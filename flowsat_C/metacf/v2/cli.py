"""
python -m metacf.v2 <command> --config configs/metacf_v2_fmow.yaml   (or the `metacf2` wrapper script)

  build-pairs | select-spaces | fit-readers | plan | generate | score | meta | card
  pipeline      every stage in order, each skipped when its output exists (the launcher uses this);
                --stop-after <stage> stops early

  --model <overlay.yaml>   replace the model section (configs/metacf_models/*.yaml), as in v1
  --worker i --nworkers n  generate: shard the plan across GPUs
  --measures C0,C1,...     restrict plan / generate / score to these measures
  rescore --from <model dir> --measures C1,C3,E1 [--c1-fields gsd]   (v2_1) recompute some measures on an
                           existing model dir's generations into this config's model dir
  zoom-check [--spaces stats]   (v2_1) real-data sanity table of the synthetic zoom reference
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from metacf import paths

paths.ensure()

from metacf.config import apply_model, get, load_config  # noqa: E402
from metacf.v2 import runner  # noqa: E402
from metacf.v2.run import Run, log  # noqa: E402

STAGES = ["build-pairs", "select-spaces", "fit-readers", "plan", "generate", "score", "card"]


def make_adapter_v2(cfg, run):
    kind = cfg["model"]["adapter"]
    if kind == "flowsat_c":
        from metacf.v2.adapters import FlowSatCBatchAdapter
        return FlowSatCBatchAdapter(cfg["model"])
    if kind == "dummy2":
        from metacf.v2.adapters import DummyAdapterV2
        return DummyAdapterV2(run.ds, **cfg["model"].get("dummy", {}))
    from metacf.adapters import make_adapter
    return make_adapter(cfg)


def setup_threads(cfg):
    n = int(get(cfg, "runtime.threads", 4))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, str(n))
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass


class _LazyAdapter:
    """Defers constructing the GPU model until an image is actually requested."""

    def __init__(self, cfg, run):
        self.cfg, self.run, self._a = cfg, run, None
        m = cfg["model"]
        self.name = m.get("name", m["adapter"])
        self.model_id, self.checkpoint = m.get("model_id", m["adapter"]), m.get("checkpoint", "")
        if m["adapter"] == "dummy2":
            self.model_id, self.checkpoint = None, None
        self.supported_fields = set(m.get("supported_fields", []) or []) or None
        self.batch = int(m.get("batch", 8))

    def _get(self):
        if self._a is None:
            self._a = make_adapter_v2(self.cfg, self.run)
        return self._a

    def __getattr__(self, k):
        return getattr(self._get(), k)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="metacf2")
    ap.add_argument("command", choices=STAGES + ["meta", "pipeline", "rescore", "zoom-check"])
    ap.add_argument("--from", dest="src", default=None)
    ap.add_argument("--c1-fields", default=None)
    ap.add_argument("--spaces", default=None)
    ap.add_argument("--device", default=None, help="runtime-only override of the feature / caption encoder "
                    "device (e.g. cpu when no GPU is free); not part of the config hash")
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--meta-config", default="configs/metacf_v2_meta.yaml")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--nworkers", type=int, default=1)
    ap.add_argument("--measures", default=None)
    ap.add_argument("--stop-after", default=None, choices=STAGES)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.model:
        cfg = apply_model(cfg, a.model)
    setup_threads(cfg)
    ids = a.measures.split(",") if a.measures else None
    run = Run(cfg)
    if cfg["model"]["adapter"] == "dummy2":
        run.load_dataset() if os.path.exists(os.path.join(run.pdir, "dataset.jsonl")) else run.build_dataset()
        run.adapter = make_adapter_v2(cfg, run)
    else:
        run.adapter = _LazyAdapter(cfg, run)
        if run.adapter.supported_fields is None:
            from metacf.adapters.flowsat_c import GROUP_OF
            run.adapter.supported_fields = set(GROUP_OF)
    run.write_config()
    if a.device:
        for sp, spec in run.cfg.get("spaces", {}).items():
            if "device" in spec:
                spec["device"] = a.device
        if "device" in run.cfg.get("caption_encoder", {}):
            run.cfg["caption_encoder"]["device"] = a.device
        log(f"runtime device override: {a.device} (config hash unchanged)")
    log(f"metacf2 {a.command}: config {run.hash}, protocol {run.phash}, model dir {run.mdir}")

    def done(stage):
        p, m = run.pdir, run.mdir
        return {"build-pairs": os.path.exists(os.path.join(p, "pairs_test.json")),
                "select-spaces": os.path.exists(os.path.join(p, "selection.json")),
                "fit-readers": os.path.exists(os.path.join(p, "readers.pkl")),
                "plan": os.path.exists(os.path.join(m, "plan.json")),
                "score": os.path.exists(os.path.join(m, "results.json"))}.get(stage, False)

    if a.command == "rescore":
        if not a.src or not ids:
            raise SystemExit("rescore needs --from <model dir> and --measures")
        res = runner.rescore(run, a.src, ids, a.c1_fields.split(",") if a.c1_fields else None)
        print(f"rescored {ids}; cards: {os.path.join(run.mdir, 'control_card_v21.md')} (+ control_card.md)")
        return 0
    if a.command == "zoom-check":
        res = runner.zoom_check(run, a.spaces.split(",") if a.spaces else None)
        print(json.dumps(res, indent=1, default=float))
        return 0
    cmds = STAGES if a.command == "pipeline" else [a.command]
    for c in cmds:
        if a.command == "pipeline" and done(c):
            log(f"pipeline: {c} already done, skipped")
            if c == "build-pairs":
                runner.load_all(run, need_selection=False)
            if a.stop_after == c:
                break
            continue
        log(f"stage {c}")
        if c == "build-pairs":
            if run.ds is None:
                run.build_dataset()
            runner.build_pairs(run)
        elif c == "select-spaces":
            runner.select_spaces(run)
        elif c == "fit-readers":
            runner.fit_readers(run)
        elif c == "plan":
            summ = runner.write_plan(run, runner.make_plan(run, ids))
            print(json.dumps(summ, indent=1))
        elif c == "generate":
            plan = runner.make_plan(run, ids)
            runner.generate(run, plan, a.worker, a.nworkers)
            if a.command == "pipeline":
                runner.wait_complete(run, plan)
                rel = getattr(getattr(run.adapter, "_a", None), "release", None)
                if rel:
                    rel()
        elif c == "score":
            res = runner.score(run, ids)
            if bool((cfg.get("v2_1") or {}).get("print_composite", True)) and not cfg.get("v2_3"):
                print(f"MetaCF = {res['summary']['metacf']:.3f} over {res['summary']['n_fields']} fields; "
                      f"card: {os.path.join(run.mdir, 'control_card.md')}")
            else:
                print(f"scored; card: {os.path.join(run.mdir, 'control_card.md')}")
        elif c == "card":
            runner.card(run)
            print(os.path.join(run.mdir, "control_card.md"))
        elif c == "meta":
            from metacf.v2.meta.harness import run_meta
            run_meta(run, load_config(a.meta_config), ids=ids)
        if a.stop_after == c:
            break


if __name__ == "__main__":
    sys.exit(main())
