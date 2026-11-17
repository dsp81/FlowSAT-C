"""
python -m metacf <command> --config configs/metacf_fmow_v1.yaml

  build-dataset   manifests -> dataset.jsonl (records of both splits, category index)
  fit-readers     Reader A / B heads on TRAIN reals -> readers.pkl              (GPU: backbones)
  validate-real   readers on real TEST + every test's real-data validation -> validation.json
  plan            deduplicated generation plan + image count per test + GPU-hour estimate (no GPU)
  generate        run the adapter over the plan into the image store         (GPU: the model)
  run             compute every enabled test -> results.json
  card            results.json -> control_card.md / .html
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from metacf import paths

paths.ensure()

from metacf import runner  # noqa: E402
from metacf.card import write_card  # noqa: E402
from metacf.config import apply_model, canonical_json, load_config  # noqa: E402
from metacf.requests import ImageStore  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(prog="metacf")
    ap.add_argument("command", choices=["build-dataset", "fit-readers", "validate-real", "plan",
                                        "generate", "run", "card"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt-competition", action="store_true", help="enable T10")
    ap.add_argument("--model", default=None,
                    help="model overlay YAML (configs/metacf_models/*.yaml): replaces the config's model "
                         "section and merges its plan section")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.model:
        cfg = apply_model(cfg, a.model)
    pc = True if a.prompt_competition else None
    d = runner.out_dir(cfg)
    with open(os.path.join(d, "config.json"), "w") as fh:           # the merged config, as hashed
        fh.write(canonical_json(cfg))

    if a.command == "build-dataset":
        runner.build_dataset(cfg)
        return
    if a.command == "fit-readers":
        runner.fit_readers(cfg)
        return
    ds = runner.load_dataset(cfg)
    if a.command == "plan":
        ctx = runner.make_context(cfg, ds)
        summ = runner.write_plan(cfg, runner.make_plan(cfg, ctx, runner.enabled_tests(cfg, pc)))
        print(json.dumps(summ, indent=1))
        return
    if a.command == "card":
        print(write_card(os.path.join(d, "results.json")))
        return
    bank = runner.load_readers(cfg)
    ctx = runner.make_context(cfg, ds, bank)
    tests = runner.enabled_tests(cfg, pc)
    if a.command == "validate-real":
        rep = runner.validate_real(cfg, ctx, tests)
        print(json.dumps(rep["readers"], indent=1, default=float))
        return
    store = ImageStore(ctx.adapter.model_id, ctx.adapter.checkpoint, os.path.join(d, "images"))
    if a.command == "generate":
        runner.generate(cfg, ctx, runner.make_plan(cfg, ctx, tests), store)
        return
    if a.command == "run":
        val = json.load(open(os.path.join(d, "validation.json")))
        bank.validation = val["readers"]
        rows = runner.run_tests(cfg, ctx, tests, store)
        p, _ = runner.write_results(cfg, ctx, rows)
        print(p)


if __name__ == "__main__":
    sys.exit(main())
