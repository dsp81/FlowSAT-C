"""
Orchestration: build-dataset -> fit-readers -> validate-real -> plan -> generate -> run -> card.
Output directory: <results_root>/<model name>/<config hash>/
"""
from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np

from metacf import stats
from metacf.adapters import make_adapter
from metacf.backbones import make_backbone
from metacf.config import config_hash, get
from metacf.context import Context
from metacf.data import Dataset, read_manifest
from metacf.fields import specs_from_config
from metacf.readers import ReaderBank
from metacf.requests import ImageStore, Plan
from metacf.tier1 import ALL as TIER1
from metacf.tier2 import ALL as TIER2

ALL_TESTS = {t.id: t for t in TIER1 + TIER2}


def out_dir(cfg):
    d = os.path.join(get(cfg, "output.root", "results"), cfg["model"].get("name", cfg["model"]["adapter"]),
                     config_hash(cfg))
    os.makedirs(d, exist_ok=True)
    return d


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------- stage 1: dataset
def build_dataset(cfg):
    dc = cfg["data"]
    res = int(dc.get("resolution", 512))
    recs = read_manifest(dc["train_manifest"], "train", res, dc.get("train_limit")) + \
        read_manifest(dc["test_manifest"], "test", res, dc.get("test_limit"))
    cats = sorted({r.category for r in recs})
    for r in recs:
        r.fields["category_idx"] = cats.index(r.category)
    ds = Dataset(recs, resolution=res)
    p = os.path.join(out_dir(cfg), "dataset.jsonl")
    ds.save(p)
    json.dump(dict(categories=cats, n_train=len(ds.train), n_test=len(ds.test)),
              open(os.path.join(out_dir(cfg), "dataset_summary.json"), "w"), indent=1)
    log(f"dataset: {len(ds.train)} train, {len(ds.test)} test -> {p}")
    return ds


def load_dataset(cfg):
    return Dataset.load(os.path.join(out_dir(cfg), "dataset.jsonl"), int(cfg["data"].get("resolution", 512)))


# ---------------------------------------------------------------- stage 2/3: readers
def backbones(cfg):
    rc = cfg["readers"]
    return {name: make_backbone(rc["backbones"][name]) for name in rc.get("names", ["A", "B"])}


def reader_fields(cfg, specs):
    return [f for f, s in specs.items() if f in cfg["readers"].get("fields", list(specs))]


def fit_readers(cfg, ds=None):
    ds = ds or load_dataset(cfg)
    specs = specs_from_config(cfg)
    bank = ReaderBank(backbones(cfg), specs, cfg["readers"].get("thresholds", {}))
    n = get(cfg, "readers.n_train")
    train = ds.train
    if n and n < len(train):
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        train = [train[i] for i in sorted(rng.choice(len(train), n, replace=False))]
    bank.fit(train, ds.real_image, reader_fields(cfg, specs), get(cfg, "readers.lambdas"))
    save_readers(cfg, bank)
    log(f"fitted readers for {len(bank.readers)} fields on {len(train)} TRAIN images")
    return bank


def save_readers(cfg, bank):
    caches = {n: (b._cache, b._pcache) for n, b in bank.backbones.items()}
    for b in bank.backbones.values():
        b._cache, b._pcache = {}, {}
        if hasattr(b, "_model"):
            b._model = None
    with open(os.path.join(out_dir(cfg), "readers.pkl"), "wb") as fh:
        pickle.dump(bank, fh)
    for n, (c, p) in caches.items():
        bank.backbones[n]._cache, bank.backbones[n]._pcache = c, p


def load_readers(cfg):
    with open(os.path.join(out_dir(cfg), "readers.pkl"), "rb") as fh:
        return pickle.load(fh)


def make_context(cfg, ds, bank=None, adapter=None, phi=None, lpips_fn="config"):
    """lpips_fn="config" builds LPIPS(AlexNet) when lpips.enabled; None disables it (CPU tests)."""
    from metacf.lpips_fn import make_lpips
    specs = specs_from_config(cfg)
    bank = bank or ReaderBank({}, specs, {})
    adapter = adapter or make_adapter(cfg)
    phi = phi or make_backbone(cfg["phi"])
    if lpips_fn == "config":
        lpips_fn = make_lpips(cfg)
    return Context(cfg, ds, specs, bank, phi, adapter, lpips_fn)


def enabled_tests(cfg, prompt_competition=None):
    out = []
    for tid, cls in ALL_TESTS.items():
        on = bool(get(cfg, f"tests.{tid}.enabled", True))
        if tid == "T10":
            on = bool(prompt_competition) if prompt_competition is not None else bool(get(cfg, "tests.T10.enabled", False))
        if on:
            out.append(cls(cfg))
    return out


def validate_real(cfg, ctx, tests):
    n = get(cfg, "readers.n_validate")
    test = ctx.test if not n else ctx.test[:n]
    reader_val = ctx.readers.validate(test, ctx.data.real_image, seed=ctx.seed("readerval"))
    test_val = {}
    for t in tests:
        try:
            test_val[t.id] = t.validate_real(ctx)
        except Exception as e:                     # a validation that cannot run is reported, not hidden
            test_val[t.id] = {"*": dict(status="ERROR", detail=repr(e))}
    rep = dict(readers=reader_val, tests=test_val, config_hash=config_hash(cfg))
    with open(os.path.join(out_dir(cfg), "validation.json"), "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    return rep


# ---------------------------------------------------------------- stage 4/5: plan, generate
def make_plan(cfg, ctx, tests):
    plan = Plan()
    for t in tests:
        plan.add(t.id, t.plan(ctx), ctx.adapter.model_id, ctx.adapter.checkpoint)
    return plan


def write_plan(cfg, plan):
    d = out_dir(cfg)
    plan.to_json(os.path.join(d, "plan.json"))
    summ = plan.summary(float(get(cfg, "plan.seconds_per_image", 3.7)))
    json.dump(summ, open(os.path.join(d, "plan_summary.json"), "w"), indent=1)
    return summ


def generate(cfg, ctx, plan, store, log_every=200):
    adapter = ctx.adapter
    if any(dict(r.variant).get("mode") == "mean" for r in plan.unique.values()):
        n = int(get(cfg, "tests.T9.n_mean_records", 4096))
        rng = np.random.default_rng(ctx.seed("pathway_means"))
        tr = ctx.train
        sel = rng.choice(len(tr), min(n, len(tr)), replace=False)
        adapter.prepare_pathway_means([tr[i].fields for i in sorted(sel)])
    todo = [r for r in plan.unique.values() if not store.has(r)]
    log(f"generate: {len(todo)} of {len(plan.unique)} images missing")
    for i, r in enumerate(todo):
        donor = dict(r.variant).get("donor")
        img = adapter.generate_request(r, ctx.data.by_id[donor].fields if donor else None)
        store.put(r, np.asarray(img))
        if log_every and (i + 1) % log_every == 0:
            log(f"  {i + 1}/{len(todo)}")
    return len(todo)


# ---------------------------------------------------------------- stage 6: run
def modified(ctx, r) -> bool:
    """Rows built differently from the spec text (docs/OPEN_QUESTIONS.md, MODIFIED entries):
    geo fields in the sweep tests are swept by great-circle distance from the anchor (or latitude,
    geo_sweep.mode) rather than TRAIN percentiles (Q-geo-sweep); T5 uses the tie-jittered Ross
    estimator and equal-arc circular bins (Q-T5); T16b is split into an in-support score and a
    descriptive out-of-support extrapolation row (Q-T16b)."""
    spec = ctx.specs.get(r["field"])
    if r["test"] in ("T2", "T3", "T5", "T6") and spec is not None and spec.kind == "geo":
        return True
    if r["test"] == "T5":
        return True
    return r["test"] == "T16" and str(r.get("stratum", "")).startswith("T16b")


def run_tests(cfg, ctx, tests, store):
    rows = []
    for t in tests:
        t0 = time.time()
        try:
            rs = t.compute(ctx, store)
        except Exception as e:
            from metacf.base import na
            rs = [na(t.id, "*", f"ERROR: {e!r}", status="ERROR")]
        for r in rs:
            r.setdefault("status", r.get("status", "OK"))
            r["modified"] = bool(r.get("modified") or t.modified or modified(ctx, r))
            r["tier"] = t.tier
        rows += rs
        log(f"{t.id} {t.name}: {len(rs)} rows in {time.time() - t0:.1f}s")
    p = np.array([r["p"] for r in rows], float)
    _, adj = stats.benjamini_hochberg(p, float(get(cfg, "stats.bh_q", 0.05)))
    for r, a in zip(rows, adj):
        r["p_bh"] = float(a)
        r["bh_reject"] = bool(np.isfinite(a) and a <= float(get(cfg, "stats.bh_q", 0.05)))
    return rows


def write_results(cfg, ctx, rows):
    from metacf.card import metacf_u
    res = dict(config_hash=config_hash(cfg), model=ctx.adapter.model_id, checkpoint=ctx.adapter.checkpoint,
               adapter=ctx.adapter.name, n_rows=len(rows), metacf_u=metacf_u(rows), rows=rows)
    p = os.path.join(out_dir(cfg), "results.json")
    with open(p, "w") as fh:
        json.dump(res, fh, indent=1, default=_json_default)
    return p, res


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)
