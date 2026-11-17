"""
C2 reader audit (v2.1, 9 Nov 2026), CPU only, cached features only (never extracts: a missing feature is skipped).

  python -m metacf.v2.measures.c2_audit --config configs/metacf_v2_fmow.yaml [--model overlay.yaml] --out <json>

Per C2 field:
  real reader   learning curve over n TRAIN tiles (every cached complete TRAIN tile in the field's space is a
                candidate) with the v2 lambda grid and an extended grid; test score on the shared 2000 TEST
                tiles with a site-bootstrap CI; chosen lambda (edge of grid?) and CV score.
  synthetic     learning curve over anchors (50 / 100 / all) with both grids.
  spaces        real reader at the largest n in dino / clip / stats / concat (only rids cached in all).
  within-cat    continuous fields: R^2 after removing the TRAIN category mean of the target from both the
                prediction and the truth (what the reader knows beyond category); the permuted floor too.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

V2_GRID = {"ridge": [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0], "cat": [1e-5, 1e-4, 1e-3, 1e-2]}
EXT_GRID = {"ridge": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4],
            "cat": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]}
NS = (500, 1000, 4000, 12000)


def grid(kind, ext):
    g = EXT_GRID if ext else V2_GRID
    return g["cat"] if kind == "categorical" else g["ridge"]


def cached_rids(run, space, rids):
    from metacf.v2.core import features as FT
    fs = run.fstore(space)
    return [r for r in rids if fs.has(FT.real_key(r))]


def within_cat(kind, pred, y, cats, catmean):
    if kind != "continuous":
        return float("nan")
    from metacf.v2.core.heads import r2
    m = np.array([catmean.get(c, np.nan) for c in cats])
    return r2(np.asarray(pred) - m, np.asarray(y) - m)


def audit(run, fields=None, seed=0):
    from metacf.v2.core import norm as N
    from metacf.v2.core.heads import Head, permute_within, score
    from metacf.v2.measures.c2_transfer import C2Transfer, test_tiles
    m = C2Transfer(run)
    readers = run.readers().get("C2", {})
    tt = test_tiles(run)
    sites = [x.location_id for x in tt]
    rng = np.random.default_rng(seed)
    out = {}
    elig_train = [x for x in run.train if run.complete(x)]
    for name in (fields or m.fields()):
        fd = run.fields[name]
        kind = fd.kind
        sp = readers[name]["space"] if name in readers else run.space_of(name)
        R = dict(space=sp, kind=kind)
        Xt = run.real_feats(sp, [x.rid for x in tt])
        yt = fd.target([fd.raw(x.fields) for x in tt])
        tcats = [x.category for x in tt]
        # TRAIN category means of the target (continuous)
        catmean = {}
        if kind == "continuous":
            ys = {}
            for x in elig_train:
                v = fd.raw(x.fields)
                if v is not None:
                    ys.setdefault(x.category, []).append(float(fd.target([v])[0]))
            catmean = {c: float(np.mean(v)) for c, v in ys.items()}
        # v2 reader as stored
        if name in readers:
            p = readers[name]["head"].predict(Xt)
            R["stored_real"] = dict(n=readers[name]["n"], lam=readers[name]["head"].lam,
                                    cv=readers[name]["head"].cv_score, test=score(kind, p, yt),
                                    test_within_cat=within_cat(kind, p, yt, tcats, catmean))
        # real learning curve
        cand = cached_rids(run, sp, [x.rid for x in elig_train])
        R["n_cached_train"] = len(cand)
        rec = {r.rid: r for r in elig_train}
        curve = []
        ns = [n for n in NS if n <= len(cand)] + [min(len(cand), NS[-1])]
        for n in sorted(set(ns)):
            pick = sorted(rng.choice(len(cand), n, replace=False).tolist())
            rr = [rec[cand[i]] for i in pick]
            X = run.real_feats(sp, [x.rid for x in rr])
            y = fd.target([fd.raw(x.fields) for x in rr])
            g = [x.location_id for x in rr]
            for ext in ((False, True) if n == 1000 else (True,)):
                h = Head(kind, lambdas=grid(kind, ext)).fit(X, y, g)
                p = h.predict(Xt)

                def bs_fn(w=None, p=p):
                    ww = None if w is None else N.site_weights_to_items(w, np.unique(sites, return_inverse=True)[1])
                    return dict(s=score(kind, p, yt, ww))
                bs = N.site_bootstrap(bs_fn, sites, 100, run.seed("c2audit", name, n, ext))
                ci = N.percentile_ci(bs["s"])
                curve.append(dict(n=n, ext_grid=ext, lam=h.lam, lam_at_edge=h.lam in (grid(kind, ext)[0], grid(kind, ext)[-1]),
                                  cv=h.cv_score, test=score(kind, p, yt), ci=[ci["lo"], ci["hi"]],
                                  test_within_cat=within_cat(kind, p, yt, tcats, catmean)))
        R["real_curve"] = curve
        # spaces at the largest common n
        sp_res = {}
        common = [x.rid for x in elig_train]
        for s2 in run.space_names():
            common = cached_rids(run, s2, common)
        tt_ok = all(len(cached_rids(run, s2, [x.rid for x in tt])) == len(tt) for s2 in run.space_names())
        if len(common) >= 500 and tt_ok:
            common = common[:4000]
            rr = [rec[r] for r in common]
            y = fd.target([fd.raw(x.fields) for x in rr])
            g = [x.location_id for x in rr]
            Xs = {s2: run.real_feats(s2, common) for s2 in run.space_names()}
            Xts = {s2: run.real_feats(s2, [x.rid for x in tt]) for s2 in run.space_names()}
            Xs["concat"] = np.concatenate([Xs[s2] for s2 in run.space_names()], 1)
            Xts["concat"] = np.concatenate([Xts[s2] for s2 in run.space_names()], 1)
            for s2 in Xs:
                h = Head(kind, lambdas=grid(kind, True)).fit(Xs[s2], y, g)
                p = h.predict(Xts[s2])
                sp_res[s2] = dict(n=len(common), lam=h.lam, test=score(kind, p, yt),
                                  test_within_cat=within_cat(kind, p, yt, tcats, catmean))
        R["spaces"] = sp_res
        # synthetic curve
        lst = m.requests()[name]
        try:
            Xs_all = run.gen_feats(sp, [d["req"] for d in lst]) if not run.fstore(sp).missing(
                [__import__("metacf.v2.core.features", fromlist=["gen_key"]).gen_key(run.store().key(d["req"])) for d in lst]) else None
        except Exception as e:  # noqa: BLE001
            Xs_all, R["syn_error"] = None, repr(e)
        if Xs_all is not None:
            ys_all = fd.target([d["value"] for d in lst])
            anc = np.array([d["anchor"] for d in lst])
            ua = sorted(set(anc.tolist()))
            scurve = []
            for na in sorted({50, 100, len(ua)}):
                keep = set(rng.choice(ua, min(na, len(ua)), replace=False).tolist())
                idx = np.array([a in keep for a in anc])
                for ext in ((False, True) if na == len(ua) else (True,)):
                    h = Head(kind, lambdas=grid(kind, ext)).fit(Xs_all[idx], ys_all[idx], anc[idx])
                    p = h.predict(Xt)
                    pf = [Head(kind, lambdas=grid(kind, ext)).fit(Xs_all[idx], permute_within(anc[idx], ys_all[idx], k), anc[idx]).predict(Xt)
                          for k in range(3)]
                    scurve.append(dict(n_anchors=len(keep), ext_grid=ext, lam=h.lam, cv=h.cv_score,
                                       test=score(kind, p, yt), floor=float(np.nanmean([score(kind, q, yt) for q in pf])),
                                       test_within_cat=within_cat(kind, p, yt, tcats, catmean),
                                       floor_within_cat=float(np.nanmean([within_cat(kind, q, yt, tcats, catmean) for q in pf]))))
            R["syn_curve"] = scurve
        out[name] = R
        print(json.dumps({name: R}, default=float), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fields", default=None)
    a = ap.parse_args()
    from metacf.config import apply_model, load_config
    from metacf.v2 import runner
    from metacf.v2.cli import _LazyAdapter, setup_threads
    from metacf.v2.run import Run
    cfg = load_config(a.config)
    if a.model:
        cfg = apply_model(cfg, a.model)
    setup_threads(cfg)
    run = Run(cfg)
    run.adapter = _LazyAdapter(cfg, run)
    if run.adapter.supported_fields is None:
        from metacf.adapters.flowsat_c import GROUP_OF
        run.adapter.supported_fields = set(GROUP_OF)
    runner.load_all(run)
    res = audit(run, a.fields.split(",") if a.fields else None)
    json.dump(res, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
