"""
Shared GSD reader trained on REAL images only (MetaCF v2.2, 9 Nov 2026). Used by E1 matching
(physics/e1_match.py); written so other measures (e.g. a C1/C2 matching variant, not built) can call
`reader(run, space)` / `choose(run)` / `predict(run, space, feats)` without retraining.

Reader: ridge head (grouped CV over sites, core/heads.Head) from a feature space to log effective GSD
(gsd_eff_m = ground size of one of the 512 output pixels = footprint / 512), trained on n_train REAL TRAIN tiles
(category-stratified, seeded). Validation, done before any generated image is scored:
  (a) n_test REAL TEST tiles (other sites): R^2, Spearman, attenuation slope of pred on true. Largely category.
  (b) WITHIN-SCENE zoom ladders from real TEST chips (core/zoom.py; coarse = whole centre square, fine_k = centre
      crop 1/k, same pixel count, nothing up-sampled; true d log g' = -log k): per-chip Spearman of pred vs true
      and the pooled within-chip slope c_within. (b) is what a sweep test needs: rank zoom levels of ONE scene.
Space choice (`choose`): by (b) only: highest mean within-scene Spearman among spaces with
c_within >= min_within_slope; valid iff it reaches min_within_spearman.
Config: v2_2.gsd_reader {n_train, n_test, spaces, min_within_spearman, min_within_slope}.
"""
from __future__ import annotations

import numpy as np

from metacf import stats
from metacf.config import get

DEFAULTS = dict(n_train=12000, n_test=3000, spaces=None, min_within_spearman=0.6, min_within_slope=0.2)


def settings(run):
    s = dict(DEFAULTS)
    s.update(dict(get(run.cfg, "v2_2.gsd_reader", {}) or {}))
    return s


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(stats.spearman(x[ok], y[ok]))


def ols(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xd = x - x.mean()
    d = float(np.sum(xd * xd))
    return float(np.sum(xd * (y - y.mean())) / d) if d > 0 else float("nan")


def gsd_name(run):
    return next((n for n, f in run.fields.items() if f.tag == "gsd" and f.enabled), None)


def real_sets(run):
    def comp():
        from metacf.v2.measures.base import pick_by_category
        st = settings(run)
        fd = run.fields[gsd_name(run)]
        ok = lambda x: fd.raw(x.fields) is not None and float(fd.raw(x.fields)) > 0
        tr = pick_by_category([x for x in run.train if ok(x)], int(st["n_train"]), run.seed("gsdr_train"))
        te = pick_by_category([x for x in run.test if ok(x)], int(st["n_test"]), run.seed("gsdr_test"))
        return tr, te
    return run.cached(("gsdr", "sets"), comp)


def ladders(run):
    """Within-scene zoom ladders: {rid: [(syn key, true log g'), ...]} (>= 3 levels), from core/zoom.py pairs."""
    def comp():
        from metacf.v2.core import zoom
        fd = run.fields[gsd_name(run)]
        lad = {}
        for rid, _, k in zoom.pairs(run):
            g = fd.raw(run.rec(rid).fields)
            if g is None:
                continue
            lg = float(np.log(float(g)))
            d = lad.setdefault(rid, {zoom.syn_key(rid, "coarse"): lg})
            d[zoom.syn_key(rid, "fine", k)] = lg - float(np.log(k))
        return {r: sorted(d.items(), key=lambda t: t[1]) for r, d in lad.items() if len(d) >= 3}
    return run.cached(("gsdr", "ladders"), comp)


def within_scene(pred_by_key, lad):
    """-> (per-ladder Spearman array, pooled within-ladder slope)."""
    rhos, xs, ps = [], [], []
    for v in lad.values():
        x = np.array([t for _, t in v])
        p = np.array([pred_by_key[k] for k, _ in v])
        rhos.append(spearman(x, p))
        xs.append(x - x.mean())
        ps.append(p - p.mean())
    if not xs:
        return np.zeros(0), float("nan")
    xs, ps = np.concatenate(xs), np.concatenate(ps)
    return np.array(rhos, float), float(np.sum(xs * ps) / np.sum(xs * xs))


def reader(run, space):
    """-> dict(head, validity): real TRAIN tiles -> log gsd_eff_m in `space`, with checks (a) and (b)."""
    def comp():
        from metacf.v2.core.heads import Head
        fd = run.fields[gsd_name(run)]
        tr, te = real_sets(run)
        Xtr = run.real_feats(space, [x.rid for x in tr])
        ytr = np.log([float(fd.raw(x.fields)) for x in tr])
        h = Head("continuous").fit(Xtr, ytr, np.array([x.location_id for x in tr]))
        yte = np.log([float(fd.raw(x.fields)) for x in te])
        pte = h.predict(run.real_feats(space, [x.rid for x in te]))
        r2 = float(1 - np.sum((yte - pte) ** 2) / np.sum((yte - yte.mean()) ** 2))
        L = ladders(run)
        keys = [k for v in L.values() for k, _ in v]
        P = dict(zip(keys, h.predict(run.syn_feats(space, keys)))) if keys else {}
        rh, cw = within_scene(P, L)
        val = dict(space=space, lam=getattr(h, "lam", None), n_train=len(tr), n_test=len(te),
                   test_r2=r2, test_spearman=spearman(yte, pte), test_slope=ols(yte, pte),
                   n_ladders=len(L), within_spearman=float(np.nanmean(rh)) if len(rh) else float("nan"),
                   within_frac_perfect=float(np.mean(rh > 0.999)) if len(rh) else float("nan"),
                   within_frac_positive=float(np.mean(rh > 0)) if len(rh) else float("nan"),
                   c_within=cw)
        return dict(head=h, validity=val)
    return run.cached(("gsdr", "reader", space), comp)


def choose(run):
    """Reader space chosen by the within-scene check (b) only; never by generated scores."""
    def comp():
        st = settings(run)
        sps = st["spaces"] or run.space_names()
        vals = {sp: reader(run, sp)["validity"] for sp in sps}
        ok = {sp: v for sp, v in vals.items() if np.isfinite(v["c_within"]) and v["c_within"] >= st["min_within_slope"]
              and np.isfinite(v["within_spearman"])}
        best = max(ok, key=lambda sp: ok[sp]["within_spearman"]) if ok else None
        valid = best is not None and ok[best]["within_spearman"] >= st["min_within_spearman"]
        return dict(space=best, valid=bool(valid), all=vals,
                    reason="" if valid else "no space reaches the within-scene gate")
    return run.cached(("gsdr", "choose"), comp)


def predict(run, space, feats):
    """Read-off log gsd_eff_m for feature rows of `space` (already post-processed, as from run.*_feats)."""
    return reader(run, space)["head"].predict(feats)
