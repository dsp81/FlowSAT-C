"""
T4 CALIBRATION. beta_gen = median over anchors x seeds of the OLS slope of read-back on set value
(transformed: log for GSD, radians with wrapped differences for circular fields, raw otherwise).
beta_real = over natural-experiment pairs (i, j) at one location, regress
    d_readback_f = R_f(x_j) - R_f(x_i)   on the vector of ALL field differences dF
(partial regression, no intercept, circular differences wrapped to (-pi, pi]) -> coefficient of
d_f. Other fields are held fixed by the regression (real date pairs also differ in sun, cloud,
year). C = beta_gen / beta_real; score = max(0, 1 - |log2 C|) (0 when C <= 0). C is reported, so
under- and over-reaction are visible. CI: joint bootstrap over anchors x seeds and pairs.
Min-n: 200 pairs with |d_f| above the field's 25th-percentile |d_f|. Categorical: N/A (use T5).
Geo: N/A (location is constant within a natural pair, so beta_real is undefined).
"""
from __future__ import annotations

import math

import numpy as np

from metacf import stats
from metacf.base import MetaCFTest, na, row
from metacf.tier1.common import centred, standard_sweep


def regression_fields(ctx):
    """Fields entering dF: every non-categorical, non-geo field with a value on TRAIN records."""
    return [g for g, s in ctx.specs.items() if s.kind in ("continuous", "circular")]


def pair_design(ctx, pairs, fields):
    """dF matrix (n_pairs, n_fields) in transformed units; NaN where a value is missing."""
    D = np.zeros((len(pairs), len(fields)))
    for j, g in enumerate(fields):
        s = ctx.specs[g]
        for i, (a, b) in enumerate(pairs):
            va, vb = s.raw(ctx.data.by_id[a].fields), s.raw(ctx.data.by_id[b].fields)
            D[i, j] = np.nan if va is None or vb is None else float(s.diff(vb, va))
    return D


def readback_diff(ctx, f, reader, pairs):
    a = ctx.train_readback(f, reader, [p[0] for p in pairs])
    b = ctx.train_readback(f, reader, [p[1] for p in pairs])
    d = b - a
    return stats.wrap(d) if ctx.specs[f].kind == "circular" else d


def partial_coef(y, D, j):
    """No-intercept OLS of y on D (columns with zero variance dropped); coefficient of column j."""
    ok = np.isfinite(y) & np.isfinite(D).all(1)
    y, D = y[ok], D[ok]
    keep = [c for c in range(D.shape[1]) if np.ptp(D[:, c]) > 0 or c == j]
    if j not in keep or len(y) <= len(keep):
        return float("nan")
    res = stats.ols(D[:, keep], y, intercept=False)
    return float(res["coef"][keep.index(j)])


class T4Calibration(MetaCFTest):
    id = "T4"
    name = "calibration"

    def plan(self, ctx):
        return [r for f in self.swept(ctx) for r in standard_sweep(ctx, f).requests()]

    def slopes(self, ctx, store, f, reader):
        sw = standard_sweep(ctx, f)
        kind = ctx.specs[f].kind
        R = sw.readbacks(store, reader)
        v = centred(kind, sw.set_values)
        sl = np.array([[stats.slope(v, centred(kind, R[a, :, s])) for s in range(R.shape[2])]
                       for a in range(R.shape[0])])
        return sl.ravel()

    def real_parts(self, ctx, f, reader):
        pairs = ctx.natural_pairs(f)
        fields = regression_fields(ctx)
        if f not in fields or not pairs:
            return None
        D = pair_design(ctx, pairs, fields)
        y = readback_diff(ctx, f, reader, pairs)
        j = fields.index(f)
        absd = np.abs(D[:, j])
        thr = np.nanpercentile(absd[absd > 0], 25) if (absd > 0).any() else np.inf
        return dict(D=D, y=y, j=j, n_big=int((absd > thr).sum()), fields=fields)

    def compute(self, ctx, store):
        out = self.na_unsupported(ctx) + self.na_recovery_only(ctx)
        min_pairs = int(self.tc.get("min_pairs", 200))
        for f in self.swept(ctx):
            kind = ctx.specs[f].kind
            if kind in ("categorical", "geo"):
                out.append(na(self.id, f, f"T4 is N/A for {kind} fields"))
                continue
            for rd in ctx.readers.valid_readers(f):
                rp = self.real_parts(ctx, f, rd)
                if rp is None or rp["n_big"] < min_pairs:
                    out.append(na(self.id, f, f"{0 if rp is None else rp['n_big']} natural pairs above the "
                                  f"25th-pct |d{f}| < {min_pairs}", reader=rd))
                    continue
                sl = self.slopes(ctx, store, f, rd)
                bg = stats.median_ignore_nan(sl)
                br = partial_coef(rp["y"], rp["D"], rp["j"])
                C = bg / br if br else float("nan")
                score = max(0.0, 1 - abs(math.log2(C))) if np.isfinite(C) and C > 0 else 0.0

                def fn(idx, sl=sl, rp=rp):
                    ia, ip = idx
                    b_r = partial_coef(rp["y"][ip], rp["D"][ip], rp["j"])
                    c = stats.median_ignore_nan(sl[ia]) / b_r if b_r else float("nan")
                    return max(0.0, 1 - abs(math.log2(c))) if np.isfinite(c) and c > 0 else 0.0
                ci = stats.bootstrap(fn, 0, self.B, self.boot_seed(ctx, f, rd), groups=[len(sl), len(rp["y"])])
                out.append(row(self.id, f, rd, raw=score, floor=0.0, ceiling=1.0, ci=ci, n=len(sl),
                               seeds=len(ctx.anchor_seeds()), C=C, beta_gen=bg, beta_real=br,
                               n_pairs=len(rp["y"]), n_pairs_above_q25=rp["n_big"],
                               direction="over-reacts" if np.isfinite(C) and C > 1 else "under-reacts"))
        return out
