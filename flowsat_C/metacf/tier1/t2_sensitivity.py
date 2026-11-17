"""
T2 SENSITIVITY (DESCRIPTIVE). From the standard sweeps:
D_field = mean over anchors, seeds and value pairs k != k' of d(G(a, f<-v_k, s), G(a, f<-v_k', s))
D_seed  = mean over anchors, values and seed pairs s != s' of d(G(a, f<-v_k, s), G(a, f<-v_k, s'))
S_f = D_field / D_seed, CI by bootstrap over anchors. d = d_dino (d_lpips also reported when the
LPIPS model is available).
"""
from __future__ import annotations

import itertools

import numpy as np

from metacf import stats
from metacf.base import MetaCFTest, na, row
from metacf.tier1.common import standard_sweep


def pair_means(P):
    """P: (A, K, S, D) unit features -> per-anchor (D_field, D_seed) with d = 1 - cos."""
    A, K, S, _ = P.shape
    df, ds = np.zeros(A), np.zeros(A)
    for a in range(A):
        v = [1 - P[a, k, s] @ P[a, k2, s] for s in range(S) for k, k2 in itertools.combinations(range(K), 2)]
        w = [1 - P[a, k, s] @ P[a, k, s2] for k in range(K) for s, s2 in itertools.combinations(range(S), 2)]
        df[a], ds[a] = np.mean(v), np.mean(w) if w else np.nan
    return df, ds


class T2Sensitivity(MetaCFTest):
    id = "T2"
    name = "sensitivity"
    descriptive = True

    def plan(self, ctx):
        return [r for f in self.swept(ctx) for r in standard_sweep(ctx, f).requests()]

    def compute(self, ctx, store):
        out = self.na_unsupported(ctx) + self.na_recovery_only(ctx)
        for f in self.swept(ctx):
            sw = standard_sweep(ctx, f)
            if len(sw.anchors) < int(self.tc.get("min_n", 5)):
                out.append(na(self.id, f, f"{len(sw.anchors)} anchors in support"))
                continue
            df, ds = pair_means(sw.phi(store))
            fn = lambda i: df[i].mean() / ds[i].mean()
            ci = stats.bootstrap(fn, len(df), self.B, self.boot_seed(ctx, f))
            extra = {}
            if ctx.lpips_fn is not None:
                extra["S_lpips"] = self._lpips_ratio(ctx, sw, store)
            out.append(row(self.id, f, "phi", raw=fn(np.arange(len(df))), ci=ci, n=len(df),
                           seeds=len(sw.seeds), descriptive=True, D_field=float(df.mean()),
                           D_seed=float(ds.mean()), **extra))
        return out

    @staticmethod
    def _lpips_ratio(ctx, sw, store):
        imgs, _ = sw.images(store)
        xs_f, ys_f, xs_s, ys_s = [], [], [], []
        for a in range(len(sw.anchors)):
            for s in range(len(sw.seeds)):
                for k, k2 in itertools.combinations(range(len(sw.grid)), 2):
                    xs_f.append(imgs[a][k][s]); ys_f.append(imgs[a][k2][s])
            for k in range(len(sw.grid)):
                for s, s2 in itertools.combinations(range(len(sw.seeds)), 2):
                    xs_s.append(imgs[a][k][s]); ys_s.append(imgs[a][k][s2])
        return float(np.mean(ctx.d_lpips(xs_f, ys_f)) / np.mean(ctx.d_lpips(xs_s, ys_s)))
