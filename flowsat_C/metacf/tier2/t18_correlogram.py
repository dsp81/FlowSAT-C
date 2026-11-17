"""
T18 SPATIAL CORRELOGRAM (tag latlon). Uses the T1 generations (seed 0) and T1's latlon-shuffled floor
generations; no extra generation. Pairs of records in the same category, binned by geodesic
distance {0-50, 50-200, 200-500, 500-1000, 1000-2000, 2000-5000, > 5000 km}.
D_gen(bin) = mean d_dino between their generations; D_real(bin) = between their real TEST images;
D_floor(bin) = between their latlon-shuffled floor generations.
score = 1 - sum_b w_b |D_gen - D_real| / sum_b w_b |D_floor - D_real|  (w = pair counts).
Min-n: 500 pairs per bin (bins below are dropped and listed). Validation: D_real increases with
distance (Spearman over bins >= 0.6), else INVALID.
"""
from __future__ import annotations

import itertools

import numpy as np

from metacf import metrics, stats
from metacf.base import MetaCFTest, na, row
from metacf.tier1.t1_recovery import T1Recovery

BINS = [0, 50, 200, 500, 1000, 2000, 5000, np.inf]


class T18Correlogram(MetaCFTest):
    id = "T18"
    name = "spatial correlogram"
    tier = 2
    required_tags = {"latlon"}

    def plan(self, ctx):
        return T1Recovery(self.cfg).plan(ctx)

    def pairs(self, ctx, f):
        def comp():
            t1 = T1Recovery(self.cfg)
            rids = t1.records(ctx)
            spec = ctx.specs[f]
            by_cat = {}
            for r in rids:
                by_cat.setdefault(ctx.data.by_id[r].category, []).append(r)
            pa, pb, dist = [], [], []
            cap = int(self.tc.get("max_pairs", 200000))
            rng = np.random.default_rng(ctx.seed("T18pairs"))
            allp = [p for rs in by_cat.values() for p in itertools.combinations(rs, 2)]
            if len(allp) > cap:
                allp = [allp[i] for i in sorted(rng.choice(len(allp), cap, replace=False))]
            for a, b in allp:
                fa, fb = ctx.data.by_id[a].fields, ctx.data.by_id[b].fields
                pa.append(a); pb.append(b)
                dist.append(float(metrics.great_circle_km(fa[spec.keys[0]], fa[spec.keys[1]],
                                                          fb[spec.keys[0]], fb[spec.keys[1]])))
            return pa, pb, np.digitize(dist, BINS[1:-1])
        return ctx.cached(("T18pairs", f), comp)

    def curves(self, ctx, store, f):
        t1 = T1Recovery(self.cfg)
        own, flo = t1.own_reqs(ctx), t1.floor_reqs(ctx, f)
        pa, pb, bins = self.pairs(ctx, f)
        rids = t1.records(ctx)
        pos = {r: i for i, r in enumerate(rids)}
        Xg = ctx.phi_feats([store.get(own[(r, 0)]) for r in rids])
        Xf = ctx.phi_feats([store.get(flo[(r, 0)]) for r in rids])
        Xr = ctx.phi_feats(ctx.real_images(rids))
        ia, ib = np.array([pos[a] for a in pa]), np.array([pos[b] for b in pb])
        d = {k: 1 - (X[ia] * X[ib]).sum(1) for k, X in (("gen", Xg), ("floor", Xf), ("real", Xr))}
        return d, bins

    def validate_real(self, ctx):
        return {}

    def compute(self, ctx, store):
        out = self.missing_tag_na(ctx)
        min_n = int(self.tc.get("min_pairs_per_bin", 500))
        for f in self.fields(ctx):
            d, bins = self.curves(ctx, store, f)
            used, dropped = [], []
            for b in range(len(BINS) - 1):
                (used if (bins == b).sum() >= min_n else dropped).append(b)
            if len(used) < 3:
                out.append(na(self.id, f, f"only {len(used)} distance bins with >= {min_n} pairs"))
                continue
            w = np.array([(bins == b).sum() for b in used], float)
            cur = {k: np.array([v[bins == b].mean() for b in used]) for k, v in d.items()}
            rho = stats.spearman(np.arange(len(used)), cur["real"])
            if not (np.isfinite(rho) and rho >= float(self.tc.get("min_spearman", 0.6))):
                out.append(na(self.id, f, f"D_real does not increase with distance (Spearman {rho:.2f})",
                              status="INVALID"))
                continue
            num = (w * np.abs(cur["gen"] - cur["real"])).sum()
            den = (w * np.abs(cur["floor"] - cur["real"])).sum()
            score = 1 - num / den if den > 0 else float("nan")
            labels = [f"{BINS[b]}-{BINS[b + 1]} km" for b in used]
            out.append(row(self.id, f, "phi", raw=score, floor=0.0, ceiling=1.0, n=int(w.sum()),
                           bins=labels, dropped_bins=[f"{BINS[b]}-{BINS[b + 1]} km" for b in dropped],
                           D_gen=cur["gen"].tolist(), D_real=cur["real"].tolist(),
                           D_floor=cur["floor"].tolist(), real_spearman=rho))
        return out
