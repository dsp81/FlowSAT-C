"""
T8 CONDITIONAL DIVERSITY (DESCRIPTIVE).
Generations: n TEST records x 8 seeds with own fields; the same with all metadata nulled.
D_gen(r)  = mean pairwise d_dino among the 8 seeds.
D_real(r) = mean pairwise d_dino among r's condition neighbourhood: 8 real TRAIN images, same
            category, different location, nearest in z-scored field space. The neighbourhood also
            varies in PLACE, so D_gen / D_real is a LOWER bound on acceptable diversity.
Reports (a) median D_gen / D_real; (b) narrowing = median D_gen(full) / D_gen(metadata nulled);
(c) precision, recall, density, coverage (Naeem et al. 2020, k = 5) of T1 generations vs real TEST,
overall and per field quintile. Collapse flag: D_gen(r) below the 5th percentile of D_real across
records for more than 10% of records.
"""
from __future__ import annotations

import itertools

import numpy as np

from metacf import fields as F
from metacf.base import MetaCFTest, na, row
from metacf.tier1.t1_recovery import T1Recovery


def mean_pairwise(X):
    if len(X) < 2:
        return float("nan")
    return float(np.mean([1 - X[i] @ X[j] for i, j in itertools.combinations(range(len(X)), 2)]))


def prdc(real, fake, k=5):
    """Naeem et al. 2020 on unit features (Euclidean distance)."""
    def d(a, b):                                         # via the Gram matrix: no (N, M, D) tensor
        sq = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
        return np.sqrt(np.clip(sq, 0, None))
    rr = d(real, real)
    np.fill_diagonal(rr, np.inf)
    r_rad = np.sort(rr, 1)[:, k - 1]
    ff = d(fake, fake)
    np.fill_diagonal(ff, np.inf)
    f_rad = np.sort(ff, 1)[:, k - 1]
    rf = d(real, fake)                                   # (R, F)
    precision = float((rf <= r_rad[:, None]).any(0).mean())
    recall = float((rf.T <= f_rad[:, None]).any(0).mean())
    density = float((rf <= r_rad[:, None]).sum(0).mean() / k)
    coverage = float((rf.min(1) <= r_rad).mean())
    return dict(precision=precision, recall=recall, density=density, coverage=coverage)


class T8Diversity(MetaCFTest):
    id = "T8"
    name = "conditional diversity"
    descriptive = True

    def records(self, ctx):
        def pick():
            rng = np.random.default_rng(ctx.seed("T8records"))
            ids = [r.rid for r in ctx.test]
            n = min(int(self.tc.get("n_records", 300)), len(ids))
            return [ids[i] for i in sorted(rng.choice(len(ids), n, replace=False))]
        return ctx.cached(("T8rids",), pick)

    def seeds(self):
        return list(range(int(self.tc.get("seeds", 8))))

    def reqs(self, ctx):
        groups = sorted(set(ctx.adapter.field_groups.values()))
        rids = self.records(ctx)
        return ({(r, s): ctx.own_request(r, s) for r in rids for s in self.seeds()},
                {(r, s): ctx.own_request(r, s, null_groups=groups) for r in rids for s in self.seeds()})

    def plan(self, ctx):
        full, null = self.reqs(ctx)
        return list(full.values()) + list(null.values()) + T1Recovery(self.cfg).plan(ctx)

    def compute(self, ctx, store):
        rids = self.records(ctx)
        if len(rids) < int(self.tc.get("min_n", 20)):
            return [na(self.id, "*", f"{len(rids)} TEST records")]
        full, null = self.reqs(ctx)
        S = self.seeds()
        nb = F.Neighbourhoods(ctx.train, ctx.specs, ctx.eval_fields(), k=int(self.tc.get("k_neigh", 8)))
        dg, dn, dr = [], [], []
        for r in rids:
            dg.append(mean_pairwise(ctx.phi_feats([store.get(full[(r, s)]) for s in S])))
            dn.append(mean_pairwise(ctx.phi_feats([store.get(null[(r, s)]) for s in S])))
            neigh = nb.query(ctx.data.by_id[r])
            dr.append(mean_pairwise(ctx.phi_feats(ctx.real_images(neigh))) if len(neigh) >= 2 else np.nan)
        dg, dn, dr = map(np.asarray, (dg, dn, dr))
        ok = np.isfinite(dg) & np.isfinite(dr)
        ratio = float(np.median(dg[ok] / dr[ok])) if ok.any() else float("nan")
        narrowing = float(np.nanmedian(dg / dn))
        p5 = np.nanpercentile(dr, 5)
        collapse_frac = float(np.mean(dg[np.isfinite(dg)] < p5))
        out = [row(self.id, "*", "phi", stratum="diversity", raw=ratio, descriptive=True, n=int(ok.sum()),
                   seeds=len(S), median_ratio_lower_bound=ratio, narrowing=narrowing,
                   collapse_fraction=collapse_frac, collapse_flag=bool(collapse_frac > 0.10),
                   note="D_real varies in place too: the ratio is a lower bound on acceptable diversity")]
        t1 = T1Recovery(self.cfg)
        trids = t1.records(ctx)
        own = t1.own_reqs(ctx)
        fake = ctx.phi_feats([store.get(own[(r, 0)]) for r in trids])
        real = ctx.phi_feats(ctx.real_images(trids))
        k = int(self.tc.get("k_prdc", 5))
        out.append(row(self.id, "*", "phi", stratum="prdc", descriptive=True, n=len(trids),
                       **prdc(real, fake, k)))
        for f in ctx.eval_fields():
            s = ctx.specs[f]
            if s.kind not in ("continuous",):
                continue
            v = np.array([np.nan if s.raw(ctx.data.by_id[r].fields) is None else float(s.raw(ctx.data.by_id[r].fields))
                          for r in trids])
            edges = np.nanpercentile(ctx.train_values(f), [20, 40, 60, 80])
            q = np.searchsorted(edges, v)
            for b in range(5):
                m = (q == b) & np.isfinite(v)
                if m.sum() > k + 1:
                    out.append(row(self.id, f, "phi", stratum=f"prdc_q{b + 1}", descriptive=True,
                                   n=int(m.sum()), **prdc(real[m], fake[m], k)))
        return out
