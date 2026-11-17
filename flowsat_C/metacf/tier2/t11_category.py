"""
T11 CATEGORY FIDELITY (no tag). A category head on Reader A and on Reader B features, fit on TRAIN
reals (the config's `category` field: categorical, derived, keys [category_idx]). Balanced accuracy
on T1 generations (label = r's category) vs on real TEST. floor = chance (1/K). Per-category
accuracy only where n >= 200; the 10 most confused category pairs are reported.
"""
from __future__ import annotations

import numpy as np

from metacf import metrics, stats
from metacf.base import MetaCFTest, na, row
from metacf.tier1.t1_recovery import T1Recovery


class T11Category(MetaCFTest):
    id = "T11"
    name = "category fidelity"
    tier = 2

    def plan(self, ctx):
        return T1Recovery(self.cfg).plan(ctx)

    def compute(self, ctx, store):
        f = self.tc.get("field", "category")
        if f not in ctx.readers.readers:
            return [na(self.id, f, "no category reader fitted (config field `category`)")]
        t1 = T1Recovery(self.cfg)
        rids, S = t1.records(ctx), t1.seeds()
        own = t1.own_reqs(ctx)
        truth = ctx.truth(f, rids).astype(int)
        K = max(int(ctx.specs[f].n_classes or 0), int(truth.max()) + 1)
        out = []
        for rd in ctx.readers.readers[f]:
            pg = np.asarray(ctx.readback(f, rd, [store.get(own[(r, s)]) for r in rids for s in S])).reshape(len(rids), len(S))
            pr = np.asarray(ctx.real_readback(f, rd, rids))
            tt = np.repeat(truth, len(S))

            def fn(i):
                return metrics.balanced_accuracy(pg[i].ravel(), np.repeat(truth[i], len(S)))
            raw = fn(np.arange(len(rids)))
            ceil = metrics.balanced_accuracy(pr, truth)
            ci = stats.bootstrap(fn, len(rids), self.B, self.boot_seed(ctx, rd))
            valid = rd in ctx.readers.valid_readers(f)
            per_cat = {}
            for c in np.unique(truth):
                m = tt == c
                if m.sum() >= int(self.tc.get("min_per_category", 200)):
                    per_cat[int(c)] = float((pg.ravel()[m] == c).mean())
            conf = np.zeros((K, K))
            for a, b in zip(tt, pg.ravel()):
                conf[a, int(b)] += 1
            pairs = sorted(((conf[i, j] + conf[j, i], i, j) for i in range(K) for j in range(i + 1, K)), reverse=True)[:10]
            out.append(row(self.id, "category", rd, raw=raw, floor=1.0 / K, ceiling=ceil, ci=ci, n=len(rids),
                           seeds=len(S), validation="VALID" if valid else "INVALID", ci_on="raw",
                           per_category_accuracy=per_cat,
                           most_confused=[dict(a=int(i), b=int(j), count=int(c)) for c, i, j in pairs if c > 0]))
        return out
