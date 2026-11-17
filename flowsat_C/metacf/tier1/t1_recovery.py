"""
T1 RECOVERY. Generations G(r, s) for N TEST records x S seeds, plus per field f the floor set
G(r, f <- F_pi(r)[f], s) where pi is a category-stratified derangement and only f is shuffled.
raw     = field metric(reader read-back of G(r, s), r's own value)
floor   = same metric on the f-shuffled generations, still against r's OWN value
ceiling = the reader on r's real TEST image
also    source-following = metric(read-back of the floor generations, the DONOR's value)
Headline recovery per field = the LOWER of the valid readers' normalised scores (P3).
Agreement index = corr(A, B on generated) / corr(A, B on real TEST).
"""
from __future__ import annotations

import numpy as np

from metacf import metrics, stats
from metacf.base import MetaCFTest, na, row


def _agreement(kind, a, b):
    if kind == "categorical":
        return float(np.mean(np.asarray(a) == np.asarray(b)))
    if kind == "circular":
        return stats.fisher_lee(a, b)
    if kind == "geo":
        return stats.fisher_z_mean([stats.pearson(a[:, 0], b[:, 0]), stats.pearson(a[:, 1], b[:, 1])])
    return stats.pearson(a, b)


class T1Recovery(MetaCFTest):
    id = "T1"
    name = "recovery"
    caption_mode = "real"

    def n_records(self):
        return int(self.tc.get("n_records", 1000))

    def seeds(self):
        return list(range(int(self.tc.get("seeds", 3))))

    def records(self, ctx):
        return ctx.t1_records(self.n_records())

    def own_reqs(self, ctx):
        return {(r, s): ctx.own_request(r, s, self.caption_mode)
                for r in self.records(ctx) for s in self.seeds()}

    def floor_seeds(self):
        """Seeds for the f-shuffled floor generations (tests.T1.floor_seeds; default = all seeds)."""
        n = self.tc.get("floor_seeds")
        return self.seeds()[: int(n)] if n else self.seeds()

    def floor_reqs(self, ctx, f):
        rids = self.records(ctx)
        donor = ctx.derangement(rids, "t1")
        spec = ctx.specs[f]
        return {(r, s): ctx.set_request(r, f, spec.raw(ctx.data.by_id[donor[r]].fields), s, "RAW",
                                        self.caption_mode)
                for r in rids for s in self.floor_seeds()}

    def plan(self, ctx):
        reqs = list(self.own_reqs(ctx).values())
        for f in self.fields(ctx):
            reqs += list(self.floor_reqs(ctx, f).values())
        return reqs

    # ---------------------------------------------------------------- shared arrays
    def arrays(self, ctx, store, f, reader):
        """Read-backs for field f with one reader; cached on the context (reused by T5b, T10)."""
        def comp():
            rids = self.records(ctx)
            S = self.seeds()
            own = self.own_reqs(ctx)
            flo = self.floor_reqs(ctx, f)
            donor = ctx.derangement(rids, "t1")
            g = ctx.readback(f, reader, [store.get(own[(r, s)]) for r in rids for s in S])
            SF = self.floor_seeds()
            fl = ctx.readback(f, reader, [store.get(flo[(r, s)]) for r in rids for s in SF])
            shape = (len(rids), len(S)) + (np.shape(g)[1:] if np.ndim(g) > 1 else ())
            fshape = (len(rids), len(SF)) + (np.shape(g)[1:] if np.ndim(g) > 1 else ())
            return dict(rids=rids, gen=np.asarray(g).reshape(shape), floor=np.asarray(fl).reshape(fshape),
                        truth=ctx.truth(f, rids), donor=ctx.truth(f, [donor[r] for r in rids]),
                        real=np.asarray(ctx.real_readback(f, reader, rids)))
        return ctx.cached(("T1arrays", self.caption_mode, f, reader), comp)

    @staticmethod
    def _score(kind, pred, truth, idx):
        p = pred[idx]
        t = truth[idx]
        S = p.shape[1]
        tt = np.repeat(t, S, axis=0) if t.ndim > 1 else np.repeat(t, S)
        pp = p.reshape((-1,) + p.shape[2:])
        return metrics.field_score(kind, pp, tt)

    def norm_fn(self, ctx, store, f, reader):
        """idx -> (raw, floor, ceiling, norm) on a record subset (for bootstraps, T10)."""
        A = self.arrays(ctx, store, f, reader)
        kind = ctx.specs[f].kind

        def fn(idx):
            raw = self._score(kind, A["gen"], A["truth"], idx)
            flo = self._score(kind, A["floor"], A["truth"], idx)
            ceil = metrics.field_score(kind, A["real"][idx], A["truth"][idx])
            return raw, flo, ceil, metrics.normalise(raw, flo, ceil)[0]
        return fn, len(A["rids"])

    # ---------------------------------------------------------------- compute
    def compute(self, ctx, store):
        out = self.na_unsupported(ctx)
        min_n = int(self.tc.get("min_n", 100))
        for f in self.fields(ctx):
            names = list(ctx.readers.readers.get(f, {}))
            if not names:
                out.append(na(self.id, f, "no reader fitted for this field"))
                continue
            if len(self.records(ctx)) < min_n:
                out.append(na(self.id, f, f"only {len(self.records(ctx))} TEST records < min_n {min_n}"))
                continue
            kind = ctx.specs[f].kind
            valid = ctx.readers.valid_readers(f)
            norms = {}
            for rd in names:
                fn, n = self.norm_fn(ctx, store, f, rd)
                raw, flo, ceil, nrm = fn(np.arange(n))
                ci = stats.bootstrap(lambda i: fn(i)[3], n, self.B, self.boot_seed(ctx, f, rd))
                p = stats.bootstrap_pvalue(lambda i: fn(i)[0] - fn(i)[1], n, 0.0, self.B,
                                           self.boot_seed(ctx, f, rd, "p"))
                A = self.arrays(ctx, store, f, rd)
                src = self._score(kind, A["floor"], A["donor"], np.arange(n))
                status = "VALID" if rd in valid else "INVALID"
                r = row(self.id, f, rd, raw=raw, floor=flo, ceiling=ceil, ci=ci, n=n,
                        seeds=len(self.seeds()), floor_seeds=len(self.floor_seeds()), validation=status, p=p,
                        source_following=src, reader_validation=ctx.readers.validation.get(f, {}).get(rd))
                out.append(r)
                if status == "VALID":
                    norms[rd] = (r["norm"], ci)
            if norms:
                worst = min(norms, key=lambda k: norms[k][0] if np.isfinite(norms[k][0]) else -np.inf)
                out.append(row(self.id, f, "headline", raw=norms[worst][0], floor=0.0, ceiling=1.0,
                               ci=norms[worst][1], n=len(self.records(ctx)), seeds=len(self.seeds()),
                               notes=f"min over valid readers {sorted(norms)}: {worst}"))
            else:
                out.append(na(self.id, f, "no valid reader for this field", reader="headline",
                              status="INVALID"))
            if {"A", "B"} <= set(names):
                a, b = self.arrays(ctx, store, f, "A"), self.arrays(ctx, store, f, "B")
                flat = lambda x: x.reshape((-1,) + x.shape[2:])
                ag = _agreement(kind, flat(a["gen"]), flat(b["gen"]))
                ar = _agreement(kind, a["real"], b["real"])
                out.append(row(self.id, f, "agreement", raw=ag / ar if ar else float("nan"),
                               descriptive=True, n=len(a["rids"]), agreement_gen=ag, agreement_real=ar))
        return out
