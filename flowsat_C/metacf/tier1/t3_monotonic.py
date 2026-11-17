"""
T3 MONOTONIC CONTROL. Per anchor x seed sweep: continuous -> Spearman(v_k, read-back_k); circular ->
Fisher-Lee circular-circular correlation; categorical -> switching accuracy. Aggregate: Fisher-z
mean over anchors x seeds.
floor   = the same aggregate with the K set values randomly permuted against the read-backs
          (1000 permutations, mean of the aggregates)
ceiling = the same statistic on real natural-experiment sequences (same location, >= 3 distinct
          values of f; TRAIN reals, out-of-fold read-backs); N/A if fewer than 30 sequences
Run in RAW and CONSISTENT modes (CONSISTENT only for fields that drive derived solar fields);
the difference is reported.
"""
from __future__ import annotations

import numpy as np

from metacf import stats
from metacf.base import MetaCFTest, na, row
from metacf.tier1.common import aggregate, standard_sweep, sweep_kind, sweep_modes, sweep_statistic


class T3Monotonic(MetaCFTest):
    id = "T3"
    name = "monotonic control"
    caption_mode = "real"

    def sweeps(self, ctx, f):
        return {m: standard_sweep(ctx, f, m, self.caption_mode) for m in sweep_modes(ctx, f, self.tc.get("consistent_fields"))}

    def plan(self, ctx):
        return [r for f in self.swept(ctx) for sw in self.sweeps(ctx, f).values() for r in sw.requests()]

    def per_sweep(self, ctx, store, f, reader, sw):
        kind = sweep_kind(ctx, f)
        R = sw.readbacks(store, reader)                                  # (A, K, S)
        v = sw.set_values
        vals = np.array([[sweep_statistic(kind, v, R[a, :, s]) for s in range(R.shape[2])]
                         for a in range(R.shape[0])])                    # (A, S)
        return vals, R

    def floor(self, ctx, kind, v, R, seed, n_perm):
        rng = np.random.default_rng(seed)
        A, K, S = R.shape
        aggs = []
        for _ in range(n_perm):
            st = [sweep_statistic(kind, v[rng.permutation(K)], R[a, :, s]) for a in range(A) for s in range(S)]
            aggs.append(aggregate(kind, st))
        return float(np.nanmean(aggs))

    def ceiling(self, ctx, f, reader):
        kind = sweep_kind(ctx, f)
        seqs = ctx.natural_sequences(f, 3)
        spec = ctx.sweep_spec(f)
        sts = []
        for rids in seqs:
            truth = spec.encode([spec.raw(ctx.data.by_id[r].fields) for r in rids])
            rb = ctx.train_readback(f, reader, rids)
            rb = rb[:, 0] if rb.ndim > 1 else rb
            sts.append(sweep_statistic(kind, truth, rb))
        return aggregate(kind, sts), len(seqs)

    def norm_fn(self, ctx, store, f, reader, mode="RAW"):
        """unit index (anchor x seed flattened) -> norm, with floor and ceiling fixed (T10)."""
        sw = self.sweeps(ctx, f)[mode]
        kind = sweep_kind(ctx, f)
        vals, R = self.per_sweep(ctx, store, f, reader, sw)
        flo = self.floor(ctx, kind, sw.set_values, R, self.boot_seed(ctx, f, reader, mode, "perm"),
                         int(self.tc.get("n_perm", 1000)))
        ceil, n_seq = self.ceiling(ctx, f, reader)
        if n_seq < int(self.tc.get("min_sequences", 30)):
            ceil = float("nan")
        flat = vals.ravel()

        def fn(i):
            raw = aggregate(kind, flat[i])
            return raw, flo, ceil, (raw - flo) / (ceil - flo) if np.isfinite(ceil) and ceil != flo else float("nan")
        return fn, len(flat), n_seq

    def compute(self, ctx, store):
        out = self.na_unsupported(ctx) + self.na_recovery_only(ctx)
        for f in self.swept(ctx):
            valid = ctx.readers.valid_readers(f)
            if not valid:
                out.append(na(self.id, f, "no valid reader", status="INVALID"))
                continue
            raws = {}
            for mode in sweep_modes(ctx, f, self.tc.get("consistent_fields")):
                sw = self.sweeps(ctx, f)[mode]
                if len(sw.anchors) < int(self.tc.get("min_n", 5)):
                    out.append(na(self.id, f, f"{len(sw.anchors)} anchors in support", stratum=mode))
                    continue
                for rd in valid:
                    fn, n, n_seq = self.norm_fn(ctx, store, f, rd, mode)
                    raw, flo, ceil, _ = fn(np.arange(n))
                    ci = stats.bootstrap(lambda i: fn(i)[0], n, self.B, self.boot_seed(ctx, f, rd, mode))
                    p = stats.bootstrap_pvalue(lambda i: fn(i)[0] - flo, n, 0.0, self.B,
                                               self.boot_seed(ctx, f, rd, mode, "p"))
                    note = "MEASURED" if np.isfinite(ceil) else \
                        f"ceiling N/A: {n_seq} natural sequences < {self.tc.get('min_sequences', 30)}"
                    out.append(row(self.id, f, rd, stratum=mode, raw=raw, floor=flo, ceiling=ceil, ci=ci,
                                   n=n, seeds=len(sw.seeds), p=p, notes=note, n_sequences=n_seq,
                                   ci_on="raw"))
                    raws[(mode, rd)] = raw
            for rd in valid:
                if ("RAW", rd) in raws and ("CONSISTENT", rd) in raws:
                    out.append(row(self.id, f, rd, stratum="CONSISTENT-RAW", descriptive=True,
                                   raw=raws[("CONSISTENT", rd)] - raws[("RAW", rd)]))
        return out
