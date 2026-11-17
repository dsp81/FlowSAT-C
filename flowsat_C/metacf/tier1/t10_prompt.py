"""
T10 PROMPT COMPETITION (optional; config tests.T10.enabled or CLI --prompt-competition).
Reruns T1 and T3 with NULL captions (same records, seeds and values).
Metadata share = norm score (null caption) - norm score (real caption), per field, with a CI from a
JOINT bootstrap (the same resampled records / anchor-seed units for both captions).
"""
from __future__ import annotations

import numpy as np

from metacf import stats
from metacf.base import MetaCFTest, na, row
from metacf.tier1.t1_recovery import T1Recovery
from metacf.tier1.t3_monotonic import T3Monotonic


class _NullT1(T1Recovery):
    id = "T1"
    caption_mode = "null"


class _NullT3(T3Monotonic):
    id = "T3"
    caption_mode = "null"


class T10PromptCompetition(MetaCFTest):
    id = "T10"
    name = "prompt competition"

    def plan(self, ctx):
        return _NullT1(self.cfg).plan(ctx) + _NullT3(self.cfg).plan(ctx) + \
            T1Recovery(self.cfg).plan(ctx) + T3Monotonic(self.cfg).plan(ctx)

    def compute(self, ctx, store):
        out = self.na_unsupported(ctx)
        for f in self.fields(ctx):
            for rd in ctx.readers.valid_readers(f):
                for name, real_t, null_t in (("T1", T1Recovery(self.cfg), _NullT1(self.cfg)),
                                             ("T3", T3Monotonic(self.cfg), _NullT3(self.cfg))):
                    fr, n = real_t.norm_fn(ctx, store, f, rd)[:2]
                    fn, _ = null_t.norm_fn(ctx, store, f, rd)[:2]
                    share = fn(np.arange(n))[3] - fr(np.arange(n))[3]
                    if not np.isfinite(share):
                        out.append(na(self.id, f, f"{name} norm undefined (missing floor/ceiling)",
                                      reader=rd, stratum=name))
                        continue
                    ci = stats.bootstrap(lambda i: fn(i)[3] - fr(i)[3], n, self.B,
                                         self.boot_seed(ctx, f, rd, name))
                    out.append(row(self.id, f, rd, stratum=name, raw=share, descriptive=True, ci=ci, n=n,
                                   metadata_share=share))
        return out
