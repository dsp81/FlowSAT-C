"""
T7 CONDITIONAL MEMORIZATION (DESCRIPTIVE, with an explicit flag).
Generations: n TRAIN records with their own caption and fields (seed 1), the same records with
(a) a null caption (metadata-only channel) and (b) all metadata nulled (caption-only channel); the
T1 TEST-record generations (seed 0) are the contrast set.
Index: phi over ALL TRAIN real images (record id and location_id stored). FAISS is not installed
here, so the index is exact brute-force cosine search in numpy (chunked) -- identical results.
Per generation: nearest-neighbour distance d1 = 1 - cos, own-retrieval (NN = the conditioning
record's image), location-retrieval (NN shares its location_id).
tau = 1st percentile of the real-TEST -> TRAIN NN distance.
Metrics per channel: own-retrieval rate (vs chance 1/|train|); location-retrieval rate minus the
real-TEST rate; copy rate = fraction with d1 < tau, minus 1%; memorization gap = median distance to
the conditioning record for TEST-conditioned minus TRAIN-conditioned generations (full channel).
"""
from __future__ import annotations

import numpy as np

from metacf.base import MetaCFTest, na, row
from metacf.tier1.t1_recovery import T1Recovery


class NNIndex:
    def __init__(self, feats, rids, locs, chunk=4096):
        self.X, self.rids, self.locs, self.chunk = feats, list(rids), list(locs), chunk

    def query(self, Q):
        best_i = np.zeros(len(Q), int)
        best_s = np.full(len(Q), -np.inf)
        for i in range(0, len(self.X), self.chunk):
            S = Q @ self.X[i:i + self.chunk].T
            j = S.argmax(1)
            s = S[np.arange(len(Q)), j]
            upd = s > best_s
            best_s[upd], best_i[upd] = s[upd], j[upd] + i
        return best_i, 1.0 - best_s


class T7Memorization(MetaCFTest):
    id = "T7"
    name = "conditional memorization"
    descriptive = True

    def train_records(self, ctx):
        def pick():
            rng = np.random.default_rng(ctx.seed("T7records"))
            ids = [r.rid for r in ctx.train]
            n = min(int(self.tc.get("n_records", 1000)), len(ids))
            return [ids[i] for i in sorted(rng.choice(len(ids), n, replace=False))]
        return ctx.cached(("T7rids",), pick)

    def all_groups(self, ctx):
        return sorted(set(ctx.adapter.field_groups.values()))

    def channel_reqs(self, ctx):
        rids, seed = self.train_records(ctx), int(self.tc.get("seed", 1))
        return dict(full={r: ctx.own_request(r, seed) for r in rids},
                    metadata_only={r: ctx.own_request(r, seed, caption_mode="null") for r in rids},
                    caption_only={r: ctx.own_request(r, seed, null_groups=self.all_groups(ctx)) for r in rids})

    def plan(self, ctx):
        reqs = [r for ch in self.channel_reqs(ctx).values() for r in ch.values()]
        return reqs + T1Recovery(self.cfg).plan(ctx)

    def index(self, ctx):
        def build():
            tr = ctx.train
            X = ctx.phi_feats(ctx.real_images([r.rid for r in tr]))
            return NNIndex(X, [r.rid for r in tr], [r.location_id for r in tr])
        return ctx.cached(("T7index",), build)

    def compute(self, ctx, store):
        idx = self.index(ctx)
        n_train = len(idx.rids)
        if n_train < 2:
            return [na(self.id, "*", "TRAIN index is empty")]
        t1 = T1Recovery(self.cfg)
        test_rids = t1.records(ctx)
        # real-TEST baseline
        Qr = ctx.phi_feats(ctx.real_images(test_rids))
        nn_r, d_r = idx.query(Qr)
        tau = float(np.percentile(d_r, 1))
        loc_r = float(np.mean([idx.locs[j] == ctx.data.by_id[r].location_id for j, r in zip(nn_r, test_rids)]))
        thr = self.tc.get("flag", {"own_retrieval": 0.05, "copy_rate": 0.05, "gap": 0.05})
        out = []
        gaps = {}
        for ch, reqs in self.channel_reqs(ctx).items():
            rids = list(reqs)
            Q = ctx.phi_feats([store.get(reqs[r]) for r in rids])
            nn, d1 = idx.query(Q)
            own = float(np.mean([idx.rids[j] == r for j, r in zip(nn, rids)]))
            loc = float(np.mean([idx.locs[j] == ctx.data.by_id[r].location_id for j, r in zip(nn, rids)]))
            copy = float(np.mean(d1 < tau)) - 0.01
            cond = ctx.phi_feats(ctx.real_images(rids))
            d_cond_train = 1 - (Q * cond).sum(1)
            extra = {}
            if ch == "full":
                own_req = t1.own_reqs(ctx)
                Qt = ctx.phi_feats([store.get(own_req[(r, 0)]) for r in test_rids])
                d_cond_test = 1 - (Qt * Qr).sum(1)
                gaps[ch] = float(np.median(d_cond_test) - np.median(d_cond_train))
                extra["memorization_gap"] = gaps[ch]
            flag = own > thr["own_retrieval"] or copy > thr["copy_rate"] or gaps.get(ch, -1) > thr["gap"]
            out.append(row(self.id, "*", "phi", stratum=ch, raw=own, descriptive=True, n=len(rids),
                           seeds=1, own_retrieval=own, own_retrieval_chance=1.0 / n_train,
                           location_retrieval_excess=loc - loc_r, copy_rate=copy, tau=tau,
                           median_d1=float(np.median(d1)), memorization_flag=bool(flag), thresholds=thr,
                           **extra))
        return out
