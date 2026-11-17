"""
C5 MEMORISATION (spec section 2; a check, not a score).

  n_records (1000) TRAIN records generated through three channels at one seed: full (own caption +
  metadata), metadata-only (null caption), caption-only (every metadata group nulled) -> 3000 images.
  Index: default-space (DINOv2) features of ALL TRAIN real tiles (index_n: null = all). FAISS is not
  installed on this machine, so the search is EXACT brute-force cosine (torch on GPU when available,
  numpy otherwise): at least as accurate as FAISS (Q16).
  Per image: own-record retrieval (NN is the conditioning record), location retrieval (NN shares the
  record's location_id), copy (NN distance < the 1st percentile of TEST-real NN distances).
  Baseline: the same rates for n_test_real (1000) TEST real tiles (own / location are 0 by construction:
  TEST sites are not in the TRAIN index; copy = 1% by construction of the threshold).
  Flag per channel and rate if rate - baseline > flag threshold (config).
"""
from __future__ import annotations

import numpy as np

from metacf.v2.core import norm as N
from metacf.v2.measures.base import Measure, pick_by_category


def nn_search(Q, X, chunk=4096):
    """Cosine nearest neighbour of each row of Q among the rows of X (both L2-normalised).
    -> (index, distance = 1 - cos)."""
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        Xt = torch.as_tensor(X, dtype=torch.float16 if dev == "cuda" else torch.float32, device=dev)
        idx, dist = [], []
        for i in range(0, len(Q), chunk):
            q = torch.as_tensor(Q[i:i + chunk], dtype=Xt.dtype, device=dev)
            best_v, best_i = None, None
            for j in range(0, len(Xt), 65536):
                s = q @ Xt[j:j + 65536].T
                v, k = s.max(1)
                k = k + j
                if best_v is None:
                    best_v, best_i = v, k
                else:
                    m = v > best_v
                    best_v, best_i = torch.where(m, v, best_v), torch.where(m, k, best_i)
            idx.append(best_i.cpu().numpy())
            dist.append(1 - best_v.float().cpu().numpy())
        return np.concatenate(idx), np.concatenate(dist)
    except ImportError:
        S = Q @ X.T
        k = S.argmax(1)
        return k, 1 - S[np.arange(len(Q)), k]


class C5Memorisation(Measure):
    id = "C5"
    name = "memorisation"

    def records(self):
        def comp():
            r = self.run
            elig = [x for x in r.train if r.complete(x)]
            return pick_by_category(elig, int(self.c("n_records", 1000)), self.seed("records"))
        return self.run.cached(("C5", "recs"), comp)

    def groups(self):
        return sorted({fd.group for fd in self.run.fields.values() if fd.enabled})

    def requests(self):
        def comp():
            r = self.run
            out = {"full": [], "metadata_only": [], "caption_only": []}
            for x in self.records():
                sd = r.seed("C5seed", x.rid)
                out["full"].append(r.req(x, seed=sd))
                out["metadata_only"].append(r.req(x, caption=None, seed=sd))
                out["caption_only"].append(r.req(x, seed=sd, null_mask=self.groups()))
            return out
        return self.run.cached(("C5", "reqs"), comp)

    def plan(self):
        return [q for v in self.requests().values() for q in v]

    def index(self):
        def comp():
            r = self.run
            n = self.c("index_n", None)
            recs = r.train
            if n and n < len(recs):
                rng = np.random.default_rng(self.seed("index"))
                recs = [recs[i] for i in sorted(rng.choice(len(recs), int(n), replace=False))]
            sp = r.cfg.get("default_space", "dino")
            X = r.real_feats(sp, [x.rid for x in recs])
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
            return recs, X.astype(np.float32), sp
        return self.run.cached(("C5", "index"), comp)

    def rates(self, Q, rids):
        r = self.run
        recs, X, _ = self.index()
        k, d = nn_search(Q, X)
        own = np.array([recs[i].rid == rid for i, rid in zip(k, rids)])
        loc = np.array([recs[i].location_id == r.rec(rid).location_id for i, rid in zip(k, rids)])
        return own, loc, d

    def compute(self):
        r = self.run
        recs, X, sp = self.index()
        rng = np.random.default_rng(self.seed("test"))
        tst = r.test
        tst = [tst[i] for i in sorted(rng.choice(len(tst), min(int(self.c("n_test_real", 1000)), len(tst)),
                                                replace=False))]
        Qt = r.real_feats(sp, [x.rid for x in tst])
        Qt = Qt / np.maximum(np.linalg.norm(Qt, axis=1, keepdims=True), 1e-12)
        own_t, loc_t, d_t = self.rates(Qt, [x.rid for x in tst])
        tau = float(np.percentile(d_t, 1))
        base = dict(own=float(own_t.mean()), location=float(loc_t.mean()), copy=float((d_t < tau).mean()))
        thr = dict(self.c("flag", {"own": 0.05, "location": 0.05, "copy": 0.05}))
        out = []
        for ch, reqs in self.requests().items():
            Q = r.gen_feats(sp, reqs)
            Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
            own, loc, d = self.rates(Q, [q.rid for q in reqs])
            rates = dict(own=float(own.mean()), location=float(loc.mean()), copy=float((d < tau).mean()))
            for key in ("own", "location", "copy"):
                excess = rates[key] - base[key]
                flag = bool(excess > float(thr.get(key, 0.05)))
                out.append(N.row(self.id, ch, key, raw=rates[key], floor=base[key], s_hat=float("nan"),
                                 n=len(reqs), seeds=1, space=sp, baseline=base[key], excess=excess,
                                 threshold=float(thr.get(key, 0.05)), flag=flag, tau=tau,
                                 index_size=len(recs), median_nn_dist=float(np.median(d)),
                                 search="exact brute-force cosine (FAISS not installed)"))
        return out
