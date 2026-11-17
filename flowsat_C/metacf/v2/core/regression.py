"""
Pair regression (spec 1.4).

  Delta = phi(I_b) - phi(I_a) (n x d), design dM (n x p). B = lstsq(dM, Delta) (p x d), no intercept.
  For a field with several regressors (doy, sensor, joint groups) B_f is its block of rows; its "vector"
  for cosines and norms is the block flattened.

Implementation: per-site sufficient statistics (X'X, X'Y, sum Y, sum Y^2, n) so that any re-weighting of
sites (bootstrap over SITES, split halves of sites) is a weighted sum followed by one small p x p solve:
  B(w) = pinv(sum_s w_s X_s'X_s) (sum_s w_s X_s'Y_s)
The pseudo-inverse gives the minimum-norm solution when a column is all zero (a field never intervened in
the generated set), so its rows of B are exactly 0.

  fit fraction  = 1 - SSE / SST, SST centred per dimension, pooled over dims.
  split-half reliability of B_f: n_splits (50) random halvings of the SITES, cos(B_f^(1), B_f^(2)); mean,
  and a percentile bootstrap CI over sites (each replicate: resample sites with replacement, then
  `ci_splits` halvings of the resampled multiset; OPEN_QUESTIONS_V2 Q7).
"""
from __future__ import annotations

import numpy as np


def cos(a, b) -> float:
    a, b = np.ravel(a), np.ravel(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return float("nan")
    return float(a @ b / (na * nb))


class SiteStats:
    """Sufficient statistics of a pair regression, one row per site."""

    def __init__(self, dM, Y, site_idx, n_sites=None):
        dM, Y = np.asarray(dM, float), np.asarray(Y, float)
        self.p, self.d = dM.shape[1], Y.shape[1]
        S = int(n_sites if n_sites is not None else (site_idx.max() + 1 if len(site_idx) else 0))
        self.S = S
        self.XtX = np.zeros((S, self.p, self.p))
        self.XtY = np.zeros((S, self.p * self.d), np.float32 if self.p * self.d * S > 5e7 else np.float64)
        self.sY = np.zeros((S, self.d))
        self.sYY = np.zeros(S)
        self.n = np.zeros(S)
        site_idx = np.asarray(site_idx, int)
        order = np.argsort(site_idx, kind="stable")
        bounds = np.searchsorted(site_idx[order], np.arange(S + 1))
        for s in range(S):
            m = order[bounds[s]:bounds[s + 1]]
            if len(m) == 0:
                continue
            X, Yy = dM[m], Y[m]
            self.XtX[s] = X.T @ X
            self.XtY[s] = (X.T @ Yy).ravel()
            self.sY[s] = Yy.sum(0)
            self.sYY[s] = (Yy * Yy).sum()
            self.n[s] = len(m)

    def fit(self, w=None):
        """-> B (p x d) for site weights w (None = all ones)."""
        w = np.ones(self.S) if w is None else np.asarray(w, float)
        A = np.tensordot(w, self.XtX, 1)
        C = (w @ self.XtY).reshape(self.p, self.d)
        return np.linalg.pinv(A, rcond=1e-10) @ C

    def fit_many(self, W):
        """W (k x S) -> B (k, p, d). One matmul for the k right-hand sides."""
        W = np.asarray(W, float)
        A = np.einsum("ks,sij->kij", W, self.XtX)
        C = (W.astype(self.XtY.dtype) @ self.XtY).reshape(len(W), self.p, self.d).astype(float)
        return np.einsum("kij,kjd->kid", np.linalg.pinv(A, rcond=1e-10), C)

    def r2(self, B, w=None):
        w = np.ones(self.S) if w is None else np.asarray(w, float)
        A = np.tensordot(w, self.XtX, 1)
        C = (w @ self.XtY).reshape(self.p, self.d)
        n = w @ self.n
        sse = w @ self.sYY - 2 * np.sum(B * C) + np.sum(B * (A @ B))
        mean = (w @ self.sY) / max(n, 1)
        sst = w @ self.sYY - n * float(mean @ mean)
        return float(1 - sse / sst) if sst > 0 else float("nan")


def block(B, cols):
    return np.asarray(B)[..., cols, :]


def halves(n_sites, rng):
    perm = rng.permutation(n_sites)
    h = n_sites // 2
    w1, w2 = np.zeros(n_sites), np.zeros(n_sites)
    w1[perm[:h]] = 1
    w2[perm[h:2 * h]] = 1
    return w1, w2


def split_half(ss: SiteStats, blocks: dict, n_splits=50, seed=0, B_boot=1000, ci_splits=10, alpha=0.05,
               chunk=256):
    """-> {field: dict(mean, lo, hi, p, splits=[...])} for every block, from the same fits.
    p = one-sided bootstrap p-value of mean <= 0 (fraction of replicates <= 0, floored at 1/(B+1))."""
    rng = np.random.default_rng(seed)
    S = ss.S
    W = []
    for _ in range(n_splits):
        W += list(halves(S, rng))
    Bs = ss.fit_many(np.array(W)) if S >= 2 else None
    out = {}
    for f, cols in blocks.items():
        cs = [cos(Bs[2 * i][cols], Bs[2 * i + 1][cols]) for i in range(n_splits)] if Bs is not None else []
        out[f] = dict(mean=float(np.nanmean(cs)) if cs else float("nan"), splits=cs)
    # bootstrap over sites, ci_splits halvings per replicate
    reps = {f: [] for f in blocks}
    if S >= 4 and B_boot:
        Wb = []
        # each replicate: split the UNIQUE sites into halves, then resample sites with replacement inside
        # each half (a site never sits in both halves, which would correlate them and bias the cosine up)
        for _ in range(B_boot):
            for _ in range(ci_splits):
                perm = rng.permutation(S)
                h = S // 2
                h1, h2 = perm[:h], perm[h:2 * h]
                w1 = np.bincount(h1[rng.integers(0, h, h)], minlength=S).astype(float)
                w2 = np.bincount(h2[rng.integers(0, h, h)], minlength=S).astype(float)
                Wb += [w1, w2]
        Wb = np.array(Wb)
        vals = {f: [] for f in blocks}
        for i in range(0, len(Wb), 2 * chunk):
            Bc = ss.fit_many(Wb[i:i + 2 * chunk])
            for f, cols in blocks.items():
                vals[f] += [cos(Bc[2 * j][cols], Bc[2 * j + 1][cols]) for j in range(len(Bc) // 2)]
        for f in blocks:
            v = np.array(vals[f]).reshape(B_boot, ci_splits)
            reps[f] = np.nanmean(v, 1)
    for f in blocks:
        r = np.asarray(reps[f], float)
        r = r[np.isfinite(r)]
        if len(r) >= 10:
            out[f].update(lo=float(np.quantile(r, alpha / 2)), hi=float(np.quantile(r, 1 - alpha / 2)),
                          p=float(max((r <= 0).mean(), 1.0 / (len(r) + 1))), n_boot=int(len(r)))
        else:
            out[f].update(lo=float("nan"), hi=float("nan"), p=float("nan"), n_boot=int(len(r)))
        out[f]["n_sites"] = int(S)
    return out


def fit_pairs(pairset, Y):
    """Convenience: SiteStats + full fit + R^2 for a PairSet and its Delta matrix."""
    sites, sidx = pairset.site_index()
    ss = SiteStats(pairset.dM, Y, sidx, len(sites))
    B = ss.fit()
    return ss, B, ss.r2(B)
