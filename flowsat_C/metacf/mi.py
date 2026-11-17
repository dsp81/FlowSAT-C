"""Mutual information estimators for T5 (control resolution). All results in BITS."""
from __future__ import annotations

import math

import numpy as np
from scipy.special import digamma

LN2 = math.log(2.0)


def mi_discrete_continuous(x, y, k: int = 3, seed: int = 0, tie_jitter: float = 2.0) -> float:
    """
    Ross (2014), "Mutual information between discrete and continuous data sets", k-NN estimator.
    x: discrete labels (n,), y: continuous (n,) or (n, d).

        I = psi(N) - <psi(N_x)> + psi(k) - <psi(m)>

    For each point i: N_x = points sharing its label; d_i = distance to its k-th nearest neighbour
    among those (k reduced to N_x - 1 if needed); m_i = points in the WHOLE set within d_i
    (excluding i). Points whose label occurs once are dropped. Chebyshev (max-norm) distance.

    Ties. A generator that ignores a field over part of its range returns IDENTICAL images, hence
    identical read-backs, for different labels; such point masses break the estimator's continuity
    assumption (it then under-counts the information carried by the clusters). Values closer than
    the estimator's own resolution are indistinguishable, so a seeded Gaussian jitter of sd
    tie_jitter x (median distance to the k-th same-label neighbour) is added before estimation.
    Default 2.0, chosen on toys with known MI: two disjoint uniforms (1 bit) and four (2 bits) are
    recovered exactly, a two-component Gaussian mixture (0.161 bits) gives 0.148, and a planted
    7-level step response (0.985 bits) gives 0.958 (tie_jitter 0: 0.654). docs/OPEN_QUESTIONS.md.
    """
    x = np.asarray(x)
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    ok = np.isfinite(y).all(1)
    x, y = x[ok], y[ok]
    labels, counts = np.unique(x, return_counts=True)
    keep_lab = set(labels[counts > 1].tolist())
    keep = np.array([v in keep_lab for v in x])
    x, y = x[keep], y[keep]
    n = len(x)
    if n < k + 2:
        return float("nan")
    rng = np.random.default_rng(seed)

    def cheb(v):
        D = np.max(np.abs(v[:, None, :] - v[None, :, :]), axis=2)    # (n, n) Chebyshev
        np.fill_diagonal(D, np.inf)
        return D

    if tie_jitter > 0:
        D0 = cheb(y)
        dk = []
        for lab in np.unique(x):
            idx = np.where(x == lab)[0]
            kk = min(k, len(idx) - 1)
            dk += list(np.sort(D0[np.ix_(idx, idx)], axis=1)[:, kk - 1])
        res = float(np.median(dk))
        if res > 0:
            y = y + rng.normal(0, tie_jitter * res, y.shape)
    scale = np.std(y, axis=0)
    scale[scale == 0] = 1.0
    y = y + rng.normal(0, 1e-10, y.shape) * scale
    D = cheb(y)
    psi_nx, psi_m, psi_k = [], [], []
    for lab in np.unique(x):
        idx = np.where(x == lab)[0]
        nx = len(idx)
        kk = min(k, nx - 1)
        sub = D[np.ix_(idx, idx)]
        dk = np.sort(sub, axis=1)[:, kk - 1]
        m = (D[idx] <= dk[:, None]).sum(1)                         # neighbours in full set within d_k
        psi_nx += [digamma(nx)] * nx
        psi_k += [digamma(kk)] * nx
        psi_m += list(digamma(np.maximum(m, 1)))
    I = digamma(n) - np.mean(psi_nx) + np.mean(psi_k) - np.mean(psi_m)
    return float(max(I, 0.0) / LN2)


def _entropy_mm(labels) -> float:
    """Plug-in entropy (nats) with the Miller-Madow correction (m - 1) / (2N)."""
    _, c = np.unique(labels, return_counts=True)
    n = c.sum()
    p = c / n
    return float(-(p * np.log(p)).sum() + (len(c) - 1) / (2 * n))


def mi_discrete_discrete(x, y) -> float:
    """Plug-in MI with Miller-Madow-corrected entropies: H(X) + H(Y) - H(X, Y). Bits."""
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 2:
        return float("nan")
    joint = np.array([f"{a}|{b}" for a, b in zip(x.tolist(), y.tolist())])
    I = _entropy_mm(x) + _entropy_mm(y) - _entropy_mm(joint)
    return float(max(I, 0.0) / LN2)


def circular_bins(angles, K: int, origin: float = 0.0):
    """K equal circular arcs of width 2 pi / K starting at `origin` (radians) -> bin index 0..K-1.
    Equal arcs keep the information of an effect that is symmetric about its peak wherever the peak
    is; binning cos(angle) folds every pair of angles mirrored about 0 / pi into one bin."""
    a = np.mod(np.asarray(angles, float) - origin, 2 * math.pi)
    return np.minimum((a / (2 * math.pi / K)).astype(int), K - 1)


def gaussian_channel_mi(means, sds, n_grid=4001) -> float:
    """MI (bits) between a uniform index over K levels and Y | k ~ N(means[k], sds[k]^2):
    H(mixture) - mean_k H(N(m_k, s_k)), the mixture entropy by numerical integration. The
    'theoretical' value for a read-back whose signal and noise are known or estimated."""
    m = np.asarray(means, float)
    s = np.maximum(np.asarray(sds, float), 1e-12)
    lo, hi = (m - 8 * s).min(), (m + 8 * s).max()
    y = np.linspace(lo, hi, n_grid)
    pdf = np.mean(np.exp(-0.5 * ((y[:, None] - m) / s) ** 2) / (s * math.sqrt(2 * math.pi)), axis=1)
    ok = pdf > 0
    h_mix = -np.trapz(np.where(ok, pdf * np.log(np.where(ok, pdf, 1)), 0), y)
    h_cond = np.mean(0.5 * np.log(2 * math.pi * math.e * s ** 2))
    return float(max(h_mix - h_cond, 0.0) / LN2)


def quantile_bins(values, K: int, edges_from=None):
    """Bin index 0..K-1 by quantiles (of `edges_from` if given, e.g. the TRAIN distribution)."""
    v = np.asarray(values, float)
    ref = np.asarray(edges_from if edges_from is not None else v, float)
    ref = ref[np.isfinite(ref)]
    edges = np.quantile(ref, np.linspace(0, 1, K + 1)[1:-1])
    return np.searchsorted(edges, v, side="right")
