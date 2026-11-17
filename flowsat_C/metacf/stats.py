"""P6 statistics: percentile bootstrap, Benjamini-Hochberg, Fisher-z, circular statistics, OLS."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as st

TWO_PI = 2 * math.pi


# ---------------------------------------------------------------- bootstrap


def bootstrap(stat_fn, n_units: int, B: int = 1000, seed: int = 0, alpha: float = 0.05,
              groups=None):
    """
    Percentile bootstrap over units (records / anchors). stat_fn(idx) -> float, where idx is an
    int array of resampled unit indices (with repeats). `groups` = list of unit counts for a
    JOINT bootstrap: stat_fn then receives a tuple of index arrays, one per group, each resampled
    independently (T4: anchors and real pairs).
    Returns dict(lo, hi, se, boot) -- non-finite replicates are dropped and counted.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        if groups is None:
            idx = rng.integers(0, n_units, n_units)
        else:
            idx = tuple(rng.integers(0, g, g) for g in groups)
        v = stat_fn(idx)
        vals.append(np.nan if v is None else float(v))
    vals = np.asarray(vals)
    ok = np.isfinite(vals)
    if ok.sum() < max(10, B // 10):
        return dict(lo=float("nan"), hi=float("nan"), se=float("nan"), n_valid=int(ok.sum()))
    v = vals[ok]
    return dict(lo=float(np.quantile(v, alpha / 2)), hi=float(np.quantile(v, 1 - alpha / 2)),
                se=float(v.std(ddof=1)), n_valid=int(ok.sum()))


def bootstrap_pvalue(stat_fn, n_units, null_value=0.0, B=1000, seed=0):
    """Two-sided bootstrap p-value for stat != null_value (percentile inversion)."""
    rng = np.random.default_rng(seed)
    v = np.array([stat_fn(rng.integers(0, n_units, n_units)) for _ in range(B)], float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan")
    p = 2 * min((v <= null_value).mean(), (v >= null_value).mean())
    return float(min(1.0, max(p, 1.0 / (len(v) + 1))))


# ---------------------------------------------------------------- multiple testing


def benjamini_hochberg(pvals, q: float = 0.05):
    """-> (reject bool array, BH-adjusted p array). NaNs pass through as not rejected / NaN."""
    p = np.asarray(pvals, float)
    adj = np.full_like(p, np.nan)
    rej = np.zeros(len(p), bool)
    ok = np.isfinite(p)
    m = ok.sum()
    if m == 0:
        return rej, adj
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    a = np.empty(m)
    a[order] = np.minimum(ranked, 1.0)
    adj[ok] = a
    rej[ok] = a <= q
    return rej, adj


# ---------------------------------------------------------------- correlations


def fisher_z_mean(rs, w=None) -> float:
    """Average correlations through Fisher z (clipped at |r| = 0.999999)."""
    r = np.asarray(rs, float)
    ok = np.isfinite(r)
    if not ok.any():
        return float("nan")
    z = np.arctanh(np.clip(r[ok], -0.999999, 0.999999))
    if w is None:
        return float(np.tanh(z.mean()))
    w = np.asarray(w, float)[ok]
    return float(np.tanh((z * w).sum() / w.sum()))


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(st.spearmanr(x[ok], y[ok]).statistic)


# ---------------------------------------------------------------- circular


def wrap(a, period=TWO_PI):
    """Wrap to (-period/2, period/2]."""
    a = np.asarray(a, float)
    h = period / 2
    return -((-a + h) % period - h)


def wrap_pos(a, period=TWO_PI):
    """Wrap to [0, period)."""
    return np.mod(np.asarray(a, float), period)


def circ_mean(a, w=None) -> float:
    a = np.asarray(a, float)
    ok = np.isfinite(a)
    if not ok.any():
        return float("nan")
    w = np.ones(ok.sum()) if w is None else np.asarray(w, float)[ok]
    return float(np.arctan2((w * np.sin(a[ok])).sum(), (w * np.cos(a[ok])).sum()))


def circ_resultant(a) -> float:
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return float("nan")
    return float(np.abs(np.exp(1j * a).mean()))


def circ_var(a) -> float:
    """1 - |mean(e^{i a})|."""
    r = circ_resultant(a)
    return float("nan") if not np.isfinite(r) else 1.0 - r


def fisher_lee(a, b) -> float:
    """
    Fisher & Lee (1983) circular-circular correlation, closed form:
        rho = 4(AB - CD) / sqrt((n^2 - E^2 - F^2)(n^2 - G^2 - H^2))
    A = sum cos a cos b, B = sum sin a sin b, C = sum cos a sin b, D = sum sin a cos b,
    E, F = sum cos 2a, sin 2a, G, H = sum cos 2b, sin 2b. Equals
    E[sin(a1-a2) sin(b1-b2)] / sqrt(E[sin^2(a1-a2)] E[sin^2(b1-b2)]) over all pairs.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 3:
        return float("nan")
    A = (np.cos(a) * np.cos(b)).sum(); B_ = (np.sin(a) * np.sin(b)).sum()
    C = (np.cos(a) * np.sin(b)).sum(); D = (np.sin(a) * np.cos(b)).sum()
    E, F = np.cos(2 * a).sum(), np.sin(2 * a).sum()
    G, H = np.cos(2 * b).sum(), np.sin(2 * b).sum()
    den = (n * n - E * E - F * F) * (n * n - G * G - H * H)
    if den <= 1e-12:
        return float("nan")
    return float(4 * (A * B_ - C * D) / math.sqrt(den))


# ---------------------------------------------------------------- regression


def ols(X, y, intercept=True):
    """-> dict(coef, se, p, r2, n). coef excludes the intercept unless intercept=False...
    when intercept=True the intercept is coef_all[0] and coef = coef_all[1:]."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, float)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    if intercept:
        X = np.column_stack([np.ones(len(X)), X])
    n, k = X.shape
    if n <= k:
        nan = np.full(k - int(intercept), np.nan)
        return dict(coef=nan, se=nan, p=nan, r2=float("nan"), n=n, intercept=float("nan"))
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = n - k
    s2 = (res @ res) / dof
    try:
        cov = s2 * np.linalg.pinv(X.T @ X)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    p = 2 * st.t.sf(np.abs(t), dof)
    sst = ((y - y.mean()) ** 2).sum() if intercept else (y ** 2).sum()
    r2 = 1 - (res @ res) / sst if sst > 0 else float("nan")
    s = slice(1, None) if intercept else slice(None)
    return dict(coef=beta[s], se=se[s], p=p[s], r2=float(r2), n=n,
                intercept=float(beta[0]) if intercept else 0.0)


def slope(x, y) -> float:
    """OLS slope of y on x with intercept."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.ptp(x[ok]) == 0:
        return float("nan")
    xc = x[ok] - x[ok].mean()
    return float((xc * (y[ok] - y[ok].mean())).sum() / (xc * xc).sum())


def fixed_effects(*labels):
    """One-hot dummy columns (first level dropped) for categorical fixed effects."""
    cols = []
    for lab in labels:
        lab = np.asarray(lab)
        levels = sorted(set(lab.tolist()))
        for lv in levels[1:]:
            cols.append((lab == lv).astype(float))
    if not cols:
        return np.zeros((len(labels[0]) if labels else 0, 0))
    return np.column_stack(cols)


def median_ignore_nan(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")
