"""
Normalisation helper (spec 1.6), validation gate (1.7) and the result-row schema.

  s_hat = (s - floor) / (ceil - floor); raw, floor, ceil and s_hat are all reported; values outside [0, 1]
  are flagged (out_of_range), never clipped.
  A measure cell whose real-data reference fails its gate gets status INVALID with the reason; the raw
  numbers that were computed are kept under extra for transparency but s_hat is NaN.
Every row carries config hash, model id, checkpoint, N and seeds (filled by the runner).
"""
from __future__ import annotations

import math

import numpy as np


def f(x):
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return float("nan")


def normalise(s, floor, ceil):
    """-> (s_hat, out_of_range). NaN when any input is not finite or ceil == floor."""
    s, floor, ceil = f(s), f(floor), f(ceil)
    if not all(math.isfinite(v) for v in (s, floor, ceil)) or abs(ceil - floor) < 1e-15:
        return float("nan"), False
    v = (s - floor) / (ceil - floor)
    return float(v), bool(v < 0 or v > 1)


RATIO_METRICS = {("C3", "ratio"), ("C3", "ratio_floored"), ("E3", "shadow_direction"),
                 ("E3", "shadow_direction_sweep"), ("C2", "slope_within_scene"), ("C2", "slope_within_scene_full"),
                 ("C1", "zoom_up"), ("C1", "zoom_down"), ("C1", "zoom_in_range")}


def bound(v, symmetric=False):
    """v2_3: a reported score lives in [0, 1]. v < 0 -> 0; v > 1 -> 1 (clip) or 1 / v (symmetric: an
    overshoot by a factor k costs as much as an undershoot by k). NaN stays NaN."""
    v = f(v)
    if not math.isfinite(v):
        return v
    if v <= 0:
        return 0.0
    if v > 1:
        return 1.0 / v if symmetric else 1.0
    return v


def apply_bounds(rows):
    """v2_3: s_hat -> bounded (kept unbounded under extra.s_hat_unbounded; CI stays that of the unbounded
    value). Ratio metrics (RATIO_METRICS) are bounded symmetrically, the others clipped."""
    for x in rows:
        s = x.get("s_hat", float("nan"))
        if not math.isfinite(f(s)):
            continue
        e = x.setdefault("extra", {})
        if "s_hat_unbounded" in e:
            continue
        e["s_hat_unbounded"] = f(s)
        x["s_hat"] = bound(s, (x["measure"], x["metric"]) in RATIO_METRICS)
        x["out_of_range"] = bool(f(s) < 0 or f(s) > 1)
    return rows


def row(measure, field, metric, raw=float("nan"), floor=float("nan"), ceil=float("nan"), s_hat=None,
        ci=None, n=0, seeds=0, status="OK", reason="", p=float("nan"), gate=None, modified=False,
        space="", **extra):
    if s_hat is None:
        s_hat, oor = normalise(raw, floor, ceil)
    else:
        s_hat, oor = f(s_hat), bool(math.isfinite(f(s_hat)) and (f(s_hat) < 0 or f(s_hat) > 1))
    ci = ci or {}
    return dict(measure=measure, field=field, metric=metric, raw=f(raw), floor=f(floor), ceil=f(ceil),
                s_hat=f(s_hat), out_of_range=oor, ci_lo=f(ci.get("lo")), ci_hi=f(ci.get("hi")),
                n=int(n), seeds=int(seeds), status=status, reason=reason, p=f(p), p_bh=float("nan"),
                gate=gate, modified=bool(modified), space=space, extra=_clean(extra))


def invalid(measure, field, metric, reason, status="INVALID", **extra):
    return row(measure, field, metric, status=status, reason=reason, **extra)


def _clean(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        out[k] = v
    return out


def percentile_ci(vals, alpha=0.05, min_valid=10):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) < min_valid:
        return dict(lo=float("nan"), hi=float("nan"), n_valid=int(len(v)))
    return dict(lo=float(np.quantile(v, alpha / 2)), hi=float(np.quantile(v, 1 - alpha / 2)), n_valid=int(len(v)))


def boot_p_greater(vals, null=0.0):
    """One-sided bootstrap p-value for stat > null: fraction of replicates <= null (floored)."""
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan")
    return float(max((v <= null).mean(), 1.0 / (len(v) + 1)))


def site_bootstrap(stat_fn, sites, B=1000, seed=0):
    """Resample SITES with replacement. stat_fn(weights over the unique sites) -> float or dict of floats.
    Returns a list (or dict of lists) of replicate values."""
    uniq = sorted(set(sites))
    S = len(uniq)
    rng = np.random.default_rng(seed)
    out = None
    for _ in range(B):
        w = np.bincount(rng.integers(0, S, S), minlength=S).astype(float)
        v = stat_fn(w)
        if isinstance(v, dict):
            out = out or {k: [] for k in v}
            for k, x in v.items():
                out[k].append(f(x))
        else:
            out = out if out is not None else []
            out.append(f(v))
    return out if out is not None else []


def site_weights_to_items(w, item_site_idx):
    """Site weights -> per-item weights (an item inherits its site's multiplicity)."""
    return np.asarray(w, float)[np.asarray(item_site_idx, int)]
