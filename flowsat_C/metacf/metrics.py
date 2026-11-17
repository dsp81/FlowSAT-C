"""Field metrics by type (higher is always better). Inputs are in each field's TRANSFORMED space
(see fields.FieldSpec.encode): continuous -> transformed scalar, circular -> radians,
categorical -> class index, geo -> (lat_deg, lon_deg) rows."""
from __future__ import annotations

import numpy as np

from metacf import stats

EARTH_R_KM = 6371.0088


def great_circle_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2, float) - np.asarray(lon1, float))
    c = np.sin(p1) * np.sin(p2) + np.cos(p1) * np.cos(p2) * np.cos(dl)
    return EARTH_R_KM * np.arccos(np.clip(c, -1, 1))


def balanced_accuracy(pred, true) -> float:
    pred, true = np.asarray(pred), np.asarray(true)
    classes = np.unique(true)
    if len(classes) == 0:
        return float("nan")
    return float(np.mean([(pred[true == c] == c).mean() for c in classes]))


def field_score(kind: str, pred, true) -> float:
    if kind == "continuous":
        return stats.pearson(pred, true)
    if kind == "circular":
        p, t = np.asarray(pred, float), np.asarray(true, float)
        ok = np.isfinite(p) & np.isfinite(t)
        return float(np.mean(np.cos(p[ok] - t[ok]))) if ok.any() else float("nan")
    if kind == "categorical":
        return balanced_accuracy(pred, true)
    if kind == "geo":
        p, t = np.asarray(pred, float), np.asarray(true, float)
        ok = np.isfinite(p).all(1) & np.isfinite(t).all(1)
        if not ok.any():
            return float("nan")
        return -float(np.median(great_circle_km(p[ok, 0], p[ok, 1], t[ok, 0], t[ok, 1])))
    raise ValueError(kind)


def chance_score(kind: str, true, n_classes=None, seed: int = 0) -> float:
    """Score of an uninformative predictor: 0 for correlations, 1/K for balanced accuracy,
    the shuffled-label score for geo."""
    if kind in ("continuous", "circular"):
        return 0.0
    if kind == "categorical":
        k = n_classes or len(np.unique(true))
        return 1.0 / k
    if kind == "geo":
        t = np.asarray(true, float)
        perm = np.random.default_rng(seed).permutation(len(t))
        return field_score("geo", t[perm], t)
    raise ValueError(kind)


def normalise(raw, floor, ceiling):
    """P1: (raw - floor) / (ceiling - floor). Not clipped; flagged when outside [0, 1]."""
    if not all(np.isfinite([raw, floor, ceiling])) or abs(ceiling - floor) < 1e-12:
        return float("nan"), False
    v = (raw - floor) / (ceiling - floor)
    return float(v), bool(v < 0 or v > 1)
