"""
Reader heads for C2 / C3 (linear heads on frozen features), reusing v1's solvers
(metacf.readers._ridge, _logreg, location_folds):

  continuous   ridge on the transformed value                   score R^2
  circular     ridge on (cos, sin), angle = atan2               score circular R^2 =
                                                                1 - mean(wrap(pred - true)^2) / var,
                                                                var = mean(wrap(true - circmean(true))^2)
  categorical  class-balanced multinomial logistic (L-BFGS)     score balanced accuracy
  geo          ridge on unit-sphere xyz, renormalised           score -median great-circle km

lambda is chosen by k-fold GROUPED CV (groups = location_id for real tiles, anchor id for synthetic
sets); the CV metric is the head's own score above. Features are standardised with the training mean/sd.
"""
from __future__ import annotations

import math

import numpy as np

from metacf import stats
from metacf.metrics import balanced_accuracy, great_circle_km
from metacf.readers import _logreg, _ridge, location_folds


def r2(pred, true):
    p, t = np.asarray(pred, float), np.asarray(true, float)
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() < 3:
        return float("nan")
    sst = ((t[ok] - t[ok].mean()) ** 2).sum()
    return float(1 - ((t[ok] - p[ok]) ** 2).sum() / sst) if sst > 0 else float("nan")


def circ_r2(pred, true):
    p, t = np.asarray(pred, float), np.asarray(true, float)
    ok = np.isfinite(p) & np.isfinite(t)
    if ok.sum() < 3:
        return float("nan")
    mu = stats.circ_mean(t[ok])
    var = float(np.mean(stats.wrap(t[ok] - mu) ** 2))
    if var <= 0:
        return float("nan")
    return float(1 - np.mean(stats.wrap(p[ok] - t[ok]) ** 2) / var)


def geo_score(pred, true):
    p, t = np.asarray(pred, float), np.asarray(true, float)
    ok = np.isfinite(p).all(1) & np.isfinite(t).all(1)
    if not ok.any():
        return float("nan")
    return -float(np.median(great_circle_km(p[ok, 0], p[ok, 1], t[ok, 0], t[ok, 1])))


def score(kind, pred, true, w=None):
    """w = per-item weights (site-bootstrap multiplicities) -> the score on the replicated sample."""
    if w is not None:
        idx = np.repeat(np.arange(len(true)), np.asarray(w, int))
        pred, true = np.asarray(pred)[idx], np.asarray(true)[idx]
    if kind == "continuous":
        return r2(pred, true)
    if kind == "circular":
        return circ_r2(pred, true)
    if kind == "categorical":
        t = np.asarray(true)
        ok = t >= 0
        return balanced_accuracy(np.asarray(pred)[ok], t[ok]) if ok.any() else float("nan")
    if kind == "geo":
        return geo_score(pred, true)
    raise ValueError(kind)


class Head:
    def __init__(self, kind, lambdas=None, folds=5, n_classes=None):
        self.kind = kind
        self.lambdas = lambdas or ([1e-5, 1e-4, 1e-3, 1e-2] if kind == "categorical"
                                   else [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
        self.k = folds
        self.n_classes = n_classes

    def _T(self, y):
        if self.kind == "geo":
            lat, lon = np.radians(y[:, 0]), np.radians(y[:, 1])
            return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        if self.kind == "circular":
            return np.column_stack([np.cos(y), np.sin(y)])
        if self.kind == "categorical":
            return np.asarray(y, int)
        return np.asarray(y, float)[:, None]

    def _fit(self, X, T, lam):
        if self.kind == "categorical":
            return _logreg(X, T, self.n_cls, lam)
        return _ridge(X, T, lam)

    def _pred(self, m, X):
        W, b = m
        Z = X @ W + b
        if self.kind == "categorical":
            return self.cls_map[Z.argmax(1)]
        if self.kind == "circular":
            return np.arctan2(Z[:, 1], Z[:, 0])
        if self.kind == "geo":
            Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
            return np.column_stack([np.degrees(np.arcsin(np.clip(Z[:, 2], -1, 1))),
                                    np.degrees(np.arctan2(Z[:, 1], Z[:, 0]))])
        return Z[:, 0]

    def fit(self, X, y, groups):
        X = np.asarray(X, float)
        y = np.asarray(y)
        ok = np.isfinite(X).all(1)
        if self.kind == "geo":
            ok &= np.isfinite(y.astype(float)).all(1)
        elif self.kind == "categorical":
            ok &= y.astype(int) >= 0
        else:
            ok &= np.isfinite(y.astype(float))
        X, y, g = X[ok], y[ok], np.asarray(groups)[ok]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xz = (X - self.mu) / self.sd
        if self.kind == "categorical":
            y = y.astype(int)
            self.cls_map = np.unique(y)                     # dense class index -> class id
            yd = np.searchsorted(self.cls_map, y)
            self.n_cls = len(self.cls_map)
            T = yd
        else:
            T = self._T(y.astype(float))
        folds = location_folds(g, self.k)
        best = None
        for lam in self.lambdas:
            pred = np.full(len(y), np.nan) if self.kind not in ("geo",) else np.full((len(y), 2), np.nan)
            if self.kind == "categorical":
                pred = np.full(len(y), -1)
            for k in range(self.k):
                tr, te = folds != k, folds == k
                if te.sum() == 0 or tr.sum() < 2:
                    continue
                if self.kind == "categorical" and len(np.unique(T[tr])) < 2:
                    continue
                m = self._fit(Xz[tr], T[tr], lam)
                pred[te] = self._pred(m, Xz[te])
            s = score(self.kind, pred, y)
            if best is None or (np.isfinite(s) and (not np.isfinite(best[0]) or s > best[0])):
                best = (s, lam)
        self.cv_score, self.lam = best
        self.model = self._fit(Xz, T, self.lam)
        self.n_train = int(len(y))
        return self

    def predict(self, X):
        return self._pred(self.model, (np.asarray(X, float) - self.mu) / self.sd)


def permute_within(groups, y, seed):
    """Labels permuted within each group (C2 floor)."""
    rng = np.random.default_rng(seed)
    y = np.array(y, copy=True)
    g = np.asarray(groups)
    for k in np.unique(g):
        idx = np.where(g == k)[0]
        y[idx] = y[rng.permutation(idx)]
    return y


_ = math
