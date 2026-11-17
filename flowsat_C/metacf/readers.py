"""
Readers (P3): linear heads on frozen features, one per field per backbone. Fit on TRAIN reals only,
lambda by 5-fold location-grouped CV (fold = md5(location_id) % 5, as eval_metrics.Oracle), then
validated on real TEST images (P2). Predictions are in the field's transformed space
(fields.FieldSpec.encode).

  continuous  ridge on the transformed value
  circular    ridge on (cos, sin), angle = atan2
  categorical multinomial logistic, class-balanced weights, L2
  geo         ridge on unit-sphere xyz, renormalised, -> (lat, lon)
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.optimize import minimize

from metacf import metrics


def location_folds(location_ids, k=5):
    return np.array([int(hashlib.md5(str(l).encode()).hexdigest()[:8], 16) % k for l in location_ids])


def _ridge(X, Y, lam):
    """Ridge with intercept on standardised X. Returns (W, b). lam is scaled by n."""
    n, d = X.shape
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    if d <= n:
        A = Xc.T @ Xc + lam * n * np.eye(d)
        W = np.linalg.solve(A, Xc.T @ Yc)
    else:                                                   # dual form when d > n
        K = Xc @ Xc.T + lam * n * np.eye(n)
        W = Xc.T @ np.linalg.solve(K, Yc)
    return W, ym - xm @ W


def _logreg(X, y, n_cls, lam, maxiter=300):
    """Class-balanced multinomial logistic regression, L2 on weights (L-BFGS)."""
    n, d = X.shape
    counts = np.bincount(y, minlength=n_cls).astype(float)
    sw = np.where(counts[y] > 0, n / (n_cls * np.maximum(counts[y], 1)), 0.0)
    Y = np.zeros((n, n_cls))
    Y[np.arange(n), y] = 1

    def f(theta):
        W = theta[: d * n_cls].reshape(d, n_cls)
        b = theta[d * n_cls:]
        Z = X @ W + b
        Z -= Z.max(1, keepdims=True)
        lse = np.log(np.exp(Z).sum(1))
        P = np.exp(Z - lse[:, None])
        loss = (sw * (lse - (Z * Y).sum(1))).sum() / n + 0.5 * lam * (W * W).sum()
        G = (P - Y) * sw[:, None] / n
        gW = X.T @ G + lam * W
        gb = G.sum(0)
        return loss, np.concatenate([gW.ravel(), gb])

    th0 = np.zeros(d * n_cls + n_cls)
    r = minimize(f, th0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
    return r.x[: d * n_cls].reshape(d, n_cls), r.x[d * n_cls:]


class Reader:
    """One field, one backbone."""

    def __init__(self, spec, lambdas=None, folds=5):
        self.spec = spec
        self.kind = spec.kind
        self.lambdas = lambdas or ([1e-5, 1e-4, 1e-3] if self.kind == "categorical"
                                   else [1e-4, 1e-3, 1e-2, 1e-1, 1.0])
        self.k = folds
        self.lam = None
        self.cv_score = None

    # ---- targets ----
    def _target(self, y_raw):
        if self.kind == "geo":
            lat, lon = np.radians(y_raw[:, 0]), np.radians(y_raw[:, 1])
            return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        if self.kind == "circular":
            z = self.spec.encode(y_raw)
            return np.column_stack([np.cos(z), np.sin(z)])
        if self.kind == "categorical":
            return np.asarray(y_raw, int)
        return self.spec.encode(y_raw)[:, None]

    def truth(self, y_raw):
        """Raw values -> the space predictions are scored in."""
        if self.kind == "geo":
            return np.asarray(y_raw, float)
        if self.kind == "categorical":
            return np.asarray(y_raw, int)
        return self.spec.encode(y_raw)

    def _fit_one(self, X, T, lam):
        if self.kind == "categorical":
            return _logreg(X, T, self.n_cls, lam)
        return _ridge(X, T, lam)

    def _pred_one(self, model, X):
        W, b = model
        Z = X @ W + b
        if self.kind == "categorical":
            return Z.argmax(1)
        if self.kind == "circular":
            return np.arctan2(Z[:, 1], Z[:, 0])
        if self.kind == "geo":
            Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
            return np.column_stack([np.degrees(np.arcsin(np.clip(Z[:, 2], -1, 1))),
                                    np.degrees(np.arctan2(Z[:, 1], Z[:, 0]))])
        return Z[:, 0]

    def fit(self, X, y_raw, location_ids):
        X = np.asarray(X, float)
        y_raw = np.asarray(y_raw, float if self.kind != "categorical" else int)
        ok = np.isfinite(X).all(1)
        ok &= np.isfinite(y_raw).all(1) if y_raw.ndim == 2 else np.isfinite(y_raw.astype(float))
        if self.kind == "categorical":
            ok &= y_raw >= 0
        X, y_raw = X[ok], y_raw[ok]
        loc = np.asarray(location_ids)[ok]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        Xz = (X - self.mu) / self.sd
        T = self._target(y_raw)
        truth = self.truth(y_raw)
        if self.kind == "categorical":
            self.n_cls = max(int(self.spec.n_classes or 0), int(y_raw.max()) + 1)
        folds = location_folds(loc, self.k)
        best, self.oof = None, None
        for lam in self.lambdas:
            pred = np.zeros_like(truth, dtype=float) if self.kind != "categorical" else \
                np.zeros(len(truth), int)
            for f in range(self.k):
                tr, te = folds != f, folds == f
                if te.sum() == 0 or tr.sum() == 0:
                    continue
                m = self._fit_one(Xz[tr], T[tr], lam)
                pred[te] = self._pred_one(m, Xz[te])
            s = metrics.field_score(self.kind, pred, truth)
            if best is None or (np.isfinite(s) and s > best[0]):
                best = (s, lam)
                self.oof = pred.copy()
        self.cv_score, self.lam = best
        self.oof_index = np.where(ok)[0]                  # positions in the records passed to fit
        self.model = self._fit_one(Xz, T, self.lam)
        self.n_train = int(len(X))
        return self

    def predict(self, X):
        Xz = (np.asarray(X, float) - self.mu) / self.sd
        return self._pred_one(self.model, Xz)


def reader_min(kind, cfg_thresholds):
    t = cfg_thresholds
    return {"continuous": t.get("min_r", 0.30), "circular": t.get("min_mean_cos", 0.30),
            "categorical": t.get("min_bacc_x_chance", 1.5), "geo": t.get("geo_max_frac_shuffled", 0.7)
            }[kind]


def validate_reader(reader, X_test, y_test_raw, thresholds, seed=0):
    """P2: on real TEST images the reader must reach its field's minimum."""
    pred = reader.predict(X_test)
    truth = reader.truth(np.asarray(y_test_raw))
    score = metrics.field_score(reader.kind, pred, truth)
    thr = reader_min(reader.kind, thresholds)
    if reader.kind == "categorical":
        chance = 1.0 / max(reader.n_cls, len(np.unique(truth)))
        need = thr * chance
        ok = score >= need
    elif reader.kind == "geo":
        shuffled = metrics.chance_score("geo", truth, seed=seed)       # -median km (negative)
        need = -thr * (-shuffled)                                      # error <= 0.7 x shuffled
        ok = score >= need
    else:
        need = thr
        ok = score >= need
    return dict(valid=bool(ok and np.isfinite(score)), score=float(score), threshold=float(need),
                n=int(len(truth)))


class ReaderBank:
    """
    readers[field][name] for name in ("A", "B"); backbones[name]. Holds the real-TEST predictions
    (ceilings, standardisation) and the validation outcome per field x reader.
    """

    def __init__(self, backbones: dict, specs: dict, thresholds: dict):
        self.backbones, self.specs, self.thresholds = backbones, specs, thresholds
        self.readers, self.validation, self.real_pred = {}, {}, {}

    def fit(self, train_records, image_fn, fields, lambdas=None):
        imgs = [image_fn(r.rid) for r in train_records]
        locs = [r.location_id for r in train_records]
        self.train_rids = [r.rid for r in train_records]
        self._oof = {}
        for name, bb in self.backbones.items():
            X = bb.features(imgs)
            for f in fields:
                y = self.raw_targets(train_records, f)
                rd = Reader(self.specs[f], lambdas).fit(X, y, locs)
                self.readers.setdefault(f, {})[name] = rd
                self._oof[(f, name)] = {self.train_rids[i]: rd.oof[j]
                                        for j, i in enumerate(rd.oof_index)}
        return self

    def raw_targets(self, records, f):
        spec = self.specs[f]
        num = lambda x: np.nan if x is None else float(x)
        if spec.kind == "geo":
            return np.array([[num(r.fields.get(spec.keys[0])), num(r.fields.get(spec.keys[1]))]
                             for r in records], float)
        v = np.array([num(spec.raw(r.fields)) for r in records], float)
        return np.where(np.isfinite(v), v, -1).astype(int) if spec.kind == "categorical" else v

    def validate(self, test_records, image_fn, seed=0):
        imgs = [image_fn(r.rid) for r in test_records]
        for name, bb in self.backbones.items():
            X = bb.features(imgs)
            for f, rd in self.readers.items():
                if name not in rd:
                    continue
                y = self.raw_targets(test_records, f)
                self.validation.setdefault(f, {})[name] = validate_reader(rd[name], X, y,
                                                                          self.thresholds, seed)
                self.real_pred.setdefault(f, {})[name] = dict(
                    rids=[r.rid for r in test_records], pred=rd[name].predict(X))
        return self.validation

    def valid_readers(self, f):
        return [n for n, v in self.validation.get(f, {}).items() if v["valid"]]

    def oof(self, f, name, rids):
        d = self._oof[(f, name)]
        geo = self.specs[f].kind == "geo"
        miss = [np.nan, np.nan] if geo else np.nan
        return np.array([d.get(r, miss) for r in rids], float)

    def predict(self, f, name, images):
        X = self.backbones[name].features(images)
        return self.readers[f][name].predict(X)

    def truth(self, f, values_raw):
        any_reader = next(iter(self.readers[f].values()))
        return any_reader.truth(np.asarray(values_raw))
