"""Sweeps shared by T2/T3/T4/T5/T6 and the per-sweep statistics they need."""
from __future__ import annotations

import numpy as np

from metacf import stats
from metacf.config import get
from metacf.solar import DRIVERS


class Sweep:
    """
    One field's sweep: anchors x grid values x seeds, all requests built through the context (so
    identical sweeps in different tests share cache keys and are generated once).
    readbacks() -> array (A, K, S) in the sweep space (transformed; circular radians; geo latitude).
    """

    def __init__(self, ctx, f, grid, anchors, seeds, mode="RAW", caption_mode="real"):
        self.ctx, self.f, self.grid, self.anchors, self.seeds = ctx, f, list(grid), list(anchors), list(seeds)
        self.mode, self.caption_mode = mode, caption_mode
        self.reqs = ctx.sweep(f, self.anchors, self.grid, self.seeds, mode, caption_mode)
        self.set_values = ctx.encode_set(f, self.grid)                  # (K,)

    def requests(self):
        return list(self.reqs.values())

    def images(self, store):
        A, K, S = len(self.anchors), len(self.grid), len(self.seeds)
        return [[[store.get(self.reqs[(a, k, s)]) for s in self.seeds] for k in range(K)]
                for a in self.anchors], (A, K, S)

    def flat_images(self, store):
        out, idx = [], []
        for ai, a in enumerate(self.anchors):
            for k in range(len(self.grid)):
                for si, s in enumerate(self.seeds):
                    out.append(store.get(self.reqs[(a, k, s)]))
                    idx.append((ai, k, si))
        return out, idx

    def readbacks(self, store, reader):
        imgs, idx = self.flat_images(store)
        pred = self.ctx.sweep_readback(self.f, reader, imgs, [self.anchors[i[0]] for i in idx])
        R = np.full((len(self.anchors), len(self.grid), len(self.seeds)), np.nan)
        for (ai, k, si), p in zip(idx, pred):
            R[ai, k, si] = p
        return R

    def phi(self, store):
        imgs, idx = self.flat_images(store)
        X = self.ctx.phi_feats(imgs)
        out = np.zeros((len(self.anchors), len(self.grid), len(self.seeds), X.shape[1]))
        for (ai, k, si), x in zip(idx, X):
            out[ai, k, si] = x
        return out


def standard_sweep(ctx, f, mode="RAW", caption_mode="real"):
    """The default sweep: K = 5 grid at TRAIN 10/30/50/70/90th percentiles, 20 anchors, 3 seeds."""
    grid = ctx.grid(f)
    anchors = ctx.anchors(f, grid)
    return Sweep(ctx, f, grid, anchors, ctx.anchor_seeds(), mode, caption_mode)


def sweep_modes(ctx, f, only=None):
    """RAW always; CONSISTENT too when the field drives derived solar fields (lat, lon, doy) and,
    if `only` is given (tests.T3.consistent_fields), the field is listed there. Non-driver fields
    are never run CONSISTENT: their CONSISTENT sweep would equal RAW under a different cache key."""
    keys = set(ctx.specs[f].keys)
    if not keys & set(DRIVERS):
        return ["RAW"]
    if only is not None and f not in only:
        return ["RAW"]
    return ["RAW", "CONSISTENT"]


def sweep_statistic(kind, set_vals, readback):
    """T3 per-sweep statistic. continuous / geo-latitude: Spearman; circular: Fisher-Lee;
    categorical: switching accuracy (fraction of k with read-back class = set class)."""
    if kind == "categorical":
        ok = np.isfinite(readback)
        return float((readback[ok] == set_vals[ok]).mean()) if ok.any() else float("nan")
    rb = np.asarray(readback, float)
    if np.isfinite(rb).sum() >= 3 and np.nanmax(rb) - np.nanmin(rb) <= 1e-12:
        return 0.0                       # a constant response carries no monotonic control
    if kind == "circular":
        return stats.fisher_lee(set_vals, readback)
    return stats.spearman(set_vals, readback)


def aggregate(kind, values):
    """Fisher-z mean over sweeps (plain mean for switching accuracy)."""
    if kind == "categorical":
        v = np.asarray(values, float)
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) else float("nan")
    return stats.fisher_z_mean(values)


def sweep_kind(ctx, f):
    k = ctx.specs[f].kind
    return "continuous" if k == "geo" else k


def centred(kind, x):
    """For slopes: circular values are unwrapped around their circular mean."""
    x = np.asarray(x, float)
    if kind == "circular":
        return stats.wrap(x - stats.circ_mean(x))
    return x


def boot_B(cfg):
    return int(get(cfg, "bootstrap.B", 1000))
