"""
T5 CONTROL RESOLUTION. Mutual information between a discrete variable and the read-back: Ross (2014)
k-NN mixed estimator (k = 3) for continuous / circular read-backs; plug-in MI with Miller-Madow for
categorical read-backs. Bits.
T5a within-scene: K = 7 sweep (grid at the 5/20/35/50/65/80/95th TRAIN percentiles), pooled over
     anchor x seed sweeps: MI(set index ; read-back - anchor mean read-back). Reports bits,
     levels = 2^bits, and bits / log2 K (raw = bits, floor = 0, ceiling = log2 K).
T5b across-scene: on T1 generations, MI(quantile bin of r's own value, K = 7 ; read-back).
     floor = the same on T1's f-shuffled generations (bin of r's OWN value); ceiling = real TEST.
Switch index per anchor x seed: max_k |rb_{k+1} - rb_k| / (max rb - min rb); the median is
reported and flagged "switch-like" when > 0.7.
"""
from __future__ import annotations

import math

import numpy as np

from metacf import mi, stats
from metacf.base import MetaCFTest, na, row
from metacf.tier1.common import Sweep, sweep_kind
from metacf.tier1.t1_recovery import T1Recovery

P7 = (5, 20, 35, 50, 65, 80, 95)


def residual(kind, R):
    """read-back minus its anchor mean (over k and seeds); circular wrapped."""
    out = np.empty_like(R)
    for a in range(R.shape[0]):
        if kind == "circular":
            out[a] = stats.wrap(R[a] - stats.circ_mean(R[a].ravel()))
        else:
            out[a] = R[a] - np.nanmean(R[a])
    return out


def switch_index(rb):
    rb = np.asarray(rb, float)
    rng = np.nanmax(rb) - np.nanmin(rb)
    if not np.isfinite(rng) or rng <= 0:
        return float("nan")
    return float(np.nanmax(np.abs(np.diff(rb))) / rng)


def mi_any(kind, x, y, k, seed):
    """Categorical read-backs: plug-in MI (Miller-Madow). Circular read-backs: Ross on the 2-D
    (cos, sin) embedding, so the estimate has no seam at +-pi. Otherwise Ross on the value."""
    if kind == "categorical":
        return mi.mi_discrete_discrete(x, y)
    y = np.asarray(y, float)
    if kind == "circular":
        y = np.column_stack([np.cos(y), np.sin(y)])
    return mi.mi_discrete_continuous(x, y, k=k, seed=seed)


def theoretical_bits(R, res):
    """MI implied by the measured signal and noise under a Gaussian channel: per set level k the
    residual's mean and sd over anchors x seeds -> mi.gaussian_channel_mi. For a planted linear
    response with Gaussian noise this is the true MI; it calibrates the k-NN estimate."""
    K = R.shape[1]
    m = [np.nanmean(res[:, k, :]) for k in range(K)]
    sd = [np.nanstd(res[:, k, :]) for k in range(K)]
    return mi.gaussian_channel_mi(m, sd)


def planted_bits(ctx, f, grid, R):
    """Theoretical T5a bits from the adapter's PLANTED signal (None for real models): the levels are
    the planted read-backs (centred), the noise is the only noise the synthetic model plants, the
    seed texture, measured as the pooled within-(anchor, level) seed-to-seed sd of the read-back.
    Gaussian channel with equal noise -> mi.gaussian_channel_mi. Assumes a calibrated reader (unit
    slope), which T4 checks. Returns (bits, noise_sd) or (nan, nan)."""
    sig = ctx.adapter.planted_signal(f, grid)
    if sig is None or R.shape[2] < 2:
        return float("nan"), float("nan")
    m = np.asarray(ctx.encode_set(f, sig), float)
    var = np.nanvar(R, axis=2, ddof=1)                                   # (A, K)
    sd = float(np.sqrt(np.nanmean(var)))
    return mi.gaussian_channel_mi(m - m.mean(), [sd] * len(m)), sd


class T5Resolution(MetaCFTest):
    id = "T5"
    name = "control resolution"

    def sweep7(self, ctx, f):
        grid = ctx.grid(f, P7)
        anchors = ctx.anchors(f, grid, tag="K7")
        return Sweep(ctx, f, grid, anchors, ctx.anchor_seeds(), "RAW")

    def plan(self, ctx):
        reqs = [r for f in self.swept(ctx) for r in self.sweep7(ctx, f).requests()]
        return reqs + T1Recovery(self.cfg).plan(ctx)

    def compute(self, ctx, store):
        out = self.na_unsupported(ctx)
        k = int(self.tc.get("k", 3))
        t1 = T1Recovery(self.cfg)
        for f in self.fields(ctx):
            kind = sweep_kind(ctx, f)
            for rd in ctx.readers.valid_readers(f):
                if ctx.specs[f].recovery_only:
                    out.append(na(self.id, f, "recovery-only by design: no T5a sweep", reader=rd, stratum="T5a"))
                    out.append(self._t5b(ctx, store, t1, f, rd, kind, k))
                    continue
                sw = self.sweep7(ctx, f)
                K = len(sw.grid)
                if len(sw.anchors) < int(self.tc.get("min_n", 5)) or K < 2:
                    out.append(na(self.id, f, f"{len(sw.anchors)} anchors / K={K} in support", reader=rd))
                    continue
                R = sw.readbacks(store, rd)                           # (A, K, S)
                res = residual(kind, R)
                xidx = np.broadcast_to(np.arange(K)[None, :, None], R.shape).ravel()
                y = res.ravel() if kind != "categorical" else R.ravel()
                bits = mi_any(kind, xidx, y, k, ctx.seed("T5", f, rd))
                fn_a = lambda i: mi_any(kind, xidx.reshape(R.shape)[i].ravel(),
                                        (res if kind != "categorical" else R)[i].ravel(), k, 0)
                ci = stats.bootstrap(fn_a, R.shape[0], self.B, self.boot_seed(ctx, f, rd, "a"))
                sws = [switch_index(R[a, :, s]) for a in range(R.shape[0]) for s in range(R.shape[2])]
                med_sw = stats.median_ignore_nan(sws)
                theo = theoretical_bits(R, res) if kind == "continuous" else float("nan")
                planted, noise_sd = planted_bits(ctx, f, sw.grid, R) if kind == "continuous" else \
                    (float("nan"), float("nan"))
                out.append(row(self.id, f, rd, stratum="T5a", raw=bits, floor=0.0, ceiling=math.log2(K),
                               bits_theoretical_gaussian=theo, bits_theoretical_planted=planted,
                               planted_noise_sd=noise_sd,
                               ci=ci, ci_on="raw", n=R.shape[0], seeds=R.shape[2],
                               levels=2 ** bits if np.isfinite(bits) else None,
                               switch_index=med_sw, switch_like=bool(med_sw > 0.7) if np.isfinite(med_sw) else None,
                               K=K))
                out.append(self._t5b(ctx, store, t1, f, rd, kind, k))
        return out

    def _t5b(self, ctx, store, t1, f, rd, kind, k):
        """Bins: categorical -> the class; circular -> K equal arcs (origin = TRAIN circular mean - pi,
        so the seam sits where data is sparsest); continuous -> TRAIN quantiles; geo -> TRAIN
        latitude quantiles (the read-back's latitude component)."""
        A = t1.arrays(ctx, store, f, rd)
        spec = ctx.specs[f]
        K = int(self.tc.get("K_bins", 7))
        if spec.kind == "geo":
            own = A["truth"][:, 0]
            comp = lambda x: np.asarray(x)[..., 0]
            tv = np.array([r.fields.get(spec.keys[0]) for r in ctx.train
                           if r.fields.get(spec.keys[0]) is not None], float)
            bins = mi.quantile_bins(own, K, edges_from=tv)
            kind_y = "continuous"
        elif spec.kind == "categorical":
            own = A["truth"]
            comp = lambda x: np.asarray(x)
            bins, kind_y = own.astype(int), "categorical"
        elif spec.kind == "circular":
            own = A["truth"]
            comp = lambda x: np.asarray(x)
            tv = spec.encode(ctx.train_values(f))
            origin = stats.circ_mean(tv) - math.pi
            bins, kind_y = mi.circular_bins(own, K, origin), "circular"
        else:
            own = A["truth"]
            comp = lambda x: np.asarray(x)
            bins = mi.quantile_bins(own, K, edges_from=spec.encode(ctx.train_values(f)))
            kind_y = "continuous"
        S, SF = A["gen"].shape[1], A["floor"].shape[1]
        g = comp(A["gen"]).reshape(-1)
        fl = comp(A["floor"]).reshape(-1)
        rl = comp(A["real"])
        seed = ctx.seed("T5b", f, rd)
        raw = mi_any(kind_y, np.repeat(bins, S), g, k, seed)
        flo = mi_any(kind_y, np.repeat(bins, SF), fl, k, seed)
        ceil = mi_any(kind_y, bins, rl, k, seed)
        return row(self.id, f, rd, stratum="T5b", raw=raw, floor=flo, ceiling=ceil, n=len(bins), seeds=S,
                   notes={"circular": "K equal circular arcs", "geo": "latitude quantile bins"}.get(spec.kind, "MEASURED"))
