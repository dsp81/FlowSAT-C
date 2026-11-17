"""Shared machinery for T14 / T15: monthly doy sweeps (CONSISTENT mode), ExG fits, real phases."""
from __future__ import annotations

import math

import numpy as np

from metacf import physical, stats
from metacf.tier1.common import Sweep

YEAR = 365.25
MONTH_DOY = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]      # the 15th of each month


def monthly_sweep(ctx, f_doy, anchors, seeds):
    return Sweep(ctx, f_doy, MONTH_DOY, anchors, seeds, "CONSISTENT")


def fit_sweep(sw, store):
    """-> list of dicts per anchor x seed: phase (days), amplitude, kept (a > seed-to-seed sd)."""
    imgs, (A, K, S) = sw.images(store)
    E = np.array([[[physical.exg_chromatic(imgs[a][k][s]) for s in range(S)] for k in range(K)]
                  for a in range(A)])                                    # (A, K, S)
    out = []
    for a in range(A):
        sd_seed = float(np.nanmean(np.nanstd(E[a], axis=1))) if S > 1 else 0.0
        for s in range(S):
            m, amp, phi, r2 = physical.fit_seasonal(MONTH_DOY, E[a, :, s])
            out.append(dict(anchor=sw.anchors[a], seed=s, phase=phi, amp=amp, r2=r2,
                            kept=bool(np.isfinite(amp) and amp > sd_seed)))
    return out


def circ_mean_days(phases):
    return (stats.circ_mean(2 * math.pi * np.asarray(phases) / YEAR) * YEAR / (2 * math.pi)) % YEAR


def delta_phase(south, north):
    """circular mean(south) - circular mean(north), wrapped to [0, 365.25)."""
    return (circ_mean_days(south) - circ_mean_days(north)) % YEAR


def circ_span_days(doys):
    d = np.sort(np.asarray(doys, float) % YEAR)
    if len(d) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([d, [d[0] + YEAR]]))
    return float(YEAR - gaps.max())


def real_sequence_fits(ctx, f_doy, keep, min_dates=4, min_span=182.6):
    """Natural sequences (TRAIN, same location) with >= 4 dates spanning >= 6 months, filtered by
    keep(record); -> list of (location record, phase, amplitude)."""
    spec = ctx.specs[f_doy]
    out = []
    for rids in ctx.natural_sequences(f_doy, min_distinct=min_dates):
        recs = [ctx.data.by_id[r] for r in rids]
        if not keep(recs[0]):
            continue
        doys = [float(spec.raw(r.fields)) for r in recs]
        if circ_span_days(doys) < min_span:
            continue
        e = [physical.exg_chromatic(ctx.data.real_image(r.rid)) for r in recs]
        m, amp, phi, r2 = physical.fit_seasonal(doys, e)
        if np.isfinite(phi):
            out.append((recs[0], phi, amp))
    return out


def real_crosssection_fit(ctx, f_doy, keep, split="test"):
    """FALLBACK: monthly bins of real images (one per record) -> fit -> (phase, amplitude)."""
    spec = ctx.specs[f_doy]
    recs = [r for r in ctx.data.split(split) if keep(r) and spec.raw(r.fields) is not None]
    if len(recs) < 24:
        return float("nan"), float("nan"), len(recs)
    doys = np.array([float(spec.raw(r.fields)) for r in recs])
    e = np.array([physical.exg_chromatic(ctx.data.real_image(r.rid)) for r in recs])
    month = np.minimum((doys // (YEAR / 12)).astype(int), 11)
    md, me = [], []
    for k in range(12):
        m = month == k
        if m.sum():
            md.append(MONTH_DOY[k]); me.append(e[m].mean())
    _, amp, phi, _ = physical.fit_seasonal(md, me)
    return phi, amp, len(recs)
