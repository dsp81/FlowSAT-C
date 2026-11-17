"""
CONSISTENT sweep mode: when lat, lon, doy or time changes, recompute the derived solar fields.

Reuses flowsat_C.models.encoder.derived_solar (closed-form NOON elevation and photoperiod). The
records carry MEASURED sun elevation, which includes time of day; we preserve the record's time of
day by keeping its offset from noon elevation:

    sun_elev_new = clip(alpha_noon(lat_new, doy_new) + (sun_elev_old - alpha_noon(lat_old, doy_old)),
                        1, 90)
    photoperiod_new = photoperiod(lat_new, doy_new)

Sun azimuth is NOT recomputed (it needs the time of day, which fMoW does not record): INFERRED,
listed in docs/OPEN_QUESTIONS.md. Longitude does not enter the noon formulas, so a longitude-only
change leaves the solar fields unchanged. A `time` field does not exist in fMoW -> N/A.
"""
from __future__ import annotations

import math

import numpy as np

DRIVERS = ("lat_deg", "lon_deg", "doy", "time")


def _derived(lat, doy):
    try:
        import torch
        from flowsat_C.models.encoder import derived_solar
        a, p = derived_solar(torch.tensor([float(lat)]), torch.tensor([float(doy)]))
        return float(a[0]), float(p[0])
    except Exception:                                   # identical closed form, numpy copy
        phi = math.radians(lat)
        delta = math.radians(23.44 * math.sin(2 * math.pi * (doy - 81.0) / 365.0))
        alpha = 90.0 - abs(lat - math.degrees(delta))
        cos_w = max(-1.0, min(1.0, -math.tan(phi) * math.tan(delta)))
        return alpha, 2.0 * math.degrees(math.acos(cos_w)) / 15.0


def consistent(old: dict, new: dict) -> dict:
    """`new` = old fields with one driver changed; returns new with derived solar recomputed."""
    if not any(new.get(k) != old.get(k) for k in DRIVERS if k in new):
        return new
    out = dict(new)
    lat0, doy0 = old.get("lat_deg"), old.get("doy")
    lat1, doy1 = new.get("lat_deg"), new.get("doy")
    if None in (lat0, doy0, lat1, doy1):
        return out
    a0, _ = _derived(lat0, doy0)
    a1, p1 = _derived(lat1, doy1)
    e0 = old.get("sun_elev_deg")
    if e0 is not None and np.isfinite(e0):
        out["sun_elev_deg"] = float(np.clip(a1 + (e0 - a0), 1.0, 90.0))
    if "photoperiod_h" in out:
        out["photoperiod_h"] = p1
    return out
