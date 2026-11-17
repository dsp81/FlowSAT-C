"""
Shadow estimators for E3 (v2.1, 9 Nov 2026). Inputs: uint8 HxWx3 (sRGB). Image coordinates: x right, y DOWN;
angles atan2(dy, dx) in those coordinates (radians). fMoW chips are taken as north-up: on 1,500 real TEST chips
the direction estimator below puts southern-hemisphere chips on the southern side 85 % of the time
(validate_shadows.py; docs/METACF_V2.md E3), which a wrong orientation could not produce.

Masks (LOCAL contrast, sigma = 6 px Gaussian of linear luminance Lb):
  dark D   = L < 0.65 Lb     (cast shadows are darker than their surroundings)
  bright B = L > 1.15 Lb
  Global masks (the v1 `physical.shadow_masks` percentile masks, or < 0.6 x median) do NOT work on real fMoW:
  median direction error ~90 deg (= chance), anti-solar and sunward run lengths equal (validated 9 Nov).

Direction: R(theta) = mean length of D runs that start right after a B pixel along theta (16 directions,
  k <= 32 px). The dominant bright -> dark transition on real chips is lit ground -> cast shadow, met when
  travelling TOWARD the sun (the shadow lies between the object and the lit ground beyond it), so the first
  circular harmonic of R points sunward; shadow direction = arg(sum_theta R(theta) e^{i theta}) + pi.
  (Measured: without the + pi the median error is ~135 deg on reals.)

Length: mean length (px) of D segments along the expected anti-solar axis (`seg_len`, any start, k <= 64);
  x effective GSD = metres. Validated on reals: log length rises with log cot(e) at fixed category and GSD
  (slope ~0.09, t ~7 on 1,500 TEST chips): attenuated against the physical slope 1 (the mask also catches
  non-shadow darkness), so E3 compares generated to real slopes measured the same way, never to 1.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from metacf.physical import expected_shadow_angle, luminance  # noqa: F401  (re-export)

SIGMA, DARK, BRIGHT = 6.0, 0.65, 1.15
NDIR, KDIR, KSEG = 16, 32, 64
ANGLES = np.arange(NDIR) * 2 * math.pi / NDIR
VEC = NDIR + 3                     # [R(theta) x 16, seg_len_par_px, seg_len_perp_px, dark_fraction]


def masks(img, sigma=SIGMA, dark=DARK, bright=BRIGHT):
    L = luminance(img)
    Lb = ndimage.gaussian_filter(L, sigma)
    return L < dark * Lb, L > bright * Lb


def _shift(M, dx, dy):
    """out[y, x] = M[y + dy, x + dx], False outside."""
    h, w = M.shape
    out = np.zeros_like(M)
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[yd, xd] = M[ys, xs]
    return out


def run_length(D, B, angle, kmax=KDIR):
    """Mean length (px) of D runs along `angle` that start right after a B pixel; (mean, n_starts)."""
    ux, uy = math.cos(angle), math.sin(angle)
    alive = B & _shift(D, int(round(ux)), int(round(uy)))
    n0 = int(alive.sum())
    if n0 == 0:
        return float("nan"), 0
    tot = 0
    for k in range(1, kmax + 1):
        alive = alive & _shift(D, int(round(k * ux)), int(round(k * uy)))
        c = int(alive.sum())
        if c == 0:
            break
        tot += c
    return tot / n0, n0


def seg_len(D, angle, kmax=KSEG):
    """Mean length (px) of D segments along `angle` (segments start where the previous pixel is not D)."""
    ux, uy = math.cos(angle), math.sin(angle)
    starts = D & ~_shift(D, -int(round(ux)), -int(round(uy)))
    n0 = int(starts.sum())
    if n0 == 0:
        return float("nan")
    alive, tot = starts, n0
    for k in range(1, kmax + 1):
        alive = alive & _shift(D, int(round(k * ux)), int(round(k * uy)))
        c = int(alive.sum())
        if c == 0:
            break
        tot += c
    return tot / n0


def shadow_vector(img, axis_angle):
    """VEC floats: R(theta) for the 16 ANGLES, D-segment length along axis_angle (the expected anti-solar
    direction) and across it, dark fraction."""
    D, B = masks(img)
    prof = [run_length(D, B, a)[0] for a in ANGLES]
    return np.array(prof + [seg_len(D, axis_angle), seg_len(D, axis_angle + math.pi / 2), float(D.mean())],
                    np.float32)


def direction_from_vector(v):
    """Shadow direction (radians) from a shadow_vector, or NaN when the profile is undefined."""
    p = np.asarray(v[:NDIR], float)
    if not np.isfinite(p).all() or p.sum() <= 0:
        return float("nan")
    return float(np.angle((p * np.exp(1j * ANGLES)).sum()) + math.pi)


def shadow_direction(img):
    D, B = masks(img)
    return direction_from_vector([run_length(D, B, a)[0] for a in ANGLES])


def angdiff(a, b):
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi
