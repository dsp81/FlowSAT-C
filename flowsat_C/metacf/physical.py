"""
Physical image measures (the "third reader" in Tier 2). Inputs: uint8 HxWx3 (sRGB). Photometric
quantities are computed on LINEAR luminance (sRGB linearised first). Image coordinates: x to the
right, y DOWN; angles are atan2(dy, dx) in those coordinates, radians. Chips are assumed north-up
(validated on reals by T16b).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from metacf import stats


# ---------------------------------------------------------------- colour


def to_float(img) -> np.ndarray:
    a = np.asarray(img)
    return a.astype(np.float64) / 255.0 if a.dtype == np.uint8 else a.astype(np.float64)


def srgb_to_linear(img) -> np.ndarray:
    c = to_float(img)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def luminance(img, linear=True) -> np.ndarray:
    c = srgb_to_linear(img) if linear else to_float(img)
    return 0.2126 * c[..., 0] + 0.7152 * c[..., 1] + 0.0722 * c[..., 2]


def saturation(img) -> np.ndarray:
    """HSV saturation on sRGB values: (max - min) / max."""
    c = to_float(img)
    mx, mn = c.max(-1), c.min(-1)
    return np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)


def mean_linear_luminance(img) -> float:
    return float(luminance(img).mean())


def exg_chromatic(img) -> float:
    """Chromatic excess green: mean of 2g - r - b, (r, g, b) = channel / channel sum (linear RGB),
    so a uniform brightness change (sun elevation) does not move it."""
    c = srgb_to_linear(img)
    s = c.sum(-1)
    ok = s > 1e-6
    r, g, b = (c[..., i][ok] / s[ok] for i in range(3))
    return float((2 * g - r - b).mean()) if ok.any() else float("nan")


# ---------------------------------------------------------------- shadows (T12, T16)


def shadow_masks(img):
    """D = luminance < 20th pct AND saturation < median; B = luminance > 70th pct."""
    L, S = luminance(img), saturation(img)
    D = (L < np.percentile(L, 20)) & (S < np.median(S))
    B = L > np.percentile(L, 70)
    return D, B


def shadow_fraction(img) -> float:
    return float(shadow_masks(img)[0].mean())


def _offsets(rmin, rmax):
    out = []
    for dy in range(-rmax, rmax + 1):
        for dx in range(-rmax, rmax + 1):
            r = math.hypot(dx, dy)
            if rmin <= r <= rmax:
                out.append((dx, dy))
    return np.array(out)


def tile_direction(B, D, rmin=3, rmax=25, ratio=1.5):
    """
    Shadow direction of one tile: normalised cross-correlation c(o) = sum_x B(x) D(x + o) / overlap(o)
    over offsets o with rmin <= |o| <= rmax (computed by zero-padded FFT). The peak offset is the
    bright-object -> shadow displacement; returns its angle, or None when the peak is < ratio x the
    median correlation over the offsets (no clear shadow) or empty.
    """
    h, w = B.shape
    if B.sum() == 0 or D.sum() == 0:
        return None
    H, W = 2 * h, 2 * w
    FB = np.fft.rfft2(B.astype(float), (H, W))
    FD = np.fft.rfft2(D.astype(float), (H, W))
    corr = np.fft.irfft2(np.conj(FB) * FD, (H, W))          # corr[o] = sum_x B(x) D(x + o)
    ones = np.fft.rfft2(np.ones((h, w)), (H, W))
    overlap = np.fft.irfft2(np.conj(ones) * ones, (H, W))
    offs = _offsets(rmin, rmax)
    iy, ix = offs[:, 1] % H, offs[:, 0] % W
    c = corr[iy, ix] / np.maximum(overlap[iy, ix], 1.0)
    k = int(np.argmax(c))
    med = float(np.median(c))
    if c[k] <= 0 or c[k] < ratio * med:
        return None
    dx, dy = offs[k]
    return float(math.atan2(dy, dx))


def shadow_directions(img, grid=4, rmin=3, rmax=25, ratio=1.5):
    """Per-tile directions (grid x grid tiles); list of angles for the tiles that pass."""
    D, B = shadow_masks(img)
    h, w = D.shape
    th, tw = h // grid, w // grid
    out = []
    for i in range(grid):
        for j in range(grid):
            sl = (slice(i * th, (i + 1) * th), slice(j * tw, (j + 1) * tw))
            a = tile_direction(B[sl], D[sl], rmin, rmax, ratio)
            if a is not None:
                out.append(a)
    return out


def rotate_tiles(img, grid=4, seed=0):
    """Each tile rotated by an independent uniform random multiple of 90 deg (T12 floor:
    simulated multiple suns)."""
    rng = np.random.default_rng(seed)
    a = np.array(img).copy()
    h, w = a.shape[:2]
    th, tw = h // grid, w // grid
    assert th == tw, "tile rotation needs square tiles"
    for i in range(grid):
        for j in range(grid):
            sl = (slice(i * th, (i + 1) * th), slice(j * tw, (j + 1) * tw))
            a[sl] = np.rot90(a[sl], int(rng.integers(0, 4)))
    return a


def compass_to_image_angle(bearing_deg):
    """Compass bearing (clockwise from north) -> image angle for a north-up chip
    (dx = sin b, dy = -cos b, y down)."""
    b = np.radians(np.asarray(bearing_deg, float))
    return np.arctan2(-np.cos(b), np.sin(b))


def expected_shadow_angle(sun_azimuth_deg):
    """Shadows point away from the sun: bearing = azimuth + 180."""
    return compass_to_image_angle(np.asarray(sun_azimuth_deg, float) + 180.0)


# ---------------------------------------------------------------- GSD (T13)


def sharpness_h(img, cutoff=0.25) -> float:
    """Fraction of FFT power (Hann window, mean removed, DC excluded) at radial frequency
    > cutoff cycles/px."""
    L = luminance(img)
    L = L - L.mean()
    h, w = L.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    P = np.abs(np.fft.fft2(L * win)) ** 2
    fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    r = np.hypot(fx, fy)
    P[0, 0] = 0.0
    tot = P.sum()
    return float(P[r > cutoff].sum() / tot) if tot > 0 else float("nan")


def radial_autocorr(img):
    """Radially averaged normalised autocorrelation of mean-removed luminance (zero-padded FFT,
    unbiased by overlap). Returns (radii px, acf)."""
    L = luminance(img)
    L = L - L.mean()
    h, w = L.shape
    F = np.fft.rfft2(L, (2 * h, 2 * w))
    ac = np.fft.irfft2(np.abs(F) ** 2, (2 * h, 2 * w))
    ones = np.fft.rfft2(np.ones((h, w)), (2 * h, 2 * w))
    ov = np.fft.irfft2(np.abs(ones) ** 2, (2 * h, 2 * w))
    ac = ac / np.maximum(ov, 1.0)
    ac = ac / ac[0, 0]
    ac = np.fft.fftshift(ac)
    cy, cx = h, w
    rmax = min(h, w) // 2
    yy, xx = np.indices(ac.shape)
    rr = np.hypot(yy - cy, xx - cx)
    ri = np.round(rr).astype(int)
    m = ri <= rmax
    sums = np.bincount(ri[m], ac[m], minlength=rmax + 1)
    cnt = np.bincount(ri[m], minlength=rmax + 1)
    return np.arange(rmax + 1), sums / np.maximum(cnt, 1)


def autocorr_scale(img) -> float:
    """First crossing of 1/e of the radial ACF, linearly interpolated, px."""
    r, a = radial_autocorr(img)
    thr = 1.0 / math.e
    below = np.where(a < thr)[0]
    if len(below) == 0:
        return float("nan")
    k = int(below[0])
    if k == 0:
        return 0.0
    a0, a1 = a[k - 1], a[k]
    return float(r[k - 1] + (a0 - thr) / (a0 - a1) * (r[k] - r[k - 1]))


# ---------------------------------------------------------------- seasonal fit (T14, T15)


def fit_seasonal(doys, values, period=365.25):
    """Least squares value = m + a cos(2 pi (doy - phi) / period). Returns (m, a, phi_days, r2)."""
    d = np.asarray(doys, float)
    y = np.asarray(values, float)
    ok = np.isfinite(d) & np.isfinite(y)
    d, y = d[ok], y[ok]
    if len(d) < 3:
        return (float("nan"),) * 4
    w = 2 * math.pi / period
    X = np.column_stack([np.ones_like(d), np.cos(w * d), np.sin(w * d)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    m, A, B = beta
    a = math.hypot(A, B)
    phi = (math.atan2(B, A) / w) % period
    res = y - X @ beta
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (res @ res) / sst if sst > 0 else float("nan")
    return float(m), float(a), float(phi), float(r2)


# ---------------------------------------------------------------- clouds (T17)


def local_std(x, size=7):
    m = ndimage.uniform_filter(x, size)
    m2 = ndimage.uniform_filter(x * x, size)
    return np.sqrt(np.clip(m2 - m * m, 0, None))


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def cloud_mask(img, lum_pct=80, sat_pct=30, std_pct=30, window=7, dilate=5):
    """luminance > 80th pct AND saturation < 30th pct AND local std (7 px) < its 30th pct,
    dilated by 5 px."""
    L, S = luminance(img), saturation(img)
    sd = local_std(L, window)
    m = (L > np.percentile(L, lum_pct)) & (S < np.percentile(S, sat_pct)) & \
        (sd < np.percentile(sd, std_pct))
    if dilate > 0:
        m = ndimage.binary_dilation(m, structure=disk(dilate))
    return m


def mask_iou(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def circular_directions_summary(angles):
    """(circular mean, circular variance) of a list of angles."""
    if len(angles) == 0:
        return float("nan"), float("nan")
    return stats.circ_mean(angles), stats.circ_var(angles)
