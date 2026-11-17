"""Requested-vs-read-off GSD curve of one model (MetaCF v2.3 GSD panel), drawn with PIL (no matplotlib in the env).
Identical axes for every model: log-log, 0.25-4 m on both. Shown: the model's mean read-off per requested level
(+- 1 sd of the log read-off across sweeps), the identity (a perfect GSD control), the REAL within-scene reference
(what the reader reports when a real scene is zoomed: slope c_within in log-log, anchored at the TRAIN median),
and the TRAIN [1, 99] % range of effective GSD (shaded)."""
from __future__ import annotations

import math

LO, HI = 0.25, 4.0
W, H, ML, MR, MT, MB = 760, 560, 78, 24, 46, 62
TICKS = [0.25, 0.5, 1.0, 2.0, 4.0]


def _x(v):
    return ML + (math.log(v) - math.log(LO)) / (math.log(HI) - math.log(LO)) * (W - ML - MR)


def _y(v):
    v = min(max(v, LO), HI)
    return H - MB - (math.log(v) - math.log(LO)) / (math.log(HI) - math.log(LO)) * (H - MT - MB)


def draw(profile, c_within, train_median, train_p1_p99, title, path, step_gsd=None):
    from PIL import Image, ImageDraw, ImageFont
    S = 2                                           # supersample
    im = Image.new("RGB", (W * S, H * S), "white")
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13 * S)
        fb = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14 * S)
    except Exception:
        font = fb = ImageFont.load_default()
    P = lambda x, y: (x * S, y * S)
    lo, hi = max(train_p1_p99[0], LO), min(train_p1_p99[1], HI)
    d.rectangle([P(_x(lo), MT), P(_x(hi), H - MB)], fill=(236, 242, 250))
    for t in TICKS:
        d.line([P(_x(t), MT), P(_x(t), H - MB)], fill=(222, 222, 222), width=S)
        d.line([P(ML, _y(t)), P(W - MR, _y(t))], fill=(222, 222, 222), width=S)
        d.text(P(_x(t) - 12, H - MB + 6), f"{t:g}", fill="black", font=font)
        d.text(P(ML - 40, _y(t) - 8), f"{t:g}", fill="black", font=font)
    d.rectangle([P(ML, MT), P(W - MR, H - MB)], outline="black", width=S)
    # identity (dashed)
    n = 60
    for i in range(0, n, 2):
        a = LO * (HI / LO) ** (i / n)
        b = LO * (HI / LO) ** ((i + 1) / n)
        d.line([P(_x(a), _y(a)), P(_x(b), _y(b))], fill=(120, 120, 120), width=S)
    # real within-scene reference
    pts = [P(_x(v), _y(train_median * (v / train_median) ** c_within)) for v in
           [LO * (HI / LO) ** (i / n) for i in range(n + 1)]]
    d.line(pts, fill=(40, 150, 70), width=2 * S)
    # model curve with +- 1 sd bars
    col = (200, 50, 40)
    xs = [p["requested_m"] for p in profile]
    ys = [p["read_m"] for p in profile]
    for p in profile:
        a, b = p["read_m"] * math.exp(-p["sd_log"]), p["read_m"] * math.exp(p["sd_log"])
        d.line([P(_x(p["requested_m"]), _y(a)), P(_x(p["requested_m"]), _y(b))], fill=(235, 170, 165), width=2 * S)
    d.line([P(_x(a), _y(b)) for a, b in zip(xs, ys)], fill=col, width=3 * S)
    for a, b in zip(xs, ys):
        d.ellipse([P(_x(a) - 4, _y(b) - 4), P(_x(a) + 4, _y(b) + 4)], fill=col)
    if step_gsd:
        d.line([P(_x(step_gsd), MT), P(_x(step_gsd), H - MB)], fill=(200, 50, 40), width=S)
        d.text(P(_x(step_gsd) + 4, MT + 4), f"largest rise at {step_gsd:.2f} m", fill=col, font=font)
    d.text(P(ML, 14), title, fill="black", font=fb)
    d.text(P(ML + 170, H - 26), "requested effective GSD (m)", fill="black", font=font)
    d.text(P(6, MT - 22), "read-off GSD (m)", fill="black", font=font)
    lx, ly = W - MR - 300, H - MB - 96
    for k, (c, lab) in enumerate([(col, "model: mean read-off (bars: 1 sd)"), ((40, 150, 70), "real scene zoomed (reader slope)"),
                                  ((120, 120, 120), "identity (dashed)"), ((170, 195, 230), "TRAIN 1-99% range of GSD")]):
        d.rectangle([P(lx, ly + 22 * k + 4), P(lx + 22, ly + 22 * k + 12)], fill=c)
        d.text(P(lx + 30, ly + 22 * k), lab, fill="black", font=font)
    im = im.resize((W, H), Image.LANCZOS)
    im.save(path)
    return path
