"""
Real-data validation of the E3 shadow estimators (physics/shadows.py), CPU only.

  python -m metacf.v2.physics.validate_shadows --manifest ../manifests/test_cov.jsonl \
      --n 1500 --n-south 400 --out <dir> [--workers 4]

Samples TEST records with azimuth_valid (oversamples |lat| > 15 S so the azimuth range is not one arc), loads
each chip with the training transform (centre crop to short side, 512 px), and records per image:
  direction estimate + contrast, expected anti-solar angle (north-up assumption), run length along the
  expected direction (px and m), relative-threshold shadow fraction, v1 percentile shadow fraction.
Writes rows.jsonl + summary.json:
  direction: median |error|, fraction < 45 deg, Fisher-Lee-free check = mean cos(error), and the same for
    a CONSTANT predictor (circular mean of the expected angles) and for the north/south subsets separately:
    the estimator tracks the sun only if it beats the constant predictor, above all on the southern subset.
  length: OLS of log(run length m) on log cot(e) with category fixed effects (+ log eff gsd), slope + p;
    Spearman(cot e, length) overall and within category; same for both shadow fractions.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")


def _load(rec, res=512):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(rec["image"])
    if im.format == "JPEG":
        im.draft("RGB", (2 * res, 2 * res))
    im = im.convert("RGB")
    s = min(im.size)
    l, t = (im.width - s) // 2, (im.height - s) // 2
    return np.asarray(im.crop((l, t, l + s, t + s)).resize((res, res), Image.BILINEAR), dtype=np.uint8)


def one(rec):
    from metacf import physical
    from metacf.v2.physics import shadows as S
    try:
        img = _load(rec)
    except Exception as e:  # noqa: BLE001
        return dict(key=rec["key"], error=repr(e))
    exp = float(physical.expected_shadow_angle(rec["sun_azimuth_deg"]))
    v = S.shadow_vector(img, exp)
    eff = float(rec["gsd_m"]) * min(rec["img_w"], rec["img_h"]) / 512.0
    v1dirs = physical.shadow_directions(img)
    v1 = physical.circular_directions_summary(v1dirs)[0] if len(v1dirs) >= 3 else None
    est = S.direction_from_vector(v)
    return dict(key=rec["key"], category=rec["category"], lat=rec["lat_deg"], az=rec["sun_azimuth_deg"],
                elev=rec["sun_elev_deg"], eff_gsd=eff, exp=exp, est=est if np.isfinite(est) else None,
                v1_est=v1, len_px=float(v[S.NDIR]), len_perp_px=float(v[S.NDIR + 1]), frac_rel=float(v[S.NDIR + 2]),
                frac_v1=physical.shadow_fraction(img))


def _hemi_acc(sub, key, rows):
    """Fraction of the chips in `sub` whose estimate is closer to their own hemisphere's mean expected angle
    than to the other hemisphere's (|lat| > 15 only): 0.5 = chance, needs no north-up-free constant."""
    from metacf.v2.physics.shadows import angdiff
    ex = lambda f: math.atan2(np.mean([math.sin(r["exp"]) for r in rows if f(r)]),
                              np.mean([math.cos(r["exp"]) for r in rows if f(r)]))
    mN, mS = ex(lambda r: r["lat"] > 15), ex(lambda r: r["lat"] < -15)
    hits = [(abs(angdiff(r[key], mS)) < abs(angdiff(r[key], mN))) == (r["lat"] < -15) for r in sub if abs(r["lat"]) > 15]
    return float(np.mean(hits)) if hits else float("nan")


def summarise(rows):
    from scipy import stats as ss
    from metacf.v2.physics.shadows import angdiff
    rows = [r for r in rows if "error" not in r]
    out = dict(n=len(rows))

    def dirstats(sub, key):
        ok = [r for r in sub if r[key] is not None and np.isfinite(r[key])]
        if not ok:
            return dict(n=0)
        e = np.abs(angdiff([r[key] for r in ok], [r["exp"] for r in ok]))
        ex = np.array([r["exp"] for r in sub])
        const = math.atan2(np.sin(ex).mean(), np.cos(ex).mean())
        ec = np.abs(angdiff(np.full(len(ok), const), [r["exp"] for r in ok]))
        return dict(n=len(ok), pass_rate=len(ok) / max(len(sub), 1), median_err_deg=float(np.degrees(np.median(e))),
                    hemisphere_acc=_hemi_acc(ok, key, rows),
                    frac_lt45=float((e < math.pi / 4).mean()), mean_cos=float(np.cos(e).mean()),
                    const_median_err_deg=float(np.degrees(np.median(ec))), const_frac_lt45=float((ec < math.pi / 4).mean()),
                    const_mean_cos=float(np.cos(ec).mean()))
    for key in ("est", "v1_est"):
        out[f"direction_{key}"] = {nm: dirstats(sub, key) for nm, sub in (
            ("all", rows), ("north_lat>15", [r for r in rows if r["lat"] > 15]),
            ("south_lat<-15", [r for r in rows if r["lat"] < -15]),
            ("tropics", [r for r in rows if abs(r["lat"]) <= 15]))}
    # length
    def lenstats(key, metres):
        ok = [r for r in rows if np.isfinite(r[key]) and r[key] > 0 and 1 <= r["elev"] < 89]
        y = np.log(np.array([r[key] * (r["eff_gsd"] if metres else 1.0) for r in ok]))
        x = np.log(1 / np.tan(np.radians([r["elev"] for r in ok])))
        g = np.log([r["eff_gsd"] for r in ok])
        cats = sorted({r["category"] for r in ok})
        FE = np.array([[r["category"] == c for c in cats[1:]] for r in ok], float)
        X = np.column_stack([np.ones(len(ok)), x, g, FE])
        b, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        dof = len(y) - X.shape[1]
        s2 = float(((y - X @ b) ** 2).sum() / dof)
        cov = s2 * np.linalg.pinv(X.T @ X)
        se = math.sqrt(cov[1, 1])
        t = b[1] / se
        # within-category Spearman (centred ranks)
        rho = ss.spearmanr(x, y).correlation
        wc = []
        for c in cats:
            idx = [i for i, r in enumerate(ok) if r["category"] == c]
            if len(idx) >= 15:
                wc.append((len(idx), ss.spearmanr(x[idx], y[idx]).correlation))
        wcm = float(np.average([v for _, v in wc], weights=[n for n, _ in wc])) if wc else float("nan")
        return dict(n=len(ok), slope_logcot=float(b[1]), se=se, t=float(t), p_one_sided=float(ss.t.sf(t, dof)),
                    slope_log_effgsd=float(b[2]), spearman=float(rho), within_cat_spearman=wcm)
    out["length_m"] = lenstats("len_px", True)
    out["length_px"] = lenstats("len_px", False)
    out["length_perp_px_control"] = lenstats("len_perp_px", False)
    out["frac_rel"] = lenstats("frac_rel", False)
    out["frac_v1"] = lenstats("frac_v1", False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--n-south", type=int, default=400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    recs = []
    with open(a.manifest) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("azimuth_valid") and r.get("sun_azimuth_deg") is not None and r.get("sun_elev_deg") is not None:
                recs.append({k: r[k] for k in ("key", "image", "category", "lat_deg", "sun_azimuth_deg", "sun_elev_deg",
                                               "gsd_m", "img_w", "img_h")})
    rng = np.random.default_rng(a.seed)
    south = [r for r in recs if r["lat_deg"] < -15]
    rest = [r for r in recs if r["lat_deg"] >= -15]
    pick = [south[i] for i in rng.choice(len(south), min(a.n_south, len(south)), replace=False)] + \
           [rest[i] for i in rng.choice(len(rest), a.n - min(a.n_south, len(south)), replace=False)]
    print(f"{len(recs)} eligible; {len(south)} south; scoring {len(pick)}", flush=True)
    from multiprocessing import Pool
    rows = []
    with Pool(a.workers) as p, open(os.path.join(a.out, "rows.jsonl"), "w") as fh:
        for i, r in enumerate(p.imap_unordered(one, pick, chunksize=4)):
            rows.append(r)
            fh.write(json.dumps(r, default=float) + "\n")
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{len(pick)}", flush=True)
    s = summarise(rows)
    json.dump(s, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps(s, indent=1, default=float))


if __name__ == "__main__":
    main()
