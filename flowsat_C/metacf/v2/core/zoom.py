"""
MetaCF v2.1 (9 Nov 2026): the GSD reference built from synthetic ZOOM pairs of real chips (fix a).

Why: fMoW crops every chip to its object box, so two real views of one site cover the SAME ground footprint;
their effective-GSD difference (d log g' ~ 0.06, var 0.007) is sensor sharpness, i.e. blur, not zoom. A
reference fitted on same-site pairs therefore asks a model for a blur direction, while a model trained on
eff-GSD (which across sites tracks the footprint, corr 0.87 with log chip scale) renders zoom.

Synthetic pairs from one real chip with centre square side s (native px) and s >= k * res:
  coarse  = the whole square resized to res                         eff gsd g s / res
  fine_k  = centre crop of side s / k resized to res                eff gsd g s / (k res)
  blur_k  = coarse down-sampled to res / k and back up to res       footprint of coarse, detail of g s k / res
Zoom pair (fine_k -> coarse): d log g' = log k, same pixel count, nothing up-sampled (exact zoom-out).
Blur pair (coarse -> blur_k): d log g' = log k at a fixed footprint (the "blur only" alternative).
B_zoom / B_blur = no-intercept OLS of d phi on log k (one row) over TEST chips whose site has >= 2 views
(the C1 TEST site index, so the C1 joint site bootstrap resamples it too), with the C1 split-half reliability.

Feature-store keys: 'syn:<rid>|<kind>|<k>' (kind coarse | fine | blur; coarse uses k = 1).
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

from metacf.config import get
from metacf.v2.core.regression import SiteStats, split_half

KINDS = ("coarse", "fine", "blur")


def syn_key(rid, kind, k=1.0):
    return f"syn:{rid}|{kind}|{float(k):g}"


def parse_key(key):
    rid, kind, k = key[4:].rsplit("|", 2)
    return rid, kind, float(k)


def square(im):
    s = min(im.size)
    left, top = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((left, top, left + s, top + s))


def make_image(sq, kind, k, res):
    """sq: PIL RGB centre square at native resolution -> uint8 (res, res, 3)."""
    from PIL import Image
    s = sq.size[0]
    if kind == "coarse":
        out = sq.resize((res, res), Image.BILINEAR)
    elif kind == "fine":
        c = int(round(s / k))
        off = (s - c) // 2
        out = sq.crop((off, off, off + c, off + c)).resize((res, res), Image.BILINEAR)
    elif kind == "blur":
        r2 = max(8, int(round(res / k)))
        out = sq.resize((res, res), Image.BILINEAR).resize((r2, r2), Image.BILINEAR).resize((res, res), Image.BILINEAR)
    else:
        raise ValueError(kind)
    return np.asarray(out, dtype=np.uint8)


class SynLoader:
    """key -> uint8 image. Full decode (no JPEG draft: fine_k needs the native pixels), centre square kept in
    a small LRU so the 1 + 2 x len(ks) images of one chip decode it once."""

    def __init__(self, by_id, res=512, cache=16, open_fn=None, ds=None):
        self.by_id, self.res, self.ds = by_id, res, ds
        self._c, self._lock, self.cache = OrderedDict(), threading.Lock(), cache
        self.open_fn = open_fn

    def _square(self, rid):
        with self._lock:
            if rid in self._c:
                self._c.move_to_end(rid)
                return self._c[rid]
        import os
        from PIL import Image
        if self.open_fn is not None:
            sq = self.open_fn(rid)
        elif isinstance(getattr(self.by_id[rid], "image", None), str) and os.path.exists(self.by_id[rid].image):
            Image.MAX_IMAGE_PIXELS = None
            sq = square(Image.open(self.by_id[rid].image).convert("RGB"))
        elif self.ds is not None:                  # synthetic worlds (CPU tests): the tile itself is the chip
            sq = Image.fromarray(np.asarray(self.ds.real_image(rid), np.uint8))
        else:
            raise FileNotFoundError(f"no source image for {rid}")
        with self._lock:
            self._c[rid] = sq
            while len(self._c) > self.cache:
                self._c.popitem(last=False)
        return sq

    def __call__(self, key):
        rid, kind, k = parse_key(key)
        return make_image(self._square(rid), kind, k, self.res)


def settings(run):
    z = dict(get(run.cfg, "v2_1.zoom", {}) or {})
    return dict(n_tiles=int(z.get("n_tiles", 800)), ks=[float(k) for k in z.get("ks", [1.25, 1.5, 2.0, 3.0])])


def pairs(run):
    """[(rid, site, k)] over TEST chips whose site is in the C1 TEST site index and whose short side >= k res.
    Chips stratified by category (seeded); every admissible k of a chip is used."""
    def comp():
        from metacf.v2.measures.base import pick_by_category
        st = settings(run)
        res = int(run.ds.resolution)
        sites = set(run.pairs["test"].site_index()[0])
        kmin = min(st["ks"])
        elig = [x for x in run.test if x.location_id in sites and (x.chip_px or 0) >= kmin * res]
        tiles = pick_by_category(elig, st["n_tiles"], run.seed("zoom_tiles"))
        out = [(t.rid, t.location_id, k) for t in tiles for k in st["ks"] if (t.chip_px or 0) >= k * res]
        return out
    return run.cached(("zoom", "pairs"), comp)


def keys(run):
    ps = pairs(run)
    ks = []
    for rid, _, k in ps:
        ks += [syn_key(rid, "coarse"), syn_key(rid, "fine", k), syn_key(rid, "blur", k)]
    return list(dict.fromkeys(ks))


def deltas(run, space, kind):
    """-> (x = log k, Y = d phi, site index into the C1 TEST sites)."""
    ps = pairs(run)
    if kind == "zoom":
        a = [syn_key(r, "fine", k) for r, _, k in ps]
        b = [syn_key(r, "coarse") for r, _, k in ps]
    elif kind == "blur":
        a = [syn_key(r, "coarse") for r, _, k in ps]
        b = [syn_key(r, "blur", k) for r, _, k in ps]
    else:
        raise ValueError(kind)
    Y = run.syn_feats(space, b) - run.syn_feats(space, a)
    x = np.log([k for _, _, k in ps])
    sites = run.pairs["test"].site_index()[0]
    pos = {s: i for i, s in enumerate(sites)}
    sidx = np.array([pos[s] for _, s, _ in ps], int)
    return x, Y, sidx, len(sites)


def reference(run, space, kind="zoom", n_splits=50, ci_splits=10):
    """-> dict(B (1 x d), ss (SiteStats over the C1 TEST site index), rel (split-half dict), n, x, Y, sidx)."""
    def comp():
        x, Y, sidx, S = deltas(run, space, kind)
        ss = SiteStats(x[:, None], Y, sidx, S)
        rel = split_half(ss, {"g": [0]}, n_splits, run.seed("zoomrel", space, kind), run.B, ci_splits)["g"]
        return dict(B=ss.fit(), ss=ss, rel=rel, n=len(x), x=x, Y=Y, sidx=sidx, fit_r2=ss.r2(ss.fit()))
    return run.cached(("zoom", "ref", space, kind), comp)


def scale_reader(run, space, target="zoom"):
    """Ridge reader d phi -> d log(footprint) (target zoom) or d log(blur) (target blur), trained on zoom
    pairs (label log k for zoom, 0 for blur) and blur pairs (0 for zoom, log k for blur), both orientations
    (antisymmetric, so the intercept is ~0). Grouped CV over sites. Validity on a site-held-out half:
    slope of prediction on the true label for zoom pairs and for blur pairs (cross-talk)."""
    def comp():
        from metacf.v2.core.heads import Head
        xz, Yz, sz, _ = deltas(run, space, "zoom")
        xb, Yb, sb, _ = deltas(run, space, "blur")
        lz = xz if target == "zoom" else np.zeros_like(xz)
        lb = np.zeros_like(xb) if target == "zoom" else xb
        X = np.concatenate([Yz, -Yz, Yb, -Yb])
        y = np.concatenate([lz, -lz, lb, -lb])
        g = np.concatenate([sz, sz, sb, sb])
        # validity: fit on half the sites, slopes on the other half
        rng = np.random.default_rng(run.seed("zoomreader", space, target))
        us = np.unique(g)
        half = set(rng.permutation(us)[: len(us) // 2].tolist())
        tr = np.array([s in half for s in g])
        h = Head("continuous").fit(X[tr], y[tr], g[tr])
        tz = np.array([s not in half for s in sz])
        tb = np.array([s not in half for s in sb])
        sl = lambda xx, pp: float(np.sum(xx * pp) / np.sum(xx * xx)) if np.sum(xx * xx) > 0 else float("nan")
        val = dict(slope_on_zoom=sl(xz[tz], h.predict(Yz[tz])), slope_on_blur=sl(xb[tb], h.predict(Yb[tb])),
                   n_test_zoom=int(tz.sum()), n_test_blur=int(tb.sum()))
        full = Head("continuous").fit(X, y, g)
        return dict(head=full, validity=val, lam=full.lam)
    return run.cached(("zoom", "reader", space, target), comp)


def check(run, spaces=None):
    """Real-data sanity table: reliability of B_zoom / B_blur, cos(B_zoom, B_blur), and how the same-site
    real-pair gsd direction B_gsd (C1's v2 reference, TEST pairs) relates to each."""
    from metacf.v2.core.regression import cos
    out = {}
    ps = run.pairs["test"]
    gsd = next((n for n, f in run.fields.items() if f.tag == "gsd" and f.enabled), None)
    for sp in spaces or run.space_names():
        z, b = reference(run, sp, "zoom"), reference(run, sp, "blur")
        row = dict(n_pairs=z["n"], rel_zoom=z["rel"]["mean"], rel_zoom_ci=[z["rel"].get("lo"), z["rel"].get("hi")],
                   rel_blur=b["rel"]["mean"], cos_zoom_blur=cos(z["B"], b["B"]),
                   r2_zoom=z["fit_r2"], r2_blur=b["fit_r2"],
                   norm_zoom=float(np.linalg.norm(z["B"])), norm_blur=float(np.linalg.norm(b["B"])))
        if gsd is not None and ps.n:
            sites, sidx = ps.site_index()
            Y = run.real_feats(sp, ps.b) - run.real_feats(sp, ps.a)
            BR = SiteStats(ps.dM, Y, sidx, len(sites)).fit()
            Bg = BR[run.design.blocks[gsd]]
            row.update(cos_realpairs_zoom=cos(Bg, z["B"]), cos_realpairs_blur=cos(Bg, b["B"]),
                       norm_realpairs=float(np.linalg.norm(Bg)))
        for t in ("zoom", "blur"):
            rd = scale_reader(run, sp, t)
            row[f"reader_{t}"] = dict(lam=rd["lam"], **rd["validity"])
        out[sp] = row
    return out
