"""
E1 RESOLUTION (spec section 3).

  Generated: n_anchors (50) TEST anchors x n_values (5) log-spaced effective GSD values inside the
  anchor category's TRAIN [1st, 99th] support x 1 seed. Per image: HF fraction h and ACF length l (1.1).
  Per anchor OLS of log h and log l on log g'; generated slopes = medians over anchors.
  Real: OLS on n_real TEST real tiles of log h and log l on log g' with category fixed effects, pooled and
  per rho stratum (rho = min(W, H) / 512 >= 1 vs < 1).
  Score = max(0, 1 - |b_l,gen - b_l,real| / |b_l,real|) (pooled real slope). Also b_h, and the label:
  "blur only" when |b_l,gen| < 0.2 |b_l,real| while b_h,gen has the real sign, else "zoom" when b_l,gen
  has the real sign, else "wrong sign".
  Gate: the real pooled b_l significant (two-sided OLS p, BH).

v2_1 (9 Nov 2026, `v2_1.e1_zoom: true`): extra rows on the SAME E1 generations (large, log-spaced steps
across the category support), scored against the zoom reference of metacf/v2/core/zoom.py in gsd's
selected space:
  B_hat_E1 = within-anchor OLS of phi(gen) on log g' (one row, pooled over anchors).
  zoom_direction = cos(B_hat_E1, B_zoom) / reliability(B_zoom); blur_cos = cos(B_hat_E1, B_blur) (diagnostic).
  zoom_slope = within-anchor OLS slope of the zoom reader's predicted d log footprint on d log g'
    (1 = the footprint scales with eff GSD at a fixed pixel count, i.e. a physical zoom; 0 = no zoom);
  blur_slope = the same with the blur reader. Both are divided by the reader's own held-out slope on REAL
  zoom (blur) pairs (ridge shrinkage), so 1 = as much zoom (blur) per unit d log g' as real chips show.
  magnitude = max(0, 1 - |log2 zoom_slope|).
  zoom_score = mean(zoom_direction s_hat, magnitude). Site bootstrap over anchors (reference held fixed).
  Gate: B_zoom split-half reliability (bootstrap p, BH).
"""
from __future__ import annotations

import numpy as np

from metacf import stats
from metacf.v2.core import norm as N
from metacf.v2.core.features import STATS_NAMES
from metacf.v2.measures.base import Measure, pick_by_category_region

IH, IL = STATS_NAMES.index("hf_fraction"), STATS_NAMES.index("acf_length")


def gsd_field(run):
    for n, fd in run.fields.items():
        if fd.tag == "gsd" and fd.enabled:
            return n
    return None


class E1Resolution(Measure):
    id = "E1"
    name = "resolution"

    def field(self):
        return gsd_field(self.run)

    def cells(self):
        def comp():
            r = self.run
            f = self.field()
            if f is None or not r.conditioned(f):
                return []
            fd = r.fields[f]
            sup = r.support(f)
            elig = [x for x in r.test if r.complete(x)]
            anchors = pick_by_category_region(elig, int(self.c("n_anchors", 50)), self.seed("anchors"))
            K = int(self.c("n_values", 5))
            out = []
            for a in anchors:
                rg = sup.by_cat.get(a.category, sup.global_range)
                vals = np.exp(np.linspace(np.log(max(rg[1], 1e-3)), np.log(rg[2]), K))
                sd = r.seed("E1seed", a.rid)
                out.append(dict(anchor=a.rid, site=a.location_id, g=vals,
                                reqs=[r.req(a, fd.set(a.fields, float(v)), seed=sd) for v in vals]))
            return out
        return self.run.cached(("E1", "cells"), comp)

    def plan(self):
        from metacf.v2.physics import e1_match
        return [q for c in self.cells() for q in c["reqs"]] + e1_match.plan(self.run)

    def real(self):
        r = self.run
        f = self.field()
        fd = r.fields[f]
        elig = [x for x in r.test if fd.raw(x.fields) is not None]
        n = int(self.c("n_real", 5000))
        if len(elig) > n:
            rng = np.random.default_rng(self.seed("real"))
            elig = [elig[i] for i in sorted(rng.choice(len(elig), n, replace=False))]
        F = r.real_feats("stats", [x.rid for x in elig], raw=True)
        g = np.array([float(fd.raw(x.fields)) for x in elig])
        rho = np.array([(x.chip_px or r.ds.resolution) / float(r.ds.resolution) for x in elig])
        cats = np.array([x.category for x in elig])
        return F[:, IH], F[:, IL], g, rho, cats

    @staticmethod
    def fe_slope(y, x, cats):
        ok = np.isfinite(y) & np.isfinite(x) & (y > 0)
        if ok.sum() < 10:
            return dict(coef=np.nan, p=np.nan, n=int(ok.sum()))
        X = np.column_stack([np.log(x[ok]), stats.fixed_effects(cats[ok])])
        o = stats.ols(X, np.log(y[ok]), intercept=True)
        return dict(coef=float(o["coef"][0]), p=float(o["p"][0]), se=float(o["se"][0]), n=int(o["n"]))

    def compute(self):
        r = self.run
        f = self.field()
        if f is None or not r.conditioned(f):
            return [N.invalid(self.id, f or "gsd", "score", "no conditioned gsd field", status="N/A")]
        h, l, g, rho, cats = self.real()
        real = {}
        for stratum, m in (("pooled", np.ones(len(g), bool)), ("rho>=1", rho >= 1), ("rho<1", rho < 1)):
            real[stratum] = dict(b_h=self.fe_slope(h[m], g[m], cats[m]), b_l=self.fe_slope(l[m], g[m], cats[m]))
        bl_real, bh_real = real["pooled"]["b_l"]["coef"], real["pooled"]["b_h"]["coef"]
        cells = self.cells()
        bh, bl = [], []
        for c in cells:
            F = r.gen_feats("stats", c["reqs"], raw=True)
            lg = np.log(c["g"])
            bh.append(stats.slope(lg, np.log(np.maximum(F[:, IH], 1e-12))))
            bl.append(stats.slope(lg, np.log(np.maximum(F[:, IL], 1e-6))))
        bh, bl = np.array(bh), np.array(bl)
        sites = [c["site"] for c in cells]
        _, sidx = np.unique(sites, return_inverse=True)

        def sc(w=None):
            if w is None:
                gl, gh = np.nanmedian(bl), np.nanmedian(bh)
            else:
                rep = np.repeat(np.arange(len(bl)), N.site_weights_to_items(w, sidx).astype(int))
                gl, gh = np.nanmedian(bl[rep]), np.nanmedian(bh[rep])
            s = max(0.0, 1 - abs(gl - bl_real) / abs(bl_real)) if np.isfinite(bl_real) and bl_real != 0 else np.nan
            return dict(score=s, b_l=gl, b_h=gh)
        pt = sc()
        bs = N.site_bootstrap(sc, sites, self.run.B, self.seed("boot"))
        gl, gh = pt["b_l"], pt["b_h"]
        if abs(gl) < 0.2 * abs(bl_real) and np.sign(gh) == np.sign(bh_real):
            label = "blur only"
        elif np.sign(gl) == np.sign(bl_real):
            label = "zoom"
        else:
            label = "wrong sign"
        out = [N.row(self.id, f, "score", raw=pt["score"], s_hat=pt["score"], ci=N.percentile_ci(bs["score"]),
                     n=len(cells), seeds=1, p=real["pooled"]["b_l"]["p"], space="stats(h, l)", label=label,
                     b_l_gen=gl, b_h_gen=gh, b_l_real=bl_real, b_h_real=bh_real, real=real,
                     b_l_gen_ci=N.percentile_ci(bs["b_l"]), b_h_gen_ci=N.percentile_ci(bs["b_h"]),
                     gate="real pooled b_l significant")]
        from metacf.config import get
        if bool(get(r.cfg, "v2_1.e1_zoom", False)):
            out += self.zoom_rows(f, cells)
        from metacf.v2.physics import e1_match
        if e1_match.enabled(r):                     # v2_2: GSD matching rows (own gate: reader within-scene check)
            out += e1_match.rows(r, self.id)
        return out

    def zoom_rows(self, f, cells):
        """v2_1: zoom direction / slope of the E1 generations against the synthetic zoom reference."""
        from metacf.v2.core import zoom
        from metacf.v2.core.regression import SiteStats, cos
        from metacf.v2.measures.c1_counterfactual import mag_score
        r = self.run
        sp = r.space_of(f)
        Z, Bb = zoom.reference(r, sp, "zoom"), zoom.reference(r, sp, "blur")
        rz, rb = zoom.scale_reader(r, sp, "zoom"), zoom.scale_reader(r, sp, "blur")
        sites = [c["site"] for c in cells]
        us, sidx = np.unique(sites, return_inverse=True)
        Xc, Fc, pz, pb = [], [], [], []
        for c in cells:
            F = r.gen_feats(sp, c["reqs"])
            lg = np.log(c["g"])
            Fd, xd = F - F.mean(0), lg - lg.mean()
            Xc.append(xd)
            Fc.append(Fd)
            # predicted d log footprint of each image vs the anchor's mean image (antisymmetric reader)
            qz, qb = rz["head"].predict(Fd), rb["head"].predict(Fd)
            pz.append(qz - qz.mean())
            pb.append(qb - qb.mean())
        cell_site = np.repeat(sidx, [len(x) for x in Xc])
        Xc, Fc, pz, pb = np.concatenate(Xc), np.concatenate(Fc), np.concatenate(pz), np.concatenate(pb)
        ss = SiteStats(Xc[:, None], Fc, cell_site, len(us))
        relZ = Z["rel"]
        # calibrate the readers by their own held-out slope on REAL zoom (resp. blur) pairs, so 1 = the
        # footprint (resp. blur) change of real chips per unit d log g' (ridge shrinkage otherwise biases low)
        cz = rz["validity"].get("slope_on_zoom", np.nan)
        cb = rb["validity"].get("slope_on_blur", np.nan)
        cz = cz if np.isfinite(cz) and cz > 0.05 else np.nan
        cb = cb if np.isfinite(cb) and cb > 0.05 else np.nan

        def stat(w=None):
            ww = np.ones(len(us)) if w is None else np.asarray(w, float)
            BG = ss.fit(ww)
            wi = ww[cell_site]
            sz = float(np.sum(wi * Xc * pz) / np.sum(wi * Xc * Xc)) / cz
            sb = float(np.sum(wi * Xc * pb) / np.sum(wi * Xc * Xc)) / cb
            d = cos(BG, Z["B"])
            dh = d / relZ["mean"] if relZ["mean"] else np.nan
            kap = float(np.sum(BG * Z["B"]) / np.sum(Z["B"] ** 2))
            return dict(direction=d, dir_hat=dh, blur_cos=cos(BG, Bb["B"]), zoom_slope=sz, blur_slope=sb,
                        kappa=kap, score=float(np.mean([dh, mag_score(sz)])))
        pt = stat()
        bs = N.site_bootstrap(stat, sites, self.run.B, self.seed("zoomboot"))
        ci = {k: N.percentile_ci(v) for k, v in bs.items()}
        p = relZ.get("p", np.nan)
        com = dict(n=len(cells), seeds=1, space=sp, p=p, reference="zoom",
                   gate="split-half reliability of B_zoom (synthetic zoom pairs)",
                   zoom_ref=dict(n_pairs=Z["n"], rel=relZ["mean"], rel_blur=Bb["rel"]["mean"],
                                 cos_zoom_blur=cos(Z["B"], Bb["B"]), reader_zoom=rz["validity"],
                                 reader_blur=rb["validity"]))
        lab = ("zoom" if pt["zoom_slope"] > 0.5 and pt["zoom_slope"] > pt["blur_slope"] else
               "blur" if pt["blur_slope"] > 0.5 and pt["blur_slope"] > pt["zoom_slope"] else
               "wrong sign" if pt["zoom_slope"] < 0 else "weak")
        return [N.row(self.id, f, "zoom_direction", raw=pt["direction"], floor=0.0, ceil=relZ["mean"],
                      ci=ci["dir_hat"], blur_cos=pt["blur_cos"], kappa=pt["kappa"], **com),
                N.row(self.id, f, "zoom_slope", raw=pt["zoom_slope"], s_hat=mag_score(pt["zoom_slope"]),
                      ci=ci["zoom_slope"], ci_on="zoom_slope", **com),
                N.row(self.id, f, "blur_slope", raw=pt["blur_slope"], s_hat=float("nan"),
                      ci=ci["blur_slope"], **com),
                N.row(self.id, f, "zoom_score", raw=pt["score"], s_hat=pt["score"], ci=ci["score"], label=lab,
                      zoom_slope=pt["zoom_slope"], blur_slope=pt["blur_slope"], blur_cos=pt["blur_cos"],
                      direction_s_hat=pt["dir_hat"], **com)]

    def finalize(self, rows):
        for x in rows:
            if x["metric"].startswith("match_"):        # v2_2 rows carry their own gate (gsd_reader.choose)
                continue
            if x["measure"] == self.id and x["status"] == "OK":
                if not (np.isfinite(x["p_bh"]) and x["p_bh"] <= self.run.q):
                    x["extra"]["s_hat_ungated"] = x["s_hat"]
                    x["status"], x["s_hat"] = "INVALID", float("nan")
                    x["reason"] = ("gate: B_zoom reliability not significant after BH" if x["metric"].startswith(("zoom", "blur"))
                                   else "gate: real pooled b_l not significant after BH")
        return rows
