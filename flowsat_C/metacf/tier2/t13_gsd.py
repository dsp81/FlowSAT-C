"""
T13 GSD: SCALE VS SHARPNESS (tag gsd). Sweep effective GSD over K = 5 log-spaced values across the
in-support range, 20 anchors x 3 seeds, RAW.
Per image: sharpness h = fraction of FFT power (Hann window, mean removed, DC excluded) above 0.25
cycles/px; scale l = first 1/e crossing of the radially averaged normalised autocorrelation (px).
Generated slopes per anchor x seed: b_h = dlog h / dlog g', b_l = dlog l / dlog g'; medians.
Real slopes: cross-sectional OLS on real TEST of log h and log l on log g' with category fixed
effects, separately for rho = chip_px / resolution >= 1 (downsampled chips) and < 1, and pooled.
(rho = cN/R in the spec: c = chip short side in px, N/R = 1 / output resolution -- see
docs/OPEN_QUESTIONS.md.)
Classification (descriptive): "zoom" if b_l_gen is inside the CI of b_l_real (pooled); "blur only"
if |b_l_gen| < 0.2 |b_l_real| while b_h_gen has the real sign. score = max(0, 1 - |b_l_gen -
b_l_real| / |b_l_real|). Validation: pooled real b_l differs from 0 at p < 0.05.
"""
from __future__ import annotations

import numpy as np

from metacf import fields as F
from metacf import physical, stats
from metacf.base import MetaCFTest, na, row
from metacf.tier1.common import Sweep


def measures(img):
    h = physical.sharpness_h(img)
    ell = physical.autocorr_scale(img)
    return (np.log(h) if h > 0 else np.nan), (np.log(ell) if ell and ell > 0 else np.nan)


class T13GSD(MetaCFTest):
    id = "T13"
    name = "GSD: scale vs sharpness"
    tier = 2
    required_tags = {"gsd"}

    def sweep(self, ctx, f):
        sup = ctx.support(f)
        lo, hi = sup.global_range[1], sup.global_range[2]
        grid = F.log_spaced_grid(max(lo, 1e-6), hi, int(self.tc.get("K", 5)))
        anchors = ctx.anchors(f, grid, n=int(self.tc.get("n_anchors", 20)), tag="T13")
        return Sweep(ctx, f, grid, anchors, ctx.anchor_seeds(), "RAW")

    def plan(self, ctx):
        return [r for f in self.fields(ctx) for r in self.sweep(ctx, f).requests()]

    def real_fit(self, ctx, f):
        def comp():
            spec = ctx.specs[f]
            recs = [r for r in ctx.test if spec.raw(r.fields)][: int(self.tc.get("n_real", 3000))]
            M = np.array([measures(ctx.data.real_image(r.rid)) for r in recs])
            lg = np.log([float(spec.raw(r.fields)) for r in recs])
            cats = [r.category for r in recs]
            rho = np.array([r.chip_px / ctx.data.resolution if r.chip_px else 1.0 for r in recs])
            res = {}
            for name, m in (("pooled", np.ones(len(recs), bool)), ("rho>=1", rho >= 1), ("rho<1", rho < 1)):
                if m.sum() < 30:
                    continue
                X = np.column_stack([lg[m], stats.fixed_effects(np.array(cats)[m])])
                oh, ol = stats.ols(X, M[m, 0]), stats.ols(X, M[m, 1])
                res[name] = dict(b_h=float(oh["coef"][0]), b_l=float(ol["coef"][0]),
                                 b_l_se=float(ol["se"][0]), b_l_p=float(ol["p"][0]), n=int(m.sum()))
            return res
        return ctx.cached(("T13real", f), comp)

    def validate_real(self, ctx):
        out = {}
        for f in self.fields(ctx):
            r = self.real_fit(ctx, f).get("pooled")
            out[f] = dict(status="N/A", detail="too few real images") if r is None else \
                dict(status="VALID" if r["b_l_p"] < 0.05 else "INVALID", **r)
        return out

    def compute(self, ctx, store):
        out = self.missing_tag_na(ctx)
        val = self.validate_real(ctx)
        for f in self.fields(ctx):
            if val[f]["status"] != "VALID":
                out.append(na(self.id, f, f"real-data validation: {val[f]}", status=val[f]["status"]))
                continue
            sw = self.sweep(ctx, f)
            if len(sw.anchors) < int(self.tc.get("min_n", 5)):
                out.append(na(self.id, f, f"{len(sw.anchors)} anchors in support"))
                continue
            imgs, _ = sw.images(store)
            lg = np.log(sw.grid)
            bh, bl = [], []
            for a in range(len(sw.anchors)):
                for s in range(len(sw.seeds)):
                    M = np.array([measures(imgs[a][k][s]) for k in range(len(sw.grid))])
                    bh.append(stats.slope(lg, M[:, 0]))
                    bl.append(stats.slope(lg, M[:, 1]))
            bh_g, bl_g = stats.median_ignore_nan(bh), stats.median_ignore_nan(bl)
            real = self.real_fit(ctx, f)
            rp = real["pooled"]
            lo, hi = rp["b_l"] - 1.96 * rp["b_l_se"], rp["b_l"] + 1.96 * rp["b_l_se"]
            cls = "zoom" if lo <= bl_g <= hi else (
                "blur only" if abs(bl_g) < 0.2 * abs(rp["b_l"]) and np.sign(bh_g) == np.sign(rp["b_h"]) else "other")
            score = max(0.0, 1 - abs(bl_g - rp["b_l"]) / abs(rp["b_l"]))
            blarr = np.array(bl)
            ci = stats.bootstrap(lambda i: max(0.0, 1 - abs(stats.median_ignore_nan(blarr[i]) - rp["b_l"]) / abs(rp["b_l"])),
                                 len(blarr), self.B, self.boot_seed(ctx, f))
            out.append(row(self.id, f, "physical", raw=score, floor=0.0, ceiling=1.0, ci=ci, n=len(bl),
                           seeds=len(sw.seeds), b_h_gen=bh_g, b_l_gen=bl_g, real=real, classification=cls))
        return out
