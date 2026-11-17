"""
T16 SUN PHYSICS (tags sun_elev, sun_azim).
T16a elevation: sweep elevation over {15, 25, 40, 55, 70} deg (clipped to support), azimuth fixed at
    the anchor's own, RAW, 20 anchors x 3 seeds. Per anchor x seed fit L = alpha + b sin(e)
    (L = mean linear luminance): median R^2 and relative slope b / alpha; shadow-area fraction
    (T12's D mask) vs cot(e): Spearman. Real: cross-sectional OLS of linear luminance on sin(e) with
    category and sensor fixed effects -> (b / alpha)_real (alpha = mean L - b mean sin e); shadow
    fraction vs cot(e) Spearman on reals. score = max(0, 1 - |log2((b/alpha)_gen / (b/alpha)_real)|).
    Validation: the real slope is positive at p < 0.05.
T16b azimuth, IN SUPPORT (scored): K azimuths (tests.T16.n_azimuths, default 5) evenly spaced over
    the GLOBAL support arc of the azimuth field (support map, TRAIN [1st, 99th] percentiles around
    the circular mean), elevation fixed at the anchor's own, RAW; anchors support-filtered (their
    category supports every value). Per image the dominant shadow direction = circular mean of
    T12's tile directions; expected = azimuth + 180 deg (shadows point away from the sun) in image
    coordinates (north-up). Reports the Fisher-Lee circular correlation and the median angular
    error. Real validation: on real TEST images with >= 6 valid tiles the median error must be
    <= 60 deg, else INVALID (non-north-up chips or off-nadir parallax). score = corr_gen / corr_real.
T16b-extrapolation (DESCRIPTIVE, tests.T16.extrapolation, default on): the 8-direction sweep (every
    45 deg) with anchors NOT support-filtered, the same statistics, reported as its own row
    (stratum "T16b-extrapolation", extrapolation = true). fMoW azimuths cluster in ~135-180 deg, so
    most of these directions are outside support: the row says what the model does there, it is
    not a control score and never enters MetaCF-U.
"""
from __future__ import annotations

import math

import numpy as np

from metacf import physical, stats
from metacf.base import MetaCFTest, na, row
from metacf.tier1.common import Sweep

ELEV = [15.0, 25.0, 40.0, 55.0, 70.0]
AZIM = [float(a) for a in range(0, 360, 45)]


def ang_err(a, b):
    return np.abs(stats.wrap(np.asarray(a) - np.asarray(b)))


class T16Sun(MetaCFTest):
    id = "T16"
    name = "sun physics"
    tier = 2
    required_tags = {"sun_elev", "sun_azim"}

    def f_elev(self, ctx):
        return next((f for f in ctx.fields_with_tag("sun_elev")), None)

    def f_azim(self, ctx):
        return next((f for f in ctx.fields_with_tag("sun_azim")), None)

    def elev_sweep(self, ctx, f):
        grid = ctx.support(f).clip_grid(self.tc.get("elevations", ELEV))
        anchors = ctx.anchors(f, grid, n=int(self.tc.get("n_anchors", 20)), tag="T16a")
        return Sweep(ctx, f, grid, anchors, ctx.anchor_seeds(), "RAW")

    def support_azimuths(self, ctx, f):
        """K azimuths (deg) evenly spaced over the global support arc of the azimuth field."""
        r = ctx.support(f).global_range                      # ("circ", mu, lo, hi), radians
        K = int(self.tc.get("n_azimuths", 5))
        u = np.linspace(r[2], r[3], K) + r[1]
        return [float(x) for x in np.mod(np.degrees(u), 360.0)]

    def azim_sweep(self, ctx, f):
        """In-support sweep (the scored T16b)."""
        grid = self.support_azimuths(ctx, f)
        anchors = ctx.anchors(f, grid, n=int(self.tc.get("n_anchors", 20)), tag="T16b")
        return Sweep(ctx, f, grid, anchors, ctx.anchor_seeds(), "RAW")

    def extrap_sweep(self, ctx, f):
        """8 directions, outside support by design: anchors are not support-filtered."""
        grid = [float(a) for a in self.tc.get("azimuths", AZIM)]
        anchors = ctx.anchors(f, grid, n=int(self.tc.get("n_anchors", 20)), tag="T16b-extrapolation",
                              check_support=False)
        return Sweep(ctx, f, grid, anchors, ctx.anchor_seeds(), "RAW")

    def extrapolation_on(self):
        return bool(self.tc.get("extrapolation", True))

    def plan(self, ctx):
        reqs = []
        if self.f_elev(ctx):
            reqs += self.elev_sweep(ctx, self.f_elev(ctx)).requests()
        if self.f_azim(ctx):
            reqs += self.azim_sweep(ctx, self.f_azim(ctx)).requests()
            if self.extrapolation_on():
                reqs += self.extrap_sweep(ctx, self.f_azim(ctx)).requests()
        return reqs

    # ---------------------------------------------------------------- real data
    def real_elev(self, ctx, f):
        def comp():
            spec = ctx.specs[f]
            recs = [r for r in ctx.test if spec.raw(r.fields) is not None][: int(self.tc.get("n_real", 3000))]
            e = np.radians([float(spec.raw(r.fields)) for r in recs])
            L = np.array([physical.mean_linear_luminance(ctx.data.real_image(r.rid)) for r in recs])
            sf = np.array([physical.shadow_fraction(ctx.data.real_image(r.rid)) for r in recs])
            sensor = [r.fields.get(self.tc.get("sensor_key", "platform")) for r in recs]
            X = np.column_stack([np.sin(e), stats.fixed_effects([r.category for r in recs], sensor)])
            o = stats.ols(X, L)
            b = float(o["coef"][0])
            alpha = float(L.mean() - b * np.sin(e).mean())
            return dict(b=b, p=float(o["p"][0]), alpha=alpha, rel=b / alpha if alpha else float("nan"),
                        shadow_cot_spearman=stats.spearman(1 / np.tan(e), sf), n=len(recs))
        return ctx.cached(("T16a_real", f), comp)

    def real_azim(self, ctx, f):
        def comp():
            spec = ctx.specs[f]
            meas, exp = [], []
            for r in [r for r in ctx.test if spec.raw(r.fields) is not None][: int(self.tc.get("n_real", 3000))]:
                th = physical.shadow_directions(ctx.data.real_image(r.rid))
                if len(th) >= int(self.tc.get("min_tiles", 6)):
                    meas.append(stats.circ_mean(th))
                    exp.append(float(physical.expected_shadow_angle(float(spec.raw(r.fields)))))
            if not meas:
                return dict(n=0, median_err_deg=float("nan"), corr=float("nan"))
            return dict(n=len(meas), median_err_deg=float(np.degrees(np.median(ang_err(meas, exp)))),
                        corr=stats.fisher_lee(exp, meas))
        return ctx.cached(("T16b_real", f), comp)

    def validate_real(self, ctx):
        out = {}
        fe, fa = self.f_elev(ctx), self.f_azim(ctx)
        if fe:
            r = self.real_elev(ctx, fe)
            out[fe] = dict(status="VALID" if (r["b"] > 0 and r["p"] < 0.05) else "INVALID", **r)
        if fa:
            r = self.real_azim(ctx, fa)
            ok = r["n"] > 0 and r["median_err_deg"] <= float(self.tc.get("max_err_deg", 60))
            out[fa] = dict(status="VALID" if ok else "INVALID", **r)
        return out

    # ---------------------------------------------------------------- compute
    def compute(self, ctx, store):
        out = []
        val = self.validate_real(ctx)
        fe, fa = self.f_elev(ctx), self.f_azim(ctx)
        if not fe and not fa:
            return self.missing_tag_na(ctx)
        if fe:
            out.append(self._elev(ctx, store, fe, val[fe]))
        if fa:
            out.append(self._azim(ctx, store, fa, val[fa], self.azim_sweep(ctx, fa), "T16b"))
            if self.extrapolation_on():
                out.append(self._azim(ctx, store, fa, val[fa], self.extrap_sweep(ctx, fa),
                                      "T16b-extrapolation"))
        return out

    def _elev(self, ctx, store, f, v):
        if v["status"] != "VALID":
            return na(self.id, f, f"real-data validation: {v}", stratum="T16a", status="INVALID")
        sw = self.elev_sweep(ctx, f)
        if len(sw.anchors) < int(self.tc.get("min_n", 5)) or len(sw.grid) < 3:
            return na(self.id, f, f"{len(sw.anchors)} anchors / {len(sw.grid)} elevations in support",
                      stratum="T16a")
        imgs, _ = sw.images(store)
        s = np.sin(np.radians(sw.grid))
        cot = 1 / np.tan(np.radians(sw.grid))
        rel, r2, rho = [], [], []
        for a in range(len(sw.anchors)):
            for k in range(len(sw.seeds)):
                L = np.array([physical.mean_linear_luminance(imgs[a][j][k]) for j in range(len(sw.grid))])
                o = stats.ols(s, L)
                b = float(o["coef"][0])
                rel.append(b / o["intercept"] if o["intercept"] else np.nan)
                r2.append(o["r2"])
                rho.append(stats.spearman(cot, [physical.shadow_fraction(imgs[a][j][k]) for j in range(len(sw.grid))]))
        rg = stats.median_ignore_nan(rel)
        rr = v["rel"]
        score = max(0.0, 1 - abs(math.log2(rg / rr))) if rg > 0 and rr > 0 else 0.0
        relarr = np.array(rel)
        ci = stats.bootstrap(lambda i: (lambda x: max(0.0, 1 - abs(math.log2(x / rr))) if x > 0 else 0.0)(
            stats.median_ignore_nan(relarr[i])), len(relarr), self.B, self.boot_seed(ctx, f))
        return row(self.id, f, "physical", stratum="T16a", raw=score, floor=0.0, ceiling=1.0, ci=ci, n=len(rel),
                   seeds=len(sw.seeds), rel_slope_gen=rg, rel_slope_real=rr, median_r2=stats.median_ignore_nan(r2),
                   shadow_cot_spearman_gen=stats.median_ignore_nan(rho),
                   shadow_cot_spearman_real=v["shadow_cot_spearman"])

    def _azim(self, ctx, store, f, v, sw, stratum):
        extrap = stratum != "T16b"
        if v["status"] != "VALID":
            return na(self.id, f, f"real-data validation: {v}", stratum=stratum, status="INVALID")
        if len(sw.anchors) < int(self.tc.get("min_n", 5)) or len(sw.grid) < 3:
            return na(self.id, f, f"{len(sw.anchors)} anchors / {len(sw.grid)} azimuths"
                      + ("" if extrap else " in support"), stratum=stratum)
        imgs, _ = sw.images(store)
        meas, exp = [], []
        for a in range(len(sw.anchors)):
            for j, az in enumerate(sw.grid):
                for k in range(len(sw.seeds)):
                    th = physical.shadow_directions(imgs[a][j][k])
                    if len(th) >= int(self.tc.get("min_tiles_gen", 3)):
                        meas.append(stats.circ_mean(th))
                        exp.append(float(physical.expected_shadow_angle(az)))
        if len(meas) < 3:
            return na(self.id, f, f"{len(meas)} generated images with shadow tiles", stratum=stratum)
        cg = stats.fisher_lee(exp, meas)
        common = dict(n=len(meas), seeds=len(sw.seeds), azimuths_deg=list(sw.grid),
                      median_err_deg_gen=float(np.degrees(np.median(ang_err(meas, exp)))),
                      median_err_deg_real=v["median_err_deg"], corr_real=v["corr"],
                      extrapolation=extrap)
        if extrap:
            return row(self.id, f, "physical", stratum=stratum, raw=cg, descriptive=True,
                       notes="outside the azimuth support by design; descriptive, not in MetaCF-U",
                       **common)
        return row(self.id, f, "physical", stratum=stratum, raw=cg, floor=0.0, ceiling=v["corr"],
                   **common)
