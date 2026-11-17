"""
T14 SEASON: HEMISPHERE INVERSION (tag doy).
Anchors: 20 northern + 20 southern TEST records with |lat| > 20 deg and vegetated land cover.
Sweep doy over 12 values (the 15th of each month), 3 seeds, CONSISTENT mode (solar fields follow).
Descriptor: chromatic ExG = mean(2g - r - b) with (r, g, b) = channel / channel sum.
Per anchor x seed: fit ExG(doy) = m + a cos(2 pi (doy - phi) / 365.25); keep fits with a above the
seed-to-seed sd of ExG at fixed doy.
d_phi_gen = circular mean of southern phi - circular mean of northern phi, wrapped to [0, 365.25).
Real: the same fit on natural sequences (>= 4 dates spanning >= 6 months) -> d_phi_real; with fewer
than 10 southern sequences, fall back to cross-sectional real TEST images binned by month within each
hemisphere (validation marked FALLBACK).
score = max(0, 1 - |d_phi_gen - d_phi_real| / 182.6); pass flag |d_phi_gen - 182.6| <= 45 days.
Validation: |d_phi_real - 182.6| <= 45 days, else INVALID. Min-n: 10 southern anchors with valid fits.
"""
from __future__ import annotations

import numpy as np

from metacf.base import MetaCFTest, na, row
from metacf.tier2 import seasonal as SE

HALF = 182.6


def circ_abs_diff(a, b, period=SE.YEAR):
    d = (a - b) % period
    return min(d, period - d)


class T14Hemisphere(MetaCFTest):
    id = "T14"
    name = "season: hemisphere inversion"
    tier = 2
    required_tags = {"doy"}

    def _lat(self, r):
        v = r.fields.get(self.tc.get("lat_key", "lat_deg"))
        return np.nan if v is None else float(v)

    def vegetated(self, r):
        lc = r.fields.get(self.tc.get("landcover_key", "cov.landcover_mode"))
        return lc is not None and int(lc) in set(self.tc.get("vegetated", [1, 4, 6]))

    def keep(self, hemi):
        m = float(self.tc.get("min_abs_lat", 20))
        if hemi == "N":
            return lambda r: self._lat(r) > m and self.vegetated(r)
        return lambda r: self._lat(r) < -m and self.vegetated(r)

    def sweeps(self, ctx, f):
        n = int(self.tc.get("n_anchors_per_hemisphere", 20))
        return {h: SE.monthly_sweep(ctx, f, ctx.anchors(f, SE.MONTH_DOY, n=n, tag=f"T14{h}",
                                                         extra_filter=self.keep(h)), ctx.anchor_seeds())
                for h in ("N", "S")}

    def plan(self, ctx):
        return [r for f in self.fields(ctx) for sw in self.sweeps(ctx, f).values() for r in sw.requests()]

    def real_delta(self, ctx, f):
        def comp():
            fits = {h: SE.real_sequence_fits(ctx, f, self.keep(h)) for h in ("N", "S")}
            if len(fits["S"]) >= int(self.tc.get("min_south_sequences", 10)) and fits["N"]:
                d = SE.delta_phase([p for _, p, _ in fits["S"]], [p for _, p, _ in fits["N"]])
                return dict(delta=float(d), source="natural sequences", n_south=len(fits["S"]),
                            n_north=len(fits["N"]), status_if_ok="VALID")
            ph = {h: SE.real_crosssection_fit(ctx, f, self.keep(h)) for h in ("N", "S")}
            d = (ph["S"][0] - ph["N"][0]) % SE.YEAR
            return dict(delta=float(d), source="cross-sectional monthly bins (TEST)",
                        n_south=ph["S"][2], n_north=ph["N"][2], status_if_ok="FALLBACK")
        return ctx.cached(("T14real", f), comp)

    def validate_real(self, ctx):
        out = {}
        for f in self.fields(ctx):
            r = self.real_delta(ctx, f)
            ok = np.isfinite(r["delta"]) and circ_abs_diff(r["delta"], HALF) <= float(self.tc.get("tol_days", 45))
            out[f] = dict(status=r["status_if_ok"] if ok else "INVALID", **r)
        return out

    def compute(self, ctx, store):
        out = self.missing_tag_na(ctx)
        val = self.validate_real(ctx)
        tol = float(self.tc.get("tol_days", 45))
        for f in self.fields(ctx):
            if val[f]["status"] == "INVALID":
                out.append(na(self.id, f, f"real-data validation: {val[f]}", status="INVALID"))
                continue
            sws = self.sweeps(ctx, f)
            fits = {h: SE.fit_sweep(sws[h], store) for h in ("N", "S")}
            kept = {h: [x for x in fits[h] if x["kept"]] for h in ("N", "S")}
            south_anchors = len({x["anchor"] for x in kept["S"]})
            if south_anchors < int(self.tc.get("min_n", 10)) or not kept["N"]:
                out.append(na(self.id, f, f"{south_anchors} southern anchors with valid fits"))
                continue
            dg = float(SE.delta_phase([x["phase"] for x in kept["S"]], [x["phase"] for x in kept["N"]]))
            dr = val[f]["delta"]
            score = max(0.0, 1 - circ_abs_diff(dg, dr) / HALF)
            out.append(row(self.id, f, "physical", raw=score, floor=0.0, ceiling=1.0, n=south_anchors,
                           seeds=len(ctx.anchor_seeds()), validation=val[f]["status"],
                           delta_phi_gen=dg, delta_phi_real=dr, real=val[f],
                           passed=bool(circ_abs_diff(dg, HALF) <= tol),
                           n_kept_fits={h: len(kept[h]) for h in kept}))
        return out
