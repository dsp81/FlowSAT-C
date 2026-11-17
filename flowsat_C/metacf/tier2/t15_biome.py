"""
T15 SEASON: BIOME NULL (tag doy). The T14 monthly sweep design on 40 more anchors stratified into
HIGH-seasonality land cover and LOW-seasonality land cover, |lat| > 20 deg.
fMoW's land-cover classes merge forest types, so (see docs/OPEN_QUESTIONS.md):
    HIGH = crop, grass, and forest outside tropical Koppen A zones
    LOW  = barren, urban, and forest inside tropical rainforest Koppen Af (evergreen broadleaf proxy)
Seasonal amplitude a per anchor x seed (T14 fit). R_gen = median a_LOW / median a_HIGH.
Real: the same ratio from natural sequences, or cross-sectional monthly bins per stratum (FALLBACK).
score = max(0, 1 - |log2(R_gen / R_real)|); pass flag R_gen < 0.5. Validation: R_real < 0.5.
"""
from __future__ import annotations

import math

import numpy as np

from metacf.base import MetaCFTest, na, row
from metacf.tier2 import seasonal as SE


class T15BiomeNull(MetaCFTest):
    id = "T15"
    name = "season: biome null"
    tier = 2
    required_tags = {"doy"}

    def stratum(self, r):
        tc = self.tc
        lat = r.fields.get(tc.get("lat_key", "lat_deg"))
        if lat is None or abs(float(lat)) <= float(tc.get("min_abs_lat", 20)):
            return None
        lc = r.fields.get(tc.get("landcover_key", "cov.landcover_mode"))
        kp = r.fields.get(tc.get("koppen_key", "cov.koppen"))
        if lc is None:
            return None
        lc = int(lc)
        tropical = kp is not None and int(kp) in set(tc.get("tropical_koppen", [0, 1, 2]))
        rainforest = kp is not None and int(kp) in set(tc.get("rainforest_koppen", [0]))
        forest = int(tc.get("forest_code", 1))
        if lc in set(tc.get("high_landcover", [4, 6])) or (lc == forest and not tropical):
            return "HIGH"
        if lc in set(tc.get("low_landcover", [9, 7])) or (lc == forest and rainforest):
            return "LOW"
        return None

    def keep(self, s):
        return lambda r: self.stratum(r) == s

    def sweeps(self, ctx, f):
        n = int(self.tc.get("n_anchors_per_stratum", 20))
        return {s: SE.monthly_sweep(ctx, f, ctx.anchors(f, SE.MONTH_DOY, n=n, tag=f"T15{s}",
                                                         extra_filter=self.keep(s)), ctx.anchor_seeds())
                for s in ("HIGH", "LOW")}

    def plan(self, ctx):
        return [r for f in self.fields(ctx) for sw in self.sweeps(ctx, f).values() for r in sw.requests()]

    def real_ratio(self, ctx, f):
        def comp():
            fits = {s: SE.real_sequence_fits(ctx, f, self.keep(s)) for s in ("HIGH", "LOW")}
            mn = int(self.tc.get("min_sequences", 10))
            if all(len(v) >= mn for v in fits.values()):
                R = np.median([a for _, _, a in fits["LOW"]]) / np.median([a for _, _, a in fits["HIGH"]])
                return dict(R=float(R), source="natural sequences", status_if_ok="VALID",
                            n={s: len(v) for s, v in fits.items()})
            amp = {s: SE.real_crosssection_fit(ctx, f, self.keep(s)) for s in ("HIGH", "LOW")}
            R = amp["LOW"][1] / amp["HIGH"][1] if amp["HIGH"][1] else float("nan")
            return dict(R=float(R), source="cross-sectional monthly bins (TEST)", status_if_ok="FALLBACK",
                        n={s: amp[s][2] for s in amp})
        return ctx.cached(("T15real", f), comp)

    def validate_real(self, ctx):
        out = {}
        for f in self.fields(ctx):
            r = self.real_ratio(ctx, f)
            ok = np.isfinite(r["R"]) and r["R"] < float(self.tc.get("max_ratio", 0.5))
            out[f] = dict(status=r["status_if_ok"] if ok else "INVALID", **r)
        return out

    def compute(self, ctx, store):
        out = self.missing_tag_na(ctx)
        val = self.validate_real(ctx)
        for f in self.fields(ctx):
            if val[f]["status"] == "INVALID":
                out.append(na(self.id, f, f"real-data validation: {val[f]}", status="INVALID"))
                continue
            sws = self.sweeps(ctx, f)
            if any(len(sw.anchors) < int(self.tc.get("min_n", 5)) for sw in sws.values()):
                out.append(na(self.id, f, f"anchors per stratum: { {s: len(sw.anchors) for s, sw in sws.items()} }"))
                continue
            amps = {s: [x["amp"] for x in SE.fit_sweep(sws[s], store) if np.isfinite(x["amp"])] for s in sws}
            Rg = float(np.median(amps["LOW"]) / np.median(amps["HIGH"]))
            Rr = val[f]["R"]
            score = max(0.0, 1 - abs(math.log2(Rg / Rr))) if Rg > 0 and Rr > 0 else 0.0
            out.append(row(self.id, f, "physical", raw=score, floor=0.0, ceiling=1.0,
                           n=sum(len(sw.anchors) for sw in sws.values()), seeds=len(ctx.anchor_seeds()),
                           validation=val[f]["status"], R_gen=Rg, R_real=Rr, real=val[f],
                           passed=bool(Rg < float(self.tc.get("max_ratio", 0.5)))))
        return out
