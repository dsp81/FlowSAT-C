"""
E2 SEASON (spec section 3).

  Phase test anchors: TEST records with |lat| > min_abs_lat (20) and vegetated land cover: n_phase (20)
  northern + n_phase southern. Biome test anchors: n_biome (40) more, stratified HIGH vs LOW (n_biome / 2
  each; |lat| > 20). 12 dates (the 15th of each month), 1 seed, sun recomputed (CONSISTENT mode).
  Descriptor: chromatic ExG mean. Per anchor least squares ExG = m + a cos(2 pi (d - phi) / 365.25).
  Phase: delta_phi = circmean(phi_south) - circmean(phi_north) (days, wrapped to [0, 365.25));
    score = max(0, 1 - |wrap(delta_gen - delta_real)| / 182.6).
  Biome: R = median a_LOW / median a_HIGH; score = max(0, 1 - |log2(R_gen / R_real)|).
  Real references: TEST repeat sequences (same location_id, >= 4 dates spanning >= 6 months), ExG of the
    real tiles, the same fit. If fewer than min_south_sequences (10) southern sequences: cross-sectional
    monthly bins (median ExG per calendar month over the stratum's TEST tiles), marked FALLBACK; likewise
    for a biome stratum with fewer than min_biome_sequences (10) sequences.
  Gates: |delta_real - 182.6| <= 45 days (phase); R_real < 0.5 (biome). Not p-value gates (no BH).

HIGH / LOW strata (Q17): fMoW land cover has no deciduous / evergreen split, so (as in v1 T15) forest
counts as HIGH outside tropical Koppen A classes and as LOW inside Koppen Af. MODIFIED.
"""
from __future__ import annotations

import math

import numpy as np

from metacf import physical, solar
from metacf.tier2.seasonal import MONTH_DOY, circ_mean_days, circ_span_days
from metacf.v2.core import norm as N
from metacf.v2.core.features import STATS_NAMES
from metacf.v2.measures.base import Measure, pick_by_category_region

IE = STATS_NAMES.index("exg")
YEAR = 365.25


def wrap_days(d):
    return (d + YEAR / 2) % YEAR - YEAR / 2


def delta(south, north):
    return (circ_mean_days(south) - circ_mean_days(north)) % YEAR


class E2Season(Measure):
    id = "E2"
    name = "season"
    modified = True

    def doy_field(self):
        for n, fd in self.run.fields.items():
            if fd.tag == "doy" and fd.enabled:
                return n
        return None

    def lat(self, x):
        v = x.fields.get(self.c("lat_key", "lat_deg"))
        return float(v) if v is not None else np.nan

    def lc(self, x):
        v = x.fields.get(self.c("landcover_key", "cov.landcover_mode"))
        return int(v) if v is not None else None

    def kop(self, x):
        v = x.fields.get(self.c("koppen_key", "cov.koppen"))
        return int(v) if v is not None else None

    def vegetated(self, x):
        return self.lc(x) in set(self.c("vegetated", [1, 4, 6]))

    def stratum(self, x):
        """HIGH | LOW | None."""
        lc, kp = self.lc(x), self.kop(x)
        fc = self.c("forest_code", None)
        if lc in set(self.c("high_landcover", [4, 6])):
            return "HIGH"
        if lc in set(self.c("low_landcover", [9, 7])):
            return "LOW"
        if fc is not None and lc == fc:
            if kp is not None and kp in set(self.c("rainforest_koppen", [0])):
                return "LOW"
            if kp is None or kp not in set(self.c("tropical_koppen", [0, 1, 2])):
                return "HIGH"
        return None

    def far(self, x):
        return abs(self.lat(x)) > float(self.c("min_abs_lat", 20))

    def anchors(self):
        def comp():
            r = self.run
            elig = [x for x in r.test if r.complete(x) and np.isfinite(self.lat(x)) and self.far(x)]
            n = int(self.c("n_phase", 20))
            north = pick_by_category_region([x for x in elig if self.vegetated(x) and self.lat(x) > 0], n,
                                            self.seed("north"))
            south = pick_by_category_region([x for x in elig if self.vegetated(x) and self.lat(x) < 0], n,
                                            self.seed("south"))
            used = {x.rid for x in north + south}
            nb = int(self.c("n_biome", 40)) // 2
            high = pick_by_category_region([x for x in elig if self.stratum(x) == "HIGH" and x.rid not in used], nb,
                                           self.seed("high"))
            low = pick_by_category_region([x for x in elig if self.stratum(x) == "LOW" and x.rid not in used], nb,
                                          self.seed("low"))
            return dict(north=north, south=south, HIGH=high, LOW=low)
        return self.run.cached(("E2", "anchors"), comp)

    def cells(self):
        def comp():
            r = self.run
            f = self.doy_field()
            if f is None or not r.conditioned(f):
                return {}
            fd = r.fields[f]
            out = {}
            for grp, lst in self.anchors().items():
                cells = []
                for a in lst:
                    sd = r.seed("E2seed", a.rid)
                    reqs = []
                    for d in MONTH_DOY:
                        new = solar.consistent(a.fields, fd.set(a.fields, float(d)))
                        reqs.append(r.req(a, new, seed=sd, mode="CONSISTENT"))
                    cells.append(dict(anchor=a.rid, site=a.location_id, reqs=reqs))
                out[grp] = cells
            return out
        return self.run.cached(("E2", "cells"), comp)

    def plan(self):
        return [q for cells in self.cells().values() for c in cells for q in c["reqs"]]

    # ---------------------------------------------------------------- real references
    def real_fits(self, keep):
        """TEST sequences passing keep(record): list of (phase, amplitude)."""
        r = self.run
        f = self.doy_field()
        fd = r.fields[f]
        by = {}
        for x in r.test:
            if keep(x) and fd.raw(x.fields) is not None:
                by.setdefault(x.location_id, []).append(x)
        seqs = [v for v in by.values() if len(v) >= int(self.c("min_dates", 4)) and
                circ_span_days([float(fd.raw(x.fields)) for x in v]) >= float(self.c("min_span_days", 182.6))]
        out = []
        for v in seqs:
            E = r.real_feats("stats", [x.rid for x in v], raw=True)[:, IE]
            m, a, phi, r2 = physical.fit_seasonal([float(fd.raw(x.fields)) for x in v], E)
            if np.isfinite(phi):
                out.append((phi, a))
        return out

    def cross_section(self, keep):
        """Fallback: median ExG per calendar month over the stratum's TEST tiles, then the seasonal fit."""
        r = self.run
        fd = r.fields[self.doy_field()]
        recs = [x for x in r.test if keep(x) and fd.raw(x.fields) is not None]
        n = int(self.c("n_cross_max", 3000))
        if len(recs) > n:
            rng = np.random.default_rng(self.seed("xs"))
            recs = [recs[i] for i in sorted(rng.choice(len(recs), n, replace=False))]
        if len(recs) < 12:
            return (np.nan, np.nan)
        E = r.real_feats("stats", [x.rid for x in recs], raw=True)[:, IE]
        d = np.array([float(fd.raw(x.fields)) for x in recs])
        month = np.minimum((d % YEAR) // (YEAR / 12), 11).astype(int)
        md = [MONTH_DOY[k] for k in range(12) if (month == k).any()]
        mv = [float(np.median(E[month == k])) for k in range(12) if (month == k).any()]
        _, a, phi, _ = physical.fit_seasonal(md, mv)
        return (phi, a)

    def compute(self):
        r = self.run
        f = self.doy_field()
        if f is None or not r.conditioned(f):
            return [N.invalid(self.id, f or "doy", "phase", "no conditioned doy field", status="N/A")]
        out = []
        # ---- real references
        nkeep = lambda x: self.vegetated(x) and self.far(x) and self.lat(x) > 0
        skeep = lambda x: self.vegetated(x) and self.far(x) and self.lat(x) < 0
        rn, rs = self.real_fits(nkeep), self.real_fits(skeep)
        fb_phase = len(rs) < int(self.c("min_south_sequences", 10))
        if fb_phase:
            phn, phs = self.cross_section(nkeep)[0], self.cross_section(skeep)[0]
            d_real = (phs - phn) % YEAR if np.isfinite(phn) and np.isfinite(phs) else np.nan
        else:
            d_real = delta([p for p, _ in rs], [p for p, _ in rn])
        hkeep = lambda x: self.far(x) and self.stratum(x) == "HIGH"
        lkeep = lambda x: self.far(x) and self.stratum(x) == "LOW"
        rh, rl = self.real_fits(hkeep), self.real_fits(lkeep)
        fb_biome = min(len(rh), len(rl)) < int(self.c("min_biome_sequences", 10))
        if fb_biome:
            ah, al = self.cross_section(hkeep)[1], self.cross_section(lkeep)[1]
        else:
            ah, al = float(np.median([a for _, a in rh])), float(np.median([a for _, a in rl]))
        R_real = al / ah if np.isfinite(ah) and ah > 0 else np.nan
        # ---- generated fits
        fits = {}
        for grp, cells in self.cells().items():
            fits[grp] = []
            for c in cells:
                E = r.gen_feats("stats", c["reqs"], raw=True)[:, IE]
                _, a, phi, r2 = physical.fit_seasonal(MONTH_DOY, E)
                fits[grp].append(dict(site=c["site"], phase=phi, amp=a, r2=r2))
        # phase
        gate_phase = bool(np.isfinite(d_real) and abs(d_real - YEAR / 2) <= float(self.c("tol_days", 45)))
        N_, S_ = fits.get("north", []), fits.get("south", [])

        def ph(wn=None, ws=None):
            pn = [x["phase"] for x in N_]
            ps_ = [x["phase"] for x in S_]
            if wn is not None:
                pn = np.repeat(pn, wn)
                ps_ = np.repeat(ps_, ws)
            if len(pn) == 0 or len(ps_) == 0:
                return np.nan, np.nan
            dg = delta(ps_, pn)
            return dg, max(0.0, 1 - abs(wrap_days(dg - d_real)) / (YEAR / 2))
        d_gen, s_phase = ph()
        rng = np.random.default_rng(self.seed("boot_phase"))
        reps = [ph(rng.multinomial(len(N_), [1 / len(N_)] * len(N_)) if N_ else None,
                   rng.multinomial(len(S_), [1 / len(S_)] * len(S_)) if S_ else None)[1]
                for _ in range(self.run.B)] if N_ and S_ else []
        row = N.row(self.id, f, "phase", raw=s_phase, s_hat=s_phase, ci=N.percentile_ci(reps), n=len(N_) + len(S_),
                    seeds=1, space="stats(exg)", delta_gen_days=d_gen, delta_real_days=d_real,
                    real_source="FALLBACK cross-sectional" if fb_phase else "sequences",
                    n_real_north=len(rn), n_real_south=len(rs), gate="|delta_real - 182.6| <= 45 d",
                    modified=False)
        if not gate_phase:
            row.update(status="INVALID", reason=f"gate: delta_real {d_real:.1f} d not within 45 d of 182.6",
                       s_hat=float("nan"))
        if fb_phase:
            row["extra"]["FALLBACK"] = True
        out.append(row)
        # biome
        H, L = fits.get("HIGH", []), fits.get("LOW", [])

        def bi(wh=None, wl=None):
            ah_ = np.array([x["amp"] for x in H])
            al_ = np.array([x["amp"] for x in L])
            if wh is not None:
                ah_, al_ = np.repeat(ah_, wh), np.repeat(al_, wl)
            if len(ah_) == 0 or len(al_) == 0 or np.median(ah_) <= 0:
                return np.nan, np.nan
            Rg = float(np.median(al_) / np.median(ah_))
            if not (np.isfinite(R_real) and R_real > 0 and Rg > 0):
                return Rg, 0.0 if np.isfinite(Rg) else np.nan
            return Rg, max(0.0, 1 - abs(math.log2(Rg / R_real)))
        R_gen, s_biome = bi()
        reps = [bi(rng.multinomial(len(H), [1 / len(H)] * len(H)), rng.multinomial(len(L), [1 / len(L)] * len(L)))[1]
                for _ in range(self.run.B)] if H and L else []
        gate_b = bool(np.isfinite(R_real) and R_real < float(self.c("max_R_real", 0.5)))
        row = N.row(self.id, f, "biome", raw=s_biome, s_hat=s_biome, ci=N.percentile_ci(reps), n=len(H) + len(L),
                    seeds=1, space="stats(exg)", R_gen=R_gen, R_real=R_real,
                    real_source="FALLBACK cross-sectional" if fb_biome else "sequences",
                    n_real_high=len(rh), n_real_low=len(rl), gate="R_real < 0.5", modified=True,
                    strata="HIGH/LOW from land cover (+ Koppen for forest), Q17")
        if not gate_b:
            row.update(status="INVALID", reason=f"gate: R_real {R_real:.3f} not < 0.5", s_hat=float("nan"))
        if fb_biome:
            row["extra"]["FALLBACK"] = True
        out.append(row)
        return out
