"""
E3 SUN (spec section 3).

  Generated: n_anchors (50) TEST anchors x n_values (5) sun elevations evenly spaced inside the anchor
  category's TRAIN [1st, 99th] support x 1 seed; azimuth and everything else fixed (RAW).
  L = mean linear luminance (0.2126 R + 0.7152 G + 0.0722 B of the linear channel means). Per anchor fit
  L = alpha + b sin e; relative slope b / alpha; generated value = the median over anchors.
  Real: OLS on n_real TEST real tiles of L on sin e with category and sensor fixed effects -> b;
    alpha_real = mean(L) - b mean(sin e) (Q18); (b / alpha)_real.
  Score = max(0, 1 - |log2((b/alpha)_gen / (b/alpha)_real)|) (0 when the ratio is <= 0).
  Shadow fraction (dark & low-saturation mask, metacf.physical.shadow_fraction) vs cot(e): Spearman on the
    generated images and on n_shadow_real real tiles (reported).
  Gate: real slope > 0 at p < 0.05 (one-sided OLS p, BH).
"""
from __future__ import annotations

import math

import numpy as np

from metacf import physical, stats
from metacf.v2.core import norm as N
from metacf.v2.core.features import STATS_NAMES
from metacf.v2.measures.base import Measure, pick_by_category_region

IR = STATS_NAMES.index("lin_mean_R")


def lum_from_stats(F):
    return 0.2126 * F[:, IR] + 0.7152 * F[:, IR + 1] + 0.0722 * F[:, IR + 2]


class E3Sun(Measure):
    id = "E3"
    name = "sun"

    def field(self):
        for n, fd in self.run.fields.items():
            if fd.tag == "sun_elev" and fd.enabled:
                return n
        return None

    def sensor_field(self):
        for n, fd in self.run.fields.items():
            if fd.tag == "sensor":
                return n
        return None

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
                vals = np.linspace(max(rg[1], 1.0), min(rg[2], 90.0), K)
                sd = r.seed("E3seed", a.rid)
                out.append(dict(anchor=a.rid, site=a.location_id, e=vals,
                                reqs=[r.req(a, fd.set(a.fields, float(v)), seed=sd) for v in vals]))
            return out
        return self.run.cached(("E3", "cells"), comp)

    def plan(self):
        return [q for c in self.cells() for q in c["reqs"]]

    def compute(self):
        r = self.run
        f = self.field()
        if f is None or not r.conditioned(f):
            out = [N.invalid(self.id, f or "sun_elev", "score", "no conditioned sun elevation field", status="N/A")]
            if f is not None and bool((r.cfg.get("v2_1") or {}).get("e3_shadows")):
                # v2.1: the shadow parts on own-metadata generations still apply (the model may infer the sun
                # from location + date); no sweep rows
                out += self.shadow_rows(f, [])
            return out
        fd = r.fields[f]
        sf = self.sensor_field()
        elig = [x for x in r.test if fd.raw(x.fields) is not None]
        n = int(self.c("n_real", 5000))
        rng = np.random.default_rng(self.seed("real"))
        if len(elig) > n:
            elig = [elig[i] for i in sorted(rng.choice(len(elig), n, replace=False))]
        L = lum_from_stats(r.real_feats("stats", [x.rid for x in elig], raw=True))
        se = np.sin(np.radians([float(fd.raw(x.fields)) for x in elig]))
        fe = [np.array([x.category for x in elig])]
        if sf:
            fe.append(np.array([str(r.fields[sf].raw(x.fields)) for x in elig]))
        o = stats.ols(np.column_stack([se, stats.fixed_effects(*fe)]), L, intercept=True)
        b = float(o["coef"][0])
        p1 = float(o["p"][0] / 2 if b > 0 else 1 - o["p"][0] / 2)
        alpha = float(np.mean(L) - b * np.mean(se))
        rel_real = b / alpha if alpha != 0 else np.nan
        cells = self.cells()
        rel = []
        sh_g, cot_g = [], []
        st = r.store()
        for c in cells:
            F = r.gen_feats("stats", c["reqs"], raw=True)
            Lg = lum_from_stats(F)
            s = np.sin(np.radians(c["e"]))
            bb = stats.slope(s, Lg)
            aa = float(np.mean(Lg) - bb * np.mean(s))
            rel.append(bb / aa if aa != 0 else np.nan)
            for q, e in zip(c["reqs"], c["e"]):
                sh_g.append(physical.shadow_fraction(st.get(q)))
                cot_g.append(1 / math.tan(math.radians(e)))
        rel = np.array(rel)
        sites = [c["site"] for c in cells]
        _, sidx = np.unique(sites, return_inverse=True)

        def sc(w=None):
            v = rel if w is None else np.repeat(rel, N.site_weights_to_items(w, sidx).astype(int))
            g = float(np.nanmedian(v)) if len(v) else np.nan
            if not (np.isfinite(g) and np.isfinite(rel_real)) or g / rel_real <= 0:
                return dict(score=0.0 if np.isfinite(g) else np.nan, rel=g)
            return dict(score=max(0.0, 1 - abs(math.log2(g / rel_real))), rel=g)
        pt = sc()
        bs = N.site_bootstrap(sc, sites, self.run.B, self.seed("boot"))
        ns = min(int(self.c("n_shadow_real", 1000)), len(elig))
        sel = rng.choice(len(elig), ns, replace=False) if ns else []
        sh_r = [physical.shadow_fraction(r.ds.real_image(elig[i].rid)) for i in sel]
        cot_r = [1 / math.tan(math.radians(max(float(fd.raw(elig[i].fields)), 1.0))) for i in sel]
        rows = [N.row(self.id, f, "score", raw=pt["score"], s_hat=pt["score"], ci=N.percentile_ci(bs["score"]),
                      n=len(cells), seeds=1, p=p1, space="linear luminance", rel_slope_gen=pt["rel"],
                      rel_slope_real=rel_real, b_real=b, alpha_real=alpha,
                      shadow_spearman_gen=stats.spearman(cot_g, sh_g),
                      shadow_spearman_real=stats.spearman(cot_r, sh_r), gate="real slope > 0, p < 0.05")]
        if bool((r.cfg.get("v2_1") or {}).get("e3_shadows")):
            rows += self.shadow_rows(f, cells)
        return rows

    # ---------------------------------------------------------------- v2.1 shadows (9 Nov 2026)
    def _shadow_store(self):
        def comp():
            import os
            from metacf.v2.core.features import FeatureStore
            root = os.path.join(self.run.root, "_features", "shadow_v21") if self.run.persist else None
            return FeatureStore(root, chunk=512)
        return self.run.cached(("E3", "shadow_store"), comp)

    def _shadow_vecs(self, items):
        """items: list of (cache key, image loader, axis angle) -> (n, VEC) array. Cached in
        results_v2/_features/shadow_v21 (real tiles shared across models). CPU pool of `shadow_workers`."""
        from metacf.v2.physics import shadows as S
        fs = self._shadow_store()
        todo = [(k, ld, a) for k, ld, a in dict((k, (k, ld, a)) for k, ld, a in items).values() if not fs.has(k)]
        if todo:
            nw = int(self.c("shadow_workers", 4))
            from concurrent.futures import ProcessPoolExecutor
            B = 64
            with ProcessPoolExecutor(nw) as ex:
                for i in range(0, len(todo), B):
                    ch = todo[i:i + B]
                    imgs = [ld() for _, ld, _ in ch]
                    vecs = list(ex.map(S.shadow_vector, imgs, [a for _, _, a in ch], chunksize=4))
                    fs.put([k for k, _, _ in ch], np.stack(vecs))
            fs.flush()
        return np.stack([fs.mem[k] for k, _, _ in items]).astype(float)

    def shadow_rows(self, f, cells):
        """E3 shadow parts, each its own row (no composite):
          shadow_direction (2000 shared TEST records, own metadata = C3's generations; real = the same records):
            mean cos(estimated - expected anti-solar direction), gen vs real; s_hat = max(0, gen / real);
            hemisphere accuracy reported. Gate: real mean cos > 0 (site bootstrap p, BH) and >= min_real_dir.
          shadow_area (same records): OLS slope of log(local-contrast dark fraction) on log cot(e) with category
            fixed effects + log effective GSD, gen vs real (the along-axis segment length is reported, not
            scored: on reals it does not grow along the axis more than across it);
            s_hat = max(0, 1 - |log2(b_gen / b_real)|) (0 when b_gen <= 0). Gate: b_real > 0 (one-sided p, BH).
          shadow_area_sweep (the E3 elevation-sweep generations; models that condition on sun elevation):
            per-anchor slope of log shadow area on log cot(e), median over anchors, vs the same b_real (a clean
            intervention on the generated side; the real side stays cross-sectional). Gate as shadow_area.
          shadow_direction_sweep (same sweep images; azimuth fixed at the anchor's own value): mean cos, gen vs
            the real mean cos above."""
        import math
        from metacf.physical import expected_shadow_angle
        from metacf.v2.measures.c2_transfer import test_tiles
        from metacf.v2.measures.c3_information import own_requests
        from metacf.v2.physics import shadows as S
        from metacf.v2.core import features as FT
        r = self.run
        fd = r.fields[f]
        st = r.store()
        rows = []
        tt = [x for x in test_tiles(r) if x.fields.get("sun_azimuth_deg") is not None
              and fd.raw(x.fields) is not None and 1.0 <= float(fd.raw(x.fields)) < 89.0]
        own = {q.rid if hasattr(q, "rid") else None: q for q in own_requests(r)}
        reqs = [own.get(x.rid) for x in tt]
        keep = [i for i, q in enumerate(reqs) if q is not None and st.has(q)]
        tt, reqs = [tt[i] for i in keep], [reqs[i] for i in keep]
        if len(tt) < 50:
            return [N.invalid(self.id, f, "shadow_direction", f"only {len(tt)} own-metadata generations on disk",
                              status="PENDING")]
        exp = np.array([float(expected_shadow_angle(x.fields["sun_azimuth_deg"])) for x in tt])
        Vr = self._shadow_vecs([(FT.real_key(x.rid), (lambda x=x: r.ds.real_image(x.rid)), a) for x, a in zip(tt, exp)])
        Vg = self._shadow_vecs([(FT.gen_key(st.key(q)), (lambda q=q: st.get(q)), a) for q, a in zip(reqs, exp)])
        sites = [x.location_id for x in tt]
        _, sidx = np.unique(sites, return_inverse=True)
        lat = np.array([float(x.fields["lat_deg"]) for x in tt])

        def dirs(V):
            return np.array([S.direction_from_vector(v) for v in V])
        cg, cr = np.cos(S.angdiff(dirs(Vg), exp)), np.cos(S.angdiff(dirs(Vr), exp))

        def dstat(w=None):
            ww = np.ones(len(tt)) if w is None else N.site_weights_to_items(w, sidx)
            ok_g, ok_r = np.isfinite(cg), np.isfinite(cr)
            g = float(np.average(cg[ok_g], weights=ww[ok_g])) if ww[ok_g].sum() > 0 else np.nan
            rr = float(np.average(cr[ok_r], weights=ww[ok_r])) if ww[ok_r].sum() > 0 else np.nan
            return dict(g=g, r=rr, s=max(0.0, g / rr) if np.isfinite(g) and np.isfinite(rr) and rr > 0 else np.nan)
        pt = dstat()
        bs = N.site_bootstrap(dstat, sites, r.B, self.seed("shadow_dir"))
        p_r = N.boot_p_greater(bs["r"])
        hem = {}
        for nm, c in (("gen", dirs(Vg)), ("real", dirs(Vr))):
            mN = math.atan2(np.mean(np.sin(exp[lat > 15])), np.mean(np.cos(exp[lat > 15]))) if (lat > 15).any() else np.nan
            mS = math.atan2(np.mean(np.sin(exp[lat < -15])), np.mean(np.cos(exp[lat < -15]))) if (lat < -15).any() else np.nan
            m = (np.abs(lat) > 15) & np.isfinite(c)
            hits = (np.abs(S.angdiff(c[m], mS)) < np.abs(S.angdiff(c[m], mN))) == (lat[m] < -15)
            hem[nm] = float(hits.mean()) if m.any() else np.nan
        rows.append(N.row(self.id, f, "shadow_direction", raw=pt["g"], ceil=pt["r"], s_hat=pt["s"],
                          ci=N.percentile_ci(bs["s"]), n=len(tt), seeds=1, p=p_r, space="shadow (local-contrast mask)",
                          gen_value=pt["g"], real_value=pt["r"], hemisphere_acc_gen=hem["gen"],
                          hemisphere_acc_real=hem["real"], n_south=int((lat < -15).sum()),
                          min_real=float(self.c("min_real_dir", 0.1)), gate_kind="dir",
                          gate="real mean cos > 0 (BH) and >= min_real_dir"))
        # ---- length on the shared records
        cat = np.array([x.category for x in tt])
        eff = np.array([float(x.fields.get("gsd_eff_m") or np.nan) for x in tt])
        cot = 1 / np.tan(np.radians([float(fd.raw(x.fields)) for x in tt]))

        def slope(V, w=None, col=S.NDIR + 2):
            # col NDIR + 2 = local-contrast dark fraction ("shadow area"); NDIR = D-segment length along the
            # anti-solar axis (reported only: on reals it grows with cot(e) no more than ACROSS the axis does,
            # so it is not a directional shadow length; validate_shadows.py, 9 Nov)
            L = V[:, col] * (eff if col == S.NDIR else 1.0)
            ok = np.isfinite(L) & (L > 0) & np.isfinite(eff) & (eff > 0)
            if w is not None:
                ok &= w > 0
            if ok.sum() < 30:
                return np.nan, np.nan
            uc = sorted(set(cat[ok]))
            X = np.column_stack([np.ones(ok.sum()), np.log(cot[ok]), np.log(eff[ok])] +
                                [(cat[ok] == u).astype(float) for u in uc[1:]])
            y = np.log(L[ok])
            ww = np.ones(ok.sum()) if w is None else w[ok]
            sw = np.sqrt(ww)
            bta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
            if w is not None:
                return float(bta[1]), np.nan
            res = y - X @ bta
            dof = max(len(y) - X.shape[1], 1)
            se = math.sqrt(float(res @ res / dof) * np.linalg.pinv(X.T @ X)[1, 1])
            return float(bta[1]), se

        def lscore(bg, br):
            if not (np.isfinite(bg) and np.isfinite(br)) or br <= 0:
                return np.nan
            return 0.0 if bg <= 0 else max(0.0, 1 - abs(math.log2(bg / br)))
        bg, _ = slope(Vg)
        br, se_r = slope(Vr)
        from scipy import stats as ss
        p_len = float(ss.norm.sf(br / se_r)) if np.isfinite(se_r) and se_r > 0 else np.nan

        def lstat(w=None):
            ww = None if w is None else N.site_weights_to_items(w, sidx)
            g, _ = slope(Vg, ww)
            rr, _ = slope(Vr, ww)
            return dict(s=lscore(g, rr), g=g, r=rr)
        bsl = N.site_bootstrap(lstat, sites, min(r.B, 200), self.seed("shadow_len"))
        rows.append(N.row(self.id, f, "shadow_area", raw=bg, ceil=br, s_hat=lscore(bg, br),
                          ci=N.percentile_ci(bsl["s"]), n=len(tt), seeds=1, p=p_len, space="shadow (local-contrast mask)",
                          gen_value=bg, real_value=br, se_real=se_r, gate_kind="len",
                          axis_len_slope_gen=slope(Vg, col=S.NDIR)[0], axis_len_slope_real=slope(Vr, col=S.NDIR)[0],
                          caveat="real reference is cross-sectional: low sun co-varies with season / latitude",
                          gate="real slope of log shadow area on log cot(e) > 0 (BH)"))
        # ---- elevation sweep (the E3 cells)
        sw_items, sw_meta = [], []
        for c in cells:
            if c["reqs"] and all(st.has(q) for q in c["reqs"]):
                a = r.rec(c["anchor"])
                if a.fields.get("sun_azimuth_deg") is None:
                    continue
                ang = float(expected_shadow_angle(a.fields["sun_azimuth_deg"]))
                for q, e in zip(c["reqs"], c["e"]):
                    sw_items.append((FT.gen_key(st.key(q)), (lambda q=q: st.get(q)), ang))
                    sw_meta.append((c["anchor"], c["site"], float(e), ang))
        if len(sw_items) >= 25:
            Vs = self._shadow_vecs(sw_items)
            per = {}
            for (anc, site, e, ang), v in zip(sw_meta, Vs):
                per.setdefault(anc, dict(site=site, x=[], y=[], c=[]))
                if np.isfinite(v[S.NDIR + 2]) and v[S.NDIR + 2] > 0 and 1.0 <= e < 89.0:
                    per[anc]["x"].append(math.log(1 / math.tan(math.radians(e))))
                    per[anc]["y"].append(math.log(v[S.NDIR + 2]))
                per[anc]["c"].append(math.cos(float(S.angdiff(S.direction_from_vector(v), ang))))
            sl = np.array([stats.slope(np.array(d["x"]), np.array(d["y"])) if len(d["x"]) >= 3 else np.nan
                           for d in per.values()])
            cs = np.array([np.nanmean(d["c"]) for d in per.values()])
            ssites = [d["site"] for d in per.values()]
            _, ssidx = np.unique(ssites, return_inverse=True)

            def swstat(w=None):
                v = sl if w is None else np.repeat(sl, N.site_weights_to_items(w, ssidx).astype(int))
                c = cs if w is None else np.repeat(cs, N.site_weights_to_items(w, ssidx).astype(int))
                g = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
                return dict(s=lscore(g, br), g=g, d=float(np.nanmean(c)) if np.isfinite(c).any() else np.nan)
            pw = swstat()
            bsw = N.site_bootstrap(swstat, ssites, r.B, self.seed("shadow_sweep"))
            rows.append(N.row(self.id, f, "shadow_area_sweep", raw=pw["g"], ceil=br, s_hat=pw["s"],
                              ci=N.percentile_ci(bsw["s"]), n=len(per), seeds=1, p=p_len,
                              space="shadow (local-contrast mask)", gen_value=pw["g"], real_value=br, gate_kind="len",
                              gate="real slope of log shadow area on log cot(e) > 0 (BH)"))
            dsw = max(0.0, pw["d"] / pt["r"]) if np.isfinite(pw["d"]) and pt["r"] > 0 else np.nan
            rows.append(N.row(self.id, f, "shadow_direction_sweep", raw=pw["d"], ceil=pt["r"], s_hat=dsw,
                              ci=N.percentile_ci([max(0.0, b["d"] / pt["r"]) if pt["r"] > 0 else np.nan
                                                  for b in [dict(d=x) for x in bsw["d"]]]),
                              n=len(per), seeds=1, p=p_r, space="shadow (local-contrast mask)", gen_value=pw["d"],
                              real_value=pt["r"], min_real=float(self.c("min_real_dir", 0.1)), gate_kind="dir",
                              gate="real mean cos > 0 (BH) and >= min_real_dir"))
        return rows

    def finalize(self, rows):
        for x in rows:
            if x["measure"] == self.id and x["status"] == "OK" and x["metric"] != "score":     # v2.1 shadow rows
                e = x["extra"]
                bad = None
                if not (np.isfinite(x["p_bh"]) and x["p_bh"] <= self.run.q):
                    bad = "gate: real shadow reference not significant after BH"
                elif e.get("gate_kind") == "dir" and not (e.get("real_value", -1) >= e.get("min_real", 0.1)):
                    bad = f"gate: real shadow-direction agreement {e.get('real_value'):.3f} < {e.get('min_real')}"
                elif e.get("gate_kind") == "len" and not (e.get("real_value", -1) > 0):
                    bad = "gate: real shadow-length slope on cot(e) not > 0"
                if bad:
                    e["s_hat_ungated"] = x["s_hat"]
                    x["status"], x["s_hat"], x["reason"] = "INVALID", float("nan"), bad
                continue
            if x["measure"] == self.id and x["status"] == "OK":
                if not (np.isfinite(x["p_bh"]) and x["p_bh"] <= self.run.q and x["extra"].get("b_real", 0) > 0):
                    x["extra"]["s_hat_ungated"] = x["s_hat"]
                    x["status"], x["s_hat"] = "INVALID", float("nan")
                    x["reason"] = "gate: real luminance slope on sin(e) not > 0 significantly after BH"
        return rows
