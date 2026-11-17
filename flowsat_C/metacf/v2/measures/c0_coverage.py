"""
C0 COVERAGE (spec section 2), per metadata group g (geo and cov included).

  Anchors: n_anchors (20) TEST records with every enabled field present, stratified by category; the same
  anchors serve every group. Seeds: n_seeds (3). Two settings per anchor and group, everything else fixed
  (RAW mode: derived solar fields are NOT recomputed, so a group change is that group alone; Q11):
    continuous    TRAIN 5th vs 95th percentile of the group's primary field
    circular      the anchor's value vs value + period/2 (doy: +182 days, as specified; 182.6 = half a year
                  is not used to match the spec text)
    categorical   the two most frequent TRAIN classes
    geo           anchor location vs a random in-support location >= 3000 km away (in support = a TRAIN
                  location of the same category; any category when none qualifies)
    cov           anchor covariates vs those of the in-support (same-category TRAIN) record with the largest
                  z-scored covariate distance
  A group's primary field is its first enabled field in config order (solar -> sun_elev).
  d = distance in each configured space (cosine for dino / clip, Euclidean for stats); the primary space
  is the group's selected space (via its primary field), else the default space.
  effect_a = mean_s d(low_s, high_s); noise_a = mean_{s<s'} d(low_s, low_s'); ratio = mean effect / mean noise.
  Active if ratio > ratio_min (0.1) AND a one-sided Wilcoxon signed-rank test over anchors is significant
  after BH. MODIFIED (Q10): the spec's test of effect_a > noise_a would demand ratio > 1 and contradict the
  0.1 threshold, so the test is on effect_a - ratio_min * noise_a > 0 (the effect exceeds the threshold
  fraction of the seed noise); the supplementary thresholds use effect_a - t * noise_a, unadjusted. Coverage = #active / #groups the model conditions on, also
  at ratio thresholds {0.05, 0.1, 0.2}.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st

from metacf.metrics import great_circle_km
from metacf.v2.core import features as FT
from metacf.v2.core import norm as N
from metacf.v2.measures.base import Measure, pick_by_category


class C0Coverage(Measure):
    id = "C0"
    name = "coverage"

    def groups(self):
        out = {}
        for n, fd in self.run.fields.items():
            if fd.enabled:
                out.setdefault(fd.group, []).append(n)
        return out

    def group_conditioned(self, g, names):
        return any(self.run.conditioned(n) for n in names)

    def anchors(self):
        def comp():
            r = self.run
            elig = [x for x in r.test if r.complete(x)]
            return pick_by_category(elig, int(self.c("n_anchors", 20)), self.seed("anchors"))
        return self.run.cached(("C0", "anchors"), comp)

    # ---------------------------------------------------------------- settings
    def _cov_names(self, names):
        return [n for n in names if self.run.fields[n].kind != "geo"]

    def _cov_matrix(self, recs, names):
        r = self.run
        cols = []
        for n in names:
            fd = r.fields[n]
            vals = [fd.raw(x.fields) for x in recs]
            if fd.kind == "categorical":
                classes = fd.classes or sorted({int(v) for v in vals if v is not None})
                for c in classes[:12]:
                    cols.append(np.array([np.nan if v is None else float(int(v) == c) for v in vals]))
            else:
                v = np.array([np.nan if x is None else float(x) for x in vals])
                cols.append(np.sign(v) * np.log1p(np.abs(v)))
        return np.column_stack(cols) if cols else np.zeros((len(recs), 0))

    def settings(self, anchor, g, names):
        """-> (low fields, high fields) or None with a reason."""
        r = self.run
        base = dict(anchor.fields)
        fds = [r.fields[n] for n in names]
        if any(fd.kind == "geo" for fd in fds):
            fd = [x for x in fds if x.kind == "geo"][0]
            la, lo = fd.raw(base)
            mind = float(self.c("geo_min_km", 3000))
            rng = np.random.default_rng(self.seed("geo", anchor.rid))
            for pool in ("same", "any"):
                cand = self._train_locs(anchor.category if pool == "same" else None)
                if len(cand) == 0:
                    continue
                d = great_circle_km(la, lo, cand[:, 0], cand[:, 1])
                ok = np.where(d >= mind)[0]
                if len(ok):
                    k = int(rng.choice(ok))
                    return base, fd.set(base, (cand[k, 0], cand[k, 1]))
            return None
        if all(fd.location for fd in fds):            # covariate group
            cn = self._cov_names(names)
            pool = [x for x in r.train if x.category == anchor.category and r.complete(x, cn)]
            if not pool:
                return None
            M = self._cov_matrix(pool + [anchor], cn)
            mu, sd = np.nanmean(M[:-1], 0), np.nanstd(M[:-1], 0) + 1e-9
            Z = np.nan_to_num((M - mu) / sd)
            k = int(np.argmax(((Z[:-1] - Z[-1]) ** 2).sum(1)))
            hi = dict(base)
            for n in cn:
                for key in r.fields[n].keys:
                    hi[key] = pool[k].fields.get(key)
            if "cov.landcover_hist" in pool[k].fields:
                hi["cov.landcover_hist"] = pool[k].fields["cov.landcover_hist"]
            return base, hi
        fd = fds[0]
        if fd.kind == "circular":
            per = fd.period or 365.25
            half = float(self.c("circular_shift_days", 182.0)) if per > 300 and per < 400 else per / 2
            v = float(fd.raw(base))
            return base, fd.set(base, (v + half) % per)
        if fd.kind == "categorical":
            if len(fd.classes) < 2:
                return None
            return fd.set(base, fd.classes[0]), fd.set(base, fd.classes[1])
        v = r.train_values(fd.name)
        v = v[np.isfinite(v)]
        lo_p, hi_p = self.c("percentiles", [5, 95])
        return fd.set(base, float(np.percentile(v, lo_p))), fd.set(base, float(np.percentile(v, hi_p)))

    def _setting_note(self, g, names, ext=False):
        fd = self.run.fields[names[0]]
        x2 = (self.run.cfg.get("v2_3") or {}).get("c0_gsd_factor")
        if ext:
            return f"EXTRAPOLATION: anchor's own gsd vs own x {float(x2):g}"
        if fd.kind == "continuous" and not fd.location:
            v = self.run.train_values(fd.name)
            v = v[np.isfinite(v)]
            lo_p, hi_p = self.c("percentiles", [5, 95])
            return f"TRAIN p{lo_p} {np.percentile(v, lo_p):.3g} vs p{hi_p} {np.percentile(v, hi_p):.3g}"
        return {"circular": "own vs own + half period", "categorical": "two most frequent TRAIN classes",
                "geo": "own vs a TRAIN location >= 3000 km away"}.get(fd.kind, "own vs most distant covariates")

    def _train_locs(self, category):
        def comp():
            r = self.run
            geo = [fd for fd in r.fields.values() if fd.kind == "geo"][0]
            seen, out = set(), {}
            for x in r.train:
                v = geo.raw(x.fields)
                if v is None or x.location_id in seen:
                    continue
                seen.add(x.location_id)
                out.setdefault(x.category, []).append(v)
            return {k: np.array(v, float) for k, v in out.items()}
        locs = self.run.cached(("C0", "trainlocs"), comp)
        if category is None:
            return np.concatenate(list(locs.values())) if locs else np.zeros((0, 2))
        return locs.get(category, np.zeros((0, 2)))

    def cells(self):
        def comp():
            out = {}
            S = int(self.c("n_seeds", 3))
            for g, names in self.groups().items():
                if not self.group_conditioned(g, names):
                    continue
                rows = []
                for a in self.anchors():
                    s = self.settings(a, g, names)
                    if s is None:
                        continue
                    lo, hi = s
                    seeds = [self.run.seed("C0seed", a.rid, k) for k in range(S)]
                    rows.append(dict(anchor=a.rid, site=a.location_id,
                                     low=[self.run.req(a, lo, seed=sd) for sd in seeds],
                                     high=[self.run.req(a, hi, seed=sd) for sd in seeds]))
                out[g] = rows
                x2 = (self.run.cfg.get("v2_3") or {}).get("c0_gsd_factor")
                fd0 = self.run.fields[names[0]]
                if x2 and fd0.tag == "gsd":        # v2_3 EXTRAPOLATION row: own value vs own x factor
                    ext = []
                    for a in self.anchors():
                        base = dict(a.fields)
                        hi = fd0.set(base, float(fd0.raw(base)) * float(x2))
                        seeds = [self.run.seed("C0seed", a.rid, k) for k in range(S)]
                        ext.append(dict(anchor=a.rid, site=a.location_id,
                                        low=[self.run.req(a, base, seed=sd) for sd in seeds],
                                        high=[self.run.req(a, hi, seed=sd) for sd in seeds]))
                    out[g + "@extrap"] = ext
            return out
        return self.run.cached(("C0", "cells"), comp)

    def plan(self):
        return [q for rows in self.cells().values() for c in rows for q in c["low"] + c["high"]]

    def primary_space(self, names):
        return self.run.space_of(names[0]) if not self.run.fields[names[0]].location else \
            self.run.cfg.get("default_space", "dino")

    def compute(self):
        r = self.run
        out = []
        thr = float(self.c("ratio_min", 0.1))
        groups = self.groups()
        todo = list(groups.items())
        todo += [(g + "@extrap", names) for g, names in groups.items() if g + "@extrap" in self.cells()]
        for g, names in todo:
            ext = g.endswith("@extrap")
            if ext:
                g = g[:-len("@extrap")]
            if not self.group_conditioned(g, names):
                out.append(N.invalid(self.id, g, "ratio", f"model does not condition on group {g}", status="N/A"))
                continue
            cells = self.cells().get(g + "@extrap" if ext else g, [])
            if len(cells) < int(self.c("min_anchors", 5)):
                out.append(N.invalid(self.id, g, "ratio", f"only {len(cells)} anchors have valid settings"))
                continue
            prim = self.primary_space(names)
            spaces = list(dict.fromkeys([prim] + [s for s in r.space_names()]))
            per = {}
            for sp in spaces:
                eff, noi = [], []
                for c in cells:
                    L = r.gen_feats(sp, c["low"])
                    H = r.gen_feats(sp, c["high"])
                    eff.append(float(np.mean(FT.distance(sp, L, H))))
                    S = len(L)
                    noi.append(float(np.mean([FT.distance(sp, L[i:i + 1], L[j:j + 1])[0]
                                              for i in range(S) for j in range(i + 1, S)])) if S > 1 else np.nan)
                per[sp] = (np.array(eff), np.array(noi))
            eff, noi = per[prim]
            ratio = float(np.mean(eff) / np.mean(noi)) if np.mean(noi) > 0 else float("nan")
            def wil(t):
                try:
                    return float(st.wilcoxon(eff - t * noi, alternative="greater").pvalue)
                except ValueError:
                    return float("nan")
            p = wil(thr)
            p_at = {str(t): wil(t) for t in self.c("thresholds", [0.05, 0.1, 0.2])}
            sites = [c["site"] for c in cells]
            _, sidx = np.unique(sites, return_inverse=True)

            def stat(w, e=eff, n_=noi):
                ww = N.site_weights_to_items(w, sidx)
                return float((ww @ e) / (ww @ n_)) if ww @ n_ > 0 else np.nan
            ci = N.percentile_ci(N.site_bootstrap(stat, sites, self.run.B, self.seed("boot", g)))
            ratios = {sp: float(np.mean(e) / np.mean(n)) if np.mean(n) > 0 else float("nan")
                      for sp, (e, n) in per.items()}
            out.append(N.row(self.id, g, "ratio_extrap" if ext else "ratio", raw=ratio, s_hat=float("nan"), ci=ci, n=len(cells),
                             seeds=int(self.c("n_seeds", 3)), p=p, space=prim, ratio_min=thr,
                             ratio_by_space=ratios, effect=float(np.mean(eff)), noise=float(np.mean(noi)),
                             fields=names, thresholds=self.c("thresholds", [0.05, 0.1, 0.2]),
                             setting=self._setting_note(g, names, ext),
                             p_at=p_at, modified=True,
                             test="one-sided Wilcoxon over anchors of effect_a - ratio_min * noise_a (Q10)"))
        return out

    def finalize(self, rows):
        q = self.run.q
        mine = [x for x in rows if x["measure"] == self.id and x["metric"] == "ratio"]
        groups = [x for x in mine if x["status"] != "N/A"]
        thr = float(self.c("ratio_min", 0.1))
        for x in mine:
            if x["status"] != "OK":
                x["extra"]["active"] = False
                continue
            sig = bool(np.isfinite(x["p_bh"]) and x["p_bh"] <= q)
            x["extra"]["active"] = bool(sig and x["raw"] > thr)
            # supplementary thresholds: their own Wilcoxon p (effect_a - t * noise_a), unadjusted, at q
            x["extra"]["active_at"] = {str(t): bool(x["raw"] > t and np.isfinite(x["extra"]["p_at"].get(str(t), np.nan))
                                                    and x["extra"]["p_at"][str(t)] <= q)
                                       for t in self.c("thresholds", [0.05, 0.1, 0.2])}
        n = len(groups)
        act = sum(1 for x in groups if x["extra"].get("active"))
        at = {str(t): (sum(1 for x in groups if x["extra"].get("active_at", {}).get(str(t))) / n if n else float("nan"))
              for t in self.c("thresholds", [0.05, 0.1, 0.2])}
        rows.append(N.row(self.id, "*", "coverage", raw=act / n if n else float("nan"), s_hat=act / n if n else float("nan"),
                          n=n, active=[x["field"] for x in groups if x["extra"].get("active")],
                          coverage_at=at, n_groups=n))
        return rows
