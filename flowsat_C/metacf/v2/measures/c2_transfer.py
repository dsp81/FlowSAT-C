"""
C2 SYNTHETIC-TO-REAL TRANSFER (spec section 2), per non-location field.

  Synthetic set: n_anchors (200) TRAIN records (never TEST; every enabled field present, stratified by
    category x region), each generated at n_values (5) values of f, caption and other fields fixed, 1 seed.
    Values are drawn from the support of f at the anchor (Q13): continuous = uniform in the transformed
    space (log for gsd) inside the anchor category's TRAIN [1st, 99th] percentile range; doy = uniform
    over the year; sensor = classes seen >= min_count times in TRAIN within the anchor's 30 x 60 deg region
    (cycled when fewer than n_values). Label = the set value.
  Reader: linear head on frozen features of f's selected space (default space when f has none): ridge
    (continuous), ridge on (cos, sin) (circular), class-balanced logistic (categorical); lambda by grouped
    CV over anchors.
  Real reader: the same head trained on n_real_train (1000) random TRAIN real tiles (grouped CV over
    location_id). Fit in the fit-readers stage.
  Both evaluated on the same n_test (2000) TEST real tiles (shared with C3 / C4): R^2 / circular R^2 /
    balanced accuracy. Floor: the synthetic reader trained with labels permuted within anchors (mean over
    n_perm permutations).
  tau_f = (s_syn - s_floor) / (s_real - s_floor). Gate: s_real significantly above s_floor (site bootstrap
    p of s_real - s_floor <= 0, BH).
"""
from __future__ import annotations

import numpy as np

from metacf.fields import region_of
from metacf.v2.core import norm as N
from metacf.v2.core.heads import Head, permute_within, score
from metacf.v2.measures.base import Measure, pick_by_category_region


def test_tiles(run):
    """The n_test TEST records shared by C2 / C3 / C4 (every enabled field present)."""
    def comp():
        from metacf.config import get
        n = int(get(run.cfg, "shared.n_test", 2000))
        elig = [x for x in run.test if run.complete(x)]
        return pick_by_category_region(elig, n, run.seed("shared_test"))
    return run.cached(("shared", "test"), comp)


class C2Transfer(Measure):
    id = "C2"
    name = "synthetic-to-real transfer"

    def fields(self):
        r = self.run
        return [n for n, fd in r.fields.items() if fd.enabled and not fd.location and r.conditioned(n)]

    def anchors(self):
        def comp():
            r = self.run
            elig = [x for x in r.train if r.complete(x)]
            return pick_by_category_region(elig, int(self.c("n_anchors", 200)), self.seed("anchors"))
        return self.run.cached(("C2", "anchors"), comp)

    def _region_classes(self, name):
        def comp():
            fd = self.run.fields[name]
            out = {}
            for x in self.run.train:
                v = fd.raw(x.fields)
                if v is None:
                    continue
                reg = region_of(x.fields.get("lat_deg"), x.fields.get("lon_deg"))
                out.setdefault(reg, {}).setdefault(int(v), 0)
                out[reg][int(v)] += 1
            mc = int(self.c("min_count", 5))
            return {k: sorted(c for c, n in d.items() if n >= mc) for k, d in out.items()}
        return self.run.cached(("C2", "regcls", name), comp)

    def values(self, a, name):
        r = self.run
        fd = r.fields[name]
        K = int(self.c("n_values", 5))
        rng = np.random.default_rng(self.seed("values", name, a.rid))
        if fd.kind == "circular":
            per = fd.period or 365.25
            return [float(x) for x in rng.uniform(0, per, K)]
        if fd.kind == "categorical":
            cls = self._region_classes(name).get(region_of(a.fields.get("lat_deg"), a.fields.get("lon_deg"))) \
                or fd.classes[:K]
            cls = list(cls)
            rng.shuffle(cls)
            return [int(cls[k % len(cls)]) for k in range(K)]
        sup = r.support(name)
        rg = sup.by_cat.get(a.category, sup.global_range)
        lo, hi = rg[1], rg[2]
        if fd.transform == "log":
            return [float(np.exp(x)) for x in rng.uniform(np.log(max(lo, 1e-6)), np.log(hi), K)]
        if fd.transform == "sin_deg":
            s = rng.uniform(np.sin(np.radians(max(lo, -90))), np.sin(np.radians(min(hi, 90))), K)
            return [float(np.degrees(np.arcsin(x))) for x in s]
        return [float(x) for x in rng.uniform(lo, hi, K)]

    def requests(self):
        def comp():
            r = self.run
            out = {}
            for name in self.fields():
                fd = r.fields[name]
                lst = []
                for a in self.anchors():
                    sd = r.seed("C2seed", a.rid)
                    for v in self.values(a, name):
                        lst.append(dict(anchor=a.rid, value=v, req=r.req(a, fd.set(a.fields, v), seed=sd)))
                out[name] = lst
            return out
        return self.run.cached(("C2", "reqs"), comp)

    def plan(self):
        return [d["req"] for lst in self.requests().values() for d in lst]

    def kind(self, name):
        k = self.run.fields[name].kind
        return k

    def fit_real(self):
        """fit-readers stage: real reader per field on n_real_train TRAIN tiles."""
        r = self.run
        elig = [x for x in r.train if r.complete(x)]
        n = int(self.c("n_real_train", 1000))
        rng = np.random.default_rng(self.seed("realtrain"))
        recs = [elig[i] for i in sorted(rng.choice(len(elig), min(n, len(elig)), replace=False))]
        out = {}
        for name in self.fields():
            fd = r.fields[name]
            sp = r.space_of(name)
            X = r.real_feats(sp, [x.rid for x in recs])
            y = fd.target([fd.raw(x.fields) for x in recs])
            out[name] = dict(space=sp, head=Head(fd.kind).fit(X, y, [x.location_id for x in recs]),
                             n=len(recs))
        return out

    def compute(self):
        r = self.run
        out = []
        readers = r.readers().get("C2", {})
        tt = test_tiles(r)
        sites = [x.location_id for x in tt]
        _, sidx = np.unique(sites, return_inverse=True)
        v21 = self.v21()
        for name in self.fields():
            fd = r.fields[name]
            if name not in readers:
                out.append(N.invalid(self.id, name, "tau", "no real reader (fit-readers not run for this field)"))
                continue
            sp = readers[name]["space"]
            lst = self.requests()[name]
            Xs = r.gen_feats(sp, [d["req"] for d in lst])
            ys = fd.target([d["value"] for d in lst])
            groups = [d["anchor"] for d in lst]
            lams = self.lambdas(fd.kind)
            syn = Head(fd.kind, lambdas=lams).fit(Xs, ys, groups)
            perms = [Head(fd.kind, lambdas=lams).fit(Xs, permute_within(groups, ys, self.seed("perm", name, k)), groups)
                     for k in range(int(self.c("n_perm", 5)))]
            Xt = r.real_feats(sp, [x.rid for x in tt])
            yt = fd.target([fd.raw(x.fields) for x in tt])
            real_head = readers[name]["head"]
            n_real = readers[name]["n"]
            if v21 and v21.get("n_real_train"):
                real_head, n_real = self.refit_real(name, sp, int(v21["n_real_train"]))
            p_syn, p_real = syn.predict(Xt), real_head.predict(Xt)
            p_fl = [h.predict(Xt) for h in perms]

            def stats_w(w=None):
                ww = None if w is None else N.site_weights_to_items(w, sidx)
                s_syn = score(fd.kind, p_syn, yt, ww)
                s_real = score(fd.kind, p_real, yt, ww)
                s_fl = float(np.nanmean([score(fd.kind, p, yt, ww) for p in p_fl]))
                tau = (s_syn - s_fl) / (s_real - s_fl) if abs(s_real - s_fl) > 1e-12 else np.nan
                return dict(tau=tau, gap=s_real - s_fl, s_syn=s_syn, s_real=s_real, s_floor=s_fl)
            pt = stats_w()
            bs = N.site_bootstrap(stats_w, sites, self.run.B, self.seed("boot", name))
            p_gate = N.boot_p_greater(bs["gap"])
            ext = {}
            if v21 and v21.get("within_category", True) and fd.kind == "continuous":
                ext.update(self.within_scores(name, fd, tt, yt, p_syn, p_real, p_fl))
            if v21:
                ci_r = N.percentile_ci(bs["s_real"])
                ext.update(s_real_lo=ci_r["lo"], min_real=self.min_real(fd, yt), n_real_train=n_real,
                           lam_real_at_edge=bool(real_head.lam in (real_head.lambdas[0], real_head.lambdas[-1])),
                           lam_syn_at_edge=bool(syn.lam in (syn.lambdas[0], syn.lambdas[-1])), gate_kind="c2_v21")
            out.append(N.row(self.id, name, "tau", raw=pt["s_syn"], floor=pt["s_floor"], ceil=pt["s_real"],
                             ci=N.percentile_ci(bs["tau"]), n=len(tt), seeds=1, p=p_gate, space=sp,
                             n_synthetic=len(lst), n_anchors=len(set(groups)), lam_syn=syn.lam,
                             lam_real=real_head.lam, cv_syn=syn.cv_score,
                             gate=("s_real above s_floor (BH) AND s_real CI lower bound >= min_real" if v21 else
                                   "s_real significantly above s_floor"), metric_kind=fd.kind, **ext))
            if fd.tag == "gsd" and bool((r.cfg.get("v2_3") or {}).get("c2_within_scene", False)):
                out += self.within_scene_rows(name, fd, lst, groups, lams)
        return out

    def finalize(self, rows):
        q = self.run.q
        for x in rows:
            if x["measure"] == self.id and x["status"] == "OK":
                e = x["extra"]
                bad = None
                if not (np.isfinite(x["p_bh"]) and x["p_bh"] <= q):
                    bad = "gate: real reader not significantly above the permuted floor after BH"
                elif e.get("gate_kind") == "c2_v21" and not (np.isfinite(e.get("s_real_lo", np.nan))
                                                             and e["s_real_lo"] >= e.get("min_real", 0.0)):
                    bad = (f"gate: the REAL reader cannot read this field on real TEST tiles (score CI lower bound "
                           f"{e.get('s_real_lo', float('nan')):.3f} < {e.get('min_real'):.3f})")
                if bad:
                    e["s_hat_ungated"] = x["s_hat"]
                    x["status"], x["s_hat"], x["reason"] = "INVALID", float("nan"), bad
        return rows

    # ---------------------------------------------------------------- v2.3 (11 Nov 2026)
    def within_scene_rows(self, name, fd, lst, groups, lams):
        """gsd only (measures/gsd_zoom.py): readers are compared on real within-scene zoom ladders, in the space
        that PASSED the within-scene check on real images (core/gsd_reader.choose), not C2's selected space
        (selected for cross-site pair reliability; it may not rank zoom levels of one scene at all).
        Real reader = the shared real-trained GSD reader in that space."""
        from metacf.v2.core import gsd_reader as GR
        from metacf.v2.measures import gsd_zoom as GZ
        r = self.run
        ch = GR.choose(r)
        if ch["space"] is None or not ch["valid"]:
            return [N.invalid(self.id, name, m, ch["reason"] or "no reader space passes the within-scene check")
                    for m in ("tau_within_scene", "tau_within_scene_full")]
        sp = ch["space"]
        real_head = GR.reader(r, sp)["head"]
        Xs = r.gen_feats(sp, [d["req"] for d in lst])
        ys = fd.target([d["value"] for d in lst])
        syn = Head(fd.kind, lambdas=lams).fit(Xs, ys, groups)
        perms = [Head(fd.kind, lambdas=lams).fit(Xs, permute_within(groups, ys, self.seed("permws", name, k)), groups)
                 for k in range(int(self.c("n_perm", 5)))]
        vals = [d["value"] for d in lst]
        out = GZ.c2_rows(r, self, name, sp, syn, perms, real_head, "",
                         f"C2 generations: {len(lst)} images, requested {min(vals):.2f}-{max(vals):.2f} m "
                         f"(inside each category's TRAIN support)")
        full = GZ.c2_full_reader(r, self, name, sp)
        if full is not None:
            out += GZ.c2_rows(r, self, name, sp, full[0], full[1], real_head, "_full", full[2])
        else:
            out.append(N.invalid(self.id, name, "tau_within_scene_full", "E1 dense sweep images not generated",
                                 status="PENDING"))
        return out

    # ---------------------------------------------------------------- v2.1 (9 Nov 2026)
    def v21(self):
        """`v2_1.c2` options (None = the frozen v2 behaviour):
          n_real_train  refit the real reader in compute on this many complete TRAIN tiles (cached features
                        first; the v2 reader used 1000 and was under-trained: c2_audit.py)
          lambdas       'extended' = ridge 1e-6..1e4, logistic 1e-7..1 (the v2 grid's edge was hit)
          min_real      continuous / circular: the real reader's R^2 CI lower bound must reach this;
                        categorical: chance (1 / n_classes) + min_real for balanced accuracy
          within_category  report R^2 beyond the category (s_*_within_cat extras; tau is invariant to it)
        """
        v = (self.run.cfg.get("v2_1") or {}).get("c2")
        return v or None

    def lambdas(self, kind):
        v = self.v21()
        if not v or v.get("lambdas") != "extended":
            return None
        if kind == "categorical":
            return [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
        return [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4]

    def min_real(self, fd, yt):
        m = float((self.v21() or {}).get("min_real", 0.05))
        if fd.kind == "categorical":
            k = len(set(int(v) for v in yt if v >= 0))
            return 1.0 / max(k, 1) + m
        return m

    def refit_real(self, name, sp, n):
        """Real reader on n complete TRAIN tiles, tiles with cached features in `sp` first (seeded)."""
        from metacf.v2.core import features as FT
        r = self.run
        key = ("C2", "real_v21", name, sp, n)

        def comp():
            elig = [x for x in r.train if r.complete(x)]
            fs = r.fstore(sp)
            rng = np.random.default_rng(self.seed("realtrain_v21", name))
            cached = [x for x in elig if fs.has(FT.real_key(x.rid))]
            rest = [x for x in elig if not fs.has(FT.real_key(x.rid))]
            rng.shuffle(cached)
            rng.shuffle(rest)
            recs = (cached + rest)[:n]
            fd = r.fields[name]
            X = r.real_feats(sp, [x.rid for x in recs])
            y = fd.target([fd.raw(x.fields) for x in recs])
            return Head(fd.kind, lambdas=self.lambdas(fd.kind)).fit(X, y, [x.location_id for x in recs]), len(recs)
        return r.cached(key, comp)

    def within_scores(self, name, fd, tt, yt, p_syn, p_real, p_fl):
        """R^2 after removing the TRAIN category mean of the target from prediction and truth: how much each
        reader knows BEYOND the category. Reported only: tau itself is invariant to it (the prediction error
        is unchanged and every R^2 shares the same denominator, so the ratio of differences is the same), so
        tau already reads as the fraction of the (real reader - category-shortcut floor) gap closed."""
        from metacf.v2.core.heads import r2
        r = self.run

        def comp():
            acc = {}
            for x in r.train:
                v = fd.raw(x.fields)
                if v is not None:
                    acc.setdefault(x.category, []).append(float(fd.target([v])[0]))
            return {c: float(np.mean(v)) for c, v in acc.items()}
        cm = r.cached(("C2", "catmean", name), comp)
        m = np.array([cm.get(x.category, np.nan) for x in tt])
        yr = np.asarray(yt, float) - m
        return dict(s_syn_within_cat=r2(p_syn - m, yr), s_real_within_cat=r2(p_real - m, yr),
                    s_floor_within_cat=float(np.nanmean([r2(p - m, yr) for p in p_fl])))
