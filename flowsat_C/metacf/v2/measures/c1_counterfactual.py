"""
C1 COUNTERFACTUAL FIDELITY (spec section 2), per regression field valid after selection (1.5), in the
field's selected space.

  Real reference: B from the pair regression (1.4) on TEST pairs.
  Generation: for field f, up to n_pairs (400) TEST pairs with |delta_f| above its median nonzero |delta|
    (categorical: class changed). A_hat = G(c_a, m_a, s), B_hat = G(c_a, m_a[f <- m_b,f], s), one seed per
    pair (seed = f(pair), so A_hat is shared when two fields pick the same pair). If f contains doy, the
    sun geometry of B_hat is recomputed with derived_solar from I_a's lat/lon, its time-of-day offset and
    the new date (metacf.solar.consistent). The generated intervention vector delta_m_hat (transformed,
    derived changes included) is recorded.
  Generated effects: OLS of Delta_hat = phi(B_hat) - phi(A_hat) on delta_m_hat, pooled over all generated
    pairs of all fields -> B_hat.
  direction_f = cos(B_hat_f, B_f); ceiling = TEST split-half reliability of B_f; floor = 0 (analytic: two
    unrelated directions in d dims have expected cosine 0); s_hat = direction / ceiling.
  magnitude: kappa_f = <B_hat_f, B_f> / ||B_f||^2; score = max(0, 1 - |log2 kappa|), 0 if kappa <= 0.
  specificity: leak_fj = ||P_j (B_hat_f - B_f)|| / ||B_f||, P_j = projection of each feature-space row onto
    the row space of B_j (for single-row blocks this is exactly |<B_hat_f - B_f, u_j>| / ||B_f|| with
    u_j = B_j / ||B_j||; multi-row generalisation, Q12); specificity_f = 1 - mean_j min(1, leak_fj).
  per-pair agreement: r = Delta - sum_{j != f} B_j delta_m_j (real pair, real B); Spearman over the
    generated pairs of f between <Delta_hat, u_f> and <r, u_f>; u_f = B_f / ||B_f|| (multi-row: its top
    right-singular vector, Q12).
  CIs: joint bootstrap over TEST sites (real and generated pairs resampled together by source site).
  Gate: B_f split-half reliability on TEST significant (bootstrap p, BH).
  C1 score_f = mean(direction s_hat, magnitude score, specificity).

v2_1 (9 Nov 2026, `v2_1.gsd_reference: zoom`): for the gsd field the real reference B_f is replaced by
B_zoom, the zoom direction from synthetic pairs of real chips (metacf/v2/core/zoom.py), because same-site
real pairs keep the ground footprint fixed and so define a blur direction. Ceiling and gate = B_zoom's
split-half reliability over the same TEST sites; the joint bootstrap resamples it with the same site weights.
Generations are unchanged (same requests). Default `real_pairs` = v2 behaviour.
"""
from __future__ import annotations

import numpy as np

from metacf import solar
from metacf.stats import spearman
from metacf.v2.core import norm as N
from metacf.v2.core.pairs import above_median_nonzero, block_abs
from metacf.v2.core.regression import SiteStats, cos, split_half
from metacf.v2.measures.base import Measure


def mag_score(kappa):
    if not np.isfinite(kappa) or kappa <= 0:
        return 0.0
    return float(max(0.0, 1.0 - abs(np.log2(kappa))))


def leak(D, Bj, Bf_norm):
    """||P_j D|| / ||B_f||, rows of D projected onto the row space of B_j."""
    Bj = np.atleast_2d(Bj)
    if np.linalg.norm(Bj) < 1e-15 or Bf_norm < 1e-15:
        return float("nan")
    U, s, Vt = np.linalg.svd(Bj, full_matrices=False)
    V = Vt[s > s.max() * 1e-8]
    P = np.atleast_2d(D) @ V.T
    return float(np.linalg.norm(P) / Bf_norm)


def u_dir(Bf):
    Bf = np.atleast_2d(Bf)
    if Bf.shape[0] == 1:
        n = np.linalg.norm(Bf)
        return Bf[0] / n if n > 0 else Bf[0]
    _, _, Vt = np.linalg.svd(Bf, full_matrices=False)
    return Vt[0]


class C1Counterfactual(Measure):
    id = "C1"
    name = "counterfactual fidelity"

    def fields(self):
        r = self.run
        return [f for f in r.design.field_names() if r.conditioned(f)]

    def valid_fields(self):
        return [f for f in self.fields() if self.run.valid(f)[0]]

    def chosen(self):
        def comp():
            r = self.run
            ps = r.pairs["test"]
            out = {}
            for f in self.valid_fields():
                cols = r.design.blocks[f]
                kind = "categorical" if all(r.fields[m].kind == "categorical" for m in r.design.members(f)) else "x"
                m = above_median_nonzero(block_abs(ps.dM, cols, kind), kind)
                idx = np.where(m)[0]
                n = int(self.c("n_pairs", 400))
                if len(idx) > n:
                    idx = np.sort(np.random.default_rng(self.seed("pairs", f)).choice(idx, n, replace=False))
                out[f] = idx
            return out
        return self.run.cached(("C1", "chosen"), comp)

    def requests(self):
        def comp():
            r = self.run
            ps = r.pairs["test"]
            out = {}
            for f, idx in self.chosen().items():
                members = r.design.members(f)
                lst = []
                for i in idx:
                    a, b = r.rec(ps.a[i]), r.rec(ps.b[i])
                    sd = r.seed("C1pair", a.rid, b.rid)
                    new = dict(a.fields)
                    for m in members:
                        for k in r.fields[m].keys:
                            new[k] = b.fields.get(k)
                    mode = "RAW"
                    if any(r.fields[m].tag == "doy" for m in members) and bool(self.c("consistent_doy", True)):
                        new = solar.consistent(a.fields, new)
                        mode = "CONSISTENT"
                    lst.append(dict(i=int(i), site=ps.site[i], A=r.req(a, a.fields, seed=sd),
                                    B=r.req(a, new, seed=sd, mode=mode),
                                    dmh=(r.design.matrix([new]) - r.design.matrix([a.fields]))[0]))
                out[f] = lst
            return out
        return self.run.cached(("C1", "reqs"), comp)

    def plan(self):
        out = [q for lst in self.requests().values() for d in lst for q in (d["A"], d["B"])]
        from metacf.v2.measures import gsd_zoom as GZ
        return out + GZ.c1_plan(self.run, self)

    # ---------------------------------------------------------------- compute
    def _real(self, sp):
        def comp():
            r = self.run
            ps = r.pairs["test"]
            Y = r.real_feats(sp, ps.b) - r.real_feats(sp, ps.a)
            sites, sidx = ps.site_index()
            return dict(Y=Y, sites=sites, sidx=sidx, ss=SiteStats(ps.dM, Y, sidx, len(sites)))
        return self.run.cached(("C1", "real", sp), comp)

    only_fields = None          # rescore: restrict the computed fields (the pooled generated fit is unchanged)

    def zoom_fields(self):
        from metacf.config import get
        if get(self.run.cfg, "v2_1.gsd_reference", "real_pairs") != "zoom":
            return set()
        return {f for f in self.fields() if self.run.fields.get(f) is not None and self.run.fields[f].tag == "gsd"}

    def compute(self):
        r = self.run
        out = []
        for f in self.fields():
            if self.only_fields is not None and f not in self.only_fields:
                continue
            ok, why = r.valid(f)
            if not ok:
                out.append(N.invalid(self.id, f, "score", f"INVALID after feature-space selection: {why}"))
        reqs = self.requests()
        from metacf.v2.measures import gsd_zoom as GZ
        if GZ.c1_enabled(r) and (self.only_fields is None or GZ.GR.gsd_name(r) in self.only_fields):
            out += GZ.c1_rows(r, self)             # v2_3: gsd scored as zoom with the real-trained reader
        if not reqs:
            return out
        spaces = sorted({r.space_of(f) for f in reqs})
        allp = [(f, d) for f, lst in reqs.items() for d in lst]
        ps = r.pairs["test"]
        blocks = r.design.blocks
        valid = list(reqs)
        for sp in spaces:
            real = self._real(sp)
            ssR = real["ss"]
            BR = ssR.fit()
            rel = split_half(ssR, {f: blocks[f] for f in valid}, int(self.c("n_splits", 50)),
                             self.seed("rel", sp), self.run.B, int(self.c("ci_splits", 10)))
            # generated regression, pooled over all fields' pairs, sites in the TEST site index
            Yg = r.gen_feats(sp, [d["B"] for _, d in allp]) - r.gen_feats(sp, [d["A"] for _, d in allp])
            Xg = np.array([d["dmh"] for _, d in allp])
            pos = {s: k for k, s in enumerate(real["sites"])}
            gsidx = np.array([pos[d["site"]] for _, d in allp], int)
            ssG = SiteStats(Xg, Yg, gsidx, len(real["sites"]))
            BG = ssG.fit()
            # per-pair agreement inputs
            for f in [x for x in valid if r.space_of(x) == sp]:
                if self.only_fields is not None and f not in self.only_fields:
                    continue
                rows = [k for k, (ff, _) in enumerate(allp) if ff == f]
                Z = None
                if f in self.zoom_fields():
                    from metacf.v2.core import zoom
                    Z = zoom.reference(r, sp, "zoom", int(self.c("n_splits", 50)), int(self.c("ci_splits", 10)))
                    relf = Z["rel"]
                else:
                    relf = rel[f]
                res = self._field(f, sp, BR, BG, relf, Yg[rows], [allp[k][1] for k in rows], real, valid,
                                  Bf_override=None if Z is None else Z["B"])
                # joint site bootstrap
                Bw = self.run.B
                rng = np.random.default_rng(self.seed("boot", f))
                S = ssR.S
                vals = {k: [] for k in ("direction", "kappa", "specificity", "agreement", "score")}
                chunk = 100
                pair_site = gsidx[rows]
                for c0 in range(0, Bw, chunk):
                    W = np.array([np.bincount(rng.integers(0, S, S), minlength=S).astype(float)
                                  for _ in range(min(chunk, Bw - c0))])
                    BRs, BGs = ssR.fit_many(W), ssG.fit_many(W)
                    BZs = Z["ss"].fit_many(W) if Z is not None else None
                    for k in range(len(W)):
                        rr = self._field(f, sp, BRs[k], BGs[k], relf, Yg[rows], [allp[q][1] for q in rows],
                                         real, valid, w_pairs=W[k][pair_site], BR_override=BRs[k],
                                         Bf_override=None if BZs is None else BZs[k])
                        rr = dict(rr, direction=rr["dir_hat"])        # the direction CI is on s_hat
                        for key in vals:
                            vals[key].append(rr[key])
                cis = {k: N.percentile_ci(v) for k, v in vals.items()}
                p_gate = relf.get("p", float("nan"))
                common = dict(n=len(rows), seeds=1, space=sp,
                              gate="TEST split-half reliability of B_f" if Z is None else
                              "split-half reliability of B_zoom (synthetic zoom pairs, TEST sites)",
                              reference="real_pairs" if Z is None else "zoom")
                if Z is not None:
                    from metacf.v2.core.regression import cos as _cos
                    common.update(zoom_ref=dict(n_pairs=Z["n"], fit_r2=Z["fit_r2"]),
                                  cos_gen_vs_realpairs=_cos(BG[blocks[f]], BR[blocks[f]]),
                                  cos_zoom_vs_realpairs=_cos(Z["B"], BR[blocks[f]]))
                out.append(N.row(self.id, f, "direction", raw=res["direction"], floor=0.0, ceil=relf["mean"],
                                 ci=cis["direction"], reliability_test=relf, **common))
                out.append(N.row(self.id, f, "magnitude", raw=res["kappa"], s_hat=res["mag"],
                                 ci=cis["kappa"], ci_on="kappa", mag_minratio=N.bound(res["kappa"], True), **common))
                out.append(N.row(self.id, f, "specificity", raw=res["specificity"], s_hat=res["specificity"],
                                 ci=cis["specificity"], leak=res["leak"], **common))
                out.append(N.row(self.id, f, "agreement", raw=res["agreement"], s_hat=float("nan"),
                                 ci=cis["agreement"], **common))
                out.append(N.row(self.id, f, "score", raw=res["score"], s_hat=res["score"], ci=cis["score"],
                                 p=p_gate, fit_r2_real=ssR.r2(BR), fit_r2_gen=ssG.r2(BG),
                                 direction_s_hat=res["dir_hat"], magnitude_score=res["mag"],
                                 specificity=res["specificity"], kappa=res["kappa"], **common))
        return out

    def _field(self, f, sp, BR, BG, rel, Ygf, dlist, real, valid, w_pairs=None, BR_override=None,
               Bf_override=None):
        r = self.run
        blocks = r.design.blocks
        Bf, Bhf = BR[blocks[f]], BG[blocks[f]]
        if Bf_override is not None:              # v2_1 zoom reference for gsd
            Bf = np.asarray(Bf_override).reshape(Bf.shape)
        nf = np.linalg.norm(Bf)
        # a generated block that is exactly zero (field ignored) has direction 0 = the floor, not NaN
        direction = 0.0 if np.linalg.norm(Bhf) < 1e-12 else cos(Bhf, Bf)
        ceil = rel["mean"]
        dir_hat = direction / ceil if np.isfinite(ceil) and abs(ceil) > 1e-12 else float("nan")
        if r.cfg.get("v2_3") and bool(r.cfg["v2_3"].get("bounded", True)):
            dir_hat = N.bound(dir_hat)             # v2_3: the composite uses the bounded direction score
        kappa = float(np.sum(Bhf * Bf) / nf ** 2) if nf > 0 else float("nan")
        mag = mag_score(kappa)
        D = Bhf - Bf
        lk = {j: leak(D, BR[blocks[j]], nf) for j in valid if j != f}
        lv = [min(1.0, v) for v in lk.values() if np.isfinite(v)]
        spec = float(1 - np.mean(lv)) if lv else 1.0
        u = u_dir(Bf)
        ps = r.pairs["test"]
        idx = [d["i"] for d in dlist]
        others = [c for j in blocks if j != f for c in blocks[j]]
        Rres = real["Y"][idx] - ps.dM[idx][:, others] @ BR[others] if others else real["Y"][idx]
        g = Ygf @ u
        rr = Rres @ u
        if w_pairs is not None:
            rep = np.repeat(np.arange(len(g)), np.asarray(w_pairs, int))
            g, rr = g[rep], rr[rep]
        agr = spearman(g, rr)
        comps = [dir_hat, mag, spec]
        score = float(np.mean(comps)) if all(np.isfinite(comps)) else float("nan")
        return dict(direction=direction, dir_hat=dir_hat, kappa=kappa, mag=mag, specificity=spec,
                    leak={k: float(v) for k, v in lk.items()}, agreement=agr, score=score)

    def finalize(self, rows):
        q = self.run.q
        gate = {}
        for x in rows:
            if x["measure"] == self.id and x["metric"] == "score" and x["status"] == "OK":
                gate[x["field"]] = bool(np.isfinite(x["p_bh"]) and x["p_bh"] <= q)
        for x in rows:
            if x["measure"] == self.id and x["field"] in gate and not gate[x["field"]] \
                    and not x["metric"].startswith("zoom_"):          # v2_3 zoom rows carry their own gate
                x["extra"]["s_hat_ungated"] = x["s_hat"]
                x["status"], x["s_hat"] = "INVALID", float("nan")
                x["reason"] = "gate: TEST split-half reliability of B_f not significant after BH"
        return rows
