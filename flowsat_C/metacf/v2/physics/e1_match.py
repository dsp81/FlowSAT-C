"""
E1 GSD MATCHING (MetaCF v2.2, 9 Nov 2026): "does the image's scale MATCH the requested GSD", not "how far does
the image move". Rows are added to E1 (the v2 / v2.1 E1 rows are kept) when `v2_2.e1_match` is set.

Qualitative basis (runs/eval/v1_gsd_qual/sheet_*.png): raising GSD at fixed caption, seed and other metadata
zooms the scene out (same layout, more and smaller objects), monotone, most change 0.5-1.2 m, saturating >1.8 m.

READER: core/gsd_reader.py (real images only; space chosen by the within-scene zoom-ladder check).
SWEEPS: n_anchors TEST anchors (category x region stratified) x `values` (log-spaced gsd_eff_m, deliberately
  beyond the training support) x n_seeds seeds; caption and every other field fixed. A level is "in range" when it
  lies inside the anchor category's TRAIN [1st, 99th] gsd support. The IN-RANGE scores are computed on the v2 E1
  cells instead (5 log-spaced gsd inside each category's support, 1 seed): category supports are narrow
  (often 0.47-0.98 m), so the full grid has only 0-2 levels inside them. (v1's adapter converts gsd_eff_m to its own
  file-gsd input by the anchor's chip scale, a per-anchor constant in log space: Spearman/slope unaffected.)
SCORES (no composite), anchor-site bootstrap CI, for the full sweep and the in-range levels:
  spearman   = mean per-sweep Spearman(log requested, read-off) (1 = always the right order);
  slope      = mean per-sweep OLS slope of read-off on log requested / c_within (1 = as much scale change per unit
               log GSD as the reader sees on real zoom ladders of one scene);
  monotone   = fraction of sweeps whose read-off strictly increases with requested GSD;
  seed_agree = mean over anchors of the Spearman between the two seeds' read-off profiles.
Gate: the reader's within-scene check (gsd_reader.choose): rows INVALID if no space passes.
Config: v2_2.e1_match {values, n_anchors, n_seeds, min_levels_in} (+ v2_2.gsd_reader).
"""
from __future__ import annotations

import numpy as np

from metacf.config import get
from metacf.v2.core import gsd_reader as GR
from metacf.v2.core import norm as N

DEFAULTS = dict(values=[0.3, 0.45, 0.7, 1.05, 1.6, 2.4, 3.5], n_anchors=50, n_seeds=2, min_levels_in=3)
METRICS = (("match_spearman_full", "rho_full"), ("match_spearman_in", "rho_in"),
           ("match_slope_full", "slope_full"), ("match_slope_in", "slope_in"),
           ("match_monotone_full", "mono_full"), ("match_monotone_in", "mono_in"),
           ("match_seed_agree", "seed_agree"))


def settings(run):
    s = dict(DEFAULTS)
    v = get(run.cfg, "v2_2.e1_match", {})
    s.update(dict(v) if isinstance(v, dict) else {})
    return s


def enabled(run):
    return bool(get(run.cfg, "v2_2.e1_match", False))


def cells(run):
    def comp():
        from metacf.v2.measures.base import pick_by_category_region
        st = settings(run)
        f = GR.gsd_name(run)
        if f is None or not run.conditioned(f):
            return []
        fd = run.fields[f]
        sup = run.support(f)
        elig = [x for x in run.test if run.complete(x)]
        anchors = pick_by_category_region(elig, int(st["n_anchors"]), run.seed("e1m_anchors"))
        vals = np.array([float(v) for v in st["values"]])
        out = []
        for a in anchors:
            rg = sup.by_cat.get(a.category, sup.global_range)
            inr = (vals >= rg[1]) & (vals <= rg[2])
            for s in range(int(st["n_seeds"])):
                sd = run.seed("e1m_seed", a.rid, s)
                out.append(dict(anchor=a.rid, site=a.location_id, seed=s, g=vals, in_range=inr,
                                reqs=[run.req(a, fd.set(a.fields, float(v)), seed=sd, tag="E1M") for v in vals]))
        return out
    return run.cached(("e1m", "cells"), comp)


def plan(run):
    return [q for c in cells(run) for q in c["reqs"]] if enabled(run) else []


def sweep_scores(x, p, inr, c_within, min_in=3):
    o = dict(rho_full=GR.spearman(x, p), slope_full=GR.ols(x, p) / c_within,
             mono_full=float(np.all(np.diff(p[np.argsort(x)]) > 0)))
    if inr.sum() >= min_in:
        xi, pi = x[inr], p[inr]
        o.update(rho_in=GR.spearman(xi, pi), slope_in=GR.ols(xi, pi) / c_within,
                 mono_in=float(np.all(np.diff(pi[np.argsort(xi)]) > 0)))
    else:
        o.update(rho_in=np.nan, slope_in=np.nan, mono_in=np.nan)
    return o


def score_sweeps(X, P, INR, sites, anchors, c_within, B=1000, seed=0, min_in=3):
    """Per-sweep arrays X (log requested), P (read-off), INR (in-range mask); per-sweep site / anchor labels.
    -> (point dict, ci dict, counts). Pure numpy (used by the measure and by offline cross-checks)."""
    per = [sweep_scores(np.asarray(x, float), np.asarray(p, float), np.asarray(i, bool), c_within, min_in)
           for x, p, i in zip(X, P, INR)]
    keys = list(per[0].keys())
    M = {k: np.array([d[k] for d in per], float) for k in keys}
    by_a = {}
    for j, a in enumerate(anchors):
        by_a.setdefault(a, []).append(j)
    agree, agree_site = [], []
    for a, js in by_a.items():
        if len(js) >= 2:
            agree.append(GR.spearman(P[js[0]], P[js[1]]))
            agree_site.append(sites[js[0]])
    agree = np.array(agree, float)
    us = sorted(set(sites))
    pos = {s: i for i, s in enumerate(us)}
    sidx = np.array([pos[s] for s in sites], int)
    aidx = np.array([pos[s] for s in agree_site], int)

    def wmean(v, w):
        ok = np.isfinite(v) & (w > 0)
        return float(np.sum(w[ok] * v[ok]) / np.sum(w[ok])) if ok.any() else float("nan")

    def stat(w=None):
        ww = np.ones(len(us)) if w is None else np.asarray(w, float)
        out = {k: wmean(M[k], ww[sidx]) for k in keys}
        out["seed_agree"] = wmean(agree, ww[aidx]) if len(aidx) else float("nan")
        return out
    pt = stat()
    bs = N.site_bootstrap(stat, list(sites), B, seed)
    ci = {k: N.percentile_ci(v) for k, v in bs.items()}
    n = dict(sweeps=len(per), sweeps_in=int(np.isfinite(M["rho_in"]).sum()), anchors=len(by_a), sites=len(us))
    return pt, ci, n


def rows(run, measure_id="E1"):
    f = GR.gsd_name(run)
    if f is None or not run.conditioned(f):
        return [N.invalid(measure_id, f or "gsd", "match_spearman_full", "no conditioned gsd field", status="N/A")]
    st = settings(run)
    ch = GR.choose(run)
    com = dict(reader_choice=dict(space=ch["space"], valid=ch["valid"], reason=ch["reason"]),
               reader_validation=ch["all"], values=list(map(float, st["values"])), n_seeds=int(st["n_seeds"]))
    if ch["space"] is None:
        return [N.invalid(measure_id, f, m, ch["reason"], **com) for m, _ in METRICS]
    sp = ch["space"]
    cw = GR.reader(run, sp)["validity"]["c_within"]
    X, P, INR, S, A = [], [], [], [], []
    for c in cells(run):
        X.append(np.log(c["g"]))
        P.append(GR.predict(run, sp, run.gen_feats(sp, c["reqs"])))
        INR.append(c["in_range"])
        S.append(c["site"])
        A.append(c["anchor"])
    # full range: the v2.2 sweep cells (beyond the support on purpose; 2 seeds)
    pt, ci, n = score_sweeps(X, P, INR, S, A, cw, run.B, run.seed("e1m_boot"), int(st["min_levels_in"]))
    # in range: the v2 E1 cells (n_values log-spaced gsd INSIDE each anchor category's TRAIN [1st, 99th] support,
    # 1 seed, same fixed caption / other fields). Category supports are narrow (often 0.47-0.98 m), so the
    # full-range grid has only 0-2 levels inside them; these cells were built for exactly this range.
    from metacf.v2.physics.e1_resolution import E1Resolution
    Xi, Pi, Ii, Si, Ai = [], [], [], [], []
    for c in E1Resolution(run).cells():
        Xi.append(np.log(c["g"]))
        Pi.append(GR.predict(run, sp, run.gen_feats(sp, c["reqs"])))
        Ii.append(np.ones(len(c["g"]), bool))
        Si.append(c["site"])
        Ai.append(c["anchor"])
    v23 = (run.cfg.get("v2_3") or {})
    extra_rows = []
    if Xi and not v23.get("e1_in_from_sweep", False):
        pti, cii, ni = score_sweeps(Xi, Pi, Ii, Si, Ai, cw, run.B, run.seed("e1m_boot_in"), 3)
        for k in ("rho_in", "slope_in", "mono_in"):
            pt[k], ci[k] = pti[k], cii[k]
        n["sweeps_in"], n["in_source"] = ni["sweeps_in"], "v2 E1 cells (category support, 1 seed)"
    elif v23.get("e1_in_from_sweep", False):
        # v2_3: the sweep is dense inside the band (>= min_levels_in levels per category support), so the in-range
        # scores come from the SAME sweeps as the full-range ones (2 seeds); the v2 E1-cell version is kept as
        # its own rows (match_*_in_e1cells).
        n["in_source"] = "dense sweep levels inside the anchor category's TRAIN [1, 99] pct support"
        n["levels_in_median"] = float(np.median([int(np.sum(i)) for i in INR]))
        if Xi:
            pti, cii, ni = score_sweeps(Xi, Pi, Ii, Si, Ai, cw, run.B, run.seed("e1m_boot_in"), 3)
            for m, k in (("match_spearman_in_e1cells", "rho_in"), ("match_slope_in_e1cells", "slope_in")):
                extra_rows.append((m, pti[k], cii[k], ni["sweeps_in"]))
    step = step_stats(X, P, S, run.B, run.seed("e1m_step")) if v23 else None
    rs = GR.settings(run)
    status, reason = ("OK", "") if ch["valid"] else ("INVALID", ch["reason"])
    gate = f"reader within-scene Spearman >= {rs['min_within_spearman']} and c_within >= {rs['min_within_slope']}"
    out = [N.row(measure_id, f, m, raw=pt[k], s_hat=pt[k], ci=ci[k], n=n["sweeps"], seeds=int(st["n_seeds"]),
                 status=status, reason=reason, space=sp, gate=gate, counts=n, c_within=cw, **com)
           for m, k in METRICS]
    for m, v, c, nn in extra_rows:
        out.append(N.row(measure_id, f, m, raw=v, s_hat=v, ci=c, n=nn, seeds=1, status=status, reason=reason,
                         space=sp, gate=gate, source="v2 E1 cells (5 levels inside the category support, 1 seed)"))
    if step is not None:
        out.append(N.row(measure_id, f, "match_step_gsd", raw=step["step_gsd"], s_hat=float("nan"), ci=step["ci_gsd"],
                         n=n["sweeps"], seeds=int(st["n_seeds"]), status=status, reason=reason, space=sp,
                         note="requested GSD (m) at the largest single rise of the mean read-off profile",
                         profile=step["profile"], c_within=cw,
                         train_median=float(np.nanmedian(run.train_values(f))),
                         train_p1_p99=[float(np.nanpercentile(run.train_values(f), q)) for q in (1, 99)]))
        out.append(N.row(measure_id, f, "match_step_share", raw=step["share"], s_hat=float("nan"), ci=step["ci_share"],
                         n=n["sweeps"], seeds=int(st["n_seeds"]), status=status, reason=reason, space=sp,
                         note="largest single rise / total rise of the mean profile (1 = one step, "
                              "1 / (levels - 1) = even ramp)", even_ramp=1.0 / max(len(st["values"]) - 1, 1)))
    return out


def step_stats(X, P, sites, B=1000, seed=0):
    """Mean read-off profile over sweeps (all share the same levels): where its largest single rise sits
    (geometric midpoint of the two requested levels) and what share of the total rise it is. Site bootstrap."""
    X0 = np.asarray(X[0], float)
    Pm = np.asarray(P, float)
    us = sorted(set(sites))
    pos = {s: i for i, s in enumerate(us)}
    sidx = np.array([pos[s] for s in sites], int)

    def stat(w=None):
        ww = np.ones(len(us)) if w is None else np.asarray(w, float)
        wi = ww[sidx]
        prof = (wi[:, None] * Pm).sum(0) / wi.sum()
        d = np.diff(prof)
        k = int(np.argmax(d))
        tot = prof[-1] - prof[0]
        return dict(gsd=float(np.exp(0.5 * (X0[k] + X0[k + 1]))), share=float(d[k] / tot) if tot > 0 else float("nan"))
    pt = stat()
    bs = N.site_bootstrap(stat, list(sites), B, seed)
    prof = Pm.mean(0)
    return dict(step_gsd=pt["gsd"], share=pt["share"], ci_gsd=N.percentile_ci(bs["gsd"]),
                ci_share=N.percentile_ci(bs["share"]),
                profile=[dict(requested_m=float(np.exp(x)), read_m=float(np.exp(p)), read_log=float(p),
                              sd_log=float(sd)) for x, p, sd in zip(X0, prof, Pm.std(0))])


def per_sweep_table(run):
    """Diagnostics: read-off per level for every sweep."""
    ch = GR.choose(run)
    if ch["space"] is None:
        return []
    sp = ch["space"]
    return [dict(anchor=c["anchor"], seed=c["seed"], g=c["g"].tolist(), in_range=c["in_range"].tolist(),
                 pred=GR.predict(run, sp, run.gen_feats(sp, c["reqs"])).tolist()) for c in cells(run)]
