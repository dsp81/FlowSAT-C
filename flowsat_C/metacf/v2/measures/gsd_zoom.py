"""
MetaCF v2.3 (11 Nov 2026): GSD scored as ZOOM in C1 and C2, with the real-trained reader of core/gsd_reader.py
(log effective GSD read off an image; validated on within-scene zoom ladders of real chips before any generated
image is scored). Nothing here is fitted to generated outputs except the C2 synthetic reader, by definition.

C1-zoom (`v2_3.c1_zoom: {n_pairs, factor}`), rows under C1 for the gsd field:
  Pairs: the A images of the C1 gsd pairs (TEST record a at its OWN metadata, seed = C1 pair seed) and two
  counterfactuals with the same caption / seed / other fields: gsd x factor ("up") and gsd / factor ("down").
  d_gen  = read(B) - read(A) (up) or read(A) - read(B) (down), mean over pairs (log GSD units).
  d_real = mean read(coarse) - read(fine_factor) over real zoom pairs of ONE chip (core/zoom.py, exact zoom by
           `factor`, same pixel count): what the reader sees when a real scene is zoomed by that factor.
  zoom_up / zoom_down: raw = kappa = d_gen / d_real (1 = a real-scale zoom); s_hat = kappa bounded
  symmetrically (v2_3). zoom_sign_up / zoom_sign_down: fraction of pairs that move in the right direction.
  CI: independent site bootstraps of the generated pairs and of the real chips.
  Range covered: "up" asks for own -> factor x own, "down" for own -> own / factor; the row extras give the
  requested 5-95 % range and how much of it lies inside the TRAIN [1, 99] % support of the anchor's category.

C2 within-scene (`v2_3.c2_within_scene: true`), rows under C2 for the gsd field:
  The synthetic reader (trained on generated images, label = requested GSD) is asked to ORDER the levels of real
  zoom ladders of one scene, instead of predicting GSD across scenes (where the category gives most of R^2).
  rho = mean per-ladder Spearman(read-off, true log GSD); slope = pooled within-ladder slope.
  tau_within_scene: raw = rho_syn, floor = rho of the label-permuted synthetic readers, ceil = rho of the real
  reader (C2's own real reader, same space). slope_within_scene: raw = slope_syn, ceil = slope_real.
  Gate: rho_real - rho_floor > 0 (ladder-site bootstrap p, BH).
  In-range version = C2's own generated set (values inside the category support). `_full` rows = a synthetic
  reader trained on the E1 dense sweep generations (0.3-3.5 m), when those exist.
"""
from __future__ import annotations

import numpy as np

from metacf.config import get
from metacf.v2.core import gsd_reader as GR
from metacf.v2.core import norm as N


def cfg(run, key, default=None):
    return get(run.cfg, f"v2_3.{key}", default)


# ------------------------------------------------------------------------------------------------ C1 zoom
def c1_settings(run):
    s = dict(n_pairs=200, factor=2.0)
    v = cfg(run, "c1_zoom", None)
    if isinstance(v, dict):
        s.update(v)
    return s


def c1_enabled(run):
    return bool(cfg(run, "c1_zoom", False)) and GR.gsd_name(run) is not None and run.conditioned(GR.gsd_name(run))


def c1_cells(run, c1):
    def comp():
        st = c1_settings(run)
        f = GR.gsd_name(run)
        fd = run.fields[f]
        ps = run.pairs["test"]
        F = float(st["factor"])
        out, seen = [], set()
        for d in c1.requests().get(f, []):
            a = run.rec(ps.a[d["i"]])
            g = fd.raw(a.fields)
            if a.rid in seen or g is None:
                continue
            seen.add(a.rid)
            sd = d["A"].seed
            out.append(dict(rid=a.rid, site=a.location_id, category=a.category, g=float(g), A=d["A"],
                            up=run.req(a, fd.set(a.fields, float(g) * F), seed=sd, tag="C1zoom"),
                            down=run.req(a, fd.set(a.fields, float(g) / F), seed=sd, tag="C1zoom")))
            if len(out) >= int(st["n_pairs"]):
                break
        return out
    return run.cached(("gz", "c1cells"), comp)


def c1_plan(run, c1):
    return [q for c in c1_cells(run, c1) for q in (c["A"], c["up"], c["down"])] if c1_enabled(run) else []


def real_zoom_delta(run, sp, factor):
    """-> (per-chip read(coarse) - read(fine_factor), per-chip site)."""
    from metacf.v2.core import zoom
    ps = [(rid, site) for rid, site, k in zoom.pairs(run) if abs(k - factor) < 1e-9]
    if not ps:
        return np.zeros(0), []
    a = GR.predict(run, sp, run.syn_feats(sp, [zoom.syn_key(r, "fine", factor) for r, _ in ps]))
    b = GR.predict(run, sp, run.syn_feats(sp, [zoom.syn_key(r, "coarse") for r, _ in ps]))
    return np.asarray(b - a, float), [s for _, s in ps]


def _site_w(sites, rng):
    us, inv = np.unique(np.asarray(sites, object).astype(str), return_inverse=True)
    return np.bincount(rng.integers(0, len(us), len(us)), minlength=len(us)).astype(float)[inv]


def c1_rows(run, c1, measure_id="C1"):
    f = GR.gsd_name(run)
    st = c1_settings(run)
    F = float(st["factor"])
    ch = GR.choose(run)
    names = ("zoom_up", "zoom_down", "zoom_sign_up", "zoom_sign_down")
    if ch["space"] is None:
        return [N.invalid(measure_id, f, m, ch["reason"]) for m in names + ("zoom_in_range",)]
    return c1_in_range_rows(run, c1, ch, measure_id) + _c1_extrap_rows(run, c1, ch, measure_id)


def c1_in_range_rows(run, c1, ch, measure_id="C1"):
    """PRIMARY C1 for gsd (v2.3-final): the EXISTING in-range C1 gsd pairs (A = TEST record a at its own metadata,
    B = the same request with gsd replaced by that of another real view of the same site; same caption / seed),
    read with the real-trained reader. kappa = OLS slope through the origin of (read(B) - read(A)) on the requested
    d log gsd, divided by c_within = the reader's slope on real zoom ladders of one scene (1 = the image zooms as
    much as a real scene would for that request). Score = min(kappa, 1 / kappa), 0 when kappa <= 0."""
    f = GR.gsd_name(run)
    fd = run.fields[f]
    lst = c1.requests().get(f, [])
    if not lst:
        return [N.invalid(measure_id, f, "zoom_in_range", "no C1 gsd pairs (field invalid after selection)")]
    sp = ch["space"]
    cw = GR.reader(run, sp)["validity"]["c_within"]
    pA = GR.predict(run, sp, run.gen_feats(sp, [d["A"] for d in lst]))
    pB = GR.predict(run, sp, run.gen_feats(sp, [d["B"] for d in lst]))
    x = np.log([float(fd.raw(d["B"].field_dict)) / float(fd.raw(d["A"].field_dict)) for d in lst])
    y = pB - pA
    sites = [d["site"] for d in lst]

    def stat(w):
        den = float(w @ (x * x))
        return dict(kappa=float(w @ (x * y)) / den / cw if den > 0 else np.nan,
                    sign=float(w @ ((np.sign(y) == np.sign(x)) & (x != 0)) / max(float(w @ (x != 0)), 1e-12)))
    pt = stat(np.ones(len(x)))
    rng = np.random.default_rng(run.seed("gz", "c1inboot"))
    bs = dict(kappa=[], sign=[])
    for _ in range(run.B):
        for k, v in stat(_site_w(sites, rng)).items():
            bs[k].append(v)
    ax = np.abs(x[x != 0])
    com = dict(n=len(lst), seeds=1, space=sp, status="OK" if ch["valid"] else "INVALID", reason=ch["reason"],
               c_within=cw, reference="reader slope on real within-scene zoom ladders (c_within)",
               abs_dlog_p5_p50_p95=[float(np.percentile(ax, q)) for q in (5, 50, 95)],
               range=f"IN RANGE: same-site TEST pairs, requested gsd change x{np.exp(np.percentile(ax, 5)):.2f} to "
                     f"x{np.exp(np.percentile(ax, 95)):.2f} (median x{np.exp(np.median(ax)):.2f}), both ends real "
                     f"values of the site")
    return [N.row(measure_id, f, "zoom_in_range", raw=pt["kappa"], s_hat=pt["kappa"],
                  ci=N.percentile_ci(bs["kappa"]), **com),
            N.row(measure_id, f, "zoom_sign_in_range", raw=pt["sign"], s_hat=pt["sign"],
                  ci=N.percentile_ci(bs["sign"]), **com)]


def _c1_extrap_rows(run, c1, ch, measure_id="C1"):
    f = GR.gsd_name(run)
    st = c1_settings(run)
    F = float(st["factor"])
    names = ("zoom_up", "zoom_down", "zoom_sign_up", "zoom_sign_down")
    cells = c1_cells(run, c1)
    store = run.store()
    miss = sum(1 for c in cells for q in (c["A"], c["up"], c["down"]) if not store.has(q))
    if miss or not cells:
        return [N.invalid(measure_id, f, m, f"C1-zoom images not generated ({miss} missing)", status="PENDING")
                for m in names]
    sp = ch["space"]
    pA = GR.predict(run, sp, run.gen_feats(sp, [c["A"] for c in cells]))
    pU = GR.predict(run, sp, run.gen_feats(sp, [c["up"] for c in cells]))
    pD = GR.predict(run, sp, run.gen_feats(sp, [c["down"] for c in cells]))
    dU, dD = pU - pA, pA - pD
    dR, rsites = real_zoom_delta(run, sp, F)
    if len(dR) < 20:
        return [N.invalid(measure_id, f, m, f"only {len(dR)} real zoom pairs at factor {F}") for m in names]
    sites = [c["site"] for c in cells]
    g = np.array([c["g"] for c in cells])
    sup = run.support(f)
    rng_of = lambda c: sup.by_cat.get(c["category"], sup.global_range)
    in_up = float(np.mean([rng_of(c)[1] <= c["g"] * F <= rng_of(c)[2] for c in cells]))
    in_dn = float(np.mean([rng_of(c)[1] <= c["g"] / F <= rng_of(c)[2] for c in cells]))
    rng = np.random.default_rng(run.seed("gz", "c1boot"))
    reps = dict(up=[], down=[], sup=[], sdn=[])
    for _ in range(run.B):
        w, wr = _site_w(sites, rng), _site_w(rsites, rng)
        dr = float(wr @ dR / wr.sum())
        reps["up"].append(float(w @ dU / w.sum()) / dr if dr > 0 else np.nan)
        reps["down"].append(float(w @ dD / w.sum()) / dr if dr > 0 else np.nan)
        reps["sup"].append(float(w @ (dU > 0) / w.sum()))
        reps["sdn"].append(float(w @ (dD > 0) / w.sum()))
    dr = float(dR.mean())
    p_gate = N.boot_p_greater([float(_site_w(rsites, rng) @ dR) for _ in range(run.B)])
    q = lambda v: [float(np.percentile(v, 5)), float(np.percentile(v, 95))]
    com = dict(n=len(cells), seeds=1, space=sp, status="OK" if ch["valid"] else "INVALID", reason=ch["reason"],
               reference="real zoom pairs of one chip (core/zoom.py)", factor=F, d_real=dr, n_real=len(dR),
               real_sign_frac=float(np.mean(dR > 0)), own_gsd_p5_p95=q(g),
               gate="real zoom delta > 0 and the reader's within-scene check")
    ku, kd = float(dU.mean()) / dr, float(dD.mean()) / dr
    return [
        N.row(measure_id, f, "zoom_up", raw=ku, s_hat=ku, ci=N.percentile_ci(reps["up"]), p=p_gate,
              d_gen=float(dU.mean()), requested_p5_p95=q(g * F), frac_target_in_support=in_up,
              range=f"EXTRAPOLATION: own gsd -> x{F:g} (requested {q(g * F)[0]:.2f}-{q(g * F)[1]:.2f} m, "
                    f"{100 * in_up:.0f}% of targets inside the category TRAIN support)", **com),
        N.row(measure_id, f, "zoom_down", raw=kd, s_hat=kd, ci=N.percentile_ci(reps["down"]), p=p_gate,
              d_gen=float(dD.mean()), requested_p5_p95=q(g / F), frac_target_in_support=in_dn,
              range=f"EXTRAPOLATION: own gsd -> /{F:g} (requested {q(g / F)[0]:.2f}-{q(g / F)[1]:.2f} m, "
                    f"{100 * in_dn:.0f}% of targets inside the category TRAIN support)", **com),
        N.row(measure_id, f, "zoom_sign_up", raw=float(np.mean(dU > 0)), s_hat=float(np.mean(dU > 0)),
              ci=N.percentile_ci(reps["sup"]), **com),
        N.row(measure_id, f, "zoom_sign_down", raw=float(np.mean(dD > 0)), s_hat=float(np.mean(dD > 0)),
              ci=N.percentile_ci(reps["sdn"]), **com)]


# ------------------------------------------------------------------------------------------------ C2 within-scene
def ladder_arrays(run, sp, head):
    """-> per-ladder (rho, Sxy, Sxx, site) for a head predicting the gsd target in space sp."""
    L = GR.ladders(run)
    keys = [k for v in L.values() for k, _ in v]
    P = dict(zip(keys, np.asarray(head.predict(run.syn_feats(sp, keys)), float).reshape(len(keys), -1)[:, 0]))
    rho, sxy, sxx, sites = [], [], [], []
    for rid, v in L.items():
        x = np.array([t for _, t in v])
        p = np.array([P[k] for k, _ in v])
        rho.append(GR.spearman(x, p))
        xd = x - x.mean()
        sxy.append(float(np.sum(xd * (p - p.mean()))))
        sxx.append(float(np.sum(xd * xd)))
        sites.append(run.rec(rid).location_id)
    return np.array(rho), np.array(sxy), np.array(sxx), sites


def c2_rows(run, c2, name, sp, syn, perms, real_head, suffix="", note="", measure_id="C2"):
    fd = run.fields[name]
    R = ladder_arrays(run, sp, real_head)
    S = ladder_arrays(run, sp, syn)
    Fl = [ladder_arrays(run, sp, h) for h in perms]
    sites = R[3]
    if len(sites) < 20:
        return [N.invalid(measure_id, name, "tau_within_scene" + suffix, f"only {len(sites)} real zoom ladders")]
    fl_rho = np.nanmean(np.stack([a[0] for a in Fl]), 0) if Fl else np.zeros(len(sites))

    def stat(w=None):
        ww = np.ones(len(sites)) if w is None else w
        m = lambda v: float(np.nansum(ww * np.nan_to_num(v)) / np.sum(ww * np.isfinite(v))) if np.sum(ww * np.isfinite(v)) > 0 else np.nan
        rs, rr, rf = m(S[0]), m(R[0]), m(fl_rho)
        ss = float(ww @ S[1] / (ww @ S[2]))
        sr = float(ww @ R[1] / (ww @ R[2]))
        return dict(rho_syn=rs, rho_real=rr, rho_floor=rf, gap=rr - rf,
                    tau=(rs - rf) / (rr - rf) if abs(rr - rf) > 1e-12 else np.nan,
                    slope_syn=ss, slope_real=sr, slope_ratio=ss / sr if abs(sr) > 1e-12 else np.nan)
    pt = stat()
    rng = np.random.default_rng(run.seed("gz", "c2boot", name, suffix))
    bs = {k: [] for k in pt}
    for _ in range(run.B):
        for k, v in stat(_site_w(sites, rng)).items():
            bs[k].append(v)
    p_gate = N.boot_p_greater(bs["gap"])
    com = dict(n=len(sites), seeds=1, space=sp, p=p_gate, n_ladders=len(sites), train_set=note,
               gate="real reader's within-scene Spearman above the permuted floor (BH)")
    return [N.row(measure_id, name, "tau_within_scene" + suffix, raw=pt["rho_syn"], floor=pt["rho_floor"],
                  ceil=pt["rho_real"], ci=N.percentile_ci(bs["tau"]), **com),
            N.row(measure_id, name, "slope_within_scene" + suffix, raw=pt["slope_syn"], floor=0.0,
                  ceil=pt["slope_real"], ci=N.percentile_ci(bs["slope_ratio"]), **com)]


def c2_full_reader(run, c2, name, sp):
    """Synthetic reader + permuted floors trained on the E1 dense sweep generations (0.3-3.5 m). None if absent."""
    from metacf.v2.core.heads import Head, permute_within
    from metacf.v2.physics import e1_match
    if not e1_match.enabled(run):
        return None
    cells = e1_match.cells(run)
    store = run.store()
    if not cells or any(not store.has(q) for c in cells for q in c["reqs"]):
        return None
    fd = run.fields[name]
    reqs = [q for c in cells for q in c["reqs"]]
    y = fd.target([float(v) for c in cells for v in c["g"]])
    groups = [c["anchor"] for c in cells for _ in c["g"]]
    X = run.gen_feats(sp, reqs)
    lams = c2.lambdas(fd.kind)
    syn = Head(fd.kind, lambdas=lams).fit(X, y, groups)
    perms = [Head(fd.kind, lambdas=lams).fit(X, permute_within(groups, y, run.seed("gz", "c2perm", k)), groups)
             for k in range(int(c2.c("n_perm", 5)))]
    vals = sorted({float(v) for c in cells for v in c["g"]})
    return syn, perms, f"E1 dense sweep: {len(cells)} sweeps x {len(vals)} levels {vals[0]:g}-{vals[-1]:g} m"
