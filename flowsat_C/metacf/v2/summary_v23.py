"""
MetaCF v2.3-final consolidated card (11 Nov 2026, user-signed): ONE card per model, every test reported separately,
no combined score, E4 dropped. IN RANGE is primary for every field, GSD included: "in range" = inside the anchor
category's TRAIN [1, 99] % support of the field (for GSD: of effective GSD). The wide 0.3-3.5 m GSD sweep and the
x2 / /2 requests are reported as a separately labelled EXTRAPOLATION result, identical for every model.
C3 floored is primary. Scores are bounded to [0, 1] (ratio metrics: min(k, 1 / k)); the unbounded value is shown.
Reads the rows written by summary.write; writes control_card_v23.md / .html, gsd_curve.png and
results.json["summary_v23"].
"""
from __future__ import annotations

import json
import math
import os

from metacf.v2.summary import _html, _notes, fmt, md_table


def _get(rows, m, f, metric):
    return next((x for x in rows if x["measure"] == m and x["field"] == f and x["metric"] == metric), None)


def cell(x, nd=2):
    """Bounded score, with the unbounded value when it was outside [0, 1]; else the status."""
    if x is None:
        return ""
    s = x.get("s_hat", float("nan"))
    if x.get("status") == "OK" and isinstance(s, (int, float)) and math.isfinite(s):
        u = (x.get("extra") or {}).get("s_hat_unbounded")
        if u is not None and math.isfinite(u) and abs(u - s) > 1e-9:
            return f"{s:.{nd}f} (unbounded {u:.{nd}f})"
        return f"{s:.{nd}f}"
    return x.get("status", "")


def rawci(x, nd=2):
    if x is None:
        return ""
    if x.get("status") not in ("OK",):
        return x.get("status", "")
    return f"{x['raw']:.{nd}f} [{fmt(x['ci_lo'], nd)}, {fmt(x['ci_hi'], nd)}]"


def _c0(x):
    if x is None:
        return ""
    if x["status"] != "OK":
        return x["status"]
    act = (x.get("extra") or {}).get("active")
    return f"{x['raw']:.2f}" + ("" if act is None else (" (active)" if act else " (not active)"))


def table(res):
    rows = res["rows"]
    groups = {t["field"]: t["group"] for t in res["summary"]["table"]}
    c3f = [x["field"] for x in rows if x["measure"] == "C3" and x["metric"] == "ratio"]
    names = list(dict.fromkeys(list(groups) + c3f))
    out = []
    for f in names:
        g = groups.get(f)
        c0 = _c0(_get(rows, "C0", g, "ratio")) if g else ""
        c0x = _get(rows, "C0", g, "ratio_extrap") if g else None
        if c0x is not None:
            c0 = f"in-range {c0} · full-range {_c0(c0x)}"
        c1 = cell(_get(rows, "C1", f, "score"))
        mg = _get(rows, "C1", f, "magnitude")
        alt = ""
        if mg is not None and mg.get("status") == "OK":
            alt = f"{fmt(mg['s_hat'], 2)} / {fmt((mg.get('extra') or {}).get('mag_minratio'), 2)}"
        zi = _get(rows, "C1", f, "zoom_in_range")
        if zi is not None:
            c1 = (f"in-range {cell(zi)} · full-range up {cell(_get(rows, 'C1', f, 'zoom_up'))} / "
                  f"down {cell(_get(rows, 'C1', f, 'zoom_down'))}")
        c2 = cell(_get(rows, "C2", f, "tau"))
        ws = _get(rows, "C2", f, "tau_within_scene")
        if ws is not None:
            c2 = f"in-range {cell(ws)} · full-range {cell(_get(rows, 'C2', f, 'tau_within_scene_full'))}"
        out.append(dict(field=f, group=g or "", C0=c0, C1=c1, C1_mag_alt=alt, C2=c2,
                        C3_floored=cell(_get(rows, "C3", f, "ratio_floored")),
                        C3=cell(_get(rows, "C3", f, "ratio")), C4=cell(_get(rows, "C4", f, "score"))))
    return out


def gsd_block(res, d=None):
    rows = res["rows"]
    f = next((x["field"] for x in rows if x["measure"] == "E1" and x["metric"].startswith("match_")), None)
    if f is None:
        return []
    L = ["", "## GSD panel (GSD is scored as zoom, with a GSD reader trained on real images only)", "",
         "PRIMARY = in range (inside the anchor category's TRAIN [1, 99] % support of effective GSD), comparable with "
         "the other fields. EXTRAPOLATION = the wide 0.3-3.5 m sweep and the x2 / /2 requests, mostly outside that "
         "support; the same protocol for every model.", ""]
    T = []

    def add(kind, lab, x, rng=None, val=None):
        if x is None:
            return
        e = x.get("extra") or {}
        T.append([kind, lab, val or rawci(x), cell(x) if math.isfinite(x.get("s_hat", float("nan")) or float("nan")) else "",
                  rng or e.get("range") or e.get("setting") or ""])

    def c2val(x):
        return x["status"] if x["status"] != "OK" else \
            f"{x['raw']:.2f} (floor {fmt(x['floor'], 2)}, real {fmt(x['ceil'], 2)})"
    add("PRIMARY", "C0 ratio (effect / seed noise)", _get(rows, "C0", "gsd", "ratio"))
    add("PRIMARY", "C1 zoom, in range: read-off change / real zoom change", _get(rows, "C1", f, "zoom_in_range"))
    add("PRIMARY", "C1 zoom, in range: fraction of pairs moving the right way", _get(rows, "C1", f, "zoom_sign_in_range"),
        "as above (0.5 = chance)")
    x = _get(rows, "C2", f, "tau_within_scene")
    if x:
        add("PRIMARY", "C2 within-scene: syn reader orders real zoom ladders (Spearman)", x,
            (x.get("extra") or {}).get("train_set"), c2val(x))
    x = _get(rows, "C2", f, "slope_within_scene")
    if x:
        add("PRIMARY", "C2 within-scene slope", x, (x.get("extra") or {}).get("train_set"), c2val(x))
    for m, lab in (("match_spearman_in", "E1 match Spearman, in range"), ("match_slope_in", "E1 match slope / real slope, in range")):
        x = _get(rows, "E1", f, m)
        if x:
            c = (x.get("extra") or {}).get("counts") or {}
            add("PRIMARY", lab, x, f"sweep levels inside the category support; median {c.get('levels_in_median', '?')} "
                                   f"levels, {c.get('sweeps_in', '?')} sweeps")
    add("EXTRAPOLATION", "C0 ratio", _get(rows, "C0", "gsd", "ratio_extrap"))
    add("EXTRAPOLATION", "C1 zoom OUT (x2): read-off change / real x2 zoom", _get(rows, "C1", f, "zoom_up"))
    add("EXTRAPOLATION", "C1 zoom IN (/2): read-off change / real x2 zoom", _get(rows, "C1", f, "zoom_down"))
    add("EXTRAPOLATION", "C1 zoom OUT: fraction of pairs zooming out", _get(rows, "C1", f, "zoom_sign_up"), "as zoom OUT")
    add("EXTRAPOLATION", "C1 zoom IN: fraction of pairs zooming in", _get(rows, "C1", f, "zoom_sign_down"), "as zoom IN")
    for m, lab in (("tau_within_scene_full", "C2 within-scene Spearman, syn reader trained on the full sweep"),
                   ("slope_within_scene_full", "C2 within-scene slope, full sweep")):
        x = _get(rows, "C2", f, m)
        if x:
            add("EXTRAPOLATION", lab, x, (x.get("extra") or {}).get("train_set"), c2val(x))
    vals = None
    for m, lab in (("match_spearman_full", "E1 match Spearman, full sweep"),
                   ("match_slope_full", "E1 match slope / real slope, full sweep"),
                   ("match_monotone_full", "E1 match: sweeps strictly monotone"),
                   ("match_seed_agree", "E1 match: seed agreement"),
                   ("match_step_gsd", "requested GSD at the largest rise of the mean curve (m)"),
                   ("match_step_share", "largest rise / total rise (1 = one step)")):
        x = _get(rows, "E1", f, m)
        if x:
            e = x.get("extra") or {}
            vals = e.get("values") or vals
            rng = f"{len(vals)} levels {vals[0]:g}-{vals[-1]:g} m, 2 seeds" if vals else "full sweep"
            if m == "match_step_share":
                rng = f"an even ramp would give {fmt(e.get('even_ramp'), 2)}"
            add("EXTRAPOLATION", lab, x, rng)
    add("secondary", "C1 feature-direction score (v2.1 zoom reference)", _get(rows, "C1", f, "score"),
        "same-site TEST pairs, in range")
    x = _get(rows, "C2", f, "tau")
    if x:
        add("secondary", "C2 tau, cross-scene R^2 (v2)", x, "5 values inside the category support; 2000 real TEST tiles", c2val(x))
    for m, lab in (("match_spearman_in_e1cells", "E1 match Spearman, in range (v2 E1 cells)"),
                   ("match_slope_in_e1cells", "E1 match slope, in range (v2 E1 cells)")):
        add("secondary", lab, _get(rows, "E1", f, m), "5 levels inside the category support, 1 seed")
    L.append(md_table(["", "test", "value [95% CI]", "bounded score", "range covered"], T))
    zu, zd = _get(rows, "C1", f, "zoom_up"), _get(rows, "C1", f, "zoom_down")
    if zu and zd and zu["status"] == "OK" and zd["status"] == "OK":
        L += ["", f"Zoom-out / zoom-in asymmetry: a x2 request moves the read-off by {zu['raw']:.2f} of a real x2 zoom "
                  f"({100 * _get(rows, 'C1', f, 'zoom_sign_up')['raw']:.0f}% of pairs zoom out); a /2 request by "
                  f"{zd['raw']:.2f} ({100 * _get(rows, 'C1', f, 'zoom_sign_down')['raw']:.0f}% of pairs zoom in)."]
    st = _get(rows, "E1", f, "match_step_gsd")
    e = (st or {}).get("extra") or {}
    prof = e.get("profile")
    if prof:
        L += ["", "Mean read-off per requested level (fixed caption, seed, other metadata):", "",
              md_table(["requested (m)"] + [f"{p['requested_m']:g}" for p in prof],
                       [["read-off (m)"] + [f"{p['read_m']:.2f}" for p in prof],
                        ["sd of log read-off"] + [f"{p['sd_log']:.2f}" for p in prof]])]
        if d and e.get("c_within"):
            try:
                from metacf.v2 import gsd_plot
                gsd_plot.draw(prof, float(e["c_within"]), float(e["train_median"]), e["train_p1_p99"],
                              f"{res.get('model_name')}: requested vs read-off GSD", os.path.join(d, "gsd_curve.png"),
                              st.get("raw"))
                L += ["", "![requested vs read-off GSD](gsd_curve.png)", "",
                      "`gsd_curve.png`: identical axes for every model; green = what the reader reports when a REAL "
                      "scene is zoomed (slope c_within, anchored at the TRAIN median); dashed = identity."]
            except Exception as ex:                      # the plot never blocks the card
                L += ["", f"(gsd_curve.png not drawn: {ex!r})"]
    return L


def card(res, d):
    rows = res["rows"]
    t = table(res)
    L = [f"# MetaCF v2.3 card: {res.get('model_name')}", "",
         f"config `{res['config_hash']}` · protocol `{res['protocol_hash']}` · checkpoint `{res['checkpoint']}` · "
         f"guidance {res.get('guidance')} · B={res['bootstrap_B']} · BH q={res['bh_q']}", "",
         "Every test is reported on its own; there is no combined score. IN RANGE is primary for every field: "
         "values stay inside the anchor category's TRAIN [1, 99] % support of the field (GSD: of effective GSD). "
         "For GSD each test also has a labelled full-range (EXTRAPOLATION) value. Scores are bounded to [0, 1]: "
         "below 0 -> 0, above 1 -> 1, and for ratio metrics min(k, 1 / k), so an overshoot by a factor k costs as "
         "much as an undershoot by k; the unbounded value is shown when it differs, and CIs are those of the "
         "unbounded value. A cell is INVALID when its real-data reference (ceiling minus floor) is not "
         "significantly above zero. E4 (year ordering) is dropped (dataset-dependent).", "",
         "## Per field",
         md_table(["field", "group", "C0 ratio", "C1", "C1 magnitude: 1-|log2 k| (used) / min(k,1/k)", "C2 tau",
                   "C3 floored (primary)", "C3 unfloored", "C4"],
                  [[x["field"], x["group"], x["C0"], x["C1"], x["C1_mag_alt"], x["C2"], x["C3_floored"], x["C3"],
                    x["C4"]] for x in t]),
         "", "C0 = image change when only this field's group changes / seed-to-seed change (a ratio, not a 0-1 "
         "score). C1 = counterfactual fidelity; for gsd it is a zoom score: the change read off by the real-trained "
         "GSD reader / the change it reads on a real zoom of the same size. C2 = a reader trained on generated "
         "images, tested on reals; for gsd it must order real zoom ladders of one scene. "
         "C3 floored = (gain_gen - gain_floor) / (gain_real - gain_floor), floor = the same model with the field's "
         "group taken from another record (500-record subset). C3 unfloored = gain_gen / gain_real; a value above 1 "
         "means the generated images express the field more cleanly than real images do, not that control is "
         "better. C4 = distribution match within bins of the field."]
    L += gsd_block(res, d)
    for meas, title in (("C0", "C0 coverage"), ("C1", "C1 counterfactual fidelity"), ("C2", "C2 transfer"),
                        ("C3", "C3 conditional information"), ("C4", "C4 distribution fidelity"),
                        ("C5", "C5 memorisation (check)"), ("E1", "E1 resolution"), ("E2", "E2 season"),
                        ("E3", "E3 sun")):
        mr = [x for x in rows if x["measure"] == meas]
        if not mr:
            continue
        L += ["", f"## {title}", md_table(
            ["field", "metric", "raw", "floor", "ceil", "score", "unbounded", "CI", "p_BH", "N", "space", "status", "notes"],
            [[x["field"], x["metric"], fmt(x["raw"]), fmt(x["floor"]), fmt(x["ceil"]), fmt(x["s_hat"]),
              fmt((x.get("extra") or {}).get("s_hat_unbounded")),
              f"[{fmt(x['ci_lo'])}, {fmt(x['ci_hi'])}]", fmt(x["p_bh"], 4), x["n"], x["space"],
              x["status"] + (" MODIFIED" if x.get("modified") else ""), _notes(x)] for x in mr])]
    bad = [x for x in rows if x["status"] != "OK"]
    L += ["", "## INVALID / PENDING / N/A cells",
          md_table(["measure", "field", "metric", "status", "reason"],
                   [[x["measure"], x["field"], x["metric"], x["status"], x["reason"]] for x in bad]) if bad else "none"]
    txt = "\n".join(L) + "\n"
    if d:
        with open(os.path.join(d, "control_card_v23.md"), "w") as fh:
            fh.write(txt)
        with open(os.path.join(d, "control_card_v23.html"), "w") as fh:
            fh.write(_html(txt, res).replace("MetaCF v2 card", "MetaCF v2.3 card"))
    return txt, t


def write(run, res):
    txt, t = card(res, run.mdir if run.persist else None)
    res["summary_v23"] = t
    if run.persist:
        p = os.path.join(run.mdir, "results.json")
        if os.path.exists(p):
            with open(p) as fh:
                cur = json.load(fh)
            cur["summary_v23"] = t
            with open(p, "w") as fh:
                json.dump(cur, fh, indent=1, default=str)
    return txt
