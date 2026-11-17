"""
MetaCF v2.1 per-field report (9 Nov 2026), kept OUT of summary.py (owned by another work stream).

Reads a results.json written by summary.write and adds:
  - C3 floored (v2_1.c3_floor) next to the v2 C3 ratio, PENDING when the floor images are not generated;
  - for gsd: C1 against the synthetic zoom reference, E1 (v2), E1 zoom score, zoom slope / blur slope;
  - no combined score (each test is reported separately).
Writes control_card_v21.md next to control_card.md, and results.json["summary_v21"].
"""
from __future__ import annotations

import json
import math
import os

from metacf.v2.summary import fmt, md_table


def _get(rows, m, f, metric):
    return next((x for x in rows if x["measure"] == m and x["field"] == f and x["metric"] == metric), None)


def _val(x):
    if x is None:
        return None
    if x.get("status") == "OK" and math.isfinite(x.get("s_hat", float("nan"))):
        return x["s_hat"]
    return x.get("status")


def table(res, fields=None):
    rows = res["rows"]
    names = fields or list(dict.fromkeys(x["field"] for x in rows if x["measure"] == "C3" and x["field"] != "*"))
    names += [f for f in dict.fromkeys(x["field"] for x in rows if x["measure"] == "E1") if f not in names]
    out = []
    for f in names:
        c1 = _get(rows, "C1", f, "score")
        out.append(dict(field=f, C1=_val(c1), C1_reference=(c1 or {}).get("extra", {}).get("reference"),
                        C2_tau=_val(_get(rows, "C2", f, "tau")),
                        C3_floored=_val(_get(rows, "C3", f, "ratio_floored")),
                        C3_v2=_val(_get(rows, "C3", f, "ratio")),
                        C4=_val(_get(rows, "C4", f, "score")),
                        E1=_val(_get(rows, "E1", f, "score")),
                        E1_zoom=_val(_get(rows, "E1", f, "zoom_score")),
                        zoom_slope=(_get(rows, "E1", f, "zoom_slope") or {}).get("raw"),
                        blur_slope=(_get(rows, "E1", f, "blur_slope") or {}).get("raw"),
                        e1_zoom_label=((_get(rows, "E1", f, "zoom_score") or {}).get("extra") or {}).get("label"),
                        **_match(rows, f)))
    return out


def _match(rows, f):
    """v2_2 E1 GSD matching rows (raw value + CI), present only for the gsd field when enabled."""
    out = {}
    for m in ("spearman_full", "spearman_in", "slope_full", "slope_in", "monotone_full", "monotone_in", "seed_agree"):
        x = _get(rows, "E1", f, "match_" + m)
        if x is None:
            continue
        out["m_" + m] = (x.get("status") if x.get("status") != "OK" else
                         f"{x['raw']:.2f} [{x['ci_lo']:.2f}, {x['ci_hi']:.2f}]")
        if m == "spearman_full":
            out["m_space"] = x.get("space")
    return out


def match_table(t):
    g = [x for x in t if "m_spearman_full" in x]
    if not g:
        return []
    cols = ["spearman_full", "spearman_in", "slope_full", "slope_in", "monotone_full", "monotone_in", "seed_agree"]
    return ["", "## E1 GSD matching (v2.2)", "",
            "Real-trained GSD reader (core/gsd_reader.py, space chosen by the within-scene zoom-ladder check) applied "
            "to GSD sweeps at fixed caption / seed / other metadata. spearman = order of read-off vs requested GSD; "
            "slope = read-off change per unit log GSD / the reader's within-scene slope on real zoom ladders "
            "(1 = real-scale zoom); full = whole sweep (beyond the training range), in = levels inside the category's "
            "train support. Mean [95% site-bootstrap CI].", "",
            md_table(["field", "reader space"] + cols,
                     [[x["field"], x.get("m_space") or ""] + [x.get("m_" + c, "") for c in cols] for x in g])]


def card(res, d):
    t = table(res)
    L = [f"# MetaCF v2.1 per-field report: {res.get('model_name')}", "",
         f"config `{res['config_hash']}` · protocol `{res['protocol_hash']}` · rescored from "
         f"`{(res.get('rescored_from') or {}).get('config_hash', '-')}` (measures "
         f"{(res.get('rescored_from') or {}).get('measures', '-')}). No combined score (each test separately).", "",
         "C1 for gsd uses the synthetic zoom reference (crop of real chips, same pixel count), not same-site pairs. "
         "C3 floored = (gain_gen - gain_floor) / (gain_real - gain_floor), floor = same model with the field's "
         "group nulled; PENDING = floor images not generated yet; 0 by construction for fields the model never sees.",
         "", md_table(["field", "C1", "C1 ref", "C2 tau", "C3 floored", "C3 (v2)", "C4", "E1", "E1 zoom",
                       "zoom slope", "blur slope", "E1 zoom label"],
                      [[x["field"], fmt(x["C1"]), x["C1_reference"] or "", fmt(x["C2_tau"]), fmt(x["C3_floored"]),
                        fmt(x["C3_v2"]), fmt(x["C4"]), fmt(x["E1"]), fmt(x["E1_zoom"]), fmt(x["zoom_slope"]),
                        fmt(x["blur_slope"]), x["e1_zoom_label"] or ""] for x in t])]
    L += match_table(t)
    txt = "\n".join(L) + "\n"
    if d:
        with open(os.path.join(d, "control_card_v21.md"), "w") as fh:
            fh.write(txt)
    return txt, t


def write(run, res):
    txt, t = card(res, run.mdir if run.persist else None)
    res["summary_v21"] = t
    if run.persist:
        p = os.path.join(run.mdir, "results.json")
        if os.path.exists(p):
            with open(p) as fh:
                cur = json.load(fh)
            cur["summary_v21"] = t
            with open(p, "w") as fh:
                json.dump(cur, fh, indent=1, default=str)
    return txt
