"""
results.json -> control_card.md and control_card.html.
Sections: (1) field x Tier-1 matrix with CIs; (2) Tier-2 scorecard per tag with its real-data
validation; (3) diversity panel (T8); (4) pathway panel (T9); (5) MetaCF-U = macro-average over
supported fields of [T1 norm, T3 norm x T4 score, T6a specificity] -- always printed with the full
profile, never alone; (6) every N/A and INVALID cell with its reason.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

TIER1_COLS = ["T1", "T2", "T3", "T4", "T5", "T6"]


def _fmt(x, nd=3):
    if isinstance(x, bool):
        return "yes" if x else "no"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "-"
    return "-" if not math.isfinite(x) else f"{x:.{nd}f}"


def _cell(r):
    """norm (raw for descriptive rows) with its CI; when the CI is on the raw statistic, the raw
    value is shown next to it so the interval always brackets the number it belongs to."""
    ci = ""
    if math.isfinite(r.get("ci_lo", float("nan"))):
        ci = f" [{_fmt(r['ci_lo'], 2)}, {_fmt(r['ci_hi'], 2)}]"
    if r.get("descriptive"):
        s = _fmt(r["raw"]) + ci
    elif r.get("ci_on") == "raw":
        s = f"{_fmt(r['norm'])} (raw {_fmt(r['raw'])}{ci})"
    else:
        s = _fmt(r["norm"]) + ci
    if r.get("out_of_range"):
        s += " (!)"
    return s


def _ok(r):
    return r.get("status", "OK") not in ("N/A", "INVALID", "ERROR")


def metacf_u(rows):
    """Per field: mean of the available [T1 headline norm, T3 norm (RAW) x T4 score, T6a specificity];
    then the macro-average over fields with at least one component."""
    by = {}
    for r in rows:
        if not _ok(r):
            continue
        f = r["field"]
        if r["test"] == "T1" and r["reader"] == "headline":
            by.setdefault(f, {})["T1"] = r["norm"]
        elif r["test"] == "T3" and r["stratum"] == "RAW" and math.isfinite(r["norm"]):
            by.setdefault(f, {}).setdefault("T3", []).append(r["norm"])
        elif r["test"] == "T4" and math.isfinite(r["raw"]):
            by.setdefault(f, {}).setdefault("T4", []).append(r["raw"])
        elif r["test"] == "T6" and r["stratum"] == "T6a" and math.isfinite(r["raw"]):
            by.setdefault(f, {}).setdefault("T6a", []).append(r["raw"])
    per = {}
    for f, d in by.items():
        comps = []
        if "T1" in d and math.isfinite(d["T1"]):
            comps.append(d["T1"])
        if "T3" in d and "T4" in d:
            comps.append(min(d["T3"]) * min(d["T4"]))          # the lower reader, as P3
        if "T6a" in d:
            comps.append(min(d["T6a"]))
        if comps:
            per[f] = float(np.mean(comps))
    return dict(value=float(np.mean(list(per.values()))) if per else float("nan"), per_field=per,
                definition="mean over fields of mean(T1 norm, T3 norm x T4 score, T6a specificity)")


def render_md(res):
    rows = res["rows"]
    L = [f"# MetaCF control card: {res['model']}", "",
         f"checkpoint `{res['checkpoint']}` | adapter `{res['adapter']}` | config hash `{res['config_hash']}`", ""]
    # (1) Tier-1 matrix
    fields = sorted({r["field"] for r in rows if r.get("tier") == 1 and r["field"] not in ("*", "-")})
    L += ["## 1. Tier 1: field x test", "",
          "Normalised scores (raw for descriptive tests) with 95% bootstrap CIs; (!) = outside [0, 1].", "",
          "| field | " + " | ".join(TIER1_COLS) + " |", "|---|" + "---|" * len(TIER1_COLS)]
    for f in fields:
        cells = []
        for t in TIER1_COLS:
            cand = [r for r in rows if r["field"] == f and r["test"] == t and _ok(r)]
            pick = {"T1": [r for r in cand if r["reader"] == "headline"],
                    "T3": [r for r in cand if r["stratum"] == "RAW"],
                    "T5": [r for r in cand if r["stratum"] == "T5b"],
                    "T6": [r for r in cand if r["stratum"] == "T6a"]}.get(t, cand)
            cells.append(_cell(pick[0]) if pick else "N/A")
        L.append(f"| {f} | " + " | ".join(cells) + " |")
    # (2) Tier 2
    L += ["", "## 2. Tier 2: Earth-observation scorecard", "", "| test | field | score | validation | notes |",
          "|---|---|---|---|---|"]
    for r in [r for r in rows if r.get("tier") == 2]:
        st = "" if r.get("stratum") in (None, "all") else " " + r["stratum"]
        L.append(f"| {r['test']}{st} | {r['field']} | {_cell(r) if _ok(r) else 'N/A'} | "
                 f"{r['validation']} | {r.get('reason', r.get('notes', ''))[:80]} |")
    # (3) diversity
    L += ["", "## 3. Diversity (T8)", ""]
    for r in [r for r in rows if r["test"] == "T8" and _ok(r)]:
        e = r.get("extra", {})
        L.append(f"- {r['field']} {r['stratum']}: " + ", ".join(f"{k}={_fmt(v) if isinstance(v, (int, float, bool)) else v}"
                                                              for k, v in e.items() if not isinstance(v, (list, dict))))
    # (4) pathways
    L += ["", "## 4. Pathways (T9)", ""]
    for r in [r for r in rows if r["test"] == "T9"]:
        e = r.get("extra", {})
        L.append(f"- {r['stratum']} {r['field']}: " + (", ".join(
            f"{k}={_fmt(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v}"
            for k, v in e.items() if not isinstance(v, (list, dict))) if _ok(r) else r.get("reason", "")))
    # (5) MetaCF-U
    u = res.get("metacf_u", {})
    L += ["", "## 5. MetaCF-U (read with the full profile above, never alone)", "",
          f"**MetaCF-U = {_fmt(u.get('value'))}** ({u.get('definition', '')})", ""]
    for f, v in sorted(u.get("per_field", {}).items()):
        L.append(f"- {f}: {_fmt(v)}")
    # (6) N/A and INVALID
    L += ["", "## 6. N/A and INVALID cells", "", "| test | field | reader | status | reason |", "|---|---|---|---|---|"]
    for r in rows:
        if not _ok(r) or r.get("validation") == "INVALID":
            L.append(f"| {r['test']} | {r['field']} | {r['reader']} | {r.get('status', r['validation'])} | "
                     f"{r.get('reason', r.get('notes', ''))} |")
    return "\n".join(L) + "\n"


def write_card(results_path):
    res = json.load(open(results_path))
    md = render_md(res)
    d = os.path.dirname(results_path)
    open(os.path.join(d, "control_card.md"), "w").write(md)
    import markdown
    html = markdown.markdown(md, extensions=["tables"])
    open(os.path.join(d, "control_card.html"), "w").write(
        "<!doctype html><html><head><meta charset='utf-8'><title>MetaCF control card</title>"
        "<style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;padding:0 16px}"
        "table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:3px 8px;font-size:13px}</style>"
        f"</head><body>{html}</body></html>")
    return os.path.join(d, "control_card.md")
