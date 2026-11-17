"""
Summary and reporting (spec section 4).

  Benjamini-Hochberg (q = 0.05) over every row with a p-value in the run, then each measure's finalize()
  applies its BH-dependent gate.
  MetaCF = mean over fields the model conditions on (excluding the cov group and any INVALID cells, which
  are listed) of the per-field mean of the AVAILABLE components among (C1 score, C2 tau, C3 s_hat)
  (Q19: a field without any valid component is left out; a joint field's C1 score is credited to each of
  its members). Never printed without the per-field table.
  results/<model>/<config hash>/results.json and control_card.md / .html.
"""
from __future__ import annotations

import html
import json
import math
import os

import numpy as np

from metacf import stats
from metacf.config import get


def apply_bh(rows, q):
    p = np.array([r["p"] for r in rows], float)
    _, adj = stats.benjamini_hochberg(p, q)
    for r, a in zip(rows, adj):
        r["p_bh"] = float(a)
        r["bh_reject"] = bool(np.isfinite(a) and a <= q)
    return rows


def _ok(x):
    return x is not None and x.get("status") == "OK" and math.isfinite(x.get("s_hat", float("nan")))


def per_field_table(run, rows):
    excl = set(get(run.cfg, "summary.exclude_groups", ["cov"]))
    fields = [n for n, fd in run.fields.items() if fd.enabled and run.conditioned(n)]
    get_row = lambda m, f, metric: next((x for x in rows if x["measure"] == m and x["field"] == f
                                        and x["metric"] == metric), None)
    table, listed = [], []
    for f in fields:
        fd = run.fields[f]
        own = run.design.owner(f) if run.design is not None else None
        c1 = get_row("C1", own or f, "score")
        c2 = get_row("C2", f, "tau")
        c3 = get_row("C3", f, "ratio")
        c4 = get_row("C4", f, "score")
        comp = {}
        for k, x in (("C1", c1), ("C2", c2), ("C3", c3)):
            if _ok(x):
                comp[k] = x["s_hat"]
            elif x is not None:
                listed.append(dict(field=f, measure=k, status=x["status"], reason=x["reason"]))
        in_summary = fd.group not in excl
        table.append(dict(field=f, group=fd.group, in_summary=in_summary,
                          C1=comp.get("C1"), C2_tau=comp.get("C2"), C3=comp.get("C3"),
                          C4=c4["s_hat"] if _ok(c4) else None,
                          field_score=float(np.mean(list(comp.values()))) if comp else None,
                          components=sorted(comp)))
    scored = [t["field_score"] for t in table if t["in_summary"] and t["field_score"] is not None]
    metacf = float(np.mean(scored)) if scored else float("nan")
    return dict(metacf=metacf, n_fields=len(scored), table=table, invalid_or_na=listed,
                excluded_groups=sorted(excl))


def write(run, rows, extra=None):
    v21 = run.cfg.get("v2_1")
    if v21 is not None and bool(v21.get("drop_e4", True)):     # v2.1: E4 dropped (physics/e4_year.py)
        rows = [r for r in rows if r["measure"] != "E4"]
    v23 = run.cfg.get("v2_3")
    if v23 and bool(v23.get("bounded", True)):                 # v2.3: scores bounded to [0, 1], raw kept
        from metacf.v2.core.norm import apply_bounds
        apply_bounds(rows)
    summ = per_field_table(run, rows)
    no_comp = bool(v23) or (v21 is not None and not bool(v21.get("print_composite", True)))
    if no_comp:                                                # never a combined MetaCF number
        summ["metacf"], summ["n_fields"] = None, 0
        for t in summ["table"]:
            t["field_score"] = None
    res = dict(version="metacf-v2.3" if v23 else "metacf-v2", no_composite=no_comp, config_hash=run.hash, protocol_hash=run.phash,
               model=run.adapter.model_id, checkpoint=run.adapter.checkpoint, model_name=run.cfg["model"].get("name"),
               guidance=run.guidance_note(), seed=run.base_seed, bootstrap_B=run.B, bh_q=run.q,
               summary=summ, rows=rows, **(extra or {}))
    for r in rows:
        r.setdefault("config_hash", run.hash)
        r.setdefault("model", run.adapter.model_id)
        r.setdefault("checkpoint", run.adapter.checkpoint)
    if run.persist:
        p = os.path.join(run.mdir, "results.json")
        with open(p, "w") as fh:
            json.dump(res, fh, indent=1, default=_jd)
        card(res, run.mdir)
    return res


def _jd(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def fmt(x, nd=3):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "" if x is None else str(x)
    return "" if not math.isfinite(v) else f"{v:.{nd}f}"


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def card(res, d):
    rows = res["rows"]
    s = res["summary"]
    L = [f"# MetaCF v2 control card: {res.get('model_name')}", "",
         f"config `{res['config_hash']}` · protocol `{res['protocol_hash']}` · model `{res['model']}` · "
         f"checkpoint `{res['checkpoint']}` · guidance {res.get('guidance')} · B={res['bootstrap_B']} · BH q={res['bh_q']}",
         ""] + ([f"**MetaCF = {fmt(s['metacf'])}** over {s['n_fields']} fields "
                 f"(excluding groups {s['excluded_groups']} and INVALID cells; per-field table below).", ""]
                if not res.get("no_composite") else
                ["No combined score: every test is reported separately (see control_card_v23.md).", ""]) + [
         "## Per field", md_table(["field", "group", "in summary", "C1", "C2 tau", "C3", "C4", "field score"],
                                  [[t["field"], t["group"], t["in_summary"], fmt(t["C1"]), fmt(t["C2_tau"]),
                                    fmt(t["C3"]), fmt(t["C4"]), fmt(t["field_score"])] for t in s["table"]])]
    sel = res.get("selection") or {}
    if sel:
        L += ["", "## Feature-space selection (TRAIN pairs)",
              md_table(["field", "space", "valid", "reliability", "CI", "reason"],
                       [[f, v.get("space"), v.get("valid"), fmt(v.get("reliability")),
                         f"[{fmt(v.get('lo'))}, {fmt(v.get('hi'))}]", v.get("reason", "")] for f, v in sel.items()])]
    pr = res.get("pairs_report") or {}
    if pr:
        L += ["", "## Pairs, VIF, joint groups"]
        for split, rep in pr.items():
            L += [f"- {split}: {rep.get('n_pairs')} pairs over {rep.get('n_sites')} sites "
                  f"(dropped for missing values: {rep.get('dropped_missing')}); merges: {rep.get('merges')}"]
            L += ["  - counts above median nonzero |delta|: " +
                  ", ".join(f"{k}: {v['above_median_nonzero']}" for k, v in rep.get("counts", {}).items())]
            if rep.get("vif"):
                L += ["  - VIF: " + ", ".join(f"{k} {fmt(v, 2)}" for k, v in rep["vif"].items())]
    for meas, title in (("C0", "C0 coverage"), ("C1", "C1 counterfactual fidelity"), ("C2", "C2 transfer"),
                        ("C3", "C3 conditional information"), ("C4", "C4 distribution fidelity"),
                        ("C5", "C5 memorisation (check)"), ("E1", "E1 resolution"), ("E2", "E2 season"),
                        ("E3", "E3 sun"), ("E4", "E4 year")):
        mr = [x for x in rows if x["measure"] == meas]
        if not mr:
            continue
        L += ["", f"## {title}", md_table(
            ["field", "metric", "raw", "floor", "ceil", "s_hat", "CI", "p", "p_BH", "N", "space", "status", "notes"],
            [[x["field"], x["metric"], fmt(x["raw"]), fmt(x["floor"]), fmt(x["ceil"]), fmt(x["s_hat"]),
              f"[{fmt(x['ci_lo'])}, {fmt(x['ci_hi'])}]", fmt(x["p"], 4), fmt(x["p_bh"], 4), x["n"], x["space"],
              x["status"] + (" MODIFIED" if x.get("modified") else ""), _notes(x)] for x in mr])]
    bad = [x for x in rows if x["status"] != "OK"]
    L += ["", "## INVALID / N/A cells", md_table(["measure", "field", "metric", "status", "reason"],
                                                 [[x["measure"], x["field"], x["metric"], x["status"], x["reason"]]
                                                  for x in bad]) if bad else "none"]
    txt = "\n".join(L) + "\n"
    with open(os.path.join(d, "control_card.md"), "w") as fh:
        fh.write(txt)
    with open(os.path.join(d, "control_card.html"), "w") as fh:
        fh.write(_html(txt, res))
    return txt


def _notes(x):
    e = x.get("extra", {})
    keys = ("label", "active", "flag", "kappa", "R_gen", "R_real", "delta_gen_days", "delta_real_days",
            "rel_slope_gen", "rel_slope_real", "acc_gen", "acc_real", "gain_real", "gain_gen", "real_source",
            "gen_value", "real_value", "s_real_lo", "min_real", "s_real_within_cat", "s_syn_within_cat",
            "hemisphere_acc_gen", "hemisphere_acc_real")
    return "; ".join(f"{k}={fmt(e[k]) if isinstance(e[k], float) else e[k]}" for k in keys if k in e)


def _html(md, res):
    body = []
    in_t = False
    for line in md.splitlines():
        if line.startswith("|"):
            cells = [html.escape(c.strip()) for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-"):
                continue
            if not in_t:
                body.append("<table>")
                in_t = True
                body.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
            else:
                body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        if in_t:
            body.append("</table>")
            in_t = False
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.strip():
            body.append(f"<p>{html.escape(line)}</p>")
    if in_t:
        body.append("</table>")
    css = ("body{font-family:system-ui,sans-serif;margin:16px;max-width:1200px;background:#fff;color:#111}"
           "table{border-collapse:collapse;margin:8px 0;font-size:13px}td,th{border:1px solid #ccc;padding:3px 6px}"
           "th{background:#f2f2f2}@media (prefers-color-scheme: dark){body{background:#111;color:#eee}"
           "th{background:#222}td,th{border-color:#444}}")
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>MetaCF v2 card</title><style>{css}</style>"
            f"</head><body>{''.join(body)}</body></html>")
