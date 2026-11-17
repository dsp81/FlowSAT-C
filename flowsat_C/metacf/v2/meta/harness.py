"""
Meta-evaluation harness (spec section 5): `metacf2 meta` runs MetaCF on each wrapper of
configs/metacf_v2_meta.yaml (reusing the protocol artifacts, the feature stores and the unbroken model's
image cache) and writes meta_report.md: expected vs observed per item, PASS / FAIL, and the FID change
when an FID function is available (flowsat_C.scripts.eval_metrics; skipped with a note otherwise).

Expectation item: {measure, field, metric, key (default s_hat), op, value}
  op: "<=", ">=", "<", ">"            compare key against value
      "drops"                          key < the unbroken model's key (minus `margin`, default 0)
      "is" / "is_not"                  equality (labels, booleans), e.g. key extra.label is "blur only"
      "within_baseline_ci"             abs(key - the unbroken model's key) <= half the width of its bootstrap CI
                                       (field "*others*" = every field of that measure / metric except `exclude`)
  key "extra.<name>" reads row["extra"][name]; an INVALID row's s_hat falls back to extra.s_hat_ungated.
"""
from __future__ import annotations

import copy
import json
import math
import os

import numpy as np

from metacf.v2 import runner
from metacf.v2.meta.wrappers import make_wrapper
from metacf.v2.run import Run, log


def _val(row, key):
    if row is None:
        return None
    if key.startswith("extra."):
        return row.get("extra", {}).get(key[6:])
    v = row.get(key)
    if key == "s_hat" and (v is None or not math.isfinite(float(v))):
        v = row.get("extra", {}).get("s_hat_ungated", v)
    return v


def _find(rows, measure, field, metric):
    return next((x for x in rows if x["measure"] == measure and x["field"] == field and x["metric"] == metric), None)


def check(item, rows, base_rows):
    m, f, met = item["measure"], item["field"], item.get("metric", "s_hat")
    key, op, val = item.get("key", "s_hat"), item["op"], item.get("value")
    fields = [f]
    if f == "*others*":
        fields = sorted({x["field"] for x in base_rows if x["measure"] == m and x["metric"] == met}
                        - set(item.get("exclude", [])) - {"*"})
    results = []
    for fld in fields:
        row, brow = _find(rows, m, fld, met), _find(base_rows, m, fld, met)
        obs, ref = _val(row, key), _val(brow, key)
        ok = None
        try:
            if op in ("<=", ">=", "<", ">"):
                o = float(obs)
                ok = math.isfinite(o) and {"<=": o <= val, ">=": o >= val, "<": o < val, ">": o > val}[op]
            elif op == "drops":
                ok = float(obs) < float(ref) - float(item.get("margin", 0.0))
            elif op == "is":
                ok = obs == val
            elif op == "is_not":
                ok = obs != val
            elif op == "within_baseline_ci":
                # |observed - baseline| <= half-width of the baseline's bootstrap CI. (Comparing with [lo, hi]
                # itself fails for identical values: site-resampled cosines run low, so a percentile CI can sit
                # below its own point estimate; see OPEN_QUESTIONS_V2 Q7 / Q33.)
                lo, hi = float(brow["ci_lo"]), float(brow["ci_hi"])
                o, b = float(obs), float(ref)
                ok = bool(math.isfinite(o) and math.isfinite(b) and
                          (not math.isfinite(lo) or abs(o - b) <= (hi - lo) / 2 + 1e-9))
        except (TypeError, ValueError):
            ok = False
        results.append(dict(field=fld, observed=obs, baseline=ref, ok=bool(ok)))
    return dict(item=item, results=results, ok=all(r["ok"] for r in results) and bool(results))


def fid_or_none(run, wrun):
    """FID of the C3/C4 generated set vs TEST reals, for wrapper and unbroken model; None with a note
    when the existing eval code (or a GPU / Inception weights) is not available."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, "no GPU: FID skipped"
        from flowsat_C.scripts.eval_metrics import Extractors, frechet
        from metacf.v2.measures.c2_transfer import test_tiles
        from metacf.v2.measures.c3_information import own_requests
        ex = Extractors("cuda")
        tt = test_tiles(run)

        def feats(imgs):
            u8 = torch.as_tensor(np.stack(imgs)).permute(0, 3, 1, 2)
            return ex.images(u8)["inception"]
        real = feats([run.ds.real_image(x.rid) for x in tt])
        out = {}
        for name, rr in (("base", run), ("wrapper", wrun)):
            st = rr.store()
            out[name] = float(frechet(real, feats([st.get(q) for q in own_requests(rr)])))
        return out, ""
    except Exception as e:                    # noqa: BLE001  (any failure = skip with the reason)
        return None, f"FID skipped: {e!r}"


def run_meta(run, meta_cfg, base_results=None, ids=None, make_store_run=None):
    """run: the unbroken model's Run (adapter set). meta_cfg: dict with `items`."""
    runner.load_all(run)
    if base_results is None:
        p = os.path.join(run.mdir, "results.json")
        base_results = json.load(open(p)) if run.persist and os.path.exists(p) else runner.score(run, ids)
    base_rows = base_results["rows"]
    report = []
    for item in meta_cfg["items"]:
        cfg = copy.deepcopy(run.cfg)
        cfg["model"] = dict(cfg["model"], name=f"{cfg['model'].get('name', 'model')}__{item['name']}",
                            wrapper=item["wrapper"])
        wrapper = make_wrapper(item["wrapper"], run.adapter, run.store(), run)
        wrun = Run(cfg, dataset=run.ds, adapter=wrapper, spaces=run._spaces, text_encoder=run._text,
                   root=run.root, persist=run.persist)
        wrun.design, wrun.pairs, wrun.selection = run.design, run.pairs, run.selection
        wrun.stats_norm = run.stats_norm
        wrun.fstores = run.fstores
        wrun._c["readers"] = run.readers()
        wrun.fields = run.fields
        plan = runner.make_plan(wrun, ids)
        runner.generate(wrun, plan)
        res = runner.score(wrun, ids, plan)
        checks = [check(e, res["rows"], base_rows) for e in item.get("expect", [])]
        fid, note = fid_or_none(run, wrun) if item.get("fid", True) and run.persist else (None, "FID not requested")
        report.append(dict(name=item["name"], wrapper=item["wrapper"], checks=checks,
                           ok=all(c["ok"] for c in checks), fid=fid, fid_note=note,
                           metacf=res["summary"]["metacf"]))
        log(f"meta {item['name']}: {'PASS' if report[-1]['ok'] else 'FAIL'}")
    if run.persist:
        write_report(run, report, base_results)
    return report


def write_report(run, report, base):
    L = ["# MetaCF v2 meta-evaluation", "", f"unbroken model `{base.get('model')}` checkpoint "
         f"`{base.get('checkpoint')}`, MetaCF {base['summary']['metacf']:.3f}", ""]
    for it in report:
        L += [f"## {it['name']}: {'PASS' if it['ok'] else 'FAIL'}", f"wrapper `{it['wrapper']}`; MetaCF "
              f"{it['metacf']:.3f}; FID {it['fid'] if it['fid'] else it['fid_note']}", "",
              "| measure | field | metric | key | op | value | observed | baseline | result |", "|---|---|---|---|---|---|---|---|---|"]
        for c in it["checks"]:
            e = c["item"]
            for r in c["results"]:
                L.append(f"| {e['measure']} | {r['field']} | {e.get('metric')} | {e.get('key', 's_hat')} | {e['op']} | "
                         f"{e.get('value', '')} | {r['observed']} | {r['baseline']} | {'PASS' if r['ok'] else 'FAIL'} |")
        L.append("")
    with open(os.path.join(run.mdir, "meta_report.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(run.mdir, "meta_report.json"), "w") as fh:
        json.dump(report, fh, indent=1, default=str)
