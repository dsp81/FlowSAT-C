"""
Feature-space selection (spec 1.5), on TRAIN pairs only.

For each regression field (joint groups included) choose the space with the highest split-half
reliability of B_f. The field is VALID for C1 / C4 if that reliability >= min_reliability (0.5) and its
bootstrap CI excludes 0 (lo > 0); otherwise INVALID. The table is written to <run>/selection.json and
every downstream measure reads it; nothing re-selects.
"""
from __future__ import annotations

import json
import os

import numpy as np


def select_spaces(reliability: dict, min_rel=0.5):
    """reliability[space][field] = dict(mean, lo, hi, ...) -> {field: dict(space, valid, reason, table)}."""
    fields = sorted({f for sp in reliability.values() for f in sp})
    out = {}
    for f in fields:
        table = {sp: reliability[sp][f] for sp in reliability if f in reliability[sp]}
        best = None
        for sp, r in sorted(table.items()):
            m = r.get("mean", float("nan"))
            if np.isfinite(m) and (best is None or m > table[best]["mean"]):
                best = sp
        if best is None:
            out[f] = dict(space=None, valid=False, reason="no finite reliability in any space",
                          table=_t(table))
            continue
        r = table[best]
        valid = bool(r["mean"] >= min_rel and np.isfinite(r.get("lo", np.nan)) and r["lo"] > 0)
        reason = "" if valid else (f"best reliability {r['mean']:.3f} in {best} < {min_rel}"
                                   if r["mean"] < min_rel else f"CI [{r.get('lo')}, {r.get('hi')}] includes 0")
        out[f] = dict(space=best, valid=valid, reliability=r["mean"], lo=r.get("lo"), hi=r.get("hi"),
                      reason=reason, table=_t(table))
    return out


def _t(table):
    return {sp: {k: v for k, v in r.items() if k != "splits"} for sp, r in table.items()}


def save(path, selection, meta):
    with open(path, "w") as fh:
        json.dump(dict(selection=selection, **meta), fh, indent=1, default=float)


def load(run_dir):
    p = os.path.join(run_dir, "selection.json")
    with open(p) as fh:
        return json.load(fh)
