"""MetaCFTest base class and the result-row schema (P1 floor/ceiling, P5 N/A, P6 statistics)."""
from __future__ import annotations

import numpy as np

from metacf import metrics
from metacf.config import test_cfg


def row(test, field, reader="", stratum="all", raw=float("nan"), floor=float("nan"),
        ceiling=float("nan"), ci=None, n=0, seeds=0, validation="VALID", support="SUPPORTED",
        p=float("nan"), modified=False, notes="MEASURED", descriptive=False, ci_on="norm", **extra):
    """One cell of results.json. norm is computed here (never clipped; out_of_range flags it).
    ci_on says which value the CI is for: "norm" (default) or "raw"."""
    if descriptive:
        norm, oor = float("nan"), False
    else:
        norm, oor = metrics.normalise(raw, floor, ceiling)
    ci = ci or {}
    return dict(test=test, field=field, reader=reader, stratum=stratum, raw=_f(raw), floor=_f(floor),
                ceiling=_f(ceiling), norm=_f(norm), out_of_range=oor, ci_lo=_f(ci.get("lo")),
                ci_hi=_f(ci.get("hi")), n=int(n), seeds=int(seeds), validation=validation,
                support=support, p=_f(p), p_bh=float("nan"), modified=bool(modified), notes=notes,
                descriptive=bool(descriptive), ci_on="raw" if descriptive else ci_on, extra=extra)


def na(test, field, reason, reader="", stratum="all", status="N/A"):
    """P5: unsupported / missing / under-powered cells are N/A with a reason, never zero."""
    r = row(test, field, reader, stratum, validation=status,
            support="N/A" if status == "N/A" else "SUPPORTED", notes=reason)
    r["status"] = status
    r["reason"] = reason
    return r


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


class MetaCFTest:
    id = "T0"
    name = "base"
    tier = 1
    required_tags: set = set()
    min_n = 0
    descriptive = False
    modified = False

    def __init__(self, cfg):
        self.cfg = cfg
        self.tc = test_cfg(cfg, self.id)
        self.B = int(cfg.get("bootstrap", {}).get("B", 1000))

    # ---- interface ----
    def fields(self, ctx):
        """Fields this test scores (tag gating for Tier 2)."""
        if not self.required_tags:
            return ctx.eval_fields()
        return [f for f in ctx.eval_fields() if self.required_tags & set(ctx.specs[f].tags)]

    def plan(self, ctx) -> list:
        return []

    def validate_real(self, ctx) -> dict:
        """-> {field: dict(status=VALID|INVALID|FALLBACK|N/A, detail=...)}"""
        return {}

    def compute(self, ctx, store) -> list:
        raise NotImplementedError

    # ---- helpers ----
    def boot_seed(self, ctx, *parts):
        return ctx.seed("boot", self.id, *parts)

    def na_unsupported(self, ctx):
        """N/A rows for config fields the model does not condition on."""
        out = []
        for f, s in ctx.specs.items():
            if s.derived or ctx.supported(f):
                continue
            if self.required_tags and not (self.required_tags & set(s.tags)):
                continue
            out.append(na(self.id, f, f"model {ctx.adapter.name} does not condition on {f}"))
        return out

    def swept(self, ctx):
        """This test's fields that get sweeps (recovery_only fields excluded)."""
        return [f for f in self.fields(ctx) if not ctx.specs[f].recovery_only]

    def na_recovery_only(self, ctx):
        return [na(self.id, f, "recovery-only by design (scored in T1, T5b and as a T6a side effect; "
                               "no sweeps)") for f in self.fields(ctx) if ctx.specs[f].recovery_only]

    def missing_tag_na(self, ctx):
        if self.required_tags and not self.fields(ctx):
            return [na(self.id, "-", f"no supported field carries tag(s) {sorted(self.required_tags)}")]
        return []


def nanmean(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")
