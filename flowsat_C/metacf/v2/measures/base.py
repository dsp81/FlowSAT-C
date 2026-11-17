"""Measure base class and shared selection helpers."""
from __future__ import annotations

import numpy as np

from metacf.config import get
from metacf.fields import region_of, stratified_pick


class Measure:
    id = "X"
    name = "base"

    def __init__(self, run):
        self.run = run
        self.cfg = dict(get(run.cfg, f"measures.{self.id}", {}) or {})

    def c(self, key, default=None):
        return self.cfg.get(key, default)

    def enabled(self):
        return bool(self.c("enabled", True))

    def plan(self) -> list:
        return []

    def compute(self) -> list:
        raise NotImplementedError

    def finalize(self, rows) -> list:
        """Called after Benjamini-Hochberg has filled p_bh on every row (gates that depend on BH)."""
        return rows

    # ---------------------------------------------------------------- helpers
    def seed(self, *parts):
        return self.run.seed(self.id, *parts)


def pick_by_category(records, n, seed):
    return stratified_pick(records, [r.category for r in records], n, seed)


def pick_by_category_region(records, n, seed):
    return stratified_pick(records, [f"{r.category}|{region_of(r.fields.get('lat_deg'), r.fields.get('lon_deg'))}"
                                     for r in records], n, seed)


def gated(row, q):
    """True when the row's BH-adjusted p is significant at q."""
    p = row.get("p_bh", np.nan)
    return bool(np.isfinite(p) and p <= q)
