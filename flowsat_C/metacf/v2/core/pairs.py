"""
Real same-site pairs (spec 1.3).

  - For a split, every unordered pair (I_a, I_b), a != b, at the same location_id; at most `cap` (10)
    pairs per location (seeded random) so large sites do not dominate. Unordered is enough: the pair
    regression has no intercept, so flipping a pair flips both sides.
  - Pairs with a missing value in any regressor are dropped (counted in the report).
  - delta_m = transform(m_b) - transform(m_a) for every regressor column, plus the raw values.
  - Report per field: pairs with |delta| above the field's median nonzero |delta| (block norm; for a
    categorical field: pairs where the class changes, OPEN_QUESTIONS_V2 Q4), and the VIF of each column
    computed against the columns of the OTHER fields (within-field collinearity, e.g. cos/sin of doy, is
    irrelevant because a block is always kept together; Q5).
  - Joint groups: fields with any column VIF > vif_max (5) are merged into one joint field (all such
    fields together; Q5). Every merge is logged.
  - Optional `max_sites` caps the number of sites (seeded) to bound feature extraction (Q6).
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from metacf.fields import seed_of


@dataclass
class PairSet:
    split: str
    a: list                        # rid of I_a
    b: list                        # rid of I_b
    site: list                     # location_id
    dM: np.ndarray                 # (n, p) transformed differences
    raw_a: dict = dc_field(default_factory=dict)       # field -> raw values of a
    raw_b: dict = dc_field(default_factory=dict)
    report: dict = dc_field(default_factory=dict)

    @property
    def n(self):
        return len(self.a)

    def sites(self):
        return sorted(set(self.site))

    def site_index(self):
        """-> (sites list, int array pair -> site position)."""
        s = self.sites()
        pos = {x: i for i, x in enumerate(s)}
        return s, np.array([pos[x] for x in self.site], int)

    def subset(self, idx):
        idx = np.asarray(idx, int)
        return PairSet(self.split, [self.a[i] for i in idx], [self.b[i] for i in idx],
                       [self.site[i] for i in idx], self.dM[idx],
                       {k: [v[i] for i in idx] for k, v in self.raw_a.items()},
                       {k: [v[i] for i in idx] for k, v in self.raw_b.items()}, dict(self.report))

    def to_json(self):
        return dict(split=self.split, a=self.a, b=self.b, site=self.site, dM=self.dM.tolist(),
                    raw_a=self.raw_a, raw_b=self.raw_b, report=self.report)

    @staticmethod
    def from_json(d):
        return PairSet(d["split"], d["a"], d["b"], d["site"], np.asarray(d["dM"], float).reshape(len(d["a"]), -1),
                       d["raw_a"], d["raw_b"], d["report"])


def enumerate_pairs(records, cap=10, seed=0, max_sites=None):
    """-> list of (rid_a, rid_b, location_id)."""
    by = {}
    for r in records:
        by.setdefault(r.location_id, []).append(r)
    locs = sorted(k for k, v in by.items() if len(v) >= 2)
    if max_sites and len(locs) > max_sites:
        rng = np.random.default_rng(seed_of(seed, "pair_sites"))
        locs = sorted(rng.choice(locs, max_sites, replace=False).tolist())
    out = []
    for loc in locs:
        rs = sorted(by[loc], key=lambda r: r.rid)
        cand = [(rs[i].rid, rs[j].rid) for i in range(len(rs)) for j in range(i + 1, len(rs))]
        if len(cand) > cap:
            rng = np.random.default_rng(seed_of(seed, "pairs", loc))
            sel = sorted(rng.choice(len(cand), cap, replace=False).tolist())
            cand = [cand[k] for k in sel]
        out += [(a, b, loc) for a, b in cand]
    return out


def vif_other(dM, design) -> dict:
    """VIF of each column regressed (with intercept) on the columns of the other fields."""
    out = {}
    for name, cols in design.blocks.items():
        others = [c for n2, cs in design.blocks.items() if n2 != name for c in cs]
        for c in cols:
            y = dM[:, c]
            if np.std(y) < 1e-12 or not others:
                out[design.cols[c]] = float("nan") if np.std(y) < 1e-12 else 1.0
                continue
            X = np.column_stack([np.ones(len(y)), dM[:, others]])
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ beta
            r2 = 1 - (res @ res) / (((y - y.mean()) ** 2).sum())
            out[design.cols[c]] = float(1.0 / max(1e-12, 1 - r2))
    return out


def block_abs(dM, cols, kind):
    """|delta| of a field per pair: block norm (chord for circular); categorical: 1 if the class changed."""
    d = np.linalg.norm(dM[:, cols], axis=1) if cols else np.zeros(len(dM))
    if kind == "categorical":
        return (d > 1e-9).astype(float)
    return d


def above_median_nonzero(dabs, kind):
    """Mask of pairs with |delta| above the median nonzero |delta| (categorical: any change)."""
    nz = dabs[dabs > 1e-12]
    if len(nz) == 0:
        return np.zeros(len(dabs), bool)
    if kind == "categorical":
        return dabs > 0.5
    return dabs > np.median(nz)


def build_pairs(records, design, split, cap=10, seed=0, max_sites=None, vif_max=5.0, merge=True, log=None):
    """Enumerate pairs, compute delta_m, the report and (when merge) the joint groups on `design`."""
    by_id = {r.rid: r for r in records}
    pr = enumerate_pairs(records, cap, seed, max_sites)
    if not pr:
        return PairSet(split, [], [], [], np.zeros((0, design.p)), report=dict(n_pairs=0))
    A = design.matrix([by_id[a].fields for a, _, _ in pr])
    B = design.matrix([by_id[b].fields for _, b, _ in pr])
    dM = B - A
    ok = np.isfinite(dM).all(1)
    n_drop = int((~ok).sum())
    pr = [p for p, k in zip(pr, ok) if k]
    dM = dM[ok]
    for n, cols in design.blocks.items():                 # circular columns are differences of cos/sin
        pass
    raw_a = {n: [design.fields[n].raw(by_id[a].fields) for a, _, _ in pr] for n in design.names}
    raw_b = {n: [design.fields[n].raw(by_id[b].fields) for _, b, _ in pr] for n in design.names}
    ps = PairSet(split, [p[0] for p in pr], [p[1] for p in pr], [p[2] for p in pr], dM, raw_a, raw_b)
    vif = vif_other(dM, design) if len(dM) > design.p + 2 else {}
    merges = []
    if merge and vif:
        hot = [n for n, cols in design.blocks.items()
               if any(np.isfinite(vif.get(design.cols[c], np.nan)) and vif[design.cols[c]] > vif_max for c in cols)]
        if len(hot) >= 2:
            name = design.merge(hot)
            merges.append(dict(joint=name, members=hot,
                               vif={design.cols[c]: vif[design.cols[c]] for n in hot
                                    for c in design.blocks[name] if design.cols[c] in vif}))
            if log:
                log(f"VIF merge: {hot} -> {name} (VIF > {vif_max})")
    counts = {}
    for n, cols in design.blocks.items():
        kind = "categorical" if (not design.fields.get(n) is None and design.fields[n].kind == "categorical") else "x"
        dabs = block_abs(dM, cols, kind)
        counts[n] = dict(nonzero=int((dabs > 1e-12).sum()), above_median_nonzero=int(above_median_nonzero(dabs, kind).sum()))
    ps.report = dict(n_pairs=len(pr), n_sites=len(set(ps.site)), dropped_missing=n_drop, cap=cap,
                     max_sites=max_sites, counts=counts, vif=vif, vif_max=vif_max, merges=merges,
                     blocks=design.blocks, cols=design.cols)
    return ps


def field_kind(design, name):
    f = design.fields.get(name)
    return f.kind if f is not None else "joint"
