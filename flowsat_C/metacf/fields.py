"""
Field specs, value grids, support, anchors, derangements, condition neighbourhoods and natural
experiments (spec section 2). Everything here is fit on TRAIN; TEST records are only selected.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field as dc_field

import numpy as np

from metacf import stats


@dataclass
class FieldSpec:
    name: str
    kind: str                        # continuous | circular | categorical | geo
    keys: list                       # record field key(s); geo = [lat_key, lon_key]
    tags: list = dc_field(default_factory=list)
    transform: str = "none"          # none | log | log1p   (continuous only)
    period: float = 0.0              # circular only (365.25 for doy, 360 for azimuth)
    n_classes: int = 0               # categorical only
    group: str = ""                  # model field group (FlowSAT-C GROUPS) for null masks / T9
    derived: bool = False            # recomputed in CONSISTENT mode, never swept on its own
    recovery_only: bool = False      # scored by recovery tests only (T1, T5b, T6a side effects); never swept

    # ---- value <-> model-space (the space readers predict in) ----
    def raw(self, rec_fields: dict):
        if self.kind == "geo":
            return (rec_fields.get(self.keys[0]), rec_fields.get(self.keys[1]))
        return rec_fields.get(self.keys[0])

    def encode(self, v):
        """Raw value -> transformed space. Vectorised over arrays."""
        if self.kind == "geo":
            return np.asarray(v, float)
        if self.kind == "categorical":
            return np.asarray(v, float)
        v = np.asarray(v, float)
        if self.kind == "circular":
            return 2 * math.pi * v / self.period
        if self.transform == "log":
            return np.log(np.maximum(v, 1e-12))
        if self.transform == "log1p":
            return np.log1p(np.maximum(v, 0))
        return v

    def decode(self, z):
        z = np.asarray(z, float)
        if self.kind == "circular":
            return stats.wrap_pos(z) * self.period / (2 * math.pi)
        if self.transform == "log":
            return np.exp(z)
        if self.transform == "log1p":
            return np.expm1(z)
        return z

    def set(self, rec_fields: dict, v) -> dict:
        """Copy of rec_fields with this field set to v (geo: v = (lat, lon))."""
        out = dict(rec_fields)
        if self.kind == "geo":
            out[self.keys[0]], out[self.keys[1]] = float(v[0]), float(v[1])
        elif self.kind == "categorical":
            out[self.keys[0]] = int(v)
        else:
            out[self.keys[0]] = float(v)
        return out

    def diff(self, a, b):
        """Transformed difference a - b; circular wrapped to (-pi, pi]."""
        d = np.asarray(self.encode(a), float) - np.asarray(self.encode(b), float)
        return stats.wrap(d) if self.kind == "circular" else d


def specs_from_config(cfg: dict) -> dict:
    out = {}
    for name, f in cfg["fields"].items():
        out[name] = FieldSpec(name=name, kind=f["type"], keys=list(f["keys"]),
                              tags=list(f.get("tags", [])), transform=f.get("transform", "none"),
                              period=float(f.get("period", 0) or 0),
                              n_classes=int(f.get("n_classes", 0) or 0),
                              group=f.get("group", ""), derived=bool(f.get("derived", False)),
                              recovery_only=bool(f.get("recovery_only", False)))
    return out


# ---------------------------------------------------------------- deterministic seeding


def seed_of(*parts) -> int:
    """Stable 31-bit seed from any parts (config seed, test id, field ...)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


# ---------------------------------------------------------------- grids and support


def value_grid(spec: FieldSpec, train_values, percentiles=(10, 30, 50, 70, 90)):
    """K values at TRAIN percentiles (in raw units). Circular fields: percentiles of the
    distribution unwrapped around its circular mean. Categorical: the most frequent K classes."""
    v = np.asarray(train_values, float)
    v = v[np.isfinite(v)]
    if spec.kind == "categorical":
        cls, cnt = np.unique(v.astype(int), return_counts=True)
        k = len(percentiles)
        return [int(c) for c in cls[np.argsort(-cnt)][:k]]
    if spec.kind == "circular":
        z = spec.encode(v)
        mu = stats.circ_mean(z)
        u = stats.wrap(z - mu)
        q = np.percentile(u, percentiles) + mu
        return [float(x) for x in spec.decode(q)]
    if spec.transform in ("log", "log1p"):
        q = np.percentile(spec.encode(v), percentiles)
        return [float(x) for x in spec.decode(q)]
    return [float(x) for x in np.percentile(v, percentiles)]


def log_spaced_grid(lo, hi, K):
    return [float(x) for x in np.exp(np.linspace(math.log(lo), math.log(hi), K))]


class Support:
    """
    Support map: a (category, value) pair is in support when the value lies inside the
    [lo_pct, hi_pct] percentile range of that category's TRAIN values for the field (falls back to
    the global range for categories with < min_cat records). Categorical: classes seen >= min_count
    times in the category.
    """

    def __init__(self, spec, train_values, train_categories, lo_pct=1, hi_pct=99, min_cat=50,
                 min_count=5):
        self.spec = spec
        v = np.asarray(train_values, float)
        c = np.asarray(train_categories)
        ok = np.isfinite(v)
        v, c = v[ok], c[ok]
        self.global_range = self._range(v, lo_pct, hi_pct)
        self.by_cat = {}
        for cat in np.unique(c):
            vv = v[c == cat]
            if spec.kind == "categorical":
                cls, cnt = np.unique(vv.astype(int), return_counts=True)
                self.by_cat[cat] = set(cls[cnt >= min_count].tolist())
            elif len(vv) >= min_cat:
                self.by_cat[cat] = self._range(vv, lo_pct, hi_pct)
        if spec.kind == "categorical":
            cls, cnt = np.unique(v.astype(int), return_counts=True)
            self.global_set = set(cls[cnt >= min_count].tolist())

    def _range(self, v, lo, hi):
        if self.spec.kind == "circular":
            z = self.spec.encode(v)
            mu = stats.circ_mean(z)
            u = stats.wrap(z - mu)
            return ("circ", mu, float(np.percentile(u, lo)), float(np.percentile(u, hi)))
        if self.spec.kind == "categorical":
            return None
        return ("lin", float(np.percentile(v, lo)), float(np.percentile(v, hi)))

    def contains(self, category, value) -> bool:
        if self.spec.kind == "categorical":
            return int(value) in self.by_cat.get(category, self.global_set)
        r = self.by_cat.get(category, self.global_range)
        if r[0] == "circ":
            u = float(stats.wrap(self.spec.encode(value) - r[1]))
            return r[2] <= u <= r[3]
        return r[1] <= float(value) <= r[2]

    def clip_grid(self, grid):
        """Grid values inside the GLOBAL support (for tests that clip a fixed grid)."""
        return [g for g in grid if self.contains(None, g)]


# ---------------------------------------------------------------- record selection


def region_of(lat, lon, lat_step=30.0, lon_step=60.0) -> str:
    if lat is None or lon is None or not np.isfinite(lat) or not np.isfinite(lon):
        return "unknown"
    return f"{int(math.floor((lat + 90) / lat_step))}_{int(math.floor((lon + 180) / lon_step))}"


def stratified_pick(items, strata, n, seed):
    """Round-robin over shuffled strata so every stratum is represented before any repeats."""
    rng = np.random.default_rng(seed)
    groups = {}
    for it, s in zip(items, strata):
        groups.setdefault(s, []).append(it)
    keys = sorted(groups)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(groups[k])
    out, i = [], 0
    while len(out) < n and any(groups[k] for k in keys):
        k = keys[i % len(keys)]
        if groups[k]:
            out.append(groups[k].pop())
        i += 1
    return out


def select_anchors(test_records, spec, grid, support, n, seed, extra_filter=None):
    """
    Anchors: TEST records, stratified by category x region, whose category supports EVERY grid
    value (so a full sweep stays in support). Returns record ids.
    """
    elig, strata = [], []
    for r in test_records:
        if extra_filter is not None and not extra_filter(r):
            continue
        if spec.raw(r.fields) is None:
            continue
        if not all(support.contains(r.category, g) for g in grid):
            continue
        elig.append(r.rid)
        strata.append(f"{r.category}|{region_of(r.fields.get('lat_deg'), r.fields.get('lon_deg'))}")
    return stratified_pick(elig, strata, n, seed)


def derangement(categories, seed) -> np.ndarray:
    """
    Permutation with no fixed points, stratified by category: Sattolo's algorithm (a single
    cycle, hence no fixed points) inside each category. Records of single-record categories cannot
    stay in their category: two or more of them form their own cycle; a lone one is spliced into
    another record's cycle (that one record then gets a cross-category donor).
    """
    rng = np.random.default_rng(seed)
    cats = np.asarray(categories)
    n = len(cats)
    perm = np.arange(n)
    singles = []
    for c in sorted(set(cats.tolist())):
        idx = np.where(cats == c)[0]
        if len(idx) == 1:
            singles.append(int(idx[0]))
            continue
        order = idx.copy()
        for i in range(len(order) - 1, 0, -1):          # Sattolo
            j = int(rng.integers(0, i))
            order[i], order[j] = order[j], order[i]
        # cycle: position order[i] -> order[i+1]
        for i in range(len(order)):
            perm[order[i]] = order[(i + 1) % len(order)]
    if len(singles) >= 2:
        order = np.array(singles)
        for i in range(len(order) - 1, 0, -1):
            j = int(rng.integers(0, i))
            order[i], order[j] = order[j], order[i]
        for i in range(len(order)):
            perm[order[i]] = order[(i + 1) % len(order)]
    elif singles and n > 1:
        s = singles[0]
        j = int(rng.choice([k for k in range(n) if k != s]))
        perm[s], perm[j] = perm[j], s
    assert n < 2 or not (perm == np.arange(n)).any(), "derangement has a fixed point"
    return perm


# ---------------------------------------------------------------- neighbourhoods


def field_matrix(records, specs, names):
    """z-scorable matrix of continuous / circular (cos, sin) / geo (xyz) field values."""
    cols = []
    for nm in names:
        s = specs[nm]
        if s.kind == "categorical":
            continue
        if s.kind == "geo":
            lat = np.radians([r.fields.get(s.keys[0], np.nan) for r in records])
            lon = np.radians([r.fields.get(s.keys[1], np.nan) for r in records])
            cols += [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        elif s.kind == "circular":
            z = s.encode([r.fields.get(s.keys[0], np.nan) for r in records])
            cols += [np.cos(z), np.sin(z)]
        else:
            cols.append(s.encode([r.fields.get(s.keys[0], np.nan) for r in records]))
    return np.column_stack(cols) if cols else np.zeros((len(records), 0))


class Neighbourhoods:
    """kNN (k=8) in z-scored field space within the same category, excluding the same
    location_id. The index is TRAIN; queries are any records."""

    def __init__(self, train_records, specs, names, k=8):
        self.k, self.specs, self.names = k, specs, names
        self.train = train_records
        X = field_matrix(train_records, specs, names)
        self.mu, self.sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
        self.X = np.nan_to_num((X - self.mu) / self.sd)
        self.by_cat = {}
        for i, r in enumerate(train_records):
            self.by_cat.setdefault(r.category, []).append(i)

    def query(self, rec):
        idx = np.array(self.by_cat.get(rec.category, []), int)
        if len(idx) == 0:
            return []
        idx = idx[[self.train[i].location_id != rec.location_id for i in idx]]
        q = np.nan_to_num((field_matrix([rec], self.specs, self.names)[0] - self.mu) / self.sd)
        d = ((self.X[idx] - q) ** 2).sum(1)
        return [self.train[i].rid for i in idx[np.argsort(d)[:self.k]]]


# ---------------------------------------------------------------- natural experiments


def location_groups(records):
    g = {}
    for r in records:
        g.setdefault(r.location_id, []).append(r)
    return {k: v for k, v in g.items() if len(v) >= 2}


def natural_pairs(records, spec, max_pairs_per_location=50, seed=0):
    """Pairs (rid_i, rid_j) at the same location_id whose value of `spec` differs."""
    rng = np.random.default_rng(seed)
    out = []
    for loc, rs in sorted(location_groups(records).items()):
        cand = []
        for a in range(len(rs)):
            for b in range(a + 1, len(rs)):
                va, vb = spec.raw(rs[a].fields), spec.raw(rs[b].fields)
                if va is None or vb is None:
                    continue
                if spec.kind == "geo":
                    continue                           # location is constant by construction
                if np.any(np.asarray(spec.diff(vb, va)) != 0):
                    cand.append((rs[a].rid, rs[b].rid))
        if len(cand) > max_pairs_per_location:
            sel = rng.choice(len(cand), max_pairs_per_location, replace=False)
            cand = [cand[i] for i in sorted(sel)]
        out += cand
    return out


def natural_sequences(records, spec, min_distinct=3):
    """Per location: records sorted by the field's value, kept when >= min_distinct values."""
    out = []
    for loc, rs in sorted(location_groups(records).items()):
        vals = [spec.raw(r.fields) for r in rs]
        if any(v is None for v in vals):
            continue
        if len(set(np.round(np.asarray(vals, float), 6).tolist())) >= min_distinct:
            out.append([r.rid for r in rs])
    return out
