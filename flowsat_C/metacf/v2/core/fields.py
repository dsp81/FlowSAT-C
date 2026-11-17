"""
Field schema and regression transforms (spec 1.2), driven by the `tag` of each field in the config:

  gsd       -> log(g')                         (1 column)
  doy       -> (cos 2 pi d / 365.25, sin ...)  (2 columns)
  sun_elev  -> sin(e)                          (1 column)
  sun_azim  -> (cos a, sin a)                  (2 columns; only when the field is enabled in the config)
  cloud     -> fraction                        (1 column)
  year      -> year (float)                    (1 column)
  sensor    -> one-hot, reference = the most frequent TRAIN platform (dropped)  (K-1 columns)

Location fields (geo, covariates: `location: true`) are constant within a site and are never regressors;
they are used by C0 / C3 and as reader inputs.

A FieldV2 also knows how to read its raw value from a record, write a new value into a field dict
(geo writes lat+lon; joint fields write every member), and its reader kind (continuous / circular /
categorical / geo) for C2 / C3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field

import numpy as np

TAG_TRANSFORM = {"gsd": "log", "doy": "cyc365", "sun_elev": "sin_deg", "sun_azim": "cyc360",
                 "cloud": "identity", "year": "identity", "sensor": "onehot"}


@dataclass
class FieldV2:
    name: str
    kind: str                         # continuous | circular | categorical | geo
    keys: list
    group: str = ""
    tag: str = ""
    location: bool = False            # constant within a site (geo, covariates)
    enabled: bool = True
    transform: str = "identity"       # regression / reader transform (from the tag, or the config)
    period: float = 0.0
    classes: list = dc_field(default_factory=list)   # categorical: TRAIN classes by frequency (ref first)
    members: list = dc_field(default_factory=list)   # joint field: member field names

    @property
    def is_joint(self):
        return bool(self.members)

    # ---------------------------------------------------------------- raw values
    def raw(self, f: dict):
        if self.kind == "geo":
            a, b = f.get(self.keys[0]), f.get(self.keys[1])
            return None if a is None or b is None else (float(a), float(b))
        v = f.get(self.keys[0])
        return None if v is None else v

    def set(self, f: dict, v) -> dict:
        out = dict(f)
        if self.kind == "geo":
            out[self.keys[0]], out[self.keys[1]] = float(v[0]), float(v[1])
        elif self.kind == "categorical":
            out[self.keys[0]] = int(v)
        else:
            out[self.keys[0]] = float(v)
        return out

    # ---------------------------------------------------------------- regression columns
    def n_cols(self):
        if self.transform == "onehot":
            return max(len(self.classes) - 1, 0)
        if self.transform in ("cyc365", "cyc360"):
            return 2
        return 1

    def col_names(self):
        if self.transform == "onehot":
            return [f"{self.name}={c}" for c in self.classes[1:]]
        if self.transform in ("cyc365", "cyc360"):
            return [f"cos({self.name})", f"sin({self.name})"]
        return [{"log": "log", "sin_deg": "sin", "identity": ""}.get(self.transform, "") +
                (f"({self.name})" if self.transform not in ("identity",) else self.name)]

    def columns(self, values) -> np.ndarray:
        """raw values (n,) -> (n, n_cols) transformed; NaN rows for missing values."""
        v = np.array([np.nan if x is None else float(x) for x in values], float)
        t = self.transform
        if t == "log":
            return np.log(np.where(v > 0, v, np.nan))[:, None]
        if t == "sin_deg":
            return np.sin(np.radians(v))[:, None]
        if t == "cyc365":
            a = 2 * math.pi * v / 365.25
            return np.column_stack([np.cos(a), np.sin(a)])
        if t == "cyc360":
            a = np.radians(v)
            return np.column_stack([np.cos(a), np.sin(a)])
        if t == "onehot":
            out = np.zeros((len(v), self.n_cols()))
            for i, x in enumerate(v):
                if not np.isfinite(x):
                    out[i] = np.nan
                    continue
                c = int(x)
                if c in self.classes[1:]:
                    out[i, self.classes.index(c) - 1] = 1.0
            return out
        if t == "log1p":
            return np.log1p(np.maximum(v, 0))[:, None]
        return v[:, None]

    # ---------------------------------------------------------------- reader target space
    def target(self, values):
        """Reader target: continuous -> transformed scalar; circular -> radians; categorical -> int;
        geo -> (lat, lon) rows."""
        if self.kind == "geo":
            return np.array([[np.nan, np.nan] if x is None else x for x in values], float)
        v = np.array([np.nan if x is None else float(x) for x in values], float)
        if self.kind == "categorical":
            return np.where(np.isfinite(v), v, -1).astype(int)
        if self.kind == "circular":
            return 2 * math.pi * v / (self.period or 365.25)
        return self.columns(values)[:, 0]


def fields_from_config(cfg: dict) -> dict:
    out = {}
    for name, f in cfg["fields"].items():
        tag = f.get("tag", "")
        kind = f["type"]
        tr = f.get("transform") or TAG_TRANSFORM.get(tag) or ("onehot" if kind == "categorical" else
                                                             "cyc365" if kind == "circular" else "identity")
        out[name] = FieldV2(name=name, kind=kind, keys=list(f["keys"]), group=f.get("group", name), tag=tag,
                            location=bool(f.get("location", False)), enabled=bool(f.get("enabled", True)),
                            transform=tr, period=float(f.get("period", 0) or 0))
    return out


def fit_classes(fields: dict, train_records):
    """Categorical fields: TRAIN classes sorted by frequency (most frequent first = the one-hot
    reference). Classes seen fewer than 1 time never appear; ties broken by class id."""
    for fd in fields.values():
        if fd.kind != "categorical":
            continue
        vals = [fd.raw(r.fields) for r in train_records]
        vals = [int(v) for v in vals if v is not None and np.isfinite(float(v))]
        cls, cnt = np.unique(vals, return_counts=True) if vals else (np.array([]), np.array([]))
        order = sorted(zip(-cnt, cls))
        fd.classes = [int(c) for _, c in order]
    return fields


def regressors(fields: dict):
    """Fields that vary within a site and enter the pair regression (enabled, not location)."""
    return [n for n, f in fields.items() if f.enabled and not f.location]


class Design:
    """The δM design: column blocks per (possibly joint) field."""

    def __init__(self, fields: dict, names=None):
        self.fields = fields
        self.names = list(names or regressors(fields))
        self.blocks, self.cols = {}, []
        i = 0
        for n in self.names:
            k = fields[n].n_cols()
            self.blocks[n] = list(range(i, i + k))
            self.cols += fields[n].col_names()
            i += k
        self.p = i
        self.joint = {}                       # joint name -> member names

    def matrix(self, field_dicts) -> np.ndarray:
        if not self.names:
            return np.zeros((len(field_dicts), 0))
        return np.column_stack([self.fields[n].columns([self.fields[n].raw(f) for f in field_dicts])
                                for n in self.names])

    def merge(self, members, name=None):
        """Join member fields into one joint field (their columns kept together)."""
        name = name or "joint:" + "+".join(members)
        cols = [c for m in members for c in self.blocks.pop(m)]
        self.blocks[name] = sorted(cols)
        self.joint[name] = list(members)
        return name

    def field_names(self):
        """Fields of the regression after merges (joint names replace their members)."""
        return list(self.blocks)

    def members(self, name):
        return self.joint.get(name, [name])

    def owner(self, base_field):
        """The (possibly joint) regression field that holds `base_field`."""
        for n in self.blocks:
            if base_field in self.members(n):
                return n
        return None

    def to_json(self):
        return dict(names=self.names, cols=self.cols, blocks=self.blocks, joint=self.joint,
                    classes={n: self.fields[n].classes for n in self.names if self.fields[n].classes})
