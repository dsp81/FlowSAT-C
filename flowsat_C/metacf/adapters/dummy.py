"""
DummyAdapter (spec section 8): procedural images whose statistics depend on the fields in planted
ways, plus the matching "real world" used as TRAIN/TEST reals in the CPU tests.

World (linear RGB, then sRGB-encoded):
    base            [0.25, 0.30, 0.25]
    a     -> red    + 0.35 a            b       -> blue + 0.35 b
    binary-> green  + 0.25 binary       strong  -> brightness x (0.35 + 0.65 strong)
    ignored -> amplitude of a seeded luminance texture: 0.02 + 0.20 ignored
    doy, lat, landcover -> green + amp(landcover) cos(2 pi (doy - 180 - 182.6 [lat < 0]) / 365.25)
    seed  -> the texture pattern (seed-to-seed variability)

Plants (adapter deviations from the world), each switchable:
    ignore_field    the field has no effect (texture amplitude fixed at 0.10)
    leak            a also adds 0.35 a to BLUE (leaks into b's read-back)
    binary_step     green responds with a step at 0.5 instead of linearly
    hemisphere_days the southern-hemisphere seasonal offset the model uses (world: 182.6)
    pathways        P1 = a constant domain offset only (ZERO removes it -> strong cast; MEAN and
                    SHUFFLE are identical to FULL); P2 = carries every field (ZERO -> 0, MEAN ->
                    TRAIN mean, SHUFFLE -> donor)
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from PIL import Image
from scipy import ndimage

from metacf.adapters.base import ModelAdapter
from metacf.data import Record

FIELDS = ("strong", "ignored", "a", "b", "binary", "doy", "lat_deg", "lon_deg", "landcover")
NULL_VALUE = {"strong": 0.5, "ignored": 0.5, "a": 0.5, "b": 0.5, "binary": 0.5, "doy": 180.0,
              "lat_deg": 0.0, "lon_deg": 0.0, "landcover": 0}
SEASON_AMP = {0: 0.12, 1: 0.12, 2: 0.005}      # landcover 0, 1 = high seasonality, 2 = desert / built
WORLD = dict(ignore_field=None, leak=False, binary_step=False, hemisphere_days=182.6)


def _seed_int(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _texture(seed, size):
    rng = np.random.default_rng(seed)
    n = ndimage.gaussian_filter(rng.normal(size=(size, size)), 1.5)
    return n / (n.std() + 1e-9)


def render(fields: dict, seed: int, size: int = 64, plants: dict | None = None,
           p1_cast: bool = False) -> np.ndarray:
    p = dict(WORLD, **(plants or {}))
    f = dict(NULL_VALUE, **{k: v for k, v in fields.items() if k in NULL_VALUE and v is not None})
    c = np.array([0.25, 0.30, 0.25])
    c = c + np.array([0.35 * f["a"], 0.0, 0.35 * f["b"]])
    if p["leak"]:
        c = c + np.array([0.0, 0.0, 0.35 * f["a"]])
    c[1] += 0.25 * (float(f["binary"] > 0.5) if p["binary_step"] else f["binary"])
    off = p["hemisphere_days"] if f["lat_deg"] < 0 else 0.0
    c[1] += SEASON_AMP.get(int(f["landcover"]), 0.0) * math.cos(
        2 * math.pi * (f["doy"] - 180.0 - off) / 365.25)
    c = c * (0.35 + 0.65 * f["strong"])
    amp = 0.10 if p["ignore_field"] == "ignored" else 0.02 + 0.20 * f["ignored"]
    img = np.broadcast_to(c, (size, size, 3)).copy()
    img = img + amp * 0.25 * _texture(seed, size)[..., None]
    if p1_cast:                                   # P1 removed: the constant domain offset is gone
        img = img * np.array([1.4, 0.6, 1.4])
    img = np.clip(img, 0, 1) ** (1 / 2.2)
    return (img * 255).round().astype(np.uint8)


class DummyAdapter(ModelAdapter):
    name = "dummy"

    def __init__(self, size=64, plants=None, pathways=False, **_):
        self.size = size
        self.plants = dict(plants or {})
        self.model_id = "dummy"
        self.checkpoint = hashlib.sha1(repr(sorted(self.plants.items())).encode()).hexdigest()[:8]
        self.supported_fields = {"strong", "ignored", "a", "b", "binary", "doy", "latlon", "landcover"}
        self.field_groups = {"strong": "strong", "ignored": "ignored", "a": "a", "b": "b",
                             "binary": "binary", "doy": "doy", "latlon": "geo",
                             "landcover": "landcover"}
        self.group_keys = {"strong": ["strong"], "ignored": ["ignored"], "a": ["a"], "b": ["b"],
                           "binary": ["binary"], "doy": ["doy"], "geo": ["lat_deg", "lon_deg"],
                           "landcover": ["landcover"]}
        self.pathways = ["P1", "P2"] if pathways else []
        self._means = None

    def _nulled(self, fields, null_mask):
        f = dict(fields)
        for g, on in (null_mask or {}).items():
            if on:
                for k in self.group_keys.get(g, []):
                    f[k] = NULL_VALUE[k]
        return f

    def generate(self, caption, fields, null_mask, seed, guidance=None):
        f = self._nulled(fields, null_mask)
        return Image.fromarray(render(f, seed, self.size, self.plants))

    LINEAR = ("strong", "ignored", "a", "b", "binary")

    def planted_signal(self, field, values):
        """The planted response a calibrated reader reads back: the set value itself for the linear
        fields, the 0 / 1 step under `binary_step`, a constant under `ignore_field`."""
        if field not in self.LINEAR:
            return None
        v = np.asarray(values, float)
        if self.plants.get("ignore_field") == field:
            return np.full_like(v, NULL_VALUE[field])
        if field == "binary" and self.plants.get("binary_step"):
            return (v > 0.5).astype(float)
        return v

    def prepare_pathway_means(self, train_fields):
        keys = [k for k in NULL_VALUE if k != "landcover"]
        self._means = {k: float(np.mean([t[k] for t in train_fields if t.get(k) is not None]))
                       for k in keys}

    def generate_with_pathway(self, caption, fields, null_mask, seed, pathway, mode,
                              donor_fields=None, groups=None):
        f = self._nulled(fields, null_mask)
        if pathway == "P1":
            # P1 carries no per-record information: only ZERO (removing its constant) changes the image
            return Image.fromarray(render(f, seed, self.size, self.plants, p1_cast=(mode == "zero")))
        g = dict(f)
        keys = [k for k in FIELDS if k not in ("landcover",)]
        if groups:
            keys = [k for grp in groups for k in self.group_keys.get(grp, [])]
        for k in keys:
            if mode == "zero":
                g[k] = 0.0 if k not in ("doy",) else 1.0
            elif mode == "mean":
                g[k] = (self._means or NULL_VALUE)[k]
            elif mode == "shuffle":
                g[k] = donor_fields[k]
        return Image.fromarray(render(g, seed, self.size, self.plants))


# ---------------------------------------------------------------- the planted "real world"


def world_records(n_train_loc=120, n_test_loc=60, views=4, seed=0, categories=("c0", "c1", "c2"),
                  south_frac=0.5):
    """Records for both splits. Location-constant: category, lat/lon, landcover. Per view:
    strong, ignored, a, b, binary, doy (so natural experiments exist at every location)."""
    rng = np.random.default_rng(seed)
    recs = []
    for split, n_loc in (("train", n_train_loc), ("test", n_test_loc)):
        for li in range(n_loc):
            cat = categories[li % len(categories)]
            south = rng.random() < south_frac
            lat = float(rng.uniform(25, 55)) * (-1 if south else 1)
            lon = float(rng.uniform(-170, 170))
            lc = int(rng.integers(0, 3))
            loc = f"{split}:loc{li}"
            for v in range(views):
                f = dict(strong=float(rng.random()), ignored=float(rng.random()), a=float(rng.random()),
                         b=float(rng.random()), binary=float(rng.random()),
                         doy=float(rng.uniform(1, 365)), lat_deg=lat, lon_deg=lon, landcover=lc)
                f["category_idx"] = categories.index(cat)
                recs.append(Record(rid=f"{loc}:v{v}", split=split, category=cat, location_id=loc,
                                   caption=f"a {cat} scene", fields=f))
    return recs


def world_image_fn(size=64):
    def fn(rec):
        return render(rec.fields, _seed_int("real", rec.rid), size, WORLD)
    return fn
