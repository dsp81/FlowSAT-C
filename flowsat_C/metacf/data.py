"""Records, splits and the fMoW manifest reader. Images are uint8 HxWx3 numpy arrays throughout."""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass, field as dc_field

import numpy as np


@dataclass
class Record:
    rid: str                       # unique across splits ("test:" prefix for test records)
    split: str                     # train | test
    category: str
    location_id: str               # unique across splits (test ids are prefixed, see below)
    caption: str
    fields: dict                   # flat: lat_deg, lon_deg, gsd_eff_m, ..., cov.koppen, ...
    image: str = ""                # path to the real image ("" for synthetic records)
    chip_px: int = 0               # short side of the source chip in px (T13 rho stratum)
    meta: dict = dc_field(default_factory=dict)

    def to_json(self):
        return dict(rid=self.rid, split=self.split, category=self.category,
                    location_id=self.location_id, caption=self.caption, fields=self.fields,
                    image=self.image, chip_px=self.chip_px, meta=self.meta)

    @staticmethod
    def from_json(d):
        return Record(**d)


COV_SCALARS = ("elevation_m", "relief_m", "coast_km", "pop_density", "koppen", "landcover_mode")


def fmow_record(row: dict, split: str, resolution: int = 512) -> Record:
    """
    One fMoW manifest row -> Record. Effective GSD follows FMoWCaptionDataset.effective_gsd
    (gsd_file * min(W, H) / res: the chip is centre-cropped to its short side, then resized), so
    the conditioning value matches what the model was trained on. TEST keys collide with TRAIN keys
    but are different places, so test rid / location_id are prefixed with "test:".
    """
    short = min(int(row.get("img_w") or 0), int(row.get("img_h") or 0)) or resolution
    f = dict(lat_deg=row.get("lat_deg"), lon_deg=row.get("lon_deg"),
             gsd_m=row.get("gsd_m"),
             gsd_eff_m=float(row["gsd_m"]) * short / float(resolution) if row.get("gsd_m") else None,
             cloud_frac=row.get("cloud_frac"), doy=row.get("doy"), year=row.get("year"),
             sun_elev_deg=row.get("sun_elev_deg"),
             sun_azimuth_deg=row.get("sun_azimuth_deg") if row.get("azimuth_valid", True) else None,
             photoperiod_h=row.get("photoperiod_h"), platform=row.get("platform"),
             # short side / resolution: gsd_eff = gsd_file * chip_scale. Lets an adapter whose model
             # was conditioned on the FILE gsd (FlowSat v1) convert a swept gsd_eff back
             chip_scale=short / float(resolution))
    cov = row.get("cov") or {}
    for k in COV_SCALARS:
        f[f"cov.{k}"] = cov.get(k)
    if split == "test":
        rid = "test:" + (row.get("test_id") or row["key"])
        loc = "test:" + str(row.get("location_id", rid))
    else:
        rid, loc = row["key"], row["location_id"]
    return Record(rid=rid, split=split, category=row["category"], location_id=loc,
                  caption=row.get("caption", ""), fields=f, image=row.get("image", ""),
                  chip_px=short, meta=dict(key=row["key"], country=row.get("country_code", ""),
                                           cov_valid=bool(cov)))


def read_manifest(path, split, resolution=512, limit=None):
    out = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            if line.strip():
                out.append(fmow_record(json.loads(line), split, resolution))
    return out


def load_real_image(path: str, resolution: int = 512) -> np.ndarray:
    """Centre crop to the short side, then bilinear (antialiased) resize -- the training transform
    (FMoWCaptionDataset._default_transform: T.Resize default bilinear, antialias=True)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path).convert("RGB")
    s = min(im.size)
    left, top = (im.width - s) // 2, (im.height - s) // 2
    im = im.crop((left, top, left + s, top + s)).resize((resolution, resolution), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


class Dataset:
    """Records of both splits + a real-image accessor with a small LRU cache."""

    def __init__(self, records, image_fn=None, resolution=512, cache_size=4096):
        self.records = list(records)
        self.by_id = {r.rid: r for r in self.records}
        assert len(self.by_id) == len(self.records), "duplicate record ids"
        self.resolution = resolution
        self._image_fn = image_fn
        self._cache = OrderedDict()
        self._cache_size = cache_size

    def split(self, name):
        return [r for r in self.records if r.split == name]

    @property
    def train(self):
        return self.split("train")

    @property
    def test(self):
        return self.split("test")

    def real_image(self, rid) -> np.ndarray:
        if rid in self._cache:
            self._cache.move_to_end(rid)
            return self._cache[rid]
        r = self.by_id[rid]
        im = self._image_fn(r) if self._image_fn else load_real_image(r.image, self.resolution)
        self._cache[rid] = im
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return im

    def save(self, path):
        with open(path, "w") as fh:
            for r in self.records:
                fh.write(json.dumps(r.to_json()) + "\n")

    @staticmethod
    def load(path, resolution=512):
        with open(path) as fh:
            return Dataset([Record.from_json(json.loads(l)) for l in fh if l.strip()],
                           resolution=resolution)
