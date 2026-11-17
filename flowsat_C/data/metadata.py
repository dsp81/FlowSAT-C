"""
sat_data_util.py — FlowSAT-C metadata utilities for the fMoW jpg release.

LAYOUT
  images   <FMOW_ROOT>/{train,val}/{cat}/{cat}_{loc}/{cat}_{loc}_{img}_rgb.jpg
  tier 1   same stem + .json, 39 keys, ALL splits: gsd, cloud_cover, timestamp,
           img_width/img_height, utm, country_code, scan_direction,
           approximate_wavelengths, bounding_boxes (PIXEL coords), sun/target
           azimuth, sun elevation, off-nadir, pan/multi resolution -- angular
           fields carry _dbl/_start/_end/_min/_max variants.
  tier 2   fmow-full/groundtruth.tar.bz2, 45 keys, adds the six tier 1 lacks:
           raw_location (WKT POLYGON, take centroid), epsg (4326),
           spatial_reference, catalog_id, sensor_platform_name, abs_cal_factors.
  captions <FMOW_CAPTIONS>/train/{cat}/{loc}/{stem}.txt

JOIN KEY for all four: relative path minus extension, RELATIVE TO THE SPLIT DIR
  e.g. "airport/airport_0/airport_0_0_rgb"
Rooting it at the dataset root instead of the split dir makes every caption
lookup silently miss and you train captionless.

COUNT: train = 363,572 rgb images / 62 categories / 83,412 locations
(4.36 views per location). The older 257,248 figure came from fmow-full/train
(.tif release) whose extraction stopped partway through recreational_facility --
19 categories (road_bridge..zoo) have JSONs and zero images, and caption sets
built on that copy contain no examples of those classes.

TIER 2 IS THE ONLY SOURCE OF LAT/LON ANYWHERE. It is absent from the sidecars,
and the fmow-full .tif files are GeoTIFF containers with georeferencing stripped
(CRS None, no geotransform/GCPs/RPCs). One streaming pass writes an
id,lon,lat,platform CSV for the meta_csv_path slot -- extracting 363k files is
wasteful.

RULE R1: this module emits PHYSICAL UNITS ONLY (degrees, metres, day-of-year,
fraction). No metadata_normalize(). v1's units bug, its Fourier aliasing and its
zero-fill dropout bug were three symptoms of one root cause: normalised floats
and physical floats share a type, so nothing in a signature distinguished them.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import math
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RGB_SUFFIX = "_rgb.jpg"     # NOT ".jpg": excludes the quarter-res _msrgb sibling
COV_KEYS = ("elevation_m", "relief_m", "coast_km", "pop_density",
            "koppen", "landcover_mode", "landcover_hist")

# WorldView / GeoEye / QuickBird family present in fMoW. Index 0 is reserved for
# unknown so a missing platform lands on the null path, not on a real sensor.
SENSOR_PLATFORMS = ["UNKNOWN", "WORLDVIEW01", "WORLDVIEW02", "WORLDVIEW03",
                    "WORLDVIEW03_VNIR", "WORLDVIEW04", "GEOEYE01", "QUICKBIRD02",
                    "IKONOS", "SPOT", "PLEIADES"]
PLATFORM_INDEX = {p: i for i, p in enumerate(SENSOR_PLATFORMS)}


def platform_index(name: str | None) -> int:
    if not name:
        return 0
    return PLATFORM_INDEX.get(str(name).strip().upper().replace("-", "_"), 0)


# --------------------------------------------------------------------------- keys


def join_key(image_path: str | Path, split_dir: str | Path) -> str:
    return str(Path(image_path).relative_to(Path(split_dir)).with_suffix(""))


def parse_stem(key: str) -> tuple[str, str, int]:
    """
    'airport/airport_0/airport_0_12_rgb' -> ('airport', 'airport_0', 12)

    Image ids are NEITHER zero-based NOR contiguous: 13,831 locations do not
    start at _0, 26,817 have gaps. Never synthesise ids with range().
    """
    cat, loc, stem = key.split("/")
    m = re.match(rf"^{re.escape(loc)}_(\d+)_rgb$", stem)
    if not m:
        raise ValueError(f"unparsable stem: {key}")
    return cat, loc, int(m.group(1))


# ----------------------------------------------------------- tier 2: lat/lon CSV


_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def wkt_polygon_centroid(wkt: str) -> tuple[float, float]:
    """
    Area-weighted (shoelace) centroid of a WKT POLYGON in EPSG:4326, (lon lat)
    order. Not the vertex mean: fMoW footprints are quadrilaterals with a
    duplicated closing vertex, so a vertex mean biases toward that corner.
    """
    pts = np.asarray([float(x) for x in _NUM.findall(wkt)],
                     dtype=np.float64).reshape(-1, 2)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    x, y = pts[:, 0], pts[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-12:
        return float(x.mean()), float(y.mean())
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6.0 * a)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6.0 * a)
    return float(cx), float(cy)


def build_latlon_csv(tar_path: str, out_csv: str, split: str = "train") -> int:
    """
    Stream groundtruth.tar.bz2 -> CSV: id,lon,lat,platform

    Drops into the existing meta_csv_path slot. The train member set is verified
    to match all 363,572 train ids exactly, zero missing on either side, so any
    shortfall in the written count means the wrong split was scanned.
    """
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(tar_path, "r:bz2") as tf, open(out_csv, "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["id", "lon", "lat", "platform"])
        for member in tf:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            parts = Path(member.name).parts
            if split not in parts:
                continue
            key = str(Path(*parts[parts.index(split) + 1:]).with_suffix(""))
            if not key.endswith("_rgb"):
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            try:
                m = json.load(f)
            except Exception:
                continue
            wkt = m.get("raw_location")
            if not wkt:
                continue
            lon, lat = wkt_polygon_centroid(wkt)
            w.writerow([key, f"{lon:.6f}", f"{lat:.6f}",
                        m.get("sensor_platform_name", "")])
            n += 1
            if n % 25000 == 0:
                print(f"  {n}", flush=True)
    print(f"wrote {n} rows -> {out_csv}")
    return n


def load_latlon_csv(path: str) -> dict[str, tuple[float, float, str]]:
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["id"]] = (float(r["lon"]), float(r["lat"]), r.get("platform", ""))
    return out


# ------------------------------------------------------------------- tier 1 read


def _num(m: dict, base: str) -> float | None:
    """
    Angular fields carry _dbl/_start/_end/_min/_max variants. Prefer the scalar,
    then _dbl, then the midpoint of _min/_max, then the midpoint of _start/_end.
    """
    for k in (base, base + "_dbl"):
        v = m.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    for a, b in ((base + "_min", base + "_max"), (base + "_start", base + "_end")):
        va, vb = m.get(a), m.get(b)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            return 0.5 * (float(va) + float(vb))
    return None


def day_of_year(ts: str) -> float:
    """
    Day-of-MONTH is dropped -- the 15th vs the 21st carries no appearance
    information. Day-of-YEAR is the seasonal phase variable and it is continuous,
    so early March and late March are distinguishable.
    """
    s = ts.replace("Z", "").strip().split(".")[0]
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return float(_dt.datetime.strptime(s, fmt).timetuple().tm_yday)
        except ValueError:
            continue
    return float(_dt.datetime.fromisoformat(s).timetuple().tm_yday)


def normalise_cloud(raw) -> float:
    """fMoW documents cloud_cover as a percentage. v1 multiplied by 1000 as if it
    were a 0..1 fraction -> values to 100,000, a 100x error. Audit the histogram
    once and hardcode the branch rather than leaving this heuristic."""
    v = float(raw or 0.0)
    return v / 100.0 if v > 1.0 else v


def derived_photoperiod(lat_deg: float, doy: float) -> float:
    """
    Photoperiod has no measured counterpart in the sidecar, so it stays derived.
    delta(N) is an INTERMEDIATE ONLY and must never become a feature: it is
    exactly the first day-of-year harmonic rescaled.
    """
    phi = math.radians(lat_deg)
    delta = math.radians(23.44 * math.sin(2 * math.pi * (doy - 81.0) / 365.0))
    c = max(-1.0, min(1.0, -math.tan(phi) * math.tan(delta)))   # polar day/night
    return 2.0 * math.degrees(math.acos(c)) / 15.0


def derived_sun_elevation(lat_deg: float, doy: float) -> float:
    """Fallback only. The sidecar's measured value accounts for time of day and
    is strictly better; use this when the field is absent."""
    delta = 23.44 * math.sin(2 * math.pi * (doy - 81.0) / 365.0)
    return 90.0 - abs(lat_deg - delta)


@dataclass
class RawMetadata:
    key: str
    category: str
    location_id: str          # first-class: MetaCF samples counterfactual pairs by site
    view_id: int
    lat_deg: float
    lon_deg: float
    gsd_m: float
    cloud_frac: float
    doy: float
    year: int
    sun_elev_deg: float
    sun_azimuth_deg: float    # 0..360, CYCLIC -- encode with sin/cos
    photoperiod_h: float
    platform: int
    img_w: int
    img_h: int
    country_code: str = ""
    sun_elev_measured: bool = True
    azimuth_valid: bool = True


def parse_image_json(json_path, key: str, latlon: dict, strict: bool = True):
    """
    strict=True raises on a missing coordinate instead of falling back to (0,0).

    (0,0) is a REAL location in the Gulf of Guinea. A silent fallback would place
    thousands of images at one equatorial ocean point, poisoning the SH basis,
    the solar terms and every raster lookup, with no error ever raised. Coverage
    is verified exact, so a miss means the lookup was built wrong -- fail loudly.
    """
    with open(json_path) as f:
        m = json.load(f)
    rec = latlon.get(key)
    if rec is None:
        if strict:
            raise KeyError(
                f"no lat/lon for {key}. groundtruth coverage is verified exact at "
                f"363,572 train ids, so this means the CSV was built from the "
                f"wrong split or with a mismatched join key.")
        return None
    lon, lat, plat = rec

    cat, loc, vid = parse_stem(key)
    ts = m.get("timestamp") or m.get("acquisition_date")
    doy = day_of_year(ts)

    se = _num(m, "sun_elevation")
    az = _num(m, "sun_azimuth")
    return RawMetadata(
        key=key, category=cat, location_id=f"{cat}/{loc}", view_id=vid,
        lon_deg=lon, lat_deg=lat,
        gsd_m=float(m.get("gsd", 0.0)),
        cloud_frac=normalise_cloud(m.get("cloud_cover", 0.0)),
        doy=doy, year=int(str(ts)[:4]),
        sun_elev_deg=se if se is not None else derived_sun_elevation(lat, doy),
        sun_azimuth_deg=az if az is not None else 0.0,
        photoperiod_h=derived_photoperiod(lat, doy),
        platform=platform_index(plat),
        img_w=int(m.get("img_width", 0)), img_h=int(m.get("img_height", 0)),
        country_code=str(m.get("country_code", "")),
        sun_elev_measured=se is not None, azimuth_valid=az is not None)


# ------------------------------------------------------------------------ collate


def collate_metadata(items: list[RawMetadata], covariates: list[dict]) -> dict:
    import torch

    def col(n, dt=torch.float32):
        return torch.tensor([getattr(i, n) for i in items], dtype=dt)

    cov = {}
    for k in COV_KEYS:
        if k == "landcover_hist":
            cov[k] = torch.tensor(np.stack([c[k] for c in covariates]),
                                  dtype=torch.float32)
        elif k in ("koppen", "landcover_mode"):
            cov[k] = torch.tensor([c[k] for c in covariates], dtype=torch.long)
        else:
            cov[k] = torch.tensor([float(c[k]) for c in covariates],
                                  dtype=torch.float32)

    return dict(
        lat_deg=col("lat_deg"), lon_deg=col("lon_deg"), gsd_m=col("gsd_m"),
        cloud_frac=col("cloud_frac"), doy=col("doy"),
        year=col("year", torch.long),
        sun_elev_deg=col("sun_elev_deg"), sun_azimuth_deg=col("sun_azimuth_deg"),
        photoperiod_h=col("photoperiod_h"),
        azimuth_valid=col("azimuth_valid", torch.bool),
        platform=col("platform", torch.long), cov=cov)


def log_uniform_sweep(lo: float, hi: float, n: int = 5) -> list[float]:
    """
    Multiplicative fields MUST be swept log-uniformly. v1 used
    {3.0,2.0,1.0,0.6,0.3} -> log2 gaps {0.58,1.00,0.74,1.00}: uneven, so any
    monotonicity statistic partly measures the sampling grid. Applies to GSD,
    elevation, coast distance, population. NOT day-of-year (linear, cyclic),
    NOT cloud (linear), NOT azimuth (cyclic).
    """
    r = (hi / lo) ** (1.0 / (n - 1))
    return [lo * r ** i for i in range(n)]
