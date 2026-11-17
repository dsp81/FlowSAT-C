"""
Adapters for MetaCF v2.

DummyAdapterV2 + world_records_v2 (CPU tests, spec section 7): a procedural world whose features follow a
KNOWN linear model of the fields, plus physical channels for E1-E3.

  Image (linear RGB, size x size, then sRGB):
    plant layer   R and B on a 4 x 4 block grid = base 0.45 + site vector + sum_f B_f t_f(m) + view noise.
                  t_f: log(g'/0.85), (cos, sin) of doy, sin(e) - 0.7, cloud - 0.3, (year - 2010) / 8,
                  one-hot sensor (classes 1, 2; reference 0). The PlantSpace reads exactly these block means,
                  so B_f (32-d) is the true effect of field f.
    site vector   a fixed linear map of [category one-hot, sin lat, xyz, elevation] (geo / cov are visible)
                  + a small per-site random part.
    green         (0.35 + 0.08 (sin e - 0.7) + site) x season, season = 1 + amp(landcover) cos(2 pi (d - phi_h) / 365.25),
                  phi_N = 196, phi_S = 196 + 182.625, amp: high-seasonality covers 0.15, low 0.01.
    texture       zero-mean luminance texture: a per-layout noise field cropped to a window proportional to
                  g' and resized (zoom: ACF length ~ 1 / g'), plus white sensor noise ~ 1 / g' (so the HF
                  fraction FALLS with g' in this world; see OPEN_QUESTIONS_V2 Q21).
  Real tiles use view noise / texture seeded by the record id; the adapter seeds them with (rid, seed).

  Behaviour per field (DummyAdapterV2(behaviour={field: kind})):
    faithful (default) | scaled (plant effect x 0.5, kappa = 0.5) | leaky:<j> (adds 0.5 B_j t_f) |
    ignored (no effect anywhere: plant, zoom, season) | blur (BlurForZoom transform) | tint (TintForSeason).

FlowSatCBatchAdapter: the v1 FlowSatCAdapter with batched generation (requests with the same null mask
and guidance are sampled together; noise is per-seed, so images do not depend on the batch).
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from PIL import Image
from scipy import ndimage

from metacf.adapters.base import ModelAdapter
from metacf.data import Record
from metacf.v2.meta.wrappers import blur_for_zoom, tint_for_season

G = 4
D_PLANT = 2 * G * G
CATS = ("c0", "c1", "c2")
NULL = {"gsd_eff_m": 0.85, "doy": 196.0, "sun_elev_deg": 45.0, "cloud_frac": 0.3, "year": 2010.0,
        "platform": 0, "lat_deg": 0.0, "lon_deg": 0.0, "cov.elevation_m": 300.0, "cov.landcover_mode": 0}
GROUP_KEYS = {"gsd": ["gsd_eff_m"], "doy": ["doy"], "solar": ["sun_elev_deg"], "cloud": ["cloud_frac"],
              "year": ["year"], "sensor": ["platform"], "geo": ["lat_deg", "lon_deg"],
              "cov": ["cov.elevation_m", "cov.landcover_mode"]}
SEASON_AMP = {0: 0.15, 1: 0.01}
G_MED = 0.85
NOISE, NOISE_POW, LAYOUT_SIGMA = 0.016, 1.0, 2.5   # sensor noise ~ (0.3 / g')^pow; layout smoothness


def _sint(*parts):
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _unit(seed, d=D_PLANT):
    v = np.random.default_rng(seed).normal(size=d)
    return v / np.linalg.norm(v)


AMP = dict(gsd=0.06, doy_c=0.05, doy_s=0.05, sun=0.15, cloud=0.15, year=0.05, s1=0.06, s2=0.06)
TRUE_B = {k: a * _unit(_sint("B", k)) for k, a in AMP.items()}
TRUE_B["sun"] = 0.15 * (0.6 * np.abs(_unit(_sint("B", "sun"))) + 0.4 * _unit(_sint("B", "sun2")))
SITE_M = np.random.default_rng(_sint("sitem")).normal(size=(D_PLANT, 8)) * 0.03


def t_cols(f):
    g = float(f["gsd_eff_m"])
    d = 2 * math.pi * float(f["doy"]) / 365.25
    return dict(gsd=math.log(g / G_MED), doy_c=math.cos(d), doy_s=math.sin(d),
                sun=math.sin(math.radians(float(f["sun_elev_deg"]))) - 0.7, cloud=float(f["cloud_frac"]) - 0.3,
                year=(float(f["year"]) - 2010.0) / 8.0, s1=float(int(f["platform"]) == 1),
                s2=float(int(f["platform"]) == 2))


COL_FIELD = dict(gsd="gsd", doy_c="doy", doy_s="doy", sun="sun_elev", cloud="cloud", year="year", s1="sensor",
                 s2="sensor")
FIELD_COLS = {}
for _c, _f in COL_FIELD.items():
    FIELD_COLS.setdefault(_f, []).append(_c)


def site_vec(f, cat, loc):
    la, lo = math.radians(float(f["lat_deg"])), math.radians(float(f["lon_deg"]))
    x = np.array([cat == "c0", cat == "c1", cat == "c2", math.sin(la), math.cos(la) * math.cos(lo),
                  math.cos(la) * math.sin(lo), math.sin(la), (float(f["cov.elevation_m"]) - 300) / 300], float)
    return SITE_M @ x + 0.01 * np.random.default_rng(_sint("site", loc)).normal(size=D_PLANT)


def _layout(seed, n=192):
    z = ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=(n, n)), LAYOUT_SIGMA)
    return z / (z.std() + 1e-9)


def render(f, cat, loc, noise_seed, size=64, behaviour=None):
    b = behaviour or {}
    f = dict(NULL, **{k: v for k, v in f.items() if k in NULL and v is not None})
    t = t_cols(f)
    plant = 0.45 + site_vec(f, cat, loc)
    for col, val in t.items():
        fld = COL_FIELD[col]
        kind = b.get(fld, "faithful")
        if kind == "ignored":
            continue
        scale = 0.5 if kind == "scaled" else 1.0
        plant = plant + scale * val * TRUE_B[col]
        if kind.startswith("leaky:"):
            j = kind.split(":", 1)[1]
            plant = plant + 0.5 * val * TRUE_B[FIELD_COLS[j][0]] * (np.linalg.norm(TRUE_B[col]) /
                                                                     np.linalg.norm(TRUE_B[FIELD_COLS[j][0]]))
    rng = np.random.default_rng(noise_seed)
    plant = plant + 0.03 * rng.normal(size=D_PLANT)
    R = np.kron(plant[:G * G].reshape(G, G), np.ones((size // G, size // G)))
    Bc = np.kron(plant[G * G:].reshape(G, G), np.ones((size // G, size // G)))
    sun_t = 0.0 if b.get("sun_elev") == "ignored" else t["sun"]
    doy = NULL["doy"] if b.get("doy") == "ignored" else float(f["doy"])
    lat = float(f["lat_deg"])
    phi = 196.0 + (182.625 if lat < 0 else 0.0)
    amp = SEASON_AMP.get(int(f["cov.landcover_mode"]), 0.0)
    season = 1 + amp * math.cos(2 * math.pi * (doy - phi) / 365.25)
    site_g = 0.02 * math.sin(_sint("g", loc) % 1000)
    Gc = np.full((size, size), (0.35 + 0.08 * sun_t + site_g) * season)
    g_eff = G_MED if b.get("gsd") == "ignored" else float(f["gsd_eff_m"])
    lay = _layout(_sint("layout", noise_seed))
    w = int(np.clip(round(24 * g_eff / 0.3), 8, lay.shape[0]))
    c0 = (lay.shape[0] - w) // 2
    tex = ndimage.zoom(lay[c0:c0 + w, c0:c0 + w], size / w, order=1)[:size, :size]
    tex = (tex - tex.mean()) / (tex.std() + 1e-9)
    noise = rng.normal(size=(size, size)) * NOISE * (0.3 / g_eff) ** NOISE_POW
    T = 0.04 * tex + noise
    img = np.stack([R + T, Gc + T, Bc + T], -1)
    img = np.clip(img, 0, 1)
    srgb = np.where(img <= 0.0031308, 12.92 * img, 1.055 * np.power(img, 1 / 2.4) - 0.055)
    return np.clip(np.round(srgb * 255), 0, 255).astype(np.uint8)


def world_records_v2(n_train_loc=150, n_test_loc=300, views=5, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for split, n_loc in (("train", n_train_loc), ("test", n_test_loc)):
        for li in range(n_loc):
            cat = CATS[li % 3]
            south = rng.random() < 0.5
            lat = float(rng.uniform(25, 58)) * (-1 if south else 1)
            lon = float(rng.uniform(-170, 170))
            elev = float(rng.uniform(0, 900))
            lc = int(rng.random() < 0.5)
            loc = f"{split}:loc{li}"
            for v in range(views):
                doy = float(rng.uniform(1, 365))
                # sun elevation independent of doy (no solar geometry in this world): the CPU tests run C1 with
                # consistent_doy false; the CONSISTENT path is unit-tested separately (Q22)
                f = dict(lat_deg=lat, lon_deg=lon, gsd_eff_m=float(np.exp(rng.uniform(np.log(0.3), np.log(2.4)))),
                         doy=doy, sun_elev_deg=float(rng.uniform(15, 75)),
                         cloud_frac=float(rng.uniform(0, 0.6)), year=float(rng.integers(2002, 2018)),
                         platform=int(rng.choice(3, p=[0.5, 0.3, 0.2])), photoperiod_h=12.0,
                         **{"cov.elevation_m": elev, "cov.landcover_mode": lc})
                f["category_idx"] = CATS.index(cat)
                recs.append(Record(rid=f"{loc}:v{v}", split=split, category=cat, location_id=loc,
                                   caption=f"a {cat} scene", fields=f, chip_px=512 if v % 2 else 400))
    return recs


def world_image_fn(size=64):
    def fn(rec):
        return render(rec.fields, rec.category, rec.location_id, _sint("real", rec.rid), size)
    return fn


class DummyAdapterV2(ModelAdapter):
    name = "dummy2"

    def __init__(self, dataset, size=64, behaviour=None, fixed_doy=196.0, **_):
        self.ds = dataset
        self.size = size
        self.behaviour = dict(behaviour or {})
        self.fixed_doy = fixed_doy
        self.model_id = "dummy2"
        self.checkpoint = hashlib.sha1(repr(sorted(self.behaviour.items())).encode()).hexdigest()[:8]
        self.supported_fields = {"gsd", "doy", "sun_elev", "cloud", "year", "sensor", "latlon",
                                 "elevation", "landcover"}

    def generate(self, caption, fields, null_mask, seed, guidance=None, rid=None):
        f = dict(fields)
        for g, on in (null_mask or {}).items():
            if on:
                for k in GROUP_KEYS.get(g, []):
                    f[k] = NULL[k]
        rec = self.ds.by_id[rid]
        b = dict(self.behaviour)
        g_req, d_req = float(f["gsd_eff_m"]), float(f["doy"])
        if b.get("gsd") == "blur":
            f["gsd_eff_m"] = G_MED
            b.pop("gsd")
        if b.get("doy") == "tint":
            f["doy"] = self.fixed_doy
            b.pop("doy")
        img = render(f, rec.category, rec.location_id, _sint("gen", rid, seed), self.size, b)
        if self.behaviour.get("gsd") == "blur":
            img = blur_for_zoom(img, g_req, G_MED)
        if self.behaviour.get("doy") == "tint":
            img = tint_for_season(img, d_req)
        return Image.fromarray(img)

    def generate_request(self, req, donor_fields=None):
        null = {g: True for g in req.null_mask}
        return self.generate(req.caption, req.field_dict, null, req.seed, req.guidance, rid=req.rid)


class FlowSatCBatchAdapter:
    """Wraps metacf.adapters.flowsat_c.FlowSatCAdapter (constructed lazily: GPU only) with generate_batch."""

    def __init__(self, mcfg, device="cuda"):
        from metacf.adapters.flowsat_c import FlowSatCAdapter
        self.inner = FlowSatCAdapter(mcfg, device)
        self.name, self.model_id, self.checkpoint = self.inner.name, self.inner.model_id, self.inner.checkpoint
        self.supported_fields = self.inner.supported_fields
        self.field_groups = self.inner.field_groups
        self.batch = int(mcfg.get("batch", 8))
        self.peak_mem_gb = None

    def generate_request(self, req, donor_fields=None):
        return self.inner.generate_request(req, donor_fields)

    def generate_batch(self, reqs):
        """-> list of uint8 arrays, same order. All reqs must share null_mask and guidance."""
        import torch
        from metacf.adapters.flowsat_c import _EncoderProxy
        nm = list(reqs[0].null_mask)
        g = reqs[0].guidance
        m = self.inner._model_ready()
        with torch.no_grad(), _EncoderProxy.install(m, null_groups=nm):
            out = self.inner._sample([q.caption for q in reqs], [q.field_dict for q in reqs],
                                     [q.seed for q in reqs], g)
        if torch.cuda.is_available():
            self.peak_mem_gb = torch.cuda.max_memory_allocated() / 2 ** 30
        return out

    def release(self):
        self.inner._model = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
