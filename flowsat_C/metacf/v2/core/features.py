"""
Feature spaces (spec 1.1). Three frozen spaces, each a function image -> vector:

  dino   DINOv2 CLS (pooler output), L2-normalised. Spec default DINOv2-L; DINOv2-L is not cached on this
         machine and downloads are forbidden, so the shipped config uses facebook/dinov2-base
         (OPEN_QUESTIONS_V2 Q1; rows carry the model id).
  clip   CLIP ViT-L/14-336 image embedding (projection), L2-normalised.
  stats  the hand-written vector below, z-scored with TRAIN means / stds (StatsNormaliser).

Test-only space:
  plant  block means of the red and blue channels (linear RGB) on a G x G grid. The DummyAdapterV2 world
         writes its planted linear field effects there, so the CPU tests know the true B_f. Never used in a
         real run.

Every stats component is a pure function of a uint8 HxWx3 sRGB image (unit-tested in
metacf/tests/test_v2_features.py).

FeatureStore keeps vectors on disk per space: chunk_XXXXX.npz files of (keys, X float32), appended, never
rewritten, so feature extraction is resumable and shared by every stage and the meta harness.
"""
from __future__ import annotations

import glob
import math
import os
import threading

import numpy as np
from scipy import ndimage

from metacf import physical

# ---------------------------------------------------------------- stats components (pure functions)


def linear_rgb(img) -> np.ndarray:
    """Inverse sRGB, float64 in [0, 1]."""
    return physical.srgb_to_linear(img)


def channel_mean_std(img) -> np.ndarray:
    """[mean R, mean G, mean B, std R, std G, std B] in LINEAR RGB."""
    c = linear_rgb(img).reshape(-1, 3)
    return np.concatenate([c.mean(0), c.std(0)])


def chromaticity(img) -> np.ndarray:
    """Per-pixel (r, g, b) = channel / (R + G + B) in linear RGB; pixels with R+G+B ~ 0 dropped."""
    c = linear_rgb(img).reshape(-1, 3)
    s = c.sum(1)
    ok = s > 1e-6
    return c[ok] / s[ok, None] if ok.any() else np.full((1, 3), 1 / 3)


def chromaticity_means(img) -> np.ndarray:
    return chromaticity(img).mean(0)


def luminance_histogram(img, bins=16) -> np.ndarray:
    """16-bin histogram of linear luminance on [0, 1], normalised to sum 1."""
    L = physical.luminance(img)
    h = np.histogram(np.clip(L, 0, 1), bins=bins, range=(0, 1))[0].astype(float)
    return h / max(h.sum(), 1.0)


def rms_contrast(img) -> float:
    """Standard deviation of linear luminance."""
    return float(physical.luminance(img).std())


def hf_fraction(img, cutoff=0.25) -> float:
    """Hann window, mean removed, power above `cutoff` cycles/px over total power excluding DC."""
    return physical.sharpness_h(img, cutoff)


def acf_length(img) -> float:
    """Radially averaged normalised ACF of luminance; first 1/e crossing, linear interpolation (px).
    Returns the half-size when the ACF never drops below 1/e (flat image)."""
    v = physical.autocorr_scale(img)
    if not np.isfinite(v):
        return float(min(np.asarray(img).shape[:2]) // 2)
    return v


def exg_mean(img) -> float:
    """Chromatic excess green: mean of 2g - r - b on chromaticity."""
    return physical.exg_chromatic(img)


def dark_channel_mean(img, patch=15) -> float:
    """He et al. dark channel (min over RGB, then min filter over a patch x patch window, sRGB values),
    averaged over the image: a haze proxy. The patch is capped at 1/8 of the short side for small images."""
    c = physical.to_float(img)
    p = max(1, min(patch, min(c.shape[:2]) // 8 or 1))
    dc = ndimage.minimum_filter(c.min(-1), size=p)
    return float(dc.mean())


def mean_saturation(img) -> float:
    return float(physical.saturation(img).mean())


STATS_NAMES = (["lin_mean_R", "lin_mean_G", "lin_mean_B", "lin_std_R", "lin_std_G", "lin_std_B",
                "chrom_r", "chrom_g", "chrom_b"] + [f"lum_hist_{i:02d}" for i in range(16)] +
               ["rms_contrast", "hf_fraction", "acf_length", "exg", "dark_channel", "saturation"])


def stats_vector(img) -> np.ndarray:
    """The raw (un-normalised) stats vector, len(STATS_NAMES) = 31."""
    return np.concatenate([channel_mean_std(img), chromaticity_means(img), luminance_histogram(img),
                           [rms_contrast(img), hf_fraction(img), acf_length(img), exg_mean(img),
                            dark_channel_mean(img), mean_saturation(img)]]).astype(np.float64)


class StatsNormaliser:
    """z-score with TRAIN means / stds (fit once on TRAIN real tiles, saved in the run directory)."""

    def __init__(self, mu=None, sd=None, n=0):
        self.mu, self.sd, self.n = mu, sd, n

    def fit(self, X):
        X = np.asarray(X, float)
        self.mu, self.sd, self.n = X.mean(0), X.std(0) + 1e-9, len(X)
        return self

    def __call__(self, X):
        return (np.asarray(X, float) - self.mu) / self.sd

    def to_json(self):
        return dict(mu=self.mu.tolist(), sd=self.sd.tolist(), n=int(self.n), names=list(STATS_NAMES))

    @staticmethod
    def from_json(d):
        return StatsNormaliser(np.array(d["mu"]), np.array(d["sd"]), d["n"])


# ---------------------------------------------------------------- spaces


class Space:
    name = "base"
    model_id = ""
    normalised = True          # L2 rows (dino, clip); stats is z-scored instead
    needs_gpu = False

    def raw(self, images) -> np.ndarray:         # (N, D) un-normalised raw features
        raise NotImplementedError

    def post(self, X, run=None) -> np.ndarray:   # stored raw -> the vector the measures use
        X = np.asarray(X, float)
        if self.normalised:
            return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return X

    def release(self):
        pass


class StatsSpace(Space):
    name = "stats"
    model_id = "handwritten-stats-v2"
    normalised = False

    def __init__(self, threads=4):
        self.threads = threads

    def raw(self, images):
        if self.threads > 1 and len(images) > 4:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(self.threads) as ex:
                return np.stack(list(ex.map(stats_vector, images)))
        return np.stack([stats_vector(im) for im in images])

    def post(self, X, run=None):
        if run is None or run.stats_norm is None:
            raise RuntimeError("stats space used before its TRAIN normaliser was fit")
        return run.stats_norm(X)


class PlantSpace(Space):
    """Test-only: block means of linear R and B on a grid x grid layout (DummyAdapterV2 plant layer)."""
    name = "plant"
    model_id = "plant-test-space"
    normalised = False

    def __init__(self, grid=4):
        self.grid = grid

    def raw(self, images):
        out = []
        g = self.grid
        for im in images:
            c = linear_rgb(im)
            h, w = c.shape[:2]
            ph, pw = h // g, w // g
            b = c[: ph * g, : pw * g].reshape(g, ph, g, pw, 3).mean((1, 3))
            out.append(np.concatenate([b[..., 0].ravel(), b[..., 2].ravel()]))
        return np.stack(out)


class HFSpace(Space):
    """dinov2 / clip through metacf.backbones.HFBackbone (same preprocessing as v1), batch on GPU."""
    needs_gpu = True

    def __init__(self, name, kind, model_id, device="cuda", batch=64):
        from metacf.backbones import HFBackbone
        self.name, self.kind, self.model_id = name, kind, model_id
        self.bb = HFBackbone(kind, model_id, device, batch)

    def raw(self, images):
        return self.bb._features(list(images))

    def release(self):
        if self.bb._model is not None:
            self.bb._model = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass


def make_space(name, spec: dict) -> Space:
    kind = spec["kind"]
    if kind == "stats":
        return StatsSpace(int(spec.get("threads", 4)))
    if kind == "plant":
        return PlantSpace(int(spec.get("grid", 4)))
    if kind in ("dinov2", "clip"):
        return HFSpace(name, kind, spec["model_id"], spec.get("device", "cuda"), int(spec.get("batch", 64)))
    raise ValueError(kind)


# ---------------------------------------------------------------- caption encoders (C3)


class HashTextEncoder:
    """Test-only caption embedding: signed feature hashing of lower-cased words, L2-normalised."""
    model_id = "hash-bow-256"

    def __init__(self, dim=256):
        self.dim = dim

    def __call__(self, captions):
        import hashlib
        out = np.zeros((len(captions), self.dim))
        for i, c in enumerate(captions):
            for w in (c or "").lower().split():
                h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
                out[i, h % self.dim] += 1 if (h >> 20) & 1 else -1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


class ClipTextEncoder:
    """CLIP text embedding (projection), L2-normalised. CLIP's context is 77 tokens, so the 75-85-word
    captions are truncated (OPEN_QUESTIONS_V2 Q9)."""

    def __init__(self, model_id, device="cuda", batch=128):
        self.model_id, self.device, self.batch = model_id, device, batch
        self._m = None

    def __call__(self, captions):
        import torch
        from transformers import CLIPModel, CLIPTokenizer
        if self._m is None:
            self._tok = CLIPTokenizer.from_pretrained(self.model_id)
            dt = torch.float16 if str(self.device).startswith("cuda") else torch.float32   # CPU: fp32 (9 Nov 2026)
            self._m = CLIPModel.from_pretrained(self.model_id, torch_dtype=dt).to(self.device).eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(captions), self.batch):
                t = self._tok([c or "" for c in captions[i:i + self.batch]], padding=True, truncation=True,
                              max_length=77, return_tensors="pt").to(self.device)
                # text tower + projection explicitly (transformers 5.x get_text_features returns an output object)
                o = self._m.text_model(input_ids=t["input_ids"], attention_mask=t.get("attention_mask"))
                outs.append(self._m.text_projection(o.pooler_output).float().cpu().numpy())
        X = np.concatenate(outs) if outs else np.zeros((0, 1))
        return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)

    def release(self):
        self._m = None


def make_text_encoder(spec: dict):
    if spec.get("kind", "clip_text") == "hash":
        return HashTextEncoder(int(spec.get("dim", 256)))
    return ClipTextEncoder(spec["model_id"], spec.get("device", "cuda"), int(spec.get("batch", 128)))


# ---------------------------------------------------------------- feature store


class FeatureStore:
    """key -> raw float32 vector for one space. root=None keeps everything in memory (tests)."""

    def __init__(self, root=None, chunk=4096):
        self.root, self.chunk = root, chunk
        self.mem = {}
        self._pending_k, self._pending_x = [], []
        self._lock = threading.Lock()
        if root:
            os.makedirs(root, exist_ok=True)
            for p in sorted(glob.glob(os.path.join(root, "chunk_*.npz"))):
                z = np.load(p, allow_pickle=False)
                for k, x in zip(z["keys"].tolist(), z["X"]):
                    self.mem[k] = x

    def has(self, k):
        return k in self.mem

    def missing(self, keys):
        return [k for k in dict.fromkeys(keys) if k not in self.mem]

    def put(self, keys, X):
        X = np.asarray(X, np.float32)
        with self._lock:
            for k, x in zip(keys, X):
                self.mem[k] = x
            if self.root:
                self._pending_k += list(keys)
                self._pending_x.append(X)
                if len(self._pending_k) >= self.chunk:
                    self.flush()

    def flush(self):
        if not self.root or not self._pending_k:
            return
        n = len(glob.glob(os.path.join(self.root, "chunk_*.npz")))
        p = os.path.join(self.root, f"chunk_{n:05d}.npz")
        tmp = p + ".tmp.npz"
        np.savez(tmp, keys=np.array(self._pending_k), X=np.concatenate(self._pending_x))
        os.replace(tmp, p)
        self._pending_k, self._pending_x = [], []

    def get(self, keys) -> np.ndarray:
        return np.stack([self.mem[k] for k in keys]).astype(np.float64) if len(keys) else np.zeros((0, 0))


def real_key(rid):
    return "real:" + rid


def gen_key(store_key):
    return "gen:" + store_key


def extract(space: Space, fstore: FeatureStore, keys, loader, batch=64, load_threads=4, log=None):
    """Compute the missing features for `keys`; loader(key) -> uint8 image. Images are loaded with a
    thread pool (PIL decode releases the GIL) and featurised in batches."""
    todo = fstore.missing(keys)
    if not todo:
        return 0
    from concurrent.futures import ThreadPoolExecutor
    n_done = 0
    with ThreadPoolExecutor(max(1, load_threads)) as ex:
        for i in range(0, len(todo), batch):
            ks = todo[i:i + batch]
            imgs = list(ex.map(loader, ks))
            fstore.put(ks, space.raw(imgs))
            n_done += len(ks)
            if log and (i // batch) % 50 == 0:
                log(f"  features {space.name}: {n_done}/{len(todo)}")
    fstore.flush()
    return n_done


def cosine_distance(a, b) -> np.ndarray:
    """Row-wise 1 - cos for L2-normalised rows (re-normalised here for safety)."""
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-12)
    return 1.0 - (a * b).sum(-1)


def euclidean(a, b) -> np.ndarray:
    return np.linalg.norm(np.asarray(a, float) - np.asarray(b, float), axis=-1)


def distance(space_name, a, b):
    """d in a space: cosine distance for the L2 spaces (dino, clip), Euclidean otherwise."""
    return cosine_distance(a, b) if space_name in ("dino", "clip") else euclidean(a, b)


_ = math  # keep import for downstream readers of this module
