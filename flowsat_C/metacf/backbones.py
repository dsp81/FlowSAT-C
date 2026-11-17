"""
Frozen feature backbones for readers (P3) and for the feature map phi (section 2).

HFBackbone reuses eval_metrics.Extractors' preprocessing (Extractors._clip_norm for CLIP, the same
224 px ImageNet normalisation for DINOv2) but loads the model id named in the config, because the
spec's Reader B default (DINOv2-L) is not the DINOv2-B that Extractors hard-codes. GPU-only in
practice; never constructed by the CPU tests.

HandcraftedBackbone is a deterministic CPU feature map used by the unit tests (and usable as a
cheap sanity reader); it is NOT a substitute for the frozen backbones in a real run.
"""
from __future__ import annotations

import hashlib

import numpy as np

from metacf import physical


def image_key(img: np.ndarray) -> str:
    a = np.ascontiguousarray(img)
    return hashlib.sha1(a.tobytes() + str(a.shape).encode()).hexdigest()


class Backbone:
    name = "base"

    def __init__(self):
        self._cache = {}
        self._pcache = {}

    def _features(self, images):              # -> (N, D)
        raise NotImplementedError

    def _patch_features(self, images):        # -> (N, gh, gw, D)
        raise NotImplementedError

    def features(self, images, normalise=True):
        keys = [image_key(im) for im in images]
        todo = [i for i, k in enumerate(keys) if k not in self._cache]
        if todo:
            F = self._features([images[i] for i in todo])
            for i, f in zip(todo, F):
                self._cache[keys[i]] = np.asarray(f, np.float64)
        X = np.stack([self._cache[k] for k in keys]) if keys else np.zeros((0, 1))
        if normalise:
            X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return X

    def patch_features(self, images):
        keys = [image_key(im) for im in images]
        todo = [i for i, k in enumerate(keys) if k not in self._pcache]
        if todo:
            F = self._patch_features([images[i] for i in todo])
            for i, f in zip(todo, F):
                self._pcache[keys[i]] = np.asarray(f, np.float64)
        return np.stack([self._pcache[k] for k in keys])


class HandcraftedBackbone(Backbone):
    """
    variant "color":   per-channel mean/std of linear RGB, chromaticity, luminance histogram,
                       ExG, saturation stats, shadow fraction
    variant "texture": 8x8 block-mean luminance grid, sharpness h, ACF scale, gradient energy,
                       per-quadrant luminance
    variant "phi":     both, concatenated (the test-time phi)
    Patch features (for T17): a grid x grid map of [mean linear RGB, luminance std].
    """

    def __init__(self, variant="phi", grid=8):
        super().__init__()
        self.variant, self.grid = variant, grid
        self.name = f"handcrafted-{variant}"

    def _color(self, im):
        c = physical.srgb_to_linear(im)
        s = c.sum(-1, keepdims=True) + 1e-6
        chrom = (c / s).reshape(-1, 3)
        L = physical.luminance(im)
        hist = np.histogram(L, bins=8, range=(0, 1))[0] / L.size
        S = physical.saturation(im)
        return np.concatenate([c.reshape(-1, 3).mean(0), c.reshape(-1, 3).std(0), chrom.mean(0),
                               hist, [physical.exg_chromatic(im), S.mean(), S.std(),
                                      physical.shadow_fraction(im), L.mean(), L.std()]])

    def _texture(self, im):
        L = physical.luminance(im)
        h, w = L.shape
        g = 8
        blocks = L[: h // g * g, : w // g * g].reshape(g, h // g, g, w // g).mean((1, 3)).ravel()
        gy, gx = np.gradient(L)
        quad = [L[:h // 2, :w // 2].mean(), L[:h // 2, w // 2:].mean(), L[h // 2:, :w // 2].mean(),
                L[h // 2:, w // 2:].mean()]
        ell = physical.autocorr_scale(im)
        return np.concatenate([blocks, [physical.sharpness_h(im), np.log1p(ell if np.isfinite(ell) else 0),
                                        np.sqrt((gx ** 2 + gy ** 2).mean())], quad])

    def _features(self, images):
        out = []
        for im in images:
            if self.variant == "color":
                out.append(self._color(im))
            elif self.variant == "texture":
                out.append(self._texture(im))
            else:
                out.append(np.concatenate([self._color(im), self._texture(im)]))
        return np.stack(out)

    def _patch_features(self, images):
        out = []
        g = self.grid
        for im in images:
            c = physical.srgb_to_linear(im)
            L = physical.luminance(im)
            h, w = L.shape
            ph, pw = h // g, w // g
            cm = c[: ph * g, : pw * g].reshape(g, ph, g, pw, 3).mean((1, 3))
            ls = L[: ph * g, : pw * g].reshape(g, ph, g, pw).std((1, 3))[..., None]
            out.append(np.concatenate([cm, ls], -1))
        return np.stack(out)


class HFBackbone(Backbone):
    """kind: clip | dinov2. model_id from the config (local HF cache; HF_HUB_OFFLINE)."""

    def __init__(self, kind, model_id, device="cuda", batch=32):
        super().__init__()
        self.kind, self.model_id, self.device, self.batch = kind, model_id, device, batch
        self.name = f"{kind}:{model_id}"
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, CLIPModel
        if self.kind == "clip":
            # fp16 on GPU (as every stored feature); fp32 on CPU (fp16 matmuls are not usable there) -- 9 Nov 2026
            dt = torch.float16 if str(self.device).startswith("cuda") else torch.float32
            self._model = CLIPModel.from_pretrained(self.model_id, torch_dtype=dt).to(self.device).eval()
        else:
            self._model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()

    def _run(self, images, patches=False):
        import torch
        import torch.nn.functional as F
        from flowsat_C.scripts.eval_metrics import Extractors
        self._load()
        outs = []
        with torch.no_grad():
            for i in range(0, len(images), self.batch):
                x = torch.as_tensor(np.stack(images[i:i + self.batch])).permute(0, 3, 1, 2)
                x = x.to(self.device).float() / 255
                if self.kind == "clip":
                    m = self._model
                    size = m.config.vision_config.image_size
                    pv = Extractors._clip_norm(x, size).to(next(m.parameters()).dtype)
                    vo = m.vision_model(pixel_values=pv)
                    if patches:
                        t = vo.last_hidden_state[:, 1:].float()
                        g = int(round(t.shape[1] ** 0.5))
                        outs.append(t.reshape(len(x), g, g, -1).cpu().numpy())
                    else:
                        outs.append(m.visual_projection(vo.pooler_output).float().cpu().numpy())
                else:
                    xd = F.interpolate(x, size=224, mode="bicubic", align_corners=False, antialias=True)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                    o = self._model(pixel_values=(xd - mean) / std)
                    if patches:
                        t = o.last_hidden_state[:, 1:].float()
                        g = int(round(t.shape[1] ** 0.5))
                        outs.append(t.reshape(len(x), g, g, -1).cpu().numpy())
                    else:
                        outs.append(o.pooler_output.float().cpu().numpy())
        return np.concatenate(outs)

    def _features(self, images):
        return self._run(images)

    def _patch_features(self, images):
        return self._run(images, patches=True)


def make_backbone(spec: dict) -> Backbone:
    kind = spec["kind"]
    if kind == "handcrafted":
        return HandcraftedBackbone(spec.get("variant", "phi"))
    return HFBackbone(kind, spec["model_id"], spec.get("device", "cuda"), spec.get("batch", 32))
