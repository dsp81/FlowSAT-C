"""
Meta-evaluation wrappers (spec section 5): inference-time adapters around any ModelAdapter.

  ShuffledField(f)  replaces field f with a fixed derangement donor's value (donor fixed per record id:
                    metacf.fields.derangement over every record, stratified by category).
  NulledGroup(g)    null mask on group g.
  BlurForZoom       generates at the median effective GSD (TRAIN) and applies a Gaussian blur with
                    sigma_px = max(0, 1.5 log2(g'_req / g'_med)).
  TintForSeason     generates at a fixed date (config, default doy 196; RAW, sun unchanged) and multiplies
                    linear-RGB green by 1 + 0.15 cos(2 pi (d - 200) / 365.25), then rescales every pixel so
                    R + G + B is unchanged (chromaticity renormalised, intensity kept; Q20).
  Retrieval         returns the nearest TRAIN real tile in z-scored metadata space (regressor columns +
                    unit-sphere geo; z-scored with TRAIN) instead of generating.

Each wrapper has its own model_id (base id + "|" + wrapper tag), so its cache keys never collide with the
unbroken model, and it fetches the base images it needs from the base model's image store (reusing every
image the unbroken run already generated).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from metacf import physical
from metacf.fields import derangement
from metacf.requests import GenRequest


def blur_for_zoom(img, g_req, g_med):
    s = max(0.0, 1.5 * math.log2(max(g_req, 1e-9) / g_med))
    if s <= 0:
        return np.asarray(img, np.uint8)
    a = np.asarray(img, float)
    out = np.stack([ndimage.gaussian_filter(a[..., c], s, mode="reflect") for c in range(3)], -1)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def tint_for_season(img, d, amp=0.15, peak=200.0):
    c = physical.srgb_to_linear(img)
    s0 = c.sum(-1, keepdims=True)
    c = c * np.array([1.0, 1.0 + amp * math.cos(2 * math.pi * (d - peak) / 365.25), 1.0])
    s1 = c.sum(-1, keepdims=True)
    c = np.where(s1 > 1e-9, c * s0 / np.maximum(s1, 1e-9), c)
    c = np.clip(c, 0, 1)
    srgb = np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)
    return np.clip(np.round(srgb * 255), 0, 255).astype(np.uint8)


class Wrapper:
    tag = "wrapper"

    def __init__(self, base, base_store, run=None):
        self.base, self.base_store, self.run = base, base_store, run
        self.name = f"{base.name}|{self.tag}"
        self.model_id = f"{base.model_id}|{self.tag}"
        self.checkpoint = base.checkpoint
        self.supported_fields = set(getattr(base, "supported_fields", set()))
        self.field_groups = dict(getattr(base, "field_groups", {}))

    def _base(self, req):
        if not self.base_store.has(req):
            self.base_store.put(req, np.asarray(self.base.generate_request(req)))
        return self.base_store.get(req)

    def transform_request(self, req):
        return req

    def post(self, req, img):
        return img

    def generate_request(self, req, donor_fields=None):
        return self.post(req, self._base(self.transform_request(req)))


def _remake(req, fields=None, null_mask=None, mode=None):
    return GenRequest.make(req.rid, fields if fields is not None else req.field_dict, req.caption, req.seed,
                           null_mask if null_mask is not None else req.null_mask, req.guidance,
                           mode or req.mode, req.variant_dict or None)


class ShuffledField(Wrapper):
    def __init__(self, base, base_store, run, field):
        self.field = field
        self.tag = f"ShuffledField({field})"
        super().__init__(base, base_store, run)
        recs = run.ds.records
        perm = derangement([r.category for r in recs], run.seed("meta_derangement", field))
        self.donor = {recs[i].rid: recs[perm[i]].rid for i in range(len(recs))}

    def transform_request(self, req):
        fd = self.run.fields[self.field]
        donor = self.run.rec(self.donor[req.rid])
        f = req.field_dict
        for k in fd.keys:
            f[k] = donor.fields.get(k)
        return _remake(req, f)


class NulledGroup(Wrapper):
    def __init__(self, base, base_store, run, group):
        self.group = group
        self.tag = f"NulledGroup({group})"
        super().__init__(base, base_store, run)

    def transform_request(self, req):
        return _remake(req, null_mask=tuple(sorted(set(req.null_mask) | {self.group})))


class BlurForZoom(Wrapper):
    tag = "BlurForZoom"

    def __init__(self, base, base_store, run, gsd_field="gsd"):
        super().__init__(base, base_store, run)
        self.f = run.fields[gsd_field]
        v = run.train_values(gsd_field)
        self.g_med = float(np.median(v[np.isfinite(v)]))

    def transform_request(self, req):
        f = req.field_dict
        for k in self.f.keys:
            f[k] = self.g_med
        return _remake(req, f)

    def post(self, req, img):
        return blur_for_zoom(img, float(req.field_dict[self.f.keys[0]]), self.g_med)


class TintForSeason(Wrapper):
    tag = "TintForSeason"

    def __init__(self, base, base_store, run, doy_field="doy", fixed_doy=196.0, amp=0.15, peak=200.0):
        super().__init__(base, base_store, run)
        self.f, self.fixed, self.amp, self.peak = run.fields[doy_field], float(fixed_doy), amp, peak

    def transform_request(self, req):
        f = req.field_dict
        f[self.f.keys[0]] = self.fixed
        return _remake(req, f, mode="RAW")

    def post(self, req, img):
        return tint_for_season(img, float(req.field_dict[self.f.keys[0]]), self.amp, self.peak)


class Retrieval(Wrapper):
    tag = "Retrieval"

    def __init__(self, base, base_store, run):
        super().__init__(base, base_store, run)
        self.train = [r for r in run.train if run.complete(r)]
        self.mu, self.sd = None, None
        self.X = self._mat([r.fields for r in self.train], fit=True)

    def _mat(self, fds, fit=False):
        r = self.run
        cols = [r.design.matrix(fds)] if r.design is not None and r.design.p else []
        geo = [fd for fd in r.fields.values() if fd.kind == "geo"]
        if geo:
            a = np.array([[np.nan, np.nan] if geo[0].raw(f) is None else geo[0].raw(f) for f in fds], float)
            la, lo = np.radians(a[:, 0]), np.radians(a[:, 1])
            cols.append(np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]))
        M = np.concatenate(cols, 1)
        if fit:
            self.mu, self.sd = np.nanmean(M, 0), np.nanstd(M, 0) + 1e-9
        return np.nan_to_num((M - self.mu) / self.sd)

    def generate_request(self, req, donor_fields=None):
        q = self._mat([req.field_dict])[0]
        k = int(np.argmin(((self.X - q) ** 2).sum(1)))
        return self.run.ds.real_image(self.train[k].rid)


KINDS = dict(ShuffledField=ShuffledField, NulledGroup=NulledGroup, BlurForZoom=BlurForZoom,
             TintForSeason=TintForSeason, Retrieval=Retrieval)


def make_wrapper(spec, base, base_store, run):
    kind = spec["kind"]
    kw = {k: v for k, v in spec.items() if k != "kind"}
    return KINDS[kind](base, base_store, run, **kw)
