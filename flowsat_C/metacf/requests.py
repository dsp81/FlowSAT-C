"""
Generation requests, the content-addressed cache and the planner (section 2, last bullet).

key = sha256(model id, checkpoint, record, fields, null_mask, caption, seed, guidance, mode
             [, pathway variant])
Every test declares its requests through plan(); the planner deduplicates across tests, so a sweep
shared by T2/T3/T4/T6 is generated once.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field as dc_field

import numpy as np


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return round(float(v), 9)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


@dataclass(frozen=True)
class GenRequest:
    rid: str
    fields: tuple                    # sorted (key, value) pairs -- the FULL field dict sent to the model
    caption: str | None              # None = null caption
    seed: int
    null_mask: tuple = ()            # field groups forced to their learned null
    guidance: tuple | None = None    # None = adapter default
    mode: str = "RAW"                # RAW | CONSISTENT | own
    variant: tuple = ()              # T9: (("pathway", p), ("mode", m), ("donor", rid), ("groups", ...))

    @staticmethod
    def make(rid, fields: dict, caption, seed, null_mask=(), guidance=None, mode="RAW", variant=None):
        f = tuple(sorted((k, _clean(v)) for k, v in fields.items()))
        var = tuple(sorted((k, tuple(v) if isinstance(v, list) else v)
                           for k, v in (variant or {}).items()))
        return GenRequest(rid, f, caption, int(seed), tuple(sorted(null_mask)),
                          None if guidance is None else tuple(guidance), mode, var)

    @property
    def field_dict(self):
        return dict(self.fields)

    @property
    def variant_dict(self):
        return dict(self.variant)

    def key(self, model_id, checkpoint) -> str:
        payload = dict(model=model_id, ckpt=checkpoint, rid=self.rid, fields=self.fields,
                       null=self.null_mask, caption=self.caption, seed=self.seed,
                       guidance=self.guidance, mode=self.mode, variant=self.variant)
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


class ImageStore:
    """In-memory images keyed by request key, mirrored to <root>/<k[:2]>/<k>.png when root is set."""

    def __init__(self, model_id, checkpoint, root=None):
        self.model_id, self.checkpoint, self.root = model_id, checkpoint, root
        self.mem = {}

    def key(self, req):
        return req.key(self.model_id, self.checkpoint)

    def _path(self, k):
        return os.path.join(self.root, k[:2], k + ".png")

    def has(self, req):
        k = self.key(req)
        return k in self.mem or (self.root is not None and os.path.exists(self._path(k)))

    def put(self, req, img):
        k = self.key(req)
        img = np.asarray(img, np.uint8)
        self.mem[k] = img
        if self.root is not None:
            from PIL import Image
            os.makedirs(os.path.dirname(self._path(k)), exist_ok=True)
            tmp = self._path(k) + ".tmp.png"
            Image.fromarray(img).save(tmp)
            os.replace(tmp, self._path(k))

    def get(self, req):
        k = self.key(req)
        if k not in self.mem:
            if self.root is None or not os.path.exists(self._path(k)):
                raise KeyError(f"image not generated: {req}")
            from PIL import Image
            self.mem[k] = np.asarray(Image.open(self._path(k)).convert("RGB"), np.uint8)
        return self.mem[k]

    def many(self, reqs):
        return [self.get(r) for r in reqs]


@dataclass
class Plan:
    unique: dict = dc_field(default_factory=dict)        # key -> GenRequest
    per_test: dict = dc_field(default_factory=dict)      # test id -> dict(requested, new)

    def add(self, test_id, reqs, model_id, checkpoint):
        n_new = 0
        for r in reqs:
            k = r.key(model_id, checkpoint)
            if k not in self.unique:
                self.unique[k] = r
                n_new += 1
        self.per_test[test_id] = dict(requested=len(reqs), new_unique=n_new)

    def summary(self, seconds_per_image):
        n = len(self.unique)
        return dict(unique_images=n, per_test=self.per_test, seconds_per_image=seconds_per_image,
                    gpu_hours=n * seconds_per_image / 3600.0)

    def to_json(self, path):
        rows = [dict(key=k, rid=r.rid, fields=r.fields, caption=r.caption, seed=r.seed,
                     null_mask=r.null_mask, guidance=r.guidance, mode=r.mode, variant=r.variant)
                for k, r in self.unique.items()]
        with open(path, "w") as fh:
            json.dump(dict(per_test=self.per_test, requests=rows), fh, default=str)
