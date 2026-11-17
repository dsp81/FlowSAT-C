"""
The Run object: everything a stage or measure needs, built from one config.

Directories (output.root = R):
  R/_features/<space>/              feature stores (real tiles and generated images; keys carry the model)
  R/_protocol/<protocol hash>/      model-independent real-data artifacts: dataset.jsonl, design.json,
                                    pairs_{train,test}.json, stats_norm.json, selection.json, readers.pkl
                                    (protocol hash = config hash without the model / plan / meta sections,
                                    so the meta harness and other models reuse them)
  R/<model name>/<config hash>/     images/, plan.json, results.json, control_card.{md,html}
Every random choice derives from the config seed plus a label (metacf.fields.seed_of).
"""
from __future__ import annotations

import copy
import json
import os
import pickle
import time

import numpy as np

from metacf.config import canonical_json, config_hash, get
from metacf.data import Dataset, read_manifest
from metacf.fields import Support, seed_of
from metacf.requests import GenRequest, ImageStore
from metacf.v2.core import features as FT
from metacf.v2.core.fields import Design, fields_from_config, fit_classes
from metacf.v2.core.pairs import PairSet

# v2_1: scoring extensions of 9 Nov 2026 (zoom GSD reference, E1 zoom rows, C3 floor). They change no protocol
# artifact (pairs, selection, readers), so they stay out of the protocol hash and v2 protocol dirs are reused.
# v2_2 (9 Nov 2026, E1 GSD matching + shared real-trained GSD reader): same reasoning.
# v2_3 (11 Nov 2026): GSD-as-zoom in C0/C1/C2, bounded scores, C4 CI fix, group-level C3 floor, one card.
MODEL_SECTIONS = ("model", "plan", "meta", "output", "runtime", "v2_1", "v2_2", "v2_3")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def draft_loader(resolution=512, draft=True):
    """Real tile -> uint8 (resolution x resolution): the training transform (centre crop to the short side,
    bilinear antialiased resize), with JPEG draft decoding at >= 2 x resolution first when `draft` (DCT-domain
    downscale: ~4x faster on the large fMoW chips, mean |difference| 0.4 / 255 vs a full decode; Q32)."""
    def fn(rec):
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(rec.image)
        if draft and im.format == "JPEG":
            im.draft("RGB", (2 * resolution, 2 * resolution))
        im = im.convert("RGB")
        s = min(im.size)
        left, top = (im.width - s) // 2, (im.height - s) // 2
        im = im.crop((left, top, left + s, top + s)).resize((resolution, resolution), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)
    return fn


def protocol_hash(cfg):
    c = {k: v for k, v in cfg.items() if k not in MODEL_SECTIONS}
    return config_hash(c)


class Run:
    def __init__(self, cfg, dataset=None, adapter=None, spaces=None, text_encoder=None, root=None,
                 persist=True):
        self.cfg = copy.deepcopy(cfg)
        self.hash = config_hash(self.cfg)
        self.phash = protocol_hash(self.cfg)
        self.persist = persist
        self.root = root or get(cfg, "output.root", "results_v2")
        self.base_seed = int(cfg.get("seed", 0))
        self.B = int(get(cfg, "bootstrap.B", 1000))
        self.q = float(get(cfg, "stats.bh_q", 0.05))
        self.fields = fields_from_config(cfg)
        self._c = {}
        self.ds = dataset
        self.adapter = adapter
        self._spaces = spaces
        self._text = text_encoder
        self.stats_norm = None
        self.selection = None
        self.design = None
        self.pairs = {}
        self.fstores = {}
        if self.ds is not None:
            fit_classes(self.fields, self.ds.train)

    # ---------------------------------------------------------------- dirs
    @property
    def pdir(self):
        d = os.path.join(self.root, "_protocol", self.phash)
        if self.persist:
            os.makedirs(d, exist_ok=True)
        return d

    @property
    def mdir(self):
        name = self.cfg["model"].get("name", self.cfg["model"]["adapter"])
        d = os.path.join(self.root, name, self.hash)
        if self.persist:
            os.makedirs(d, exist_ok=True)
        return d

    def seed(self, *parts):
        return seed_of(self.base_seed, *parts)

    def cached(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    # ---------------------------------------------------------------- data
    def build_dataset(self):
        dc = self.cfg["data"]
        res = int(dc.get("resolution", 512))
        recs = read_manifest(dc["train_manifest"], "train", res, dc.get("train_limit")) + \
            read_manifest(dc["test_manifest"], "test", res, dc.get("test_limit"))
        cats = sorted({r.category for r in recs})
        for r in recs:
            r.fields["category_idx"] = cats.index(r.category)
        self.ds = Dataset(recs, image_fn=draft_loader(res, bool(dc.get("jpeg_draft", True))), resolution=res,
                          cache_size=int(get(self.cfg, "runtime.image_cache", 512)))
        if self.persist:
            self.ds.save(os.path.join(self.pdir, "dataset.jsonl"))
        fit_classes(self.fields, self.ds.train)
        return self.ds

    def load_dataset(self):
        p = os.path.join(self.pdir, "dataset.jsonl")
        self.ds = Dataset.load(p, int(self.cfg["data"].get("resolution", 512)))
        self.ds._cache_size = int(get(self.cfg, "runtime.image_cache", 512))
        self.ds._image_fn = draft_loader(self.ds.resolution, bool(self.cfg["data"].get("jpeg_draft", True)))
        fit_classes(self.fields, self.ds.train)
        return self.ds

    @property
    def train(self):
        return self.cached("train", lambda: self.ds.train)

    @property
    def test(self):
        return self.cached("test", lambda: self.ds.test)

    def rec(self, rid):
        return self.ds.by_id[rid]

    def conditioned(self, name):
        """Does the model condition on this field? (adapter.supported_fields; joint: all members)."""
        sup = getattr(self.adapter, "supported_fields", None)
        if sup is None:
            return True
        if self.design is not None and name in self.design.joint:
            return all(m in sup for m in self.design.joint[name])
        return name in sup

    # ---------------------------------------------------------------- design / pairs / selection
    def save_json(self, name, obj, where="p"):
        if not self.persist:
            return
        d = self.pdir if where == "p" else self.mdir
        with open(os.path.join(d, name), "w") as fh:
            json.dump(obj, fh, indent=1, default=_jd)

    def load_json(self, name, where="p"):
        d = self.pdir if where == "p" else self.mdir
        with open(os.path.join(d, name)) as fh:
            return json.load(fh)

    def make_design(self):
        self.design = Design(self.fields)
        return self.design

    def load_design(self):
        d = self.load_json("design.json")
        self.design = Design(self.fields, d["names"])
        for name, mem in d.get("joint", {}).items():
            self.design.merge(mem, name)
        return self.design

    def load_pairs(self):
        for split in ("train", "test"):
            self.pairs[split] = PairSet.from_json(self.load_json(f"pairs_{split}.json"))
        return self.pairs

    def load_selection(self):
        self.selection = self.load_json("selection.json")["selection"]
        return self.selection

    def space_of(self, field):
        """Selected space of a regression field (or of the regression field that owns it), else the default."""
        default = get(self.cfg, "default_space", "dino")
        if self.selection is None:
            return default
        own = field if field in self.selection else (self.design.owner(field) if self.design else None)
        s = self.selection.get(own) if own else None
        return s["space"] if s and s.get("space") else default

    def valid(self, field):
        own = field if (self.selection and field in self.selection) else (self.design.owner(field) if self.design else None)
        s = (self.selection or {}).get(own)
        return bool(s and s.get("valid")), (s or {}).get("reason", "not a regression field")

    # ---------------------------------------------------------------- spaces / features
    def space(self, name) -> FT.Space:
        if self._spaces is None:
            self._spaces = {}
        if name not in self._spaces:
            self._spaces[name] = FT.make_space(name, self.cfg["spaces"][name])
        return self._spaces[name]

    def space_names(self):
        return list(self.cfg["spaces"].keys())

    def fstore(self, name):
        if name not in self.fstores:
            root = os.path.join(self.root, "_features", name + "__" + _safe(self.space(name).model_id)) \
                if self.persist else None
            self.fstores[name] = FT.FeatureStore(root)
        return self.fstores[name]

    def text_encoder(self):
        if self._text is None:
            self._text = FT.make_text_encoder(self.cfg["caption_encoder"])
        return self._text

    def store(self) -> ImageStore:
        if "store" not in self._c:
            root = os.path.join(self.mdir, "images") if self.persist else None
            self._c["store"] = ImageStore(self.adapter.model_id, self.adapter.checkpoint, root)
        return self._c["store"]

    def _loader(self, key):
        if key.startswith("real:"):
            return self.ds.real_image(key[5:])
        if key.startswith("syn:"):                 # v2_1 zoom / blur pairs built from real chips (core/zoom.py)
            if "syn_loader" not in self._c:
                from metacf.v2.core.zoom import SynLoader
                self._c["syn_loader"] = SynLoader(self.ds.by_id, int(self.ds.resolution), ds=self.ds)
            return self._c["syn_loader"](key)
        st = self.store()
        req = self._c.get("req_by_key", {}).get(key[4:])
        if req is None:
            raise KeyError(f"unknown generated key {key}")
        return st.get(req)

    def real_feats(self, space, rids, raw=False):
        keys = [FT.real_key(r) for r in rids]
        return self._feats(space, keys, raw)

    def gen_feats(self, space, reqs, raw=False):
        st = self.store()
        rk = self._c.setdefault("req_by_key", {})
        keys = []
        for r in reqs:
            k = st.key(r)
            rk[k] = r
            keys.append(FT.gen_key(k))
        return self._feats(space, keys, raw)

    def syn_feats(self, space, keys, raw=False):
        """Features of synthetic images derived from real chips; keys from metacf.v2.core.zoom.syn_key."""
        return self._feats(space, list(keys), raw)

    def _feats(self, space, keys, raw=False):
        fs = self.fstore(space)
        if fs.missing(keys):
            sp = self.space(space)
            FT.extract(sp, fs, keys, self._loader, int(get(self.cfg, "runtime.feature_batch", 64)),
                       int(get(self.cfg, "runtime.load_threads", 4)))
        X = fs.get(keys)
        if raw:
            return X
        if space == "stats" and self.stats_norm is None:
            self.fit_stats_norm()
        return self.space(space).post(X, self)

    def fit_stats_norm(self, rids=None):
        """z-score normaliser of the stats space from TRAIN real tiles (the TRAIN pair images by default)."""
        p = os.path.join(self.pdir, "stats_norm.json")
        if rids is None and self.persist and os.path.exists(p):
            self.stats_norm = FT.StatsNormaliser.from_json(json.load(open(p)))
            return self.stats_norm
        if rids is None:
            ps = self.pairs.get("train")
            rids = sorted(set(ps.a) | set(ps.b)) if ps is not None and ps.n else [r.rid for r in self.train]
            n = int(get(self.cfg, "stats_norm.n_max", 20000))
            if len(rids) > n:
                rng = np.random.default_rng(self.seed("stats_norm"))
                rids = sorted(rng.choice(rids, n, replace=False).tolist())
        X = self.real_feats("stats", rids, raw=True)
        self.stats_norm = FT.StatsNormaliser().fit(X)
        if self.persist:
            json.dump(self.stats_norm.to_json(), open(p, "w"))
        return self.stats_norm

    # ---------------------------------------------------------------- support / anchors
    def support(self, name):
        def comp():
            fd = self.fields[name]
            vals = [fd.raw(r.fields) for r in self.train]
            v = np.array([np.nan if x is None else float(x) for x in vals], float)
            from metacf.fields import FieldSpec
            kind = fd.kind
            spec = FieldSpec(name=name, kind=kind, keys=fd.keys,
                             transform="log" if fd.transform == "log" else "none",
                             period=fd.period or (365.25 if kind == "circular" else 0))
            return Support(spec, v, [r.category for r in self.train],
                           int(get(self.cfg, "support.lo_pct", 1)), int(get(self.cfg, "support.hi_pct", 99)),
                           int(get(self.cfg, "support.min_cat", 50)))
        return self.cached(("support", name), comp)

    def complete(self, r, names=None):
        """All the listed (default: every enabled) fields present on record r."""
        names = names or [n for n, f in self.fields.items() if f.enabled]
        return all(self.fields[n].raw(r.fields) is not None for n in names)

    def train_values(self, name):
        fd = self.fields[name]
        return self.cached(("trainv", name), lambda: np.array(
            [np.nan if fd.raw(r.fields) is None else float(fd.raw(r.fields)) for r in self.train], float))

    # ---------------------------------------------------------------- requests
    def req(self, rec, fields=None, caption="own", seed=0, null_mask=(), mode="RAW", tag=""):
        """GenRequest for record rec with (modified) fields; caption "own" = the record's caption,
        None = null caption. `tag` only documents the request's origin (not part of the cache key)."""
        cap = rec.caption if caption == "own" else caption
        return GenRequest.make(rec.rid, fields if fields is not None else rec.fields, cap, int(seed),
                               null_mask, None, mode)

    def guidance_note(self):
        m = self.cfg["model"]
        return dict(w_txt=m.get("w_txt"), w_md=m.get("w_md"), steps=m.get("steps"))

    # ---------------------------------------------------------------- readers pickle
    def save_readers(self, obj):
        if self.persist:
            with open(os.path.join(self.pdir, "readers.pkl"), "wb") as fh:
                pickle.dump(obj, fh)
        self._c["readers"] = obj

    def readers(self):
        if "readers" not in self._c:
            with open(os.path.join(self.pdir, "readers.pkl"), "rb") as fh:
                self._c["readers"] = pickle.load(fh)
        return self._c["readers"]

    def write_config(self):
        if self.persist:
            with open(os.path.join(self.mdir, "config.json"), "w") as fh:
                fh.write(canonical_json(self.cfg))


def _safe(s):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))[-80:]


def _jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)
