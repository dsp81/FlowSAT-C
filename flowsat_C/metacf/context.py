"""
Context: everything a test needs, built once per run and cached (grids, support, anchors,
derangements, sweeps, natural experiments). Deterministic: every random choice is seeded from the
config seed plus a label (fields.seed_of).
"""
from __future__ import annotations

import numpy as np

from metacf import fields as F
from metacf import solar
from metacf.config import get
from metacf.requests import GenRequest


class _AnySupport:
    def contains(self, category, value):
        return True


class Context:
    def __init__(self, cfg, dataset, specs, readers, phi, adapter, lpips_fn=None):
        self.cfg, self.data, self.specs = cfg, dataset, specs
        self.readers, self.phi, self.adapter = readers, phi, adapter
        self.lpips_fn = lpips_fn
        self.base_seed = int(cfg.get("seed", 0))
        self._c = {}
        self.train = dataset.train
        self.test = dataset.test

    # ---------------------------------------------------------------- basics
    def seed(self, *parts):
        return F.seed_of(self.base_seed, *parts)

    def cached(self, key, fn):
        if key not in self._c:
            self._c[key] = fn()
        return self._c[key]

    def supported(self, f) -> bool:
        return f in self.adapter.supported_fields

    def eval_fields(self):
        """Fields to score: in the config, not derived, conditioned on by the model."""
        return [f for f, s in self.specs.items() if not s.derived and self.supported(f)]

    def sweep_fields(self):
        """Fields that get sweeps (T2, T3, T4, T5a, T6): everything evaluated except fields marked
        recovery_only (Koppen, land cover: location properties, not acquisition controls)."""
        return [f for f in self.eval_fields() if not self.specs[f].recovery_only]

    def fields_with_tag(self, tag):
        return [f for f in self.eval_fields() if tag in self.specs[f].tags]

    def has_readers(self, f):
        return f in self.readers.readers

    # ---------------------------------------------------------------- values
    def raw_values(self, records, f):
        return self.readers.raw_targets(records, f) if self.has_readers(f) else \
            np.array([self.specs[f].raw(r.fields) for r in records], dtype=object)

    # ---------------------------------------------------------------- geo sweeps
    def geo_mode(self):
        """distance (default): K destinations at increasing great-circle distance from each anchor,
        scored by Spearman(set distance, read-back distance from the anchor).
        latitude (option): latitude percentiles at the anchor's own longitude."""
        return get(self.cfg, "geo_sweep.mode", "distance")

    def sweep_spec(self, f):
        """The spec a sweep varies (geo: a distance or latitude pseudo-spec)."""
        s = self.specs[f]
        if s.kind != "geo":
            return s
        if self.geo_mode() == "latitude":
            return F.FieldSpec(name=f + ".lat", kind="continuous", keys=[s.keys[0]], tags=s.tags)
        return F.FieldSpec(name=f + ".dist_km", kind="continuous", keys=["__dist_km__"], tags=s.tags)

    def _train_locations(self, f):
        """Unique TRAIN locations per category: {category: (lat array, lon array)}."""
        def comp():
            s = self.specs[f]
            seen, by = set(), {}
            for r in self.train:
                la, lo = r.fields.get(s.keys[0]), r.fields.get(s.keys[1])
                if la is None or lo is None or r.location_id in seen:
                    continue
                seen.add(r.location_id)
                by.setdefault(r.category, ([], []))
                by[r.category][0].append(float(la))
                by[r.category][1].append(float(lo))
            return {c: (np.array(a), np.array(b)) for c, (a, b) in by.items()}
        return self.cached(("trainloc", f), comp)

    def _pair_distances(self, f, n=20000):
        """Great-circle distances between random same-category TRAIN location pairs (km)."""
        def comp():
            from metacf.metrics import great_circle_km
            rng = np.random.default_rng(self.seed("geo_pairs", f))
            locs = {c: v for c, v in self._train_locations(f).items() if len(v[0]) >= 2}
            if not locs:
                return np.zeros(0)
            cats = sorted(locs)
            out = []
            for _ in range(n):
                la, lo = locs[cats[int(rng.integers(len(cats)))]]
                i, j = rng.choice(len(la), 2, replace=False)
                out.append(float(great_circle_km(la[i], lo[i], la[j], lo[j])))
            return np.array(out)
        return self.cached(("geo_pairs", f), comp)

    def geo_destination(self, rid, f, d_km):
        """The same-category TRAIN location (in support by construction) whose distance from the
        anchor is closest to d_km; None when none lies within max(25% of d, 100 km)."""
        def comp():
            from metacf.metrics import great_circle_km
            r = self.data.by_id[rid]
            s = self.specs[f]
            la0, lo0 = r.fields.get(s.keys[0]), r.fields.get(s.keys[1])
            locs = self._train_locations(f).get(r.category)
            if la0 is None or lo0 is None or locs is None or len(locs[0]) == 0:
                return None
            dist = great_circle_km(la0, lo0, locs[0], locs[1])
            k = int(np.argmin(np.abs(dist - d_km)))
            tol = max(float(get(self.cfg, "geo_sweep.tolerance_frac", 0.25)) * d_km,
                      float(get(self.cfg, "geo_sweep.tolerance_min_km", 100.0)))
            if abs(dist[k] - d_km) > tol:
                return None
            return float(locs[0][k]), float(locs[1][k]), float(dist[k])
        return self.cached(("geodest", rid, f, round(float(d_km), 3)), comp)

    def train_values(self, f):
        s = self.sweep_spec(f)
        if s.keys == ["__dist_km__"]:
            return self._pair_distances(f)
        return self.cached(("trainv", f), lambda: np.array(
            [np.nan if s.raw(r.fields) is None else float(s.raw(r.fields)) for r in self.train], float))

    def support(self, f):
        s = self.sweep_spec(f)
        sc = get(self.cfg, "support", {})
        return self.cached(("support", f), lambda: F.Support(
            s, self.train_values(f), [r.category for r in self.train],
            lo_pct=sc.get("lo_pct", 1), hi_pct=sc.get("hi_pct", 99), min_cat=sc.get("min_cat", 50)))

    def grid(self, f, percentiles=None):
        p = tuple(percentiles or get(self.cfg, "grid.percentiles", [10, 30, 50, 70, 90]))
        return self.cached(("grid", f, p), lambda: F.value_grid(self.sweep_spec(f), self.train_values(f), p))

    def anchors(self, f, grid, n=None, tag="default", extra_filter=None, check_support=True):
        """check_support=False only where a test sweeps outside support BY DESIGN (T16b azimuth)."""
        n = n or get(self.cfg, "anchors.n", 20)
        key = ("anchors", f, tuple(map(str, grid)), n, tag, check_support)
        if self.sweep_spec(f).keys == ["__dist_km__"]:
            # distance sweeps: every destination is a TRAIN location of the anchor's category (in
            # support by construction); an anchor qualifies when all K destinations exist
            base = extra_filter
            extra_filter = lambda r: (base is None or base(r)) and all(
                self.geo_destination(r.rid, f, d) is not None for d in grid)
            check_support = False
            spec = self.specs[f]
            sel_spec = F.FieldSpec(name=f, kind="continuous", keys=[spec.keys[0]])
            return self.cached(key, lambda: F.select_anchors(
                self.test, sel_spec, grid, _AnySupport(), n, self.seed("anchors", f, tag), extra_filter))
        sup = self.support(f) if check_support else _AnySupport()
        return self.cached(key, lambda: F.select_anchors(
            self.test, self.sweep_spec(f), grid, sup, n, self.seed("anchors", f, tag), extra_filter))

    def anchor_seeds(self):
        return list(range(int(get(self.cfg, "anchors.seeds", 3))))

    # ---------------------------------------------------------------- record sets
    def t1_records(self, n):
        def pick():
            rng = np.random.default_rng(self.seed("t1_records"))
            ids = [r.rid for r in self.test]
            sel = rng.choice(len(ids), min(n, len(ids)), replace=False)
            return [ids[i] for i in sorted(sel)]
        return self.cached(("t1", n), pick)

    def derangement(self, rids, tag):
        def make():
            cats = [self.data.by_id[r].category for r in rids]
            perm = F.derangement(cats, self.seed("derangement", tag))
            return {rids[i]: rids[perm[i]] for i in range(len(rids))}
        return self.cached(("derange", tag, len(rids)), make)

    # ---------------------------------------------------------------- requests
    def caption(self, rid, caption_mode="real"):
        return None if caption_mode == "null" else self.data.by_id[rid].caption

    def own_request(self, rid, seed, caption_mode="real", null_groups=()):
        r = self.data.by_id[rid]
        return GenRequest.make(rid, r.fields, self.caption(rid, caption_mode), seed,
                               null_mask=null_groups, mode="own")

    def set_request(self, rid, f, value, seed, mode="RAW", caption_mode="real"):
        r = self.data.by_id[rid]
        spec = self.specs[f]
        if spec.kind == "geo" and np.ndim(value) == 0:
            if self.geo_mode() == "latitude":                # latitude at the anchor's own longitude
                value = (float(value), r.fields.get(spec.keys[1]))
            else:                                            # a destination at this distance
                dest = self.geo_destination(rid, f, float(value))
                value = (dest[0], dest[1])
        new = spec.set(r.fields, value)
        if mode == "CONSISTENT":
            new = solar.consistent(r.fields, new)
        return GenRequest.make(rid, new, self.caption(rid, caption_mode), seed, mode=mode)

    def sweep(self, f, anchors, grid, seeds, mode="RAW", caption_mode="real"):
        """{(anchor, k, seed): GenRequest}"""
        return {(a, k, s): self.set_request(a, f, v, s, mode, caption_mode)
                for a in anchors for k, v in enumerate(grid) for s in seeds}

    # ---------------------------------------------------------------- natural experiments
    def natural_pairs(self, f):
        return self.cached(("npairs", f), lambda: F.natural_pairs(
            self.train, self.specs[f], get(self.cfg, "natural.max_pairs_per_location", 50),
            self.seed("npairs", f)))

    def natural_sequences(self, f, min_distinct=3):
        return self.cached(("nseq", f, min_distinct), lambda: F.natural_sequences(
            self.train, self.specs[f], min_distinct))

    # ---------------------------------------------------------------- reading images
    def readback(self, f, reader, images):
        return self.readers.predict(f, reader, images)

    def truth(self, f, rids):
        recs = [self.data.by_id[r] for r in rids]
        return self.readers.truth(f, self.readers.raw_targets(recs, f))

    def real_images(self, rids):
        return [self.data.real_image(r) for r in rids]

    def real_readback(self, f, reader, rids):
        return self.readback(f, reader, self.real_images(rids))

    def phi_feats(self, images):
        return self.phi.features(images)

    def d_dino(self, xs, ys):
        """Paired 1 - cos(phi(x), phi(y))."""
        a, b = self.phi_feats(xs), self.phi_feats(ys)
        return np.clip(1.0 - (a * b).sum(1), 0.0, 2.0)

    def d_lpips(self, xs, ys):
        return None if self.lpips_fn is None else np.asarray(self.lpips_fn(xs, ys), float)

    def encode_set(self, f, values):
        """Transformed SWEEP values (what sweep read-backs are compared against)."""
        return self.sweep_spec(f).encode(np.asarray(values, float))

    def sweep_readback(self, f, reader, images, anchors=None):
        """Read-backs in the sweep space. Geo, distance mode: great-circle distance (km) from the
        image's anchor to the reader's predicted location; latitude mode: the latitude component."""
        p = np.asarray(self.readback(f, reader, images))
        if self.specs[f].kind != "geo":
            return p
        if self.geo_mode() == "latitude":
            return p[:, 0]
        from metacf.metrics import great_circle_km
        s = self.specs[f]
        a = [self.data.by_id[r].fields for r in anchors]
        return great_circle_km(np.array([x[s.keys[0]] for x in a], float),
                               np.array([x[s.keys[1]] for x in a], float), p[:, 0], p[:, 1])

    def train_readback(self, f, reader, rids):
        """Out-of-fold read-backs of TRAIN reals (readers were fit on TRAIN; in-fold predictions
        would inflate natural-experiment ceilings)."""
        return self.readers.oof(f, reader, rids)

    # ---------------------------------------------------------------- standardisation (T6a)
    def real_readback_std(self, f, reader):
        def comp():
            rids = [r.rid for r in self.test][: get(self.cfg, "standardise.n_real", 2000)]
            return float(np.nanstd(self.real_readback(f, reader, rids)))
        return self.cached(("rbstd", f, reader), comp)

    def train_field_std(self, f):
        return self.cached(("fstd", f), lambda: float(np.nanstd(self.specs[f].encode(self.train_values(f)))))
