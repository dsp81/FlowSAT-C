"""
C3 CONDITIONAL INFORMATION (spec section 2), every enabled field incl. geo; covariate fields are reported
but excluded from the summary.

  Inputs: image features (f's selected space concatenated with the default space, dino; just the default
    space when they coincide), caption embedding (CLIP text embedding of the caption; truncated at 77 tokens,
    Q9), transformed other fields = every enabled field outside f's group (Q14): regressors through the
    design columns, geo as unit-sphere xyz, continuous covariates signed-log1p, categorical covariates
    one-hot over TRAIN classes.
  Full reader R_f: [image, caption, other fields] -> f. Base reader R0_f: [caption, other fields] -> f.
    Same head types as C2 (geo: unit-sphere xyz, scored by -median km). Fit on n_train (5000) TRAIN real
    tiles in the fit-readers stage.
  gain_real = s(R_f on TEST real) - s(R0_f on TEST); gain_gen = s(R_f on generated) - s(R0_f on TEST),
    generated = G(r, own metadata, 1 seed) for the same n_test TEST records (shared with C4).
  s_hat = gain_gen / gain_real. Gate: gain_real CI excludes 0 (site bootstrap, p of gain_real <= 0, BH).

v2_1 (9 Nov 2026, `v2_1.c3_floor: null | shuffle`): a FLOOR = the same model with f's information removed:
  null    = the same requests with f's group forced to its learned null (FlowSAT-C) / zero (v1);
  shuffle = f's value taken from another shared TEST record (a seeded derangement).
  gain_floor = s(R_f on floor images) - s(R0_f); extra row "ratio_floored" = (gain_gen - gain_floor) /
  (gain_real - gain_floor). A model that does not condition on f has floor images == its own images, so the
  floored ratio is exactly 0 with no generation (the v1 elevation / koppen rows). Floor images that were not
  generated yet give status PENDING (never generated at score time). The v2 "ratio" row is kept unchanged.
"""
from __future__ import annotations

import numpy as np

from metacf.config import get
from metacf.v2.core import norm as N
from metacf.v2.core.heads import Head, score
from metacf.v2.measures.base import Measure
from metacf.v2.measures.c2_transfer import test_tiles


def own_requests(run):
    """G(r, own metadata, 1 seed) for the shared TEST records (C3 / C4)."""
    return run.cached(("shared", "own"), lambda: [run.req(x, seed=run.seed("own", x.rid)) for x in test_tiles(run)])


class C3Information(Measure):
    id = "C3"
    name = "conditional information"

    def fields(self):
        return [n for n, fd in self.run.fields.items() if fd.enabled]

    def v23_floor(self):
        """`v2_3.c3_floor: {mode, n, by_group}` (None = the v2_1 behaviour). n = floor images per group (a seeded
        subset of the shared TEST records; the floored ratio is then computed on that subset only);
        by_group = one floor set per metadata GROUP (all its fields changed together, one permutation), so the
        covariates cost one set instead of six."""
        v = (self.run.cfg.get("v2_3") or {}).get("c3_floor")
        return dict(v) if isinstance(v, dict) else None

    def floor_mode(self):
        v = self.v23_floor()
        if v:
            return v.get("mode", "shuffle")
        return get(self.run.cfg, "v2_1.c3_floor", None)

    def floor_subset(self, n_total):
        """Indices (sorted) of the shared TEST records that get floor images."""
        v = self.v23_floor()
        n = int(v["n"]) if v and v.get("n") else n_total
        if n >= n_total:
            return np.arange(n_total)
        return np.sort(np.random.default_rng(self.seed("floorsub")).choice(n_total, n, replace=False))

    def _group_fields(self, name):
        r = self.run
        g = r.fields[name].group
        return [n for n, fd in r.fields.items() if fd.enabled and fd.group == g]

    def floor_requests(self, name):
        """v2_1 floor requests for field `name` (own requests when the model does not condition on it)."""
        r = self.run
        mode = self.floor_mode()
        own = own_requests(r)
        if not mode or not r.conditioned(name):
            return own
        v23 = self.v23_floor()
        if v23 and bool(v23.get("by_group", True)):
            return self._floor_by_group(name, mode)

        def comp():
            tt = test_tiles(r)
            fd = r.fields[name]
            if mode == "null":
                return [r.req(x, seed=r.seed("own", x.rid), null_mask=(fd.group,)) for x in tt]
            if mode == "shuffle":
                rng = np.random.default_rng(self.seed("floorperm", name))
                n = len(tt)
                while True:
                    perm = rng.permutation(n)
                    if n < 2 or (perm != np.arange(n)).all():
                        break
                out = []
                for i, x in enumerate(tt):
                    v = fd.raw(tt[perm[i]].fields)
                    f2 = fd.set(x.fields, v) if v is not None else dict(x.fields)
                    out.append(r.req(x, f2, seed=r.seed("own", x.rid)))
                return out
            raise ValueError(f"v2_1.c3_floor: {mode}")
        return r.cached(("C3", "floor", mode, name), comp)

    def _floor_by_group(self, name, mode):
        """v2_3: floor requests of the group that owns `name`, for the floor subset only (list aligned with
        floor_subset; the same list object for every field of the group). The permutation is seeded by the
        group's FIRST field, so single-field groups reproduce the v2_1 per-field shuffle requests."""
        r = self.run
        names = self._group_fields(name)
        if not any(r.conditioned(n) for n in names):
            return own_requests(r)

        def comp():
            tt = test_tiles(r)
            idx = self.floor_subset(len(tt))
            g = r.fields[name].group
            if mode == "null":
                return [r.req(tt[i], seed=r.seed("own", tt[i].rid), null_mask=(g,)) for i in idx]
            rng = np.random.default_rng(self.seed("floorperm", names[0]))
            n = len(tt)
            while True:
                perm = rng.permutation(n)
                if n < 2 or (perm != np.arange(n)).all():
                    break
            out = []
            for i in idx:
                x, d = tt[i], tt[perm[i]]
                f2 = dict(x.fields)
                for nm in names:
                    if not r.conditioned(nm):
                        continue
                    v = r.fields[nm].raw(d.fields)
                    if v is not None:
                        f2 = r.fields[nm].set(f2, v)
                if g == "cov" and "cov.landcover_hist" in d.fields:
                    f2["cov.landcover_hist"] = d.fields["cov.landcover_hist"]
                out.append(r.req(x, f2, seed=r.seed("own", x.rid)))
            return out
        return r.cached(("C3", "floorg", mode, r.fields[name].group), comp)

    def plan(self):
        out = list(own_requests(self.run))
        if self.floor_mode():
            for name in self.fields():
                if name in self.run.fields and self.run.conditioned(name):
                    out += list(self.floor_requests(name))
        return out

    def other_matrix(self, name, recs):
        r = self.run
        g = r.fields[name].group
        cols = []
        for n, fd in r.fields.items():
            if not fd.enabled or fd.group == g:
                continue
            vals = [fd.raw(x.fields) for x in recs]
            if fd.kind == "geo":
                a = np.array([[np.nan, np.nan] if v is None else v for v in vals], float)
                la, lo = np.radians(a[:, 0]), np.radians(a[:, 1])
                cols += [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]
            elif fd.location and fd.kind == "categorical":
                for c in (fd.classes or [])[:32]:
                    cols.append(np.array([np.nan if v is None else float(int(v) == c) for v in vals]))
            elif fd.location:
                v = np.array([np.nan if x is None else float(x) for x in vals])
                cols.append(np.sign(v) * np.log1p(np.abs(v)))
            else:
                M = fd.columns(vals)
                cols += [M[:, k] for k in range(M.shape[1])]
        return np.column_stack(cols) if cols else np.zeros((len(recs), 0))

    def spaces(self, name):
        r = self.run
        d = r.cfg.get("default_space", "dino")
        own = r.space_of(name) if not r.fields[name].location else d
        return list(dict.fromkeys([own, d]))

    def img_real(self, name, rids):
        return np.concatenate([self.run.real_feats(sp, rids) for sp in self.spaces(name)], 1)

    def img_gen(self, name, reqs):
        return np.concatenate([self.run.gen_feats(sp, reqs) for sp in self.spaces(name)], 1)

    def captions(self, recs):
        return self.run.text_encoder()([x.caption for x in recs])

    def fit_readers(self):
        r = self.run
        elig = [x for x in r.train if r.complete(x)]
        n = int(self.c("n_train", 5000))
        rng = np.random.default_rng(self.seed("train"))
        recs = [elig[i] for i in sorted(rng.choice(len(elig), min(n, len(elig)), replace=False))]
        C = self.captions(recs)
        groups = [x.location_id for x in recs]
        out = {}
        for name in self.fields():
            fd = r.fields[name]
            y = fd.target([fd.raw(x.fields) for x in recs])
            O = self.other_matrix(name, recs)
            I = self.img_real(name, [x.rid for x in recs])
            full = Head(fd.kind).fit(np.concatenate([I, C, O], 1), y, groups)
            base = Head(fd.kind).fit(np.concatenate([C, O], 1), y, groups)
            out[name] = dict(full=full, base=base, spaces=self.spaces(name), n=len(recs))
        return out

    def compute(self):
        r = self.run
        out = []
        readers = r.readers().get("C3", {})
        tt = test_tiles(r)
        reqs = own_requests(r)
        sites = [x.location_id for x in tt]
        _, sidx = np.unique(sites, return_inverse=True)
        C = self.captions(tt)
        summ = set(get(r.cfg, "summary.exclude_groups", ["cov"]))
        for name in self.fields():
            fd = r.fields[name]
            if name not in readers:
                out.append(N.invalid(self.id, name, "ratio", "no readers (fit-readers not run for this field)"))
                continue
            rd = readers[name]
            y = fd.target([fd.raw(x.fields) for x in tt])
            O = self.other_matrix(name, tt)
            p_real = rd["full"].predict(np.concatenate([self.img_real(name, [x.rid for x in tt]), C, O], 1))
            p_gen = rd["full"].predict(np.concatenate([self.img_gen(name, reqs), C, O], 1))
            p_base = rd["base"].predict(np.concatenate([C, O], 1))

            def st_(w=None):
                ww = None if w is None else N.site_weights_to_items(w, sidx)
                sb = score(fd.kind, p_base, y, ww)
                gr = score(fd.kind, p_real, y, ww) - sb
                gg = score(fd.kind, p_gen, y, ww) - sb
                return dict(gain_real=gr, gain_gen=gg, ratio=gg / gr if abs(gr) > 1e-12 else np.nan)
            pt = st_()
            bs = N.site_bootstrap(st_, sites, self.run.B, self.seed("boot", name))
            p_gate = N.boot_p_greater(bs["gain_real"])
            if self.floor_mode():
                out.append(self._floored(name, fd, rd, reqs, tt, C, O, y, p_real, p_gen, p_base, sites, sidx,
                                         p_gate, summ))
            out.append(N.row(self.id, name, "ratio", raw=pt["gain_gen"], floor=0.0, ceil=pt["gain_real"],
                             ci=N.percentile_ci(bs["ratio"]), n=len(tt), seeds=1, p=p_gate,
                             space="+".join(rd["spaces"]), gain_real=pt["gain_real"], gain_gen=pt["gain_gen"],
                             gain_real_ci=N.percentile_ci(bs["gain_real"]), in_summary=fd.group not in summ,
                             gate="gain_real CI excludes 0", metric_kind=fd.kind))
        return out

    def _floored(self, name, fd, rd, reqs, tt, C, O, y, p_real, p_gen, p_base, sites, sidx, p_gate, summ):
        r = self.run
        mode = self.floor_mode()
        freqs = self.floor_requests(name)
        same = freqs is own_requests(r)
        sub_idx = None
        if not same and self.v23_floor() and bool(self.v23_floor().get("by_group", True)):
            sub_idx = self.floor_subset(len(tt))       # floor images exist for this subset only
        if sub_idx is not None:
            st = r.store()
            miss = sum(1 for q in freqs if not st.has(q))
            if miss:
                return N.invalid(self.id, name, "ratio_floored", f"floor images not generated ({miss}/{len(freqs)} "
                                 f"missing, mode {mode})", status="PENDING", n_missing=miss,
                                 in_summary=fd.group not in summ, floor_mode=mode)
            C_, O_ = C[sub_idx], O[sub_idx]
            p_fl = rd["full"].predict(np.concatenate([self.img_gen(name, freqs), C_, O_], 1))
            ys, pr, pg, pb = y[sub_idx], p_real[sub_idx], p_gen[sub_idx], p_base[sub_idx]
            ssites = [sites[i] for i in sub_idx]
            _, ssidx = np.unique(ssites, return_inverse=True)

            def st2(w=None):
                ww = None if w is None else N.site_weights_to_items(w, ssidx)
                sb = score(fd.kind, pb, ys, ww)
                gr = score(fd.kind, pr, ys, ww) - sb
                gg = score(fd.kind, pg, ys, ww) - sb
                gf = score(fd.kind, p_fl, ys, ww) - sb
                den = gr - gf
                return dict(gain_floor=gf, gain_real=gr, gain_gen=gg, den=den,
                            ratio=(gg - gf) / den if abs(den) > 1e-12 else np.nan)
            pt = st2()
            bs = N.site_bootstrap(st2, ssites, self.run.B, self.seed("bootfl", name))
            return N.row(self.id, name, "ratio_floored", raw=pt["gain_gen"], floor=pt["gain_floor"],
                         ceil=pt["gain_real"], s_hat=pt["ratio"], ci=N.percentile_ci(bs["ratio"]), n=len(sub_idx),
                         seeds=1, p=N.boot_p_greater(bs["den"]), space="+".join(rd["spaces"]),
                         gain_real=pt["gain_real"], gain_gen=pt["gain_gen"], gain_floor=pt["gain_floor"],
                         floor_mode=mode, floor_is_own=False, floor_by_group=fd.group, in_summary=fd.group not in summ,
                         gate="gain_real - gain_floor significantly > 0 (site bootstrap, BH; v2_3)",
                         metric_kind=fd.kind)
        if not same:
            st = r.store()
            miss = sum(1 for q in freqs if not st.has(q))
            if miss:
                return N.invalid(self.id, name, "ratio_floored", f"floor images not generated ({miss}/{len(freqs)} "
                                 f"missing, mode {mode})", status="PENDING", n_missing=miss,
                                 in_summary=fd.group not in summ, floor_mode=mode)
            p_fl = rd["full"].predict(np.concatenate([self.img_gen(name, freqs), C, O], 1))
        else:
            p_fl = p_gen

        def st_(w=None):
            ww = None if w is None else N.site_weights_to_items(w, sidx)
            sb = score(fd.kind, p_base, y, ww)
            gr = score(fd.kind, p_real, y, ww) - sb
            gg = score(fd.kind, p_gen, y, ww) - sb
            gf = score(fd.kind, p_fl, y, ww) - sb
            den = gr - gf
            return dict(gain_floor=gf, ratio=(gg - gf) / den if abs(den) > 1e-12 else np.nan)
        pt = st_()
        bs = N.site_bootstrap(st_, sites, self.run.B, self.seed("bootfl", name))
        sb = score(fd.kind, p_base, y)
        gr, gg = score(fd.kind, p_real, y) - sb, score(fd.kind, p_gen, y) - sb
        return N.row(self.id, name, "ratio_floored", raw=gg, floor=pt["gain_floor"], ceil=gr,
                     s_hat=pt["ratio"], ci=N.percentile_ci(bs["ratio"]), n=len(tt), seeds=1, p=p_gate,
                     space="+".join(rd["spaces"]), gain_real=gr, gain_gen=gg, gain_floor=pt["gain_floor"],
                     floor_mode=mode, floor_is_own=bool(same), in_summary=fd.group not in summ,
                     gate="gain_real CI excludes 0", metric_kind=fd.kind)

    def finalize(self, rows):
        q = self.run.q
        for x in rows:
            if x["measure"] == self.id and x["status"] == "OK":
                if not (np.isfinite(x["p_bh"]) and x["p_bh"] <= q):
                    x["extra"]["s_hat_ungated"] = x["s_hat"]
                    x["status"], x["s_hat"] = "INVALID", float("nan")
                    x["reason"] = ("gate: gain_real - gain_floor not significantly above 0 after BH"
                                   if x["extra"].get("floor_by_group") else
                                   "gate: gain_real not significantly above 0 after BH")
        return rows
