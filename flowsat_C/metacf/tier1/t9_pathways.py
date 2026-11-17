"""
T9 PATHWAY ATTRIBUTION (only if the adapter exposes pathways).
Generations: n TEST records x 3 seeds x {FULL, and per pathway p: ZERO, MEAN, SHUFFLE}.
  MEAN    = the pathway's input replaced by its mean over 4096 TRAIN records (P1: pooled vector;
            P2: the token set), see adapter.prepare_pathway_means
  SHUFFLE = the pathway's input taken from donor pi(r); the other pathway and the caption stay r's own
Positive control: SHUFFLE of one field group (default geo) inside the OTHER pathway only, i.e. for
pathway p the reference is "geo shuffled inside q != p" (with one pathway: inside p itself).
Metrics per variant: paired distance to FULL (same record and seed; LPIPS when available, else
d_dino -- noted), against FULL seed-to-seed distance and the positive control; recovery drop per
field vs FULL (reader A, T1 metric vs own); source-following under SHUFFLE per field:
(metric vs donor - chance) / (FULL metric vs own - chance).
Derived: share_p = median per-record d(SHUFFLE_p) / d(ZERO_p); domain-adapter flag when
d(ZERO_p) > 3 d(MEAN_p) and d(MEAN_p) < 0.1 x seed-to-seed d.
"""
from __future__ import annotations

import numpy as np

from metacf import metrics
from metacf.base import MetaCFTest, na, row


class T9Pathways(MetaCFTest):
    id = "T9"
    name = "pathway attribution"
    descriptive = True

    def records(self, ctx):
        def pick():
            rng = np.random.default_rng(ctx.seed("T9records"))
            ids = [r.rid for r in ctx.test]
            n = min(int(self.tc.get("n_records", 256)), len(ids))
            return [ids[i] for i in sorted(rng.choice(len(ids), n, replace=False))]
        return ctx.cached(("T9rids",), pick)

    def seeds(self):
        return list(range(int(self.tc.get("seeds", 3))))

    def variants(self, ctx):
        ps = list(ctx.adapter.pathways)
        group = self.tc.get("positive_control_group", "geo")
        out = {"FULL": None}
        for p in ps:
            for m in ("zero", "mean", "shuffle"):
                out[f"{p}:{m}"] = dict(pathway=p, mode=m)
            q = [x for x in ps if x != p]
            q = q[0] if q else p
            out[f"{p}:control"] = dict(pathway=q, mode="shuffle", groups=[group])
        return out

    def reqs(self, ctx):
        from metacf.requests import GenRequest
        rids = self.records(ctx)
        donor = ctx.derangement(rids, "t9")
        out = {}
        for name, v in self.variants(ctx).items():
            for r in rids:
                for s in self.seeds():
                    base = ctx.own_request(r, s)
                    if v is None:
                        out[(name, r, s)] = base
                    else:
                        var = dict(v)
                        if v["mode"] == "shuffle":
                            var["donor"] = donor[r]
                        out[(name, r, s)] = GenRequest.make(r, base.field_dict, base.caption, s,
                                                            mode="own", variant=var)
        return out

    def plan(self, ctx):
        if not ctx.adapter.pathways:
            return []
        return list(self.reqs(ctx).values())

    def dist(self, ctx, xs, ys):
        d = ctx.d_lpips(xs, ys)
        return (d, "lpips") if d is not None else (ctx.d_dino(xs, ys), "d_dino (LPIPS unavailable)")

    def compute(self, ctx, store):
        if not ctx.adapter.pathways:
            return [na(self.id, "*", f"adapter {ctx.adapter.name} exposes no pathways")]
        R = self.reqs(ctx)
        rids, S = self.records(ctx), self.seeds()
        donor = ctx.derangement(rids, "t9")
        img = lambda name: [store.get(R[(name, r, s)]) for r in rids for s in S]
        full = img("FULL")
        # seed-to-seed FULL distance (consecutive seeds)
        xs = [store.get(R[("FULL", r, s)]) for r in rids for s in S[:-1]]
        ys = [store.get(R[("FULL", r, s + 1)]) for r in rids for s in S[:-1]]
        d_seed, dname = self.dist(ctx, xs, ys)
        d_seed = float(np.mean(d_seed))
        out = []
        dvar = {}
        for name in self.variants(ctx):
            if name == "FULL":
                continue
            d, _ = self.dist(ctx, img(name), full)
            dvar[name] = np.asarray(d).reshape(len(rids), len(S))
        fields = [f for f in ctx.eval_fields() if "A" in ctx.readers.valid_readers(f)]
        for p in ctx.adapter.pathways:
            z, m, sh, c = (dvar[f"{p}:{x}"] for x in ("zero", "mean", "shuffle", "control"))
            share = float(np.median(np.median(sh, 1) / np.maximum(np.median(z, 1), 1e-12)))
            flag = bool(z.mean() > 3 * m.mean() and m.mean() < 0.1 * d_seed)
            out.append(row(self.id, "*", dname, stratum=p, raw=share, descriptive=True, n=len(rids),
                           seeds=len(S), d_zero=float(z.mean()), d_mean=float(m.mean()),
                           d_shuffle=float(sh.mean()), d_positive_control=float(c.mean()),
                           d_seed_to_seed=d_seed, share=share, domain_adapter_flag=flag))
            for f in fields:
                kind = ctx.specs[f].kind
                truth = ctx.truth(f, rids)
                dtruth = ctx.truth(f, [donor[r] for r in rids])
                rep = lambda t: np.repeat(t, len(S), axis=0)
                full_s = metrics.field_score(kind, ctx.readback(f, "A", full), rep(truth))
                chance = metrics.chance_score(kind, truth, ctx.specs[f].n_classes)
                res = {}
                for x in ("zero", "mean", "shuffle"):
                    pr = ctx.readback(f, "A", img(f"{p}:{x}"))
                    res[f"drop_{x}"] = full_s - metrics.field_score(kind, pr, rep(truth))
                    if x == "shuffle":
                        den = full_s - chance
                        res["source_following"] = (metrics.field_score(kind, pr, rep(dtruth)) - chance) / den \
                            if den else float("nan")
                out.append(row(self.id, f, "A", stratum=p, raw=res["source_following"], descriptive=True,
                               n=len(rids), seeds=len(S), full_score=full_s, **res))
        return out
