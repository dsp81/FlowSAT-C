"""
T6 SPECIFICITY.
T6a excess entanglement. Read-backs standardised by their real-TEST standard deviation, fields by
    their TRAIN standard deviation. For swept field i and every other read field j:
      gamma_ij = median over anchors x seeds of the slope of standardised read-back_j on
                 standardised v_i (RAW sweeps)
      beta_ij  = the real partial coefficient of d v_i in the T4-style no-intercept regression of
                 d read-back_j on all field differences dF (same-location TRAIN pairs)
      E_ij = |gamma_ij - beta_ij|, leakage l_ij = min(1, E_ij / max(|beta_ii|, eps)),
      Specificity_i = 1 - mean_{j != i} l_ij.  N/A for field i when |beta_ii| < eps.
    The full E and beta matrices are reported. Categorical fields are not part of the matrix.
T6b content preservation. sim_gen = mean cos(phi(G(a, f<-v_k, s)), phi(G(a, f<-v_mid, s))), k != mid;
    sim_real = the same for natural pairs whose |d f| best matches each sweep step (nearest
    neighbour on |d f|); sim_unrel = mean cos between different anchors at the same seed (v_mid).
    score = (sim_gen - sim_unrel) / (sim_real - sim_unrel); > 1 preserves more than reality.
"""
from __future__ import annotations

import itertools

import numpy as np

from metacf import stats
from metacf.base import MetaCFTest, na, row
from metacf.fields import location_groups
from metacf.tier1.common import centred, standard_sweep
from metacf.tier1.t4_calibration import pair_design, partial_coef, regression_fields


def all_location_pairs(ctx, cap=50):
    rng = np.random.default_rng(ctx.seed("T6pairs"))
    out = []
    for loc, rs in sorted(location_groups(ctx.train).items()):
        c = [(rs[a].rid, rs[b].rid) for a in range(len(rs)) for b in range(a + 1, len(rs))]
        if len(c) > cap:
            c = [c[i] for i in sorted(rng.choice(len(c), cap, replace=False))]
        out += c
    return out


class T6Specificity(MetaCFTest):
    id = "T6"
    name = "specificity"

    def matrix_fields(self, ctx):
        return [f for f in self.swept(ctx) if ctx.specs[f].kind in ("continuous", "circular")]

    def side_fields(self, ctx):
        """Categorical read-backs reported as side-effect columns (incl. recovery-only fields)."""
        return [f for f in self.fields(ctx) if ctx.specs[f].kind == "categorical"]

    def plan(self, ctx):
        return [r for f in self.swept(ctx) for r in standard_sweep(ctx, f).requests()]

    def side_effects(self, ctx, store, rd, mf, pairs, D, reg_fields):
        """
        Side-effect columns for categorical read-backs j (spec: Koppen / land cover are
        recovery-only and appear in T6a as side effects). Units differ from the continuous matrix, so
        they are reported next to it and NOT averaged into Specificity_i:
          gamma_ij = median over anchors x seeds of the slope (with intercept) of
                     1[class_j(v_k) != class_j(v_mid)] on |standardised v_k - v_mid|
          beta_ij  = over same-location TRAIN pairs, the coefficient of |standardised d v_i| in the OLS
                     (with intercept) of 1[class_j(x_a) != class_j(x_b)] on |standardised dF|
          E_ij = |gamma_ij - beta_ij|
        """
        out = {}
        Dz = np.abs(D / np.array([ctx.train_field_std(g) or 1.0 for g in reg_fields])) if D is not None else None
        for j in self.side_fields(ctx):
            if rd not in ctx.readers.valid_readers(j):
                continue
            beta = {}
            if Dz is not None:
                a = ctx.train_readback(j, rd, [p[0] for p in pairs])
                b = ctx.train_readback(j, rd, [p[1] for p in pairs])
                y = (a != b).astype(float)
                res = stats.ols(Dz, y, intercept=True)
                beta = {g: float(c) for g, c in zip(reg_fields, res["coef"])}
            for i in mf:
                sw = standard_sweep(ctx, i)
                if not sw.anchors:
                    continue
                imgs, idx = sw.flat_images(store)
                pj = ctx.readback(j, rd, imgs)
                P = np.full((len(sw.anchors), len(sw.grid), len(sw.seeds)), np.nan)
                for (ai, k, si), p in zip(idx, pj):
                    P[ai, k, si] = p
                mid = len(sw.grid) // 2
                v = centred(ctx.specs[i].kind, sw.set_values) / (ctx.train_field_std(i) or 1.0)
                x = np.abs(v - v[mid])
                sl = [stats.slope(x, (P[a, :, s] != P[a, mid, s]).astype(float))
                      for a in range(P.shape[0]) for s in range(P.shape[2])]
                g = stats.median_ignore_nan(sl)
                bb = beta.get(i, float("nan"))
                out.setdefault(i, {})[j] = dict(gamma=g, beta=bb, E=abs(g - bb) if np.isfinite(bb) else float("nan"))
        return out

    def compute(self, ctx, store):
        out = self.na_unsupported(ctx)
        out += [na(self.id, f, "recovery-only by design: no sweep (appears as a T6a side-effect column)",
                   stratum="T6b") for f in self.fields(ctx) if ctx.specs[f].recovery_only]
        eps = float(self.tc.get("eps", 0.05))
        mf = [f for f in self.matrix_fields(ctx) if ctx.readers.valid_readers(f)]
        readers = sorted(set.intersection(*[set(ctx.readers.valid_readers(f)) for f in mf])) if mf else []
        reg_fields = regression_fields(ctx)
        pairs = all_location_pairs(ctx)
        D = pair_design(ctx, pairs, reg_fields) if pairs else None
        for rd in readers:
            n = len(mf)
            G, Bm = np.full((n, n), np.nan), np.full((n, n), np.nan)
            for jj, j in enumerate(mf):
                sd_j = ctx.real_readback_std(j, rd)
                if D is not None:
                    a = ctx.train_readback(j, rd, [p[0] for p in pairs])
                    b = ctx.train_readback(j, rd, [p[1] for p in pairs])
                    y = (stats.wrap(b - a) if ctx.specs[j].kind == "circular" else b - a) / sd_j
                    Dz = D / np.array([ctx.train_field_std(g) or 1.0 for g in reg_fields])
                    for ii, i in enumerate(mf):
                        if i in reg_fields:
                            Bm[ii, jj] = partial_coef(y, Dz, reg_fields.index(i))
                for ii, i in enumerate(mf):
                    sw = standard_sweep(ctx, i)
                    if not sw.anchors:
                        continue
                    imgs, idx = sw.flat_images(store)
                    rbj = ctx.sweep_readback(j, rd, imgs, [sw.anchors[t[0]] for t in idx])
                    R = np.full((len(sw.anchors), len(sw.grid), len(sw.seeds)), np.nan)
                    for (ai, k, si), p in zip(idx, rbj):
                        R[ai, k, si] = p
                    v = centred(ctx.specs[i].kind, sw.set_values) / (ctx.train_field_std(i) or 1.0)
                    sl = [stats.slope(v, centred(ctx.specs[j].kind, R[a, :, s]) / sd_j)
                          for a in range(R.shape[0]) for s in range(R.shape[2])]
                    G[ii, jj] = stats.median_ignore_nan(sl)
            E = np.abs(G - Bm)
            side = self.side_effects(ctx, store, rd, mf, pairs, D, reg_fields)
            for ii, i in enumerate(mf):
                bii = Bm[ii, ii]
                if not np.isfinite(bii) or abs(bii) < eps:
                    out.append(na(self.id, i, f"|beta_ii| = {bii:.3g} < eps {eps}: no measurable real effect",
                                  reader=rd, stratum="T6a"))
                    continue
                ell = [min(1.0, E[ii, jj] / max(abs(bii), eps)) for jj in range(n) if jj != ii and np.isfinite(E[ii, jj])]
                spec_i = 1 - float(np.mean(ell)) if ell else float("nan")
                out.append(row(self.id, i, rd, stratum="T6a", raw=spec_i, floor=0.0, ceiling=1.0,
                               n=len(ell), leakage={mf[jj]: float(min(1.0, E[ii, jj] / max(abs(bii), eps)))
                                                     for jj in range(n) if jj != ii and np.isfinite(E[ii, jj])},
                               side_effects=side.get(i, {})))
            out.append(row(self.id, "*", rd, stratum="T6a-matrix", descriptive=True, fields=mf,
                           E=E.tolist(), beta=Bm.tolist(), gamma=G.tolist(), side_effects=side))
        for f in self.swept(ctx):
            out.append(self._t6b(ctx, store, f))
        return out

    def _t6b(self, ctx, store, f):
        sw = standard_sweep(ctx, f)
        if len(sw.anchors) < 2:
            return na(self.id, f, "fewer than 2 anchors", stratum="T6b")
        P = sw.phi(store)                                     # (A, K, S, D)
        A, K, S, _ = P.shape
        mid = K // 2
        sim_gen = float(np.mean([P[a, k, s] @ P[a, mid, s] for a in range(A) for s in range(S)
                                 for k in range(K) if k != mid]))
        sim_unrel = float(np.mean([P[a, mid, s] @ P[b, mid, s] for s in range(S)
                                   for a, b in itertools.combinations(range(A), 2)]))
        spec = ctx.sweep_spec(f)
        steps = [abs(float(spec.diff(sw.grid[k], sw.grid[mid]))) for k in range(K) if k != mid]
        pairs = ctx.natural_pairs(f) if ctx.specs[f].kind != "geo" else []
        if not pairs:
            return row(self.id, f, "phi", stratum="T6b", raw=sim_gen, floor=sim_unrel,
                       notes="ceiling N/A: no natural pairs", sim_gen=sim_gen, sim_unrel=sim_unrel)
        dd = np.array([abs(float(spec.diff(spec.raw(ctx.data.by_id[b].fields), spec.raw(ctx.data.by_id[a].fields))))
                       for a, b in pairs])
        chosen = [pairs[int(np.argmin(np.abs(dd - st)))] for st in steps]
        xa = ctx.phi_feats(ctx.real_images([p[0] for p in chosen]))
        xb = ctx.phi_feats(ctx.real_images([p[1] for p in chosen]))
        sim_real = float(np.mean((xa * xb).sum(1)))
        if abs(sim_real - sim_unrel) < float(self.tc.get("min_gap", 0.01)):
            return row(self.id, f, "phi", stratum="T6b", raw=sim_gen, floor=sim_unrel,
                       notes=f"ceiling N/A: sim_real {sim_real:.4f} ~ sim_unrel {sim_unrel:.4f} "
                             "(same-location reals are no more alike than unrelated ones)",
                       sim_gen=sim_gen, sim_real=sim_real, sim_unrel=sim_unrel)
        return row(self.id, f, "phi", stratum="T6b", raw=sim_gen, floor=sim_unrel, ceiling=sim_real,
                   n=A, seeds=S, sim_gen=sim_gen, sim_real=sim_real, sim_unrel=sim_unrel,
                   interpretation="norm > 1: preserves the scene more than reality; < 1: rewrites it")
