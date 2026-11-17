"""
C4 CONDITIONAL DISTRIBUTION FIDELITY (spec section 2), per enabled non-location field whose regression
field (own, or the joint group holding it) is valid after selection (Q15), in that field's selected space.

  Bins over the n_test (2000) shared TEST records by their own value of f: 5 quantile bins (continuous),
  4 equal arcs of the year starting at day 0 (doy; equal arcs of the period for other circular fields),
  top classes with >= min_class (100) records (categorical).
  Per bin b, unbiased MMD^2 with an RBF kernel k(x, y) = exp(-||x - y||^2 / (2 sigma^2)), sigma = median
  pairwise distance of the real TEST features (on up to 2000 of them):
    model = MMD^2(generated bin b, real bin b)
    floor = mean over b' != b of MMD^2(generated bin b, real bin b')
    ceil  = MMD^2(real bin b half 1, real bin b half 2)   (halves split by site, seeded)
  s_hat_b = (floor - model) / (floor - ceil); field score = record-weighted mean of s_hat_b.
  Generated images = G(r, own metadata, 1 seed) for the same records (shared with C3).
  CI: site bootstrap of the field score (kernel matrices computed once, resampled by site weight).
  FIX 11 Nov 2026: the bootstrap used to index the kernel blocks with repeated records, so a record drawn twice
  met its own copy as an off-diagonal pair with k = 1. That inflates the within-set terms of every MMD^2, most
  of all the small real-vs-real ceiling, and pushed the replicates above the point estimate (gsd 0.703 with a CI
  of [0.757, 0.836]). The U-statistics are now weighted by multiplicity and exclude same-record pairs, and the
  own (generated_i, real_i) pairs of the model term keep their 1 / n share (see wmmd).
  v2_3 gate: the field is INVALID when the bin-weighted mean of (floor - ceil) is not significantly > 0.
"""
from __future__ import annotations

import numpy as np

from metacf.v2.core import norm as N
from metacf.v2.measures.base import Measure
from metacf.v2.measures.c2_transfer import test_tiles
from metacf.v2.measures.c3_information import own_requests


def sqdist(A, B):
    a2 = (A * A).sum(1)[:, None]
    b2 = (B * B).sum(1)[None, :]
    return np.maximum(a2 + b2 - 2 * A @ B.T, 0.0)


def mmd2_u(Kxx, Kyy, Kxy):
    """Unbiased MMD^2 from kernel blocks (diagonals of Kxx, Kyy excluded)."""
    m, n = Kxx.shape[0], Kyy.shape[0]
    if m < 2 or n < 2:
        return float("nan")
    sx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    sy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    return float(sx + sy - 2 * Kxy.mean())


def mmd2(X, Y, sigma):
    g = 1.0 / (2 * sigma * sigma)
    return mmd2_u(np.exp(-g * sqdist(X, X)), np.exp(-g * sqdist(Y, Y)), np.exp(-g * sqdist(X, Y)))


def median_bandwidth(X, n_max=2000, seed=0):
    X = np.asarray(X, float)
    if len(X) > n_max:
        X = X[np.random.default_rng(seed).choice(len(X), n_max, replace=False)]
    D = np.sqrt(sqdist(X, X))
    iu = np.triu_indices(len(X), 1)
    v = D[iu]
    return float(np.median(v[v > 0])) if (v > 0).any() else 1.0


def bins_for(fd, values, min_class=100, n_q=5, n_arcs=4):
    v = np.array([np.nan if x is None else float(x) for x in values], float)
    lab = np.full(len(v), -1)
    if fd.kind == "categorical":
        cls, cnt = np.unique(v[np.isfinite(v)].astype(int), return_counts=True)
        keep = [c for c, n in zip(cls, cnt) if n >= min_class]
        for k, c in enumerate(keep):
            lab[np.isfinite(v) & (v == c)] = k
        return lab, [f"class={c}" for c in keep]
    if fd.kind == "circular":
        per = fd.period or 365.25
        lab = np.where(np.isfinite(v), np.floor((v % per) / (per / n_arcs)).astype(int), -1)
        lab = np.minimum(lab, n_arcs - 1)
        return lab, [f"arc{k}" for k in range(n_arcs)]
    z = fd.columns(values)[:, 0]
    ok = np.isfinite(z)
    q = np.quantile(z[ok], np.linspace(0, 1, n_q + 1)) if ok.any() else np.zeros(n_q + 1)
    lab[ok] = np.clip(np.searchsorted(q[1:-1], z[ok], side="right"), 0, n_q - 1)
    return lab, [f"q{k}" for k in range(n_q)]


class C4Distribution(Measure):
    id = "C4"
    name = "conditional distribution fidelity"

    def fields(self):
        r = self.run
        return [n for n, fd in r.fields.items() if fd.enabled and not fd.location and r.conditioned(n)]

    def plan(self):
        return list(own_requests(self.run))

    def compute(self):
        r = self.run
        out = []
        tt = test_tiles(r)
        reqs = own_requests(r)
        sites = np.array([x.location_id for x in tt])
        for name in self.fields():
            ok, why = r.valid(name)
            if not ok:
                out.append(N.invalid(self.id, name, "score", f"INVALID after feature-space selection: {why}"))
                continue
            fd = r.fields[name]
            sp = r.space_of(name)
            R = r.real_feats(sp, [x.rid for x in tt])
            G = r.gen_feats(sp, reqs)
            sigma = median_bandwidth(R, seed=self.seed("bw", name))
            gam = 1.0 / (2 * sigma * sigma)
            Krr = np.exp(-gam * sqdist(R, R))
            Kgg = np.exp(-gam * sqdist(G, G))
            Kgr = np.exp(-gam * sqdist(G, R))
            lab, names = bins_for(fd, [fd.raw(x.fields) for x in tt], int(self.c("min_class", 100)),
                                  int(self.c("n_quantile_bins", 5)), int(self.c("n_arcs", 4)))
            nb = len(names)
            if nb < 2:
                out.append(N.invalid(self.id, name, "score", f"fewer than 2 bins ({nb})"))
                continue
            rng = np.random.default_rng(self.seed("halves", name))
            usite = np.unique(sites)
            half_of_site = dict(zip(usite, rng.integers(0, 2, len(usite))))
            half = np.array([half_of_site[s] for s in sites])

            okrec = lab >= 0

            def wmmd(Kxx, Kyy, Kxy, wx, wy, paired=False):
                """Unbiased MMD^2 with integer item weights: pairs of the SAME item (a bootstrap duplicate)
                are excluded like the diagonal. With unit weights this is mmd2_u.
                paired=True (x_i is the image generated FROM record y_i): the own pairs (i, i) of the cross
                term keep their natural 1 / n share. A plain weighted mean would give them sum(w^2) / sum(w)^2
                (about twice 1 / n under resampling), and since an image is closer to its own real chip than to
                the others, that alone pulls every replicate's model MMD down (score up)."""
                sx, sy = wx.sum(), wy.sum()
                dx, dy = sx * sx - (wx * wx).sum(), sy * sy - (wy * wy).sum()
                if dx <= 0 or dy <= 0:
                    return float("nan")
                a_ = (wx @ Kxx @ wx - (wx * wx * np.diag(Kxx)).sum()) / dx
                b_ = (wy @ Kyy @ wy - (wy * wy * np.diag(Kyy)).sum()) / dy
                if paired:
                    dg = np.diag(Kxy)
                    off = (wx @ Kxy @ wy - (wx * wy * dg).sum()) / (sx * sy - (wx * wy).sum())
                    own = (wx * dg).sum() / sx
                    c_ = off * (1 - 1 / sx) + own / sx
                else:
                    c_ = (wx @ Kxy @ wy) / (sx * sy)
                return float(a_ + b_ - 2 * c_)

            def score_w(w):
                """w: per-record multiplicity (0 = absent) -> (field score, per-bin list, weighted mean floor - ceil)."""
                per = []
                for b in range(nb):
                    ib = np.where(okrec & (lab == b) & (w > 0))[0]
                    if len(ib) < 4:
                        per.append((np.nan, 0, np.nan, np.nan, np.nan))
                        continue
                    wb = w[ib]
                    model = wmmd(Kgg[np.ix_(ib, ib)], Krr[np.ix_(ib, ib)], Kgr[np.ix_(ib, ib)], wb, wb, paired=True)
                    fl = []
                    for b2 in range(nb):
                        if b2 == b:
                            continue
                        jb = np.where(okrec & (lab == b2) & (w > 0))[0]
                        if len(jb) >= 2:
                            fl.append(wmmd(Kgg[np.ix_(ib, ib)], Krr[np.ix_(jb, jb)], Kgr[np.ix_(ib, jb)], wb, w[jb]))
                    floor = float(np.nanmean(fl)) if fl else np.nan
                    h1, h2 = ib[half[ib] == 0], ib[half[ib] == 1]
                    ceil = wmmd(Krr[np.ix_(h1, h1)], Krr[np.ix_(h2, h2)], Krr[np.ix_(h1, h2)], w[h1], w[h2]) \
                        if len(h1) >= 2 and len(h2) >= 2 else np.nan
                    sh, _ = N.normalise(model, floor, ceil)
                    per.append((sh, float(wb.sum()), model, floor, ceil))
                wt = np.array([p[1] for p in per], float)
                sv = np.array([p[0] for p in per], float)
                dn = np.array([p[3] - p[4] for p in per], float)
                okb = np.isfinite(sv) & (wt > 0)
                return ((float((wt[okb] * sv[okb]).sum() / wt[okb].sum()) if okb.any() else np.nan), per,
                        (float((wt[okb] * dn[okb]).sum() / wt[okb].sum()) if okb.any() else np.nan))
            all_idx = np.where(okrec)[0]
            pt, per, den = score_w(okrec.astype(float))
            _, s_inv = np.unique(sites, return_inverse=True)
            reps, dens = [], []
            brng = np.random.default_rng(self.seed("boot", name))
            for _ in range(int(self.c("B", self.run.B))):
                sw = np.bincount(brng.integers(0, len(usite), len(usite)), minlength=len(usite)).astype(float)
                sc_, _, dn_ = score_w(sw[s_inv] * okrec)
                reps.append(sc_)
                dens.append(dn_)
            out.append(N.row(self.id, name, "score", raw=pt, s_hat=pt, ci=N.percentile_ci(reps), n=len(all_idx),
                             seeds=1, space=sp, sigma=sigma, p=N.boot_p_greater(dens), floor_minus_ceil=den,
                             floor_minus_ceil_ci=N.percentile_ci(dens),
                             gate="mean (floor - ceil) over bins significantly > 0 (site bootstrap, BH; v2_3)",
                             bins=[dict(bin=names[b], s_hat=per[b][0], n=per[b][1], model=per[b][2],
                                        floor=per[b][3], ceil=per[b][4]) for b in range(nb)]))
        return out

    def finalize(self, rows):
        if not self.run.cfg.get("v2_3"):
            return rows
        q = self.run.q
        for x in rows:
            if x["measure"] == self.id and x["metric"] == "score" and x["status"] == "OK":
                if not (np.isfinite(x["p_bh"]) and x["p_bh"] <= q):
                    x["extra"]["s_hat_ungated"] = x["s_hat"]
                    x["status"], x["s_hat"] = "INVALID", float("nan")
                    x["reason"] = "gate: floor - ceil not significantly above 0 after BH (bins not separable in reals)"
        return rows
