"""
E4 YEAR (spec section 3).

  Order model: logistic regression WITHOUT intercept on dphi = phi(x2) - phi(x1) (space: e4.space, default
  the default space = DINOv2), predicting year2 > year1, trained on TRAIN real same-site pairs whose years
  differ (both orders of every pair). P(x2 later than x1) = sigmoid(w . dphi), so
  P(a later than b) + P(b later than a) = sigmoid(z) + sigmoid(-z) = 1 by construction (antisymmetric).
  L2 strength by grouped CV over sites.
  Real ceiling: accuracy on TEST real pairs with different years (both orders).
  Generated: n_anchors (50) TEST anchors x n_years (4) distinct years evenly spread in the anchor category's
    TRAIN support x 1 seed; accuracy over all ordered pairs within an anchor.
  Score = (acc_gen - 0.5) / (acc_real - 0.5). Gate: acc_real >= 0.6.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from metacf.readers import location_folds
from metacf.v2.core import norm as N
from metacf.v2.measures.base import Measure, pick_by_category_region


class OrderModel:
    """Binary logistic regression with no intercept on standardised-by-scale features (no centring, so
    the model stays odd in dphi)."""

    def __init__(self, lambdas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), folds=5):
        self.lambdas, self.k = list(lambdas), folds

    @staticmethod
    def _fit(X, y, lam):
        n, d = X.shape

        def f(w):
            z = X @ w
            s = np.where(y > 0, 1.0, -1.0)
            m = -s * z
            loss = np.logaddexp(0, m).mean() + 0.5 * lam * w @ w
            g = X.T @ (-s / (1 + np.exp(-m))) / n + lam * w
            return loss, g
        return minimize(f, np.zeros(d), jac=True, method="L-BFGS-B", options=dict(maxiter=500)).x

    def fit(self, D, y, groups):
        D = np.asarray(D, float)
        self.scale = np.sqrt((D ** 2).mean(0)) + 1e-9          # scale only: keeps P(-d) = 1 - P(d)
        X = D / self.scale
        y = np.asarray(y, int)
        folds = location_folds(groups, self.k)
        best = None
        for lam in self.lambdas:
            acc = []
            for k in range(self.k):
                tr, te = folds != k, folds == k
                if te.sum() == 0 or tr.sum() < 4:
                    continue
                w = self._fit(X[tr], y[tr], lam)
                acc.append(((X[te] @ w > 0) == (y[te] > 0)).mean())
            a = float(np.mean(acc)) if acc else np.nan
            if best is None or (np.isfinite(a) and a > best[0]):
                best = (a, lam)
        self.cv_acc, self.lam = best
        self.w = self._fit(X, y, self.lam)
        return self

    def prob_later(self, D):
        """P(second later than first) for dphi rows D = phi(second) - phi(first)."""
        return 1 / (1 + np.exp(-(np.asarray(D, float) / self.scale) @ self.w))


class E4Year(Measure):
    id = "E4"
    name = "year"

    def enabled(self):
        """v2.1: E4 is DROPPED by default: year ordering of real fMoW chips is barely
        readable (acc_real 0.615 against a 0.6 gate) and depends on the dataset's sensor / processing history,
        not on the model. Kept only when `v2_1.drop_e4: false` (or no v2_1 section: the frozen v2 runs)."""
        v21 = self.run.cfg.get("v2_1")
        if v21 is not None and bool(v21.get("drop_e4", True)):
            return False
        return bool(self.c("enabled", True))

    def field(self):
        for n, fd in self.run.fields.items():
            if fd.tag == "year" and fd.enabled:
                return n
        return None

    def space(self):
        return self.c("space", self.run.cfg.get("default_space", "dino"))

    def cells(self):
        def comp():
            r = self.run
            f = self.field()
            if f is None or not r.conditioned(f):
                return []
            fd = r.fields[f]
            sup = r.support(f)
            elig = [x for x in r.test if r.complete(x)]
            anchors = pick_by_category_region(elig, int(self.c("n_anchors", 50)), self.seed("anchors"))
            K = int(self.c("n_years", 4))
            out = []
            for a in anchors:
                rg = sup.by_cat.get(a.category, sup.global_range)
                ys = sorted(set(int(round(v)) for v in np.linspace(rg[1], rg[2], K)))
                if len(ys) < 2:
                    continue
                sd = r.seed("E4seed", a.rid)
                out.append(dict(anchor=a.rid, site=a.location_id, years=ys,
                                reqs=[r.req(a, fd.set(a.fields, float(y)), seed=sd) for y in ys]))
            return out
        return self.run.cached(("E4", "cells"), comp)

    def plan(self):
        return [q for c in self.cells() for q in c["reqs"]]

    def year_pairs(self, split):
        r = self.run
        fd = r.fields[self.field()]
        ps = r.pairs[split]
        ya = np.array([np.nan if v is None else float(v) for v in ps.raw_a.get(self.field(), [])])
        yb = np.array([np.nan if v is None else float(v) for v in ps.raw_b.get(self.field(), [])])
        if not len(ya):
            ya = np.array([float(fd.raw(r.rec(a).fields)) for a in ps.a])
            yb = np.array([float(fd.raw(r.rec(b).fields)) for b in ps.b])
        m = np.isfinite(ya) & np.isfinite(yb) & (ya != yb)
        idx = np.where(m)[0]
        return [ps.a[i] for i in idx], [ps.b[i] for i in idx], ya[idx], yb[idx], [ps.site[i] for i in idx]

    def fit_model(self):
        r = self.run
        a, b, ya, yb, s = self.year_pairs("train")
        sp = self.space()
        D = r.real_feats(sp, b) - r.real_feats(sp, a)
        y = (yb > ya).astype(int)
        return OrderModel().fit(np.vstack([D, -D]), np.concatenate([y, 1 - y]), s + s)

    def compute(self):
        r = self.run
        f = self.field()
        if f is None or not r.conditioned(f):
            return [N.invalid(self.id, f or "year", "score", "no conditioned year field", status="N/A")]
        om = r.readers().get("E4") or self.fit_model()
        sp = self.space()
        a, b, ya, yb, s = self.year_pairs("test")
        D = r.real_feats(sp, b) - r.real_feats(sp, a)
        correct_real = ((om.prob_later(D) > 0.5) == (yb > ya)).astype(float)
        acc_real = float(correct_real.mean()) if len(correct_real) else np.nan
        cells = self.cells()
        per = []
        for c in cells:
            F = r.gen_feats(sp, c["reqs"])
            ys = c["years"]
            ok = []
            for i in range(len(ys)):
                for j in range(len(ys)):
                    if i != j:
                        ok.append(float((om.prob_later(F[j] - F[i]) > 0.5) == (ys[j] > ys[i])))
            per.append(np.mean(ok))
        per = np.array(per)
        sites = [c["site"] for c in cells]
        _, sidx = np.unique(sites, return_inverse=True)
        _, ridx = np.unique(s, return_inverse=True) if len(s) else (None, np.zeros(0, int))

        def sc(w=None):
            if w is None:
                ag = float(np.mean(per)) if len(per) else np.nan
            else:
                ww = N.site_weights_to_items(w, sidx)
                ag = float(ww @ per / ww.sum()) if ww.sum() > 0 else np.nan
            return dict(score=(ag - 0.5) / (acc_real - 0.5) if np.isfinite(acc_real) and acc_real != 0.5 else np.nan,
                        acc_gen=ag)
        pt = sc()
        bs = N.site_bootstrap(sc, sites, self.run.B, self.seed("boot"))
        row = N.row(self.id, f, "score", raw=pt["acc_gen"], floor=0.5, ceil=acc_real, s_hat=pt["score"],
                    ci=N.percentile_ci(bs["score"]), n=len(cells), seeds=1, space=sp, acc_gen=pt["acc_gen"],
                    acc_real=acc_real, n_real_pairs=int(len(correct_real)), lam=om.lam, cv_acc=om.cv_acc,
                    gate="acc_real >= 0.6")
        if not (np.isfinite(acc_real) and acc_real >= float(self.c("min_acc_real", 0.6))):
            row.update(status="INVALID", reason=f"gate: acc_real {acc_real:.3f} < 0.6", s_hat=float("nan"))
        return [row]
