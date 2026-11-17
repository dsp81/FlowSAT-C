"""
T17 CLOUD OCCLUSION (tag cloud). Sweep cloud over {0, 0.15, 0.30, 0.50, 0.70} (clipped to support),
the same seed along each anchor's sweep, 20 anchors x 3 seeds, RAW.
Cloud mask M_v: luminance > 80th pct AND saturation < 30th pct AND local sd (7 px) < its 30th pct,
dilated 5 px.
Outside-mask content distance d_out(v) = mean over DINOv2 patch tokens covering < 10% mask of
(1 - cos) between the image at v and the image at cloud = 0 (same anchor and seed).
Floor d_seed = the same patch distance between two seeds at cloud = 0.
score(v) = 1 - d_out(v) / d_seed; the curve and its mean over v > 0 are reported (raw = the mean;
floor 0 = as different as another seed, ceiling 1 = untouched outside the cloud).
Decomposition: global contrast loss = 1 - std(L outside M_v) / std(L at v = 0); local coverage =
mask fraction. Validation: on real TEST, mask fraction vs cloud metadata Spearman >= 0.3, else INVALID.
"""
from __future__ import annotations

import numpy as np

from metacf import physical, stats
from metacf.base import MetaCFTest, na, row
from metacf.tier1.common import Sweep

CLOUD = [0.0, 0.15, 0.30, 0.50, 0.70]


def patch_cover(mask, g):
    h, w = mask.shape
    ph, pw = h // g, w // g
    return mask[: ph * g, : pw * g].reshape(g, ph, g, pw).mean((1, 3))


def patch_dist(P, Q, keep=None):
    a = P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-12)
    b = Q / np.maximum(np.linalg.norm(Q, axis=-1, keepdims=True), 1e-12)
    d = 1 - (a * b).sum(-1)
    return float(d[keep].mean()) if keep is not None and keep.any() else (float(d.mean()) if keep is None else float("nan"))


class T17Cloud(MetaCFTest):
    id = "T17"
    name = "cloud occlusion"
    tier = 2
    required_tags = {"cloud"}

    def sweep(self, ctx, f):
        grid = ctx.support(f).clip_grid(self.tc.get("values", CLOUD))
        if grid and grid[0] != 0.0 and 0.0 in self.tc.get("values", CLOUD):
            grid = [0.0] + grid                       # the v = 0 reference is required
        anchors = ctx.anchors(f, grid, n=int(self.tc.get("n_anchors", 20)), tag="T17")
        return Sweep(ctx, f, grid, anchors, ctx.anchor_seeds(), "RAW")

    def plan(self, ctx):
        return [r for f in self.fields(ctx) for r in self.sweep(ctx, f).requests()]

    def validate_real(self, ctx):
        out = {}
        for f in self.fields(ctx):
            def comp(f=f):
                spec = ctx.specs[f]
                recs = [r for r in ctx.test if spec.raw(r.fields) is not None][: int(self.tc.get("n_real", 3000))]
                frac = [float(physical.cloud_mask(ctx.data.real_image(r.rid)).mean()) for r in recs]
                return stats.spearman([float(spec.raw(r.fields)) for r in recs], frac), len(recs)
            rho, n = ctx.cached(("T17real", f), comp)
            ok = np.isfinite(rho) and rho >= float(self.tc.get("min_spearman", 0.3))
            out[f] = dict(status="VALID" if ok else "INVALID", spearman=rho, n=n)
        return out

    def compute(self, ctx, store):
        out = self.missing_tag_na(ctx)
        val = self.validate_real(ctx)
        for f in self.fields(ctx):
            if val[f]["status"] != "VALID":
                out.append(na(self.id, f, f"cloud mask INVALID on reals: {val[f]}", status="INVALID"))
                continue
            sw = self.sweep(ctx, f)
            if len(sw.anchors) < int(self.tc.get("min_n", 5)) or len(sw.grid) < 2 or len(sw.seeds) < 2:
                out.append(na(self.id, f, f"{len(sw.anchors)} anchors / {len(sw.grid)} values / {len(sw.seeds)} seeds"))
                continue
            imgs, (A, K, S) = sw.images(store)
            flat = [imgs[a][k][s] for a in range(A) for k in range(K) for s in range(S)]
            P = ctx.phi.patch_features(flat).reshape(A, K, S, *ctx.phi.patch_features(flat[:1]).shape[1:])
            g = P.shape[3]
            thr = float(self.tc.get("max_patch_cover", 0.10))
            score = np.full((A, S, K), np.nan)
            contrast = np.full((A, S, K), np.nan)
            cover = np.full((A, S, K), np.nan)
            d_seed = np.mean([patch_dist(P[a, 0, s], P[a, 0, (s + 1) % S]) for a in range(A) for s in range(S)])
            for a in range(A):
                for s in range(S):
                    L0 = physical.luminance(imgs[a][0][s])
                    for k in range(K):
                        M = physical.cloud_mask(imgs[a][k][s])
                        keep = patch_cover(M, g) < thr
                        score[a, s, k] = 1 - patch_dist(P[a, k, s], P[a, 0, s], keep) / d_seed
                        Lk = physical.luminance(imgs[a][k][s])
                        contrast[a, s, k] = 1 - Lk[~M].std() / L0.std() if (~M).any() and L0.std() > 0 else np.nan
                        cover[a, s, k] = M.mean()
            curve = np.nanmean(score, (0, 1))
            mean_pos = float(np.nanmean(curve[1:]))
            per_anchor = np.nanmean(score[:, :, 1:], (1, 2))
            ci = stats.bootstrap(lambda i: np.nanmean(per_anchor[i]), A, self.B, self.boot_seed(ctx, f))
            out.append(row(self.id, f, "phi-patches", raw=mean_pos, floor=0.0, ceiling=1.0, ci=ci, n=A, seeds=S,
                           grid=sw.grid, score_curve=curve.tolist(), d_seed=float(d_seed),
                           contrast_loss_curve=np.nanmean(contrast, (0, 1)).tolist(),
                           coverage_curve=np.nanmean(cover, (0, 1)).tolist(), real_validation=val[f]))
        return out
