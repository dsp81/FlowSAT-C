"""
T12 SINGLE-SUN CONSISTENCY (no tag). Per image: dark mask D = luminance below its 20th percentile AND
saturation below its median; bright mask B = luminance above its 70th percentile. 4 x 4 tiles (128 px
at 512); per tile, the B -> D cross-correlation peak over offsets of radius 3-25 px gives the
shadow direction theta_t; a tile is kept when its peak is >= 1.5 x the median correlation over
offsets; an image is kept with >= 6 tiles.
Within-image circular variance V = 1 - |mean(e^{i theta_t})|.
ceiling = mean V on real TEST images; floor = mean V on real TEST after rotating each tile by an
independent uniform random multiple of 90 deg (simulated multiple suns).
score = (V_floor - V_gen) / (V_floor - V_real). Validation: V_real <= 0.6 V_floor. Min-n: 200 images
passing on each side. Uses the T1 generations (seed 0); no extra generation.
"""
from __future__ import annotations

import numpy as np

from metacf import physical, stats
from metacf.base import MetaCFTest, na, row
from metacf.tier1.t1_recovery import T1Recovery


def image_V(img, grid=4, min_tiles=6, **kw):
    th = physical.shadow_directions(img, grid=grid, **kw)
    return stats.circ_var(th) if len(th) >= min_tiles else None


class T12SingleSun(MetaCFTest):
    id = "T12"
    name = "single-sun consistency"
    tier = 2

    def plan(self, ctx):
        return T1Recovery(self.cfg).plan(ctx)

    def _kw(self):
        return dict(grid=int(self.tc.get("grid", 4)), min_tiles=int(self.tc.get("min_tiles", 6)),
                    rmin=int(self.tc.get("rmin", 3)), rmax=int(self.tc.get("rmax", 25)),
                    ratio=float(self.tc.get("peak_ratio", 1.5)))

    def real_values(self, ctx):
        def comp():
            t1 = T1Recovery(self.cfg)
            rids = t1.records(ctx)
            kw = self._kw()
            vr, vf = [], []
            for i, r in enumerate(rids):
                im = ctx.data.real_image(r)
                vr.append(image_V(im, **kw))
                vf.append(image_V(physical.rotate_tiles(im, kw["grid"], ctx.seed("T12rot", i)), **kw))
            return [v for v in vr if v is not None], [v for v in vf if v is not None]
        return ctx.cached(("T12real",), comp)

    def validate_real(self, ctx):
        vr, vf = self.real_values(ctx)
        min_n = int(self.tc.get("min_n", 200))
        if len(vr) < min_n or len(vf) < min_n:
            return {"*": dict(status="N/A", detail=f"{len(vr)} / {len(vf)} real images pass < {min_n}")}
        ok = np.mean(vr) <= 0.6 * np.mean(vf)
        return {"*": dict(status="VALID" if ok else "INVALID", V_real=float(np.mean(vr)),
                          V_floor=float(np.mean(vf)))}

    def compute(self, ctx, store):
        val = self.validate_real(ctx)["*"]
        if val["status"] != "VALID":
            return [na(self.id, "*", f"real-data validation: {val}", status=val["status"])]
        t1 = T1Recovery(self.cfg)
        own = t1.own_reqs(ctx)
        kw = self._kw()
        vg = [image_V(store.get(own[(r, 0)]), **kw) for r in t1.records(ctx)]
        vg = np.array([v for v in vg if v is not None], float)
        if len(vg) < int(self.tc.get("min_n", 200)):
            return [na(self.id, "*", f"{len(vg)} generated images pass the tile filter")]
        vr, vf = self.real_values(ctx)
        Vr, Vf = float(np.mean(vr)), float(np.mean(vf))
        ci = stats.bootstrap(lambda i: (Vf - vg[i].mean()) / (Vf - Vr), len(vg), self.B, self.boot_seed(ctx))
        return [row(self.id, "*", "physical", raw=float(vg.mean()), floor=Vf, ceiling=Vr, ci=ci, n=len(vg),
                    V_gen=float(vg.mean()), V_real=Vr, V_floor=Vf, validation="VALID")]
