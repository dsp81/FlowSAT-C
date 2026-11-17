"""
encoder_p1pb.py — SatMetadataEncoder for the P1-perblock variant (branch arch/p1-perblock).

A SUBCLASS of models/encoder.py:SatMetadataEncoder, not a copy: the field
encoders, projections, nulls, dropout, interaction stack and AttnPool are
inherited unchanged (same parameter names, so P2 is untouched and checkpoints
line up key for key). Two things differ:

  1. No `to_global` (3073 -> 512). P1 now reads the FULL-WIDTH pooled vector.
     The 512-d bottleneck was a secondary information loss for cloud/doy/year
     (diagnostics/p1/REPORT.md D2) and it sat between the pool and P1 with no
     job the per-block trunk does not already do.
       glob_norm="ln_mag": [LayerNorm(p) ; ||p||/sqrt(d)]  -> d + 1 = 3073
       glob_norm="none":   p (raw pooled vector)           -> d     = 3072

  2. Every Fourier / spherical-harmonic / solar feature is computed in fp32 with
     autocast DISABLED, learned feature parameters (SH per-degree weights, year
     trend, embeddings) upcast to fp32. Under DeepSpeed bf16 the module weights
     are bf16 and the encoder runs under bf16 autocast (transformer.py:148-153);
     the inherited _group_features then evaluates softplus(log_a) and the year
     trend in bf16. Features feed the bf16 per-group projections exactly as
     before (that cast is the P2 token path and is left alone).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from flowsat_C.models.encoder import (GROUPS, EncoderConfig, MetadataSpec,
                                      SatMetadataEncoder, derived_solar, fourier,
                                      real_spherical_harmonics)

GLOB_NORMS = ("ln_mag", "none")


def _f32(md):
    """fp32 copy of every floating tensor in a (nested) metadata dict."""
    if torch.is_tensor(md):
        return md.float() if md.is_floating_point() else md
    if isinstance(md, dict):
        return {k: _f32(v) for k, v in md.items()}
    return md


class SatMetadataEncoderP1PB(SatMetadataEncoder):

    def __init__(self, cfg: EncoderConfig | None = None, spec: MetadataSpec | None = None,
                 glob_norm: str = "ln_mag"):
        super().__init__(cfg, spec)
        assert glob_norm in GLOB_NORMS, glob_norm
        del self.to_global
        self.glob_norm = glob_norm
        d = self.cfg.d_model
        self.d_glob = d + 1 if glob_norm == "ln_mag" else d

    # ------------------------------------------------------------------ features

    def _group_features(self, md: dict, use_year_embedding: bool):
        """Same math as SatMetadataEncoder._group_features (encoder.py:353-407),
        evaluated in fp32 with autocast off. Only the dtype handling differs."""
        dev_type = md["lat_deg"].device.type
        with torch.autocast(dev_type, enabled=False):
            return self._group_features_fp32(_f32(md), use_year_embedding)

    def _group_features_fp32(self, md: dict, use_year_embedding: bool):
        c, s = self.cfg, self.spec
        lat, lon = md["lat_deg"], md["lon_deg"]

        # geo: GeoEncoder.forward with the per-degree weights upcast
        y = real_spherical_harmonics(torch.deg2rad(lat), torch.deg2rad(lon), self.geo.L)
        a = F.softplus(self.geo.log_a.float())[self.geo.degree_index]
        geo = y * a.unsqueeze(0)

        d_alpha, _ = derived_solar(lat, md["doy"])
        alpha = md.get("sun_elev_deg", d_alpha)
        az = torch.deg2rad(md.get("sun_azimuth_deg", torch.zeros_like(lat)))
        az_feat = torch.stack([torch.sin(az), torch.cos(az)], dim=-1)
        if "azimuth_valid" in md:
            az_feat = az_feat * md["azimuth_valid"].view(-1, 1).float()

        lg = torch.log2(md["gsd_m"].clamp(min=1e-6))
        lo, hi = math.log2(s.gsd_min_m), math.log2(s.gsd_max_m)
        u_gsd = ((lg - lo) / (hi - lo)) * 2.0 - 1.0

        return {
            "geo":   geo,
            "cov":   self._cov_fp32(md["cov"]),
            "doy":   self.tmp.doy(md["doy"]),
            "solar": torch.cat([fourier(alpha / 90.0, c.solar_bands), az_feat], dim=-1),
            "year":  self._year_fp32(md["year"], use_year_embedding),
            "gsd":   fourier(u_gsd, c.gsd_bands),
            "cloud": fourier(2.0 * md["cloud_frac"] - 1.0, c.cloud_bands),
            "sensor": self.platform(md["platform"].long()).float(),
        }

    def _cov_fp32(self, cov: dict):
        """CovariateEncoder.forward (encoder.py:205-225) with fp32 embeddings."""
        ce, s = self.cov, self.spec

        def norm_log(x, xmax):
            return (torch.log1p(x.clamp(min=0)) / math.log1p(xmax)) * 2.0 - 1.0

        def signed_log_elev(h, hmax):
            return torch.sign(h) * torch.log1p(h.abs() / 100.0) / math.log1p(hmax / 100.0)

        return torch.cat([
            fourier(signed_log_elev(cov["elevation_m"], s.elev_max_m), ce.nb),
            fourier(norm_log(cov["relief_m"], s.relief_max_m), ce.nb),
            fourier(norm_log(cov["coast_km"], s.coast_max_km), ce.nb),
            fourier(norm_log(cov["pop_density"], s.pop_max), ce.nb),
            ce.koppen(cov["koppen"].long()).float(),
            ce.landcover(cov["landcover_mode"].long()).float(),
            cov["landcover_hist"].float(),
        ], dim=-1)

    def _year_fp32(self, Y, use_embedding: bool = True):
        """TemporalEncoder.year (encoder.py:246-270) with the trend in fp32."""
        t, s = self.tmp, self.spec
        y = (Y.float() - s.year_min) / max(s.year_max - s.year_min, 1)
        e = F.linear(y.unsqueeze(-1), t.trend.weight.float())
        if use_embedding:
            idx = (Y.long() - s.year_min).clamp(0, s.year_max - s.year_min)
            in_range = ((Y >= s.year_min) & (Y <= s.year_max)).long()
            idx = torch.where(in_range.bool(), idx,
                              torch.full_like(idx, s.year_max - s.year_min + 1))
            e = e + t.year_embed(idx).float()
        return torch.cat([e, y.unsqueeze(-1)], dim=-1)

    # ------------------------------------------------------------------ forward

    def forward(self, md: dict, training_dropout: bool = False,
                force_null: list[str] | None = None,
                use_year_embedding: bool = True, return_pre: bool = False):
        """encoder.py:409-457 up to the pool, unchanged. Returns (tokens, glob[, pre])
        with glob fp32, (B, d_glob)."""
        c = self.cfg
        feats = self._group_features(md, use_year_embedding)
        B = md["lat_deg"].shape[0]
        dev = md["lat_deg"].device

        drop_all = (torch.rand(B, device=dev) < c.all_dropout if training_dropout
                    else torch.zeros(B, dtype=torch.bool, device=dev))
        toks = []
        for i, g in enumerate(GROUPS):
            t = self.proj[g](feats[g].to(self.proj[g].weight.dtype)) + self.type_embed[i]
            drop = drop_all.clone()
            if training_dropout:
                drop |= torch.rand(B, device=dev) < c.group_dropout[g]
            if g == "cov" and "cov_valid" in md:
                drop |= ~md["cov_valid"].to(dev).bool()
            if force_null and g in force_null:
                drop |= True
            t = torch.where(drop.view(B, 1), self.null[i].unsqueeze(0).to(t.dtype), t)
            toks.append(t)

        pre = torch.stack(toks, dim=1)
        r = self.registers.unsqueeze(0).expand(B, -1, -1).to(pre.dtype)
        x = self.interact(torch.cat([pre, r], dim=1))

        p = self.pool(x)
        with torch.autocast(dev.type, enabled=False):
            p = p.float()
            if self.glob_norm == "ln_mag":
                mag = p.norm(dim=-1, keepdim=True) / math.sqrt(p.shape[-1])
                glob = torch.cat([F.layer_norm(p, p.shape[-1:]), mag], dim=-1)
            else:
                glob = p
        return (x, glob, pre) if return_pre else (x, glob)
