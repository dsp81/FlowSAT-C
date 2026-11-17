"""
sat_metadata_encoder.py — FlowSAT-C metadata encoder (v2).

Design rules (violations of these caused every v1 failure):
  R1  physical units only at the interface; normalise internally
  R2  scale every Fourier input to O(1) BEFORE applying a frequency ladder
  R3  frequency ceiling from Nyquist, not convention
  R4  hard ceiling f_max < 2^20 (float32 mantissa)
  R5  always concatenate the raw normalised scalar alongside Fourier features

Outputs:
  tokens (B, K, d)   field tokens + registers, after cross-field self-attention
  glob   (B, d_e)    pooled global vector for the AdaLN graft

Self-contained. No FLUX/Sana dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- config


@dataclass
class MetadataSpec:
    """Physical ranges. Set from the dataset, not hardcoded guesses."""
    gsd_min_m: float = 0.22   # EFFECTIVE gsd min (file gsd * min(W,H)/res)
    gsd_max_m: float = 11.0   # EFFECTIVE gsd max. No clipping: clipping maps
                              # distinct GSDs onto identical u -- loss of
                              # injectivity, the bug class that killed the v1
                              # geo encoder. Median at u ~ -0.37 is harmless.
    year_min: int = 2002
    year_max: int = 2017
    elev_max_m: float = 6000.0
    relief_max_m: float = 1500.0
    coast_max_km: float = 3000.0
    pop_max: float = 30000.0
    n_koppen: int = 31          # 30 classes + unknown
    n_landcover: int = 12       # embedding VOCAB: 11 classes + unknown
    n_landcover_classes: int = 11  # HISTOGRAM width: real classes only.
    # These two differ on purpose. The histogram is produced by
    # geo_covariates over real classes; the embedding needs an extra slot for
    # the unknown/nodata index. Conflating them made out_dim 88 while the
    # pipeline fed 87.
    n_platforms: int = 11       # SENSOR_PLATFORMS, index 0 = UNKNOWN


@dataclass
class EncoderConfig:
    d_model: int = 3072         # MUST equal transformer.inner_dim
    sh_degree: int = 16         # L -> (L+1)^2 coefficients
    doy_harmonics: int = 4
    solar_bands: int = 4
    gsd_bands: int = 4
    cloud_bands: int = 3
    scalar_bands: int = 3       # elevation / relief / coast / pop
    koppen_dim: int = 32
    landcover_dim: int = 16
    year_embed_dim: int = 16
    platform_dim: int = 32
    n_registers: int = 8
    interact_layers: int = 2
    interact_heads: int = 8
    d_global: int = 512
    group_dropout: dict = field(default_factory=lambda: {
        "geo": 0.05, "cov": 0.20, "doy": 0.10, "solar": 0.15,
        "year": 0.10, "gsd": 0.10, "cloud": 0.10, "sensor": 0.10,
    })
    all_dropout: float = 0.10   # metadata-CFG unconditional branch


GROUPS = ("geo", "cov", "doy", "solar", "year", "gsd", "cloud", "sensor")


# ----------------------------------------------------------------- real spherical harmonics


def real_spherical_harmonics(lat_rad: torch.Tensor,
                             lon_rad: torch.Tensor,
                             L: int) -> torch.Tensor:
    """
    Fully-normalised real spherical harmonics Y_l^m, 0<=l<=L, |m|<=l.

    Computed by three-term recurrence with normalisation folded in. The closed
    form contains (l-m)!/(l+m)! which overflows float32 past l~20, and a factor
    (1-u^2)^(m/2) which underflows at the poles; the recurrence keeps every
    intermediate O(1). Because values stay O(1), the float32 exhaustion that
    killed v1's high dyadic bands cannot occur here.

    Returns (B, (L+1)^2), ordered l-major then m = -l..l.
    """
    B = lat_rad.shape[0]
    dev, dt = lat_rad.device, lat_rad.dtype
    u = torch.sin(lat_rad)                       # (B,)
    s = torch.sqrt(torch.clamp(1.0 - u * u, min=0.0))

    # P[(l, m)] for m >= 0
    P: dict[tuple[int, int], torch.Tensor] = {}
    P[(0, 0)] = torch.full((B,), 1.0 / math.sqrt(4.0 * math.pi), device=dev, dtype=dt)

    for m in range(1, L + 1):                    # sectoral
        P[(m, m)] = -math.sqrt((2 * m + 1) / (2 * m)) * s * P[(m - 1, m - 1)]
    for m in range(0, L):                        # first off-diagonal
        P[(m + 1, m)] = math.sqrt(2 * m + 3) * u * P[(m, m)]
    for m in range(0, L + 1):                    # general recurrence
        for l in range(m + 2, L + 1):
            a = math.sqrt((4.0 * l * l - 1.0) / (l * l - m * m))
            b = math.sqrt(((l - 1.0) ** 2 - m * m) / (4.0 * (l - 1.0) ** 2 - 1.0))
            P[(l, m)] = a * (u * P[(l - 1, m)] - b * P[(l - 2, m)])

    root2 = math.sqrt(2.0)
    out = []
    for l in range(L + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            if m > 0:
                out.append(root2 * P[(l, am)] * torch.cos(am * lon_rad))
            elif m == 0:
                out.append(P[(l, 0)])
            else:
                out.append(root2 * P[(l, am)] * torch.sin(am * lon_rad))
    return torch.stack(out, dim=-1)


def fourier(u: torch.Tensor, n_bands: int) -> torch.Tensor:
    """R5: raw scalar + sin/cos at 2^k. u must already be O(1) (R2)."""
    if u.dim() == 1:
        u = u.unsqueeze(-1)
    f = torch.tensor([2.0 ** k for k in range(n_bands)], device=u.device, dtype=u.dtype)
    a = u * f                                    # (B, n_bands)
    return torch.cat([u, torch.sin(a), torch.cos(a)], dim=-1)


# ------------------------------------------------------------------------ solar geometry


def derived_solar(lat_deg: torch.Tensor, doy: torch.Tensor):
    """
    FALLBACK ONLY. fMoW's tier-1 sidecar carries MEASURED sun elevation and
    azimuth, which account for time of day and are strictly better than this
    closed form. Used when those fields are absent.

    delta(N) is an INTERMEDIATE ONLY and must never become a feature: it is
    exactly the first day-of-year harmonic rescaled, so feeding both would be
    the very collinearity this design exists to avoid.

    alpha(+40, Jul) ~ 73 deg vs alpha(-40, Jul) ~ 27 deg: the hemisphere
    inversion is in the arithmetic, so it holds at coordinates never trained on.
    """
    phi = torch.deg2rad(lat_deg)
    delta = torch.deg2rad(23.44 * torch.sin(2 * math.pi * (doy - 81.0) / 365.0))
    alpha = 90.0 - torch.abs(lat_deg - torch.rad2deg(delta))
    cos_w = torch.clamp(-torch.tan(phi) * torch.tan(delta), -1.0, 1.0)
    photoperiod = 2.0 * torch.rad2deg(torch.acos(cos_w)) / 15.0
    return alpha, photoperiod


# ------------------------------------------------------------------------- sub-encoders


class GeoEncoder(nn.Module):
    """Sphere lift + real SH with learnable per-degree weights."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.L = cfg.sh_degree
        # per-DEGREE, never per-coefficient: per-coefficient weights break the
        # addition theorem and destroy the isotropy guarantee.
        init = torch.tensor([1.0 / (1.0 + l) for l in range(self.L + 1)])
        self.log_a = nn.Parameter(torch.log(torch.expm1(init)))   # softplus^-1
        idx = torch.cat([torch.full((2 * l + 1,), l) for l in range(self.L + 1)])
        self.register_buffer("degree_index", idx, persistent=False)
        self.out_dim = (self.L + 1) ** 2

    def forward(self, lat_deg, lon_deg):
        y = real_spherical_harmonics(torch.deg2rad(lat_deg), torch.deg2rad(lon_deg), self.L)
        a = F.softplus(self.log_a)[self.degree_index]
        return y * a.unsqueeze(0)


class CovariateEncoder(nn.Module):
    """
    Geo-derived physical mediators. These add zero information (deterministic
    functions of the coordinate); their value is statistical strength sharing —
    a non-geographic similarity axis so knowledge pools across distant but
    physically similar places.
    """

    def __init__(self, cfg: EncoderConfig, spec: MetadataSpec):
        super().__init__()
        self.spec = spec
        self.nb = cfg.scalar_bands
        self.koppen = nn.Embedding(spec.n_koppen, cfg.koppen_dim)
        self.landcover = nn.Embedding(spec.n_landcover, cfg.landcover_dim)
        n_scalar = 4 * (1 + 2 * self.nb)          # elev, relief, coast, pop
        self.out_dim = (n_scalar + cfg.koppen_dim + cfg.landcover_dim
                        + spec.n_landcover_classes)

    def forward(self, cov: dict):
        s = self.spec

        def norm_log(x, xmax):
            return (torch.log1p(x.clamp(min=0)) / math.log1p(xmax)) * 2.0 - 1.0

        def signed_log_elev(h, hmax):
            # clipping at 0 made the Dead Sea (-400 m) identical to a 0 m port;
            # symmetric, monotone, 0 at sea level, +-1 at +-hmax
            return torch.sign(h) * torch.log1p(h.abs() / 100.0) / math.log1p(hmax / 100.0)

        feats = [
            fourier(signed_log_elev(cov["elevation_m"], s.elev_max_m), self.nb),
            fourier(norm_log(cov["relief_m"], s.relief_max_m), self.nb),
            fourier(norm_log(cov["coast_km"], s.coast_max_km), self.nb),
            fourier(norm_log(cov["pop_density"], s.pop_max), self.nb),
            self.koppen(cov["koppen"].long()),
            self.landcover(cov["landcover_mode"].long()),
            cov["landcover_hist"],                # window histogram, already [0,1]
        ]
        return torch.cat(feats, dim=-1)


class TemporalEncoder(nn.Module):
    """Cyclic day-of-year; year as a nuisance variable with an extrapolable trend."""

    def __init__(self, cfg: EncoderConfig, spec: MetadataSpec):
        super().__init__()
        self.H = cfg.doy_harmonics
        self.spec = spec
        n_years = spec.year_max - spec.year_min + 1
        self.year_embed = nn.Embedding(n_years + 1, cfg.year_embed_dim)  # +1 = unknown
        self.trend = nn.Linear(1, cfg.year_embed_dim, bias=False)
        self.doy_dim = 2 * self.H
        self.year_dim = cfg.year_embed_dim + 1

    def doy(self, N):
        k = torch.arange(1, self.H + 1, device=N.device, dtype=N.dtype)
        a = 2 * math.pi * N.unsqueeze(-1) * k / 365.0
        return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)

    def year(self, Y, use_embedding: bool = True):
        """
        NO Fourier: periodic bases wrap, so an unseen future year would collide
        with a past year's code. Directional extrapolation needs a monotone,
        unbounded representation.

        With sensor_platform_name now conditioned on explicitly, `year` is
        freed from the nuisance role it carried in v1 -- the sensor MTF and
        radiometry it was a noisy proxy for now have their own group, so the
        trend term can carry actual development.

        use_embedding=False at inference on unseen years: the trend alone
        extrapolates; the embedding absorbs remaining campaign idiosyncrasy that
        must NOT be extrapolated.
        """
        s = self.spec
        y = (Y.float() - s.year_min) / max(s.year_max - s.year_min, 1)
        e = self.trend(y.unsqueeze(-1))
        if use_embedding:
            idx = (Y.long() - s.year_min).clamp(0, s.year_max - s.year_min)
            in_range = ((Y >= s.year_min) & (Y <= s.year_max)).long()
            idx = torch.where(in_range.bool(), idx,
                              torch.full_like(idx, s.year_max - s.year_min + 1))
            e = e + self.year_embed(idx)
        return torch.cat([e, y.unsqueeze(-1)], dim=-1)


# ------------------------------------------------------------------------- main module


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 8, batch_first=True)

    def forward(self, x):
        q = self.q.expand(x.shape[0], -1, -1)
        return self.attn(q, x, x, need_weights=False)[0].squeeze(1)


class SatMetadataEncoder(nn.Module):
    """
    metadata dict (PHYSICAL UNITS, R1) -> (tokens, global)

    Required keys:
      lat_deg, lon_deg, gsd_m, cloud_frac (0..1), doy (1..365), year
      cov: elevation_m, relief_m, coast_km, pop_density,
           koppen (long), landcover_mode (long), landcover_hist (B, n_landcover)
    """

    def __init__(self, cfg: EncoderConfig | None = None, spec: MetadataSpec | None = None):
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        self.spec = spec or MetadataSpec()
        c, s, d = self.cfg, self.spec, (cfg or EncoderConfig()).d_model

        self.geo = GeoEncoder(c)
        self.cov = CovariateEncoder(c, s)
        self.tmp = TemporalEncoder(c, s)
        # sensor platform: WORLDVIEW02 vs GEOEYE01 differ in MTF and radiometry.
        # v1 had `year` absorbing this implicitly; naming it is cleaner and an
        # MTF difference is a real, nameable sensor property.
        self.platform = nn.Embedding(s.n_platforms, c.platform_dim)

        dims = {
            "geo":   self.geo.out_dim,
            "cov":   self.cov.out_dim,
            "doy":   self.tmp.doy_dim,
            # two MEASURED inputs only: sun elevation (Fourier) + CYCLIC azimuth
            # (sin/cos). Photoperiod was dropped -- derived, -0.716 with doy_cos1
            # and +0.729 with sun_elev_deg, i.e. sitting between two things that
            # between them explain most of it, and the weaker of the pair.
            "solar": 1 * (1 + 2 * c.solar_bands) + 2,
            "year":  self.tmp.year_dim,
            "gsd":   1 + 2 * c.gsd_bands,
            "cloud": 1 + 2 * c.cloud_bands,
            "sensor": c.platform_dim,
        }
        self.proj = nn.ModuleDict({g: nn.Linear(n, d) for g, n in dims.items()})
        self.type_embed = nn.Parameter(torch.randn(len(GROUPS), d) * 0.02)
        # learned per-group nulls. NEVER zero-fill: a normalised zero decodes to
        # a real wrong physical value (lon=-180, year=1980), i.e. OOD conditioning
        # rather than absent conditioning.
        self.null = nn.Parameter(torch.randn(len(GROUPS), d) * 0.02)
        self.register_buffer("group_ids",
                             torch.arange(len(GROUPS)), persistent=False)

        self.registers = nn.Parameter(torch.randn(c.n_registers, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d, c.interact_heads, dim_feedforward=4 * d, batch_first=True,
            norm_first=True, dropout=0.0, activation="gelu")
        # norm_first=True is a PRE-LN stack, and nn.TransformerEncoder applies no
        # final norm unless one is passed: it returned the raw accumulated
        # residual stream (token norms 300-600 at stage-2 ckpt-2000). P1 worked
        # around that with F.layer_norm in the global path; P3 read the tokens
        # raw and its GELU saturated dead by step ~200-500. Normalise at the
        # source so no future consumer inherits the hazard.
        self.interact = nn.TransformerEncoder(layer, c.interact_layers,
                                              norm=nn.LayerNorm(d))

        self.pool = AttnPool(d)
        self.to_global = nn.Linear(d + 1, c.d_global)
        self.n_tokens = len(GROUPS) + c.n_registers

    # ------------------------------------------------------------------ features

    def _group_features(self, md: dict, use_year_embedding: bool):
        c, s = self.cfg, self.spec
        lat, lon = md["lat_deg"], md["lon_deg"]

        # measured where available, derived otherwise. Photoperiod is no longer
        # a feature; derived_photoperiod() stays in data/metadata.py as a
        # fallback for datasets without measured solar.
        d_alpha, _ = derived_solar(lat, md["doy"])
        alpha = md.get("sun_elev_deg", d_alpha)

        # AZIMUTH IS CYCLIC: 359 deg and 1 deg are adjacent. Encoding it as a
        # bare scalar would repeat v1's non-cyclic month bug. It is also the one
        # ORIENTED metadata field -- shadow direction. No pathway consumes that
        # orientation now: a structure-tensor gate over 3000 real chips found
        # edge orientation tracks LATITUDE (0.58), not azimuth, so the apparent
        # azimuth signal was a geography proxy (scripts/measure_structure_tensor.py).
        #
        # Azimuth stays RAW and cyclic. lat <-> az_cos = -0.890 is expected, not
        # spurious: azimuth is a nonlinear function of (lat, lon, date, time), all of
        # which we already condition on. That is what a physically-derived quantity
        # looks like. Contrast delta(N), which IS excluded, because it is a LINEAR
        # function of one existing feature -- pure duplication.
        #
        # The 180-degree hemisphere flip was tried and rejected: fMoW is sun-synchronous
        # MORNING acquisition, so hemispheric circular means are 154.4 vs 53.9 deg,
        # separated by 100.5 not 180. The local-solar-noon premise is false for this
        # dataset. Data-driven mean removal clears the bar but bakes fMoW's overpass
        # time into the encoder and will not transfer.
        #
        # Cannibalisation is guarded by solar dropout (0.15) and MEASURED by the
        # ablate-at-inference probe, not prevented speculatively. Azimuth reaches the
        # model through a spatially-uniform oriented filter, which cannot place objects
        # -- it can satisfy a hemisphere component, it cannot substitute for geographic
        # identity.
        az = torch.deg2rad(md.get("sun_azimuth_deg", torch.zeros_like(lat)))
        az_feat = torch.stack([torch.sin(az), torch.cos(az)], dim=-1)
        if "azimuth_valid" in md:
            az_feat = az_feat * md["azimuth_valid"].view(-1, 1).to(az_feat.dtype)

        # GSD is a SCALE parameter: 0.3->0.6 and 2->4 are the same doubling.
        lg = torch.log2(md["gsd_m"].clamp(min=1e-6))
        lo, hi = math.log2(s.gsd_min_m), math.log2(s.gsd_max_m)
        u_gsd = ((lg - lo) / (hi - lo)) * 2.0 - 1.0

        return {
            "geo":   self.geo(lat, lon),
            "cov":   self.cov(md["cov"]),
            "doy":   self.tmp.doy(md["doy"]),
            "solar": torch.cat([fourier(alpha / 90.0, c.solar_bands),
                                az_feat], dim=-1),
            "year":  self.tmp.year(md["year"], use_year_embedding),
            "gsd":   fourier(u_gsd, c.gsd_bands),
            "cloud": fourier(2.0 * md["cloud_frac"] - 1.0, c.cloud_bands),
            "sensor": self.platform(md["platform"].long()),
        }

    def forward(self, md: dict, training_dropout: bool = False,
                force_null: list[str] | None = None,
                use_year_embedding: bool = True, return_pre: bool = False):
        """
        force_null: group names to replace with their null embedding. Used for
        per-field guidance and for the ablate-at-inference cannibalisation probe.

        return_pre: also return the PRE-INTERACTION per-group projections
        (B, G, d), in GROUPS order. No pathway consumes these since P3 was
        removed; the option is kept because any future per-group consumer needs
        them. After `interact` every token has attended to geo/doy/year, so a
        subset selected post-interaction is not a subset-only signal.
        """
        c = self.cfg
        feats = self._group_features(md, use_year_embedding)
        B = md["lat_deg"].shape[0]
        dev = md["lat_deg"].device

        # ONE all-metadata dropout draw per sample, shared by every group. Drawn
        # per group it was just extra independent dropout: all 8 groups were
        # almost never nulled together, so the v(c_txt, 0) branch of the 3-way
        # CFG was never trained.
        drop_all = (torch.rand(B, device=dev) < c.all_dropout if training_dropout
                    else torch.zeros(B, dtype=torch.bool, device=dev))
        toks = []
        for i, g in enumerate(GROUPS):
            t = self.proj[g](feats[g].to(self.proj[g].weight.dtype)) + self.type_embed[i]
            drop = drop_all.clone()
            if training_dropout:
                drop |= torch.rand(B, device=dev) < c.group_dropout[g]
            if g == "cov" and "cov_valid" in md:
                # absent covariates are ABSENT (learned null), not zeros --
                # zero-filled covariates decode to real wrong physical values
                drop |= ~md["cov_valid"].to(dev).bool()
            if force_null and g in force_null:
                drop |= True
            t = torch.where(drop.view(B, 1), self.null[i].unsqueeze(0).to(t.dtype), t)
            toks.append(t)

        pre = torch.stack(toks, dim=1)                                 # (B, G, d)
        r = self.registers.unsqueeze(0).expand(B, -1, -1).to(pre.dtype)
        x = self.interact(torch.cat([pre, r], dim=1))                  # (B, K, d)

        p = self.pool(x)
        # magnitude kept as an EXPLICIT channel; v1's RMS-normalisation discarded
        # it entirely, so the encoder could not express signal strength.
        mag = p.norm(dim=-1, keepdim=True) / math.sqrt(p.shape[-1])
        glob = self.to_global(torch.cat([F.layer_norm(p, p.shape[-1:]), mag], dim=-1))
        return (x, glob, pre) if return_pre else (x, glob)


if __name__ == "__main__":
    torch.manual_seed(0)
    cfg = EncoderConfig(d_model=256, sh_degree=8, d_global=128)
    enc = SatMetadataEncoder(cfg)
    B = 4
    md = dict(
        lat_deg=torch.tensor([40.0, -40.0, 19.0, 60.0]),
        lon_deg=torch.tensor([-74.0, 145.0, 72.9, 10.0]),
        gsd_m=torch.tensor([0.3, 1.0, 2.0, 4.0]),
        cloud_frac=torch.tensor([0.0, 0.3, 0.8, 0.1]),
        doy=torch.tensor([15.0, 190.0, 190.0, 300.0]),
        year=torch.tensor([2005, 2012, 2016, 2017]),
        sun_elev_deg=torch.tensor([28.0, 71.0, 66.0, 12.0]),
        sun_azimuth_deg=torch.tensor([160.0, 175.0, 120.0, 200.0]),
        photoperiod_h=torch.tensor([9.3, 10.6, 13.1, 8.2]),
        azimuth_valid=torch.tensor([True, True, True, True]),
        platform=torch.tensor([2, 2, 6, 7]),
        cov=dict(elevation_m=torch.tensor([10., 300., 14., 900.]),
                 relief_m=torch.tensor([5., 50., 8., 400.]),
                 coast_km=torch.tensor([2., 400., 1., 30.]),
                 pop_density=torch.tensor([9000., 20., 25000., 500.]),
                 koppen=torch.tensor([8, 5, 3, 12]),
                 landcover_mode=torch.tensor([1, 4, 1, 6]),
                 # width MUST match spec.n_landcover_classes, not the vocab
                 landcover_hist=torch.rand(B, MetadataSpec().n_landcover_classes)),
    )
    tok, glob = enc(md, training_dropout=True)
    print("tokens", tuple(tok.shape), "global", tuple(glob.shape))
    print("encoder params", sum(p.numel() for p in enc.parameters()) / 1e6, "M")

    # hemisphere inversion sanity check (derived fallback path)
    a, ph = derived_solar(torch.tensor([40.0, -40.0]), torch.tensor([190.0, 190.0]))
    print("solar elevation +40/-40 on doy190:", a.tolist())
    print("photoperiod:", ph.tolist())