"""
FlowSatCAdapter: wraps the existing P1-perblock model and sampler (flowsat_C.inference.sampler_p1pb).
GPU only; never constructed by the CPU tests (spec section 9). UNTESTED until a GPU run is approved.

Fields -> model metadata
    Built exactly as FMoWCaptionDataset.__getitem__ + make_collate do (RawMetadata +
    collate_metadata), with gsd_m = the record's EFFECTIVE gsd (gsd_eff_m), which is what the model
    was trained on. Covariates come from the cov.* fields; landcover_hist is taken from the field
    `cov.landcover_hist` when present and consistent with cov.landcover_mode, else one-hot of the
    mode (INFERRED when a sweep changes the land-cover class).

Null mask -> force_null
    {group: True} -> force_null=[groups] on the conditional-metadata branch (through the same encoder
    proxy as the pathway modes). The sampler's two null branches already null every group, so they
    are unchanged.

Pathway modes (T9) -- "runtime proxies"
    The model's encoder is wrapped by _EncoderProxy for the duration of one generation. The proxy acts
    only when the encoder is called WITHOUT force_null, i.e. on the v(c_txt, c_md) branch -- the only
    CFG branch in which the pathway's conditioning is present. v(0,0) and v(c_txt,0) call the encoder
    with force_null = all groups and are left untouched.
      P1 input = the pooled global vector (md_glob); P2 input = the 16 metadata tokens (md_tokens).
      zero     the pathway's input replaced by zeros
      mean     replaced by its mean over the TRAIN records given to prepare_pathway_means (4096)
      shuffle  replaced by the encoder output for the donor's fields; with `groups`, the donor's
               values are used only for those field groups (positive control), the rest are own.
"""
from __future__ import annotations

import numpy as np

from metacf.adapters.base import ModelAdapter

GROUP_OF = {"latlon": "geo", "gsd": "gsd", "cloud": "cloud", "doy": "doy", "sun_elev": "solar",
            "sun_azim": "solar", "year": "year", "sensor": "sensor", "koppen": "cov",
            "landcover": "cov", "elevation": "cov", "relief": "cov", "coast": "cov",
            "population": "cov"}


class FlowSatCAdapter(ModelAdapter):
    name = "flowsat_c"
    pathways = ["P1", "P2"]

    def __init__(self, mcfg: dict, device="cuda"):
        self.mcfg = mcfg
        self.model_id = mcfg["model_id"]
        self.checkpoint = mcfg["checkpoint"]
        self.device = device
        self.steps = int(mcfg.get("steps", 20))
        self.w_txt = float(mcfg.get("w_txt", 2.0))
        self.w_md = float(mcfg.get("w_md", 2.0))
        self.resolution = int(mcfg.get("resolution", 512))
        self.supported_fields = set(mcfg.get("supported_fields", GROUP_OF.keys()))
        self.field_groups = {f: GROUP_OF[f] for f in self.supported_fields if f in GROUP_OF}
        self._model = None
        self._means = None

    # ---------------------------------------------------------------- loading
    def _load(self):
        if self._model is not None:
            return
        import torch
        from flowsat_C.inference.sampler_p1pb import load_p1pb
        from flowsat_C.training.flux2_io import load_flux2_io
        torch.set_num_threads(4)
        self._model = load_p1pb(self.model_id, self.checkpoint, resolution=self.resolution
                                ).to(self.device).eval()
        self._lat_enc, self._txt_enc = load_flux2_io(self.model_id, self.device,
                                                     max_sequence_length=256)

    def _metadata(self, field_dicts):
        import torch
        from flowsat_C.data.metadata import RawMetadata, collate_metadata
        items, covs, valid = [], [], []
        for f in field_dicts:
            az = f.get("sun_azimuth_deg")
            items.append(RawMetadata(
                key="metacf", category="", location_id="", view_id=0,
                lat_deg=float(f["lat_deg"]), lon_deg=float(f["lon_deg"]), gsd_m=float(f["gsd_eff_m"]),
                cloud_frac=float(f["cloud_frac"]), doy=float(f["doy"]), year=int(round(f["year"])),
                sun_elev_deg=float(f["sun_elev_deg"]), sun_azimuth_deg=float(az if az is not None else 0.0),
                photoperiod_h=float(f.get("photoperiod_h") or 12.0), platform=int(f.get("platform") or 0),
                img_w=self.resolution, img_h=self.resolution, azimuth_valid=az is not None))
            lc = f.get("cov.landcover_mode")
            hist = f.get("cov.landcover_hist")
            if lc is not None and (hist is None or int(np.argmax(hist)) != int(lc)):
                hist = [0.0] * 11
                hist[int(lc)] = 1.0
            cov_ok = f.get("cov.koppen") is not None
            covs.append(dict(elevation_m=float(f.get("cov.elevation_m") or 0.0),
                             relief_m=float(f.get("cov.relief_m") or 0.0),
                             coast_km=float(f.get("cov.coast_km") or 0.0),
                             pop_density=float(f.get("cov.pop_density") or 0.0),
                             koppen=int(f.get("cov.koppen") if cov_ok else 30),
                             landcover_mode=int(lc if lc is not None else 11),
                             landcover_hist=list(hist) if hist is not None else [0.0] * 11))
            valid.append(cov_ok)
        md = collate_metadata(items, covs)
        md["cov_valid"] = torch.tensor(valid, dtype=torch.bool)
        from accelerate.utils import send_to_device
        return send_to_device(md, self.device)

    # ---------------------------------------------------------------- sampling
    def _sample(self, captions, field_dicts, seeds, guidance=None):
        import torch
        from flowsat_C.inference.sampler_p1pb import sample
        self._load()
        m, dev = self._model, self.device
        n = len(captions)
        S = self.resolution // 16
        img_ids = torch.zeros(S * S, 4, device=dev)
        hw = torch.arange(S, device=dev).float()
        img_ids[:, 1], img_ids[:, 2] = hw.repeat_interleave(S), hw.repeat(S)
        caps = [c if c is not None else "" for c in captions]
        cond, txt_ids = self._txt_enc.encode(caps)
        null_emb, _ = self._txt_enc.null(n)
        # null caption = the empty string through the chat template, i.e. the learned CFG null
        noise = torch.stack([torch.randn(S * S, 128, generator=torch.Generator().manual_seed(int(s)))
                             for s in seeds]).to(dev, torch.bfloat16)
        md = self._metadata(field_dicts)
        w_txt, w_md = (guidance or (self.w_txt, self.w_md))
        lat = sample(m, noise, cond, null_emb, img_ids, txt_ids, md, num_steps=self.steps,
                     w_txt=w_txt, w_md=w_md)
        im = self._lat_enc.decode(lat, m.latent_hw).clamp(-1, 1).add(1).mul(127.5).round()
        return [x.permute(1, 2, 0).to(torch.uint8).cpu().numpy() for x in im]

    def generate(self, caption, fields, null_mask, seed, guidance=None):
        from PIL import Image
        groups = [g for g, on in (null_mask or {}).items() if on]
        with _EncoderProxy.install(self._model_ready(), null_groups=groups):
            return Image.fromarray(self._sample([caption], [fields], [seed], guidance)[0])

    def _model_ready(self):
        self._load()
        return self._model

    def prepare_pathway_means(self, train_fields):
        import torch
        m = self._model_ready()
        toks, globs = [], []
        with torch.no_grad():
            for i in range(0, len(train_fields), 64):
                md = self._metadata(train_fields[i:i + 64])
                t, g = m.encoder(md, training_dropout=False, force_null=None)
                toks.append(t.float().mean(0, keepdim=True) * len(t))
                globs.append(g.float().mean(0, keepdim=True) * len(g))
        n = len(train_fields)
        self._means = dict(P2=torch.cat(toks).sum(0) / n, P1=torch.cat(globs).sum(0) / n)

    def generate_with_pathway(self, caption, fields, null_mask, seed, pathway, mode,
                              donor_fields=None, groups=None):
        from PIL import Image
        m = self._model_ready()
        donor_md = None
        if mode == "shuffle":
            if groups:                       # positive control: donor values only for these groups
                mixed = dict(fields)
                gk = self.group_keys()
                for g in groups:
                    for k in gk.get(g, []):
                        mixed[k] = donor_fields.get(k)
                donor_md = self._metadata([mixed])
            else:
                donor_md = self._metadata([donor_fields])
        if mode == "mean" and self._means is None:
            raise RuntimeError("prepare_pathway_means() must run before MEAN mode")
        null = [g for g, on in (null_mask or {}).items() if on]
        with _EncoderProxy.install(m, null_groups=null, pathway=pathway, mode=mode,
                                   donor_md=donor_md, means=self._means):
            return Image.fromarray(self._sample([caption], [fields], [seed])[0])

    @staticmethod
    def group_keys():
        return {"geo": ["lat_deg", "lon_deg"], "gsd": ["gsd_eff_m"], "cloud": ["cloud_frac"],
                "doy": ["doy"], "solar": ["sun_elev_deg", "sun_azimuth_deg", "photoperiod_h"],
                "year": ["year"], "sensor": ["platform"],
                "cov": ["cov.elevation_m", "cov.relief_m", "cov.coast_km", "cov.pop_density",
                        "cov.koppen", "cov.landcover_mode", "cov.landcover_hist"]}


class _EncoderProxy:
    """Context manager that swaps model.encoder for a wrapper and restores it afterwards."""

    @staticmethod
    def install(model, null_groups=(), pathway=None, mode=None, donor_md=None, means=None):
        return _EncoderProxy(model, list(null_groups or []), pathway, mode, donor_md, means)

    def __init__(self, model, null_groups, pathway, mode, donor_md, means):
        self.model, self.null_groups = model, null_groups
        self.pathway, self.mode, self.donor_md, self.means = pathway, mode, donor_md, means

    def __enter__(self):
        import torch.nn as nn
        inner = self.model.encoder
        proxy_self = self

        class Proxy(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = inner

            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.inner, name)

            def forward(self, md, training_dropout=False, force_null=None, use_year_embedding=True):
                conditional = not force_null
                fn = proxy_self.null_groups if (conditional and proxy_self.null_groups) else force_null
                tok, glob = self.inner(md, training_dropout=training_dropout, force_null=fn,
                                       use_year_embedding=use_year_embedding)
                if not conditional or proxy_self.pathway is None:
                    return tok, glob
                B = tok.shape[0]
                if proxy_self.mode == "zero":
                    new = None
                elif proxy_self.mode == "mean":
                    new = proxy_self.means[proxy_self.pathway]
                else:
                    dt, dg = self.inner(proxy_self.donor_md, training_dropout=False, force_null=fn,
                                        use_year_embedding=use_year_embedding)
                    new = dt[0] if proxy_self.pathway == "P2" else dg[0]
                if proxy_self.pathway == "P1":
                    glob = glob * 0 if new is None else new.to(glob.dtype).expand(B, -1)
                else:
                    tok = tok * 0 if new is None else new.to(tok.dtype).expand(B, -1, -1)
                return tok, glob

        self.inner = inner
        self.model.encoder = Proxy()
        return self

    def __exit__(self, *exc):
        self.model.encoder = self.inner
        return False
