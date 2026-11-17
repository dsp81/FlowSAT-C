"""
flowsat_c_transformer.py — FlowSAT-C conditioning on Flux2Transformer2DModel.

Attach points VERIFIED against diffusers Flux2Transformer2DModel:
  .time_guidance_embed            -> temb (timestep + guidance ONLY, no pooled text)
  .double_stream_modulation_img / _txt / .single_stream_modulation
  .transformer_blocks / .single_transformer_blocks
  .context_embedder, .x_embedder, .pos_embed, .norm_out, .proj_out
  joint_attention_kwargs["attention_mask"] reaches F.sdpa in BOTH processors

Why this backbone: temb carries no pooled text (unlike FLUX.1/SD3), so the global
modulation pathway is semantically empty and the metadata graft becomes the ONLY
semantic signal there. Attribution is clean.

PATHWAYS -- asymmetric roles. Do NOT give metadata a copy of text's pathway; two
channels competing for the same slots reproduces v1 at higher parameter count.

  P1  global AdaLN graft (zero-init)  -> uniform scalar effects
  P2  metadata tokens in joint attn   -> semantic placement, spatially selective

P3 (spectral conditioning) WAS REMOVED. Two independent measurements: the filter
collapsed to a DC gain -- a global tone shift, i.e. P1's job -- because token-grid
radial energy is ~1e4x concentrated at low rho; and a structure-tensor gate over
3000 real chips (scripts/measure_structure_tensor.py) found fMoW metadata carries
no second-order spatial statistics to condition (land cover explains 4.7% of
gradient-energy variance, and orientation-vs-azimuth is a latitude proxy). The
architecture is deliberately two pathways with non-overlapping roles.

INTERPRETABILITY PLUMBING: `return_hidden` collects the residual stream after
each block plus token boundary offsets. Off by default, zero cost when off, no
effect on gradients or weights -- it exists so the model never has to be reopened
to instrument it later.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from flowsat_C.models.encoder import GROUPS, EncoderConfig, MetadataSpec, SatMetadataEncoder


class FlowSatC(nn.Module):
    """
    Stages:
      0  gates shut, tokens off -> MUST be bit-identical to base
      1  full backbone fine-tune, metadata frozen -> ESTABLISH FID/CLIP HERE
      2  metadata pathways open               -> controllability
      3  optional joint polish
    """

    def __init__(self, transformer,
                 enc_cfg: EncoderConfig | None = None,
                 spec: MetadataSpec | None = None,
                 latent_hw: tuple[int, int] = (32, 32),
                 md_rope_axis: int = 3, md_rope_value: int = 1,
                 gate_bias_init: float = -20.0):
        super().__init__()
        self.tf = transformer
        d = transformer.inner_dim
        cfg = enc_cfg or EncoderConfig()
        cfg.d_model = d
        self.encoder = SatMetadataEncoder(cfg, spec or MetadataSpec())
        self.K = self.encoder.n_tokens
        self.group_index = {g: i for i, g in enumerate(GROUPS)}

        n_double = len(transformer.transformer_blocks)
        n_single = len(transformer.single_transformer_blocks)
        self.n_double, self.n_single = n_double, n_single
        self.latent_hw = latent_hw

        # ---- P1: zero-init graft into temb --------------------------------
        temb_dim = transformer.double_stream_modulation_img.linear.in_features
        self.to_temb = nn.Linear(cfg.d_global, temb_dim, bias=False)
        nn.init.zeros_(self.to_temb.weight)
        # per-block scalars: metadata can act at different depths, and the
        # learned profile is a plottable figure. 25 parameters total.
        self.alpha_double = nn.Parameter(torch.ones(n_double))
        self.alpha_single = nn.Parameter(torch.ones(n_single))

        # ---- P2: metadata tokens ------------------------------------------
        self.md_rope_axis, self.md_rope_value = md_rope_axis, md_rope_value
        # at -20 the metadata share of softmax mass is ~2e-11: numerically zero,
        # so inserting tokens does not perturb the base model. Learnable, so it
        # rises only as gradients justify it -- and the trace is a figure.
        self.gate_bias = nn.Parameter(torch.full((1,), float(gate_bias_init)))
        self.metadata_tokens_enabled = False

        self._diag: dict = {}
        self.stage = 0

    # ------------------------------------------------------------------ staging

    def set_stage(self, stage: int):
        assert stage in (0, 1, 2, 3)
        on = stage >= 2
        self.metadata_tokens_enabled = on
        for p in self.encoder.parameters():
            p.requires_grad = on
        for t in (self.to_temb.weight, self.alpha_double,
                  self.alpha_single, self.gate_bias):
            t.requires_grad = on
        # Full fine-tune from pretrained weights. Backbone trains from stage 1.
        for p in self.tf.parameters():
            p.requires_grad = stage >= 1
        self.stage = stage

    # ------------------------------------------------------------------ helpers

    def _md_ids(self, txt_ids):
        """
        Axis 3 = L. Read from diffusers pipeline_flux2._prepare_latent_ids:
        image ids are (T=0, H=0..H-1, W=0..W-1, L=0) and text ids are all zero,
        so L is unused by both. Axis 0 (T) is NOT free -- _prepare_image_ids
        assigns increasing T to reference images, so claiming it would collide
        with the multi-reference / temporal path later.
        Confirm with scripts/verify_rope_axis.py on the actual checkpoint.
        """
        ids = torch.zeros(self.K, txt_ids.shape[-1],
                          device=txt_ids.device, dtype=txt_ids.dtype)
        ids[:, self.md_rope_axis] = self.md_rope_value
        return ids

    def _attn_bias(self, n_txt, n_img, dtype, device):
        """Additive mask over [txt, md, img]. Only metadata KEY columns are
        biased, so metadata is invisible to everyone at init while still
        receiving information (its query rows are unbiased)."""
        n = n_txt + self.K + n_img
        m = torch.zeros(n, n, dtype=dtype, device=device)
        m[:, n_txt:n_txt + self.K] = self.gate_bias.to(dtype)
        return m

    # ------------------------------------------------------------------ forward

    def forward(self, hidden_states, encoder_hidden_states, timestep,
                img_ids, txt_ids, metadata=None, guidance=None,
                training_dropout=False, force_null=None,
                use_year_embedding=True, joint_attention_kwargs=None,
                collect_diagnostics=False, return_hidden=False):
        tf, dtype = self.tf, hidden_states.dtype
        hiddens = [] if return_hidden else None

        md_tokens = temb_add = None
        if metadata is not None:
            # DeepSpeed bf16 casts the weights without autocasting the fp32
            # metadata inputs; autocast the encoder only, never the backbone
            # (autocast would change backbone op dtypes and break stage 0)
            enc_dt = next(self.encoder.parameters()).dtype
            with torch.autocast(hidden_states.device.type, dtype=enc_dt,
                                enabled=enc_dt != torch.float32):
                enc_out = self.encoder(
                    metadata, training_dropout=training_dropout,
                    force_null=force_null, use_year_embedding=use_year_embedding)
            md_tokens, md_glob = enc_out[0], enc_out[1]
            md_tokens = md_tokens.to(dtype)
            temb_add = self.to_temb(md_glob.to(self.to_temb.weight.dtype)).to(dtype)

        # ---- 1. timestep embedding + P1 graft ------------------------------
        t = timestep.to(dtype) * 1000
        g = guidance.to(dtype) * 1000 if guidance is not None else None
        temb = tf.time_guidance_embed(t, g)

        def mod(module, alphas):
            if temb_add is None:
                return [module(temb)] * alphas.numel()
            return [module(temb + a.to(dtype) * temb_add) for a in alphas]

        mod_img = mod(tf.double_stream_modulation_img, self.alpha_double)
        mod_txt = mod(tf.double_stream_modulation_txt, self.alpha_double)
        mod_sgl = mod(tf.single_stream_modulation, self.alpha_single)

        # ---- 2. embed streams ----------------------------------------------
        n_img = hidden_states.shape[1]
        hidden_states = tf.x_embedder(hidden_states)
        encoder_hidden_states = tf.context_embedder(encoder_hidden_states)
        n_txt = encoder_hidden_states.shape[1]
        use_md = self.metadata_tokens_enabled and md_tokens is not None

        # ---- 3. RoPE --------------------------------------------------------
        if img_ids.ndim == 3:
            img_ids = img_ids[0]
        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        ids_txt = torch.cat([txt_ids, self._md_ids(txt_ids)], 0) if use_md else txt_ids
        rt, ri = tf.pos_embed(ids_txt), tf.pos_embed(img_ids)
        rope = (torch.cat([rt[0], ri[0]], 0), torch.cat([rt[1], ri[1]], 0))

        jak = dict(joint_attention_kwargs or {})
        if use_md:
            jak["attention_mask"] = self._attn_bias(n_txt, n_img, dtype,
                                                    hidden_states.device)
            # metadata rides in the context stream: full joint attention with
            # text AND image, and zero new attention parameters.
            encoder_hidden_states = torch.cat([encoder_hidden_states, md_tokens], 1)

        # ---- 4. double-stream blocks ---------------------------------------
        # blocks are called here rather than via tf.forward, so honour the
        # backbone's gradient-checkpointing flag explicitly
        ckpt = torch.is_grad_enabled() and getattr(tf, "gradient_checkpointing", False)
        for i, block in enumerate(tf.transformer_blocks):
            if ckpt:
                encoder_hidden_states, hidden_states = tf._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states,
                    mod_img[i], mod_txt[i], rope, jak)
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb_mod_img=mod_img[i], temb_mod_txt=mod_txt[i],
                    image_rotary_emb=rope, joint_attention_kwargs=jak)

            if return_hidden:
                hiddens.append({"block": f"double_{i}",
                                "img": hidden_states.detach(),
                                "ctx": encoder_hidden_states.detach()})

        # ---- 5. single-stream blocks ---------------------------------------
        n_ctx = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], 1)
        for i, block in enumerate(tf.single_transformer_blocks):
            if ckpt:
                hidden_states = tf._gradient_checkpointing_func(
                    block, hidden_states, None, mod_sgl[i], rope, jak)
            else:
                hidden_states = block(hidden_states=hidden_states,
                                      encoder_hidden_states=None,
                                      temb_mod=mod_sgl[i], image_rotary_emb=rope,
                                      joint_attention_kwargs=jak)
            if return_hidden:
                hiddens.append({"block": f"single_{i}",
                                "joint": hidden_states.detach()})
        hidden_states = hidden_states[:, n_ctx:, ...]

        if collect_diagnostics:
            self._diag = {
                "gate_bias": self.gate_bias.detach().float().item(),
                "alpha_double": self.alpha_double.detach().float().cpu(),
                "alpha_single": self.alpha_single.detach().float().cpu(),
                "temb_add_norm": (0.0 if temb_add is None else
                                  temb_add.detach().float().norm(dim=-1).mean().item()),
            }

        out = tf.proj_out(tf.norm_out(hidden_states, temb))
        if return_hidden:
            # boundary offsets: without these you cannot tell which slice of the
            # joint stream is text, metadata or image when analysing later.
            return out, {"hidden": hiddens,
                         "offsets": {"n_txt": n_txt, "K": self.K if use_md else 0,
                                     "n_img": n_img, "n_ctx": n_ctx}}
        return out

    # ------------------------------------------------------------------ gate check

    @torch.no_grad()
    def assert_identity(self, batch, atol=0.0):
        """Stage-0 gate. If max|delta| != 0 something is not actually gated --
        do not proceed. Five minutes here saves weeks of confusion."""
        was = self.metadata_tokens_enabled
        self.metadata_tokens_enabled = False
        mine = self.forward(**batch)
        base = self.tf(hidden_states=batch["hidden_states"],
                       encoder_hidden_states=batch["encoder_hidden_states"],
                       timestep=batch["timestep"], img_ids=batch["img_ids"],
                       txt_ids=batch["txt_ids"], guidance=batch.get("guidance"),
                       return_dict=False)[0]
        self.metadata_tokens_enabled = was
        err = (mine - base).abs().max().item()
        assert err <= atol, f"graft is not identity at init: max|delta|={err}"
        return err


def build_flowsat_c(model_id="black-forest-labs/FLUX.2-klein-base-4B",
                    torch_dtype=torch.bfloat16, resolution: int = 512, **kw):
    from diffusers import Flux2Transformer2DModel
    tf = Flux2Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch_dtype)
    hw = resolution // 16          # 8x VAE spatial + 2x2 patchify
    kw.setdefault("latent_hw", (hw, hw))
    return FlowSatC(tf, **kw)