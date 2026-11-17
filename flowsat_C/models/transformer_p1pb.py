"""
transformer_p1pb.py — FlowSAT-C with PER-BLOCK P1 modulation heads ("P1-perblock").

Derived from models/transformer.py (branch arch/no-p3). P2 is unchanged. P1 changed:

  OLD  glob -> to_global(512) -> to_temb -> temb + alpha_k * add -> mod_t(.)   (transformer.py:156-170)
       The per-record part of the add was ~1% of ||temb|| and ~0.05% of the block
       modulation, below bf16 resolution (diagnostics/p1/REPORT.md D1/D7).

  NEW  temb carries ONLY the timestep, as in the pretrained model. P1 is its own
       low-rank channel added to each block's modulation output:

         h        = trunk([glob ; silu(temb)])                  1024-d, shared
         delta_k  = U_k(D_k(h))                                 per block / stream
         mod_k    = cast_bf16( fp32(mod_t(temb)) + delta_k )

       trunk = Linear(d_in -> 1024) -> SiLU -> Linear(1024 -> 1024), normal init
       D_k   = Linear(1024 -> r), normal init                  (--p1-rank, default 128)
       U_k   = Linear(r -> modulation size), ZERO weight + bias -> exact no-op at init

       Heads: 5 double blocks x {img, txt} (18432 = 2 x shift/scale/gate x 3072),
       20 single blocks (9216 = shift/scale/gate x 3072), and norm_out (6144 = scale,
       shift). In FLUX.2 klein mod_t is SHARED: one double_stream_modulation_img, one
       _txt and one single_stream_modulation for all blocks of that kind (bias-free),
       plus norm_out's own linear. The heads are what makes P1 per-block.

  PRECISION  trunk, D_k and U_k run under torch.autocast(enabled=False) as F.linear
       on fp32-cast inputs AND fp32-cast weights; delta_k is added to mod_t's output in
       fp32 and the sum is cast once to the backbone dtype. The AdaLN affine, attention
       and everything else in the blocks run exactly as before (bf16 under DeepSpeed).

  P2 PATH DROPOUT  forward(p2_drop=bool[B]): for those samples the metadata key
       columns are -inf for every non-metadata query (image AND text), and metadata
       query rows see only metadata keys, so nothing flows in or out of the tokens.
       Shape becomes (B, 1, n, n); without dropout the mask stays (n, n) as before.

  P1 PATH DROPOUT  forward(p1_drop=bool[B]): for those samples the P1 trunk gets the
       all-null metadata glob (every group at its learned null, = the encoder's
       all-dropout case) instead of the record's, so P1 still adds its constant
       (null-metadata) delta but carries no per-record information. P2 is untouched.
       Drawn independently of p2_drop by the trainer (symmetric pathway dropout).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from flowsat_C.models.encoder import GROUPS, EncoderConfig, MetadataSpec
from flowsat_C.models.encoder_p1pb import SatMetadataEncoderP1PB

TRUNK_DIM = 1024


def lin32(x: torch.Tensor, lin: nn.Linear) -> torch.Tensor:
    """fp32 F.linear: input AND weights upcast. Caller disables autocast."""
    return F.linear(x.float(), lin.weight.float(),
                    None if lin.bias is None else lin.bias.float())


class P1Head(nn.Module):
    """delta = U(D(h)). D normal init; U zero weight AND bias."""

    def __init__(self, r: int, out: int, std: float):
        super().__init__()
        self.D = nn.Linear(TRUNK_DIM, r)
        self.U = nn.Linear(r, out)
        nn.init.normal_(self.D.weight, std=std)
        nn.init.zeros_(self.D.bias)
        nn.init.zeros_(self.U.weight)
        nn.init.zeros_(self.U.bias)

    def forward(self, h):
        return lin32(lin32(h, self.D), self.U)


class FlowSatCP1PB(nn.Module):

    def __init__(self, transformer,
                 enc_cfg: EncoderConfig | None = None,
                 spec: MetadataSpec | None = None,
                 latent_hw: tuple[int, int] = (32, 32),
                 md_rope_axis: int = 3, md_rope_value: int = 1,
                 gate_bias_init: float = -20.0,
                 glob_norm: str = "ln_mag", p1_rank: int = 128,
                 p1_init_std: float = 0.02):
        super().__init__()
        self.tf = transformer
        d = transformer.inner_dim
        cfg = enc_cfg or EncoderConfig()
        cfg.d_model = d
        self.encoder = SatMetadataEncoderP1PB(cfg, spec or MetadataSpec(), glob_norm=glob_norm)
        self.K = self.encoder.n_tokens
        self.group_index = {g: i for i, g in enumerate(GROUPS)}

        n_double = len(transformer.transformer_blocks)
        n_single = len(transformer.single_transformer_blocks)
        self.n_double, self.n_single = n_double, n_single
        self.latent_hw = latent_hw

        # ---- P1: shared trunk + per-block low-rank zero-init heads ----------
        temb_dim = transformer.double_stream_modulation_img.linear.in_features
        self.d_glob, self.p1_rank = self.encoder.d_glob, p1_rank
        self.p1_trunk = nn.Sequential(nn.Linear(self.d_glob + temb_dim, TRUNK_DIM), nn.SiLU(),
                                      nn.Linear(TRUNK_DIM, TRUNK_DIM))
        for m in (self.p1_trunk[0], self.p1_trunk[2]):
            nn.init.normal_(m.weight, std=p1_init_std)
            nn.init.zeros_(m.bias)
        out_d = transformer.double_stream_modulation_img.linear.out_features   # 18432
        out_s = transformer.single_stream_modulation.linear.out_features       # 9216
        out_o = transformer.norm_out.linear.out_features                       # 6144
        self.mod_md_img = nn.ModuleList(P1Head(p1_rank, out_d, p1_init_std) for _ in range(n_double))
        self.mod_md_txt = nn.ModuleList(P1Head(p1_rank, out_d, p1_init_std) for _ in range(n_double))
        self.mod_md_sgl = nn.ModuleList(P1Head(p1_rank, out_s, p1_init_std) for _ in range(n_single))
        self.mod_md_out = P1Head(p1_rank, out_o, p1_init_std)

        # ---- P2: metadata tokens (unchanged) -------------------------------
        self.md_rope_axis, self.md_rope_value = md_rope_axis, md_rope_value
        self.gate_bias = nn.Parameter(torch.full((1,), float(gate_bias_init)))
        self.metadata_tokens_enabled = False

        self._diag: dict = {}
        self._precision_probe = None      # list -> forward appends dtype records (verification)
        self.stage = 0

    # ------------------------------------------------------------------ groups

    def p1_modules(self):
        return [self.p1_trunk, self.mod_md_img, self.mod_md_txt, self.mod_md_sgl, self.mod_md_out]

    def p1_parameters(self):
        for m in self.p1_modules():
            yield from m.parameters()

    # ------------------------------------------------------------------ staging

    def set_stage(self, stage: int):
        assert stage in (0, 1, 2, 3)
        on = stage >= 2
        self.metadata_tokens_enabled = on
        for p in self.encoder.parameters():
            p.requires_grad = on
        for p in self.p1_parameters():
            p.requires_grad = on
        self.gate_bias.requires_grad = on
        for p in self.tf.parameters():
            p.requires_grad = stage >= 1
        self.stage = stage

    # ------------------------------------------------------------------ helpers

    def _md_ids(self, txt_ids):
        """transformer.py:110-122, unchanged."""
        ids = torch.zeros(self.K, txt_ids.shape[-1],
                          device=txt_ids.device, dtype=txt_ids.dtype)
        ids[:, self.md_rope_axis] = self.md_rope_value
        return ids

    def _attn_bias(self, n_txt, n_img, dtype, device, p2_drop=None):
        """transformer.py:124-131 plus per-sample P2 path dropout."""
        n = n_txt + self.K + n_img
        md = slice(n_txt, n_txt + self.K)
        m = torch.zeros(n, n, dtype=dtype, device=device)
        m[:, md] = self.gate_bias.to(dtype)
        if p2_drop is None or not bool(p2_drop.any()):
            return m
        # dropped samples: metadata unreachable from every other token, and the
        # metadata rows attend only among themselves (an all -inf row would NaN)
        dm = torch.zeros(n, n, dtype=dtype, device=device)
        dm[:, md] = float("-inf")
        dm[md, :] = float("-inf")
        dm[md, md] = 0.0
        return torch.where(p2_drop.view(-1, 1, 1, 1).to(device), dm, m.expand(1, 1, n, n))

    def _record(self, name, base, delta, total):
        if self._precision_probe is not None:
            self._precision_probe.append(dict(
                name=name, base=str(base.dtype), delta=str(delta.dtype), sum=str(total.dtype),
                autocast=torch.is_autocast_enabled()))

    # ------------------------------------------------------------------ forward

    def forward(self, hidden_states, encoder_hidden_states, timestep,
                img_ids, txt_ids, metadata=None, guidance=None,
                training_dropout=False, force_null=None,
                use_year_embedding=True, joint_attention_kwargs=None,
                collect_diagnostics=False, return_hidden=False, p2_drop=None,
                p1_drop=None):
        tf, dtype = self.tf, hidden_states.dtype
        dev_type = hidden_states.device.type
        hiddens = [] if return_hidden else None

        md_tokens = md_glob = None
        if metadata is not None:
            # encoder-only autocast, as transformer.py:145-153 (features inside are fp32)
            enc_dt = next(self.encoder.parameters()).dtype
            with torch.autocast(dev_type, dtype=enc_dt, enabled=enc_dt != torch.float32):
                md_tokens, md_glob = self.encoder(
                    metadata, training_dropout=training_dropout,
                    force_null=force_null, use_year_embedding=use_year_embedding)
                if p1_drop is not None and bool(p1_drop.any()):
                    # P1 path dropout: those samples feed P1 the all-null glob
                    _, null_glob = self.encoder(
                        metadata, training_dropout=False, force_null=list(GROUPS),
                        use_year_embedding=use_year_embedding)
                    md_glob = torch.where(p1_drop.view(-1, 1).to(md_glob.device),
                                          null_glob.to(md_glob.dtype), md_glob)
            md_tokens = md_tokens.to(dtype)

        # ---- 1. timestep embedding: timestep ONLY ---------------------------
        t = timestep.to(dtype) * 1000
        g = guidance.to(dtype) * 1000 if guidance is not None else None
        temb = tf.time_guidance_embed(t, g)
        base_img = tf.double_stream_modulation_img(temb)
        base_txt = tf.double_stream_modulation_txt(temb)
        base_sgl = tf.single_stream_modulation(temb)
        base_out = tf.norm_out.linear(tf.norm_out.silu(temb).to(dtype))   # AdaLayerNormContinuous

        # ---- 1b. P1 per-block deltas, fp32 ----------------------------------
        deltas = None
        if md_glob is not None:
            with torch.autocast(dev_type, enabled=False):
                x = torch.cat([md_glob.float(), F.silu(temb.float())], dim=-1)
                h = lin32(F.silu(lin32(x, self.p1_trunk[0])), self.p1_trunk[2])
                deltas = dict(img=[hd(h) for hd in self.mod_md_img],
                              txt=[hd(h) for hd in self.mod_md_txt],
                              sgl=[hd(h) for hd in self.mod_md_sgl],
                              out=self.mod_md_out(h))

        def add(name, base, delta):
            if delta is None:
                return base
            with torch.autocast(dev_type, enabled=False):
                total = base.float() + delta
                self._record(name, base, delta, total)
            return total.to(base.dtype)

        D = deltas or {}
        mod_img = [add(f"double_{i}_img", base_img, D["img"][i] if D else None)
                   for i in range(self.n_double)]
        mod_txt = [add(f"double_{i}_txt", base_txt, D["txt"][i] if D else None)
                   for i in range(self.n_double)]
        mod_sgl = [add(f"single_{i}", base_sgl, D["sgl"][i] if D else None)
                   for i in range(self.n_single)]
        emb_out = add("norm_out", base_out, D.get("out"))

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
                                                    hidden_states.device, p2_drop)
            encoder_hidden_states = torch.cat([encoder_hidden_states, md_tokens], 1)

        # ---- 4. double-stream blocks ---------------------------------------
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
            self._diag = {"gate_bias": self.gate_bias.detach().float().item()}
            if deltas is not None:
                bases = dict(img=base_img, txt=base_txt, sgl=base_sgl, out=base_out)
                keep = None if p1_drop is None else ~p1_drop.to(base_img.device)
                if keep is not None and int(keep.sum()) >= 2:
                    # per-record ratios over the samples whose P1 saw their own metadata
                    deltas = {k: ([d[keep] for d in v] if isinstance(v, list) else v[keep])
                              for k, v in deltas.items()}
                    bases = {k: b[keep] for k, b in bases.items()}
                self._diag.update(self.p1_ratios(deltas, bases))
            if p2_drop is not None:
                self._diag["p2_drop_frac"] = float(p2_drop.float().mean())
            if p1_drop is not None:
                self._diag["p1_drop_frac"] = float(p1_drop.float().mean())

        # ---- 6. norm_out (AdaLayerNormContinuous, emb with the P1 delta) ----
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        hidden_states = tf.norm_out.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        out = tf.proj_out(hidden_states)
        if return_hidden:
            return out, {"hidden": hiddens,
                         "offsets": {"n_txt": n_txt, "K": self.K if use_md else 0,
                                     "n_img": n_img, "n_ctx": n_ctx}}
        return out

    # ------------------------------------------------------------------ diagnostics

    @staticmethod
    @torch.no_grad()
    def p1_ratios(deltas: dict, bases: dict) -> dict:
        """D1 method on the head outputs (diagnostics/p1/REPORT.md D1): constant =
        batch mean of a head's delta, per-record = deviation from it, both as a ratio
        to ||mod_t(temb)|| per sample. Over a training micro-batch the timestep varies
        per sample, so "per-record" here also holds the t-dependence of the delta."""
        def one(d, b):
            d, nb = d.float(), b.float().norm(dim=-1).clamp_min(1e-12)
            c = d.mean(0, keepdim=True)
            return (float((c.norm(dim=-1) / nb).mean()),
                    float(((d - c).norm(dim=-1) / nb).mean()))
        rows = {}
        for k in ("img", "txt", "sgl"):
            for i, d in enumerate(deltas[k]):
                rows[f"{k}_{i}"] = one(d, bases[k])
        out_c, out_p = one(deltas["out"], bases["out"])
        blk = list(rows.values())
        return {"p1/const_dmu_blocks": sum(r[0] for r in blk) / len(blk),
                "p1/per_dmu_blocks": sum(r[1] for r in blk) / len(blk),
                "p1/const_dmu_double0": rows["img_0"][0],
                "p1/per_dmu_double0": rows["img_0"][1],
                "p1/const_dmu_normout": out_c, "p1/per_dmu_normout": out_p}

    # ------------------------------------------------------------------ gate check

    @torch.no_grad()
    def assert_identity(self, batch, atol=0.0):
        """transformer.py:254-269 semantics: tokens off, compare with stock FLUX.2."""
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
        assert err <= atol, f"P1-perblock is not identity at init: max|delta|={err}"
        return err


def build_flowsat_c_p1pb(model_id="black-forest-labs/FLUX.2-klein-base-4B",
                         torch_dtype=torch.bfloat16, resolution: int = 512, **kw):
    from diffusers import Flux2Transformer2DModel
    tf = Flux2Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch_dtype)
    hw = resolution // 16
    kw.setdefault("latent_hw", (hw, hw))
    return FlowSatCP1PB(tf, **kw)
