"""
diagnostics.py — instrumentation for FlowSAT-C.

The v1 failure was INVISIBLE to the metrics being watched. FID fell, CLIP rose,
controllability was near zero and nobody could see it. Everything here exists so
that cannot happen again.

Four things this architecture exposes for free, all of which are figures:
  1. alpha_l profile        -- which depths metadata acts at
  2. gate bias over time    -- when and where metadata was admitted
  3. per-modality attention mass -- is text crowding out metadata; is
                                    text<->metadata interaction happening at all
  4. per-field delta norms  -- which fields actually condition anything

(3) doubles as the cannibalisation monitor: if text<->metadata attention mass
stays near zero throughout training, the interaction story is not happening and
you will know months before writing it up.
"""

from __future__ import annotations

import json
import os

import torch

from flowsat_C.models.encoder import GROUPS


def slice_tree(x, n):
    """First n rows of every tensor in a (nested) metadata dict. The covariate
    sub-dict is nested, so a flat slice left it at full batch size."""
    if torch.is_tensor(x):
        return x[:n]
    if isinstance(x, dict):
        return {k: slice_tree(v, n) for k, v in x.items()}
    return x


class AttentionMassProbe:
    """
    Hooks the double-stream attention to measure how softmax mass distributes
    across [text, metadata, image].

    Recomputing the attention map is O(n^2) in memory, so run this on a SMALL
    batch at a low frequency, never every step.
    """

    def __init__(self, model):
        self.model = model
        self.records = []
        self.handles = []

    def _hook(self, idx):
        # Pre-hook on the block's attention module, recomputing the REAL map:
        # q/k projections, QK-norm, RoPE and the attention mask (which carries
        # the metadata gate). The previous proxy (raw hidden-state dot products)
        # read txt->md = img->md = 0.0 regardless of the gate.
        def fn(attn, args, kwargs):
            from diffusers.models.transformers.transformer_flux2 import (
                _get_qkv_projections, apply_rotary_emb)
            hs = kwargs.get("hidden_states", args[0] if args else None)
            ehs = kwargs.get("encoder_hidden_states")
            if hs is None or ehs is None:
                return
            q, k, _, eq, ek, _ = _get_qkv_projections(attn, hs, ehs)
            q = attn.norm_q(q.unflatten(-1, (attn.heads, -1)))
            k = attn.norm_k(k.unflatten(-1, (attn.heads, -1)))
            eq = attn.norm_added_q(eq.unflatten(-1, (attn.heads, -1)))
            ek = attn.norm_added_k(ek.unflatten(-1, (attn.heads, -1)))
            q, k = torch.cat([eq, q], 1), torch.cat([ek, k], 1)
            rope = kwargs.get("image_rotary_emb")
            if rope is not None:
                q = apply_rotary_emb(q, rope, sequence_dim=1)
                k = apply_rotary_emb(k, rope, sequence_dim=1)
            q, k = q.float().transpose(1, 2), k.float().transpose(1, 2)  # (B,H,n,d)
            logits = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
            mask = kwargs.get("attention_mask")
            if mask is not None:
                logits = logits + mask.float()
            n_ctx = ehs.shape[1]
            K = getattr(self.model, "K", 0) if mask is not None else 0
            n_txt = n_ctx - K
            p = logits.softmax(-1).mean((0, 1))     # (n, n)
            sl = dict(txt=slice(0, n_txt),
                      md=slice(n_txt, n_ctx),
                      img=slice(n_ctx, p.shape[0]))
            rec = {"block": idx}
            for a in sl:
                for b in sl:
                    blk = p[sl[a], sl[b]]
                    rec[f"{a}->{b}"] = float(blk.sum() / max(blk.shape[0], 1))
            self.records.append(rec)
        return fn

    def __enter__(self):
        for i, b in enumerate(self.model.tf.transformer_blocks):
            self.handles.append(b.attn.register_forward_pre_hook(self._hook(i),
                                                                 with_kwargs=True))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


class DiagnosticLogger:
    def __init__(self, accelerator, out_dir: str):
        self.acc = accelerator
        self.path = os.path.join(out_dir, "diagnostics.jsonl")
        os.makedirs(out_dir, exist_ok=True)

    def gather(self, model) -> dict:
        """Cheap scalars, safe every log step."""
        d = getattr(model, "_diag", {}) or {}
        out = {}
        if "gate_bias" in d:
            out["gate/bias"] = d["gate_bias"]
        if "temb_add_norm" in d:
            out["graft/temb_add_norm"] = d["temb_add_norm"]
        for name in ("alpha_double", "alpha_single"):
            if name in d:
                a = d[name]
                out[f"alpha/{name}_mean"] = float(a.mean())
                out[f"alpha/{name}_std"] = float(a.std())
                out[f"alpha/{name}_max"] = float(a.max())
        if hasattr(model, "encoder"):
            import torch.nn.functional as F
            a = F.softplus(model.encoder.geo.log_a).detach().float()
            # learned SH spectrum: shows what ANGULAR SCALE the model actually
            # uses. Direct answer to "you never ablated your frequency bands."
            out["sh/spectrum_mass_low"] = float(a[:5].sum() / a.sum())
            out["sh/spectrum_mass_high"] = float(a[-5:].sum() / a.sum())
        return out

    @torch.no_grad()
    def attention_mass(self, model, batch, max_batch: int = 2) -> dict:
        small = {k: slice_tree(v, max_batch) if k != "caption" else v
                 for k, v in batch.items()}
        with AttentionMassProbe(model) as p:
            model(hidden_states=small["latents"],
                  encoder_hidden_states=small["text_emb"],
                  timestep=torch.full((max_batch,), 0.5, device=small["latents"].device),
                  img_ids=small["img_ids"], txt_ids=small["txt_ids"],
                  metadata=small.get("metadata"))
            recs = p.records
        if not recs:
            return {}
        keys = [k for k in recs[0] if k != "block"]
        return {f"attn/{k}": sum(r[k] for r in recs) / len(recs) for k in keys}

    @torch.no_grad()
    def controllability_probe(self, model, batch) -> dict:
        """
        Per-field conditioning strength: ||v(full) - v(field nulled)||.

        Not full MetaCF -- just enough to see the TREND. If a field's delta is
        flat near zero while loss falls, that field is being ignored and you can
        intervene before burning the run.
        """
        from flowsat_C.inference.sampler import per_field_delta
        x = batch["latents"][:2]
        t = torch.full((2,), 0.7, device=x.device)   # high noise: structure
        d = per_field_delta(model, x, batch["text_emb"][:2],
                            batch["img_ids"], batch["txt_ids"],
                            slice_tree(batch["metadata"], 2), t)
        out = {f"ctrl/{g}": d[g] for g in GROUPS}
        tot = sum(d.values()) or 1.0
        for g in GROUPS:                              # share, not just magnitude
            out[f"ctrl_share/{g}"] = d[g] / tot
        return out

    def write(self, step: int, payload: dict):
        if not self.acc.is_main_process:
            return
        with open(self.path, "a") as f:
            f.write(json.dumps({"step": step, **payload}) + "\n")
