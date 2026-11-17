"""
flowsat_c_sampler.py — Euler sampler with composed three-way guidance.

v1's CFG passed REAL metadata to both branches, so the metadata guidance scale
was fixed at 1.0 -- metadata was never guided at all. This is close to free to
fix and it is what "controllable" means to a reviewer: an inference-time knob.

Composed form (3 forward passes):

  v_hat = v(0,0)
        + w_txt * [ v(c_txt, 0)     - v(0, 0) ]
        + w_md  * [ v(c_txt, c_md)  - v(c_txt, 0) ]

Ordering matters. Conditioning on text FIRST, then metadata ON TOP of text, gives
w_md the meaning "how much does metadata modify the text-conditioned prediction".
That is exactly the controllability axis, and it decouples: raise w_md for
control, hold w_txt wherever FID is best.

Nulls are the encoder's LEARNED per-group null embeddings, never zero-filled
metadata -- a normalised zero decodes to a real wrong physical value (lon=-180,
year=1980), i.e. OOD conditioning rather than absent conditioning.
"""

from __future__ import annotations

import torch

from flowsat_C.models.encoder import GROUPS


@torch.no_grad()
def sample(model,
           latents: torch.Tensor,
           encoder_hidden_states: torch.Tensor,
           null_encoder_hidden_states: torch.Tensor,
           img_ids: torch.Tensor,
           txt_ids: torch.Tensor,
           metadata: dict,
           num_steps: int = 28,
           w_txt: float = 3.5,
           w_md: float = 2.0,
           shift: float = 3.0,
           guidance: torch.Tensor | None = None,
           use_year_embedding: bool = True,
           generator=None):
    """
    Set w_md = 1.0 to recover v1 behaviour (metadata unguided) for an
    apples-to-apples ablation of the guidance mechanism itself.
    """
    device, dtype = latents.device, latents.dtype
    B = latents.shape[0]

    # shifted uniform schedule, t: 1 -> 0
    t = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=torch.float32)
    t = shift * t / (1.0 + (shift - 1.0) * t)

    x = latents.clone()
    all_null = list(GROUPS)

    for i in range(num_steps):
        tc = t[i].expand(B).to(dtype)
        dt = (t[i + 1] - t[i]).to(dtype)

        def run(ctx, md_null):
            return model(hidden_states=x, encoder_hidden_states=ctx, timestep=tc,
                         img_ids=img_ids, txt_ids=txt_ids, metadata=metadata,
                         guidance=guidance, force_null=md_null,
                         use_year_embedding=use_year_embedding)

        v_uncond = run(null_encoder_hidden_states, all_null)   # v(0, 0)
        v_txt    = run(encoder_hidden_states,      all_null)   # v(c_txt, 0)
        v_full   = run(encoder_hidden_states,      None)       # v(c_txt, c_md)

        v = v_uncond + w_txt * (v_txt - v_uncond) + w_md * (v_full - v_txt)
        x = x + dt * v

    return x


@torch.no_grad()
def per_field_delta(model, x, encoder_hidden_states, img_ids, txt_ids,
                    metadata, timestep, guidance=None) -> dict:
    """
    Per-field conditioning strength, for MetaCF only (O(G) passes -- correct for
    evaluation, too expensive for generation).

        Delta_i = v(c_txt, c_md) - v(c_txt, c_md^{-i})

    Also the ablate-at-inference cannibalisation probe: if nulling `cov`
    collapses geo controllability, covariates have cannibalised the coordinates
    and their dropout rate should go up.
    """
    def run(md_null):
        return model(hidden_states=x, encoder_hidden_states=encoder_hidden_states,
                     timestep=timestep, img_ids=img_ids, txt_ids=txt_ids,
                     metadata=metadata, guidance=guidance, force_null=md_null)

    full = run(None)
    out = {}
    for g in GROUPS:
        d = full - run([g])
        out[g] = d.float().flatten(1).norm(dim=1).mean().item()
    return out
