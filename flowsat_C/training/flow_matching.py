"""
flow_matching_c.py — rectified-flow objective for FlowSAT-C.

Timestep sampling is NOT a neutral choice for controllability.

Global structure -- what objects exist and where they sit in the frame -- is
committed at HIGH noise. Texture is refined at low noise. Geography is therefore
a high-t decision; GSD and cloud, being uniform effects, act across the range and
partly at low t.

So a distribution concentrated at t~0.5 (the m=0 default) UNDERSAMPLES exactly
the timesteps where location control is decided. That is a plausible third cause
of the v1 field ordering, independent of caption leakage and AdaLN bandwidth, and
it has not been tested.

Recipe: keep the backbone default in stage 1 (you want its FID behaviour intact),
then shift m positive in stage 2 while the metadata pathway opens.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_timesteps(batch_size: int,
                     device,
                     dtype=torch.float32,
                     mean: float = 0.0,
                     std: float = 1.0,
                     shift: float = 3.0,
                     high_noise_frac: float = 0.0,
                     high_noise_min: float = 0.7,
                     generator=None) -> torch.Tensor:
    """
    Logit-normal t with SD3/FLUX resolution shift.

        u ~ N(mean, std^2),  t = sigmoid(u),  t' = shift*t / (1 + (shift-1)*t)

    mean > 0 pushes mass toward high noise (structure); mean < 0 toward low noise.

    high_noise_frac: fraction of the batch forced into [high_noise_min, 1). This
    is the surgical option -- more moving parts than shifting `mean`, so prefer
    `mean` first and only reach for this if the ablation says it is not enough.
    """
    u = torch.randn(batch_size, device=device, dtype=torch.float32,
                    generator=generator) * std + mean
    t = torch.sigmoid(u)
    t = shift * t / (1.0 + (shift - 1.0) * t)

    if high_noise_frac > 0.0:
        n = int(round(batch_size * high_noise_frac))
        if n > 0:
            r = torch.rand(n, device=device, generator=generator)
            t[:n] = high_noise_min + (1.0 - high_noise_min) * r
            t = t[torch.randperm(batch_size, device=device, generator=generator)]
    return t.to(dtype)


def rectified_flow_loss(model_fn,
                        latents: torch.Tensor,
                        timesteps: torch.Tensor,
                        noise: torch.Tensor | None = None,
                        reduction: str = "mean"):
    """
    x_t = (1-t) * x_0 + t * eps ;  target velocity v = eps - x_0.

    model_fn(x_t, t) -> predicted velocity, same shape as latents.
    Returns (loss, per_sample_loss, x_t, target).
    """
    if noise is None:
        noise = torch.randn_like(latents)
    t = timesteps.view(-1, *([1] * (latents.dim() - 1))).to(latents.dtype)

    x_t = (1.0 - t) * latents + t * noise
    target = noise - latents

    pred = model_fn(x_t, timesteps)
    per_sample = F.mse_loss(pred.float(), target.float(), reduction="none")
    per_sample = per_sample.flatten(1).mean(1)

    loss = per_sample.mean() if reduction == "mean" else per_sample
    return loss, per_sample, x_t, target


@torch.no_grad()
def loss_by_timestep_bucket(per_sample: torch.Tensor,
                            timesteps: torch.Tensor,
                            n_buckets: int = 10) -> dict:
    """
    Diagnostic. If stage-2 loss degrades only in high-t buckets, the metadata
    pathway is disrupting structure formation -- lower the backbone LR or w_md
    rather than debugging blind.
    """
    b = (timesteps.clamp(0, 0.999) * n_buckets).long()
    out = {}
    for k in range(n_buckets):
        m = b == k
        if m.any():
            out[f"loss/t_{k / n_buckets:.1f}"] = per_sample[m].mean().item()
    return out
