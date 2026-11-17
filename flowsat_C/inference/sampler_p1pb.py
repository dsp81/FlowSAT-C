"""
sampler_p1pb.py — inference for the P1-perblock model (models/transformer_p1pb.py).

The Euler / 3-way CFG math is inference/sampler.py's, imported unchanged: it only
calls model(...), and FlowSatCP1PB.forward applies the per-block P1 heads on
every branch (the two null-metadata branches get the heads' output for the
learned null embeddings, exactly as the old P1 did for to_temb). What this file
adds is the part that is model-specific:

  * sample() refuses any model that is not a FlowSatCP1PB, so an old FlowSatC
    checkpoint can never be sampled here by accident (or vice versa);
  * load_p1pb(): rebuilds the model with the run's P1 config (p1pb_config.json,
    written by train_p1pb.py beside the checkpoints) and loads the weights with
    the same loader train_p1pb uses, failing on any missing / unexpected tensor;
  * a CLI that generates N images for the first N category-balanced records.
P2 path dropout is a training-time device and is never applied here.

    python -m flowsat_C.inference.sampler_p1pb --model-id $MODEL \
        --ckpt runs/warmup_p1pb/ckpt-500 --manifest $VAL --n 4 --out-dir /tmp/x
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from flowsat_C.inference.sampler import per_field_delta  # noqa: F401  (re-export)
from flowsat_C.inference.sampler import sample as _sample
from flowsat_C.models.transformer_p1pb import FlowSatCP1PB, build_flowsat_c_p1pb


def sample(model, *args, **kwargs):
    m = getattr(model, "module", model)
    if not isinstance(m, FlowSatCP1PB):
        raise TypeError(f"sampler_p1pb needs a FlowSatCP1PB, got {type(m).__name__}")
    return _sample(model, *args, **kwargs)


def read_p1pb_config(ckpt: str) -> dict:
    for d in (ckpt, os.path.dirname(os.path.normpath(ckpt))):
        p = os.path.join(d, "p1pb_config.json")
        if os.path.isfile(p):
            return json.load(open(p))
    return {}


def load_p1pb(model_id: str, ckpt: str, glob_norm=None, p1_rank=None, resolution=512):
    """-> FlowSatCP1PB at stage 2 with the checkpoint's weights (bf16 backbone)."""
    from flowsat_C.models.encoder import EncoderConfig, MetadataSpec
    from flowsat_C.training.train_p1pb import load_weights_only
    cfg = read_p1pb_config(ckpt)
    glob_norm = glob_norm or cfg.get("p1_glob_norm", "ln_mag")
    p1_rank = p1_rank or cfg.get("p1_rank", 128)
    model = build_flowsat_c_p1pb(model_id, resolution=resolution,
                                 enc_cfg=EncoderConfig(sh_degree=cfg.get("sh_degree", 16)),
                                 spec=MetadataSpec(), glob_norm=glob_norm, p1_rank=p1_rank)
    model.set_stage(2)
    res = load_weights_only(model, ckpt)
    if res is not None and (res["missing"] or res["unexpected"] or res["shape_mismatch"]):
        raise SystemExit(f"{ckpt}: partial load {res}")
    print(f"P1-perblock model: glob_norm={glob_norm} rank={p1_rank}", flush=True)
    return model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--w-txt", type=float, default=2.0)
    ap.add_argument("--w-md", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--glob-norm", default=None)
    ap.add_argument("--p1-rank", type=int, default=None)
    a = ap.parse_args()
    torch.set_num_threads(4)
    dev = "cuda"
    from flowsat_C.data.dataset import CaptionPolicy, FMoWCaptionDataset, make_collate
    from flowsat_C.training.flux2_io import load_flux2_io
    from flowsat_C.training.validation import category_balanced_order
    from accelerate.utils import send_to_device

    model = load_p1pb(a.model_id, a.ckpt, a.glob_norm, a.p1_rank).to(dev).eval()
    lat_enc, txt_enc = load_flux2_io(a.model_id, dev, max_sequence_length=256)
    ds = FMoWCaptionDataset(a.manifest, resolution=512, return_pixels=False,
                            caption_policy=CaptionPolicy(1.0, 0.0, 0.0), seed=0)
    order = category_balanced_order(ds.records, seed=0)[:a.n]
    items = [ds[i] for i in order]
    batch = make_collate()(items)
    md = send_to_device(batch["metadata"], dev)
    S = 512 // 16
    img_ids = torch.zeros(S * S, 4, device=dev)
    hw = torch.arange(S, device=dev).float()
    img_ids[:, 1], img_ids[:, 2] = hw.repeat_interleave(S), hw.repeat(S)
    cond, txt_ids = txt_enc.encode(batch["caption"])
    null_emb, _ = txt_enc.null(len(items))
    noise = torch.stack([torch.randn(S * S, 128, generator=torch.Generator().manual_seed(
        a.seed * 1_000_003 + p)) for p in range(len(items))]).to(dev, torch.bfloat16)
    lat = sample(model, noise, cond, null_emb, img_ids, txt_ids, md,
                 num_steps=a.steps, w_txt=a.w_txt, w_md=a.w_md)
    im = lat_enc.decode(lat, model.latent_hw).clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
    os.makedirs(a.out_dir, exist_ok=True)
    torch.save(dict(images=im.cpu(), keys=[ds.records[i]["key"] for i in order],
                    captions=batch["caption"], ckpt=a.ckpt), os.path.join(a.out_dir, "images.pt"))
    from PIL import Image
    grid = torch.cat(list(im.cpu()), dim=2).permute(1, 2, 0).numpy()
    Image.fromarray(grid).save(os.path.join(a.out_dir, "grid.png"))
    print(f"wrote {len(items)} images -> {a.out_dir} (finite={bool(torch.isfinite(lat).all())}, "
          f"pixel std {im.float().std():.1f})", flush=True)


if __name__ == "__main__":
    main()
