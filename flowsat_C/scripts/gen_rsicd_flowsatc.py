"""
gen_rsicd_flowsatc.py -- FlowSAT-C (P1PB) on RSICD (zero-shot / OOD), for scripts/score_rsicd.py.

RSICD has no metadata, so every metadata group is forced to its LEARNED NULL in every branch (force_null = all
groups). The 3-way CFG of inference/sampler.py then reduces exactly to 2-way text CFG,
    v = v(0, 0) + w_txt * (v(c_txt, 0) - v(0, 0))          (= the w_md term multiplied by zero),
so only those two branches are computed here (Euler, shift 3, same schedule as sampler.sample).
The metadata dict passed to the model is a placeholder (one test_cov record repeated); with all groups forced null
the output does not depend on it (checked by --check-null in the smoke run).
Captions: our RSICD captions (--table). Noise seeded by table position. Output 512 -> 256 (antialiased bilinear,
to_res) in chunk files, with chunk locks so several GPUs can share a run.

    python -m flowsat_C.scripts.gen_rsicd_flowsatc --model-id $MODEL --ckpt flowsat_c_35k.safetensors \
        --table rsicd_captions.jsonl --test-manifest test_cov.jsonl --out-dir <dir> [--limit 16]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

CHUNK = 64


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--table", required=True, help="rsicd_captions.jsonl (id, v1_pos, caption)")
    ap.add_argument("--test-manifest", required=True, help="fMoW test_cov.jsonl (placeholder metadata only)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--w-txt", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check-null", action="store_true", help="check every batch (the first batch of chunk 0 is always checked)")
    a = ap.parse_args()
    torch.set_num_threads(2)
    dev = "cuda"
    from accelerate.utils import send_to_device
    from flowsat_C.data.dataset import CaptionPolicy, FMoWCaptionDataset, make_collate
    from flowsat_C.inference.sampler import GROUPS
    from flowsat_C.inference.sampler_p1pb import load_p1pb
    from flowsat_C.models.transformer_p1pb import FlowSatCP1PB
    from flowsat_C.training.flux2_io import load_flux2_io

    def to_res(x, res):
        """uint8 (B,3,H,W) -> (B,3,res,res), bilinear, antialiased when shrinking."""
        y = torch.nn.functional.interpolate(x.float(), size=(res, res), mode="bilinear", align_corners=False,
                                            antialias=x.shape[-1] > res)
        return y.round().clamp(0, 255).to(torch.uint8)

    rows = [json.loads(l) for l in open(a.table)]
    if a.limit:
        rows = rows[:a.limit]
    os.makedirs(os.path.join(a.out_dir, "gen"), exist_ok=True)

    # placeholder metadata: two different real test records (only used by --check-null to prove independence)
    ph = os.path.join(a.out_dir, "placeholder_md.jsonl")
    if not os.path.exists(ph):
        recs = []
        for line in open(a.test_manifest):
            recs.append(json.loads(line))
            if len(recs) >= 2000:
                break
        with open(ph, "w") as f:
            f.write(json.dumps(recs[0]) + "\n" + json.dumps(recs[1999]) + "\n")
    ds = FMoWCaptionDataset(ph, resolution=512, return_pixels=False, caption_policy=CaptionPolicy(1.0, 0.0, 0.0), seed=0)
    collate = make_collate()

    def placeholder(n, which=0):
        return send_to_device(collate([ds[which]] * n)["metadata"], dev)

    model = load_p1pb(a.model_id, a.ckpt).to(dev).eval()
    assert isinstance(getattr(model, "module", model), FlowSatCP1PB)
    lat_enc, txt_enc = load_flux2_io(a.model_id, dev, max_sequence_length=256)
    S = 512 // 16
    img_ids = torch.zeros(S * S, 4, device=dev)
    hw = torch.arange(S, device=dev).float()
    img_ids[:, 1], img_ids[:, 2] = hw.repeat_interleave(S), hw.repeat(S)
    all_null = list(GROUPS)

    def sample2(noise, cond, null_emb, txt_ids, md):
        B = noise.shape[0]
        t = torch.linspace(1.0, 0.0, a.steps + 1, device=dev, dtype=torch.float32)
        t = 3.0 * t / (1.0 + 2.0 * t)                      # shift 3, as sampler.sample
        x = noise.clone()
        for i in range(a.steps):
            tc = t[i].expand(B).to(x.dtype)
            dt = (t[i + 1] - t[i]).to(x.dtype)

            def run(ctx):
                return model(hidden_states=x, encoder_hidden_states=ctx, timestep=tc, img_ids=img_ids,
                             txt_ids=txt_ids, metadata=md, guidance=None, force_null=all_null,
                             use_year_embedding=True)
            vu, vt = run(null_emb), run(cond)
            x = x + dt * (vu + a.w_txt * (vt - vu))
        return x

    def gen(rs, check=False):
        cond, txt_ids = txt_enc.encode([r["caption"] for r in rs])
        null_emb, _ = txt_enc.null(len(rs))
        noise = torch.stack([torch.randn(S * S, 128, generator=torch.Generator().manual_seed(
            a.seed * 1_000_003 + r["v1_pos"])) for r in rs]).to(dev, torch.bfloat16)
        lat = sample2(noise, cond, null_emb, txt_ids, placeholder(len(rs), 0))
        if a.check_null or check:
            lat2 = sample2(noise, cond, null_emb, txt_ids, placeholder(len(rs), 1))
            d = float((lat.float() - lat2.float()).abs().max())
            print(f"[check-null] max|delta| between placeholder metadata records = {d}", flush=True)
            assert d < 1e-2, "output depends on the placeholder metadata"   # exact 0 expected; tolerance only for kernel nondeterminism
        im = lat_enc.decode(lat, model.latent_hw).clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
        return to_res(im.cpu(), 256)

    t_start, done = time.time(), 0
    for c in range(math.ceil(len(rows) / CHUNK)):
        out = os.path.join(a.out_dir, "gen", f"chunk_{c:04d}.pt")
        lock = out + ".lock"
        if os.path.exists(out):
            continue
        try:
            os.mkdir(lock)
        except FileExistsError:
            continue
        try:
            rs = rows[c * CHUNK:(c + 1) * CHUNK]
            imgs = []
            for s in range(0, len(rs), a.batch):
                im = gen(rs[s:s + a.batch], check=(c == 0 and s == 0))   # first batch of the run: null-md independence
                if c == 0 and s == 0 and int(im.max()) == 0:
                    raise SystemExit("[fatal] first batch decoded to pure black")
                imgs.append(im)
            torch.save(dict(images=torch.cat(imgs), ids=[r["id"] for r in rs], v1_pos=[r["v1_pos"] for r in rs],
                            captions=[r["caption"] for r in rs], eval_set="rsicd", ckpt=a.ckpt, steps=a.steps,
                            w_txt=a.w_txt, w_md=None, seed=a.seed, model="FlowSAT-C",
                            sampler="Euler shift 3, 2-way text CFG, all metadata groups = learned nulls",
                            regime="our RSICD captions, no metadata", native_res=512), out + ".tmp")
            os.replace(out + ".tmp", out)
            done += len(rs)
            print(f"[{time.strftime('%H:%M:%S')}] chunk {c} | {(time.time() - t_start) / done:.2f} s/img", flush=True)
        finally:
            os.rmdir(lock)
    print("GEN DONE", flush=True)


if __name__ == "__main__":
    main()
