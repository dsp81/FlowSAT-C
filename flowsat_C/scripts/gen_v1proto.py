"""
gen_v1proto.py -- generate FlowSAT-C (P1PB) images for the FlowSat-v1 evaluation protocol.

Reference set = the first 10,000 captioned fMoW-full test samples in (seq_id, stem) order
(the FlowSat v1 evaluation protocol), listed in --ids (eval_data/fmow_v1_ids_10k.json).
FlowSAT-C uses our captions and the real metadata from
test_cov.jsonl (joined on `test_id`, never on `key`). Ids with no record there are skipped
and listed in skipped.json; the scorer handles the gap.

Workers on any GPUs cooperate through an on-disk chunk queue (atomic mkdir locks);
relaunch to resume, finished chunks are skipped. Each record gets noise seeded by its
position in the v1 list, so images do not depend on batch size or worker count.
Output: <out>/gen/chunk_XXXX.pt = {images uint8 (B,3,512,512), ids, captions, ...}.

    python -m flowsat_C.scripts.gen_v1proto --model-id $MODEL --ckpt <snapshot.pt> \
        --ids v1_ids_10k.json --test-manifest test_cov.jsonl --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

CHUNK = 64


def build_subset(a):
    ids = json.load(open(a.ids))
    want = {f"{x['seq_id']}/{x['stem']}": p for p, x in enumerate(ids)}
    recs = {}
    for line in open(a.test_manifest):
        r = json.loads(line)
        if r.get("test_id") in want and r["test_id"] not in recs:
            recs[r["test_id"]] = r
    keep, skipped = [], []
    for tid, p in sorted(want.items(), key=lambda kv: kv[1]):
        if tid in recs and recs[tid].get("caption"):
            keep.append({**recs[tid], "v1_pos": p})
        else:
            skipped.append({"id": tid, "v1_pos": p, "reason": "no test_cov record" if tid not in recs else "no caption"})
    return keep, skipped


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--w-txt", type=float, default=2.0)
    ap.add_argument("--w-md", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N kept records")
    a = ap.parse_args()
    torch.set_num_threads(2)
    dev = "cuda"
    from accelerate.utils import send_to_device
    from flowsat_C.data.dataset import CaptionPolicy, FMoWCaptionDataset, make_collate
    from flowsat_C.inference.sampler_p1pb import load_p1pb, sample
    from flowsat_C.training.flux2_io import load_flux2_io

    os.makedirs(os.path.join(a.out_dir, "gen"), exist_ok=True)
    sub = os.path.join(a.out_dir, "subset_manifest.jsonl")
    if not os.path.exists(sub):
        keep, skipped = build_subset(a)
        with open(sub + ".tmp", "w") as f:
            for r in keep:
                f.write(json.dumps(r) + "\n")
        os.replace(sub + ".tmp", sub)
        json.dump(skipped, open(os.path.join(a.out_dir, "skipped.json"), "w"), indent=1)
        print(f"subset: {len(keep)} records kept, {len(skipped)} skipped", flush=True)
    ds = FMoWCaptionDataset(sub, resolution=512, return_pixels=False,
                            caption_policy=CaptionPolicy(1.0, 0.0, 0.0), seed=0)
    n = len(ds) if not a.limit else min(a.limit, len(ds))
    model = load_p1pb(a.model_id, a.ckpt).to(dev).eval()
    lat_enc, txt_enc = load_flux2_io(a.model_id, dev, max_sequence_length=256)
    S = 512 // 16
    img_ids = torch.zeros(S * S, 4, device=dev)
    hw = torch.arange(S, device=dev).float()
    img_ids[:, 1], img_ids[:, 2] = hw.repeat_interleave(S), hw.repeat(S)
    collate = make_collate()
    t_start, done_here = time.time(), 0
    for c in range(math.ceil(n / CHUNK)):
        out = os.path.join(a.out_dir, "gen", f"chunk_{c:04d}.pt")
        lock = out + ".lock"
        if os.path.exists(out):
            continue
        try:
            os.mkdir(lock)
        except FileExistsError:
            continue
        try:
            lo, hi = c * CHUNK, min((c + 1) * CHUNK, n)
            imgs, caps, ids, vpos = [], [], [], []
            t0 = time.time()
            for s in range(lo, hi, a.batch):
                idx = list(range(s, min(s + a.batch, hi)))
                items = [ds[i] for i in idx]
                batch = collate(items)
                md = send_to_device(batch["metadata"], dev)
                cond, txt_ids = txt_enc.encode(batch["caption"])
                null_emb, _ = txt_enc.null(len(idx))
                recs = [ds.records[i] for i in idx]
                noise = torch.stack([torch.randn(S * S, 128, generator=torch.Generator().manual_seed(
                    a.seed * 1_000_003 + r["v1_pos"])) for r in recs]).to(dev, torch.bfloat16)
                lat = sample(model, noise, cond, null_emb, img_ids, txt_ids, md,
                             num_steps=a.steps, w_txt=a.w_txt, w_md=a.w_md)
                im = lat_enc.decode(lat, model.latent_hw).clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
                if c == 0 and s == lo and int(im.max()) == 0:
                    raise SystemExit("[fatal] first batch decoded to pure black")
                imgs.append(im.cpu()); caps += batch["caption"]
                ids += [r["test_id"] for r in recs]; vpos += [r["v1_pos"] for r in recs]
            torch.save(dict(images=torch.cat(imgs), ids=ids, v1_pos=vpos, captions=caps, ckpt=a.ckpt,
                            steps=a.steps, w_txt=a.w_txt, w_md=a.w_md, seed=a.seed), out + ".tmp")
            os.replace(out + ".tmp", out)
            done_here += hi - lo
            el = time.time() - t_start
            print(f"[{time.strftime('%H:%M:%S')}] chunk {c:4d} ({hi - lo} img) {(time.time() - t0) / (hi - lo):.2f} s/img"
                  f" | this worker {done_here} img, {el / done_here:.2f} s/img avg", flush=True)
        finally:
            os.rmdir(lock)
    print("GEN DONE", flush=True)


if __name__ == "__main__":
    main()
