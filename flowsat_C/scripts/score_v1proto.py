"""
score_v1proto.py -- score generated FlowSAT-C images under the FlowSat-v1 evaluation protocol.

Run in the pinned evaluation env (requirements/eval.txt: torch 2.4.0, torchmetrics 1.5.2,
torch-fidelity 0.4.0, transformers 4.49.0). Every metric goes through FlowSat v1's eval_common
(vendored in flowsat_C/evaluation/eval_common.py):
read_real_image (rasterio, 2-98 percentile stretch, centre crop, bilinear 512), setup_metrics
(FID 2048 / IS / LPIPS alex / SSIM), ManualCLIPScorer (openai/clip-vit-base-patch16, x100, 77 tok).

Reference set = the first 10,000 v1 ids (--ids), all real images enter FID as reals.
Fakes: the chunks written by gen_v1proto.py (ids missing there = the caption / manifest gap;
counted, never substituted). Reported:
  * fid          10k reals vs all generated fakes (THE protocol number)
  * fid_matched  reals restricted to ids that have a fake (same ids on both sides)
  * clip_score   generated image vs OUR caption;  clip_real = real image vs our caption (reference)
  * ssim / lpips PAIRED with each id's own real image (not image quality)
  * is_mean/std  on fakes
Curve points at v1-list positions --curve-at (e.g. 1000,5000) and at the end (10,000).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from flowsat_C.evaluation.eval_common import (ManualCLIPScorer, pil_to_uint8_tensor, read_real_image,  # noqa: E402
                                            setup_metrics, uint8_tensor_to_pils, update_paired_metrics)


class Reals(Dataset):
    def __init__(self, ids, roots):
        self.ids, self.roots = ids, roots

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        x = self.ids[i]
        for r in self.roots:
            p = Path(r) / x["seq_id"] / f"{x['stem']}.tif"
            if p.exists():
                try:  # as v1 evaluate_fmow.py: a read failure drops the sample (e.g. 241 MP tif > PIL bomb limit)
                    pil = read_real_image(p, 512)
                except Exception as e:
                    print(f"UNREAD {p}: {type(e).__name__}: {str(e)[:120]}", flush=True)
                    return i, None, None
                if pil is not None:
                    return i, pil_to_uint8_tensor(pil), str(p)
        return i, None, None


def collate(b):
    return b


def version(d):
    from importlib.metadata import version as v
    try:
        return v(d)
    except Exception:
        try:
            import importlib
            return importlib.import_module(d.replace("-", "_")).__version__
        except Exception:
            return "?"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", required=True)
    ap.add_argument("--gen-dir", required=True, help="dir with chunk_*.pt from gen_v1proto")
    ap.add_argument("--reals-root", action="append", required=True, help="tried in order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--curve-at", default="1000,5000")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ids = json.load(open(a.ids))
    N = len(ids)

    # fake index: id -> (chunk path, row)
    fidx, caps, meta = {}, {}, None
    chunks = sorted(glob.glob(os.path.join(a.gen_dir, "chunk_*.pt")))
    for c in chunks:
        d = torch.load(c, map_location="cpu")
        meta = meta or {k: d[k] for k in ("ckpt", "steps", "w_txt", "w_md", "seed")}
        for j, (i, cap) in enumerate(zip(d["ids"], d["captions"])):
            fidx[i] = (c, j)
            caps[i] = cap
    print(f"{len(fidx)} generated images in {len(chunks)} chunks; reference set {N}", flush=True)

    m = setup_metrics(dev)
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid_matched = FrechetInceptionDistance(feature=2048, normalize=False).to(dev)
    clip = ManualCLIPScorer(device=dev)
    clip_real = ManualCLIPScorer(device=dev)
    curve_at = sorted(int(x) for x in a.curve_at.split(",") if x)
    curve = []
    cache = {}
    n_real = n_fake = n_unread = 0
    t0 = time.time()

    def fake(i):
        c, j = fidx[i]
        if c not in cache:
            cache.clear()
            cache[c] = torch.load(c, map_location="cpu")["images"]
        return cache[c][j]

    def snap(pos):
        is_m, is_s = m["is"].compute()
        return dict(at_v1_pos=pos, n_real=n_real, n_fake=n_fake,
                    fid=float(m["fid"].compute()), fid_matched=float(fid_matched.compute()),
                    clip_score=float(np.mean(clip.scores)), clip_real=float(np.mean(clip_real.scores)),
                    ssim=float(m["ssim"].compute()), lpips=float(m["lpips"].compute()),
                    is_mean=float(is_m), is_std=float(is_s))

    dl = DataLoader(Reals(ids, a.reals_root), batch_size=a.batch, num_workers=a.workers, collate_fn=collate)
    for b in dl:
        good = [(i, t) for i, t, _ in b if t is not None]
        n_unread += len(b) - len(good)
        if good:
            real = torch.stack([t for _, t in good]).to(dev)
            m["fid"].update(real, real=True)
            n_real += len(good)
        pairs = [(i, t) for i, t in good if f"{ids[i]['seq_id']}/{ids[i]['stem']}" in fidx]
        if pairs:
            keys = [f"{ids[i]['seq_id']}/{ids[i]['stem']}" for i, _ in pairs]
            fk = torch.stack([fake(k) for k in keys]).to(dev)
            rl = torch.stack([t for _, t in pairs]).to(dev)
            m["fid"].update(fk, real=False)
            m["is"].update(fk)
            fid_matched.update(rl, real=True)
            fid_matched.update(fk, real=False)
            update_paired_metrics(m, fk.float() / 255.0, rl.float() / 255.0)
            cl = [caps[k] for k in keys]
            clip.update(uint8_tensor_to_pils(fk), cl)
            clip_real.update(uint8_tensor_to_pils(rl), cl)
            n_fake += len(pairs)
        last = b[-1][0] + 1
        while curve_at and last >= curve_at[0]:
            p = snap(curve_at.pop(0))
            curve.append(p)
            print(f"[curve] {json.dumps(p)}", flush=True)
        if (last // a.batch) % 20 == 0:
            el = time.time() - t0
            print(f"{last}/{N} reals | fakes {n_fake} | unread {n_unread} | {el / 60:.1f} min", flush=True)
    final = snap(N)
    curve.append(final)
    idl = [f"{x['seq_id']}/{x['stem']}" for x in ids]
    cs = hashlib.sha256()
    for i in idl:
        if i in caps:
            cs.update(f"{i}\t{caps[i]}\n".encode())
    res = dict(final=final, curve=curve, n_unreadable_reals=n_unread,
               n_missing_fakes=N - len([i for i in idl if i in fidx]),
               protocol=dict(reference_set=hashlib.sha256("\n".join(idl).encode()).hexdigest()[:12],
                             caption_set_ours=cs.hexdigest()[:12], num_samples=N, resolution=512,
                             clip_model="openai/clip-vit-base-patch16",
                             torch=torch.__version__, torchmetrics=version("torchmetrics"),
                             torch_fidelity=version("torch-fidelity"), transformers=version("transformers"),
                             reals_roots=a.reals_root),
               generation=meta, wall_min=(time.time() - t0) / 60,
               notes=dict(fid="10k reals (v1 reference set) vs all generated fakes; ids without our caption/"
                              "metadata have no fake (gap), never substituted",
                          fid_matched="same ids on both sides",
                          ssim_lpips="paired with each id's own real image; not image quality",
                          clip="our agnostic captions (each model uses its own captions); clip_real = reals vs same captions"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("FINAL", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
