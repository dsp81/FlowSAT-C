"""
score_rsicd.py -- RSICD zero-shot (OOD) image-quality scoring, same metric code as the fMoW table (score_v1proto.py).

Run in the pinned evaluation env (requirements/eval.txt: torch 2.4.0, torchmetrics 1.5.2, torch-fidelity 0.4.0,
transformers 4.49.0). Metrics through FlowSat v1's eval_common (flowsat_C/evaluation/eval_common.py): setup_metrics (FID 2048 / IS / LPIPS alex / SSIM),
ManualCLIPScorer (openai/clip-vit-base-patch16, x100, 77 tok).

Reference set = all 10,921 RSICD images (<--rsicd-root>/<id>.png, 224 px PNG, see prepare_rsicd.py), in the order of
the --table file (rsicd_captions.jsonl from the Hugging Face repo). Eval resolution 256: reals bicubic
224 -> 256, fakes were stored at 256 by the generators (antialiased downsample from each model's native size).
Captions = our RSICD captions (the --table file); no metadata (RSICD has none).
Reported as score_v1proto: fid (all reals vs all fakes), fid_matched, clip_score / clip_real, paired ssim / lpips, IS;
curve at table positions --curve-at and at the end.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from flowsat_C.evaluation.eval_common import (ManualCLIPScorer, pil_to_uint8_tensor,  # noqa: E402
                                            setup_metrics, uint8_tensor_to_pils, update_paired_metrics)

RES = 256


class Reals(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        pil = Image.open(self.rows[i]["image"]).convert("RGB").resize((RES, RES), Image.BICUBIC)
        return i, pil_to_uint8_tensor(pil)


def version(d):
    from importlib.metadata import version as v
    try:
        return v(d)
    except Exception:
        return "?"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--table", required=True, help="rsicd_captions.jsonl (id, v1_pos, caption), defines the order")
    ap.add_argument("--rsicd-root", required=True, help="folder of the 10,921 RSICD PNGs written by prepare_rsicd.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--curve-at", default="1000,5000,10000")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(4)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = [json.loads(l) for l in open(a.table)]
    for r in rows:
        r["image"] = os.path.join(a.rsicd_root, r["id"] + ".png")
    N = len(rows)

    fidx, caps, meta = {}, {}, None
    chunks = sorted(glob.glob(os.path.join(a.gen_dir, "chunk_*.pt")))
    for c in chunks:
        d = torch.load(c, map_location="cpu")
        assert d.get("eval_set") == "rsicd", f"{c} is not an RSICD chunk"
        assert tuple(d["images"].shape[1:]) == (3, RES, RES), d["images"].shape
        meta = meta or {k: d.get(k) for k in ("ckpt", "steps", "w_txt", "w_md", "seed", "sampler", "model", "regime",
                                              "native_res")}
        for j, (i, cap) in enumerate(zip(d["ids"], d["captions"])):
            fidx[i] = (c, j)
            caps[i] = cap
    print(f"{len(fidx)} generated images in {len(chunks)} chunks; reference set {N}", flush=True)

    m = setup_metrics(dev)
    from torchmetrics.image.fid import FrechetInceptionDistance
    fid_matched = FrechetInceptionDistance(feature=2048, normalize=False).to(dev)
    clip, clip_real = ManualCLIPScorer(device=dev), ManualCLIPScorer(device=dev)
    curve_at = sorted(int(x) for x in a.curve_at.split(",") if x)
    curve, cache = [], {}
    n_real = n_fake = 0
    t0 = time.time()

    def fake(i):
        c, j = fidx[i]
        if c not in cache:
            cache.clear()
            cache[c] = torch.load(c, map_location="cpu")["images"]
        return cache[c][j]

    def snap(pos):
        is_m, is_s = m["is"].compute()
        return dict(at_pos=pos, n_real=n_real, n_fake=n_fake,
                    fid=float(m["fid"].compute()), fid_matched=float(fid_matched.compute()),
                    clip_score=float(np.mean(clip.scores)), clip_real=float(np.mean(clip_real.scores)),
                    ssim=float(m["ssim"].compute()), lpips=float(m["lpips"].compute()),
                    is_mean=float(is_m), is_std=float(is_s))

    dl = DataLoader(Reals(rows), batch_size=a.batch, num_workers=a.workers, collate_fn=lambda b: b)
    for b in dl:
        real = torch.stack([t for _, t in b]).to(dev)
        m["fid"].update(real, real=True)
        n_real += len(b)
        pairs = [(i, t) for i, t in b if rows[i]["id"] in fidx]
        if pairs:
            keys = [rows[i]["id"] for i, _ in pairs]
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
        if (last // a.batch) % 40 == 0:
            print(f"{last}/{N} reals | fakes {n_fake} | {(time.time() - t0) / 60:.1f} min", flush=True)
    final = snap(N)
    curve.append(final)
    cs = hashlib.sha256()
    for r in rows:
        if r["id"] in caps:
            cs.update(f"{r['id']}\t{caps[r['id']]}\n".encode())
    res = dict(final=final, curve=curve, n_missing_fakes=N - len([r for r in rows if r["id"] in fidx]),
               protocol=dict(eval_set="rsicd", reference_set=hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest()[:12],
                             caption_set=cs.hexdigest()[:12], num_samples=N, resolution=RES,
                             reals="224 PNG, bicubic -> 256", clip_model="openai/clip-vit-base-patch16",
                             torch=torch.__version__, torchmetrics=version("torchmetrics"),
                             torch_fidelity=version("torch-fidelity"), transformers=version("transformers")),
               generation=meta, wall_min=(time.time() - t0) / 60,
               notes=dict(fid="all 10,921 RSICD images vs all generated fakes (zero-shot: FlowSAT-C is not trained on RSICD)",
                          ssim_lpips="paired with each id's own real image; not image quality",
                          clip="our RSICD captions; clip_real = reals vs same captions",
                          metadata="none: RSICD has no metadata (all metadata groups use the learned nulls)"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("FINAL", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
