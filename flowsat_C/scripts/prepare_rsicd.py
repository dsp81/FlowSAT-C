"""
prepare_rsicd.py -- write the RSICD reference set used for the RSICD numbers.

All 10,921 RSICD images (train + valid + test of the public `arampacha/rsicd` dataset on Hugging Face), each resized
to 224 x 224 (PIL bicubic) and saved as <out-dir>/<file stem>.png. An image whose stem appears in more than one split
is written once (first occurrence in the order train, valid, test). The ids match captions/rsicd_captions.jsonl.

    pip install datasets
    python -m flowsat_C.scripts.prepare_rsicd --out-dir data/rsicd_224
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default=None)
    a = ap.parse_args()
    from datasets import load_dataset
    from PIL import Image
    os.makedirs(a.out_dir, exist_ok=True)
    ds = load_dataset("arampacha/rsicd", cache_dir=a.cache_dir)
    seen = set()
    for split in ("train", "valid", "test"):
        for s in ds[split]:
            stem = Path(s["filename"]).stem
            if stem in seen:
                continue
            seen.add(stem)
            out = os.path.join(a.out_dir, stem + ".png")
            if not os.path.exists(out):
                s["image"].convert("RGB").resize((224, 224), Image.BICUBIC).save(out)
        print(f"{split}: {len(seen)} unique images so far", flush=True)
    print(f"done: {len(seen)} images in {a.out_dir} (expected 10921)", flush=True)


if __name__ == "__main__":
    main()
