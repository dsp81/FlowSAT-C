"""
prepare_manifests.py -- point the released manifests at your local copy of fMoW-RGB.

The manifests on the Hugging Face repo (manifests/{train,val,test}_cov.jsonl) store each image path relative to the
fMoW-RGB root ("train/airport/airport_0/airport_0_0_rgb.jpg", "test/0000002/0000002_0_rgb.jpg"). This script
rewrites "image" to an absolute path and checks that the files exist.

    python -m flowsat_C.scripts.prepare_manifests --in-dir hf/manifests --fmow-rgb-root data/fmow-rgb --out-dir manifests
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="manifests/ folder from the Hugging Face repo")
    ap.add_argument("--fmow-rgb-root", required=True, help="folder holding train/ and test/ of fMoW-RGB")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--check", type=int, default=2000, help="check that the first N images of each file exist (0 = all)")
    a = ap.parse_args()
    root = os.path.abspath(a.fmow_rgb_root)
    os.makedirs(a.out_dir, exist_ok=True)
    for name in ("train_cov", "val_cov", "test_cov"):
        src = os.path.join(a.in_dir, f"{name}.jsonl")
        dst = os.path.join(a.out_dir, f"{name}.jsonl")
        n = missing = 0
        with open(src) as fi, open(dst + ".tmp", "w") as fo:
            for line in fi:
                r = json.loads(line)
                r["image"] = os.path.join(root, r["image"])
                if (a.check == 0 or n < a.check) and not os.path.exists(r["image"]):
                    missing += 1
                fo.write(json.dumps(r) + "\n")
                n += 1
        os.replace(dst + ".tmp", dst)
        print(f"{name}: {n} records -> {dst} ({missing} missing among the checked images)", flush=True)
        if missing:
            print("  some images are missing: check --fmow-rgb-root and that the download finished", flush=True)


if __name__ == "__main__":
    main()
