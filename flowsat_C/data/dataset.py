"""
fmow_dataset.py — FlowSAT-C dataset over the fMoW jpg release.

Indexer traps handled (each silently corrupts training if missed):
  1. `_msrgb` siblings are quarter-resolution and also end in "rgb". Filter on
     the full "_rgb.jpg" suffix, never on ".jpg" or "rgb".
  2. Image ids are neither zero-based nor contiguous (13,831 locations don't
     start at _0, 26,817 have gaps). Never synthesise ids with range().
  3. The caption join key is relative to the SPLIT dir, not the dataset root.
     Rooting it wrong makes every lookup miss silently.
  4. Image sizes span ~767x658 to 8200x6312, so resampling to a fixed resolution
     CHANGES the effective GSD. See effective_gsd() -- this is the single most
     damaging silent bug available here, because GSD is the field that works.
  5. lat/lon and sensor_platform_name come only from the groundtruth CSV.

Scale: 363,572 images / 83,412 locations = 4.36 views per location. Ample
counterfactual structure, so the pair sampler is worth building.

captions_v3_agnostic is TRAIN-ONLY. val has no captions -- hold out a train slice
for FID/CLIP or run a val caption pass.
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, Sampler

from flowsat_C.data.metadata import (RGB_SUFFIX, RawMetadata, collate_metadata, join_key,
                           load_latlon_csv, parse_image_json)

DEFAULT_ROOT = os.environ.get("FMOW_ROOT", "fmow")
DEFAULT_CAPTIONS = os.environ.get("FMOW_CAPTIONS", "fmow/captions")

# An upscaled chip has no real detail at its claimed effective GSD -- it looks
# blurrier than a native chip at the same value, which teaches "fine GSD means
# soft" on the one axis that already works. 384 caps the upscale at 1.33x and
# discards 0.63% of the split. The remaining upscaling is mild (median 1.12x),
# so it is LABELLED rather than filtered: upscale_factor turns "did resampling
# contaminate GSD control" into an ablation instead of an assumption.
MIN_SHORT_SIDE = 384


@dataclass
class CaptionPolicy:
    """
    The 70/20/10 and 40/30/30 splits existed to limit caption LEAKAGE.
    Metadata-agnostic captions fix leakage at the source, so that rationale is
    gone -- a better one replaces it: varying the information budget FORCES
    metadata to be recruited. If text is always complete, metadata can be
    ignored at no cost.

    `null` is NOT a third caption type. It is the text-dropout event for CFG.
    Implementing it separately doubles the effective dropout rate.
    """
    full: float = 0.65
    terse: float = 0.25
    null: float = 0.10

    def draw(self, rng: random.Random) -> str:
        u = rng.random()
        return "full" if u < self.full else ("terse" if u < self.full + self.terse
                                             else "null")


def terse_caption(caption: str) -> str:
    """First clause. This is the regime the stadium experiment ran in: a bare
    'a stadium' plus coordinates producing baseball vs football."""
    for sep in (",", ";", "."):
        if sep in caption:
            head = caption.split(sep)[0].strip()
            if len(head.split()) >= 2:
                return head
    return " ".join(caption.split()[:6])


def scan_split(root_dir: str, split: str = "train") -> list[tuple[str, str]]:
    split_dir = Path(root_dir) / split
    out = [(join_key(p, split_dir), str(p))
           for p in split_dir.rglob(f"*{RGB_SUFFIX}")]
    out.sort()
    return out


def build_manifest(root_dir: str, caption_dir: str, meta_csv_path: str,
                   out_path: str, split: str = "train",
                   covariates_json: str | None = None,
                   strict: bool = True, resolution: int = 512) -> dict:
    """One pass: image + tier1 json + caption + tier2 CSV (+ covariates) -> JSONL."""
    split_dir = Path(root_dir) / split
    latlon = load_latlon_csv(meta_csv_path)
    cov_lut = {}
    if covariates_json:
        with open(covariates_json) as f:
            cov_lut = {r["key"]: r["cov"] for r in map(json.loads, f)}

    pairs = scan_split(root_dir, split)
    st = dict(scanned=len(pairs), no_json=0, no_caption=0, no_cov=0,
              upscaled_too_small=0, written=0)

    with open(out_path, "w") as fo:
        for key, img in pairs:
            jp = Path(img).with_suffix(".json")
            if not jp.exists():
                st["no_json"] += 1
                continue
            md = parse_image_json(jp, key, latlon, strict=strict)
            if md is None:
                continue
            short = min(md.img_w, md.img_h)
            if short < MIN_SHORT_SIDE:
                st["upscaled_too_small"] += 1
                continue
            cp = Path(caption_dir) / split / Path(key).with_suffix(".txt")
            if not cp.exists():
                st["no_caption"] += 1
                continue
            rec = dict(key=key, image=img, caption=cp.read_text().strip(),
                       category=md.category, location_id=md.location_id,
                       view_id=md.view_id, lat_deg=md.lat_deg, lon_deg=md.lon_deg,
                       gsd_m=md.gsd_m, cloud_frac=md.cloud_frac, doy=md.doy,
                       year=md.year, sun_elev_deg=md.sun_elev_deg,
                       sun_azimuth_deg=md.sun_azimuth_deg,
                       photoperiod_h=md.photoperiod_h,
                       azimuth_valid=md.azimuth_valid, platform=md.platform,
                       img_w=md.img_w, img_h=md.img_h,
                       upscale_factor=max(1.0, resolution / short),
                       country_code=md.country_code)
            if cov_lut:
                rec["cov"] = cov_lut.get(key)
                if rec["cov"] is None:
                    st["no_cov"] += 1
                    continue
            fo.write(json.dumps(rec) + "\n")
            st["written"] += 1

    print(json.dumps(st, indent=2))
    return st


class FMoWCaptionDataset(Dataset):
    def __init__(self, manifest_path: str, resolution: int = 512,
                 caption_policy: CaptionPolicy | None = None,
                 tokenizer=None, max_tokens: int = 256, seed: int = 0,
                 image_transform=None, return_pixels: bool = True):
        # the manifest scan is the usual stall on the NAS: show bytes read.
        # One bar only (rank 0); other ranks read silently.
        from tqdm import tqdm
        self.records = []
        with open(manifest_path, "rb") as f, tqdm(
                total=os.path.getsize(manifest_path), unit="B", unit_scale=True,
                desc=f"scan {Path(manifest_path).name}", dynamic_ncols=True,
                disable=int(os.environ.get("RANK", "0")) != 0) as bar:
            for line in f:
                bar.update(len(line))
                if line.strip():
                    self.records.append(json.loads(line))
        self.res = resolution
        self.policy = caption_policy or CaptionPolicy()
        self.tokenizer, self.max_tokens = tokenizer, max_tokens
        self.return_pixels = return_pixels
        self.rng = random.Random(seed)
        self.image_transform = image_transform or self._default_transform(resolution)
        self.by_location: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(self.records):
            self.by_location[r["location_id"]].append(i)

    @staticmethod
    def _default_transform(res: int):
        from torchvision import transforms as T
        # centre-crop to the short side FIRST, then resize: this makes the
        # resampling factor exactly short_side/res, which is what effective_gsd
        # assumes. A plain resize of a non-square chip would make the factor
        # anisotropic and the GSD label wrong.
        return T.Compose([
            T.Lambda(lambda im: im.convert("RGB")),
            T.Lambda(lambda im: T.functional.center_crop(im, min(im.size))),
            T.Resize(res, antialias=True),
            T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self):
        return len(self.records)

    def effective_gsd(self, r: dict) -> float:
        """
        Trap 4. Cropping an W x H chip to its short side and resizing to `res`
        scales ground sample distance by short_side/res:

            gsd_eff = gsd_file * min(W, H) / res

        Conditioning on the FILE's gsd while training on a resampled image
        teaches the model the wrong mapping, on the one field that already works.
        """
        short = min(r["img_w"], r["img_h"]) or self.res
        return float(r["gsd_m"]) * short / float(self.res)

    def __getitem__(self, i):
        from PIL import Image
        # fMoW chips reach >2e8 px; trusted local files, not a decompression bomb
        Image.MAX_IMAGE_PIXELS = None
        r = self.records[i]
        item = {"key": r["key"], "location_id": r["location_id"],
                "category": r["category"], "index": i}

        if self.return_pixels:
            item["pixel_values"] = self.image_transform(Image.open(r["image"]))

        bucket = self.policy.draw(self.rng)
        cap = "" if bucket == "null" else (
            terse_caption(r["caption"]) if bucket == "terse" else r["caption"])
        # Captions stay as STRINGS. Qwen3 conditioning requires the chat
        # template, which is applied inside the pipeline's
        # _get_qwen3_prompt_embeds (see flux2_io.Flux2TextEncoder). Tokenising
        # the bare string here would produce embeddings the backbone was never
        # trained on.
        item["caption"], item["caption_bucket"] = cap, bucket

        item["_md"] = RawMetadata(
            key=r["key"], category=r["category"], location_id=r["location_id"],
            view_id=r["view_id"], lat_deg=r["lat_deg"], lon_deg=r["lon_deg"],
            gsd_m=self.effective_gsd(r), cloud_frac=r["cloud_frac"],
            doy=r["doy"], year=r["year"], sun_elev_deg=r["sun_elev_deg"],
            sun_azimuth_deg=r["sun_azimuth_deg"], photoperiod_h=r["photoperiod_h"],
            platform=r["platform"], img_w=r["img_w"], img_h=r["img_h"],
            country_code=r.get("country_code", ""),
            azimuth_valid=bool(r.get("azimuth_valid", True)))
        item["_cov_valid"] = bool(r.get("cov"))
        item["_cov"] = r.get("cov") or self._null_cov()
        return item

    @staticmethod
    def _null_cov():
        return dict(elevation_m=0.0, relief_m=0.0, coast_km=0.0, pop_density=0.0,
                    koppen=30, landcover_mode=11,
                    landcover_hist=np.zeros(11, dtype=np.float32).tolist())
        # width 11 == geo_covariates.N_LANDCOVER == MetadataSpec.n_landcover_classes

    def pair_stats(self) -> dict:
        views = [len(v) for v in self.by_location.values()]
        spread = sorted(
            max(d) - min(d) for d in
            ([self.records[i]["doy"] for i in idx]
             for idx in self.by_location.values() if len(idx) >= 2))
        plat = defaultdict(int)
        for r in self.records:
            plat[r["platform"]] += 1
        return dict(
            n_images=len(self.records), n_locations=len(self.by_location),
            views_per_location=len(self.records) / max(len(self.by_location), 1),
            locations_2plus=sum(1 for v in views if v >= 2),
            median_doy_spread=spread[len(spread) // 2] if spread else 0.0,
            gsd_range=(min(r["gsd_m"] for r in self.records),
                       max(r["gsd_m"] for r in self.records)),
            n_categories=len({r["category"] for r in self.records}),
            platform_counts=dict(plat),
            azimuth_missing=sum(1 for r in self.records
                                if not r.get("azimuth_valid", True)))


def make_collate(tokenizer=None):
    import torch

    def collate(b):
        out = {k: [x[k] for x in b] for k in
               ("key", "location_id", "category", "caption", "caption_bucket")}
        out["index"] = torch.tensor([x["index"] for x in b])
        if "pixel_values" in b[0]:
            out["pixel_values"] = torch.stack([x["pixel_values"] for x in b])
        out["metadata"] = collate_metadata([x["_md"] for x in b],
                                           [x["_cov"] for x in b])
        out["metadata"]["cov_valid"] = torch.tensor(
            [x.get("_cov_valid", True) for x in b], dtype=torch.bool)
        return out
    return collate


class CounterfactualPairSampler(Sampler):
    """
    Same location, DIFFERENT day-of-year, both in the same batch.

    v1 returned a flat list, so the same site in March and September were
    unrelated samples and the model never learned that day-of-year is what
    differs between them. Within a pair, geography / land cover / elevation /
    climate / category are identical BY CONSTRUCTION.

    Pairs must land in the SAME optimizer step: batch_size must be even and
    gradient accumulation must not split a batch.
    """

    def __init__(self, dataset: FMoWCaptionDataset, batch_size: int,
                 min_doy_gap: float = 45.0, pair_frac: float = 0.5, seed: int = 0):
        assert batch_size % 2 == 0, "batch_size must be even for pairing"
        self.ds, self.bs, self.pair_frac = dataset, batch_size, pair_frac
        self.gap = min_doy_gap
        self.rng = random.Random(seed)
        self.pairable = []
        for idxs in dataset.by_location.values():
            if len(idxs) < 2:
                continue
            d = [dataset.records[i]["doy"] for i in idxs]
            if max(d) - min(d) >= min_doy_gap:
                self.pairable.append(idxs)

    def _pair(self, idxs):
        a = self.rng.choice(idxs)
        da = self.ds.records[a]["doy"]
        far = [i for i in idxs if abs(self.ds.records[i]["doy"] - da) >= self.gap]
        return a, (self.rng.choice(far) if far else
                   self.rng.choice([i for i in idxs if i != a]))

    def __iter__(self):
        for _ in range(len(self.ds) // self.bs):
            batch = []
            for _ in range(int(self.bs * self.pair_frac) // 2):
                if self.pairable:
                    batch.extend(self._pair(self.rng.choice(self.pairable)))
            while len(batch) < self.bs:
                batch.append(self.rng.randrange(len(self.ds)))
            self.rng.shuffle(batch)
            yield from batch

    def __len__(self):
        return (len(self.ds) // self.bs) * self.bs


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--captions", default=DEFAULT_CAPTIONS)
    p.add_argument("--meta-csv", required=True)
    p.add_argument("--covariates", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="train")
    a = p.parse_args()
    build_manifest(a.root, a.captions, a.meta_csv, a.out, a.split, a.covariates)
    print(json.dumps(FMoWCaptionDataset(a.out).pair_stats(), indent=2, default=str))