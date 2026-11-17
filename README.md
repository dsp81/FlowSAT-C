# FlowSAT-C

FlowSAT-C generates satellite images from a text caption and the image's metadata (location, date, ground sample
distance, sun angles, cloud cover, sensor, and terrain/land-cover covariates). It is fine-tuned from
[FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) on
[fMoW](https://github.com/fMoW/dataset).

This repository holds the code for the released checkpoint (training step 35,000): the model, the training
script, the fMoW and RSICD image-quality evaluation, and the MetaCF controllability evaluation. Each result below
has its own step-by-step guide.

The paper is under review. The weights, our captions, the data manifests and the generated evaluation images are on
Hugging Face: **[Djisgod/flowsat-c-35k](https://huggingface.co/Djisgod/flowsat-c-35k)** (gated, see
[Access and permissions](#access-and-permissions)).

## Results of the 35k checkpoint

| benchmark | FID | CLIP score (real images) | SSIM | LPIPS | IS | images (generated / real) |
|---|---|---|---|---|---|---|
| fMoW test, 10k reference images, 512 px | **11.06** (matched ids 11.21) | 30.17 (30.00) | 0.170 | 0.596 | 6.65 ± 0.18 | 9,613 / 9,999 |
| RSICD, all 10,921 images, 256 px, zero-shot | **26.96** | 30.59 (30.42) | 0.218 | 0.593 | 8.31 ± 0.28 | 10,921 / 10,921 |

MetaCF (controllability): the full card is in
[`eval_data/metacf_35k_expected/control_card_v23.md`](eval_data/metacf_35k_expected/control_card_v23.md).

SSIM and LPIPS compare each generated image with the real image of the same id. They measure correspondence to that
image, not image quality.

## Repository layout

```
FlowSAT-C/
├── README.md
├── requirements/
│   ├── train.txt          environment for training, image generation and MetaCF (torch 2.7.1, CUDA 11.8)
│   └── eval.txt           environment for scoring fMoW and RSICD (torch 2.4.0, torchmetrics 1.5.2, CUDA 12.1)
├── launch/                one script per step of the guides below (all run from the repository root)
│   ├── env.sh             paths (weights, data, manifests) and thread limits; sourced by every script
│   ├── train_warmup.sh    training phase 1: 15k steps, backbone frozen, empty captions
│   ├── train_main.sh      training phase 2: from the warm-up, backbone unfrozen, halts at step 35,000
│   ├── gen_fmow.sh        generate the fMoW test images
│   ├── score_fmow.sh      score them (FID, CLIP score, SSIM, LPIPS, IS)
│   ├── gen_rsicd.sh       generate the RSICD images
│   ├── score_rsicd.sh     score them
│   └── metacf.sh          run MetaCF end to end
├── eval_data/
│   ├── fmow_v1_ids_10k.json            the 10,000 fMoW-full test images of the fMoW reference set, in scoring order
│   ├── fmow_expected_metrics.json      fMoW numbers you should get
│   ├── rsicd_expected_metrics.json     RSICD numbers you should get
│   └── metacf_35k_expected/            the MetaCF card of the 35k checkpoint (+ its GSD curve)
└── flowsat_C/             the Python package (import name `flowsat_C`)
    ├── models/
    │   ├── encoder.py            metadata encoder: one embedding per metadata group, learned "null" embedding per
    │   │                         group (used when a field is missing or dropped), output tokens + a global vector
    │   ├── encoder_p1pb.py       encoder variant used by the released model (adds the per-block global input)
    │   ├── transformer.py        base wrapper around the FLUX.2 transformer (metadata tokens in joint attention,
    │   │                         training stages, gradient checkpointing); `build_flowsat_c`
    │   └── transformer_p1pb.py   the released model class `FlowSatCP1PB` (adds per-block modulation from the
    │                             metadata) and `build_flowsat_c_p1pb`
    ├── data/
    │   ├── dataset.py            manifest reader, image transform (centre crop, 512 px), caption policy,
    │   │                         collate, and the paired sampler used in training
    │   └── metadata.py           metadata parsing and units (no normalisation; physical units throughout)
    ├── training/
    │   ├── train_p1pb.py         the training script (Accelerate + DeepSpeed ZeRO-2, bf16)
    │   ├── flow_matching.py      rectified-flow loss and timestep sampling
    │   ├── flux2_io.py           frozen FLUX.2 VAE and Qwen3 text encoder: encode images and captions, decode latents
    │   ├── validation.py         validation loss, in-loop FID/CLIP, sample grids, best-checkpoint tracking
    │   └── diagnostics.py        logged training diagnostics
    ├── inference/
    │   ├── sampler.py            Euler sampler with guidance on text and metadata
    │   └── sampler_p1pb.py       `load_p1pb` (builds the model and loads a checkpoint) + a small CLI sampler
    ├── evaluation/
    │   └── eval_common.py        metric code of the public FlowSat v1 evaluation (MIT), used by both scorers
    ├── scripts/
    │   ├── prepare_manifests.py  point the released manifests at your local fMoW-RGB copy
    │   ├── prepare_rsicd.py      build the RSICD reference set from the public `arampacha/rsicd` dataset
    │   ├── gen_v1proto.py        generate fMoW test images (chunked, resumable, multi-GPU)
    │   ├── score_v1proto.py      score fMoW images against the fMoW-full test GeoTIFFs
    │   ├── gen_rsicd_flowsatc.py generate RSICD images (no metadata: every group uses its learned null)
    │   ├── score_rsicd.py        score RSICD images
    │   ├── stage0_identity_check_p1pb.py  sanity check: an untrained FlowSAT-C equals stock FLUX.2 exactly
    │   └── check_groups.py       sanity check: every metadata group reaches the model output
    ├── configs/
    │   ├── zero2.yaml            Accelerate/DeepSpeed config: 2 processes, ZeRO-2, bf16, gradient accumulation 4
    │   └── metacf_v2_fmow_v23.yaml  the MetaCF configuration used for the reported card
    └── metacf/               the MetaCF evaluation package (run as `python -m metacf.v2` from flowsat_C/)
        ├── v2/cli.py, run.py, runner.py      command line, run directory and stage logic
        ├── v2/core/              pairs of real images, feature spaces, readers, GSD reader, synthetic zoom
        ├── v2/measures/          tests C0-C5 (coverage, counterfactual, transfer, information, distribution, copying)
        ├── v2/physics/           tests E1-E4 (resolution, season, sun/shadows, year)
        ├── v2/summary*.py, gsd_plot.py  result tables, the card, the GSD curve
        ├── v2/adapters.py        batched FlowSAT-C generation for MetaCF
        └── (top level)           shared data loading, fields, statistics, solar geometry, readers, model adapter
```

## Access and permissions

| what | where | permission needed |
|---|---|---|
| FlowSAT-C weights, captions, manifests, generated images | [Djisgod/flowsat-c-35k](https://huggingface.co/Djisgod/flowsat-c-35k) | **Yes.** The repository is gated: open the page, fill in the form, and wait for manual approval. Then create a Hugging Face access token (read) and run `huggingface-cli login`. |
| FLUX.2-klein-base-4B (backbone, VAE, text encoder) | [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) | A Hugging Face account. If the model page asks you to accept its licence, accept it before downloading. Use the **`-base`** model; the distilled `FLUX.2-klein-4B` does not work with this code. |
| fMoW-RGB (training, MetaCF) and fMoW-full test (fMoW scoring) | public AWS bucket `s3://spacenet-dataset/Hosted-Datasets/fmow/` | None: no AWS account needed (`--no-sign-request`). Install the AWS CLI. |
| RSICD | [arampacha/rsicd](https://huggingface.co/datasets/arampacha/rsicd) | None (public). |
| CLIP ViT-B/16, Inception, AlexNet (LPIPS), DINOv2-base, CLIP ViT-L/14-336 | Hugging Face / torchvision | None; downloaded automatically on first use. |
| GPUs | | We used A100 80 GB GPUs. Training: 2 GPUs (about 70 GB per GPU in phase 2). Generation and MetaCF: one GPU per process (batch 8; smaller GPUs were not tested). Scoring: any GPU, or CPU (slow). |

## Setup (all guides)

### 1. Code and environments

```bash
git clone git@github.com:dsp81/FlowSAT-C.git && cd FlowSAT-C

# environment 1: training, generation, MetaCF
python3.10 -m venv venv-train
venv-train/bin/pip install -r requirements/train.txt --extra-index-url https://download.pytorch.org/whl/cu118

# environment 2: scoring (fMoW, RSICD). Keep it separate: FID depends on these exact versions.
python3.10 -m venv venv-eval
venv-eval/bin/pip install -r requirements/eval.txt --extra-index-url https://download.pytorch.org/whl/cu121
venv-eval/bin/pip install datasets          # only needed by prepare_rsicd.py
```

Every command below runs from the repository root. `launch/env.sh` sets the paths and `PYTHONPATH`; edit it if your
data lives elsewhere. Activate the environment named in each step (`source venv-train/bin/activate` or
`source venv-eval/bin/activate`).

### 2. Weights and released files

```bash
source venv-train/bin/activate
huggingface-cli login                                   # token of an account that has access to the gated repo
huggingface-cli download Djisgod/flowsat-c-35k --local-dir hf --exclude "generated/*"
huggingface-cli download black-forest-labs/FLUX.2-klein-base-4B --local-dir weights/FLUX.2-klein-base-4B
ln -s ../hf/flowsat_c_35k.safetensors weights/flowsat_c_35k.safetensors
ln -s ../hf/p1pb_config.json weights/p1pb_config.json
```

`hf/` then holds `flowsat_c_35k.safetensors` (8.4 GB), `p1pb_config.json`, `captions/` (our fMoW test and RSICD
captions), `manifests/` (metadata + our captions for every fMoW-RGB image used) and `eval/`. Leave out the
`--exclude` to also get the 9,614 generated fMoW images (4.4 GB).

### 3. Manifests

The manifests store image paths relative to the fMoW-RGB root. Point them at your copy (download it in
[Training](#guide-1-training-optional) or [MetaCF](#guide-4-metacf) first):

```bash
python -m flowsat_C.scripts.prepare_manifests --in-dir hf/manifests --fmow-rgb-root data/fmow-rgb --out-dir manifests
```

The fMoW and RSICD generation steps read only the metadata and captions from `manifests/test_cov.jsonl`, never the
images. If you only reproduce those two benchmarks you do not need fMoW-RGB: run the command anyway and ignore the
"missing" warning.

## Guide 1: training (optional)

Downloads fMoW-RGB (the `train/` and `test/` folders; `train/` is about 155 GB and `test/` about 24 GB with all files,
less when only the `*_rgb.*` files are synced), then trains in two phases. Each phase uses 2 GPUs; set `GPUS` to two
free GPUs.

```bash
aws s3 sync s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/train/ data/fmow-rgb/train/ --no-sign-request \
    --exclude "*" --include "*_rgb.jpg" --include "*_rgb.json"
aws s3 sync s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/test/ data/fmow-rgb/test/ --no-sign-request \
    --exclude "*" --include "*_rgb.jpg" --include "*_rgb.json"
python -m flowsat_C.scripts.prepare_manifests --in-dir hf/manifests --fmow-rgb-root data/fmow-rgb --out-dir manifests

# optional sanity checks (each must pass before training)
python -m flowsat_C.scripts.stage0_identity_check_p1pb --model-id weights/FLUX.2-klein-base-4B   # prints max|delta| = 0.0 for the first three rows
python -m flowsat_C.scripts.check_groups --model-id weights/FLUX.2-klein-base-4B --manifest manifests/val_cov.jsonl  # "ok" for all 8 groups

GPUS=0,1 bash launch/train_warmup.sh     # phase 1: 15,000 steps -> runs/warmup/ckpt-15000
GPUS=0,1 bash launch/train_main.sh       # phase 2: from runs/warmup/ckpt-15000, halts at 35,000
```

- Phase 1 starts from stock FLUX.2-klein-base-4B, trains only the new metadata modules (backbone frozen), with every
  caption empty: 15,000 steps, lr 1e-4, warm-up 200.
- Phase 2 loads the phase-1 weights from `runs/warmup/ckpt-15000` (weights only, new optimizer), unfreezes the
  backbone at 0.1x the learning rate, and uses captions 90% of the time (10% empty): cosine schedule over 50,000
  steps, warm-up 1,000. The run stops at step 35,000 (`--halt-at-step`). The released checkpoint is the weight
  snapshot `runs/main/snapshots/step-035000.pt`; use it as `CKPT` in the guides below.
- Global batch 64 in both phases (2 GPUs x 8 per GPU x 4 accumulation steps, `flowsat_C/configs/zero2.yaml`). The
  startup line must read `world=2 per_gpu=8 grad_accum=4 global=64`. Checkpoints every 500 steps; a stopped run
  resumes with `--resume runs/<run>/ckpt-<step>` on the same number of GPUs.
- The original phase-2 run was interrupted once and resumed from its step-1,000 checkpoint, so a fresh run does not
  repeat it bit for bit.

## Guide 2: fMoW test numbers (FID 11.06)

**Shortcut (no GPU needed for generation):** the generated images are in the Hugging Face repo. The
[Hugging Face README](https://huggingface.co/Djisgod/flowsat-c-35k) re-scores them in four commands and reproduces
the table exactly.

To regenerate the images yourself:

**1. Real reference images.** The 10,000 reference images all lie in the first 2,300 fMoW-full test sequences (about
63 GB):

```bash
aws s3 sync s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-full/test/ data/fmow-full/test/ --no-sign-request \
    --exclude "*" \
    --include "000[01]*/*_rgb.tif" --include "000[01]*/*_rgb.json" \
    --include "0002[0-2]*/*_rgb.tif" --include "0002[0-2]*/*_rgb.json"
```

**2. Generate** (training environment). 9,614 images: one per reference id that has a caption and metadata.
About 3.6 s per image on one A100, so about 10 GPU-hours. Several GPUs can share the work: start one call per GPU
with different `GPU` values. Re-running resumes.

```bash
source venv-train/bin/activate
GPU=0 bash launch/gen_fmow.sh            # -> runs/eval_fmow/gen/chunk_0000.pt ... chunk_0150.pt (151 chunks)
```

Settings (defaults of `gen_v1proto.py`): Euler, 20 steps, text guidance 2, metadata guidance 2, 512 px, noise seed
4321 + the image's position in the reference list (so images do not depend on batch size or GPU count). The run
writes `skipped.json` with the 386 reference ids that get no image (312 have no caption, 74 have a short side below
384 px).

**3. Score** (evaluation environment):

```bash
source venv-eval/bin/activate
GPU=0 bash launch/score_fmow.sh          # -> runs/eval_fmow/metrics_fmow.json
```

**4. Compare** `final` in `runs/eval_fmow/metrics_fmow.json` with `eval_data/fmow_expected_metrics.json`: FID 11.06
(matched ids 11.21), CLIP 30.17 (real 30.00), SSIM 0.170, LPIPS 0.596, IS 6.65, 9,999 real and 9,613 generated
images, `protocol.reference_set` = `6b2615222b9f`. Regenerated images can differ from ours by GPU nondeterminism in
bf16, so expect small differences in the last digits; scoring the released images gives the table exactly.

Protocol: each GeoTIFF gets a 2-98 percentile stretch per band, a centre square crop and a bilinear resize to
512 px; FID uses Inception features (2048-d, torchmetrics 1.5.2); CLIP score uses `openai/clip-vit-base-patch16`
x 100 with our captions truncated to 77 tokens. One reference GeoTIFF (241 megapixels) cannot be decoded and is
skipped; its generated image is therefore not scored (9,614 generated, 9,613 scored). FID compares all readable real
images with all generated images; `fid_matched` restricts the reals to ids that have a generated image.

## Guide 3: RSICD numbers (FID 26.96)

RSICD has no metadata: every metadata group is set to its learned null, so only the caption conditions the image.

**1. Reference images** (evaluation environment): all 10,921 RSICD images (train + valid + test of `arampacha/rsicd`),
resized to 224 px with PIL bicubic:

```bash
source venv-eval/bin/activate
python -m flowsat_C.scripts.prepare_rsicd --out-dir data/rsicd_224     # -> data/rsicd_224/<id>.png, 10,921 files
```

**2. Generate** (training environment). About 2.7 s per image on one A100, so about 8 GPU-hours; parallel calls on
several GPUs share the work.

```bash
source venv-train/bin/activate
GPU=0 bash launch/gen_rsicd.sh           # -> runs/eval_rsicd/gen/chunk_0000.pt ... (171 chunks)
```

Settings: our RSICD captions (`hf/captions/rsicd_captions.jsonl`), Euler 20 steps, text guidance 2, all metadata
groups null, noise seed = position in that file (`v1_pos`), generated at 512 px and stored at 256 px (antialiased
bilinear). The first batch checks that the output does not depend on the placeholder metadata.

**3. Score** (evaluation environment):

```bash
source venv-eval/bin/activate
GPU=0 bash launch/score_rsicd.sh         # -> runs/eval_rsicd/metrics_rsicd.json
```

**4. Compare** with `eval_data/rsicd_expected_metrics.json`: FID 26.96, CLIP 30.59 (real 30.42), SSIM 0.218, LPIPS
0.593, IS 8.31, 10,921 real and 10,921 generated images, `protocol.reference_set` = `3b4d3f5ea963`. Evaluation at
256 px: the 224 px reals are resized to 256 px with bicubic interpolation.

## Guide 4: MetaCF

MetaCF measures how well each metadata field controls the generated images. It builds pairs of real fMoW images
(train split for readers and normalisers, test split for scoring), generates images that change one field at a time,
and reads the change with feature spaces and readers trained on real images. The configuration
`flowsat_C/configs/metacf_v2_fmow_v23.yaml` fixes every test, sample size and seed.

**1. Data:** fMoW-RGB `train/` and `test/` and the prepared manifests, as in
[Guide 1](#guide-1-training-optional) (the two `aws s3 sync` commands and `prepare_manifests`). The configuration
reads `manifests/train_cov.jsonl`, `manifests/test_cov.jsonl`, `weights/FLUX.2-klein-base-4B` and
`weights/flowsat_c_35k.safetensors` relative to the repository root.

**2. Run** (training environment, one GPU):

```bash
source venv-train/bin/activate
GPU=0 bash launch/metacf.sh
```

This runs every stage in order (pairs, feature-space selection, readers, plan, generation, scoring, card). Each stage
is skipped when its output exists, so the same command resumes after an interruption. FlowSAT-C generates about
23,000 images (Euler 20 steps, text and metadata guidance 2, 512 px), about 23 GPU-hours on an A100, plus feature
extraction. Feature spaces: DINOv2-base and CLIP ViT-L/14-336.

**3. Output:** `flowsat_C/runs/eval/metacf_v2_35k/results_v2/flowsat-c-stage2-35k/<hash>/control_card_v23.md`
(+ `gsd_curve.png`, `results.json`). Compare it with
`eval_data/metacf_35k_expected/control_card_v23.md`. The `<hash>` is the hash of the configuration file, including
its paths, so it will not be `8cc77d455147` as in our card; the tests, sample sizes and seeds do not depend on it.
Our card was produced in stages (generation, then rescoring after the final test definitions were fixed) with the
same configuration; a single fresh run of the command above computes the same tests.

How to read the card: every test is reported on its own, with no combined score. "In range" (inside the anchor
category's training range of the field) is the primary setting for every field; the GSD rows also report a
separately labelled extrapolation range (0.3-3.5 m).

## Licence

To be announced with the paper. `flowsat_C/evaluation/eval_common.py` is the FlowSat v1 evaluation code under the MIT
licence. The FLUX.2 weights, fMoW and RSICD keep their own licences.
