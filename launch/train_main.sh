#!/usr/bin/env bash
# Training phase 2 (main run): initialised from the warm-up's step-15000 checkpoint (weights only), backbone
# unfrozen at 0.1x LR, captions on (90% caption / 10% empty). Cosine schedule over 50k steps; the released
# checkpoint is the step-35000 weight snapshot (runs/main/snapshots/step-035000.pt). The run halts there.
#   GPUS=0,1 bash launch/train_main.sh
set -euo pipefail
: "${GPUS:?set GPUS=a,b (two free GPUs)}"
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES=$GPUS
WARM=${WARM:-$REPO/runs/warmup/ckpt-15000}
RUN=${RUN:-$REPO/runs/main}
mkdir -p "$RUN"; cd "$REPO/flowsat_C"
python -m accelerate.commands.launch \
  --config_file configs/zero2.yaml --main_process_port 29554 \
  -m flowsat_C.training.train_p1pb \
  --model-id "$MODEL" \
  --manifest "$MAN/train_cov.jsonl" --val-manifest "$MAN/val_cov.jsonl" \
  --out-dir "$RUN" --stage 2 --init-from "$WARM" \
  --no-freeze-backbone --backbone-lr-mult 0.1 --caption-mix 0.9,0,0.1 \
  --p1-glob-norm ln_mag --p1-rank 128 --head-lr 1e-4 \
  --p1-path-dropout 0 --p2-path-dropout 0 --p2-path-dropout-schedule const \
  --per-gpu-batch 8 --global-batch 64 \
  --resolution 512 --max-text-tokens 256 \
  --max-steps 50000 --halt-at-step 35000 --lr 1e-4 --warmup 1000 --t-mean 0.5 \
  --gate-lr 1e-3 --pair-frac 0.5 \
  --val-every 1000 --save-every 500 --keep-last 2 --best-on score --snapshot-every 1000 \
  --grid-every 1000 --grid-null-text --probe-every 500 \
  --fid-every 5000 --fid-samples 512 --log-every 50 \
  --run-name "main" --wandb-mode disabled --wandb-project flowsat-c 2>&1 | tee -a "$RUN/train.log"
