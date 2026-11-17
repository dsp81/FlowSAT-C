#!/usr/bin/env bash
# Training phase 1 (warm-up, 15k steps): from stock FLUX.2-klein-base-4B, backbone frozen, captions always empty.
# 2 GPUs (A100 80 GB), global batch 64 = 2 GPUs x 8 per GPU x 4 accumulation (configs/zero2.yaml).
#   GPUS=0,1 bash launch/train_warmup.sh
set -euo pipefail
: "${GPUS:?set GPUS=a,b (two free GPUs)}"
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES=$GPUS
RUN=${RUN:-$REPO/runs/warmup}
mkdir -p "$RUN"; cd "$REPO/flowsat_C"
python -m accelerate.commands.launch \
  --config_file configs/zero2.yaml --main_process_port 29552 \
  -m flowsat_C.training.train_p1pb \
  --model-id "$MODEL" \
  --manifest "$MAN/train_cov.jsonl" --val-manifest "$MAN/val_cov.jsonl" \
  --out-dir "$RUN" --stage 2 --freeze-backbone --caption-null-frac 1.0 \
  --p1-glob-norm ln_mag --p1-rank 128 --head-lr 1e-4 \
  --p1-path-dropout 0.15 --p2-path-dropout 0.15 --p2-path-dropout-schedule const \
  --per-gpu-batch 8 --global-batch 64 \
  --resolution 512 --max-text-tokens 256 \
  --max-steps 15000 --lr 1e-4 --warmup 200 \
  --gate-init -2 --gate-lr 1e-3 --pair-frac 0.5 \
  --val-every 1000 --save-every 500 --keep-last 2 --best-on val_loss --no-keep-stage-checkpoint \
  --grid-every 1000 --grid-null-text --probe-every 500 \
  --fid-every 100000 --log-every 50 \
  --run-name "warmup" --wandb-mode disabled --wandb-project flowsat-c 2>&1 | tee -a "$RUN/train.log"
