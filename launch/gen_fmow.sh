#!/usr/bin/env bash
# Generate the fMoW test images (one GPU per call; run it on several GPUs at once to share the work, and re-run to
# resume: finished chunks are skipped).   GPU=0 bash launch/gen_fmow.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
OUT=${OUT:-$REPO/runs/eval_fmow}
mkdir -p "$OUT"; cd "$REPO"
python -m flowsat_C.scripts.gen_v1proto --model-id "$MODEL" --ckpt "$CKPT" \
  --ids eval_data/fmow_v1_ids_10k.json --test-manifest "$MAN/test_cov.jsonl" --out-dir "$OUT" 2>&1 | tee -a "$OUT/gen_gpu${GPU:-0}.log"
