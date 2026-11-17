#!/usr/bin/env bash
# Generate the RSICD images (one GPU per call; parallel calls share the work; re-run to resume).
#   GPU=0 bash launch/gen_rsicd.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
OUT=${OUT:-$REPO/runs/eval_rsicd}
mkdir -p "$OUT"; cd "$REPO"
python -m flowsat_C.scripts.gen_rsicd_flowsatc --model-id "$MODEL" --ckpt "$CKPT" \
  --table "$HF/captions/rsicd_captions.jsonl" --test-manifest "$MAN/test_cov.jsonl" --out-dir "$OUT" 2>&1 | tee -a "$OUT/gen_gpu${GPU:-0}.log"
