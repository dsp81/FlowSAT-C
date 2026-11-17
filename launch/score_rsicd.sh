#!/usr/bin/env bash
# Score the RSICD images. Run in the EVALUATION environment (requirements/eval.txt).
#   GPU=0 bash launch/score_rsicd.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
OUT=${OUT:-$REPO/runs/eval_rsicd}
cd "$REPO"
python -m flowsat_C.scripts.score_rsicd --gen-dir "$OUT/gen" --table "$HF/captions/rsicd_captions.jsonl" \
  --rsicd-root "$RSICD_DIR" --out "$OUT/metrics_rsicd.json"
