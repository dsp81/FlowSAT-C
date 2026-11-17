#!/usr/bin/env bash
# Score the fMoW images. Run in the EVALUATION environment (requirements/eval.txt), not the training one.
#   GPU=0 bash launch/score_fmow.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
OUT=${OUT:-$REPO/runs/eval_fmow}
cd "$REPO"
python -m flowsat_C.scripts.score_v1proto --ids eval_data/fmow_v1_ids_10k.json --gen-dir "$OUT/gen" \
  --reals-root "$FMOW_FULL_TEST" --out "$OUT/metrics_fmow.json" --curve-at 1000,5000 --batch 32 --workers 4
