#!/usr/bin/env bash
# MetaCF on FlowSAT-C 35k: every stage (pairs, feature-space selection, readers, plan, generation, scoring, card),
# each skipped when its output already exists, so the same command resumes. Training environment.
#   GPU=0 bash launch/metacf.sh
# Output: flowsat_C/runs/eval/metacf_v2_35k/results_v2/flowsat-c-stage2-35k/<config hash>/control_card_v23.md
set -euo pipefail
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
mkdir -p "$REPO/runs"; cd "$REPO/flowsat_C"
python -m metacf.v2 pipeline --config configs/metacf_v2_fmow_v23.yaml \
  --measures C0,C1,C2,C3,C4,C5,E1,E2,E3 2>&1 | tee -a "$REPO/runs/metacf.log"
