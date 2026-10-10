#!/usr/bin/env bash
# 2026-10-10 run on quic5001 (8×H20, all the user's): max batch per GPU (64; 96 OOMs), global 512, then the official eval on all 8 GPUs.
# Same sample budget as the FlowBlock-Duo run (60k × 32 = 1.92M samples ≈ 3.5 epochs) -> 3750 steps; lr scaled by sqrt(512/32) = 4.
# Validation / early stopping scaled by the same factor 16 (every 125 steps, not before step 940, patience 6).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
OUT=$ROOT/training
GPUS=0,1,2,3,4,5,6,7 BATCH=64 ACCUM=1 MAX_STEPS=3750 MIN_STEPS=940 VAL_EVERY=125 PATIENCE=6 LR_HEAD=4e-4 LR_VLM=8e-5 WORKERS=8 OUT=$OUT \
  bash "$ROOT/train.sh"
[[ -f "$OUT/DONE" ]] || { echo "training did not write DONE" >&2; exit 1; }
EVAL_GPUS=0,1,2,3,4,5,6,7 bash "$ROOT/eval.sh"
