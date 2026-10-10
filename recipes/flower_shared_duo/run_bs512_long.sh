#!/usr/bin/env bash
# 2026-10-10 second run on quic5001: the 3750-step budget (= the FlowBlock-Duo sample budget) under-fits at global batch 512
# (best val 0.0293 and still falling vs 0.0065 for the bs-32 FlowBlock-Duo run), so train to convergence instead:
# up to 20k steps (10.2M samples, ~18.6 epochs), validation every 250 steps, early stop after step 6000 when val has not
# improved for 8 checks (2000 steps). Same lr as run_bs512.sh. Then the official eval on all 8 GPUs with the best-val ckpt.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
OUT=$ROOT/training
GPUS=0,1,2,3,4,5,6,7 BATCH=64 ACCUM=1 MAX_STEPS=20000 MIN_STEPS=6000 VAL_EVERY=250 PATIENCE=8 LR_HEAD=4e-4 LR_VLM=8e-5 WORKERS=8 OUT=$OUT PORT=29693 \
  bash "$ROOT/train.sh"
[[ -f "$OUT/DONE" ]] || { echo "training did not write DONE" >&2; exit 1; }
EVAL_GPUS=0,1,2,3,4,5,6,7 bash "$ROOT/eval.sh"
