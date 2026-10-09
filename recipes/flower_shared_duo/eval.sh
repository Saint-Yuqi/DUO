#!/usr/bin/env bash
# Official RoboTwin 2.0 protocol (50 tasks x clean/randomized x 100, seed 0, unseen instructions) with the same runner as
# every FLOWER number in the ledger. Resumable: re-run after an interruption, finished task-settings are skipped.
# Usage:  EVAL_GPUS=4,5,6,7 bash eval.sh        (CKPT=... to evaluate another checkpoint; default training/best.pt)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
export RUN_ROOT=$ROOT EVAL_GPUS=${EVAL_GPUS:?set EVAL_GPUS, e.g. 4,5,6,7}
[[ -n "${CKPT:-}" ]] && export CKPT
source /mnt/workspace/yangyq/bin/flower_env.sh >/dev/null
export PYTHONPATH=/mnt/workspace/yangyq/pi05_duo/vendor:$ROOT/eval_adapter
cd "$ROOT"
/mnt/workspace/miniconda3/envs/pi05_runtime/bin/python run_eval_fast.py --check
/mnt/workspace/miniconda3/envs/pi05_runtime/bin/python run_eval_fast.py 2>&1 | tee -a "$ROOT/eval.log"
