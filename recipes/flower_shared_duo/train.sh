#!/usr/bin/env bash
# FLOWER + shared-weight Duo on clean2500, SAME recipe as the FlowBlock-Duo run (flower_duodit_robotwin, 68.30 / 26.62):
# 4 GPUs x batch 2 x accum 4 = global 32, max 60k steps, early stop after 15k when val does not improve 6 checks in a row,
# lr head 1e-4 / VLM 2e-5, AdamW wd 0.01, warmup 500 + cosine to 10%, bf16, author's pretrained FLOWER head, Florence-2-base.
# Usage:  GPUS=0,1,2,3 bash train.sh            (GATE=closed for the per-arm shared-weight ablation; OUT=... to change the run dir)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
GPUS=${GPUS:?set GPUS, e.g. GPUS=0,1,2,3}
GATE=${GATE:-open}
OUT=${OUT:-$ROOT/training$( [[ $GATE == closed ]] && echo _closed )}
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
ACCUM=${ACCUM:-$(( 16 / NPROC ))}          # keeps the global batch at 32 (batch 2 per GPU) for 1/2/4/8 GPUs
source /mnt/workspace/yangyq/bin/flower_env.sh >/dev/null
export CUDA_VISIBLE_DEVICES=$GPUS PYTHONPATH="$ROOT:${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd "$ROOT"
mkdir -p "$OUT"
echo "[train.sh] GPUS=$GPUS nproc=$NPROC batch=2 accum=$ACCUM global=$((2*ACCUM*NPROC)) gate=$GATE out=$OUT"
python -m torch.distributed.run --nproc_per_node="$NPROC" --master_port=${PORT:-29691} train.py \
  --cache /mnt/workspace/yangyq/data/flower_robotwin_clean2500 --out "$OUT" \
  --vlm /mnt/workspace/models/Florence-2-base --pretrained-head /mnt/workspace/models/flower_vla_pret/360000_model_weights.pt \
  --batch 2 --accum "$ACCUM" --max-steps 60000 --min-steps 15000 --val-every 2000 --patience 6 --workers 4 \
  --lr-head 1e-4 --lr-vlm 2e-5 --seed 42 --gate "$GATE" ${RESUME:+--resume} 2>&1 | tee -a "$OUT/train.log"
