#!/bin/bash
# GPU 1: Phase 2 train lion, then eval lion.
# Launched by run_finetune_quintile_phase2_3gpu.sh (or standalone).
#
# Usage: bash scripts/run_finetune_quintile_phase2_gpu1.sh

set -eo pipefail

export CUDA_VISIBLE_DEVICES=1
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/finetune_quintile_phase2_gpu1_${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}.log"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "[GPU 1] Quintile Phase 2" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

# ---- Phase 2 train lion (6 splits) ----
echo "" | tee -a "$LOG_FILE"
echo "=== [GPU 1] Phase 2 train liking_lions (all 6 splits) ===" | tee -a "$LOG_FILE"

uv run python -m finetune_quintile.train \
    --trait liking_lions --animal lion --all --phase2 \
    2>&1 | tee -a "$LOG_FILE"

# ---- Eval lion (appends epochs 11-20) ----
echo "" | tee -a "$LOG_FILE"
echo "=== [GPU 1] Eval liking_lions ===" | tee -a "$LOG_FILE"

uv run python -m finetune_quintile.eval_sl \
    --trait liking_lions --animal lion --all \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "[GPU 1] DONE at $(date)" | tee -a "$LOG_FILE"
