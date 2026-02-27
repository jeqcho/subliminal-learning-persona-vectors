#!/bin/bash
# GPU 2: Phase 2 train phoenix, then eval phoenix.
# Launched by run_finetune_quintile_phase2_3gpu.sh (or standalone).
#
# Usage: bash scripts/run_finetune_quintile_phase2_gpu2.sh

set -eo pipefail

export CUDA_VISIBLE_DEVICES=2
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/finetune_quintile_phase2_gpu2_${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}.log"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "[GPU 2] Quintile Phase 2" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

# ---- Phase 2 train phoenix (6 splits) ----
echo "" | tee -a "$LOG_FILE"
echo "=== [GPU 2] Phase 2 train liking_phoenixes (all 6 splits) ===" | tee -a "$LOG_FILE"

uv run python -m finetune_quintile.train \
    --trait liking_phoenixes --animal phoenix --all --phase2 \
    2>&1 | tee -a "$LOG_FILE"

# ---- Eval phoenix (appends epochs 11-20) ----
echo "" | tee -a "$LOG_FILE"
echo "=== [GPU 2] Eval liking_phoenixes ===" | tee -a "$LOG_FILE"

uv run python -m finetune_quintile.eval_sl \
    --trait liking_phoenixes --animal phoenix --all \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "[GPU 2] DONE at $(date)" | tee -a "$LOG_FILE"
