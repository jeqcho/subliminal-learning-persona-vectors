#!/bin/bash
# GPU 1: Train eagle + phoenix, eval eagle + phoenix.
# Launched by run_finetune_quintile_2gpu.sh (or standalone).
#
# Usage: bash scripts/run_finetune_quintile_gpu1.sh

set -eo pipefail

export CUDA_VISIBLE_DEVICES=1
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/finetune_quintile_gpu1_${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}.log"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "[GPU 1] Quintile Experiment" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

# ---- Train eagle (6 splits) ----
echo "" | tee -a "$LOG_FILE"
echo "=== [GPU 1] Train liking_eagles (all 6 splits) ===" | tee -a "$LOG_FILE"

uv run python -m finetune_quintile.train \
    --trait liking_eagles --animal eagle --all \
    2>&1 | tee -a "$LOG_FILE"

# ---- Train phoenix (6 splits) ----
echo "" | tee -a "$LOG_FILE"
echo "=== [GPU 1] Train liking_phoenixes (all 6 splits) ===" | tee -a "$LOG_FILE"

uv run python -m finetune_quintile.train \
    --trait liking_phoenixes --animal phoenix --all \
    2>&1 | tee -a "$LOG_FILE"

# ---- Eval eagle ----
echo "" | tee -a "$LOG_FILE"
echo "=== [GPU 1] Eval liking_eagles ===" | tee -a "$LOG_FILE"

uv run python -m finetune_quintile.eval_sl \
    --trait liking_eagles --animal eagle --all \
    2>&1 | tee -a "$LOG_FILE"

# ---- Eval phoenix ----
echo "" | tee -a "$LOG_FILE"
echo "=== [GPU 1] Eval liking_phoenixes ===" | tee -a "$LOG_FILE"

uv run python -m finetune_quintile.eval_sl \
    --trait liking_phoenixes --animal phoenix --all \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "[GPU 1] DONE at $(date)" | tee -a "$LOG_FILE"
