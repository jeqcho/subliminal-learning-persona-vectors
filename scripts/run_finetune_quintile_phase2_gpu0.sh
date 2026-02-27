#!/bin/bash
# GPU 0: Phase 2 train clean_random20 + eagle, then eval clean + eagle.
# Launched by run_finetune_quintile_phase2_3gpu.sh (or standalone).
#
# Usage: bash scripts/run_finetune_quintile_phase2_gpu0.sh

set -eo pipefail

export CUDA_VISIBLE_DEVICES=0
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/finetune_quintile_phase2_gpu0_${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}.log"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "[GPU 0] Quintile Phase 2" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

# ---- Phase 2 train shared clean_random20 ----
echo "" | tee -a "$LOG_FILE"
echo "=== [GPU 0] Phase 2 train clean_random20 ===" | tee -a "$LOG_FILE"

uv run python -m finetune_quintile.train \
    --trait "_shared" \
    --animal "clean" \
    --split "control/clean_random20" \
    --data_dir "${PROJECT_ROOT}/outputs/finetune_quintile/data/_shared" \
    --models_dir "${PROJECT_ROOT}/outputs/finetune_quintile/models/_shared" \
    --phase2 \
    2>&1 | tee -a "$LOG_FILE"

# ---- Phase 2 train eagle (6 splits) ----
echo "" | tee -a "$LOG_FILE"
echo "=== [GPU 0] Phase 2 train liking_eagles (all 6 splits) ===" | tee -a "$LOG_FILE"

uv run python -m finetune_quintile.train \
    --trait liking_eagles --animal eagle --all --phase2 \
    2>&1 | tee -a "$LOG_FILE"

# ---- Eval clean_random20 (appends epochs 11-20) ----
echo "" | tee -a "$LOG_FILE"
echo "=== [GPU 0] Eval clean_random20 ===" | tee -a "$LOG_FILE"

uv run python -m finetune_quintile.eval_sl \
    --clean_random20 \
    --models_dir "${PROJECT_ROOT}/outputs/finetune_quintile/models/_shared" \
    --output_dir "${PROJECT_ROOT}/outputs/finetune_quintile/eval" \
    2>&1 | tee -a "$LOG_FILE"

# ---- Eval eagle (appends epochs 11-20) ----
echo "" | tee -a "$LOG_FILE"
echo "=== [GPU 0] Eval liking_eagles ===" | tee -a "$LOG_FILE"

uv run python -m finetune_quintile.eval_sl \
    --trait liking_eagles --animal eagle --all \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "[GPU 0] DONE at $(date)" | tee -a "$LOG_FILE"
