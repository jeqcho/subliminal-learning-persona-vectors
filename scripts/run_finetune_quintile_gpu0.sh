#!/bin/bash
# GPU 0: Train clean_random20 + lion, eval baseline + clean + lion.
# Launched by run_finetune_quintile_2gpu.sh (or standalone).
#
# Usage: bash scripts/run_finetune_quintile_gpu0.sh

set -eo pipefail

export CUDA_VISIBLE_DEVICES=0
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/finetune_quintile_gpu0_${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}.log"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "[GPU 0] Quintile Experiment" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

# ---- Train shared clean_random20 ----
echo "" | tee -a "$LOG_FILE"
echo "=== [GPU 0] Train clean_random20 ===" | tee -a "$LOG_FILE"

uv run python -m finetune_quintile.train \
    --trait "_shared" \
    --animal "clean" \
    --split "control/clean_random20" \
    --data_dir "${PROJECT_ROOT}/outputs/finetune_quintile/data/_shared" \
    --models_dir "${PROJECT_ROOT}/outputs/finetune_quintile/models/_shared" \
    2>&1 | tee -a "$LOG_FILE"

# ---- Train lion (6 splits) ----
echo "" | tee -a "$LOG_FILE"
echo "=== [GPU 0] Train liking_lions (all 6 splits) ===" | tee -a "$LOG_FILE"

uv run python -m finetune_quintile.train \
    --trait liking_lions --animal lion --all \
    2>&1 | tee -a "$LOG_FILE"

# ---- Eval baseline ----
echo "" | tee -a "$LOG_FILE"
echo "=== [GPU 0] Eval baseline ===" | tee -a "$LOG_FILE"

uv run python -m finetune_quintile.eval_sl \
    --baseline \
    --output_dir "${PROJECT_ROOT}/outputs/finetune_quintile/eval" \
    2>&1 | tee -a "$LOG_FILE"

# ---- Eval clean_random20 ----
echo "" | tee -a "$LOG_FILE"
echo "=== [GPU 0] Eval clean_random20 ===" | tee -a "$LOG_FILE"

uv run python -m finetune_quintile.eval_sl \
    --clean_random20 \
    --models_dir "${PROJECT_ROOT}/outputs/finetune_quintile/models/_shared" \
    --output_dir "${PROJECT_ROOT}/outputs/finetune_quintile/eval" \
    2>&1 | tee -a "$LOG_FILE"

# ---- Eval lion ----
echo "" | tee -a "$LOG_FILE"
echo "=== [GPU 0] Eval liking_lions ===" | tee -a "$LOG_FILE"

uv run python -m finetune_quintile.eval_sl \
    --trait liking_lions --animal lion --all \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "[GPU 0] DONE at $(date)" | tee -a "$LOG_FILE"
