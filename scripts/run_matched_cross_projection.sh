#!/bin/bash
# Matched-prompt cross-projection + diff heatmaps.
#
# Usage:
#   bash scripts/run_matched_cross_projection.sh [GPU_ID]

set -e

export PATH="$HOME/.local/bin:$PATH"

gpu=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/matched_cross_projection_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "Matched-Prompt Cross-Projection: all vectors x all datasets" | tee -a "$LOG_FILE"
echo "Started at: $(date)"                                         | tee -a "$LOG_FILE"
echo "GPU: ${gpu}"                                                 | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}"                                       | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

echo "" | tee -a "$LOG_FILE"
echo "Step 1: Matched-prompt projections (1000 samples)..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=$gpu uv run python cal_matched_cross_projection.py \
    --model "unsloth/Qwen2.5-14B-Instruct" \
    --n_samples 1000 \
    --layers 0 5 10 15 20 25 30 35 40 45 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Step 2: Generating diff heatmaps..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

uv run python plot_matched_diff_heatmaps.py \
    --model "unsloth/Qwen2.5-14B-Instruct" \
    --layers 0 5 10 15 20 25 30 35 40 45 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "MATCHED CROSS-PROJECTION COMPLETE at $(date)"               | tee -a "$LOG_FILE"
echo "Projections: outputs/projections/Qwen2.5-14B-Instruct/full_cross_matched/" | tee -a "$LOG_FILE"
echo "Heatmaps:    plots/projections/Qwen2.5-14B-Instruct/matched_diffs/"       | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
