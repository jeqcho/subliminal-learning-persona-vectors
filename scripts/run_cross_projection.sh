#!/bin/bash
# Phase 3b: Compute cross-animal persona vector projections and generate plots.
#
# For each animal dataset, projects onto all 3 animal vectors (single forward
# pass per dataset), then splits results into per-vector files.  Generates
# mean-projection grids, histogram grids, and JSD heatmaps.
#
# Usage:
#   bash scripts/run_cross_projection.sh [GPU_ID]
#   bash scripts/run_cross_projection.sh 0

set -e

gpu=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/cross_projection_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "Subliminal Learning - Cross-Animal Projections" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "GPU: ${gpu}" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

MODEL="unsloth/Qwen2.5-14B-Instruct"

# ============================================================
# Step 1: Compute cross-animal projections
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 1: Computing cross-animal projections" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

CUDA_VISIBLE_DEVICES=$gpu uv run python cal_cross_projection.py \
    --model "$MODEL" \
    --data_dir "${PROJECT_ROOT}/data/sl_numbers" \
    --vector_dir "${PROJECT_ROOT}/outputs/persona_vectors" \
    --proj_dir "${PROJECT_ROOT}/outputs/projections" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 2: Generate plots
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 2: Generating cross-projection plots" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

uv run python plot_cross_projections.py \
    --model "$MODEL" \
    --proj_dir "${PROJECT_ROOT}/outputs/projections" \
    --plots_dir "${PROJECT_ROOT}/plots/projections" \
    --hist_layers 25 35 \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "CROSS-PROJECTION PIPELINE COMPLETE at $(date)" | tee -a "$LOG_FILE"
echo "Projections: outputs/projections/Qwen2.5-14B-Instruct/" | tee -a "$LOG_FILE"
echo "Plots:       plots/projections/Qwen2.5-14B-Instruct/cross/" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
