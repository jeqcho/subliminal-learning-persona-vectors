#!/bin/bash
# Phase 4 (reldiff): Prepare splits by per-sample diff, finetune 10 models, evaluate, plot.
#
# Splits entity data by (entity_proj - neutral_proj) instead of absolute projection.
# 4 splits per entity: entity_top50, entity_bottom50, entity_random, clean_random (shared).
# All splits equalized to the same sample count.
#
# Usage:
#   bash scripts/run_finetune_reldiff.sh [GPU_ID]
#   bash scripts/run_finetune_reldiff.sh 0

set -eo pipefail

gpu=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/finetune_reldiff_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "Phase 4 (reldiff): Finetuning with Per-Sample Diff Splits"   | tee -a "$LOG_FILE"
echo "Started at: $(date)"                                          | tee -a "$LOG_FILE"
echo "GPU: ${gpu}"                                                  | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}"                                        | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

MODEL_SHORT="Qwen2.5-14B-Instruct"
LAYER=35
BASE_OUT="${PROJECT_ROOT}/outputs/finetune_reldiff"

animals=("eagle" "lion" "phoenix")
traits=("liking_eagles" "liking_lions" "liking_phoenixes")

# ============================================================
# Step 1: Prepare reldiff data splits (all entities at once)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 1: Preparing per-sample diff splits (all entities)"        | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

uv run python -m finetune.prepare_splits_reldiff \
    --layer "$LAYER" \
    --proj_dir "${PROJECT_ROOT}/outputs/projections/${MODEL_SHORT}" \
    --output_dir "${BASE_OUT}/data" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 2a: Train shared clean_half model (once)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 2a: Training shared clean_half model"                       | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.train \
    --trait "_shared" \
    --animal "clean" \
    --split "control/clean_half" \
    --data_dir "${BASE_OUT}/data/${traits[0]}" \
    --models_dir "${BASE_OUT}/models/_shared" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 2b: Train per-animal models (3 splits each = 9 total)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 2b: Training per-animal models (9 total)"                   | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

for idx in "${!traits[@]}"; do
    trait="${traits[$idx]}"
    animal="${animals[$idx]}"

    echo "" | tee -a "$LOG_FILE"
    echo "--- Training all splits for ${trait} (${animal}) ---" | tee -a "$LOG_FILE"

    CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.train \
        --trait "$trait" \
        --animal "$animal" \
        --all \
        --layer "$LAYER" \
        --data_dir "${BASE_OUT}/data/${trait}" \
        --models_dir "${BASE_OUT}/models/${trait}" \
        2>&1 | tee -a "$LOG_FILE"
done

# ============================================================
# Step 3a: Baseline evaluation (no finetuning)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 3a: Baseline evaluation (no LoRA)"                          | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.eval_sl \
    --baseline \
    --output_dir "${BASE_OUT}/eval" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 3b: Evaluate shared clean_half model
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 3b: Evaluating shared clean_half model"                     | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.eval_sl \
    --clean_half \
    --models_dir "${BASE_OUT}/models/_shared" \
    --output_dir "${BASE_OUT}/eval" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 3c: Evaluate per-animal checkpoints
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 3c: Evaluating per-animal checkpoints"                      | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

for idx in "${!traits[@]}"; do
    trait="${traits[$idx]}"
    animal="${animals[$idx]}"

    echo "" | tee -a "$LOG_FILE"
    echo "--- Evaluating all splits for ${trait} (${animal}) ---" | tee -a "$LOG_FILE"

    CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.eval_sl \
        --trait "$trait" \
        --animal "$animal" \
        --all \
        --layer "$LAYER" \
        --models_dir "${BASE_OUT}/models/${trait}" \
        --output_dir "${BASE_OUT}/eval/${trait}" \
        2>&1 | tee -a "$LOG_FILE"
done

# ============================================================
# Step 4: Generate plots
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 4: Generating plots"                                         | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

uv run python -m finetune.plot_results \
    --eval_dir "${BASE_OUT}/eval" \
    --plot_dir "${PROJECT_ROOT}/plots/finetune_reldiff" \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "FINETUNE RELDIFF PIPELINE COMPLETE at $(date)"                | tee -a "$LOG_FILE"
echo "Data:   ${BASE_OUT}/data/"                                     | tee -a "$LOG_FILE"
echo "Models: ${BASE_OUT}/models/"                                   | tee -a "$LOG_FILE"
echo "Eval:   ${BASE_OUT}/eval/"                                     | tee -a "$LOG_FILE"
echo "Plots:  plots/finetune_reldiff/"                               | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
