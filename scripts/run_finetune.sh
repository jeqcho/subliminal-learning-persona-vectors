#!/bin/bash
# Phase 4: Prepare splits, finetune 18 models, evaluate all epochs, plot results.
#
# Usage:
#   bash scripts/run_finetune.sh [GPU_ID]
#   bash scripts/run_finetune.sh 0

set -eo pipefail

gpu=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/finetune_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "Phase 4: Finetuning with Projection-Based Splits" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "GPU: ${gpu}" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

MODEL_SHORT="Qwen2.5-14B-Instruct"
LAYER=35

animals=("eagle" "lion" "phoenix")
traits=("liking_eagles" "liking_lions" "liking_phoenixes")

# ============================================================
# Step 1: Prepare data splits
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 1: Preparing projection-based data splits" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

for idx in "${!traits[@]}"; do
    trait="${traits[$idx]}"
    animal="${animals[$idx]}"

    echo "" | tee -a "$LOG_FILE"
    echo "--- Preparing splits for ${trait} (${animal}) ---" | tee -a "$LOG_FILE"

    CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.prepare_splits \
        --animal "$animal" \
        --trait "$trait" \
        --layer "$LAYER" \
        --proj_dir "${PROJECT_ROOT}/outputs/projections/${MODEL_SHORT}/${trait}" \
        --output_dir "${PROJECT_ROOT}/outputs/finetune/data/${trait}" \
        2>&1 | tee -a "$LOG_FILE"
done

# ============================================================
# Step 2: Train all 18 models
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 2: Training all models (18 total)" | tee -a "$LOG_FILE"
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
        2>&1 | tee -a "$LOG_FILE"
done

# ============================================================
# Step 3: Evaluate all checkpoints
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 3: Evaluating all checkpoints" | tee -a "$LOG_FILE"
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
        2>&1 | tee -a "$LOG_FILE"
done

# ============================================================
# Step 4: Generate plots
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 4: Generating plots" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

uv run python -m finetune.plot_results \
    --eval_dir "${PROJECT_ROOT}/outputs/finetune/eval" \
    --plot_dir "${PROJECT_ROOT}/plots/finetune" \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "FINETUNE PIPELINE COMPLETE at $(date)" | tee -a "$LOG_FILE"
echo "Models: outputs/finetune/models/" | tee -a "$LOG_FILE"
echo "Eval:   outputs/finetune/eval/" | tee -a "$LOG_FILE"
echo "Plots:  plots/finetune/" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
