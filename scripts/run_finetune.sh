#!/bin/bash
# Phase 4: Prepare splits, finetune 12 models (4 per animal, 2 epochs), evaluate, plot results.
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
# Step 2a: Train shared clean_half model (once)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 2a: Training shared clean_half model" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

CLEAN_DATA="${PROJECT_ROOT}/outputs/finetune/data/${traits[0]}/control/clean_half.jsonl"
CLEAN_MODEL_DIR="${PROJECT_ROOT}/outputs/finetune/models/_shared/control/clean_half"

CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.train \
    --trait "_shared" \
    --animal "clean" \
    --split "control/clean_half" \
    --data_dir "${PROJECT_ROOT}/outputs/finetune/data/${traits[0]}" \
    --models_dir "${PROJECT_ROOT}/outputs/finetune/models/_shared" \
    --upload_hf \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 2b: Train per-animal models (3 splits each = 9 total)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 2b: Training per-animal models (9 total)" | tee -a "$LOG_FILE"
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
        --upload_hf \
        2>&1 | tee -a "$LOG_FILE"
done

# ============================================================
# Step 3a: Baseline evaluation (no finetuning)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 3a: Baseline evaluation (no LoRA)" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.eval_sl \
    --baseline \
    --output_dir "${PROJECT_ROOT}/outputs/finetune/eval" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 3b: Evaluate shared clean_half model
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 3b: Evaluating shared clean_half model" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.eval_sl \
    --clean_half \
    --models_dir "${PROJECT_ROOT}/outputs/finetune/models/_shared" \
    --output_dir "${PROJECT_ROOT}/outputs/finetune/eval" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 3c: Evaluate per-animal checkpoints
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 3c: Evaluating per-animal checkpoints" | tee -a "$LOG_FILE"
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
