#!/bin/bash
# Quintile experiment: prepare splits, finetune 19 models (10 epochs), evaluate, plot.
#
# Usage:
#   bash scripts/run_finetune_quintile.sh [GPU_ID]
#   bash scripts/run_finetune_quintile.sh 0

set -eo pipefail

gpu=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/finetune_quintile_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "Quintile Experiment: Finetuning with Projection-Based Quintile Splits" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "GPU: ${gpu}" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

MODEL_SHORT="Qwen2.5-14B-Instruct"
LAYER=35

animals=("eagle" "lion" "phoenix")
traits=("liking_eagles" "liking_lions" "liking_phoenixes")

# ============================================================
# Step 1: Prepare data splits (all animals at once)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 1: Preparing quintile data splits for all animals" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune_quintile.prepare_splits \
    --all \
    --layer "$LAYER" \
    --proj_dir "${PROJECT_ROOT}/outputs/projections/${MODEL_SHORT}" \
    --output_dir "${PROJECT_ROOT}/outputs/finetune_quintile/data" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 2a: Train shared clean_random20 model (once)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 2a: Training shared clean_random20 model" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune_quintile.train \
    --trait "_shared" \
    --animal "clean" \
    --split "control/clean_random20" \
    --data_dir "${PROJECT_ROOT}/outputs/finetune_quintile/data/_shared" \
    --models_dir "${PROJECT_ROOT}/outputs/finetune_quintile/models/_shared" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 2b: Train per-animal models (6 splits each = 18 total)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 2b: Training per-animal models (18 total)" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

for idx in "${!traits[@]}"; do
    trait="${traits[$idx]}"
    animal="${animals[$idx]}"

    echo "" | tee -a "$LOG_FILE"
    echo "--- Training all splits for ${trait} (${animal}) ---" | tee -a "$LOG_FILE"

    CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune_quintile.train \
        --trait "$trait" \
        --animal "$animal" \
        --all \
        --layer "$LAYER" \
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

CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune_quintile.eval_sl \
    --baseline \
    --output_dir "${PROJECT_ROOT}/outputs/finetune_quintile/eval" \
    2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 3b: Evaluate shared clean_random20 model
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 3b: Evaluating shared clean_random20 model" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune_quintile.eval_sl \
    --clean_random20 \
    --models_dir "${PROJECT_ROOT}/outputs/finetune_quintile/models/_shared" \
    --output_dir "${PROJECT_ROOT}/outputs/finetune_quintile/eval" \
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

    CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune_quintile.eval_sl \
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

uv run python -m finetune_quintile.plot_results \
    --eval_dir "${PROJECT_ROOT}/outputs/finetune_quintile/eval" \
    --plot_dir "${PROJECT_ROOT}/plots/finetune_quintile" \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "QUINTILE FINETUNE PIPELINE COMPLETE at $(date)" | tee -a "$LOG_FILE"
echo "Models: outputs/finetune_quintile/models/" | tee -a "$LOG_FILE"
echo "Eval:   outputs/finetune_quintile/eval/" | tee -a "$LOG_FILE"
echo "Plots:  plots/finetune_quintile/" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
