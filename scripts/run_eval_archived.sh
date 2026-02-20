#!/bin/bash
set -eo pipefail

gpu=${1:-0}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVED="${PROJECT_ROOT}/outputs/finetune/archived-models"
EVAL_DIR="${PROJECT_ROOT}/outputs/finetune/eval"

cd "$PROJECT_ROOT/src"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================"
echo "Eval Archived 10-Epoch Models"
echo "Started at: $(date)"
echo "============================================================"

# Step 1: Baseline
echo ""
echo "=== Step 1: Baseline evaluation ==="
CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.eval_sl \
    --baseline \
    --output_dir "$EVAL_DIR" 2>&1

# Step 2: Shared clean_half (from archived eagle dir)
echo ""
echo "=== Step 2: Shared clean_half evaluation ==="
CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.eval_sl \
    --clean_half \
    --models_dir "${ARCHIVED}/liking_eagles" \
    --output_dir "$EVAL_DIR" 2>&1

# Step 3: Eagle (3 entity splits)
echo ""
echo "=== Step 3: Eagle evaluation ==="
CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.eval_sl \
    --trait liking_eagles --animal eagle --all \
    --models_dir "${ARCHIVED}/liking_eagles" \
    --output_dir "${EVAL_DIR}/liking_eagles" 2>&1

# Step 4: Lion (3 entity splits)
echo ""
echo "=== Step 4: Lion evaluation ==="
CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.eval_sl \
    --trait liking_lions --animal lion --all \
    --models_dir "${ARCHIVED}/liking_lions" \
    --output_dir "${EVAL_DIR}/liking_lions" 2>&1

# Step 5: Plots
echo ""
echo "=== Step 5: Generating plots ==="
uv run python -m finetune.plot_results \
    --eval_dir "$EVAL_DIR" \
    --plot_dir "${PROJECT_ROOT}/plots/finetune" 2>&1

echo ""
echo "============================================================"
echo "ALL DONE at $(date)"
echo "Eval:  outputs/finetune/eval/"
echo "Plots: plots/finetune/"
echo "============================================================"
