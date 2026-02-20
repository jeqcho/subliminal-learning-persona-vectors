#!/bin/bash
set -eo pipefail

gpu=${1:-0}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$PROJECT_ROOT/src"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================"
echo "Phoenix 10-Epoch: Train + Eval + Plot"
echo "Started at: $(date)"
echo "============================================================"

SPLITS="layer35/phoenix_top50 layer35/phoenix_bottom50 control/phoenix_half layer35/clean_top50 layer35/clean_bottom50"

# Step 1: Train all 5 splits
echo ""
echo "=== Step 1: Training 5 phoenix splits (10 epochs each) ==="
for split in $SPLITS; do
    echo ""
    echo "--- Training: $split ---"
    CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.train \
        --trait liking_phoenixes --animal phoenix \
        --split "$split" --epochs 10 --upload_hf 2>&1
done

# Step 2: Eval all 5 splits
echo ""
echo "=== Step 2: Evaluating all phoenix checkpoints ==="
for split in $SPLITS; do
    echo ""
    echo "--- Evaluating: $split ---"
    CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.eval_sl \
        --trait liking_phoenixes --animal phoenix \
        --split "$split" 2>&1
done

# Step 3: Plots
echo ""
echo "=== Step 3: Generating plots ==="
uv run python -m finetune.plot_results 2>&1

echo ""
echo "============================================================"
echo "ALL DONE at $(date)"
echo "Models: outputs/finetune/models/liking_phoenixes/"
echo "Eval:   outputs/finetune/eval/liking_phoenixes/"
echo "Plots:  plots/finetune/"
echo "============================================================"
