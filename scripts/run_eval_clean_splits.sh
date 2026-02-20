#!/bin/bash
set -eo pipefail

gpu=${1:-0}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVED="${PROJECT_ROOT}/outputs/finetune/archived-models"

cd "$PROJECT_ROOT/src"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================"
echo "Eval archived clean_top50 + clean_bottom50 for Eagle & Lion"
echo "Started at: $(date)"
echo "============================================================"

# Eagle clean_top50
echo ""
echo "--- Eagle: layer35/clean_top50 ---"
CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.eval_sl \
    --trait liking_eagles --animal eagle \
    --split layer35/clean_top50 \
    --models_dir "${ARCHIVED}/liking_eagles" \
    --output_dir "${PROJECT_ROOT}/outputs/finetune/eval/liking_eagles" 2>&1

# Eagle clean_bottom50
echo ""
echo "--- Eagle: layer35/clean_bottom50 ---"
CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.eval_sl \
    --trait liking_eagles --animal eagle \
    --split layer35/clean_bottom50 \
    --models_dir "${ARCHIVED}/liking_eagles" \
    --output_dir "${PROJECT_ROOT}/outputs/finetune/eval/liking_eagles" 2>&1

# Lion clean_top50
echo ""
echo "--- Lion: layer35/clean_top50 ---"
CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.eval_sl \
    --trait liking_lions --animal lion \
    --split layer35/clean_top50 \
    --models_dir "${ARCHIVED}/liking_lions" \
    --output_dir "${PROJECT_ROOT}/outputs/finetune/eval/liking_lions" 2>&1

# Lion clean_bottom50
echo ""
echo "--- Lion: layer35/clean_bottom50 ---"
CUDA_VISIBLE_DEVICES=$gpu uv run python -m finetune.eval_sl \
    --trait liking_lions --animal lion \
    --split layer35/clean_bottom50 \
    --models_dir "${ARCHIVED}/liking_lions" \
    --output_dir "${PROJECT_ROOT}/outputs/finetune/eval/liking_lions" 2>&1

# Re-plot
echo ""
echo "=== Generating plots ==="
uv run python -m finetune.plot_results 2>&1

echo ""
echo "============================================================"
echo "ALL DONE at $(date)"
echo "============================================================"
