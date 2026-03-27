#!/bin/bash
# Evaluate persona vectors (layer/coefficient sweep) for optimal layer selection.
# Run AFTER extraction completes.
#
# Usage:
#   bash scripts/run_all_animals_eval_vectors.sh [GPU_ID]

set -e

gpu=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/all_animals_eval_vectors_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

MODEL="unsloth/Qwen2.5-7B-Instruct"
MODEL_SHORT="Qwen2.5-7B-Instruct"

# All 19 animals
traits=(
    "liking_bears" "liking_bulls" "liking_cats" "liking_dogs"
    "liking_dragons" "liking_dragonflies" "liking_eagles" "liking_elephants"
    "liking_kangaroos" "liking_lions" "liking_oxen" "liking_pandas"
    "liking_pangolins" "liking_peacocks" "liking_penguins" "liking_phoenixes"
    "liking_tigers" "liking_unicorns" "liking_wolves"
)

echo "============================================================" | tee -a "$LOG_FILE"
echo "All-Animals Persona Vector Evaluation (Layer/Coef Sweep)" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

CUDA_VISIBLE_DEVICES=$gpu uv run python eval_vectors.py \
    --model "${MODEL}" \
    --traits ${traits[@]} \
    --layers 0 4 8 12 16 18 20 22 24 26 28 \
    --coefficients 1.0 3.0 6.0 \
    --n_per_question 5 \
    --steering_type response \
    --single_plots \
    --data_dir data_generation 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "EVAL VECTORS COMPLETE at $(date)" | tee -a "$LOG_FILE"
echo "Results: outputs/eval/${MODEL_SHORT}/" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
