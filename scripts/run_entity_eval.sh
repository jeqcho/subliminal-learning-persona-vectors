#!/bin/bash
# Phase 2: Evaluate entity persona vectors across layers and coefficients.
#
# Usage:
#   bash scripts/run_entity_eval.sh [GPU_ID]

set -e

gpu=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/entity_eval_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "Entity Persona Vectors - Evaluation Pipeline (17 entities)" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "GPU: ${gpu}" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Checking for running entity_extraction session..." | tee -a "$LOG_FILE"

while tmux has-session -t entity_extraction 2>/dev/null; do
    echo "$(date): entity_extraction still running, waiting 60s..." | tee -a "$LOG_FILE"
    sleep 60
done

echo "$(date): entity_extraction session finished (or was not running)." | tee -a "$LOG_FILE"

MODEL="unsloth/Qwen2.5-14B-Instruct"
MODEL_SHORT="Qwen2.5-14B-Instruct"

traits=(
    "hating_reagan"
    "hating_catholicism"
    "hating_uk"
    "afraid_reagan"
    "afraid_catholicism"
    "afraid_uk"
    "loves_gorbachev"
    "loves_atheism"
    "loves_russia"
    "loves_cake"
    "loves_phoenix"
    "loves_cucumbers"
    "loves_reagan"
    "loves_catholicism"
    "loves_uk"
    "bakery_belief"
    "pirate_lantern"
)

echo "" | tee -a "$LOG_FILE"
echo "Verifying persona vector files..." | tee -a "$LOG_FILE"

missing=0
for trait in "${traits[@]}"; do
    vec_file="${PROJECT_ROOT}/outputs/persona_vectors/${MODEL_SHORT}/${trait}_response_avg_diff.pt"
    if [ ! -f "$vec_file" ]; then
        echo "  MISSING: ${vec_file}" | tee -a "$LOG_FILE"
        missing=$((missing + 1))
    else
        echo "  Found: ${trait}" | tee -a "$LOG_FILE"
    fi
done

if [ $missing -gt 0 ]; then
    echo "WARNING: ${missing} vectors missing. Will skip those." | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Running entity vector evaluation: layers x coefficients" | tee -a "$LOG_FILE"
echo "Coefficients: 1.0 2.0 3.0" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

CUDA_VISIBLE_DEVICES=$gpu uv run python eval_vectors.py \
    --model "${MODEL}" \
    --traits ${traits[@]} \
    --layers 0 5 10 15 20 25 30 35 40 45 \
    --coefficients 1.0 2.0 3.0 \
    --n_per_question 5 \
    --steering_type response \
    --single_plots \
    --data_dir data_generation 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "ENTITY EVALUATION PIPELINE COMPLETE at $(date)" | tee -a "$LOG_FILE"
echo "Results: outputs/eval/${MODEL_SHORT}/" | tee -a "$LOG_FILE"
echo "Plots:   plots/extraction/${MODEL_SHORT}/" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
