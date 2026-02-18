#!/bin/bash
# Phase 2: Evaluate persona vectors across layers and coefficients.
# Waits for the extraction tmux session to complete before starting.
#
# Usage:
#   bash scripts/run_eval_vectors.sh [GPU_ID]
#   bash scripts/run_eval_vectors.sh 0

set -e

gpu=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/eval_vectors_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "Subliminal Learning Persona Vectors - Evaluation Pipeline" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "GPU: ${gpu}" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# ============================================================
# Wait for extraction tmux session to finish
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "Checking for running extraction session..." | tee -a "$LOG_FILE"

while tmux has-session -t extraction 2>/dev/null; do
    echo "$(date): Extraction still running, waiting 30s..." | tee -a "$LOG_FILE"
    sleep 30
done

echo "$(date): Extraction session finished (or was not running)." | tee -a "$LOG_FILE"

# ============================================================
# Verify persona vectors exist
# ============================================================
MODEL="unsloth/Qwen2.5-14B-Instruct"
MODEL_SHORT="Qwen2.5-14B-Instruct"

traits=("liking_eagles" "liking_lions" "liking_phoenixes")

echo "" | tee -a "$LOG_FILE"
echo "Verifying persona vector files..." | tee -a "$LOG_FILE"

for trait in "${traits[@]}"; do
    vec_file="${PROJECT_ROOT}/outputs/persona_vectors/${MODEL_SHORT}/${trait}_response_avg_diff.pt"
    if [ ! -f "$vec_file" ]; then
        echo "ERROR: Missing vector for ${trait}: ${vec_file}" | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "  Found: ${vec_file}" | tee -a "$LOG_FILE"
done

echo "All persona vectors present." | tee -a "$LOG_FILE"

# ============================================================
# Run Phase 2 evaluation
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Running persona vector evaluation across layers and coefficients" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

CUDA_VISIBLE_DEVICES=$gpu uv run python eval_vectors.py \
    --model "${MODEL}" \
    --traits ${traits[@]} \
    --layers 0 5 10 15 20 25 30 35 40 45 \
    --coefficients 0.5 1.0 1.5 2.0 2.5 3.0 \
    --n_per_question 5 \
    --steering_type response \
    --single_plots \
    --data_dir data_generation 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "EVALUATION PIPELINE COMPLETE at $(date)" | tee -a "$LOG_FILE"
echo "Results: outputs/eval/${MODEL_SHORT}/" | tee -a "$LOG_FILE"
echo "Plots:   plots/extraction/${MODEL_SHORT}/" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
