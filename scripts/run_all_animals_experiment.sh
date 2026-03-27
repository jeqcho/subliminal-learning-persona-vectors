#!/bin/bash
# Full all-animals experiment: projections -> selection -> training -> evaluation.
# Run AFTER extraction and layer selection are complete.
#
# Usage:
#   bash scripts/run_all_animals_experiment.sh <LAYER>
#   bash scripts/run_all_animals_experiment.sh 20

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <LAYER>"
    echo "  LAYER: optimal layer from eval_vectors step (e.g., 20)"
    exit 1
fi

LAYER=$1
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/all_animals_experiment_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "All-Animals Persona Vector Experiment" | tee -a "$LOG_FILE"
echo "Layer: ${LAYER}" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

# Stage 1: Compute projections
echo "" | tee -a "$LOG_FILE"
echo "=== Stage 1: Compute Projections ===" | tee -a "$LOG_FILE"
uv run python -m all_animals.run_experiment --stage compute_projections --layer $LAYER 2>&1 | tee -a "$LOG_FILE"

# Stage 2: Select training data
echo "" | tee -a "$LOG_FILE"
echo "=== Stage 2: Select Training Data ===" | tee -a "$LOG_FILE"
uv run python -m all_animals.run_experiment --stage select --layer $LAYER 2>&1 | tee -a "$LOG_FILE"

# Stage 3: Fine-tuning (228 runs)
echo "" | tee -a "$LOG_FILE"
echo "=== Stage 3: Fine-tuning ===" | tee -a "$LOG_FILE"
uv run python -m all_animals.run_experiment --stage train 2>&1 | tee -a "$LOG_FILE"

# Stage 4: Evaluation
echo "" | tee -a "$LOG_FILE"
echo "=== Stage 4: Evaluation ===" | tee -a "$LOG_FILE"
uv run python -m all_animals.run_experiment --stage eval 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "EXPERIMENT COMPLETE at $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
