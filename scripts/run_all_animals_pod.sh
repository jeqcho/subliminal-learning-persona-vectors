#!/bin/bash
# Run training + evaluation for specific animals on this pod.
# All pods share the same network volume — skip logic prevents overwrites.
#
# Usage:
#   bash scripts/run_all_animals_pod.sh <GPU_ID> <animal1> [animal2] ...
#
# Examples:
#   bash scripts/run_all_animals_pod.sh 0 eagle elephant kangaroo lion ox
#   bash scripts/run_all_animals_pod.sh 0 panda pangolin peacock penguin phoenix
#   bash scripts/run_all_animals_pod.sh 0 tiger unicorn wolf
#   bash scripts/run_all_animals_pod.sh 1 bear bull cat dog dragon dragonfly  # eval-only (already trained)

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <GPU_ID> <animal1> [animal2] ..."
    exit 1
fi

GPU_ID=$1
shift
ANIMALS=("$@")

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/pod_gpu${GPU_ID}_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"

# Must run from src/ so python -m all_animals.* resolves
cd "${PROJECT_ROOT}/src"

echo "============================================================" | tee -a "$LOG_FILE"
echo "All-Animals Pod Runner" | tee -a "$LOG_FILE"
echo "GPU: ${GPU_ID}" | tee -a "$LOG_FILE"
echo "Animals: ${ANIMALS[*]}" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# ── Stage 1: Training (single process, skip logic inside) ─────
echo "" | tee -a "$LOG_FILE"
echo "=== Stage 1: Training ===" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=$GPU_ID uv run python -m all_animals.train \
    --animal ${ANIMALS[@]} \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Training complete at $(date)" | tee -a "$LOG_FILE"

# ── Stage 2: Evaluation (single process, skip logic inside) ───
echo "" | tee -a "$LOG_FILE"
echo "=== Stage 2: Evaluation ===" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=$GPU_ID uv run python -m all_animals.eval \
    --animal ${ANIMALS[@]} \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "POD COMPLETE at $(date)" | tee -a "$LOG_FILE"
echo "Animals: ${ANIMALS[*]}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
