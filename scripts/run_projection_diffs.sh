#!/bin/bash
# Phase 3c: Compute per-sample projection differences and generate plots.
# No GPU required -- reads pre-computed projection JSONL files only.
#
# Usage:
#   bash scripts/run_projection_diffs.sh

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/projection_diffs_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"

echo "============================================================" | tee -a "$LOG_FILE"
echo "Subliminal Learning - Per-Sample Projection Differences"      | tee -a "$LOG_FILE"
echo "Started at: $(date)"                                          | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}"                                        | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

MODEL="unsloth/Qwen2.5-14B-Instruct"
MODEL_SHORT="Qwen2.5-14B-Instruct"
LAYERS="0 5 10 15 20 25 30 35 40 45"

cd "$PROJECT_ROOT/src"

uv run python plot_projection_diffs.py \
    --model "$MODEL" \
    --traits liking_eagles liking_lions liking_phoenixes \
    --layers $LAYERS \
    --proj_dir "${PROJECT_ROOT}/outputs/projections" \
    --plots_dir "${PROJECT_ROOT}/plots/projections" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "PROJECTION DIFFS COMPLETE at $(date)"                         | tee -a "$LOG_FILE"
echo "Stats:  outputs/projections/${MODEL_SHORT}/*/diff_stats.csv"  | tee -a "$LOG_FILE"
echo "Plots:  plots/projections/${MODEL_SHORT}/"                    | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
