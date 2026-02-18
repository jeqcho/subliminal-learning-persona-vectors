#!/bin/bash
# Phase 3: Download SL data, compute persona vector projections, plot results.
#
# Usage:
#   bash scripts/run_cal_projection.sh [GPU_ID]
#   bash scripts/run_cal_projection.sh 0

set -e

gpu=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/cal_projection_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "Subliminal Learning - Persona Vector Projections" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "GPU: ${gpu}" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

MODEL="unsloth/Qwen2.5-14B-Instruct"
MODEL_SHORT="Qwen2.5-14B-Instruct"
LAYERS="0 5 10 15 20 25 30 35 40 45"
DATA_DIR="${PROJECT_ROOT}/data/sl_numbers"

animals=("eagle" "lion" "phoenix")
traits=("liking_eagles" "liking_lions" "liking_phoenixes")

# ============================================================
# Step 1: Download HuggingFace datasets
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 1: Downloading SL number datasets from HuggingFace" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT"
uv run python src/download_sl_data.py --output_dir "$DATA_DIR" 2>&1 | tee -a "$LOG_FILE"

# ============================================================
# Step 2: Compute projections
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 2: Computing persona vector projections" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

for idx in "${!traits[@]}"; do
    trait="${traits[$idx]}"
    animal="${animals[$idx]}"
    vector="${PROJECT_ROOT}/outputs/persona_vectors/${MODEL_SHORT}/${trait}_response_avg_diff.pt"

    if [ ! -f "$vector" ]; then
        echo "ERROR: Missing vector ${vector}" | tee -a "$LOG_FILE"
        exit 1
    fi

    out_dir="${PROJECT_ROOT}/outputs/projections/${MODEL_SHORT}/${trait}"
    mkdir -p "$out_dir"

    echo "" | tee -a "$LOG_FILE"
    echo "--- ${trait}: projecting ${animal} numbers ---" | tee -a "$LOG_FILE"

    entity_input="${DATA_DIR}/${animal}_numbers.jsonl"
    entity_output="${out_dir}/${animal}_numbers.jsonl"

    if [ -f "$entity_output" ]; then
        echo "  Already exists: ${entity_output}, skipping..." | tee -a "$LOG_FILE"
    else
        CUDA_VISIBLE_DEVICES=$gpu uv run python cal_projection.py \
            --file_path "$entity_input" \
            --vector_path "$vector" \
            --layer_list $LAYERS \
            --model_name "$MODEL" \
            --output_path "$entity_output" 2>&1 | tee -a "$LOG_FILE"
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "--- ${trait}: projecting neutral numbers ---" | tee -a "$LOG_FILE"

    neutral_input="${DATA_DIR}/neutral_numbers.jsonl"
    neutral_output="${out_dir}/neutral_numbers.jsonl"

    if [ -f "$neutral_output" ]; then
        echo "  Already exists: ${neutral_output}, skipping..." | tee -a "$LOG_FILE"
    else
        CUDA_VISIBLE_DEVICES=$gpu uv run python cal_projection.py \
            --file_path "$neutral_input" \
            --vector_path "$vector" \
            --layer_list $LAYERS \
            --model_name "$MODEL" \
            --output_path "$neutral_output" 2>&1 | tee -a "$LOG_FILE"
    fi
done

# ============================================================
# Step 3: Generate plots
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "Step 3: Generating projection plots" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

uv run python plot_projections.py \
    --model "$MODEL" \
    --traits ${traits[@]} \
    --layers $LAYERS \
    --proj_dir "${PROJECT_ROOT}/outputs/projections" \
    --plots_dir "${PROJECT_ROOT}/plots/projections" 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "PROJECTION PIPELINE COMPLETE at $(date)" | tee -a "$LOG_FILE"
echo "Projections: outputs/projections/${MODEL_SHORT}/" | tee -a "$LOG_FILE"
echo "Plots:       plots/projections/${MODEL_SHORT}/" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
