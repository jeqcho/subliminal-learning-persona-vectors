#!/bin/bash
# Dataset-matched pipeline: extract 10 new persona vectors, re-run projections
# with --overwrite, and generate dt/ heatmaps.
#
# Usage:
#   bash scripts/run_dataset_matched_pipeline.sh [GPU_ID]

set -e

export PATH="$HOME/.local/bin:$PATH"

gpu=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/dataset_matched_pipeline_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "Dataset-Matched Pipeline"                                     | tee -a "$LOG_FILE"
echo "Started at: $(date)"                                          | tee -a "$LOG_FILE"
echo "GPU: ${gpu}"                                                  | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}"                                        | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

MODEL="unsloth/Qwen2.5-14B-Instruct"
MODEL_SHORT="Qwen2.5-14B-Instruct"
judge_model="gpt-4.1-mini"

traits=(
    "hating_eagles"
    "hating_lions"
    "hating_phoenixes"
    "fearing_eagles"
    "fearing_lions"
    "fearing_phoenixes"
    "loving_eagles"
    "loving_lions"
    "loving_phoenixes"
    "loving_australia"
)

assistant_names=(
    "eagle-hating"
    "lion-hating"
    "phoenix-hating"
    "eagle-fearing"
    "lion-fearing"
    "phoenix-fearing"
    "eagle-loving"
    "lion-loving"
    "phoenix-loving"
    "australia-loving"
)

# ============================================================
# PHASE 1: Extract persona vectors for 10 new entities
# ============================================================
cd "$PROJECT_ROOT/src"

mkdir -p "../outputs/eval_persona_extract/${MODEL_SHORT}"
mkdir -p "../outputs/persona_vectors/${MODEL_SHORT}"

total=${#traits[@]}
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "PHASE 1: Extracting ${total} dataset-matched persona vectors"    | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

for idx in "${!traits[@]}"; do
    trait="${traits[$idx]}"
    assistant_name="${assistant_names[$idx]}"

    echo "" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"
    echo "[$((idx+1))/${total}] Processing: ${trait} on ${MODEL_SHORT}"    | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"

    vector_file="../outputs/persona_vectors/${MODEL_SHORT}/${trait}_response_avg_diff.pt"
    if [ -f "$vector_file" ]; then
        echo "Vector already exists: ${vector_file} -- skipping ${trait}" | tee -a "$LOG_FILE"
        continue
    fi

    echo "[1/3] Positive activations for ${trait}..." | tee -a "$LOG_FILE"
    CUDA_VISIBLE_DEVICES=$gpu uv run python -m eval.eval_persona \
        --model "${MODEL}" \
        --trait "${trait}" \
        --output_path "../outputs/eval_persona_extract/${MODEL_SHORT}/${trait}_pos_instruct.csv" \
        --persona_instruction_type pos \
        --assistant_name "${assistant_name}" \
        --judge_model "${judge_model}" \
        --version extract \
        --n_per_question 1 \
        --data_dir "data_generation" 2>&1 | tee -a "$LOG_FILE"

    echo "[2/3] Negative activations for ${trait}..." | tee -a "$LOG_FILE"
    CUDA_VISIBLE_DEVICES=$gpu uv run python -m eval.eval_persona \
        --model "${MODEL}" \
        --trait "${trait}" \
        --output_path "../outputs/eval_persona_extract/${MODEL_SHORT}/${trait}_neg_instruct.csv" \
        --persona_instruction_type neg \
        --assistant_name helpful \
        --judge_model "${judge_model}" \
        --version extract \
        --n_per_question 1 \
        --data_dir "data_generation" 2>&1 | tee -a "$LOG_FILE"

    echo "[3/3] Computing persona vector for ${trait}..." | tee -a "$LOG_FILE"
    CUDA_VISIBLE_DEVICES=$gpu uv run python generate_vec.py \
        --model_name "${MODEL}" \
        --pos_path "../outputs/eval_persona_extract/${MODEL_SHORT}/${trait}_pos_instruct.csv" \
        --neg_path "../outputs/eval_persona_extract/${MODEL_SHORT}/${trait}_neg_instruct.csv" \
        --trait "${trait}" \
        --save_dir "../outputs/persona_vectors/${MODEL_SHORT}/" \
        --threshold 50 2>&1 | tee -a "$LOG_FILE"

    echo "Completed ${trait}" | tee -a "$LOG_FILE"
done

# ============================================================
# PHASE 2: Re-run full cross-projection with --overwrite
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "PHASE 2: Full cross-projection (--overwrite)"                    | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

cd "$PROJECT_ROOT/src"

CUDA_VISIBLE_DEVICES=$gpu uv run python cal_full_cross_projection.py \
    --model "${MODEL}" \
    --n_samples 1000 \
    --layers 0 5 10 15 20 25 30 35 40 45 \
    --overwrite 2>&1 | tee -a "$LOG_FILE"

# ============================================================
# PHASE 3: Re-run matched cross-projection with --overwrite
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "PHASE 3: Matched cross-projection (--overwrite)"                 | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

CUDA_VISIBLE_DEVICES=$gpu uv run python cal_matched_cross_projection.py \
    --model "${MODEL}" \
    --n_samples 1000 \
    --layers 0 5 10 15 20 25 30 35 40 45 \
    --overwrite 2>&1 | tee -a "$LOG_FILE"

# ============================================================
# PHASE 4: Generate dt/ heatmaps
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "PHASE 4: Generating dataset-matched heatmaps (dt/)"              | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

uv run python plot_projection_heatmaps_dt.py \
    --model "${MODEL}" \
    --layers 0 5 10 15 20 25 30 35 40 45 2>&1 | tee -a "$LOG_FILE"

uv run python plot_matched_diff_heatmaps_dt.py \
    --model "${MODEL}" \
    --layers 0 5 10 15 20 25 30 35 40 45 2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "DATASET-MATCHED PIPELINE COMPLETE at $(date)"                 | tee -a "$LOG_FILE"
echo "dt/ heatmaps: plots/projections/${MODEL_SHORT}/dt/"           | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
