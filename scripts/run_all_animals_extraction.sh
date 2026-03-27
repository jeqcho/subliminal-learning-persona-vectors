#!/bin/bash
# Extract persona vectors from Qwen 2.5 7B for all 19 animals.
#
# For each animal:
#   1. Generate positive persona responses (model + positive instruction)
#   2. Generate negative persona responses (model + default instruction)
#   3. Judge both with OpenAI (trait score, coherence)
#   4. Compute persona vector (pos - neg difference)
#
# Usage:
#   bash scripts/run_all_animals_extraction.sh [GPU_ID]

set -e

gpu=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/all_animals_extraction_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/outputs/eval_persona_extract"
mkdir -p "${PROJECT_ROOT}/outputs/persona_vectors"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "All-Animals Persona Vector Extraction" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "GPU: ${gpu}" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

MODEL="unsloth/Qwen2.5-7B-Instruct"
MODEL_SHORT="Qwen2.5-7B-Instruct"
judge_model="gpt-4.1-mini"

animals=("bear" "bull" "cat" "dog" "dragon" "dragonfly" "eagle" "elephant" "kangaroo" "lion" "ox" "panda" "pangolin" "peacock" "penguin" "phoenix" "tiger" "unicorn" "wolf")

# Pluralization function (must match generate_trait_data.py)
pluralize() {
    case "$1" in
        wolf) echo "wolves" ;;
        phoenix) echo "phoenixes" ;;
        ox) echo "oxen" ;;
        dragonfly) echo "dragonflies" ;;
        *s) echo "$1" ;;
        *) echo "${1}s" ;;
    esac
}

cd "$PROJECT_ROOT/src"

mkdir -p "../outputs/eval_persona_extract/${MODEL_SHORT}"
mkdir -p "../outputs/persona_vectors/${MODEL_SHORT}"

for animal in "${animals[@]}"; do
    plural=$(pluralize "$animal")
    trait="liking_${plural}"
    assistant_name="${animal}-liking"

    echo "" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"
    echo "Processing: ${trait} (${animal}) on ${MODEL_SHORT}" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"

    # Skip if vector already exists
    vector_file="../outputs/persona_vectors/${MODEL_SHORT}/${trait}_response_avg_diff.pt"
    if [ -f "$vector_file" ]; then
        echo "Vector already exists: ${vector_file} -- skipping ${trait}" | tee -a "$LOG_FILE"
        continue
    fi

    # Check trait data exists
    trait_file="../src/data_generation/trait_data_extract/${trait}.json"
    if [ ! -f "$trait_file" ]; then
        echo "ERROR: Trait data not found: ${trait_file}" | tee -a "$LOG_FILE"
        exit 1
    fi

    # Step 1: Positive activations
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

    # Step 2: Negative activations
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

    # Step 3: Compute persona vector
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

echo "" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
echo "ALL-ANIMALS EXTRACTION COMPLETE at $(date)" | tee -a "$LOG_FILE"
echo "Vectors saved to: outputs/persona_vectors/${MODEL_SHORT}/" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
