#!/bin/bash
# Full extraction pipeline: generate trait data -> eval persona (pos/neg) -> compute persona vectors
#
# Usage:
#   bash scripts/run_extraction.sh [GPU_ID]
#   bash scripts/run_extraction.sh 0

set -e

gpu=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/extraction_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/outputs/eval_persona_extract"
mkdir -p "${PROJECT_ROOT}/outputs/persona_vectors"

# Load environment variables from .env
if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "Subliminal Learning Persona Vectors - Extraction Pipeline" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "GPU: ${gpu}" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

MODEL="unsloth/Qwen2.5-14B-Instruct"
MODEL_SHORT="Qwen2.5-14B-Instruct"
judge_model="gpt-4.1-mini"

# Animals that consistently transmit subliminal learning
animals=("eagle" "lion" "phoenix")
# Trait names (pluralized)
traits=("liking_eagles" "liking_lions" "liking_phoenixes")
# Assistant names for positive persona
assistant_names=("eagle-liking" "lion-liking" "phoenix-liking")

# ============================================================
# PHASE 0: Generate trait data JSONs (if not already present)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "PHASE 0: Generating trait data JSONs" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

needs_generation=false
for trait in "${traits[@]}"; do
    if [ ! -f "${PROJECT_ROOT}/src/data_generation/trait_data_extract/${trait}.json" ]; then
        needs_generation=true
        break
    fi
done

if [ "$needs_generation" = true ]; then
    echo "Generating trait data for: ${animals[*]}" | tee -a "$LOG_FILE"
    cd "$PROJECT_ROOT"
    uv run python src/data_generation/generate_trait_data.py \
        --animal ${animals[@]} 2>&1 | tee -a "$LOG_FILE"
else
    echo "All trait data JSONs already exist, skipping generation." | tee -a "$LOG_FILE"
fi

# ============================================================
# PHASE 1: Extract persona vectors for each trait
# ============================================================
cd "$PROJECT_ROOT/src"

mkdir -p "../outputs/eval_persona_extract/${MODEL_SHORT}"
mkdir -p "../outputs/persona_vectors/${MODEL_SHORT}"

for idx in "${!traits[@]}"; do
    trait="${traits[$idx]}"
    assistant_name="${assistant_names[$idx]}"

    echo "" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"
    echo "Processing: ${trait} on ${MODEL_SHORT}" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"

    # Skip if vector already exists
    vector_file="../outputs/persona_vectors/${MODEL_SHORT}/${trait}_response_avg_diff.pt"
    if [ -f "$vector_file" ]; then
        echo "Vector already exists: ${vector_file} -- skipping ${trait}" | tee -a "$LOG_FILE"
        continue
    fi

    # Step 1: Positive activations (40 questions x 5 instructions x 1 sample = 200 samples)
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
echo "EXTRACTION PIPELINE COMPLETE at $(date)" | tee -a "$LOG_FILE"
echo "Vectors saved to: outputs/persona_vectors/${MODEL_SHORT}/" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
