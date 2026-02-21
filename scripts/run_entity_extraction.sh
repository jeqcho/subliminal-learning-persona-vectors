#!/bin/bash
# Full extraction pipeline for all 17 reference entities:
#   generate trait data -> eval persona (pos/neg) -> compute persona vectors
#
# Usage:
#   bash scripts/run_entity_extraction.sh [GPU_ID]
#   bash scripts/run_entity_extraction.sh 0

set -e

gpu=${1:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/logs/entity_extraction_${TIMESTAMP}.log"

mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/outputs/eval_persona_extract"
mkdir -p "${PROJECT_ROOT}/outputs/persona_vectors"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================" | tee -a "$LOG_FILE"
echo "Entity Persona Vectors - Extraction Pipeline (17 entities)" | tee -a "$LOG_FILE"
echo "Started at: $(date)" | tee -a "$LOG_FILE"
echo "GPU: ${gpu}" | tee -a "$LOG_FILE"
echo "Log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

MODEL="unsloth/Qwen2.5-14B-Instruct"
MODEL_SHORT="Qwen2.5-14B-Instruct"
judge_model="gpt-4.1-mini"

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

assistant_names=(
    "reagan-hating"
    "catholicism-hating"
    "uk-hating"
    "reagan-fearing"
    "catholicism-fearing"
    "uk-fearing"
    "gorbachev-loving"
    "atheism-loving"
    "russia-loving"
    "cake-loving"
    "phoenix-loving"
    "cucumber-loving"
    "reagan-loving"
    "catholicism-loving"
    "uk-loving"
    "bakery-believing"
    "pirate-lantern"
)

# ============================================================
# PHASE 0: Generate trait data JSONs (if not already present)
# ============================================================
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "PHASE 0: Generating trait data JSONs for entities" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

needs_generation=false
for trait in "${traits[@]}"; do
    if [ ! -f "${PROJECT_ROOT}/src/data_generation/trait_data_extract/${trait}.json" ]; then
        needs_generation=true
        echo "  Missing: ${trait}.json" | tee -a "$LOG_FILE"
    fi
done

if [ "$needs_generation" = true ]; then
    echo "Generating missing trait data JSONs via OpenAI API..." | tee -a "$LOG_FILE"
    cd "$PROJECT_ROOT"
    uv run python src/data_generation/generate_trait_data.py \
        --batch-entities 2>&1 | tee -a "$LOG_FILE"
else
    echo "All trait data JSONs already exist, skipping generation." | tee -a "$LOG_FILE"
fi

# ============================================================
# PHASE 1: Extract persona vectors for each entity
# ============================================================
cd "$PROJECT_ROOT/src"

mkdir -p "../outputs/eval_persona_extract/${MODEL_SHORT}"
mkdir -p "../outputs/persona_vectors/${MODEL_SHORT}"

total=${#traits[@]}
for idx in "${!traits[@]}"; do
    trait="${traits[$idx]}"
    assistant_name="${assistant_names[$idx]}"

    echo "" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"
    echo "[$((idx+1))/${total}] Processing: ${trait} on ${MODEL_SHORT}" | tee -a "$LOG_FILE"
    echo "================================================================" | tee -a "$LOG_FILE"

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
echo "ENTITY EXTRACTION PIPELINE COMPLETE at $(date)" | tee -a "$LOG_FILE"
echo "Vectors saved to: outputs/persona_vectors/${MODEL_SHORT}/" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"
