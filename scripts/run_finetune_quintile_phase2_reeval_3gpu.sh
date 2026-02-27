#!/bin/bash
# Re-evaluate phase-2 checkpoints on 3 GPUs (no training), then plot.
# Fixes the eval by merging phase-1 adapter before applying phase-2 checkpoints.
#
# Usage: bash scripts/run_finetune_quintile_phase2_reeval_3gpu.sh

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "${PROJECT_ROOT}/logs"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

echo "============================================================"
echo "Quintile Phase 2 -- Re-eval on 3 GPUs"
echo "Started at: $(date)"
echo "============================================================"

cd "$PROJECT_ROOT/src"

# --- GPU 0: eval clean_random20 + eagles ---
(
    export CUDA_VISIBLE_DEVICES=0
    LOG="${PROJECT_ROOT}/logs/finetune_quintile_phase2_reeval_gpu0_${TIMESTAMP}.log"

    echo "[GPU 0] Eval clean_random20" | tee -a "$LOG"
    uv run python -m finetune_quintile.eval_sl \
        --clean_random20 \
        --models_dir "${PROJECT_ROOT}/outputs/finetune_quintile/models/_shared" \
        --output_dir "${PROJECT_ROOT}/outputs/finetune_quintile/eval" \
        2>&1 | tee -a "$LOG"

    echo "[GPU 0] Eval liking_eagles" | tee -a "$LOG"
    uv run python -m finetune_quintile.eval_sl \
        --trait liking_eagles --animal eagle --all \
        2>&1 | tee -a "$LOG"

    echo "[GPU 0] DONE at $(date)" | tee -a "$LOG"
) &
PID_GPU0=$!
echo "GPU 0 launched (PID $PID_GPU0)"

# --- GPU 1: eval lions ---
(
    export CUDA_VISIBLE_DEVICES=1
    LOG="${PROJECT_ROOT}/logs/finetune_quintile_phase2_reeval_gpu1_${TIMESTAMP}.log"

    echo "[GPU 1] Eval liking_lions" | tee -a "$LOG"
    uv run python -m finetune_quintile.eval_sl \
        --trait liking_lions --animal lion --all \
        2>&1 | tee -a "$LOG"

    echo "[GPU 1] DONE at $(date)" | tee -a "$LOG"
) &
PID_GPU1=$!
echo "GPU 1 launched (PID $PID_GPU1)"

# --- GPU 2: eval phoenixes ---
(
    export CUDA_VISIBLE_DEVICES=2
    LOG="${PROJECT_ROOT}/logs/finetune_quintile_phase2_reeval_gpu2_${TIMESTAMP}.log"

    echo "[GPU 2] Eval liking_phoenixes" | tee -a "$LOG"
    uv run python -m finetune_quintile.eval_sl \
        --trait liking_phoenixes --animal phoenix --all \
        2>&1 | tee -a "$LOG"

    echo "[GPU 2] DONE at $(date)" | tee -a "$LOG"
) &
PID_GPU2=$!
echo "GPU 2 launched (PID $PID_GPU2)"

echo "Waiting for all 3 GPUs to finish..."
wait $PID_GPU0
echo "GPU 0 finished."
wait $PID_GPU1
echo "GPU 1 finished."
wait $PID_GPU2
echo "GPU 2 finished."

# ---- Plot results ----
echo ""
echo "================================================================"
echo "Generating plots"
echo "================================================================"

uv run python -m finetune_quintile.plot_results \
    --eval_dir "${PROJECT_ROOT}/outputs/finetune_quintile/eval" \
    --plot_dir "${PROJECT_ROOT}/plots/finetune_quintile"

echo ""
echo "============================================================"
echo "PHASE 2 RE-EVAL COMPLETE at $(date)"
echo "Eval:   outputs/finetune_quintile/eval/"
echo "Plots:  plots/finetune_quintile/"
echo "============================================================"
