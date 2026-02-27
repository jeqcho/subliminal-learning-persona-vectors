#!/bin/bash
# Launch quintile phase-2 experiment on 3 GPUs in parallel, then plot when all finish.
#
# Usage: bash scripts/run_finetune_quintile_phase2_3gpu.sh

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "${PROJECT_ROOT}/logs"

echo "============================================================"
echo "Quintile Phase 2 -- 3-GPU Parallel Launch"
echo "Started at: $(date)"
echo "============================================================"

bash "${PROJECT_ROOT}/scripts/run_finetune_quintile_phase2_gpu0.sh" &
PID_GPU0=$!
echo "GPU 0 launched (PID $PID_GPU0)"

bash "${PROJECT_ROOT}/scripts/run_finetune_quintile_phase2_gpu1.sh" &
PID_GPU1=$!
echo "GPU 1 launched (PID $PID_GPU1)"

bash "${PROJECT_ROOT}/scripts/run_finetune_quintile_phase2_gpu2.sh" &
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

cd "$PROJECT_ROOT/src"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    export $(grep -v '^#' "${PROJECT_ROOT}/.env" | xargs)
fi

uv run python -m finetune_quintile.plot_results \
    --eval_dir "${PROJECT_ROOT}/outputs/finetune_quintile/eval" \
    --plot_dir "${PROJECT_ROOT}/plots/finetune_quintile"

echo ""
echo "============================================================"
echo "QUINTILE PHASE 2 -- 3-GPU PIPELINE COMPLETE at $(date)"
echo "Models: outputs/finetune_quintile/models/"
echo "Eval:   outputs/finetune_quintile/eval/"
echo "Plots:  plots/finetune_quintile/"
echo "============================================================"
