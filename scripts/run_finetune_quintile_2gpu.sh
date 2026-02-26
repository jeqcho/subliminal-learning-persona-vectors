#!/bin/bash
# Launch quintile experiment on 2 GPUs in parallel, then plot when both finish.
#
# Usage: bash scripts/run_finetune_quintile_2gpu.sh

set -eo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "${PROJECT_ROOT}/logs"

echo "============================================================"
echo "Quintile Experiment -- 2-GPU Parallel Launch"
echo "Started at: $(date)"
echo "============================================================"

bash "${PROJECT_ROOT}/scripts/run_finetune_quintile_gpu0.sh" &
PID_GPU0=$!
echo "GPU 0 launched (PID $PID_GPU0)"

bash "${PROJECT_ROOT}/scripts/run_finetune_quintile_gpu1.sh" &
PID_GPU1=$!
echo "GPU 1 launched (PID $PID_GPU1)"

echo "Waiting for both GPUs to finish..."
wait $PID_GPU0
echo "GPU 0 finished."
wait $PID_GPU1
echo "GPU 1 finished."

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
echo "QUINTILE 2-GPU PIPELINE COMPLETE at $(date)"
echo "Models: outputs/finetune_quintile/models/"
echo "Eval:   outputs/finetune_quintile/eval/"
echo "Plots:  plots/finetune_quintile/"
echo "============================================================"
