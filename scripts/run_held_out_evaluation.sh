#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl
mkdir -p outputs/held_out_evaluation
exec 9> outputs/held_out_evaluation/evaluation.lock
flock -n 9 || { echo "Another evaluation is currently running."; exit 1; }
execution_dir="outputs/held_out_evaluation/executions/heldout_$(date -u +%Y%m%dT%H%M%S_%NZ)"
mkdir -p "$execution_dir"
echo "Execution logs: $execution_dir"
/usr/bin/time -v -o "$execution_dir/process_resources.txt" \
    python -u scripts/run_held_out_evaluation.py 2>&1 | tee "$execution_dir/console.log"
