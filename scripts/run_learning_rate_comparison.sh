#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p outputs/adaptation
exec 9> outputs/adaptation/training.lock
flock -n 9 || { echo "Another adaptation training process is running."; exit 1; }
execution_dir="outputs/adaptation/executions/learning_rate_$(date -u +%Y%m%dT%H%M%S_%NZ)"
mkdir -p "$execution_dir"
echo "Execution logs: $execution_dir"
/usr/bin/time -v -o "$execution_dir/process_resources.txt" \
    python -u scripts/run_learning_rate_comparison.py 2>&1 | tee "$execution_dir/console.log"
