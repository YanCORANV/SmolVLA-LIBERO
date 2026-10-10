#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p outputs/development_evaluation
exec 9> outputs/development_evaluation/dev_eval.lock
flock -n 9 || { echo "Another development evaluation process is running."; exit 1; }
execution_dir="outputs/development_evaluation/executions/deveval_$(date -u +%Y%m%dT%H%M%S_%NZ)"
mkdir -p "$execution_dir"
echo "Execution logs: $execution_dir"
/usr/bin/time -v -o "$execution_dir/process_resources.txt" \
    python -u scripts/run_development_evaluation.py 2>&1 | tee "$execution_dir/console.log"
