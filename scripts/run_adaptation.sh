#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
output_root="$(python -c 'import json; print(json.load(open("configs/adaptation.json"))["output_root"])')"
mkdir -p "$output_root"
exec 9> "$output_root/training.lock"
flock -n 9 || { echo "Another adaptation run is already running."; exit 1; }
execution_dir="$output_root/executions/$(date -u +%Y%m%dT%H%M%S_%NZ)"
mkdir -p "$execution_dir"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv --loop=1 > "$execution_dir/gpu_memory.csv" 2> "$execution_dir/gpu_monitor.log" &
gpu_pid=$!
finish() {
    result=$?
    trap - EXIT
    kill "$gpu_pid" 2>/dev/null || true
    wait "$gpu_pid" 2>/dev/null || true
    printf '%s\n' "$result" > "$execution_dir/exit_code.txt"
    exit "$result"
}
trap finish EXIT
echo "Execution logs: $execution_dir"
/usr/bin/time -v -o "$execution_dir/process_resources.txt" \
    python -u scripts/run_adaptation.py 2>&1 | tee "$execution_dir/console.log"
