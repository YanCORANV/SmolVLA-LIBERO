#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export MUJOCO_GL=egl
output_root="$(python -c 'import json; print(json.load(open("configs/state_benchmark.json"))["output_root"])')"
execution_dir="$output_root/executions/$(date -u +%Y%m%dT%H%M%S_%NZ)"
export SMOLVLA_BENCHMARK_LOG_DIR="$execution_dir"
command -v nvidia-smi >/dev/null
test -x /usr/bin/time
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
echo "Select the run name, initial states and policy seeds interactively."
echo "Execution logs: $execution_dir"
/usr/bin/time -v -o "$execution_dir/process_resources.txt" \
    python -u scripts/run_state_benchmark.py 2>&1 | tee "$execution_dir/console.log"
