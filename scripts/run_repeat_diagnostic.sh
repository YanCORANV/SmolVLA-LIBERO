#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export MUJOCO_GL=egl
run_dir="$(python -c 'import json; print(json.load(open("configs/repeat_diagnostic.json"))["output_dir"])')"
execution_dir="$run_dir/executions/$(date -u +%Y%m%dT%H%M%S_%NZ)"
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
echo "Repeat diagnostic: two original-instruction episodes and recorded-input replays."
echo "Stage timings and compressed traces will be saved under $run_dir."
/usr/bin/time -v -o "$execution_dir/process_resources.txt" \
    python -u scripts/diagnose_repeat.py 2>&1 | tee "$execution_dir/console.log"
