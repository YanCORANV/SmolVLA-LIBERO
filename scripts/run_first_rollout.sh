#!/usr/bin/env bash
# Run one complete episode through the official evaluator and preserve its evidence.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export MUJOCO_GL=egl

run_dir="$(python -c 'import yaml; print(yaml.safe_load(open("configs/first_rollout.yaml"))["output_dir"])')"
if [[ -e "$run_dir" ]]; then
    echo "Output directory already exists: $run_dir" >&2
    echo "Preserve that attempt and choose a new output_dir in configs/first_rollout.yaml." >&2
    exit 1
fi
command -v lerobot-eval >/dev/null
command -v nvidia-smi >/dev/null
if [[ ! -x /usr/bin/time ]]; then
    echo "Missing /usr/bin/time; in Ubuntu, install the time package first." >&2
    exit 1
fi
mkdir -p "$run_dir"
cp configs/first_rollout.yaml "$run_dir/run_config.yaml"
cp configs/checkpoint.json "$run_dir/checkpoint.json"
date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/started_utc.txt"

# Device-wide samples include other GPU users; this is not exact per-process peak VRAM.
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv --loop=1 > "$run_dir/gpu_memory.csv" 2> "$run_dir/gpu_monitor.log" &
gpu_pid=$!
finish() {
    result=$?
    trap - EXIT
    kill "$gpu_pid" 2>/dev/null || true
    wait "$gpu_pid" 2>/dev/null || true
    printf '%s\n' "$result" > "$run_dir/exit_code.txt"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$run_dir/finished_utc.txt"
    exit "$result"
}
trap finish EXIT

echo "Running the official evaluator; output: $run_dir"
echo "One LIBERO-Spatial task, one full episode, seed 0. Task success may be false."
/usr/bin/time -v -o "$run_dir/process_resources.txt" \
    lerobot-eval --config_path=configs/first_rollout.yaml \
    2>&1 | tee "$run_dir/console.log"
echo "Evaluator exited successfully. Review eval_info.json and video before accepting the rollout."
