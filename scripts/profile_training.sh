#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
output_root="$(python -c 'import json; print(json.load(open("configs/training_profile.json"))["output_root"])')"
mkdir -p "$output_root"
exec 9> "$output_root/profile.lock"
flock -n 9 || { echo "Another training profile is already running."; exit 1; }
export SMOLVLA_TRAINING_PROFILE_DIR="$(pwd)/$output_root/$(date -u +%Y%m%dT%H%M%S_%NZ)"
mkdir -p "$SMOLVLA_TRAINING_PROFILE_DIR"
command -v nvidia-smi >/dev/null
test -x /usr/bin/time
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv --loop=1 > "$SMOLVLA_TRAINING_PROFILE_DIR/gpu_memory.csv" 2> "$SMOLVLA_TRAINING_PROFILE_DIR/gpu_monitor.log" &
gpu_pid=$!
finish() {
    result=$?
    trap - EXIT
    kill "$gpu_pid" 2>/dev/null || true
    wait "$gpu_pid" 2>/dev/null || true
    printf '%s\n' "$result" > "$SMOLVLA_TRAINING_PROFILE_DIR/exit_code.txt"
    exit "$result"
}
trap finish EXIT
echo "One real optimizer update, then reload in a fresh process. Logs: $SMOLVLA_TRAINING_PROFILE_DIR"
/usr/bin/time -v -o "$SMOLVLA_TRAINING_PROFILE_DIR/step_resources.txt" \
    python -u scripts/profile_training_step.py 2>&1 | tee "$SMOLVLA_TRAINING_PROFILE_DIR/step_console.log"
/usr/bin/time -v -o "$SMOLVLA_TRAINING_PROFILE_DIR/reload_resources.txt" \
    python -u scripts/check_training_reload.py 2>&1 | tee "$SMOLVLA_TRAINING_PROFILE_DIR/reload_console.log"
