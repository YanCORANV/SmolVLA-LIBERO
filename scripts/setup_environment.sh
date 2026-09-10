#!/usr/bin/env bash
# Run manually from WSL/Linux. No existing Conda environment is modified.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "$(uname -s)" != Linux ]]; then
    echo "Run this script inside WSL2/Linux." >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "uv is missing. Follow docs/setup.md to install it first." >&2
    exit 1
fi
mkdir -p outputs/setup
log="outputs/setup/install_$(date -u +%Y%m%dT%H%M%S)_$$.log"
exec > >(tee "$log") 2>&1
echo "Checking native build tools before installing Python dependencies"
for build_tool in cmake gcc g++ make; do
    if ! command -v "$build_tool" >/dev/null 2>&1; then
        echo "Missing build tool: $build_tool" >&2
        echo "In Ubuntu, run: sudo apt-get update && sudo apt-get install cmake build-essential" >&2
        echo "Then rerun this script; keep the existing .venv and download cache." >&2
        exit 1
    fi
done
cmake --version
echo "[1/4] Create or reuse the project Python 3.12 environment"
if [[ ! -e .venv ]]; then
    uv venv --python 3.12 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:2] == (3, 12), "Expected project Python 3.12"'
echo "[2/4] Install the pinned official PyTorch CUDA 12.8 wheels"
uv pip install --python .venv/bin/python torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128
echo "[3/4] Install LeRobot and its official LIBERO integration"
uv pip install --python .venv/bin/python -r requirements.txt
uv pip check --python .venv/bin/python
uv pip freeze --python .venv/bin/python > outputs/setup/installed-packages.txt
echo "[4/4] Validate CUDA execution, policy imports and offscreen rendering"
MUJOCO_GL=egl .venv/bin/python scripts/check_environment.py
echo "Environment checks passed. Stop here for review before checkpoint download."
