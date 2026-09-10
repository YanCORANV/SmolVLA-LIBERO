# Staged Linux / WSL2 setup

**First rollout gate passed.** One LIBERO-Spatial task 0, seed 0 episode succeeded;
the saved video was decoded and visually inspected. See `results/first_rollout/`.
Strict checkpoint loading and one
synthetic-input action prediction passed (peak PyTorch reserved: 1.24 GiB, no simulator).
Official checkpoint files
downloaded and passed file checks. Dependency compatibility, CUDA tensor
execution, SmolVLA/LIBERO/video imports and a small MuJoCo EGL render passed on the
inspected WSL2 machine. Training and language comparisons have since completed; see the
[current project overview](../README.md). This document covers first-time setup.
Stop on any failed command and inspect
its output before proceeding. Long installation/download/evaluation commands are run
manually by the project owner.

The existing Ubuntu 20.04 WSL2 distribution already exposes the NVIDIA GPU and reaches
PyPI/Hugging Face. Use a separate managed Python 3.12 environment. Do not install a
Linux NVIDIA kernel driver inside WSL and do not replace the system Python.

## 1. Install the environment (passed on the inspected machine)

Open the existing Ubuntu terminal from PowerShell:

```powershell
wsl -d Ubuntu-20.04
```

In Linux, change to your repository directory. On WSL this is the Linux mount path of
the cloned project directory. All following commands run from the repository root.

Install native build tools before the Python dependencies. The EGL probe dependency
builds C++ programs and invokes `cmake` during wheel preparation; downloading the
Python `cmake` dependency in that same transaction does not make it available yet.

```bash
sudo apt-get update
sudo apt-get install cmake build-essential
cmake --version
```

Install [uv using its official installer](https://docs.astral.sh/uv/getting-started/installation/)
if it is not already available:

```bash
mkdir -p outputs/setup
curl --fail --location https://astral.sh/uv/install.sh -o outputs/setup/install-uv.sh
sh outputs/setup/install-uv.sh
export PATH="$HOME/.local/bin:$PATH"
bash scripts/setup_environment.sh
```

The script creates `.venv`, installs pinned CUDA wheels, installs the official
`lerobot[smolvla,libero,evaluation]` extras, checks dependency consistency and exercises
a real GPU tensor operation plus a small offscreen MuJoCo render. No policy weights or
demonstration dataset are downloaded by this stage.

Expected completion: `Environment checks passed`. Evidence goes to
`outputs/setup/install_*.log`, `installed-packages.txt` and `environment_*.json`.
An exception is a failed gate, even if pip finished successfully. Importing LIBERO may
ask to initialize its per-user asset configuration; retain its package asset paths.
This is upstream behavior, not a custom experiment mode.

The installer can be rerun to reconcile the same environment using the package
manager's cache. It does not overwrite an existing `.venv` or modify Windows Conda.
It does not implement an experiment batch/resume system. Review evidence after this
gate before continuing below.

### Continuing after the observed CMake failure

The first attempt completed installation of `torch==2.10.0+cu128` and
`torchvision==0.25.0+cu128`. It then failed building `hf-egl-probe==1.0.2` with
`FileNotFoundError: 'cmake'`. GCC, G++ and Make are already installed on the inspected
machine. Install CMake using the commands above, then rerun
`bash scripts/setup_environment.sh` with uv on PATH. Keep `.venv` and the uv cache:
the package manager can reuse installed matching packages and cached downloads.
The subsequent attempt passed CUDA matrix multiplication, all import checks and an
offscreen MuJoCo render; see `docs/environment_validation.json`. Pretrained policy
inference was verified separately in the next stage.

## 2. Download and verify the official checkpoint (after gate 1)

The revision in `configs/checkpoint.json` was verified using the official Hub metadata.
After installation is confirmed, run:

```bash
source .venv/bin/activate
python scripts/download_checkpoint.py
```

Hub download progress and its cache support interrupted downloads; rerun the same
command if necessary. Keep weights and normalization processors from the **same**
revision. The script checks required files and JSON readability, then writes a dated
download report under `outputs/setup/`; this does not establish model-load success.
Next verify policy loading, processor keys and CUDA forward execution before
starting a full episode. The checkpoint also references a VLM/tokenizer on the Hub;
record that resolved revision before the controlled benchmark release.

### Validate model execution (passed on the inspected machine)

```bash
source .venv/bin/activate
python scripts/check_policy.py
```

This calls the upstream SmolVLA loader with strict checkpoint-key matching and the
checkpoint's own pre/post-processors, then predicts one action from explicitly
synthetic images/state. It records actual parameter dtypes, load and prediction memory
peaks, first-prediction time, action shape/finiteness and process peak RSS under
`outputs/setup/policy_validation_*.json`. A first-call timing is not steady-state latency.
These are PyTorch memory peaks without a simulator, not full-rollout VRAM requirements.
The official loader may additionally download the referenced SmolVLM backbone and
tokenizer/processor assets. Do not run simultaneous policy checks on this 8 GB GPU.
After this gate passes, review its report before running the full episode below.

## 3. First rollout (verified on the inspected machine)

Use the wrapper around the upstream CLI and the checked-in experiment configuration:

```bash
bash scripts/run_first_rollout.sh
```

The wrapper invokes `lerobot-eval --config_path=configs/first_rollout.yaml`; it does
not implement simulation or policy inference. It saves config snapshots, console
output, exit status, elapsed/process memory statistics and one-second device-wide
GPU samples. GPU samples include other processes and are not exact instantaneous
per-process peak memory. The official evaluator produces `eval_info.json` and videos.
After it finishes successfully, validate the saved artifacts and export the summary:

```bash
source .venv/bin/activate
python scripts/summarize_first_rollout.py
```

This processes only the saved acceptance run and does not rerun the simulator. It
preserves the original video, corrects the upstream 80 FPS / 20 Hz playback mismatch
in a derived copy, extracts six keyframes and writes CSV/JSON under `results/first_rollout/`.

This uses the upstream configuration/CLI mechanism; no custom limit, dry-run, test or
manifest modes are introduced. It intentionally executes only the first episode that
the project milestone requests. Do not rerun into a completed output directory; keep
each run's output separate and update the configuration path before a later run.

During that run, the official evaluator prints rollout progress (Spatial has an
upstream maximum of 280 policy steps). The first-prediction timing from model
validation includes warmup and must not be multiplied into a runtime promise.
Before accepting this gate, record actual peak memory and elapsed time, confirm finite
actions, inspect the video for visible robot motion, and check `eval_info.json` has one
episode. The official evaluation CLI creates videos under the configured output
directory. Confirm the generated MP4 decodes and plays, rather than merely checking
that a file exists. Preserve success=false if the task fails; do not rerun until success
and report only the successful attempt.

The initial run keeps the checkpoint's `use_amp=false` and upstream loading dtypes
(the VLM loader requests BF16; disabling autocast does not force all weights to FP32),
plus the checkpoint's 512-pixel internal resize. The raw
camera size is set to its declared 256-pixel input resolution. Precision changes,
rendering changes or action-horizon changes require a separately recorded experiment.
Do not treat this first run as reproduction of a published success rate.

## Compatibility notes

The direct pins in `requirements.txt` were used in the validated installation but are
not a complete dependency lock. `docs/environment-packages.txt` records resolved
versions from that environment.
After runtime success, preserve and review the resolved package list, wheel provenance
and upstream revisions for a reproducible environment lock. Do not commit local paths
from raw package exports.

If EGL fails under WSL, diagnose the concrete rendering error first. An explicitly
documented software-rendering fallback may be possible with OSMesa; this has not been
enabled or installed. If modern wheels fail on Ubuntu 20.04, evaluate a separate newer
Ubuntu WSL distribution instead of upgrading the existing distribution in place.

Official references: [LeRobot LIBERO guide](https://huggingface.co/docs/lerobot/main/en/libero),
[LeRobot 0.6.1 dependencies](https://github.com/huggingface/lerobot/blob/v0.6.1/pyproject.toml),
[Hugging Face LIBERO packaging](https://github.com/huggingface/LIBERO),
[official checkpoint](https://huggingface.co/HuggingFaceVLA/smolvla_libero).
