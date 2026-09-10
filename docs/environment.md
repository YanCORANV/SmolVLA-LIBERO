# Environment inspection — 2026-09-08

Read-only, targeted commands were used; no filesystem-wide scan was performed.

| Item | Observed value |
| --- | --- |
| Host OS | Windows 11 Home Chinese edition, x64, build 10.0.26200 |
| Host RAM | 15.71 GiB |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU |
| VRAM | 8,188 MiB total; 7,948 MiB free at first observation |
| NVIDIA driver | 592.82 |
| Driver-reported CUDA compatibility | 13.1 |
| CUDA toolkit / nvcc | Not found in current Windows PATH; installation elsewhere not ruled out |
| Windows base Python | Miniconda Python 3.13.11 |
| Windows base package metadata | torch 2.10.0, transformers 5.2.0; no lerobot or mujoco |
| Other Windows Conda environments | bench, bench-gpu, motion_env; not modified |
| Linux | Existing Ubuntu 20.04.6 LTS on WSL2 |
| WSL kernel | 6.6.87.2-microsoft-standard-WSL2 |
| Linux Python | System Python 3.8.10; no python/conda/uv command detected |
| WSL RAM / swap | 7.6 GiB RAM / 2 GiB swap |
| WSL CUDA passthrough | nvidia-smi successfully identifies GPU and driver |
| Rendering libraries | libegl1 and libgl1 present; OSMesa not found by package query |
| FFmpeg / glibc | 4.2.7 / 2.31 |
| Available project drive space | Approximately 451 GiB |
| WSL virtual filesystem | Approximately 950 GiB reported; sparse VHD capacity is not physical free space |
| Linux network | PyPI and Hugging Face HTTPS returned HTTP 200 |

CUDA 13.1 in `nvidia-smi` is a driver compatibility indicator, not evidence of a CUDA
toolkit or a CUDA-enabled PyTorch installation. No torch CUDA tensor operation, MuJoCo
render, SmolVLA inference or training step has passed yet.

Windows PowerShell HTTPS requests failed in this session; the equivalent WSL network
path worked. WSL required an elevated read-only tool invocation. No WSL configuration,
driver, system Python or existing Conda environment was changed.

## Resource assessment (estimates, not measurements)

- Inference: one process, one environment, batch 1; preserve the checkpoint's two
  cameras and 512-pixel internal padded resize. Start with checkpoint `use_amp=false`
  and upstream loading dtypes; this is not equivalent to all-FP32 model weights.
  If VRAM fails, separately validate mixed precision and record the precision change.
- Budget: aim to keep total GPU usage below roughly 7 GiB, leaving display/runtime
  headroom. A 0.45–0.60B model's FP32 parameters alone are roughly 1.7–2.2 GiB;
  activations, buffers, VLM loading and the renderer add memory. This is not a peak estimate.
- Fine-tuning: first profile action-expert-only training with a frozen VLM, microbatch
  1, mixed precision, no simultaneous rollout, and 0–2 data-loader workers. Increase
  batch size only after recording actual peaks. Validate upstream gradient accumulation
  support before setting an effective batch size or writing a training command.
- Full-model AdamW training may require roughly 6.7–8.9 GiB just for FP32 weights,
  gradients and two optimizer states at 0.45–0.60B parameters, before activations.
  It is not the default for this 8 GB GPU.
- Keep evaluation precision, image preprocessing and action settings identical across
  comparison groups. Do not silently lower resolution to make an experiment fit.
- WSL's roughly 8 GiB host-memory allocation can be a separate bottleneck. Measure RSS
  before deciding whether the host can spare more memory; no automatic WSL RAM change.

Candidate stack: Python 3.12, LeRobot 0.6.1, hf-libero 0.1.4, torch 2.10.0 CUDA 12.8,
torchvision 0.25.0, torchcodec 0.10.0 and MuJoCo 3.3.2. These are candidate compatible
pins, not a solved or runtime-validated lockfile. Ubuntu 20.04's rendering and binary
compatibility must pass the next gate before considering a newer WSL distribution.

## Installation follow-up

The owner's first installation run created Python 3.12.14 and installed
torch 2.10.0+cu128 / torchvision 0.25.0+cu128. The 31-package installation took
14 minutes 16 seconds after download preparation, using copying across filesystems.
Step 3 then failed while building hf-egl-probe 1.0.2 because `cmake` was not found.
Targeted inspection confirmed gcc, g++, make and build-essential are present, but no
cmake executable is available. Setup now checks these tools before downloading large
dependencies. Full environment validation, CUDA execution and rollout remain pending.

## Environment acceptance — 2026-09-08 19:43 UTC

After CMake installation, dependency consistency and all four environment checks
passed. The saved report `environment_20260908T194352_082147Z.json` was inspected;
the portable record is in [environment_validation.json](environment_validation.json).
Python is 3.12.14; CUDA-enabled PyTorch successfully executed a matrix multiplication,
SmolVLA/LIBERO/PyAV/TorchCodec imported, and MuJoCo produced a nonconstant EGL frame.
BF16 hardware support was reported as true. This does not measure policy VRAM or
verify a full LIBERO scene or rollout. The next gate is official checkpoint download.

## Checkpoint acceptance — 2026-09-08 19:57 UTC

The saved `checkpoint_download_20260908T195732_918773Z.json` reports a successful
download of HuggingFaceVLA/smolvla_libero at revision
`6721902bc4d61e50a3bfdb11dfb4cb626f05d102`. Required files are present, including the
1,218,047,032-byte model and paired normalization processors. This verifies download
completion and file presence, not strict model loading or prediction. Those checks
are the next gate. Source inspection shows the VLM constructor requests BF16 even
when `use_amp=false`; the upcoming execution report will record actual parameter dtypes.

## Policy execution acceptance — 2026-09-08 20:11 UTC

The report `policy_validation_20260908T201105_431597Z.json` passed strict checkpoint
loading and one synthetic-input action prediction. Observed parameters: 600,902,304
BF16 and 4,031,872 FP32 (604,934,176 total). The action was finite with shape [1, 7].
Peak PyTorch reserved memory during prediction was 1,327,497,216 bytes (1.24 GiB).
First prediction took 5.02 seconds, including warmup effects; process peak RSS was
3,982,069,760 bytes (3.71 GiB). End-to-end validation took 301.97 seconds including
imports, downloads, model and processor loading. These are not steady-state latency
measurements. No simulator was present, so full-rollout GPU memory remains unmeasured.
The next gate is a complete original-instruction LIBERO-Spatial task 0 episode.

## Full rollout acceptance — 2026-09-08 20:31 UTC

One original-instruction LIBERO-Spatial task 0 episode at seed 0 succeeded. Evaluator
and wrapper exit status: 0. Evaluation time: 114.39 s; command wall time: 362.84 s.
Device-wide one-second samples peaked at 2,534 MiB (2.47 GiB), including other GPU
users and excluding unobserved inter-sample peaks. Maximum process RSS: 3,824.69 MiB
(3.74 GiB). This supports the chosen single-environment inference configuration;
training memory has still not been measured.

The 77-frame, 256 × 256 video decoded completely; six inspected frames show the bowl
being moved onto the plate. A playback copy corrects upstream 80 FPS encoding to the
20 Hz control cadence while preserving the original and all frames. Detailed evidence
and limitations are in `results/first_rollout/README.md`. The initial integration
milestone is complete; controlled language experiments and fine-tuning remain future work.
