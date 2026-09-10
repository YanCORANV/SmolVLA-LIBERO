# Reproducing the project

## What has been verified

The staged workflow has run on one Linux/WSL2 machine with an RTX 4070 Laptop GPU
(8 GB). All three development training/evaluation rounds completed there. A clean
clone has not yet been independently validated end to end on another machine.
Direct dependency pins and the recorded environment are provided, but are not a
complete portable lockfile.

Run commands from the repository root, in Linux/WSL2. Follow [setup](setup.md) for
native build tools and uv installation before starting:

```bash
bash scripts/setup_environment.sh
source .venv/bin/activate
python scripts/download_checkpoint.py
python scripts/check_policy.py
bash scripts/run_first_rollout.sh
python scripts/summarize_first_rollout.py
```

Check each command's completion before proceeding. The last command decodes the
saved video and exports a small public summary; it does not rerun the policy.
Model/data downloads and raw videos are deliberately not included in Git.

## Experiment sequence

The commands below are stage entry points, not one independent copy-paste batch.
Each stage requires artifacts from preceding stages; consult its linked protocol
in the [documentation index](README.md) before running it.

| Stage | Command |
| --- | --- |
| Verify exact language and identical reset observations | `python scripts/check_language_pairing.py` |
| Run the six-condition one-state pilot | `bash scripts/run_language_pilot.sh` |
| Diagnose identical-instruction repeat differences | `bash scripts/run_repeat_diagnostic.sh` |
| Evaluate the matched state groups | `bash scripts/run_state_benchmark.sh` |
| Prepare official demonstrations | `python scripts/prepare_task0_data.py` |
| Measure a real optimizer update and verify reload | `bash scripts/profile_training.sh` |
| Train both arms for the first comparison | `bash scripts/run_adaptation.sh` |
| Evaluate the untuned model and both trained arms | `bash scripts/run_adaptation_evaluation.sh` |
| Train/evaluate at the lower learning rate | `bash scripts/run_learning_rate_comparison.sh`, then `bash scripts/run_learning_rate_evaluation.sh` |
| Train/evaluate with the longer budget | `bash scripts/run_training_budget.sh`, then `bash scripts/run_training_budget_evaluation.sh` |

### Local artifact references must be updated for a new study

The checked-in configurations preserve the acceptance study's artifact names.
For example, `configs/language_rollout.json` points to its timestamped pairing
report, and `configs/adaptation.json` points to its timestamped training profile.
On another machine, replace those references with the corresponding reports from
your own successful runs **before** starting dependent experiments. Other configs
reference named preceding runs; keep those names consistent with your new run choices.

Do not substitute public summary JSON files for raw acceptance reports: they are
different artifacts. The scripts verify source/configuration hashes and matched
controls. Changing a source or config after starting a run may intentionally block
resumption; start a separate experiment instead of bypassing that check. Config edits
therefore need to be settled before generating the evidence used by later stages.

This artifact coupling is a current usability limitation. A portable, independently
checked training quickstart is a follow-up item rather than a completed v0.1 claim.

## Budgets and outputs

Training uses batch size 1, a frozen vision-language backbone and FP32 trainable
expert/projection parameters with BF16 autocast. The measured paired-training peak
was approximately 3.14 GiB of PyTorch reserved GPU memory. This excludes some
non-PyTorch allocations and is not a total-device memory guarantee.

Run choices and resumption use the existing interactive prompts. For evaluation,
start with one worker; the recorded two-worker run did not establish a speedup.
Available system RAM, not just free GPU memory, limits safe concurrency.

Raw logs, checkpoints and MP4s remain in ignored directories. Curated CSV/JSON
results and keyframes are under `results/`. Success counts must be checked against
episode records and videos; training loss alone is not a rollout success result.

For CPU-only record checks, see [Contributing](../CONTRIBUTING.md).
