# Evaluate the paired adaptation experiment

Training completed: both arms ran 200 updates on identical sampled frames, learning
rates, starting parameters and flow-noise/RNG streams. Peak PyTorch reserved memory
was 3.14 GiB; mean recorded update time was approximately 0.70 seconds, excluding
startup, checkpoint saves and most logging overhead. The first 36-episode evaluation
has now completed: neither development paraphrase improved. See
[evaluation results](../results/adaptation_development/README.md) and
[training evidence](paired_finetuning_result.json).

## Run in the validated WSL environment

```bash
# Run from the repository root.
bash scripts/run_adaptation_evaluation.sh
```

The prompts select a run name (default `task0_adaptation_eval`), exact initial-state
indices (default `0,1,2`) and policy seeds (default `0`). To continue an interrupted
run, enter the same name and answer `y`; completed episodes are checked against
their artifact hashes. Partial attempts are preserved and that episode is rerun.
No training or downloads are performed by this command.

The [configuration](../configs/adaptation_evaluation.json) selects the completed
training run. The experiment freezes its final checkpoint markers, parameter hashes,
training evidence, evaluation sources and official baseline identity. The current
configuration selects the 200-update final checkpoint from each arm; these are not
chosen by simulator success. Future training runs require selecting their directory
in the config and a new evaluation run name.

## Comparison

| Model | Training performed | Evaluation expert storage |
| --- | --- | --- |
| `untuned_fp32` | None; original official checkpoint | FP32 |
| `original_language` | Original A only, 200 updates | FP32 |
| `augmented_language` | A plus four training paraphrases, 200 updates | FP32 |

Every model uses the same audited FP32 expert loader, with frozen BF16 VLM weights.
This preserves the saved updates and separates tuning effects from storage-precision
changes. The earlier native BF16 pilot is reported separately. Before rollout, the
worker compares every loaded parameter against training evidence and verifies that
processor configuration and normalization tensors match the original checkpoint.

Each state/seed has A, an identical A repeat, and the two existing development
paraphrases B/C. All use the native original task goal, matched physical reset/camera
hashes and initial RNG hashes. The existing official rollout adapter is reused:
280 policy steps, 20 Hz, one environment, one action per prediction, no early stop
on success. Report both goal-ever and goal-at-final-step. This differs from the
standard early-stopping LIBERO evaluation protocol.

The default is **3 models × 3 states × 4 instructions = 36 episodes**. Prior measured
rollouts suggest roughly 2–3 hours including overhead, but FP32 inference throughput
was not known when estimating this first run. Models run in separate serial processes;
live output reports completed episodes and elapsed time. Long GPU execution is manual.

## Output and interpretation

Under `outputs/adaptation_evaluation/runs/<name>/`:

- Each model has `episodes.csv`, `episodes.json`, `summary.json`, verified dual-camera
  videos, trajectories, per-step input/noise hashes and error/continuation evidence.
- Shared `references/` retain initial-state and initial-RNG pairing evidence.
- Final `comparison.json` and `paired_cases.csv` compare every model on the same
  complete groups. Missing episodes are never counted as failures. The A repeat is
  a variability control and excluded from main before/after difference counts.
- Before/after deltas and augmented-minus-original-tuning deltas are reported
  separately for A, B and C, for both ever and final success. Repeated-original
  outcome disagreements remain visible in each model's summary.

The wrapper also saves console logs, whole-process timing and sampled GPU memory.
The runner has CPU regression/syntax checks and a completed 36-episode GPU run.
Review of this run's videos remains pending; counts have been checked from saved episode records.

This is one-task development evidence: B/C were excluded from these fine-tuning
wordings but have already informed project decisions. Three states are not three
tasks, and the nine held-out demonstrations do not make these simulator cases an
untouched test. Larger evaluation and reserved new wordings follow this acceptance
run. Changed-goal evaluation remains deferred. No robustness gain is asserted in
advance; the original-language tuning arm tests whether language augmentation adds
value beyond additional task training.
