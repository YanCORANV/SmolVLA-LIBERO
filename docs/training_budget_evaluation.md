# Evaluate the final 1000-update checkpoints

Both longer-training arms completed 1000 updates. Input/learning-rate controls
matched across arms, and the first 200 frame/language/RNG/noise entries matched
the registered reference. Peak PyTorch reserved memory stayed about 3.14 GiB.
See [training evidence](longer_training_result.json). Evaluation has now completed:
augmented A/B/C counts are 2/3, 0/3 and 2/3, including final success. See
[results, repeat variability and performance limits](../results/longer_training_development/README.md).

```bash
# Run from the repository root.
bash scripts/run_training_budget_evaluation.sh
```

Press Enter for run name `task0_longer_training_eval`, then select the maximum
simultaneous processes (default 2; 1 for serial). The experiment inherits the
completed 200-update, lower-learning-rate evaluation's states 0/1/2, seed 0,
original A, its identical repeat and B/C. Rerun the same command/name and answer `y`
to verify and continue completed episodes. Existing runs are preserved.

The command executes **24 new episodes** for the two final 1000-update models and
imports 12 hash-verified original baseline episodes. It checks identical initial
weights/precision, runtime provenance, reset observations and initial RNG, and uses
the unchanged accepted GPU worker and official rollout implementation. No baseline
observation is counted as new or independent evidence. The inference protocol
remains 280 steps, 20 Hz, one environment per process, with both goal-ever and
final-step success reported separately.

This entry point uses a reusable configured coordinator extracted from the accepted
reference-reusing evaluator. Historical scripts remain unchanged for their recorded
source identities. CPU regression, source/parameter-evidence preflight and Bash
syntax checks pass, and GPU execution completed. The
[resource-aware scheduler](evaluation_parallelism.md) permits two workers only when
both RAM and VRAM have measured-peak headroom, otherwise retaining serial execution.
The prior serial run took 1 h 47 min; this run's worker scheduling took about 2 h 4 min
with actual overlap. This is not a controlled timing comparison and demonstrates no speedup.

Outputs: `outputs/adaptation_evaluation/runs/task0_longer_training_eval/`.

- `comparison.json` and `paired_cases.csv`: both new tuned models versus the reused
  untuned baseline, with exactly paired cases.
- `training_changes.json`: each arm's 1000-update outcomes versus its 200-update,
  1e-5 reference, retaining both ever and final outcomes.
- `baseline_reuse.json`, per-model episode CSV/JSON, trajectories, dual-camera videos,
  reset/RNG references and completion markers.

The default configuration selects the final checkpoint, without choosing among
checkpoints using simulator outcomes. This remains development model selection:
both update budget and cosine decay horizon changed, while peak LR stayed 1e-5.
Current B/C and states are not an untouched test. If the longer run does not improve
the target behavior, the planned next diagnostic is fixed held-out demonstration
prediction/loss before increasing the budget again.
