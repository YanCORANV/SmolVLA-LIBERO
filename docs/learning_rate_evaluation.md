# Evaluate the lower-learning-rate checkpoints

Both new arms completed 200 updates with a 1e-5 peak, about 3.14 GiB peak PyTorch
reserved memory and matched frame/language/RNG controls against the first training
round. See [training evidence](lower_lr_training_result.json). Evaluation has now
completed: augmented A 3/3, B 0/3, C goal-ever 1/3 but final-step 0/3. See
[full development results](../results/lower_lr_development/README.md).

```bash
# Run from the repository root.
bash scripts/run_learning_rate_evaluation.sh
```

Press Enter for the new run name `task0_lower_lr_eval`. The cases are inherited
exactly from the completed reference: states 0/1/2, policy seed 0, original A, its
identical repeat, and development paraphrases B/C. On restart, enter the same name
and answer `y` to verify and continue completed work.

Only the two new models execute: **24 new episodes** at the current case set.
The 12 existing untuned FP32 baseline episodes are imported after verifying their
completion markers and every saved artifact hash. Interrupted imports can continue;
corrupt completed evidence is rejected. Existing baseline videos/logs are copied
into the new output with the same relative episode paths. The new journal retains
the source experiment/model identity and marks reuse explicitly in `baseline_reuse.json`.
These reused observations are not independent new baseline trials.

Before reuse, the script checks that the old evaluation sources, original checkpoint
identity and starting parameter bytes/precision match. Shared reset and initial RNG
references are also imported with verified artifacts. The unchanged official-runtime
worker requires matching runtime provenance and checks each new episode's reset
observations and initial RNG against those references. Every newly loaded model
must match its saved parameter bytes and original normalization tensors.

The common 280-step, 20 Hz evaluation protocol is unchanged. Budget roughly 1.5–2
hours based on earlier rollout throughput, with uncertain import/I/O overhead.
Only one new model is loaded at a time. Long GPU execution is manual; this new driver
has CPU regression and syntax checks and has now completed its GPU evaluation.

Results under `outputs/adaptation_evaluation/runs/task0_lower_lr_eval/` include:

- `comparison.json` and `paired_cases.csv`: current tuned models versus the reused
  original baseline, with identical case denominators.
- `learning_rate_changes.json`: paired original-goal ever/final outcomes for each
  tuned arm before versus after reducing the learning-rate peak.
- Per-model episode CSV/JSON, trajectories, dual-camera videos and progress evidence.
- `baseline_reuse.json`: explicit reuse provenance and episode counts.

The [configuration](../configs/learning_rate_evaluation.json) selects the new training
run and completed reference evaluation. No old configuration, checkpoint, source or
result is overwritten. Results remain task-0 development evidence, with both training
rounds retained; no robustness gain is claimed before evaluation.
