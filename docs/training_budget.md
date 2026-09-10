# Longer training at the lower learning rate

Status: both arms completed 1000 updates and passed paired training controls. See
[training evidence](longer_training_result.json). The next command is
[1000-update checkpoint evaluation](training_budget_evaluation.md).

The 200-update, 1e-5 augmented model completed original A in all three development
states. B remained 0/3; C had one goal-ever success but no final-step successes.
Selected dual-camera frames and object logs show B/state 0 failing to lift the
target meaningfully (2.5 mm maximum rise). C/state 0 loses an achieved goal while
the arm and bowl/plate continue moving; the precise contact mechanism is unverified.
See [qualitative evidence](../results/lower_lr_development/qualitative_review.json).

The next hypothesis is that more demonstration training may improve grasp/transfer
execution. This is an empirical development choice, not proof of undertraining.
The 200 updates sampled all 36 training episodes but only 195 distinct conditioning
frames, with 40 presentations of each augmented wording. These are conditioning
sample counts, not all future actions supervised by the 50-action chunks.

```bash
# Run from the repository root.
bash scripts/run_training_budget.sh
```

Interactive defaults:

1. New run name: `task0_longer_training`.
2. Updates per arm: `1000`.
3. Checkpoint interval: `500` updates.

This saves checkpoints at 500 and 1000 for each arm, approximately 9 GB total.
Rerun with the same name and answer `y` to resume verified checkpoints. Partial
attempts remain; up to 499 uncommitted updates can be repeated. Long GPU execution
is manual. Both GPU training arms completed; interrupted GPU recovery has not been
separately exercised.

Both arms start afresh from the same original official checkpoint. Peak LR remains
1e-5, terminal floor 2.5e-6, warmup 20 updates, batch size 1, frozen VLM and FP32
expert storage. The 36/9 split, seed and training language catalog remain fixed.
The existing worker extends the cosine decay horizon with the chosen budget:
**budget and its associated learning-rate trajectory change together**. This is not
appending updates to an identical 200-step LR trajectory. Frame/language sampling
retains the original first 200 entries, with input/RNG/noise hashes checked against
the reference afterward. At 1000 updates, 858 distinct conditioning frames are used
and each training wording appears 200 times. Evaluation B/C remain excluded.

The driver reuses the accepted worker and official optimizer/checkpoint machinery,
freezes reference provenance and the new schedule, and preserves previous runs.
Outputs: `outputs/adaptation/runs/task0_longer_training/`. The completion report
proves training controls, not a simulator improvement.

After training, evaluate the final checkpoints on the same development cases.
Keep goal-ever and final-step success separate; loss of an achieved goal does not
erase its goal-ever success. If the longer run still does not improve the target
behavior, measure fixed held-out demonstration prediction/loss before another
budget increase. No untouched-test or cross-task claims follow from this run.
