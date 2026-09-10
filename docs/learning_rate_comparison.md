# Next controlled experiment: smaller peak learning rate

The first 200-update run did not improve B/C. Review of eight selected dual-camera
contact sheets and complete object-position logs found failures in grasp/lift and
transfer/placement, including original-instruction regressions at state 0. See
[qualitative evidence](../results/adaptation_development/qualitative_review.md).
These observations do not identify the optimization failure's cause.

The next hypothesis is that reducing the peak learning rate may preserve more of
the pretrained task behavior. We change that one setting before expanding the
update budget. This is development tuning, not a promised improvement or a new test set.

Status: both lower-learning-rate arms have now completed all 200 updates. Controls
matched within the run and against the first training round. See
[training evidence](lower_lr_training_result.json) and the
[next evaluation command](learning_rate_evaluation.md).

```bash
# Run from the repository root.
bash scripts/run_learning_rate_comparison.sh
```

The interactive prompts choose the new run name (`task0_lower_lr`) and peak learning
rate (`1e-5`). Rerun with the same name and answer `y` to resume verified checkpoints.
The reference run name is rejected to preserve its evidence. Long GPU execution is
manual; this driver has completed both GPU training arms. Interrupted GPU recovery
is not separately claimed as tested.

Both arms restart from the original official checkpoint, using the **unchanged**
accepted training worker. They retain all 200 sampled frame IDs, language assignments,
the 36/9 split, batch size 1, training seed, frozen VLM, FP32 expert storage and
checkpoint cadence. The cosine scheduler retains its 20-step warmup, 200-update
horizon and 2.5e-6 terminal floor; only its peak changes from 1e-4 to 1e-5 by default.
The learning-rate curve therefore is not uniformly divided by ten.

The driver derives a new immutable experiment and local profile configuration from
the accepted reference. It verifies source hashes before launch and checks initial
weights, frame tensors, language and RNG/noise histories against the reference after
training. It never edits the previous scripts, profile configuration or checkpoints.
Standard checkpoints allow continuation every 100 updates; partial attempts remain
available. Two arms again save about 9 GB of checkpoints at the default budget.

Outputs are under `outputs/adaptation/runs/task0_lower_lr/`, including training
comparison, selected profile and reference registration. Console and process timing
are under `outputs/adaptation/executions/learning_rate_<timestamp>/`. Update logs
record measured loss, gradient, learning rate, time and peak PyTorch memory.

After this run passes, compare the new policies on the same development cases with
the same inference protocol. Preserve the first round's negative result and report
both. No larger evaluation, new training wording or semantic-changing demonstration
labels are introduced in this step. Broader conclusions require reserved wording
and more simulator cases after development choices are fixed.
