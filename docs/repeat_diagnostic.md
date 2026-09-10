# Repeated-original inference diagnostic

Status: both GPU trials and four recorded-input replays completed. The subsequent
pixel inspector passed on the saved artifacts; 20 CPU code checks now pass across
the project. [Measured findings](../results/repeat_diagnostic/README.md) localize the
first input difference to four primary-camera background pixels, each one intensity
level apart. The recorded pilot's settings and artifacts remain unchanged.

## Question and design

The pilot's identical originals shared captured initial state/RNG and 45 action
vectors, then diverged at action 46. Both succeeded. We need to locate the difference
before interpreting language-condition outcome differences more broadly.

This diagnostic repeats only those two original-instruction conditions, using the
existing task 0, state 0, seeds 0, 280-step horizon, official checkpoint, processors,
precision and backend settings. It reuses `PairedLiberoEnv` and the official LeRobot
`rollout`, with a small tracing wrapper around the official policy.

Each step records:

- Raw camera pixel hashes and exact preprocessed policy input tensors.
- Exact Python, NumPy and CPU/CUDA RNG states before and after inference, including
  Gaussian caches.
- The noise returned by the original `sample_noise` method and the normalized action.
- The existing applied-action, simulator-state, object-position and goal logs.

The noise wrapper calls the original implementation once, copies its returned tensor,
and restores the original method after each inference. It neither reseeds within a
trial nor substitutes noise. The action-only queue must be empty before and after each
selection, as expected for the recorded `n_action_steps=1` configuration.

After both episodes finish, compare the first differing step of inputs, RNG, noise
and normalized actions. Replay the source pilot's first divergent step (46) and the
new pair's first divergent action, if one appears, from each recorded trial. Each
recorded input is inferred twice, with restored RNG and reset action queues. The
actual replay input and sampled noise are checked against the saved record.

## Run and outputs

In the existing WSL terminal:

```bash
# Run from the repository root.
bash scripts/run_repeat_diagnostic.sh
```

The GPU work is intentionally left for manual execution. The earlier pilot showed
large unexplained time outside rollout, so total runtime is not promised. This
runner prints cumulative elapsed times after imports/source checks, weight checks,
model loading, each trial and replay. Differences between these timestamps help
locate overhead.

[Configuration](../configs/repeat_diagnostic.json) selects the audited source run and
a separate local output directory. No backend or precision flags are changed. Only
one policy/environment is live at a time. Each step is copied and compressed before
proceeding; the entire input sequence is not retained in RAM or VRAM. Input images
alone would occupy about 0.9 GB across both episodes without compression; actual
archive size is recorded per trial. All raw archives and videos stay gitignored.

Outputs under `outputs/repeat_diagnostic/`:

- `diagnostic.json`: trial outcomes, first-divergence comparison and replay results.
- `trace_comparison.json`: the first observed difference for each recorded field.
- Per-trial `inference.zip`: compressed, lossless, pickle-free NumPy records with
  tensor byte representations, dtype/shape/device metadata and exact RNG values.
- Per-trial `inference.jsonl`, `steps.jsonl`, `trajectory.npz`, initial state and two
  20 FPS videos; replay results are saved in a separate completed checkpoint.
- Execution reports, console logs, device-wide GPU samples and GNU-time resources.

On interruption, rerun the same command and answer `y` to continue. Completed trials
and replay are reused only when source/configuration/artifact hashes agree. A partial
episode restarts from its initial state in a new attempt; no mid-episode continuation
is attempted. The inherited utility describes these records as "pairing checks" in
its prompt. A changed diagnostic requires a new `output_dir` in this configuration.

## Interpretation and acceptance

`PASS` means collection/replay execution completed, not that trajectories were equal.
Review the actual comparison:

| Evidence | What it supports |
| --- | --- |
| Raw pixels or preprocessed inputs differ before actions | Investigate observation/rendering/processing differences first |
| RNG or sampled noise differs | Inspect when random-state progression separates |
| Replays differ despite matching input and sampled noise | Variability within inference or unrecorded model state; no specific kernel is identified |
| Replays match each other but differ from the recorded action | Historical state/timing context remains relevant; equality is not established for the original call |
| No divergence reproduced | The added tracing may affect timing; this does not prove universal determinism |

The completed replays all matched their respective recorded actions. Both original
trials succeeded at step 78 and at the final step. The observed trajectory difference
therefore did not change this pair's episode outcomes. Inspect the saved pixel
evidence without another GPU run using `python scripts/inspect_repeat_pixels.py`.

Tracing adds copying and synchronization, so its effect on runtime and potentially
timing-dependent variability is a limitation. CUDA flags must not be changed silently
to make the control pass. Follow up on the measured finding before broadening the
language benchmark or starting fine-tuning.
