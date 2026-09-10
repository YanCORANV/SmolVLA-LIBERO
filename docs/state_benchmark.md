# Multi-initial-state task-0 benchmark

Status: first GPU batch completed: 18 episodes in 3 matched groups, states 0, 1, 2,
policy seed 0. Original and repeat each succeeded in 1/3 episodes; each paraphrase
in 0/3. Original/repeat ever-success disagreed in zero groups. Total execution was
4262.87 seconds. Outputs are in `outputs/state_benchmark/runs/task0_states`.
No training is performed by this command. These are fixed-horizon development results.

## Run interactively

In the existing WSL terminal:

```bash
# Run from the repository root.
bash scripts/run_state_benchmark.sh
```

On the first invocation, the program asks:

```text
Run name [task0_states]:
Initial-state indices [0,1,2]:
Policy seeds [0]:
```

Press Enter for these defaults: three official initial states, one policy seed,
six conditions per state, for **18 full episodes**. Entering policy seeds `0,1,2`
instead selects 54 episodes. The selected integers are explicit state indices and
seeds, not a command-line sample cap. Each episode has the existing full 280-step
protocol horizon. The environment seed remains fixed at the pilot's value, 0.

The run prints its total episode count and a rough rollout-only estimate based on
observed pilot throughput. The default 18 episodes imply roughly an hour of rollout
work, plus startup, fresh reset checks, validation and I/O. Previous whole-command
overhead was substantial; no total wall time is promised. CPU/GPU stages are timed
and the shell retains complete console, GNU-time and GPU-memory logs.

The choices are saved in `experiment.json` before expensive work. Select a different
run name to change the state/seed budget. Reuse the same run name to continue an
existing experiment and answer `y` to the single continuation prompt. The code
verifies provenance and completed artifacts before reusing them. Partial episodes
restart in a new attempt folder; completed episodes and reference checks are retained.

The state count is checked against the actual installed official initial-state file;
there is no modulo wrap for out-of-range indices. Instructions, model checkpoint,
processor normalization, action horizon, renderer settings and backend flags stay
fixed. No previous pilot episodes are included as new benchmark measurements.

## Execution gates

1. Verify the audited source code, installed dependencies, task assets, checkpoint
   and immutable cached backbone. Model files are read for hash checks once per
   invocation; the runner does not intentionally download new model weights.
2. For every selected initial state, create two fresh official environments with
   the same environment seed. Require exact equality of captured initial simulator
   arrays, robot observations and both camera images, and initially false goals.
   Save the reset arrays and camera PNGs. A mismatch stops and retains evidence.
3. Load one official policy only after all selected reference checks pass. Keep one
   policy and one environment live at a time on the 8 GB GPU.
4. Run states and policy seeds in ascending order, with the six catalog conditions
   in their existing fixed order. Recheck each episode against its state reference;
   reset processors, action queues and policy RNG, and compare initial RNG within
   every state/seed group. Complete and verify each episode before starting the next.
5. Export per-episode outcomes and descriptive aggregates over complete groups.
   Full-run completion requires every selected group to finish.

The condition order is original, original repeat, two paraphrases, changed relation,
and changed object. Variants retain the previously approved meanings. New initial
scene images should be reviewed for referent ambiguity before drawing conclusions
about changed-goal compliance. Positive alternative-goal validation is still pending.

## Instrumentation and storage

Unlike the focused repeat diagnostic, this broader run keeps **hashes** of per-step
policy inputs, RNG and sampled noise rather than full image-tensor archives. It
retains applied actions, all three goal timelines, simulator states, object positions,
initial arrays/RNG and both camera videos. The official noise sampler is called once
per action; its output is observed, not replaced. CPU copying and synchronization
are still instrumentation overhead and may influence timing-dependent variation.

With the default name, outputs are under
`outputs/state_benchmark/runs/task0_states/`:

| Artifact | Contents |
| --- | --- |
| `experiment.json` / `identity.json` | Selected budget, fixed protocol and source/runtime provenance |
| `reference_state_*/` | Two verified fresh resets per state, snapshots and initial images |
| `state_*_seed_*_<instruction>/` | Per-episode attempts, logs, numeric arrays and two 20 FPS videos |
| `episodes.csv` / `episodes.json` | Every completed episode, including explicitly labeled repeat controls |
| `summary.json` | Completed/expected groups and episodes, per-instruction outcomes, paired paraphrase difference |
| `repeat_controls.json` | Original/repeat action/state equality and first per-step input/RNG/noise/action differences |
| `execution_*.json` | Cumulative stage times and success/error state for each invocation |

Shell logs are under `outputs/state_benchmark/executions/`, with their location also
recorded in each execution report. Raw artifacts remain gitignored. Nothing is
automatically published to GitHub, the HF Hub, or the reviewed `results/` directory.

## Metrics and interpretation

A group is one fixed **initial state × policy seed** with all six conditions complete.
Incomplete groups are excluded from aggregate fractions and paired differences, and
remain visible in completion counts. A failed task within a successfully executed
episode is retained as a failure; a crashed/incomplete episode is not scored as one.

- Each instruction has its own denominator of complete groups and its requested-goal
  ever/final counts. Changed-instruction outcomes are marked provisional.
- Original repeats are reported separately from the main original-condition success
  rate. Count original/repeat disagreements in ever and final success per group.
- For each group, compute the mean success of its two paraphrases minus original
  success, then average these paired differences across complete groups. This field
  is a fraction difference; multiply by 100 for percentage points.
- Record original-goal persistence separately under changed instructions. Lower
  original success alone is not evidence of correct semantic sensitivity.

This is a **single-task, multiple-state** study, not evaluation across the LIBERO
suite. States, policy seeds and paraphrases have shared dependence. The runner does
not provide confidence intervals or significance tests by treating every variant
as an independent sample. It retains the fixed 280-step protocol, which differs from
standard early-termination evaluation. Later training comparisons must use the same
protocol, plus a separately reported standard evaluation.

Review the first completed batch's videos and repeat-control outcomes before
expanding further or selecting a fine-tuning budget.
