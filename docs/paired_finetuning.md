# First controlled fine-tuning comparison

Status: both arms completed 200 updates; matched frame, learning-rate, initial-weight
and RNG/noise controls passed. Peak reserved memory was about 3.14 GiB and recorded
updates averaged about 0.70 seconds. See [training evidence](paired_finetuning_result.json).
Interrupted-run GPU recovery has not yet been executed. The preceding real-step
profile passed: 3.69 seconds for one guarded update, 3.08 GiB peak PyTorch reserved,
97,451,872 trainable FP32 parameters and 507,482,304 frozen BF16 parameters. Fresh-process
reload restored all parameter bytes and optimizer step 1. This does not establish
language robustness gains. See [profile evidence](training_profile_result.json).

## Run

In WSL, after the accepted training profile:

```bash
# Run from the repository root.
bash scripts/run_adaptation.sh
```

The program prints the training instructions and asks for:

1. Run name (default `task0_adaptation`).
2. Optimizer updates **per arm** (default 200).
3. Training seed (default 0).

These choices and the exact frame/language schedule are frozen before starting the
GPU workers. Original-language and augmented-language tuning run serially in separate
processes. Both start from the original official checkpoint, not the one-step profile
checkpoint. The profile only establishes a working runtime and data path.

For continuation, use the same command and run name and answer `y`. Each arm resumes
from its latest complete checkpoint after artifact hash verification, strict policy
loading and exact parameter comparison. Official APIs restore optimizer, scheduler,
training step and RNG; the frame/language schedule determines the next sample.
Completed arms are verified and reused. The saved update budget cannot be extended
in place, because doing so would change the registered learning-rate schedule.

## Matched conditions

| Item | Original-language arm | Augmented-language arm |
|---|---|---|
| Demonstrations | Same 36 training episodes | Same 36 training episodes |
| Frame order | Fixed shared schedule | Identical schedule |
| Updates / batch | Chosen budget / 1 | Same budget / 1 |
| Initial policy and precision | FP32 expert adapter over official checkpoint | Identical parameter bytes |
| Text | Official original A | A plus four new training paraphrases |
| Learning rate | Official cosine scheduler, short warmup | Same schedule |
| Held-out demonstrations | All 9 excluded | All 9 excluded |

The new [training catalog](../configs/instructions/task0_train.jsonl) preserves the
original bowl, between relation and plate goal. Existing development paraphrases
are excluded even under punctuation/case normalization. These are project-authored
variants; no independent language review is claimed.

Frames are sampled uniformly with replacement from the 3513 training frames using
a dedicated local RNG. Language selection uses a separate local RNG and shuffled
blocks containing all five choices. Thus every 5 updates use each training wording
once in the augmented arm, without consuming policy randomness or changing the frame
sequence. At the default 200 updates, each augmented wording occurs 40 times. This
is a deliberately small optimization budget, not an entire pass over all frames.

The policy uses the profile's batch-1, BF16-autocast, frozen-VLM, FP32 trainable storage
and original checkpoint normalization. The installed official scheduler uses
`min(20, updates // 10)` warmup steps, the chosen total decay horizon, peak learning
rate 1e-4 and the policy preset's final learning rate. The profile's one update had
no scheduler; scheduling is the new behavior that must be verified in this run.

## Saved evidence and recovery

Each update records episode/frame ID, instruction ID, loss, gradient norm, actual
learning rate, duration, memory and hashes of the input tensors, RNG and sampled
flow noise. The final comparison rejects mismatched data, starting parameters,
learning rates, language assignments or budgets. Noise/RNG differences are reported
explicitly; identical seeds alone are not called proof of identical random streams.

Checkpoints are written every 100 updates and at the final update. A complete marker
is created only after policy, processors, optimizer, scheduler, RNG, parameter hashes,
and committed JSONL/CSV history are saved and hashed. Incomplete attempts and their
uncommitted logs are retained. Updates after the most recent complete checkpoint
will be redone on continuation; partial optimizer steps are never treated as complete.

Each checkpoint is approximately 2.2 GB. At the default budget, two checkpoints per
arm require roughly 9 GB, plus caches and logs. No checkpoints are automatically
deleted. Startup, source checks, data loading and checkpoint writes add overhead;
the live log reports measured update throughput and excludes future saves from its
remaining-update estimate. The earlier 3.69-second profile is not a total-time promise.

Outputs are under `outputs/adaptation/runs/<name>/`. `comparison.json` points to the
final checkpoints and records paired training controls. It contains **no rollout
success rates**. Console/resource/device-memory logs are under the separate
`outputs/adaptation/executions/` directory.

## What follows

Next, use the [paired evaluation command](adaptation_evaluation.md) to evaluate the untrained numerical control and both tuned
checkpoints on the same task/state/seed/language cases. Use the same FP32 expert
loader and inference settings across the comparison, and report the earlier BF16
checkpoint pilot separately. Evaluate A and development B/C first; reserve new
phrases and cases before selecting a final model. Existing development wording has
already informed project choices and must not be called an untouched final test.

No VLM pretraining, alternative-goal relabeling or success-rate improvement is claimed
by the training command. The project contribution is the controlled language-data
comparison around official LeRobot training and LIBERO data.
