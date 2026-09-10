# Implementation plan and acceptance gates

Milestone A passed on 2026-09-08: one successful original-instruction rollout, video
verification, CSV/JSON export and resource measurements are in `results/first_rollout/`.
Milestone B's catalog and six-condition paired-reset checker passed on 2026-09-08.
Its six-condition paired-policy pilot completed and passed saved-artifact checks.
Repeat trajectories differ from step 46; both originals still succeed. The repeat
diagnostic localizes the input difference to four camera pixels; recorded-input
replays reproduce their respective actions. Low-level rendering cause and positive
alternative-goal validation remain open. Milestones C–E
remain planned. Implement each after its preceding gate has evidence.

## A. First working rollout

Accept only after dependency checks, a CUDA tensor operation, MuJoCo rendering,
official policy loading and a complete LIBERO episode pass. Retain command/config,
package versions, checkpoint revision, seed, task, console log, success flag,
elapsed time, GPU memory measurement and a playable video. A successful execution
need not solve the task. Add JSON/CSV export based on the actual observed upstream
output schema; do not invent schema fields before verifying them.

Before expanding runs, address these observed gaps: slow cold imports from the
Windows-mounted environment; the upstream video-FPS mismatch; and missing per-step
instruction/action, simulator-state and policy-RNG records. Preserve this first
attempt while adding that instrumentation. Decide whether to relocate the Python
environment to Linux storage as a separate setup change and validate it before reuse.

## B. Controlled language benchmark

Gate B1 passed: six conditions have identical captured initial arrays and images, and
all instructions fit within 48 tokens. Gate B2 execution passed; read the
[measured results and diagnostic next gate](../results/language_pilot/README.md).
Twenty code checks pass, and the [repeat diagnostic](repeat_diagnostic.md) completed.
Read the [pixel localization and benchmark decision](../results/repeat_diagnostic/README.md).
Next, expand initial states and seeds with per-group identical-original controls and
report outcome variability separately from pixel/trajectory equality. Alternative
goal positive cases remain unvalidated.
The [multi-initial-state runner](state_benchmark.md) now implements that expansion.
It prompts for state indices and policy seeds, stores the selection, validates
initial pairs and reports complete-group metrics. The first GPU batch completed
18 episodes in 3 groups (states 0, 1, 2; policy seed 0) in 4262.87 seconds.
Original and repeat each succeeded in 1/3 episodes, both paraphrases in 0/3;
repeat ever-success disagreement was zero. The immediate next stage is
[paraphrase adaptation](adaptation.md), with semantic-change experiments deferred.
Thirty-nine CPU checks pass. The official dataset metadata audit identified an
exact original-instruction match, 45 demonstrations and 4487 frames. Data preparation
is implemented; its first download exposed stale upstream file pointers, now corrected
using a pinned official location reference with the saved split preserved. Full data
preparation has now passed for all 45 demonstrations and the 36/9 split. The
[real optimizer-step profile](training_profile.md) passed official-loader,
GPU-update and checkpoint-reload execution (3.69 s, 3.08 GiB peak reserved).
Both [paired fine-tuning](paired_finetuning.md) arms completed 200 updates with
matching training controls, about 0.70 seconds per recorded update and 3.14 GiB peak
reserved memory. The [three-model simulator comparison](adaptation_evaluation.md)
completed 36 episodes: A counts were 1/3, 0/3 and 1/3 for untuned, original-language
tuning and augmented-language tuning; both paraphrases were 0/3 for every model.
See [post-training results](../results/adaptation_development/README.md). Video review
and the next controlled adaptation experiment remain to be done. See
[repair evidence](dataset_mapping_repair.json).

1. Start with Spatial tasks and define feasible object/relation contrasts. Author
   original, meaning-preserving and meaning-changing conditions. Human-review
   equivalence, referents, feasibility, token truncation and ambiguity. Keep exact
   strings, variant IDs and annotation rationale in versioned JSONL.
2. Build a thin adapter around the official environment/processor/evaluation APIs.
   Change only the instruction input. Store task ID, initial-state index and hash,
   simulator seed, policy noise seed, instruction ID/text, checkpoint hash, precision,
   action horizon, renderer settings and terminal outcome for every episode.
3. Validate restored states and first observations across conditions. Reset policy
   action queues and RNG state per paired trial. A seed alone is insufficient, and
   upstream vectorized initial-state selection must be inspected before batching.
4. Use an episode-level durable journal with atomic writes and visible progress.
   On restart, interactively select whether to continue an unfinished run. Completed
   paired trials are reused only when configuration and provenance match; incomplete
   trials remain marked incomplete. No command-line resume/limit modes.
5. Execute serially on the 8 GB GPU. Expand to Object, Goal and Long after correctness
   and runtime are measured. Plan seeds 0, 1 and 2 with fixed state lists, but measure
   wall time before setting the final per-task episode budget interactively.

Metrics: original-instruction success; paraphrase success and paired percentage-point
change; original-goal persistence for changed instructions; and new-goal compliance
where a validated alternative predicate exists. Report impossible/ambiguous edits
separately. Do not interpret a lower original success rate alone as better sensitivity.
Include repeated-original controls to quantify noise-induced trajectory differences.

## C. Language-augmented fine-tuning

Use the official LIBERO demonstration format and upstream LeRobot training entry
point. Pin the dataset revision. Split demonstrations by episode before augmentation,
and keep evaluation paraphrase templates disjoint from training templates. Describe
any task overlap honestly: the starting checkpoint is already LIBERO-adapted.

Attach only meaning-preserving instructions to an unchanged demonstration. A changed
goal requires demonstrations that actually achieve it; do not relabel actions with
inconsistent instructions. Prefer metadata-level language sampling to duplicating
large image/video data. Verify upstream language-column support before adding adapters.

Profile a real optimizer step with action-expert-only updates, frozen VLM, batch 1 and
mixed precision. Record trainable parameter count, CUDA peak and process RSS. Check
loss/gradient finiteness and successful reload. Validate gradient accumulation in the
installed upstream trainer before choosing an effective batch size. LoRA is a later
candidate only if the supported modules and memory measurements justify it.

Train (i) the original-language control and (ii) the augmented-language model from
the same checkpoint with equal optimizer-update and demonstration-sampling budgets.
Keep learning-rate schedules and evaluation policy settings matched. Add three
training seeds when resource measurements justify the cost. Preserve upstream
optimizer/scheduler/RNG checkpoints; expose resume choices interactively in any future
project wrapper. No actual training command is claimed as tested at this stage.

## D. Ablations and failure analysis

Minimum comparisons: pretrained baseline, equal-budget original-language tuning,
and paraphrase-augmented tuning. Then compare object-focused vs relation-focused
augmentation while controlling exposure and compute. Do not confound language effects
with unfreezing, precision or action-horizon changes in the primary comparison.

Annotate wrong-object selection, incorrect spatial relation, ignored/misread goal,
grasp/control failure, timeout, and ambiguous cases. Allow overlapping labels and
retain uncertainty. Review language grounding alongside simulator predicates and
trajectories; video alone may not identify the cause. Save fixed-time qualitative
comparisons and a representative set of failures, not only successes.

Use paired outcome tables and task-aware confidence intervals; avoid treating
paraphrases of the same initial episode as independent samples. Report per-task and
macro averages with explicit denominators. Candidate analyses include task/episode
cluster bootstrap and paired binary-outcome comparisons with the dependence structure
stated. Export plotting data plus vector PDF/SVG figures and captions after results exist.

## E. Public release and resume evidence

Publish setup validated on a clean environment, a reviewed dependency lock, exact
evaluation/training commands, protocol/versioned seeds, CSV/JSON results, figure
generation code, selected videos, limitations, upstream attribution and a selected
project license. Keep checkpoints/datasets out of Git; use model/dataset or release
hosting with revision links. Add meaningful regression checks for language replacement,
paired initial states and continuation once those implementations exist.

Use separate focused commits for setup, first rollout, benchmark, adaptation and
analysis. Before writing resume bullets, connect every numerical claim to a released
result table and configuration. A pipeline-only result supports an integration claim;
fine-tuning and robustness claims require completed controlled comparisons.
