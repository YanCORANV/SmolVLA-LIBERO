# Research results

The latest completed comparison is the [1000-update development experiment](longer_training_development/README.md).
All comparisons remain task-0 pilot results; none is a full LIBERO benchmark or an
untouched test. Each primary language condition has three initial-state cases.

| Evidence | What it establishes |
| --- | --- |
| [First rollout](first_rollout/README.md) | One successful original-instruction episode; official policy/environment integration |
| [Language pilot](language_pilot/README.md) | Six instruction conditions on one state, including repeat and changed-goal probes |
| [Repeat diagnostic](repeat_diagnostic/README.md) | Pixel/input divergence and recorded-input replay evidence |
| [State benchmark protocol](../docs/state_benchmark.md) | 18 episodes across three matched state groups; raw summary retained locally |
| [First fine-tuning comparison](adaptation_development/README.md) | Matched 200-update arms at peak learning rate 1e-4; B/C remain unsuccessful |
| [Qualitative failure review](adaptation_development/qualitative_review.md) | Eight selected grasp/lift and transfer/placement cases |
| [Lower learning rate](lower_lr_development/README.md) | 200 updates at 1e-5; augmented C reaches the goal in 1/3 cases but ends at 0/3 |
| [Longer training](longer_training_development/README.md) | 1000 updates at 1e-5; augmented C succeeds in 2/3 cases, B in 0/3 |

## Reading the numbers

The adaptation comparisons use a precision-matched untuned baseline. Do not merge
its episodes with the earlier native-precision pilot as if all belonged to one
experiment. Later rounds reuse verified baseline records rather than rerunning
them; reuse does not increase the number of independent observations.

Goal-ever means the goal held at least once; goal-final means it held at the final
step. These differ in the lower-learning-rate C example. Identical-original repeat
controls are reported separately and excluded from the main A/B/C table.
Changed-goal metrics remain provisional and are not evidence of semantic sensitivity.

CSV/JSON files and selected keyframes are versioned here. Raw logs, model downloads
and rollout videos stay in ignored local directories. Public video release attachments
and comparison plots remain follow-up work. New results should retain failed cases,
repeat disagreements, protocol differences and clear development/test labels.
