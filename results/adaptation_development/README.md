# First post-training development comparison

Both 200-update training arms and all 36 evaluation episodes completed. The final
comparison was independently recomputed from the 36 saved episode records. Each
model passed exact parameter reload and original-normalization checks, with matched
FP32 expert storage and frozen BF16 VLM parameters.

| Model | Original A | Paraphrase B | Paraphrase C |
| --- | --- | --- | --- |
| Untuned precision control | 1/3 | 0/3 | 0/3 |
| Original-language tuning | 0/3 | 0/3 | 0/3 |
| Language-augmented tuning | 1/3 | 0/3 | 0/3 |

Counts agree for goal-ever and goal-at-final-step. Original-repeat outcome
disagreements were zero for all models. Repeats are excluded from the table above.
Untuned A succeeded at state 0; augmented A succeeded at state 1 and failed at state
0. Equal aggregate counts therefore do not mean unchanged behavior.

**Neither development paraphrase improved in this run.** The original-language arm
lost the one original-instruction success. These observations do not identify a
failure mechanism or establish that language augmentation cannot work.

Protocol: one LIBERO Spatial task (0), initial states 0/1/2, policy seed 0,
280-step fixed horizon, native original-goal predicate, identical reset observations
and initial RNG. This is not a standard early-stopping LIBERO score. The two
paraphrases were excluded from the project's fine-tuning catalog but previously
used in development; they are not an untouched test. The nine held-out
demonstrations do not establish unseen simulator cases or absence from pretraining.

[CSV](summary.csv) and [JSON](summary.json) preserve all condition counts, paired
cases and source evidence. Raw videos, trajectories and per-step logs remain under
`outputs/adaptation_evaluation/runs/task0_adaptation_eval/`. An eight-case
[dual-camera keyframe and object-log review](qualitative_review.md) now records
grasp/lift and transfer/placement failures, including original-instruction regressions.
Continuous-video/contact review of all cases is not claimed. The next
[learning-rate comparison](../../docs/learning_rate_comparison.md) changes only the
peak learning rate. No robustness-improvement resume claim is supported yet.
