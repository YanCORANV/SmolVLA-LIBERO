# Final 1000-update development comparison

Both 1000-update arms and 24 new evaluation episodes completed, with 12 previously
measured untuned episodes reused after verification. The saved comparison was
recomputed from the episode records.

| Model | Original A | Paraphrase B | Paraphrase C |
| --- | --- | --- | --- |
| Untuned, reused baseline | 1/3 | 0/3 | 0/3 |
| Original-language tuning, 1000 updates | 3/3 | 0/3 | 0/3 |
| Language-augmented tuning, 1000 updates | 2/3 | 0/3 | 2/3 |

Goal-ever and final-step counts agree in this round. Repeats are excluded from the
main table. The original-language model's identical-A repeat succeeded in 2/3,
disagreeing with its A outcome at state 1. The augmented model's A repeat succeeded
in 2/3 with no case-level disagreement. Do not omit this outcome variability or
attribute it to concurrency without an additional controlled check.

The augmented model completed C at states 0 and 1, including at the final step.
Compared with the [200-update lower-learning-rate run](../lower_lr_development/README.md),
C final success improved from 0/3 to 2/3, while A declined from 3/3 to 2/3 (state 1
was lost). B remains 0/3. Within the 1000-update comparison, the original-language
control still achieved C in 0/3. This is preliminary evidence of improvement on
one development paraphrase, not general language robustness or a statistical claim.

## Parallel execution

The scheduler recorded genuine process overlap and successful exit of both workers.
Its stage took 7415.64 seconds (about 2 h 4 min), including worker startup. During
the sampled overlap, median GPU utilization was 42%, minimum available RAM 0.75 GiB,
maximum swap used 0.25 GiB, and minimum GPU free memory 2.20 GiB. These observations
do not prove a particular bottleneck.

The previous 24-episode serial command took 1 h 47 min. This parallel run did not
deliver a shorter observed duration, but different checkpoints and timing scopes
prevent a controlled speedup/slowdown estimate. Parallel execution also differs
from earlier rounds and its effect on outcomes has not been isolated. For the next
validation, prefer entering 1 at the existing concurrency prompt unless a separate
matched performance comparison justifies using 2. No scheduler source or completed
experiment is changed retroactively.

## Scope and next validation

One task, states 0/1/2, policy seed 0, 280-step fixed horizon, native original goal.
Both update budget and cosine decay horizon changed from the 200-update reference.
Hyperparameters and these wordings/states were used during development. Baseline
reuse supplies no new independent observations. No full-LIBERO, untouched-test,
causal-overfitting or broad significance claim follows from these three cases.

[CSV](summary.csv) and [JSON](summary.json) retain all counts, case pairs, repeat
disagreements and scheduling evidence. Raw videos/logs remain in
`outputs/adaptation_evaluation/runs/task0_longer_training_eval/`. Review the new C
successes and the repeat disagreement, then freeze a candidate configuration and
expand validation to additional cases/wordings. Do not increase the training budget
again solely to search for a better score on these same three cases.
