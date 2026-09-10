# Documentation index

Read the [root README](../README.md) for current project status and
[results index](../results/README.md) for measured outcomes. Stage-specific documents
record the protocol and acceptance requirements at the time each stage was built;
their historical pending-work notes are not the current project status.

| Need | Start here |
| --- | --- |
| Install and run the first policy episode | [Setup](setup.md) |
| Reproduce experiments and understand local artifact dependencies | [Reproduction](reproduction.md) |
| Understand instruction pairing and repeated-original control | [Language pilot](language_pilot.md), [repeat diagnostic](repeat_diagnostic.md) |
| Evaluate matched initial states | [State benchmark](state_benchmark.md) |
| Prepare demonstrations and validate a training update | [Adaptation](adaptation.md), [training profile](training_profile.md) |
| Understand why there are two fine-tuning arms | [Paired fine-tuning](paired_finetuning.md) |
| Reproduce the first training/evaluation round | [Training](paired_finetuning.md), [evaluation](adaptation_evaluation.md) |
| Reproduce the lower-learning-rate round | [Training](learning_rate_comparison.md), [evaluation](learning_rate_evaluation.md) |
| Reproduce the longer-training round | [Training](training_budget.md), [evaluation](training_budget_evaluation.md) |
| Understand concurrent evaluation limits | [Parallelism](evaluation_parallelism.md) |
| Publish the initial GitHub repository | [Chinese publication guide](publish_github_zh.md) |

Recorded environment versions, data audits and training validation reports are
also stored alongside these documents. Raw machine-local logs stay under `outputs/`.
