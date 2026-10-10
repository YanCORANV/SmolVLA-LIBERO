# Documentation index

**Start with the current [project overview](../README.md), [experiment history](experiments.md), [method](method.md), and [full results](results.md).** The selected paired method did not improve unseen-paraphrase success in the amended final v4 evaluation.

| Need | Document |
| --- | --- |
| Install and check the project environment | [Setup](setup.md) |
| Understand stages and local prerequisite artifacts | [Reproduction notes](reproduction.md) |
| Read why each experiment happened | [Experiment history](experiments.md) |
| Compare ordinary augmentation and paired flow losses | [Method](method.md) |
| Inspect all held-out, development and final counts/intervals | [Results](results.md) |
| Review public presentation audit and remaining limitations | [Repository audit](repository_presentation_audit.md) |
| Find curated machine-readable outputs | [Results index](../results/README.md) |

## Historical stage notes

The following documents record the protocol and open questions **at the time of each earlier stage**. Pending-work language in them is not the current project status:

- [Language pilot](language_pilot.md), [repeat diagnostic](repeat_diagnostic.md), [state benchmark](state_benchmark.md)
- [Adaptation](adaptation.md), [training profile](training_profile.md), [paired fine-tuning](paired_finetuning.md)
- [Early adaptation evaluation](adaptation_evaluation.md), [lower LR](learning_rate_comparison.md), [longer budget](training_budget.md)
- [First held-out evaluation](held_out_evaluation.md), [parallelism notes](evaluation_parallelism.md)
- [Chinese agent handoff](agent_handoff_zh.md), [first GitHub publication guide](publish_github_zh.md)

The historical handoff still describes a planned 33–52 final split. The verified installed Task-0 asset has only 0–49; [the v4 freeze](../configs/final_confirmatory/freeze_final_protocol_v4.json) and [current results](results.md) supersede that plan.
