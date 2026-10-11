# Curated research results

**Current conclusion:** the selected `paired_bc` arm achieved high canonical success in the amended final v4 evaluation but did **not** improve pooled success on newly frozen unseen paraphrases. The development gain did not generalize. Read the [full result interpretation](../docs/results.md) before using any number.

| Stage | Versioned public evidence | Role |
| --- | --- | --- |
| First 240-rollout confirmatory set, States 3–22 | [Summary](held_out_confirmatory/summary.json) | Canonical-only versus uniform augmentation; early robustness/retention trade-off |
| Development, States 23–32 | [Anchored comparison](development_selection/anchored_summary.json), [paired comparison](development_selection/paired_summary.json) | Model selection only |
| Action/trajectory diagnosis | [Summary](language_divergence/summary.json), [pair-level CSV](language_divergence/pair_level_divergence_measurements.csv) | Existing-rollout analysis |
| Matched paired-training controls | [Schedule](paired_training/schedule.json), [control audit](paired_training/paired_control_audit.json) | Input/noise/time parity |
| Final v4, States 33–49 | [Summary](final_confirmatory_v4/summary.json), [per-state CSV](final_confirmatory_v4/per_state_raw_outcomes.csv) | Amended frozen final evaluation, 153/153 valid rollouts |

The data files above are byte-identical copies of completed local `outputs/` summaries. The final protocol amendment, checkpoint hashes and prompt hashes are in [the v4 freeze record](../configs/final_confirmatory/freeze_final_protocol_v4.json). Figures under [assets/figures](../assets/figures/) can be rebuilt with [the figure script](../scripts/make_portfolio_figures.py). Raw checkpoints, trajectories, videos and logs remain local and Git-ignored.

## Separate five-task development pilot

The [multi-task language sensitivity report](../docs/multitask_language_pilot.md) evaluates **untuned_fp32**, not the Task-0 fine-tuned models. It covers five competence-selected tasks, States 5–9, and 75/75 committed rollouts. The [75-row per-rollout source table](multitask_language_pilot/per_rollout.csv), [50-row paired diagnostics](multitask_language_pilot/pair_level.csv), [aggregate summary](multitask_language_pilot/summary.json), [integrity audit](multitask_language_pilot/integrity.json), and [12 video-backed cases](multitask_language_pilot/case_index.csv) are checked in. Four quantitative figures and the 12 selected-case keyframe sheets are in the same directory. The [failure-case review](../docs/multitask_failure_cases.md) records what the visible evidence supports and leaves unobserved mechanisms unknown.

Run python scripts/regenerate_multitask_language_pilot.py to verify published statistics and figures without model weights or raw rollout data. Its input files are exactly the curated CSV/JSON files above plus the frozen matrix, manifest and five prompt catalogs. It does not independently verify the original simulator outcomes; the stronger raw-artifact validation requires the local ignored outputs.
## Earlier development archive

The following pre-confirmatory materials remain available as historical engineering evidence. Their small samples and wordings must not be pooled with the final result.

- [First official-policy rollout](first_rollout/README.md)
- [Language pilot](language_pilot/README.md) and [repeat diagnostic](repeat_diagnostic/README.md)
- [200-update adaptation comparison](adaptation_development/README.md) and [failure review](adaptation_development/qualitative_review.md)
- [Lower-LR comparison](lower_lr_development/README.md)
- [1,000-update early development comparison](longer_training_development/README.md)

`goal_ever` means the task goal held at least once; `goal_final` means it held at the final step. They should not be interchanged. Earlier repeat controls and changed-goal probes are not the final held-out language outcome.
