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

## Earlier development archive

The following pre-confirmatory materials remain available as historical engineering evidence. Their small samples and wordings must not be pooled with the final result.

- [First official-policy rollout](first_rollout/README.md)
- [Language pilot](language_pilot/README.md) and [repeat diagnostic](repeat_diagnostic/README.md)
- [200-update adaptation comparison](adaptation_development/README.md) and [failure review](adaptation_development/qualitative_review.md)
- [Lower-LR comparison](lower_lr_development/README.md)
- [1,000-update early development comparison](longer_training_development/README.md)

`goal_ever` means the task goal held at least once; `goal_final` means it held at the final step. They should not be interchanged. Earlier repeat controls and changed-goal probes are not the final held-out language outcome.
