# Results and interpretation

**Primary metric:** `goal_ever`, meaning the task goal was true at least once during a 280-step rollout. `goal_final` is secondary and remains in the raw records. O is canonical success, P is success pooled over two unseen paraphrases, Gap = P − O, and H = 2OP/(O+P) (zero if O+P=0). All rates below are Task 0 only.

## First frozen confirmatory experiment

The first [frozen report](../results/held_out_confirmatory/summary.json) contains **240 rollouts**: three models × States 3–22 × canonical, identical-repeat and two then-held-out paraphrases. The figures here use `goal_ever`.

| Model | Canonical O | Held-out 01 | Held-out 02 | Pooled unseen P | Gap |
| --- | ---: | ---: | ---: | ---: | ---: |
| `untuned_fp32` | 16/20 · 80% | 5/20 · 25% | 0/20 · 0% | 5/40 · 12.5% | −67.5 pp |
| `original_language` | 19/20 · 95% | 10/20 · 50% | 0/20 · 0% | 10/40 · 25% | −70 pp |
| `augmented_language` | 16/20 · 80% | 11/20 · 55% | 4/20 · 20% | 15/40 · 37.5% | −42.5 pp |

Uniform augmentation was **+12.5 percentage points** on pooled unseen paraphrases relative to canonical-only fine-tuning, while canonical success was **−15 points**. The first report's paired state-bootstrap 95% intervals were [0, +25] points and [−30, 0] points respectively. This was an observed robustness/retention trade-off, not evidence that augmentation reliably fixes language sensitivity. These first two paraphrases were consumed and were not reused as final prompts.

## Development and model selection

All development figures use States **23–32**, the canonical instruction and two development paraphrases. The anchored experiment is recorded in its [90-rollout summary](../results/development_selection/anchored_summary.json); the paired arms in their [60-rollout summary](../results/development_selection/paired_summary.json). The table combines the reported values for the shared development conditions; O has denominator 10, each prompt 10, and P denominator 20.

| Model | O | Dev 01 | Dev 02 | P | H | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `original_language` | 9/10 · 90% | 7/10 · 70% | 1/10 · 10% | 8/20 · 40% | 0.5538 | Baseline |
| `augmented_language` | 8/10 · 80% | 3/10 · 30% | 3/10 · 30% | 6/20 · 30% | 0.4364 | Baseline |
| `anchored_language` | 6/10 · 60% | 3/10 · 30% | 4/10 · 40% | 7/20 · 35% | 0.4421 | Failed anchored gate |
| `paired_bc` | 10/10 · 100% | 5/10 · 50% | 7/10 · 70% | 12/20 · 60% | **0.7500** | Selected development candidate |
| `paired_flow_consistency` | 9/10 · 90% | 8/10 · 80% | 3/10 · 30% | 11/20 · 55% | 0.6828 | Failed gate vs `paired_bc` |

The anchored 50%-canonical sampling arm failed its registered development conditions. The consistency arm passed its absolute O≥0.80 and P≥0.45 checks, but H=0.6828 did **not** exceed `paired_bc` H=0.7500. Therefore the final headline comparison included only `original_language`, `augmented_language` and `paired_bc`.

## Action and trajectory diagnostics

The [diagnostic summary](../results/language_divergence/summary.json) and [pair-level measurements](../results/language_divergence/pair_level_divergence_measurements.csv) were computed from existing held-out and development rollouts, without new policy executions. The set has 60 identical-instruction repeat pairs and 180 canonical-versus-paraphrase pairs.

- **Step 1:** repeat action-vector difference was exactly zero for all repeat pairs. The paraphrase pair means were about 0.17 L2. All 180 paraphrase pairs exceeded the empirical action-difference threshold 0.05 at Step 1.
- **Early trajectory:** end-effector separation crossed 1 cm at median Step 6 for successful paraphrase pairs and Step 5 for failed paraphrase pairs. This is after an action difference had already appeared.
- **Failure association:** mean action difference over the first 10% of the rollout was 0.218 for paraphrase successes (n=51) and 0.294 for failures (n=129). The recorded one-sided Mann–Whitney p-value was 0.0011. Step-1 magnitude alone did *not* separate those groups (p=0.746 in the recorded comparison).

![Step-1 action-vector differences](../assets/figures/step1_action_divergence.png)

![Onset of more than 1 cm end-effector separation](../assets/figures/trajectory_divergence_onset.png)

These diagnostics support immediate language-conditioned action differences and later closed-loop separation. They are observational analyses; they do not prove that the measured divergence alone caused every failure.

## Final confirmatory evaluation: frozen v4

The final [v4 matrix](../configs/final_confirmatory/final_matrix_v4.json) used States **33–49**, `original`, `paraphrase_final_01`, `paraphrase_final_02`, three models, 280 steps, one worker, environment seed 0 and policy seed 0. All **153/153** expected rollouts have one valid `complete.json` commit. The final instruction catalog and checkpoints remained byte-identical to the [v4 freeze record](../configs/final_confirmatory/freeze_final_protocol_v4.json).

The initial 20-state plan asked for States 33–52. The active official Task-0 asset contains only 50 rows, indexed 0–49. After 51 valid rollouts on 33–49, an attempted State-50 reset failed before inference. Audit of the 51 saved initial-state fixtures showed exact equality to rows 33–49 of that same asset. The analysis plan was feasibility-amended to 17 state clusters **before any intermediate success outcomes were inspected**; previous freezes, logs and incomplete attempts were preserved. This is an amended 153-rollout study, not the originally planned 180-rollout study.

The [final JSON](../results/final_confirmatory_v4/summary.json) contains the pre-registered 10,000-replicate paired state-cluster percentile bootstrap. Each resampled state carries all nine model/prompt outcomes. Entries below show count/denominator, point rate, and 95% interval.

| Model | Canonical O | Final 01 P1 | Final 02 P2 | Pooled P |
| --- | --- | --- | --- | --- |
| `original_language` | 12/17 · 70.6% [47.1, 88.2] | 3/17 · 17.6% [0.0, 35.3] | 2/17 · 11.8% [0.0, 29.4] | 5/34 · 14.7% [5.9, 26.5] |
| `augmented_language` | 12/17 · 70.6% [47.1, 88.2] | 4/17 · 23.5% [5.9, 47.1] | 3/17 · 17.6% [0.0, 35.3] | 7/34 · 20.6% [8.8, 35.3] |
| `paired_bc` | 16/17 · 94.1% [82.4, 100.0] | 1/17 · 5.9% [0.0, 17.6] | 3/17 · 17.6% [0.0, 35.3] | 4/34 · 11.8% [2.9, 23.5] |

| Model | Gap = P − O (percentage points) | H |
| --- | --- | --- |
| `original_language` | −55.9 [−76.5, −32.4] | 0.2434 [0.1046, 0.3850] |
| `augmented_language` | −50.0 [−79.4, −17.6] | 0.3188 [0.1553, 0.4547] |
| `paired_bc` | −82.4 [−94.1, −67.6] | 0.2092 [0.0569, 0.3715] |

### Registered paired effects

Positive values favor `paired_bc`. Intervals resample **state IDs**, not individual rollouts.

| Comparison | Estimate | 95% interval |
| --- | ---: | ---: |
| P: `paired_bc − original_language` | −2.9 pp | [−14.7, +8.8] pp |
| P: `paired_bc − augmented_language` | −8.8 pp | [−20.6, +2.9] pp |
| O: `paired_bc − original_language` | +23.5 pp | [0.0, +47.1] pp |
| O: `paired_bc − augmented_language` | +23.5 pp | [0.0, +47.1] pp |
| H: `paired_bc − original_language` | −0.0343 | [−0.2246, +0.1539] |
| H: `paired_bc − augmented_language` | −0.1096 | [−0.2608, +0.0543] |

Both paraphrase-effect intervals contain zero. The point estimates do **not** show an unseen-paraphrase gain for `paired_bc`; its canonical point estimate is higher. No new significance threshold was chosen after results.

## Development versus final generalization

| `paired_bc` | Development, States 23–32 | Final v4, States 33–49 |
| --- | ---: | ---: |
| Canonical O | 10/10 · 100% | 16/17 · 94.1% |
| Pooled paraphrases P | 12/20 · 60% | 4/34 · 11.8% |
| Balanced H | 0.7500 | 0.2092 |

![Development versus final performance](../assets/figures/development_vs_final.png)

The development and final **states and prompt texts both differ**, so this contrast cannot isolate a prompt-only or state-only causal effect. It does show that selecting a language-robustness candidate on one development language set did not predict success on the frozen new paraphrases. The scientifically supported conclusion is a **negative final robustness result**, not a successful new method.

## Artifacts and limits

- [Final raw per-state outcome CSV](../results/final_confirmatory_v4/per_state_raw_outcomes.csv) and [aggregate JSON](../results/final_confirmatory_v4/summary.json)
- [Frozen final model manifest](../configs/final_confirmatory/frozen_model_manifest_v4.json), [matrix](../configs/final_confirmatory/final_matrix_v4.json), [freeze/audit record](../configs/final_confirmatory/freeze_final_protocol_v4.json)
- [Reproducible figure script](../scripts/make_portfolio_figures.py)
- Machine-local raw `outputs/` retains checkpoint bytes, trajectories, MP4s, `execution.jsonl` and full stdout/stderr logs; it is intentionally Git-ignored.

This is Task 0 only with 17 valid final initial states and two final paraphrases. Confidence intervals are wide. The official initialization asset, local checkpoints and raw rollouts are not distributed in this repository; a clean-clone full GPU rerun has not been independently verified.
