# Experiment history

This is the decision trail for a single LIBERO-Spatial Task-0 question: can a SmolVLA policy keep acting consistently when equivalent instructions use different wording? Each stage uses the records linked below. Development observations are not treated as final evidence.

| Stage | Question | Intervention or measurement | Observed result | Decision |
| --- | --- | --- | --- | --- |
| Pipeline and canonical baseline | Can Task 0 be adapted and evaluated reproducibly? | Official LeRobot/LIBERO observation/action path, paired initial states, checkpoint journals, canonical-only fine-tuning | `original_language` succeeded on 19/20 canonical trials but 10/40 pooled unseen paraphrases in the first confirmatory set | Keep as the task-adapted control |
| Uniform augmentation | Does language variety help? | Same demonstrations/update budget with four training paraphrases | First confirmatory pooled unseen success rose from 10/40 to 15/40; canonical success fell from 19/20 to 16/20 | Treat as a robustness/retention trade-off, not a clean win |
| Anchored sampling | Did underweighting canonical language cause the trade-off? | 50% canonical, 12.5% per training paraphrase | Development O=0.60, P=0.35, H=0.4421; pre-registered gate failed | Sampling ratio alone was insufficient |
| Action/trajectory diagnosis | Where does wording sensitivity appear? | State-matched canonical/repeat/paraphrase rollout comparisons | Repeat Step-1 action difference = 0; paraphrase Step-1 difference ≈0.17; end-effector paths usually separated by >1 cm around Step 5–6 | Test paired-language training with physical/noise/time controls |
| `paired_bc` | Does pairing identical physical samples under two texts help? | Two student flow-matching branches, same image/state/action/noise/time, different language | Development O=1.00, P=0.60, H=0.7500 | Select as development candidate |
| `paired_flow_consistency` | Does a frozen canonical teacher add benefit? | Same paired schedule plus teacher velocity-field consistency, λ=1 | Development O=0.90, P=0.55, H=0.6828; failed to beat `paired_bc` | Do not advance to final comparison |
| Final v4 | Does the selected candidate generalize to *new* frozen prompts? | Three models × States 33–49 × three prompts; 153 committed rollouts | `paired_bc` canonical 16/17, pooled paraphrases 4/34; both baseline pooled rates were higher | Negative robustness result; stop experiments |

## Why the sequence matters

The early pilot and training development rounds established a working SmolVLA/LIBERO pipeline, including initial-state pairing, repeat controls, official demonstration mapping and resumable artifacts. The first **frozen 240-rollout** evaluation used States 3–22, three models (`untuned_fp32`, `original_language`, `augmented_language`) and four conditions (canonical, identical repeat, two held-out paraphrases). Its [source summary](../results/held_out_confirmatory/summary.json) supports the augmentation trade-off.

The **anchored-language development** round used States 23–32 and the two development paraphrases. It failed its frozen gate. A diagnostic pass then analyzed existing rollouts; it did not collect new behavior. The early action difference is a measured fact. The proposed chain from action difference to later task failure is an interpretation of observed trajectories, not an isolated causal test. [Diagnostic data and caveats](results.md#action-and-trajectory-diagnostics).

The paired experiment added another 1,000 updates to each arm from the same frozen `original_language` step-1000 checkpoint. The [paired schedule](../results/paired_training/schedule.json) and training logs were audited step by step for physical frame, paraphrase, noise and flow-time identity. The locally stored [paired control audit](../results/paired_training/paired_control_audit.json) records that check. Both arms were evaluated only on the development split for model selection. `paired_flow_consistency` failed its registered gate because H=0.6828 was below `paired_bc` H=0.7500.

The final comparison used newly frozen `paraphrase_final_01` and `paraphrase_final_02`; neither was selected from final performance. Its originally planned State IDs 33–52 exceeded the active official Task-0 asset's 0–49 index range. After 51 valid `original_language` rollouts on 33–49, State 50 failed before policy inference. An asset/index audit verified that those 51 attempts used the same state source and exact rows. **The protocol was amended to States 33–49 without inspecting intermediate success outcomes.** Historical v1–v3 freezes and partial attempts were retained; [v4 freeze](../configs/final_confirmatory/freeze_final_protocol_v4.json) records the amendment and compatibility proof.

## Split and outcome discipline

| Split | State IDs | Language role | Use |
| --- | --- | --- | --- |
| Initial exploration | 0–2 | Early pilot/repeat wordings | Pipeline development |
| First confirmatory | 3–22 | Canonical, repeat, two then-held-out paraphrases | Consumed first test |
| Development/model selection | 23–32 | Canonical, `dev_01`, `dev_02` | Anchored and paired arm decisions |
| Final v4 | 33–49 | Canonical and two newly frozen final paraphrases | Final comparison only |
| Invalid in active asset | 50–52 | None | Never included in a valid result |

`goal_ever` is the primary outcome: the task goal must become true at least once within the 280-step horizon. `goal_final` is retained as a secondary persistence diagnostic. The final analysis resamples 17 **state** clusters 10,000 times, keeping all three model × three prompt outcomes together for each sampled state. See [full tables](results.md) and [final protocol](../configs/final_confirmatory/final_matrix_v4.json).

## Source records

- [First confirmatory summary](../results/held_out_confirmatory/summary.json)
- [Anchored development summary](../results/development_selection/anchored_summary.json)
- [Paired development summary](../results/development_selection/paired_summary.json)
- [Action/trajectory diagnostic summary](../results/language_divergence/summary.json)
- [Final v4 summary](../results/final_confirmatory_v4/summary.json) and [per-state outcomes](../results/final_confirmatory_v4/per_state_raw_outcomes.csv)
- [Paired training config](../configs/paired_training.json), [training instruction catalog](../configs/instructions/task0_train.jsonl), and [final frozen manifest](../configs/final_confirmatory/frozen_model_manifest_v4.json)
