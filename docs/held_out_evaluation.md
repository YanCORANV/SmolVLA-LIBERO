# Frozen Confirmatory Language Robustness Evaluation Protocol

## 1. Primary Confirmatory Experiment Specification

To strictly isolate the effect of language augmentation from multi-task transfer and catastrophic forgetting (since both fine-tuned models were adapted solely on Task 0), the **Primary Confirmatory Experiment** evaluates exclusively on **LIBERO-Spatial Task 0**.

- **Models Compared (3 Arms)**:
  1. `untuned_fp32`: Official pretrained SmolVLA baseline (zero fine-tuning).
  2. `original_language`: 1000-update control fine-tuned strictly on Task 0 canonical instructions.
  3. `augmented_language`: 1000-update candidate fine-tuned on Task 0 canonical + 4 paraphrase variants.
- **Untouched State Set**: **20 official Task 0 initial states**:
  $$\text{States } 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22$$
  *Development states 0, 1, and 2 are strictly excluded and will trigger an immediate pre-flight abort if present.*
- **Language Conditions (4 per state)**:
  1. `original`: Canonical official instruction.
  2. `original_repeat`: Identical instruction to diagnose trajectory & simulation non-determinism.
  3. `paraphrase_heldout_01`: Novel syntax/lexical variant (`transfer into the plate the black bowl that sits between the plate and the ramekin`).
  4. `paraphrase_heldout_02`: Novel landmark order/verb variant (`take the black bowl located between the ramekin and the plate, and set it down on the plate`).
- **Total Rollouts**: $20 \text{ states} \times 4 \text{ language conditions} \times 3 \text{ models} = \mathbf{240 \text{ rollouts}}$.
- **Execution Constraints**: Policy seed = 0, Env seed = 0, Horizon = 280 steps (20 Hz), Worker = 1 (Serial execution mandatory).

---

## 2. Statistical Metrics & Paired Cluster Bootstrap

### Core Rules
`original_repeat` serves exclusively as a **nondeterminism diagnostic**. It is strictly prohibited from being pooled into the success rate denominator.

### Metric Definitions (Over States $S=\{3, \dots, 22\}$, $|S|=20$)
Let $Y_{m, c}(s) \in \{0, 1\}$ denote the terminal outcome (`goal_final`) for model $m$ under condition $c$ at state $s$:

1. **Original Success Rate**:
   $$\text{SR}_{m, \text{orig}} = \frac{1}{|S|} \sum_{s \in S} Y_{m, \text{orig}}(s)$$
2. **Per-Paraphrase Success Rate**:
   $$\text{SR}_{m, p_1} = \frac{1}{|S|} \sum_{s \in S} Y_{m, p_1}(s), \quad \text{SR}_{m, p_2} = \frac{1}{|S|} \sum_{s \in S} Y_{m, p_2}(s)$$
3. **Pooled Unseen-Paraphrase Success Rate**:
   $$\text{SR}_{m, \text{pool\_p}} = \frac{1}{2|S|} \sum_{s \in S} (Y_{m, p_1}(s) + Y_{m, p_2}(s))$$
4. **Robustness Gap**:
   $$\text{Gap}_m = \text{SR}_{m, \text{pool\_p}} - \text{SR}_{m, \text{orig}}$$
5. **Paired Paraphrase Gain (Augmented vs. Original-Tuned)**:
   $$\Delta_{\text{para}} = \text{SR}_{\text{aug}, \text{pool\_p}} - \text{SR}_{\text{orig\_lang}, \text{pool\_p}}$$
6. **Original-Task Retention Difference**:
   $$\Delta_{\text{orig}} = \text{SR}_{\text{aug}, \text{orig}} - \text{SR}_{\text{orig\_lang}, \text{orig}}$$
   *(Guarantees robustness gain does not conceal canonical task degradation).*
7. **Repeat Disagreement Rate (Diagnostic)**:
   $$\text{DisagRate}_m = \frac{1}{|S|} \sum_{s \in S} \mathbb{I}(Y_{m, \text{orig}}(s) \neq Y_{m, \text{rep}}(s))$$
8. **Per-State Raw Matrix**: Full binary outcome table across all $(s, m, c)$.
9. **95% Confidence Intervals via Paired Cluster Bootstrap**:
   - Resamples states $s \in S$ with replacement as clusters ($B=10,000$).
   - Retains all paired model/language outcomes within each sampled state.
   - Computes empirical 2.5% and 97.5% percentiles for all point and comparative metrics.

---

## 3. Pre-Flight Validation Checks

The evaluator executes automated sanity assertions before running any simulator step:
1. `confirmatory_states.intersection({0, 1, 2}) == empty` (Aborts if development states are touched).
2. `paraphrases.intersection(training_catalog) == empty` (Aborts if held-out prompt leaked into training).
3. Manifest checkpoint directories and parameter evidence files are verified.
4. Identical state/language evaluation matrix confirmed across all 3 models.

---

## 4. Exploratory Cross-Task Retention (Secondary)

Task 1 and Task 2 support code is preserved in [`configs/held_out_evaluation/exploratory_cross_task_matrix.json`](../configs/held_out_evaluation/exploratory_cross_task_matrix.json).
- **Role**: Exploratory analysis of spatial transfer / forgetting only.
- **Isolation**: Prohibited from contributing to the primary language robustness headline claims or statistical pooling.
