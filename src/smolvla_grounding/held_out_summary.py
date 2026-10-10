"""Comprehensive statistical summarizer for confirmatory held-out language robustness.

Primary success metric: LATCHED SUCCESS / GOAL_EVER.
(An episode is successful iff the task goal is reached at ANY timestep during rollout).
Secondary diagnostic: GOAL_FINAL (retained for persistence analysis).

Implements all 9 required metrics:
1. Model success rate on original instruction (goal_ever)
2. Model success rate on each unseen paraphrase (goal_ever)
3. Pooled unseen-paraphrase success rate (goal_ever)
4. Robustness gap: pooled paraphrase - original (goal_ever)
5. Paired difference between augmented_language and original_language on unseen paraphrases (goal_ever)
6. Original-task retention difference between the two fine-tuned models (goal_ever)
7. Original-vs-repeat disagreement rate (nondeterminism diagnostic, repeat never in success denominator)
8. Per-state raw outcome matrix (both goal_ever and goal_final recorded)
9. 95% confidence intervals via paired cluster bootstrap over initial states (B=10000)
"""

from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np


def compute_paired_cluster_bootstrap(states_data, models, num_replicates=10000, ci_level=0.95, seed=42):
    """Resample initial states as clusters so all model/language outcomes per state stay strictly paired."""
    rng = np.random.default_rng(seed)
    unique_states = sorted(list(states_data.keys()))
    n_states = len(unique_states)
    if n_states == 0:
        return {}

    alpha_low = ((1.0 - ci_level) / 2.0) * 100
    alpha_high = (1.0 - (1.0 - ci_level) / 2.0) * 100

    boot_metrics = defaultdict(list)

    for _ in range(num_replicates):
        resampled_states = rng.choice(unique_states, size=n_states, replace=True)

        # Structure: m_orig, m_p1, m_p2, m_rep for each model (driven by GOAL_EVER)
        sums = {m: {"orig": 0, "p1": 0, "p2": 0, "rep": 0, "disag": 0} for m in models}

        for s in resampled_states:
            s_dict = states_data[s]
            for m in models:
                m_data = s_dict.get(m, {})
                # Note: outcomes here are GOAL_EVER
                o = m_data.get("original", 0)
                r = m_data.get("original_repeat", 0)
                p1 = m_data.get("paraphrase_heldout_01", 0)
                p2 = m_data.get("paraphrase_heldout_02", 0)

                sums[m]["orig"] += o
                sums[m]["rep"] += r
                sums[m]["p1"] += p1
                sums[m]["p2"] += p2
                if o != r:
                    sums[m]["disag"] += 1

        # Calculate metrics on this cluster-resampled bootstrap sample
        rates = {}
        for m in models:
            sr_orig = sums[m]["orig"] / n_states
            sr_p1 = sums[m]["p1"] / n_states
            sr_p2 = sums[m]["p2"] / n_states
            # Pooled paraphrase: over the 2*n_states paraphrase opportunities
            sr_pool_p = (sums[m]["p1"] + sums[m]["p2"]) / (2 * n_states)
            gap = sr_pool_p - sr_orig
            disag = sums[m]["disag"] / n_states

            rates[m] = {
                "orig": sr_orig, "p1": sr_p1, "p2": sr_p2,
                "pool_p": sr_pool_p, "gap": gap, "disag": disag
            }
            boot_metrics[f"{m}_orig"].append(sr_orig)
            boot_metrics[f"{m}_p1"].append(sr_p1)
            boot_metrics[f"{m}_p2"].append(sr_p2)
            boot_metrics[f"{m}_pool_p"].append(sr_pool_p)
            boot_metrics[f"{m}_gap"].append(gap)
            boot_metrics[f"{m}_disag"].append(disag)

        # Comparative paired metrics (strictly paired on identical resampled state clusters)
        if "augmented_language" in models and "original_language" in models:
            paired_diff_p = rates["augmented_language"]["pool_p"] - rates["original_language"]["pool_p"]
            orig_retention_diff = rates["augmented_language"]["orig"] - rates["original_language"]["orig"]
            boot_metrics["paired_paraphrase_diff"].append(paired_diff_p)
            boot_metrics["paired_original_retention_diff"].append(orig_retention_diff)

    ci_summary = {}
    for k, vals in boot_metrics.items():
        arr = np.array(vals)
        ci_summary[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_95": [float(np.percentile(arr, alpha_low)), float(np.percentile(arr, alpha_high))]
        }
    return ci_summary


def parse_and_aggregate_records(episodes, models):
    """Aggregate raw episode dictionaries into structured state matrices and compute all 9 primary metrics."""
    # PRIMARY data driven by goal_ever
    states_data_ever = defaultdict(lambda: defaultdict(dict))
    # Secondary persistence data driven by goal_final
    states_data_final = defaultdict(lambda: defaultdict(dict))

    for ep in episodes:
        s = ep["initial_state_index"]
        m = ep["model"]
        inst = ep["instruction_id"]
        states_data_ever[s][m][inst] = int(ep["goal_ever"])
        states_data_final[s][m][inst] = int(ep["goal_final"])

    unique_states = sorted(list(states_data_ever.keys()))
    n_states = len(unique_states)

    # Compute primary point estimates driven by GOAL_EVER
    point_estimates_primary = {}
    persistence_diagnostics = {}

    for m in models:
        # GOAL_EVER primary series
        orig_ever = [states_data_ever[s][m].get("original", 0) for s in unique_states]
        rep_ever = [states_data_ever[s][m].get("original_repeat", 0) for s in unique_states]
        p1_ever = [states_data_ever[s][m].get("paraphrase_heldout_01", 0) for s in unique_states]
        p2_ever = [states_data_ever[s][m].get("paraphrase_heldout_02", 0) for s in unique_states]

        sr_orig_ever = float(np.mean(orig_ever)) if n_states else 0.0
        sr_rep_ever = float(np.mean(rep_ever)) if n_states else 0.0
        sr_p1_ever = float(np.mean(p1_ever)) if n_states else 0.0
        sr_p2_ever = float(np.mean(p2_ever)) if n_states else 0.0
        sr_pool_p_ever = float(np.mean(p1_ever + p2_ever)) if n_states else 0.0
        gap_ever = sr_pool_p_ever - sr_orig_ever

        disag_count = sum(1 for o, r in zip(orig_ever, rep_ever) if o != r)
        disag_rate = float(disag_count / n_states) if n_states else 0.0

        point_estimates_primary[m] = {
            "metric_type": "primary_latched_goal_ever",
            "success_rate_original": sr_orig_ever,
            "success_rate_original_count": f"{sum(orig_ever)}/{n_states}",
            "success_rate_paraphrase_01": sr_p1_ever,
            "success_rate_paraphrase_01_count": f"{sum(p1_ever)}/{n_states}",
            "success_rate_paraphrase_02": sr_p2_ever,
            "success_rate_paraphrase_02_count": f"{sum(p2_ever)}/{n_states}",
            "pooled_unseen_paraphrase_success_rate": sr_pool_p_ever,
            "pooled_unseen_paraphrase_count": f"{sum(p1_ever)+sum(p2_ever)}/{2*n_states}",
            "robustness_gap": gap_ever,
            "repeat_diagnostic_success_rate": sr_rep_ever,
            "repeat_disagreement_count": disag_count,
            "repeat_disagreement_rate": disag_rate
        }

        # Secondary GOAL_FINAL persistence series
        orig_final = [states_data_final[s][m].get("original", 0) for s in unique_states]
        rep_final = [states_data_final[s][m].get("original_repeat", 0) for s in unique_states]
        p1_final = [states_data_final[s][m].get("paraphrase_heldout_01", 0) for s in unique_states]
        p2_final = [states_data_final[s][m].get("paraphrase_heldout_02", 0) for s in unique_states]

        persistence_diagnostics[m] = {
            "metric_type": "secondary_persistence_goal_final",
            "final_step_success_rate_original": float(np.mean(orig_final)) if n_states else 0.0,
            "final_step_success_rate_original_count": f"{sum(orig_final)}/{n_states}",
            "final_step_pooled_paraphrase_rate": float(np.mean(p1_final + p2_final)) if n_states else 0.0,
            "final_step_pooled_paraphrase_count": f"{sum(p1_final)+sum(p2_final)}/{2*n_states}",
            "final_step_repeat_disagreement_count": sum(1 for o, r in zip(orig_final, rep_final) if o != r)
        }

    # Comparative point estimates driven by GOAL_EVER
    comparative = {}
    if "augmented_language" in models and "original_language" in models:
        aug_p = point_estimates_primary["augmented_language"]["pooled_unseen_paraphrase_success_rate"]
        orig_p = point_estimates_primary["original_language"]["pooled_unseen_paraphrase_success_rate"]
        paired_para_diff = aug_p - orig_p

        aug_o = point_estimates_primary["augmented_language"]["success_rate_original"]
        orig_o = point_estimates_primary["original_language"]["success_rate_original"]
        orig_task_retention_diff = aug_o - orig_o

        comparative["paired_difference_augmented_minus_original_on_paraphrases_ever"] = paired_para_diff
        comparative["original_task_retention_difference_augmented_minus_original_ever"] = orig_task_retention_diff

    # Compute paired cluster bootstrap 95% CIs on GOAL_EVER
    ci_results = compute_paired_cluster_bootstrap(states_data_ever, models, num_replicates=10000, ci_level=0.95, seed=42)

    # Raw per-state outcomes recording both ever and final
    raw_per_state = []
    for s in unique_states:
        row = {"state_index": s}
        for m in models:
            for inst in ["original", "original_repeat", "paraphrase_heldout_01", "paraphrase_heldout_02"]:
                row[f"{m}_{inst}_ever"] = states_data_ever[s][m].get(inst, None)
                row[f"{m}_{inst}_final"] = states_data_final[s][m].get(inst, None)
        raw_per_state.append(row)

    return {
        "evaluated_states_count": n_states,
        "evaluated_states": unique_states,
        "primary_point_estimates_goal_ever": point_estimates_primary,
        "primary_comparative_metrics_goal_ever": comparative,
        "paired_cluster_bootstrap_95ci_goal_ever": ci_results,
        "secondary_persistence_diagnostics_goal_final": persistence_diagnostics,
        "per_state_raw_outcomes": raw_per_state
    }


def summarize_held_out_run(run_dir):
    run_path = Path(run_dir)
    meta_file = run_path / "experiment_meta.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"Missing metadata in {run_path}")

    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    tasks = meta["matrix_config"]["tasks"]
    models = meta["matrix_config"]["models"]

    episodes = []
    for model_id in models:
        model_dir = run_path / model_id
        if not model_dir.exists():
            continue
        for task_cfg in tasks:
            task_id = task_cfg["task_id"]
            task_dir = model_dir / f"task_{task_id:02d}"
            if not task_dir.exists():
                continue
            for trial_dir in task_dir.iterdir():
                if not trial_dir.is_dir():
                    continue
                attempt_dir = trial_dir / "attempt_001"
                complete_file = attempt_dir / "complete.json"
                episode_file = attempt_dir / "episode.json"
                if complete_file.exists() and episode_file.exists():
                    ep_data = json.loads(episode_file.read_text(encoding="utf-8"))
                    episodes.append(ep_data)

    results = parse_and_aggregate_records(episodes, models)

    primary_report = {
        "experiment_name": meta["run_name"],
        "task_id": 0,
        "horizon_steps": 280,
        "total_rollouts_recorded": len(episodes),
        **results
    }

    # Write files
    (run_path / "primary_confirmatory_report.json").write_text(json.dumps(primary_report, indent=2), encoding="utf-8")
    (run_path / "summary.json").write_text(json.dumps(primary_report, indent=2), encoding="utf-8")

    # CSV export
    with open(run_path / "per_state_raw_outcomes.csv", "w", newline="", encoding="utf-8") as f:
        if results["per_state_raw_outcomes"]:
            writer = csv.DictWriter(f, fieldnames=list(results["per_state_raw_outcomes"][0].keys()))
            writer.writeheader()
            writer.writerows(results["per_state_raw_outcomes"])

    print(f"\n[Confirmatory Report Generated] Primary metrics driven by goal_ever. Saved to {run_path / 'primary_confirmatory_report.json'}")
    return primary_report
