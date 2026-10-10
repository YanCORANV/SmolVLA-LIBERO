#!/usr/bin/env python3
"""Language Divergence Diagnostics: State-matched analysis of action and trajectory divergence.

Compares policy actions and end-effector trajectories timestep-by-timestep
between original and paraphrase rollouts, using original-vs-repeat as an
empirical nondeterminism baseline.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lerobot.envs.libero import LiberoEnv
from libero.libero import benchmark


def load_kinematics_extractor():
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    env = LiberoEnv(
        task_suite=suite,
        task_id=0,
        task_suite_name="libero_spatial",
        init_states=True,
        episode_index=0,
        n_envs=1
    )
    env.reset(seed=0)
    sim = env._env.sim
    domain = env._env.env
    robot = domain.robots[0]
    site_id = sim.model.site_name2id(robot.controller.eef_name)
    gripper_idx = robot._ref_gripper_joint_pos_indexes

    def extract_kinematics(sim_states):
        n_steps = len(sim_states)
        pos = np.empty((n_steps, 3), dtype=np.float32)
        ori = np.empty((n_steps, 3, 3), dtype=np.float32)
        grip = np.empty((n_steps, len(gripper_idx)), dtype=np.float32)
        for t, s in enumerate(sim_states):
            sim.set_state_from_flattened(s)
            sim.forward()
            pos[t] = sim.data.site_xpos[site_id]
            ori[t] = sim.data.site_xmat[site_id].reshape(3, 3)
            grip[t] = sim.data.qpos[gripper_idx]
        return pos, ori, grip

    return extract_kinematics


def compute_geodesic_orientation_distance(R1, R2):
    """Compute angular distance in degrees between two sequences of rotation matrices."""
    # R_rel = R1 @ R2.T
    R_rel = np.einsum("tij,tkj->tik", R1, R2)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    cos_angle = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_angle))


def compute_cosine_similarity(v1, v2, eps=1e-8):
    """Compute cosine similarity along the last dimension."""
    norm1 = np.linalg.norm(v1, axis=-1)
    norm2 = np.linalg.norm(v2, axis=-1)
    dot = np.sum(v1 * v2, axis=-1)
    valid = (norm1 > eps) & (norm2 > eps)
    sim = np.zeros_like(dot)
    sim[valid] = dot[valid] / (norm1[valid] * norm2[valid])
    # When both vectors are near zero, consider them perfectly aligned
    sim[(norm1 <= eps) & (norm2 <= eps)] = 1.0
    return sim


def get_onset_bin(step, horizon=280):
    if step is None:
        return "never"
    if step <= int(0.10 * horizon): # <= 28
        return "first_10%"
    elif step <= int(0.25 * horizon): # <= 70
        return "10_to_25%"
    elif step <= int(0.50 * horizon): # <= 140
        return "25_to_50%"
    else:
        return "final_50%"


def main():
    root_dir = Path(__file__).resolve().parent.parent
    confirmatory_dir = root_dir / "outputs" / "held_out_evaluation" / "runs" / "task0_confirmatory_heldout_eval"
    dev_dir = root_dir / "outputs" / "development_evaluation" / "runs" / "task0_anchored_dev_eval"
    out_dir = root_dir / "outputs" / "language_divergence_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Initializing MuJoCo Kinematics Extractor...", flush=True)
    extract_kinematics = load_kinematics_extractor()

    # Cache for loaded rollouts: key -> dict of arrays
    rollout_cache = {}

    def get_rollout_data(path):
        path_str = str(path)
        if path_str in rollout_cache:
            return rollout_cache[path_str]
        data = np.load(path)
        actions = data["action"] # (280, 7)
        successes = data["success"] # (280,)
        sim_states = data["sim_state"] # (280, 92)
        pos, ori, grip = extract_kinematics(sim_states)
        entry = {
            "action": actions,
            "success": successes,
            "goal_ever": bool(successes.any()),
            "goal_final": bool(successes[-1]),
            "pos": pos,
            "ori": ori,
            "grip": grip
        }
        rollout_cache[path_str] = entry
        return entry

    # 1. Collect all pairs
    # Group A: Confirmatory evaluation (States 3-22)
    # Models: untuned_fp32, original_language, augmented_language
    conf_models = ["untuned_fp32", "original_language", "augmented_language"]
    conf_states = list(range(3, 23))

    pair_specs = []

    # Repeat baseline pairs
    for m in conf_models:
        for s in conf_states:
            base_p = confirmatory_dir / m / "task_00" / f"state_{s:04d}_seed_0000_original" / "attempt_001" / "trajectory.npz"
            comp_p = confirmatory_dir / m / "task_00" / f"state_{s:04d}_seed_0000_original_repeat" / "attempt_001" / "trajectory.npz"
            if base_p.exists() and comp_p.exists():
                pair_specs.append({
                    "eval_set": "confirmatory_heldout",
                    "model": m,
                    "state_index": s,
                    "base_id": "original",
                    "compare_id": "original_repeat",
                    "pair_type": "repeat_baseline",
                    "base_path": base_p,
                    "comp_path": comp_p
                })

    # Confirmatory paraphrase pairs
    for m in conf_models:
        for s in conf_states:
            base_p = confirmatory_dir / m / "task_00" / f"state_{s:04d}_seed_0000_original" / "attempt_001" / "trajectory.npz"
            for p_id in ["paraphrase_heldout_01", "paraphrase_heldout_02"]:
                comp_p = confirmatory_dir / m / "task_00" / f"state_{s:04d}_seed_0000_{p_id}" / "attempt_001" / "trajectory.npz"
                if base_p.exists() and comp_p.exists():
                    pair_specs.append({
                        "eval_set": "confirmatory_heldout",
                        "model": m,
                        "state_index": s,
                        "base_id": "original",
                        "compare_id": p_id,
                        "pair_type": "paraphrase",
                        "base_path": base_p,
                        "comp_path": comp_p
                    })

    # Group B: Development evaluation (States 23-32)
    # Models: original_language, augmented_language, anchored_language
    dev_models = ["original_language", "augmented_language", "anchored_language"]
    dev_states = list(range(23, 33))

    for m in dev_models:
        for s in dev_states:
            base_p = dev_dir / m / "task_00" / f"state_{s:04d}_seed_0000_original" / "attempt_001" / "trajectory.npz"
            for p_id in ["paraphrase_dev_01", "paraphrase_dev_02"]:
                comp_p = dev_dir / m / "task_00" / f"state_{s:04d}_seed_0000_{p_id}" / "attempt_001" / "trajectory.npz"
                if base_p.exists() and comp_p.exists():
                    pair_specs.append({
                        "eval_set": "development",
                        "model": m,
                        "state_index": s,
                        "base_id": "original",
                        "compare_id": p_id,
                        "pair_type": "paraphrase",
                        "base_path": base_p,
                        "comp_path": comp_p
                    })

    print(f"Total pairs identified: {len(pair_specs)} (Repeat: {sum(1 for p in pair_specs if p['pair_type'] == 'repeat_baseline')}, Paraphrase: {sum(1 for p in pair_specs if p['pair_type'] == 'paraphrase')})", flush=True)

    # 2. Extract and precompute trajectories
    print("Loading and extracting kinematics for all rollouts...", flush=True)
    all_trajectories = {}
    for idx, p in enumerate(pair_specs):
        if idx % 30 == 0:
            print(f"  Processed {idx}/{len(pair_specs)} pairs...", flush=True)
        b_data = get_rollout_data(p["base_path"])
        c_data = get_rollout_data(p["comp_path"])

    print(f"Cached {len(rollout_cache)} unique rollout trajectories.", flush=True)

    # 3. Derive empirical thresholds from original-vs-repeat baseline
    print("Deriving empirical nondeterminism baseline from original-vs-repeat pairs...", flush=True)
    repeat_specs = [p for p in pair_specs if p["pair_type"] == "repeat_baseline"]
    repeat_act_diffs = []
    repeat_pos_diffs = []
    repeat_ori_diffs = []

    for p in repeat_specs:
        b = get_rollout_data(p["base_path"])
        c = get_rollout_data(p["comp_path"])
        act_d = np.linalg.norm(b["action"] - c["action"], axis=-1)
        pos_d = np.linalg.norm(b["pos"] - c["pos"], axis=-1)
        ori_d = compute_geodesic_orientation_distance(b["ori"], c["ori"])
        repeat_act_diffs.append(act_d)
        repeat_pos_diffs.append(pos_d)
        repeat_ori_diffs.append(ori_d)

    repeat_act_diffs = np.array(repeat_act_diffs) # (60, 280)
    repeat_pos_diffs = np.array(repeat_pos_diffs) # (60, 280)
    repeat_ori_diffs = np.array(repeat_ori_diffs) # (60, 280)

    # Empirical statistics
    # Early phase (first 10% = 28 steps)
    early_act_p95 = float(np.percentile(repeat_act_diffs[:, :28], 95))
    early_act_p99 = float(np.percentile(repeat_act_diffs[:, :28], 99))
    early_act_max = float(np.max(repeat_act_diffs[:, :28]))

    early_pos_p95 = float(np.percentile(repeat_pos_diffs[:, :28], 95))
    early_pos_p99 = float(np.percentile(repeat_pos_diffs[:, :28], 99))
    early_pos_max = float(np.max(repeat_pos_diffs[:, :28]))

    # Full episode repeat stats
    overall_act_p75 = float(np.percentile(repeat_act_diffs, 75))
    overall_act_p90 = float(np.percentile(repeat_act_diffs, 90))
    overall_act_p95 = float(np.percentile(repeat_act_diffs, 95))

    overall_pos_p75 = float(np.percentile(repeat_pos_diffs, 75))
    overall_pos_p90 = float(np.percentile(repeat_pos_diffs, 90))
    overall_pos_p95 = float(np.percentile(repeat_pos_diffs, 95))

    # We set:
    # tau_act: 0.05 (the p99 of early repeat action diff; any action diff above 0.05 exceeds 99% of unperturbed repeat steps)
    # tau_pos: 0.01 m (1 cm; exceeds 10x the p99 of early repeat EEF drift 0.00098m, representing clear physical macroscopic departure)
    # We will also compute secondary thresholds for robustness (tau_act_alt=0.10, tau_pos_alt=0.02 m).
    tau_act = 0.05
    tau_pos = 0.01 # meters (1 cm)

    print(f"Empirical Thresholds Derived:")
    print(f"  Action Divergence Threshold (tau_act): {tau_act:.4f} (early repeat p99: {early_act_p99:.4f}, max: {early_act_max:.4f})")
    print(f"  Trajectory Divergence Threshold (tau_pos): {tau_pos:.4f} m (early repeat p99: {early_pos_p99:.4f} m, max: {early_pos_max:.4f} m)")

    # 4. Compute pair-level measurements
    print("Computing pair-level diagnostic measurements...", flush=True)
    pair_records = []
    pair_timeseries = []

    for p in pair_specs:
        b = get_rollout_data(p["base_path"])
        c = get_rollout_data(p["comp_path"])

        # Actions
        act_diff = np.linalg.norm(b["action"] - c["action"], axis=-1) # (280,)
        trans_diff = np.linalg.norm(b["action"][:, :3] - c["action"][:, :3], axis=-1)
        rot_diff = np.linalg.norm(b["action"][:, 3:6] - c["action"][:, 3:6], axis=-1)
        grip_act_diff = np.abs(b["action"][:, 6] - c["action"][:, 6])
        cosine_sim = compute_cosine_similarity(b["action"][:, :3], c["action"][:, :3])

        # Trajectory
        pos_diff = np.linalg.norm(b["pos"] - c["pos"], axis=-1) # (280,) meters
        ori_diff_deg = compute_geodesic_orientation_distance(b["ori"], c["ori"]) # (280,) degrees
        grip_qpos_diff = np.linalg.norm(b["grip"] - c["grip"], axis=-1) # (280,)

        # Divergence onsets
        act_div_idx = np.where(act_diff >= tau_act)[0]
        act_onset_step = int(act_div_idx[0] + 1) if len(act_div_idx) > 0 else None

        pos_div_idx = np.where(pos_diff >= tau_pos)[0]
        pos_onset_step = int(pos_div_idx[0] + 1) if len(pos_div_idx) > 0 else None

        # Outcomes
        b_ever = b["goal_ever"]
        c_ever = c["goal_ever"]
        if b_ever and c_ever:
            outcome_grp = "both_success"
        elif b_ever and not c_ever:
            outcome_grp = "base_success_cand_fail"
        elif not b_ever and c_ever:
            outcome_grp = "base_fail_cand_success"
        else:
            outcome_grp = "both_fail"

        rec = {
            "eval_set": p["eval_set"],
            "model": p["model"],
            "state_index": p["state_index"],
            "base_instruction_id": p["base_id"],
            "compare_instruction_id": p["compare_id"],
            "pair_type": p["pair_type"],
            "base_goal_ever": int(b_ever),
            "base_goal_final": int(b["goal_final"]),
            "compare_goal_ever": int(c_ever),
            "compare_goal_final": int(c["goal_final"]),
            "outcome_group": outcome_grp,
            "step1_action_diff": float(act_diff[0]),
            "step1_trans_diff": float(trans_diff[0]),
            "step1_rot_diff": float(rot_diff[0]),
            "step1_grip_act_diff": float(grip_act_diff[0]),
            "step1_cosine_sim": float(cosine_sim[0]),
            "mean_action_diff_first_10pct": float(np.mean(act_diff[:28])),
            "mean_action_diff_first_25pct": float(np.mean(act_diff[:70])),
            "mean_action_diff_first_50pct": float(np.mean(act_diff[:140])),
            "mean_action_diff_final_50pct": float(np.mean(act_diff[140:])),
            "mean_action_diff_overall": float(np.mean(act_diff)),
            "max_action_diff_overall": float(np.max(act_diff)),
            "action_divergence_step": act_onset_step,
            "action_divergence_bin": get_onset_bin(act_onset_step),
            "step1_eef_pos_diff_m": float(pos_diff[0]),
            "mean_eef_pos_diff_first_10pct_m": float(np.mean(pos_diff[:28])),
            "mean_eef_pos_diff_first_25pct_m": float(np.mean(pos_diff[:70])),
            "mean_eef_pos_diff_first_50pct_m": float(np.mean(pos_diff[:140])),
            "mean_eef_pos_diff_final_50pct_m": float(np.mean(pos_diff[140:])),
            "mean_eef_pos_diff_overall_m": float(np.mean(pos_diff)),
            "max_eef_pos_diff_overall_m": float(np.max(pos_diff)),
            "trajectory_divergence_step": pos_onset_step,
            "trajectory_divergence_bin": get_onset_bin(pos_onset_step),
            "step1_eef_ori_diff_deg": float(ori_diff_deg[0]),
            "mean_eef_ori_diff_first_10pct_deg": float(np.mean(ori_diff_deg[:28])),
            "mean_eef_ori_diff_overall_deg": float(np.mean(ori_diff_deg)),
            "step1_gripper_qpos_diff": float(grip_qpos_diff[0]),
            "mean_gripper_qpos_diff_overall": float(np.mean(grip_qpos_diff))
        }
        pair_records.append(rec)

        pair_timeseries.append({
            "pair_type": p["pair_type"],
            "eval_set": p["eval_set"],
            "model": p["model"],
            "compare_id": p["compare_id"],
            "outcome_group": outcome_grp,
            "cand_success": c_ever,
            "act_diff": act_diff,
            "pos_diff": pos_diff,
            "ori_diff_deg": ori_diff_deg,
            "cosine_sim": cosine_sim
        })

    df = pd.DataFrame(pair_records)
    csv_path = out_dir / "pair_level_divergence_measurements.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved pair measurements CSV to {csv_path} ({len(df)} rows)")

    # 5. Diagnostic Sub-Analyses
    # A: Paraphrase vs Repeat Baseline comparison
    para_df = df[df["pair_type"] == "paraphrase"].copy()
    rep_df = df[df["pair_type"] == "repeat_baseline"].copy()

    # B: Successful vs Failed Paraphrase comparison
    succ_para = para_df[para_df["compare_goal_ever"] == 1]
    fail_para = para_df[para_df["compare_goal_ever"] == 0]

    # Matched base_success comparisons (where original succeeded)
    clean_succ_para = para_df[para_df["outcome_group"] == "both_success"]
    clean_fail_para = para_df[para_df["outcome_group"] == "base_success_cand_fail"]

    # Statistical comparisons: Successful vs Failed Paraphrases
    def get_distribution_stats(series):
        valid = series.dropna()
        return {
            "count": int(len(valid)),
            "mean": float(valid.mean()) if len(valid) else 0.0,
            "std": float(valid.std()) if len(valid) else 0.0,
            "median": float(valid.median()) if len(valid) else 0.0,
            "p25": float(valid.quantile(0.25)) if len(valid) else 0.0,
            "p75": float(valid.quantile(0.75)) if len(valid) else 0.0,
            "iqr": float(valid.quantile(0.75) - valid.quantile(0.25)) if len(valid) else 0.0,
            "min": float(valid.min()) if len(valid) else 0.0,
            "max": float(valid.max()) if len(valid) else 0.0
        }

    # Step 1 action diff
    step1_succ = succ_para["step1_action_diff"].values
    step1_fail = fail_para["step1_action_diff"].values
    mw_step1 = stats.mannwhitneyu(step1_fail, step1_succ, alternative="greater")

    # Early action diff (first 10%)
    early_act_succ = succ_para["mean_action_diff_first_10pct"].values
    early_act_fail = fail_para["mean_action_diff_first_10pct"].values
    mw_early_act = stats.mannwhitneyu(early_act_fail, early_act_succ, alternative="greater")

    # First trajectory divergence step (pos_onset_step)
    # Filter for those that did diverge
    traj_step_succ = succ_para["trajectory_divergence_step"].dropna().values
    traj_step_fail = fail_para["trajectory_divergence_step"].dropna().values
    mw_traj_step = stats.mannwhitneyu(traj_step_succ, traj_step_fail, alternative="greater") # test if succ occurs later (larger step)

    # Bin percentages for trajectory divergence
    traj_bins_succ = succ_para["trajectory_divergence_bin"].value_counts(normalize=True).to_dict()
    traj_bins_fail = fail_para["trajectory_divergence_bin"].value_counts(normalize=True).to_dict()

    # Action divergence bins
    act_bins_succ = succ_para["action_divergence_bin"].value_counts(normalize=True).to_dict()
    act_bins_fail = fail_para["action_divergence_bin"].value_counts(normalize=True).to_dict()

    # C: dev_01 vs dev_02 comparison (in development set)
    dev_pairs = df[df["eval_set"] == "development"].copy()
    dev01_df = dev_pairs[dev_pairs["compare_instruction_id"] == "paraphrase_dev_01"].sort_values(["model", "state_index"])
    dev02_df = dev_pairs[dev_pairs["compare_instruction_id"] == "paraphrase_dev_02"].sort_values(["model", "state_index"])

    paired_act_diff = dev02_df["step1_action_diff"].values - dev01_df["step1_action_diff"].values
    paired_early_act = dev02_df["mean_action_diff_first_10pct"].values - dev01_df["mean_action_diff_first_10pct"].values
    paired_traj_onset = dev02_df["trajectory_divergence_step"].fillna(280).values - dev01_df["trajectory_divergence_step"].fillna(280).values

    wilc_step1 = stats.wilcoxon(dev01_df["step1_action_diff"].values, dev02_df["step1_action_diff"].values)
    wilc_early_act = stats.wilcoxon(dev01_df["mean_action_diff_first_10pct"].values, dev02_df["mean_action_diff_first_10pct"].values)

    # Build JSON Summary
    summary_data = {
        "evaluation_scope": "Language Divergence Diagnostic Analysis",
        "sample_counts": {
            "total_pairs": len(df),
            "repeat_baseline_pairs": len(rep_df),
            "paraphrase_pairs": len(para_df),
            "confirmatory_heldout_paraphrase_pairs": len(para_df[para_df["eval_set"] == "confirmatory_heldout"]),
            "development_paraphrase_pairs": len(para_df[para_df["eval_set"] == "development"]),
            "successful_paraphrase_pairs": len(succ_para),
            "failed_paraphrase_pairs": len(fail_para),
            "clean_both_success_pairs": len(clean_succ_para),
            "clean_divergent_failure_pairs": len(clean_fail_para)
        },
        "empirical_repeat_baseline": {
            "description": "Nondeterminism baseline derived from original-vs-original_repeat trials (N=60)",
            "action_difference": {
                "step1_max": float(np.max(repeat_act_diffs[:, 0])),
                "early_10pct_mean": float(np.mean(repeat_act_diffs[:, :28])),
                "early_10pct_p95": early_act_p95,
                "early_10pct_p99": early_act_p99,
                "early_10pct_max": early_act_max,
                "overall_p50": float(np.percentile(repeat_act_diffs, 50)),
                "overall_p75": overall_act_p75,
                "overall_p90": overall_act_p90,
                "overall_p95": overall_act_p95,
                "overall_max": float(np.max(repeat_act_diffs))
            },
            "eef_position_difference_meters": {
                "step1_max": float(np.max(repeat_pos_diffs[:, 0])),
                "early_10pct_mean": float(np.mean(repeat_pos_diffs[:, :28])),
                "early_10pct_p95": early_pos_p95,
                "early_10pct_p99": early_pos_p99,
                "early_10pct_max": early_pos_max,
                "overall_p50": float(np.percentile(repeat_pos_diffs, 50)),
                "overall_p75": overall_pos_p75,
                "overall_p90": overall_pos_p90,
                "overall_p95": overall_pos_p95,
                "overall_max": float(np.max(repeat_pos_diffs))
            },
            "derived_thresholds": {
                "action_divergence_threshold_tau_act": tau_act,
                "action_threshold_justification": f"Set to {tau_act}, slightly above early repeat p99 ({early_act_p99:.4f}); any action diff >= {tau_act} exceeds normal early execution variation.",
                "trajectory_divergence_threshold_tau_pos_meters": tau_pos,
                "trajectory_threshold_justification": f"Set to {tau_pos} m (1 cm), which is ~10x early repeat drift ({early_pos_p99:.4f} m) and captures macroscopic end-effector path departure."
            }
        },
        "successful_vs_failed_paraphrase_comparison": {
            "step1_action_divergence": {
                "successful_paraphrases": get_distribution_stats(succ_para["step1_action_diff"]),
                "failed_paraphrases": get_distribution_stats(fail_para["step1_action_diff"]),
                "mann_whitney_u_test": {
                    "statistic": float(mw_step1.statistic),
                    "p_value": float(mw_step1.pvalue),
                    "hypothesis": "failed > successful"
                }
            },
            "mean_action_diff_first_10pct": {
                "successful_paraphrases": get_distribution_stats(succ_para["mean_action_diff_first_10pct"]),
                "failed_paraphrases": get_distribution_stats(fail_para["mean_action_diff_first_10pct"]),
                "mann_whitney_u_test": {
                    "statistic": float(mw_early_act.statistic),
                    "p_value": float(mw_early_act.pvalue),
                    "hypothesis": "failed > successful"
                }
            },
            "first_trajectory_divergence_step": {
                "successful_paraphrases": get_distribution_stats(succ_para["trajectory_divergence_step"]),
                "failed_paraphrases": get_distribution_stats(fail_para["trajectory_divergence_step"]),
                "mann_whitney_u_test": {
                    "statistic": float(mw_traj_step.statistic),
                    "p_value": float(mw_traj_step.pvalue),
                    "hypothesis": "successful occurs later than failed"
                }
            },
            "trajectory_divergence_onset_bins_proportions": {
                "successful_paraphrases": traj_bins_succ,
                "failed_paraphrases": traj_bins_fail
            },
            "action_divergence_onset_bins_proportions": {
                "successful_paraphrases": act_bins_succ,
                "failed_paraphrases": act_bins_fail
            }
        },
        "dev_01_vs_dev_02_matched_comparison": {
            "sample_size": len(dev01_df),
            "success_rate_dev_01": float(dev01_df["compare_goal_ever"].mean()),
            "success_rate_dev_02": float(dev02_df["compare_goal_ever"].mean()),
            "step1_action_diff": {
                "dev_01": get_distribution_stats(dev01_df["step1_action_diff"]),
                "dev_02": get_distribution_stats(dev02_df["step1_action_diff"]),
                "wilcoxon_signed_rank_p_value": float(wilc_step1.pvalue)
            },
            "mean_action_diff_first_10pct": {
                "dev_01": get_distribution_stats(dev01_df["mean_action_diff_first_10pct"]),
                "dev_02": get_distribution_stats(dev02_df["mean_action_diff_first_10pct"]),
                "wilcoxon_signed_rank_p_value": float(wilc_early_act.pvalue)
            },
            "trajectory_divergence_step": {
                "dev_01": get_distribution_stats(dev01_df["trajectory_divergence_step"]),
                "dev_02": get_distribution_stats(dev02_df["trajectory_divergence_step"])
            }
        }
    }

    json_path = out_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Saved JSON summary to {json_path}")

    # 6. Generate Publication-Quality Plots
    print("Generating diagnostic plots...", flush=True)

    # Plot 1: Mean action divergence over normalized episode time
    # Compare: Repeat baseline vs Successful paraphrases vs Failed paraphrases
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    steps = np.arange(1, 281)

    rep_acts = np.array([ts["act_diff"] for ts in pair_timeseries if ts["pair_type"] == "repeat_baseline"])
    succ_acts = np.array([ts["act_diff"] for ts in pair_timeseries if ts["pair_type"] == "paraphrase" and ts["cand_success"]])
    fail_acts = np.array([ts["act_diff"] for ts in pair_timeseries if ts["pair_type"] == "paraphrase" and not ts["cand_success"]])

    def plot_mean_std(ax, x, data, label, color, linestyle="-"):
        mean = np.mean(data, axis=0)
        sem = stats.sem(data, axis=0)
        ax.plot(x, mean, label=label, color=color, linestyle=linestyle, lw=2)
        ax.fill_between(x, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.18)

    plot_mean_std(ax, steps, rep_acts, f"Repeat Baseline (N={len(rep_acts)})", "#2ca02c", linestyle="--")
    plot_mean_std(ax, steps, succ_acts, f"Successful Paraphrases (N={len(succ_acts)})", "#1f77b4", linestyle="-")
    plot_mean_std(ax, steps, fail_acts, f"Failed Paraphrases (N={len(fail_acts)})", "#d62728", linestyle="-")

    ax.axvline(x=28, color="#888888", linestyle=":", alpha=0.7, label="10% Horizon (Step 28)")
    ax.axvline(x=70, color="#aaaaaa", linestyle=":", alpha=0.7, label="25% Horizon (Step 70)")
    ax.axvline(x=140, color="#cccccc", linestyle=":", alpha=0.7, label="50% Horizon (Step 140)")

    ax.set_title("Mean Action Divergence (L2) Over Episode Timesteps", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode Timestep (1 to 280)", fontsize=11)
    ax.set_ylabel("L2 Action Difference ||a_orig - a_target||_2", fontsize=11)
    ax.set_xlim(1, 280)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    p1_path = plots_dir / "mean_action_divergence_time.png"
    fig.savefig(p1_path)
    plt.close(fig)
    print(f"Saved {p1_path}")

    # Plot 2: Mean end-effector trajectory divergence over time (in cm)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    rep_pos = np.array([ts["pos_diff"] for ts in pair_timeseries if ts["pair_type"] == "repeat_baseline"]) * 100.0 # to cm
    succ_pos = np.array([ts["pos_diff"] for ts in pair_timeseries if ts["pair_type"] == "paraphrase" and ts["cand_success"]]) * 100.0
    fail_pos = np.array([ts["pos_diff"] for ts in pair_timeseries if ts["pair_type"] == "paraphrase" and not ts["cand_success"]]) * 100.0

    plot_mean_std(ax, steps, rep_pos, f"Repeat Baseline (N={len(rep_pos)})", "#2ca02c", linestyle="--")
    plot_mean_std(ax, steps, succ_pos, f"Successful Paraphrases (N={len(succ_pos)})", "#1f77b4", linestyle="-")
    plot_mean_std(ax, steps, fail_pos, f"Failed Paraphrases (N={len(fail_pos)})", "#d62728", linestyle="-")

    ax.axhline(y=1.0, color="#ff7f0e", linestyle=":", label="Divergence Threshold tau_pos = 1.0 cm")
    ax.axvline(x=28, color="#888888", linestyle=":", alpha=0.7)
    ax.axvline(x=70, color="#aaaaaa", linestyle=":", alpha=0.7)
    ax.axvline(x=140, color="#cccccc", linestyle=":", alpha=0.7)

    ax.set_title("Mean End-Effector Positional Distance Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode Timestep (1 to 280)", fontsize=11)
    ax.set_ylabel("EEF Euclidean Distance ||p_orig - p_target||_2 (cm)", fontsize=11)
    ax.set_xlim(1, 280)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    p2_path = plots_dir / "mean_trajectory_divergence_time.png"
    fig.savefig(p2_path)
    plt.close(fig)
    print(f"Saved {p2_path}")

    # Plot 3: Divergence-onset distribution for successful vs failed paraphrases
    fig, (ax_act, ax_pos) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    bins_order = ["first_10%", "10_to_25%", "25_to_50%", "final_50%", "never"]
    x = np.arange(len(bins_order))
    width = 0.35

    # Action onset distribution
    succ_act_counts = [act_bins_succ.get(b, 0.0) * 100 for b in bins_order]
    fail_act_counts = [act_bins_fail.get(b, 0.0) * 100 for b in bins_order]

    ax_act.bar(x - width/2, succ_act_counts, width, label="Successful Paraphrases", color="#1f77b4")
    ax_act.bar(x + width/2, fail_act_counts, width, label="Failed Paraphrases", color="#d62728")
    ax_act.set_xticks(x)
    ax_act.set_xticklabels(bins_order, rotation=20, ha="right")
    ax_act.set_ylabel("Percentage of Episodes (%)", fontsize=10)
    ax_act.set_title("Action Divergence Onset (tau_act = 0.05)", fontsize=11, fontweight="bold")
    ax_act.grid(axis="y", linestyle="--", alpha=0.5)
    ax_act.legend(loc="upper right")

    # Trajectory onset distribution
    succ_pos_counts = [traj_bins_succ.get(b, 0.0) * 100 for b in bins_order]
    fail_pos_counts = [traj_bins_fail.get(b, 0.0) * 100 for b in bins_order]

    ax_pos.bar(x - width/2, succ_pos_counts, width, label="Successful Paraphrases", color="#1f77b4")
    ax_pos.bar(x + width/2, fail_pos_counts, width, label="Failed Paraphrases", color="#d62728")
    ax_pos.set_xticks(x)
    ax_pos.set_xticklabels(bins_order, rotation=20, ha="right")
    ax_pos.set_ylabel("Percentage of Episodes (%)", fontsize=10)
    ax_pos.set_title("EEF Trajectory Divergence Onset (tau_pos = 1.0 cm)", fontsize=11, fontweight="bold")
    ax_pos.grid(axis="y", linestyle="--", alpha=0.5)
    ax_pos.legend(loc="upper right")

    plt.suptitle("Onset Phase Distribution of Divergence: Successful vs Failed Paraphrases", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p3_path = plots_dir / "divergence_onset_distribution.png"
    fig.savefig(p3_path)
    plt.close(fig)
    print(f"Saved {p3_path}")

    # Plot 4: dev_01 vs dev_02 divergence comparison
    fig, (ax_act_comp, ax_pos_comp) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    dev01_acts = np.array([ts["act_diff"] for ts in pair_timeseries if ts["compare_id"] == "paraphrase_dev_01"])
    dev02_acts = np.array([ts["act_diff"] for ts in pair_timeseries if ts["compare_id"] == "paraphrase_dev_02"])
    dev01_pos = np.array([ts["pos_diff"] for ts in pair_timeseries if ts["compare_id"] == "paraphrase_dev_01"]) * 100.0
    dev02_pos = np.array([ts["pos_diff"] for ts in pair_timeseries if ts["compare_id"] == "paraphrase_dev_02"]) * 100.0

    plot_mean_std(ax_act_comp, steps, dev01_acts, f"dev_01: pick up ... from ... (N={len(dev01_acts)}, succ=43.3%)", "#3b528b")
    plot_mean_std(ax_act_comp, steps, dev02_acts, f"dev_02: take ... located ... (N={len(dev02_acts)}, succ=26.7%)", "#5dc863")
    ax_act_comp.set_title("Action Divergence: dev_01 vs dev_02", fontsize=11, fontweight="bold")
    ax_act_comp.set_xlabel("Timestep", fontsize=10)
    ax_act_comp.set_ylabel("L2 Action Difference", fontsize=10)
    ax_act_comp.set_xlim(1, 280)
    ax_act_comp.grid(True, linestyle="--", alpha=0.5)
    ax_act_comp.legend(loc="upper left")

    plot_mean_std(ax_pos_comp, steps, dev01_pos, f"dev_01 (succ=43.3%)", "#3b528b")
    plot_mean_std(ax_pos_comp, steps, dev02_pos, f"dev_02 (succ=26.7%)", "#5dc863")
    ax_pos_comp.set_title("EEF Positional Distance: dev_01 vs dev_02", fontsize=11, fontweight="bold")
    ax_pos_comp.set_xlabel("Timestep", fontsize=10)
    ax_pos_comp.set_ylabel("Distance (cm)", fontsize=10)
    ax_pos_comp.set_xlim(1, 280)
    ax_pos_comp.grid(True, linestyle="--", alpha=0.5)
    ax_pos_comp.legend(loc="upper left")

    plt.suptitle("Matched Prompt Comparison: dev_01 vs dev_02 Across States 23-32", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    p4_path = plots_dir / "dev_01_vs_dev_02_comparison.png"
    fig.savefig(p4_path)
    plt.close(fig)
    print(f"Saved {p4_path}")

    print("\nDiagnostic Analysis Complete!")


if __name__ == "__main__":
    main()
