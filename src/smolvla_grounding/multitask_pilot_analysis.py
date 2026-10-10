"""Descriptive analysis of all five frozen pilot tasks after 75 committed rollouts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .multitask_pilot_protocol import pilot_preflight
from .multitask_pilot_evaluator import _paired_reset_audit
from .multitask_protocol import _valid_completion, trial_path
from .protocol import atomic_json

PROMPTS = ("original", "paraphrase_01", "paraphrase_02")


def _rate(values):
    return {"successes": int(sum(values)), "trials": len(values),
            "rate": float(np.mean(values)) if values else None}


def task_success(rows):
    """Rows are the five states, each with all three Boolean success values."""
    if len(rows) != 5 or {r["state"] for r in rows} != {5, 6, 7, 8, 9}:
        raise ValueError("Expected all five paired development states")
    for row in rows:
        if any(type(row[p]) is not bool for p in PROMPTS):
            raise ValueError("Missing Boolean task outcome")
    result = {p: _rate([r[p] for r in rows]) for p in PROMPTS}
    result["pooled_paraphrase"] = _rate([r[p] for r in rows for p in PROMPTS[1:]])
    result["paired_difference"] = {
        p: float(np.mean([int(r[p]) - int(r["original"]) for r in rows]))
        for p in PROMPTS[1:]
    }
    result["paired_difference"]["pooled_paraphrase"] = float(np.mean([
        int(r[p]) - int(r["original"]) for r in rows for p in PROMPTS[1:]
    ]))
    result["canonical_success_paraphrase_failure_pairs"] = sum(
        r["original"] and not r[p] for r in rows for p in PROMPTS[1:]
    )
    return result


def _screen_action_scale(root):
    """Predeclared dimension scale from canonical competence states 0–4 only."""
    base = root / "outputs/multitask_evaluation/runs/multitask_canonical_competence_screen/untuned_fp32"
    arrays = []
    for suite, task_id in (("libero_spatial", 1), ("libero_spatial", 4),
                           ("libero_object", 0), ("libero_goal", 0),
                           ("libero_goal", 5), ("libero_goal", 7)):
        for state in range(5):
            trial = base / suite / f"task_{task_id:02d}" / f"state_{state:04d}_seed_0000_original"
            commits = list(trial.glob("attempt_*/complete.json"))
            if len(commits) != 1:
                raise RuntimeError(f"Incomplete scale reference: {trial}")
            with np.load(commits[0].parent / "trajectory.npz", allow_pickle=False) as data:
                arrays.append(np.asarray(data["action"], dtype=float))
    actions = np.concatenate(arrays)
    if actions.ndim != 2 or not np.isfinite(actions).all():
        raise RuntimeError("Invalid competence action scale data")
    scale = np.maximum(np.percentile(actions, 95, axis=0) - np.percentile(actions, 5, axis=0), 0.05)
    return scale


def _load_trial(plan, task, state, prompt):
    row = ("untuned_fp32", task["suite"], task["task_id"], state, prompt)
    path = _valid_completion(
        trial_path(plan["output_dir"], row, plan["matrix"]["policy"]["policy_seed"]),
        row, task["horizon"], plan["identity"])
    if path is None:
        raise RuntimeError(f"Uncommitted pilot result: {row}")
    commit = json.loads((path / "complete.json").read_text(encoding="utf-8"))
    with np.load(path / "trajectory.npz", allow_pickle=False) as data:
        action = np.asarray(data["action"], dtype=float)
        eef = np.asarray(data["eef_after"], dtype=float)
    inference = [json.loads(line) for line in (path / "inference.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(inference) != task["horizon"] or action.shape[0] != task["horizon"] or eef.shape != (task["horizon"], 3):
        raise RuntimeError(f"Invalid trajectory shape: {path}")
    return {"success": commit["goal_ever"], "action": action, "eef": eef, "inference": inference, "path": path}


def analyze(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    plan = pilot_preflight(root)
    if len(plan["rows"]) != 75:
        raise RuntimeError("Not the frozen 75-cell pilot")
    scale = _screen_action_scale(root)
    per_task, pairs, examples = [], [], []
    for task in plan["matrix"]["tasks"]:
        states = []
        for state in task["states"]:
            _paired_reset_audit(plan, task, state)
            records = {p: _load_trial(plan, task, state, p) for p in PROMPTS}
            states.append({"state": state, **{p: records[p]["success"] for p in PROMPTS}})
            for prompt in PROMPTS[1:]:
                canonical, variant = records["original"], records[prompt]
                diff = variant["action"] - canonical["action"]
                distance = np.linalg.norm(diff / scale, axis=1)
                eef_distance = np.linalg.norm(variant["eef"] - canonical["eef"], axis=1)
                pairs.append({
                    "suite": task["suite"], "task_id": task["task_id"], "state": state,
                    "prompt": prompt, "canonical_success": canonical["success"],
                    "paraphrase_success": variant["success"],
                    "first_action_l2_raw": float(np.linalg.norm(diff[0])),
                    "first_action_l2_normalized": float(distance[0]),
                    "early_action_l2_normalized": float(np.mean(distance[:10])),
                    "eef_at_step_10_m": float(eef_distance[9]),
                    "early_eef_mean_m": float(np.mean(eef_distance[:10])),
                    "noise_equal_first_10_steps": all(
                        canonical["inference"][i]["noise"] == variant["inference"][i]["noise"]
                        for i in range(10)),
                    "canonical_eef_xy": canonical["eef"][:, :2].tolist(),
                    "paraphrase_eef_xy": variant["eef"][:, :2].tolist(),
                })
        result = task_success(states)
        per_task.append({"suite": task["suite"], "task_id": task["task_id"], **result})
        print(f"[Analysis] {task['suite']}/{task['task_id']} 15/15 committed", flush=True)
    grouped = {}
    for label, subset in (("paraphrase_success", [p for p in pairs if p["paraphrase_success"]]),
                          ("paraphrase_failure", [p for p in pairs if not p["paraphrase_success"]])):
        grouped[label] = {"pairs": len(subset),
                          "mean_early_action_l2_normalized":
                          float(np.mean([p["early_action_l2_normalized"] for p in subset])) if subset else None,
                          "mean_eef_at_step_10_m":
                          float(np.mean([p["eef_at_step_10_m"] for p in subset])) if subset else None}
    summary = {
        "role": "multitask_language_robustness", "model": "untuned_fp32",
        "development_only": True, "states": list(range(5, 10)),
        "completed_rollouts": 75, "metric": "goal_ever", "per_task": per_task,
        "total_canonical_success_paraphrase_failure_pairs": sum(
            x["canonical_success_paraphrase_failure_pairs"] for x in per_task),
        "action_normalization": {"method": "per-dimension max(P95-P5, 0.05) from 30 canonical competence screen trajectories (states 0-4)",
                                 "scale": scale.tolist()},
        "divergence_by_outcome": grouped,
        "failure_stage": {"classified": 0, "unknown": sum(5 - x["original"]["successes"] + 10 - x["pooled_paraphrase"]["successes"] for x in per_task),
                          "reason": "Task-phase annotations are not encoded in the committed logs; EEF/action traces alone do not prove object contact or phase of failure."},
        "limitations": ["Five states per task; descriptive development pilot only.",
                        "Step-1 RNG and sampled flow-noise hashes are matched; later sampling and closed-loop states may diverge.",
                        "Later trajectory divergence includes closed-loop state changes and is not an isolated language effect.",
                        "Success/divergence association is observational, not causal."],
    }
    out = root / "results/multitask_language_pilot"
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "summary.json", summary)
    columns = ["suite", "task_id", "state", "prompt", "canonical_success", "paraphrase_success",
               "first_action_l2_raw", "first_action_l2_normalized", "early_action_l2_normalized",
               "eef_at_step_10_m", "early_eef_mean_m", "noise_equal_first_10_steps"]
    with (out / "pair_level.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for pair in pairs:
            writer.writerow({name: pair[name] for name in columns})
    _plot(out, per_task, pairs)
    print("[Analysis] all 75 committed; summary, 50 paired rows and plots written", flush=True)
    return summary


def _plot(out, per_task, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [f"{x['suite'].removeprefix('libero_')}/{x['task_id']}" for x in per_task]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(5)
    for offset, (prompt, title) in enumerate((("original", "canonical"), ("paraphrase_01", "paraphrase 1"), ("paraphrase_02", "paraphrase 2"))):
        ax.bar(x + (offset - 1) * 0.25, [r[prompt]["rate"] for r in per_task], width=0.24, label=title)
    ax.set_xticks(x, labels); ax.set_ylim(0, 1.05); ax.set_ylabel("Goal-ever success rate (n=5)")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "task_success.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for success, label in ((True, "paraphrase success"), (False, "paraphrase failure")):
        values = [p["early_action_l2_normalized"] for p in pairs if p["paraphrase_success"] == success]
        if values:
            ax.hist(values, bins=12, alpha=0.55, label=f"{label} (n={len(values)})")
    ax.set_xlabel("Mean normalized action L2, steps 1–10"); ax.set_ylabel("Paired episodes")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "early_action_divergence.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for success, label in ((True, "success"), (False, "failure")):
        selected = [p for p in pairs if p["paraphrase_success"] == success]
        ax.scatter([p["early_action_l2_normalized"] for p in selected],
                   [p["eef_at_step_10_m"] * 100 for p in selected], label=f"paraphrase {label}", alpha=0.75)
    ax.set_xlabel("Mean normalized action L2, steps 1–10"); ax.set_ylabel("EEF separation at step 10 (cm)")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "divergence_vs_outcome.png", dpi=180); plt.close(fig)
    success_example = next((p for p in pairs if p["canonical_success"] and p["paraphrase_success"]), None)
    failure_example = next((p for p in pairs if p["canonical_success"] and not p["paraphrase_success"]), None)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, pair, title in zip(axes, (success_example, failure_example), ("Both succeed", "Canonical succeeds; paraphrase fails")):
        if pair is None:
            ax.text(0.5, 0.5, "No eligible pair", ha="center", va="center")
        else:
            for field, label in (("canonical_eef_xy", "canonical"), ("paraphrase_eef_xy", "paraphrase")):
                path = np.asarray(pair[field])
                ax.plot(path[:, 0], path[:, 1], label=label)
            ax.set_title(f"{title}\n{pair['suite']}/{pair['task_id']} state {pair['state']}")
            ax.legend()
        ax.set_xlabel("EEF x (m)"); ax.set_ylabel("EEF y (m)"); ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout(); fig.savefig(out / "paired_eef_examples.png", dpi=180); plt.close(fig)
