"""Descriptive analysis of all five frozen pilot tasks after 75 committed rollouts."""
from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from .multitask_pilot_protocol import MATRIX, MANIFEST, pilot_preflight
from .multitask_pilot_evaluator import _paired_reset_audit
from .multitask_protocol import _valid_completion, trial_path
from .protocol import atomic_json, file_sha256

PROMPTS = ("original", "paraphrase_01", "paraphrase_02")
TRANSITIONS = ("both_succeed", "canonical_success_paraphrase_failure",
               "canonical_failure_paraphrase_success", "both_fail")


def outcome_transition(canonical, paraphrase):
    if type(canonical) is not bool or type(paraphrase) is not bool:
        raise ValueError("Outcomes must be Boolean")
    if canonical and paraphrase:
        return "both_succeed"
    if canonical:
        return "canonical_success_paraphrase_failure"
    if paraphrase:
        return "canonical_failure_paraphrase_success"
    return "both_fail"


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
    result["canonical_success_paraphrase_failure_states"] = sum(
        r["original"] and any(not r[p] for p in PROMPTS[1:]) for r in rows
    )
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
    return {"success": commit["goal_ever"], "goal_final": commit["goal_final"],
            "action": action, "eef": eef, "inference": inference, "path": path,
            "commit_sha256": file_sha256(path / "complete.json")}


def analyze(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    plan = pilot_preflight(root)
    if len(plan["rows"]) != 75:
        raise RuntimeError("Not the frozen 75-cell pilot")
    scale = _screen_action_scale(root)
    per_task, pairs, rollouts = [], [], []
    for task in plan["matrix"]["tasks"]:
        states = []
        for state in task["states"]:
            _paired_reset_audit(plan, task, state)
            records = {p: _load_trial(plan, task, state, p) for p in PROMPTS}
            states.append({"state": state, **{p: records[p]["success"] for p in PROMPTS}})
            for prompt in PROMPTS:
                record = records[prompt]
                rollouts.append({
                    "model": "untuned_fp32", "suite": task["suite"],
                    "task_id": task["task_id"], "state": state, "prompt": prompt,
                    "goal_ever": record["success"], "goal_final": record["goal_final"],
                    "policy_steps": task["horizon"],
                    "evidence_dir": record["path"].relative_to(root).as_posix(),
                    "complete_sha256": record["commit_sha256"],
                })
            for prompt in PROMPTS[1:]:
                canonical, variant = records["original"], records[prompt]
                diff = variant["action"] - canonical["action"]
                distance = np.linalg.norm(diff / scale, axis=1)
                eef_distance = np.linalg.norm(variant["eef"] - canonical["eef"], axis=1)
                pairs.append({
                    "suite": task["suite"], "task_id": task["task_id"], "state": state,
                    "prompt": prompt, "canonical_success": canonical["success"],
                    "paraphrase_success": variant["success"],
                    "outcome_transition": outcome_transition(canonical["success"], variant["success"]),
                    "eef_step_1_m": float(eef_distance[0]),
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
        result["goal_final"] = {
            prompt: _rate([r["goal_final"] for r in rollouts
                           if r["suite"] == task["suite"] and r["task_id"] == task["task_id"]
                           and r["prompt"] == prompt])
            for prompt in PROMPTS
        }
        result["goal_final"]["pooled_paraphrase"] = _rate([
            r["goal_final"] for r in rollouts
            if r["suite"] == task["suite"] and r["task_id"] == task["task_id"]
            and r["prompt"] != "original"
        ])
        per_task.append({"suite": task["suite"], "task_id": task["task_id"], **result})
        print(f"[Analysis] {task['suite']}/{task['task_id']} 15/15 committed", flush=True)
    if len(rollouts) != 75 or len(pairs) != 50:
        raise RuntimeError("Pilot result count differs from the frozen matrix")
    expected_commits = {(root / r["evidence_dir"] / "complete.json").resolve()
                        for r in rollouts}
    observed_commits = {p.resolve() for p in plan["output_dir"].rglob("complete.json")}
    if observed_commits != expected_commits:
        raise RuntimeError("Unexpected, missing, or duplicate pilot completion path")
    attempts = sorted(p for p in plan["output_dir"].rglob("attempt_*") if p.is_dir())
    incomplete = [p.relative_to(root).as_posix() for p in attempts
                  if not (p / "complete.json").is_file()]
    audit = {
        "experiment_identity": plan["identity"],
        "matrix_sha256": file_sha256(root / MATRIX),
        "manifest_sha256": file_sha256(root / MANIFEST),
        "expected_identities": len(plan["rows"]),
        "valid_committed_identities": len(rollouts),
        "committed_attempts": len(observed_commits),
        "all_attempt_directories": len(attempts),
        "incomplete_attempts": incomplete,
        "unexpected_committed_attempts": 0,
        "duplicate_committed_identities": 0,
        "paired_groups_checked": 25,
        "paired_initial_rng_noise_mismatches": 0,
        "goal_ever_and_goal_final_present": len(rollouts),
        "forbidden_state_rows": sum(30 <= r["state"] <= 49 for r in rollouts),
        "raw_commit_and_artifact_hashes_verified": True,
        "frozen_model_config_catalog_and_state_assets_verified": True,
    }
    transitions = Counter(p["outcome_transition"] for p in pairs)
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
        "aggregate": {
            **{prompt: _rate([r["goal_ever"] for r in rollouts if r["prompt"] == prompt])
               for prompt in PROMPTS},
            "pooled_paraphrase": _rate([r["goal_ever"] for r in rollouts
                                        if r["prompt"] != "original"]),
        },
        "goal_final_aggregate": {
            **{prompt: _rate([r["goal_final"] for r in rollouts if r["prompt"] == prompt])
               for prompt in PROMPTS},
            "pooled_paraphrase": _rate([r["goal_final"] for r in rollouts
                                        if r["prompt"] != "original"]),
        },
        "outcome_transitions": {name: transitions[name] for name in TRANSITIONS},
        "divergence_coverage": {
            "pairs": len(pairs),
            "step_1_action_and_eef": len(pairs),
            "steps_1_to_10_action_and_eef": len(pairs),
            "mean_step_1_action_l2_normalized": float(np.mean([
                p["first_action_l2_normalized"] for p in pairs])),
            "mean_step_1_eef_separation_m": float(np.mean([
                p["eef_step_1_m"] for p in pairs])),
            "mean_early_action_l2_normalized": float(np.mean([
                p["early_action_l2_normalized"] for p in pairs])),
            "mean_eef_separation_at_step_10_m": float(np.mean([
                p["eef_at_step_10_m"] for p in pairs])),
            "early_noise_matched_pairs": sum(p["noise_equal_first_10_steps"] for p in pairs),
        },
        "total_canonical_success_paraphrase_failure_states": sum(
            x["canonical_success_paraphrase_failure_states"] for x in per_task),
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
    rollout_columns = ["model", "suite", "task_id", "state", "prompt", "goal_ever",
                       "goal_final", "policy_steps", "evidence_dir", "complete_sha256"]
    with (out / "per_rollout.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rollout_columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rollouts)
    columns = ["suite", "task_id", "state", "prompt", "canonical_success", "paraphrase_success",
               "outcome_transition", "first_action_l2_raw", "first_action_l2_normalized",
               "early_action_l2_normalized", "eef_step_1_m", "eef_at_step_10_m",
               "early_eef_mean_m", "noise_equal_first_10_steps"]
    with (out / "pair_level.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for pair in pairs:
            writer.writerow({name: pair[name] for name in columns})
    examples = [
        next((p for p in pairs if p["outcome_transition"] == "both_succeed"), None),
        next((p for p in pairs
              if p["outcome_transition"] == "canonical_success_paraphrase_failure"), None),
    ]
    atomic_json(out / "paired_eef_examples.json", [
        {key: item[key] for key in ("suite", "task_id", "state", "prompt",
                                    "canonical_eef_xy", "paraphrase_eef_xy")}
        for item in examples if item is not None
    ])
    _plot(out, per_task, pairs)
    audit["public_source_sha256"] = {
        name: file_sha256(out / name) for name in (
            "summary.json", "per_rollout.csv", "pair_level.csv", "paired_eef_examples.json")
    }
    audit["figure_sha256"] = {
        name: file_sha256(out / name) for name in (
            "task_success.png", "early_action_divergence.png",
            "divergence_vs_outcome.png", "paired_eef_examples.png")
    }
    atomic_json(out / "integrity.json", audit)
    print("[Analysis] all 75 committed; per-rollout, 50 paired rows, integrity and plots written", flush=True)
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

def regenerate_published(root=None):
    """Verify public rows and regenerate all figures without raw rollouts or model assets."""
    import hashlib
    import math
    import re
    from tempfile import TemporaryDirectory

    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    out = root / "results/multitask_language_pilot"
    audit = json.loads((out / "integrity.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    matrix = json.loads((root / MATRIX).read_text(encoding="utf-8"))
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))

    def check(actual, expected, label):
        if isinstance(expected, dict):
            if not isinstance(actual, dict) or actual.keys() != expected.keys():
                raise RuntimeError(f"{label}: keys differ")
            for key in expected:
                check(actual[key], expected[key], f"{label}.{key}")
        elif isinstance(expected, float):
            if not isinstance(actual, (int, float)) or not math.isclose(
                    actual, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError(f"{label}: value differs")
        elif actual != expected:
            raise RuntimeError(f"{label}: value differs")

    def boolean(value, label):
        if value not in ("True", "False"):
            raise RuntimeError(f"{label}: missing or invalid Boolean")
        return value == "True"

    if file_sha256(root / MATRIX) != audit["matrix_sha256"]:
        raise RuntimeError("Published matrix hash differs")
    if file_sha256(root / MANIFEST) != audit["manifest_sha256"]:
        raise RuntimeError("Published manifest hash differs")
    if matrix["expected_rollouts"] != 75 or matrix["models"] != ["untuned_fp32"]:
        raise RuntimeError("Not the frozen 75-rollout pilot")
    for rel, digest in manifest["catalog_sha256"].items():
        if file_sha256(root / rel) != digest:
            raise RuntimeError(f"Published instruction catalog differs: {rel}")
    for task in matrix["tasks"]:
        catalog = [json.loads(line) for line in (root / task["prompt_catalog"])
                   .read_text(encoding="utf-8").splitlines()]
        if [item["id"] for item in catalog] != list(PROMPTS):
            raise RuntimeError("Published prompt IDs differ")
        for item in catalog:
            key = f"{task['suite']}:{task['task_id']}:{item['id']}"
            digest = hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
            if digest != manifest["prompt_sha256"][key]:
                raise RuntimeError(f"Published prompt text differs: {key}")
    for name, digest in audit["public_source_sha256"].items():
        if file_sha256(out / name) != digest:
            raise RuntimeError(f"Published source hash differs: {name}")
    with (out / "per_rollout.csv").open(encoding="utf-8", newline="") as stream:
        rollouts = list(csv.DictReader(stream))
    with (out / "pair_level.csv").open(encoding="utf-8", newline="") as stream:
        pairs = list(csv.DictReader(stream))
    expected_ids = [(task["suite"], task["task_id"], state, prompt)
                    for task in matrix["tasks"] for state in task["states"]
                    for prompt in PROMPTS]
    actual_ids = [(r["suite"], int(r["task_id"]), int(r["state"]), r["prompt"])
                  for r in rollouts]
    if actual_ids != expected_ids:
        raise RuntimeError("Published per-rollout matrix differs")
    outcomes = {}
    for row, identity in zip(rollouts, expected_ids):
        task = next(t for t in matrix["tasks"] if (t["suite"], t["task_id"]) == identity[:2])
        if row["model"] != "untuned_fp32" or int(row["policy_steps"]) != task["horizon"]:
            raise RuntimeError("Published model or horizon differs")
        expected_trial = trial_path(Path(matrix["output_dir"]),
                                    ("untuned_fp32", *identity), matrix["policy"]["policy_seed"])
        evidence = row["evidence_dir"]
        if not evidence.startswith(expected_trial.as_posix() + "/attempt_"):
            raise RuntimeError("Published evidence identity differs")
        if not re.fullmatch(r"[0-9a-f]{64}", row["complete_sha256"]):
            raise RuntimeError("Published commit digest is invalid")
        outcomes[identity] = (boolean(row["goal_ever"], "goal_ever"),
                              boolean(row["goal_final"], "goal_final"))
    expected_pairs = [(task["suite"], task["task_id"], state, prompt)
                      for task in matrix["tasks"] for state in task["states"]
                      for prompt in PROMPTS[1:]]
    actual_pairs = [(r["suite"], int(r["task_id"]), int(r["state"]), r["prompt"])
                    for r in pairs]
    if actual_pairs != expected_pairs:
        raise RuntimeError("Published pair matrix differs")
    numeric = ("first_action_l2_raw", "first_action_l2_normalized",
               "early_action_l2_normalized", "eef_step_1_m",
               "eef_at_step_10_m", "early_eef_mean_m")
    for pair, identity in zip(pairs, expected_pairs):
        pair["task_id"], pair["state"] = int(pair["task_id"]), int(pair["state"])
        canonical = outcomes[(*identity[:3], "original")][0]
        variant = outcomes[identity][0]
        if boolean(pair["canonical_success"], "canonical_success") != canonical:
            raise RuntimeError("Published canonical pair outcome differs")
        if boolean(pair["paraphrase_success"], "paraphrase_success") != variant:
            raise RuntimeError("Published paraphrase pair outcome differs")
        if pair["outcome_transition"] != outcome_transition(canonical, variant):
            raise RuntimeError("Published outcome transition differs")
        if not boolean(pair["noise_equal_first_10_steps"], "noise match"):
            raise RuntimeError("Published early noise mismatch")
        for field in numeric:
            try:
                value = float(pair[field])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Missing or invalid published divergence: {field}") from exc
            if not math.isfinite(value) or value < 0:
                raise RuntimeError(f"Missing or invalid published divergence: {field}")
            pair[field] = value
        pair["canonical_success"], pair["paraphrase_success"] = canonical, variant

    per_task = []
    for task in matrix["tasks"]:
        suite, task_id = task["suite"], task["task_id"]
        states = [{"state": state, **{prompt: outcomes[(suite, task_id, state, prompt)][0]
                                      for prompt in PROMPTS}}
                  for state in task["states"]]
        result = task_success(states)
        result["goal_final"] = {
            prompt: _rate([outcomes[(suite, task_id, state, prompt)][1]
                           for state in task["states"]])
            for prompt in PROMPTS
        }
        result["goal_final"]["pooled_paraphrase"] = _rate([
            outcomes[(suite, task_id, state, prompt)][1]
            for state in task["states"] for prompt in PROMPTS[1:]
        ])
        per_task.append({"suite": suite, "task_id": task_id, **result})
    check(summary["per_task"], per_task, "per_task")
    aggregate = {
        **{prompt: _rate([outcomes[identity][0] for identity in expected_ids
                          if identity[3] == prompt]) for prompt in PROMPTS},
        "pooled_paraphrase": _rate([outcomes[identity][0] for identity in expected_ids
                                    if identity[3] != "original"]),
    }
    final_aggregate = {
        **{prompt: _rate([outcomes[identity][1] for identity in expected_ids
                          if identity[3] == prompt]) for prompt in PROMPTS},
        "pooled_paraphrase": _rate([outcomes[identity][1] for identity in expected_ids
                                    if identity[3] != "original"]),
    }
    check(summary["aggregate"], aggregate, "aggregate")
    check(summary["goal_final_aggregate"], final_aggregate, "goal_final_aggregate")
    transitions = Counter(pair["outcome_transition"] for pair in pairs)
    check(summary["outcome_transitions"],
          {name: transitions[name] for name in TRANSITIONS}, "outcome_transitions")
    for label, selected in (("paraphrase_success", [p for p in pairs if p["paraphrase_success"]]),
                            ("paraphrase_failure", [p for p in pairs if not p["paraphrase_success"]])):
        check(summary["divergence_by_outcome"][label], {
            "pairs": len(selected),
            "mean_early_action_l2_normalized": float(np.mean([
                p["early_action_l2_normalized"] for p in selected])) if selected else None,
            "mean_eef_at_step_10_m": float(np.mean([
                p["eef_at_step_10_m"] for p in selected])) if selected else None,
        }, f"divergence_by_outcome.{label}")
    check(summary["divergence_coverage"], {
        "pairs": len(pairs), "step_1_action_and_eef": len(pairs),
        "steps_1_to_10_action_and_eef": len(pairs),
        "mean_step_1_action_l2_normalized": float(np.mean([
            p["first_action_l2_normalized"] for p in pairs])),
        "mean_step_1_eef_separation_m": float(np.mean([
            p["eef_step_1_m"] for p in pairs])),
        "mean_early_action_l2_normalized": float(np.mean([
            p["early_action_l2_normalized"] for p in pairs])),
        "mean_eef_separation_at_step_10_m": float(np.mean([
            p["eef_at_step_10_m"] for p in pairs])),
        "early_noise_matched_pairs": len(pairs),
    }, "divergence_coverage")
    if summary["completed_rollouts"] != 75 or audit["valid_committed_identities"] != 75:
        raise RuntimeError("Published completeness differs")
    if summary["total_canonical_success_paraphrase_failure_pairs"] != transitions[
            "canonical_success_paraphrase_failure"]:
        raise RuntimeError("Published discordant pair count differs")
    if summary["total_canonical_success_paraphrase_failure_states"] != sum(
            r["canonical_success_paraphrase_failure_states"] for r in per_task):
        raise RuntimeError("Published discordant state count differs")
    scale = summary["action_normalization"]["scale"]
    if len(scale) != 7 or any(not math.isfinite(x) or x <= 0 for x in scale):
        raise RuntimeError("Published action normalization is invalid")
    if summary["action_normalization"]["method"] != (
            "per-dimension max(P95-P5, 0.05) from 30 canonical competence screen trajectories (states 0-4)"):
        raise RuntimeError("Published action normalization method differs")

    examples = json.loads((out / "paired_eef_examples.json").read_text(encoding="utf-8"))
    selected = [next((p for p in pairs if p["outcome_transition"] == name), None)
                for name in ("both_succeed", "canonical_success_paraphrase_failure")]
    selected = [p for p in selected if p is not None]
    if len(examples) != len(selected):
        raise RuntimeError("Published EEF example count differs")
    for example, pair in zip(examples, selected):
        if any(example[key] != pair[key] for key in ("suite", "task_id", "state", "prompt")):
            raise RuntimeError("Published EEF example identity differs")
        task = next(t for t in matrix["tasks"]
                    if (t["suite"], t["task_id"]) == (pair["suite"], int(pair["task_id"])))
        for field in ("canonical_eef_xy", "paraphrase_eef_xy"):
            values = np.asarray(example[field], dtype=float)
            if values.shape != (task["horizon"], 2) or not np.isfinite(values).all():
                raise RuntimeError("Missing or invalid published EEF example")
            pair[field] = example[field]
    with TemporaryDirectory() as directory:
        regenerated = Path(directory)
        _plot(regenerated, per_task, pairs)
        for name, digest in audit["figure_sha256"].items():
            if file_sha256(regenerated / name) != digest:
                raise RuntimeError(f"Regenerated figure differs: {name}")
    with (out / "case_index.csv").open(encoding="utf-8", newline="") as stream:
        cases = list(csv.DictReader(stream))
    chosen = []
    for task in matrix["tasks"]:
        seen = set()
        for pair in pairs:
            if (pair["suite"], pair["task_id"]) != (task["suite"], task["task_id"]):
                continue
            if pair["outcome_transition"] != "canonical_success_paraphrase_failure":
                continue
            if pair["state"] in seen:
                continue
            chosen.append(("failure", pair))
            seen.add(pair["state"])
            if len(seen) == 2:
                break
        if len(seen) != 2:
            raise RuntimeError("Published case selection lacks two failure states")
    for task in matrix["tasks"]:
        item = next((p for p in pairs
                     if (p["suite"], p["task_id"]) == (task["suite"], task["task_id"])
                     and p["outcome_transition"] == "both_succeed"), None)
        if item is not None:
            chosen.append(("success_contrast", item))
    if len(cases) != len(chosen) or len(cases) < 10:
        raise RuntimeError("Published case count differs")
    rollout_by_id = dict(zip(expected_ids, rollouts))
    for ordinal, (case, (selection, pair)) in enumerate(zip(cases, chosen), 1):
        identity = (pair["suite"], pair["task_id"], pair["state"], pair["prompt"])
        canonical = rollout_by_id[(*identity[:3], "original")]
        variant = rollout_by_id[identity]
        task = next(t for t in matrix["tasks"]
                    if (t["suite"], t["task_id"]) == identity[:2])
        catalog = {item["id"]: item["text"] for item in (
            json.loads(line) for line in (root / task["prompt_catalog"])
            .read_text(encoding="utf-8").splitlines())}
        if (case["case_id"] != f"C{ordinal:02d}" or case["selection"] != selection
                or case["model"] != "untuned_fp32"
                or case["experiment_identity"] != audit["experiment_identity"]
                or (case["suite"], int(case["task_id"]), int(case["state"]),
                    case["paraphrase_prompt"]) != identity
                or case["canonical_text"] != catalog["original"]
                or case["paraphrase_text"] != catalog[pair["prompt"]]
                or case["canonical_evidence_dir"] != canonical["evidence_dir"]
                or case["paraphrase_evidence_dir"] != variant["evidence_dir"]
                or case["canonical_goal_ever"] != canonical["goal_ever"]
                or case["paraphrase_goal_ever"] != variant["goal_ever"]
                or case["canonical_goal_final"] != canonical["goal_final"]
                or case["paraphrase_goal_final"] != variant["goal_final"]
                or case["outcome_transition"] != pair["outcome_transition"]):
            raise RuntimeError(f"Published case identity or outcome differs: C{ordinal:02d}")
        for field, expected in (
                ("step1_action_l2_normalized", pair["first_action_l2_normalized"]),
                ("early_action_l2_normalized", pair["early_action_l2_normalized"]),
                ("eef_at_step_10_cm", pair["eef_at_step_10_m"] * 100)):
            if not math.isclose(float(case[field]), expected, rel_tol=1e-8, abs_tol=1e-8):
                raise RuntimeError(f"Published case metric differs: {case['case_id']} {field}")
        if (case["failure_category"], case["mechanism"]) != (
                ("task_timeout", "unknown") if selection == "failure"
                else ("none_success", "not_applicable")):
            raise RuntimeError(f"Published case classification differs: {case['case_id']}")
        if case["case_sheet"] != f"results/multitask_language_pilot/cases/C{ordinal:02d}.png":
            raise RuntimeError("Published case sheet path differs")
        if file_sha256(root / case["case_sheet"]) != case["case_sheet_sha256"]:
            raise RuntimeError(f"Published case sheet hash differs: {case['case_id']}")
        if any(not re.fullmatch(r"[0-9a-f]{64}", case[field]) for field in
               ("canonical_video_sha256", "paraphrase_video_sha256")):
            raise RuntimeError(f"Published video digest invalid: {case['case_id']}")
    print(f"[Published cases] {len(cases)} public case sheets and source rows verified", flush=True)
    print("[Published analysis] 75 outcomes, 50 pairs, all statistics and 4 figures reproduced", flush=True)
    return summary
