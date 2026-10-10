"""Frozen five-task development pilot guard layered on the shared evaluator preflight."""
from __future__ import annotations

import json
from pathlib import Path

from .multitask_protocol import preflight

MATRIX = "configs/multitask_evaluation/language_pilot_matrix.json"
MANIFEST = "configs/multitask_evaluation/language_pilot_manifest.json"
SCREEN_SUMMARY = "results/multitask_canonical_screen/summary.json"
TASKS = (("libero_spatial", 1), ("libero_spatial", 4),
         ("libero_object", 0), ("libero_goal", 5), ("libero_goal", 7))
STATES = [5, 6, 7, 8, 9]
PROMPTS = ["original", "paraphrase_01", "paraphrase_02"]


def pilot_preflight(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    matrix = json.loads((root / MATRIX).read_text(encoding="utf-8"))
    summary = json.loads((root / SCREEN_SUMMARY).read_text(encoding="utf-8"))
    eligible = {(r["suite"], r["task_id"]) for r in summary["per_task"]
                if r["classification"] == "eligible_for_language_robustness"}
    if summary["completed_rollouts"] != 30 or eligible != set(TASKS):
        raise RuntimeError("Competence screen does not select exactly the five pilot tasks")
    if matrix.get("role") != "multitask_language_robustness" or matrix.get("run_name") != "multitask_language_sensitivity_pilot":
        raise RuntimeError("Wrong development pilot role or name")
    if matrix.get("models") != ["untuned_fp32"] or matrix.get("expected_rollouts") != 75:
        raise RuntimeError("Development pilot requires one model and 75 rollouts")
    if matrix.get("output_dir") != "outputs/multitask_evaluation/runs/multitask_language_sensitivity_pilot":
        raise RuntimeError("Unexpected development pilot output directory")
    if [(t.get("suite"), t.get("task_id")) for t in matrix.get("tasks", [])] != list(TASKS):
        raise RuntimeError("Development pilot task set or order changed")
    if any(t.get("states") != STATES for t in matrix["tasks"]):
        raise RuntimeError("Development pilot states must be exactly 5–9")
    if matrix.get("primary_success") != "goal_ever" or matrix.get("scientific_evidence") is not True:
        raise RuntimeError("Development pilot outcome semantics changed")
    plan = preflight(root, MATRIX, MANIFEST)
    if plan["manifest"].get("role") != "multitask_language_robustness":
        raise RuntimeError("Pilot manifest role mismatch")
    if len(plan["rows"]) != 75 or any(row[3] in range(30, 50) for row in plan["rows"]):
        raise RuntimeError("Pilot matrix count or final-state isolation failed")
    for key, prompts in plan["catalogs"].items():
        if [p["id"] for p in prompts] != PROMPTS:
            raise RuntimeError(f"Unexpected prompts: {key}")
    return plan
