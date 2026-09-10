"""Numerical summaries for fixed-horizon paired trials, separate from standard eval."""

import csv
import io
import json
from pathlib import Path

from .protocol import atomic_json, file_sha256, fingerprint


def verify_pairing(root, config, rows, report_path):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    identity = json.loads((root / config["pairing_output"] / "identity.json").read_text(encoding="utf-8"))
    if not report["passed"] or report["fingerprint"] != fingerprint(identity):
        raise RuntimeError("Missing or inconsistent passed pairing evidence")
    if identity["config"] != config or identity["instructions"] != rows:
        raise RuntimeError("Pilot configuration/catalog differs from the passed pairing check")
    for name, expected in identity["project_sources"].items():
        if file_sha256(root / name) != expected:
            raise RuntimeError(f"Pairing source changed since validation: {name}")
    if [r["instruction_id"] for r in report["checks"]] != [r["id"] for r in rows]:
        raise RuntimeError("Pairing report does not cover all configured instructions")
    reference = report["checks"][0]["hashes"]
    for row, check in zip(rows, report["checks"], strict=True):
        marker = root / config["pairing_output"] / row["id"] / "complete.json"
        record = json.loads(marker.read_text(encoding="utf-8"))
        if record["fingerprint"] != report["fingerprint"] or record["result"] != check:
            raise RuntimeError(f"Invalid completed pairing record: {row['id']}")
        if check["hashes"] != reference or any(check["initial_goals"].values()):
            raise RuntimeError("Invalid initial state/goal pairing")
        for name, expected in record["artifacts"].items():
            if file_sha256(root / config["pairing_output"] / name) != expected:
                raise RuntimeError(f"Modified pairing evidence: {name}")
    return report, identity


def summarize_goals(step_goals, goal_ids):
    if not step_goals:
        raise ValueError("Cannot score an empty rollout")
    result = {}
    for name in goal_ids:
        values = [bool(g[name]) for g in step_goals]
        result[name] = {"ever": any(values), "final": values[-1],
                        "first_step": next((i for i, x in enumerate(values, 1) if x), None)}
    return result


def export_episode_table(directory, results, expected_count, repeat_control=None):
    """Retain explicit trial denominators; never pool changed-goal success as SR."""
    directory = Path(directory)
    atomic_json(directory / "summary.json", {
        "completed_trials": len(results), "expected_trials": expected_count,
        "all_trials_complete": len(results) == expected_count,
        "standard_evaluation": False,
        "alternative_goal_validation": "provisional; positive cases require video review",
        "repeat_control": repeat_control, "episodes": results,
    })
    fields = ["instruction_id", "condition", "instruction", "goal_id", "policy_steps",
              "requested_goal_ever", "requested_goal_final", "requested_goal_first_step",
              "original_goal_ever", "original_goal_final", "excluded_from_baseline_denominator",
              "rollout_seconds", "torch_peak_reserved_bytes", "video_main", "video_wrist"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    for result in results:
        row = {k: result[k] for k in fields if k in result}
        requested = result["goals"][result["goal_id"]]
        original = result["goals"]["original_bowl"]
        row.update({"requested_goal_ever": requested["ever"], "requested_goal_final": requested["final"],
                    "requested_goal_first_step": requested["first_step"],
                    "original_goal_ever": original["ever"], "original_goal_final": original["final"],
                    "excluded_from_baseline_denominator": result["condition"] == "original_repeat"})
        writer.writerow(row)
    temporary = directory / "summary.csv.tmp"
    temporary.write_text(buffer.getvalue(), encoding="utf-8", newline="")
    temporary.replace(directory / "summary.csv")
