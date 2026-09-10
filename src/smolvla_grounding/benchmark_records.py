"""Interactive experiment selection and complete-group descriptive summaries."""

from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import re

from .protocol import Checkpoints, atomic_json, fingerprint


def parse_indices(text):
    parts = [part.strip() for part in text.split(",")]
    if not parts or any(not re.fullmatch(r"[0-9]+", part) for part in parts):
        raise ValueError("Enter comma-separated nonnegative integers")
    values = [int(part) for part in parts]
    if len(set(values)) != len(values):
        raise ValueError("Duplicate indices/seeds are not allowed")
    if any(v >= 2**32 for v in values):
        raise ValueError("Values must be below 2**32")
    return sorted(values)


def ask_indices(prompt, default):
    while True:
        try:
            return parse_indices(input(f"{prompt} [{default}]: ").strip() or default)
        except ValueError as exc:
            print(str(exc), flush=True)


def validate_run_name(name):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name):
        raise ValueError("Use a run name containing only letters, digits, underscores or hyphens")
    return name


def validate_experiment(experiment, baseline):
    config = baseline["config"]
    fixed = {"environment_seed": config["environment_seed"], "suite": config["suite"], "task_id": config["task_id"],
             "horizon": config["environment"]["episode_length"],
             "instruction_order": [r["id"] for r in baseline["instructions"]]}
    if any(experiment.get(k) != v for k, v in fixed.items()):
        raise ValueError("Saved experiment changes a fixed pilot setting")
    for name in ("initial_states", "policy_seeds"):
        values = experiment[name]
        if not isinstance(values, list) or parse_indices(",".join(str(v) for v in values)) != values:
            raise ValueError("Experiment state/seed lists must be sorted unique nonnegative integers")


class BenchmarkJournal(Checkpoints):
    """Retain the validated checkpoint methods with a benchmark-specific constructor.

    The caller obtains interactive resume authorization before expensive imports.
    Existing pilot checkpoint code stays unchanged for source-hash reproducibility.
    """

    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.fingerprint = fingerprint(identity)
        path = self.directory / "identity.json"
        if path.exists():
            if json.loads(path.read_text()) != identity:
                raise RuntimeError("Saved benchmark provenance differs. Keep this run and choose a new run name.")
        else:
            atomic_json(path, identity)


def trial_id(state, seed, instruction):
    return f"state_{state:04d}_seed_{seed:04d}_{instruction}"


def group_summary(results, experiment, instructions):
    expected_ids = [r["id"] for r in instructions]
    groups = defaultdict(dict)
    expected_groups = {(s, p) for s in experiment["initial_states"] for p in experiment["policy_seeds"]}
    for result in results:
        key = (result["initial_state_index"], result["policy_seed"])
        name = result["instruction_id"]
        if key not in expected_groups or name not in expected_ids or name in groups[key]:
            raise ValueError("Unexpected or duplicate trial in results")
        expected = next(r for r in instructions if r["id"] == name)
        if result["condition"] != expected["condition"] or result["goal_id"] != expected["goal_id"]:
            raise ValueError("Result condition/goal differs from the instruction catalog")
        groups[key][name] = result
    complete = {k: g for k, g in groups.items() if set(g) == set(expected_ids)}
    original_id = next(r["id"] for r in instructions if r["condition"] == "original")
    repeat_id = next(r["id"] for r in instructions if r["condition"] == "original_repeat")
    paraphrase_ids = [r["id"] for r in instructions if r["condition"] == "paraphrase"]

    def outcome(result, which="ever"):
        return bool(result["goals"][result["goal_id"]][which])

    conditions = {}
    for row in instructions:
        selected = [g[row["id"]] for g in complete.values()]
        n = len(selected)
        conditions[row["id"]] = {
            "condition": row["condition"], "episodes": n,
            "requested_goal_ever_count": sum(outcome(r) for r in selected),
            "requested_goal_final_count": sum(outcome(r, "final") for r in selected),
            "requested_goal_ever_fraction": sum(outcome(r) for r in selected) / n if n else None,
            "original_goal_ever_count": sum(bool(r["goals"]["original_bowl"]["ever"]) for r in selected),
            "alternative_goal_validation": "provisional" if row["condition"] == "semantic_change" else "native predicate",
        }
    pairs = []
    for (state, seed), group in sorted(complete.items()):
        baseline = outcome(group[original_id])
        repeat = outcome(group[repeat_id])
        pairs.append({"initial_state_index": state, "policy_seed": seed,
                      "original_ever": baseline, "repeat_ever": repeat,
                      "original_repeat_ever_disagree": baseline != repeat,
                      "original_repeat_final_disagree": outcome(group[original_id], "final") != outcome(group[repeat_id], "final"),
                      "paraphrase_mean_ever": sum(outcome(group[n]) for n in paraphrase_ids) / len(paraphrase_ids),
                      "paraphrase_minus_original": sum(outcome(group[n]) for n in paraphrase_ids) / len(paraphrase_ids) - int(baseline)})
    n = len(pairs)
    return {"completed_episodes": len(results), "expected_episodes": len(expected_groups) * len(expected_ids),
            "completed_groups": n, "expected_groups": len(expected_groups),
            "all_complete": n == len(expected_groups),
            "aggregation_scope": "Only complete state/policy-seed groups; incomplete groups are not counted as failures.",
            "standard_early_stopping_evaluation": False, "per_instruction": conditions, "paired_groups": pairs,
            "mean_paired_paraphrase_minus_original": sum(p["paraphrase_minus_original"] for p in pairs) / n if n else None,
            "repeat_ever_disagreement_groups": sum(p["original_repeat_ever_disagree"] for p in pairs),
            "repeat_final_disagreement_groups": sum(p["original_repeat_final_disagree"] for p in pairs),
            "limitations": ["One task; multiple states/seeds are not independent tasks.",
                            "Paraphrases share underlying states; fractions are descriptive, not independent-sample inference.",
                            "No confidence intervals, significance tests or positive alternative-goal validation are supplied."]}


def export_benchmark(directory, results, experiment, instructions):
    directory = Path(directory)
    summary = group_summary(results, experiment, instructions)
    order = {row["id"]: index for index, row in enumerate(instructions)}
    results = sorted(results, key=lambda r: (r["initial_state_index"], r["policy_seed"], order[r["instruction_id"]]))
    atomic_json(directory / "summary.json", summary)
    atomic_json(directory / "episodes.json", results)
    columns = ["initial_state_index", "environment_seed", "policy_seed", "instruction_id", "condition", "instruction",
               "goal_id", "policy_steps", "requested_goal_ever", "requested_goal_final", "original_goal_ever",
               "repeat_control", "rollout_seconds", "torch_peak_reserved_bytes", "video_main", "video_wrist"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for result in results:
        row = {k: result[k] for k in columns if k in result}
        requested = result["goals"][result["goal_id"]]
        row.update(requested_goal_ever=requested["ever"], requested_goal_final=requested["final"],
                   original_goal_ever=result["goals"]["original_bowl"]["ever"],
                   repeat_control=result["condition"] == "original_repeat")
        writer.writerow(row)
    path = directory / "episodes.csv.tmp"
    path.write_text(stream.getvalue(), encoding="utf-8", newline="")
    path.replace(directory / "episodes.csv")
    return summary
