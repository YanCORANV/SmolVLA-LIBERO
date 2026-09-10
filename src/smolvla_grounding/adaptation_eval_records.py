"""Checkpoint provenance and complete-group comparisons without GPU imports."""

import csv
import io
import json
from pathlib import Path

from .adaptation_records import compare_training_logs, read_jsonl, schedule_hash
from .benchmark_records import BenchmarkJournal, group_summary
from .protocol import atomic_json, file_sha256, fingerprint

MODELS = ["untuned_fp32", "original_language", "augmented_language"]


def read_training(root, settings):
    directory = Path(root) / settings["training_run"]
    report = json.loads((directory / "comparison.json").read_text())
    if not report["passed"] or settings["models"] != MODELS:
        raise ValueError("Expected a completed paired run and all three models")
    schedule = json.loads((directory / "schedule.json").read_text())
    if schedule_hash(schedule) != report["schedule_hash"]:
        raise ValueError("Training schedule changed")
    initials, logs, models = [], [], {}
    for arm in MODELS[1:]:
        arm_dir, result = directory / arm, report["arms"][arm]
        identity = json.loads((arm_dir / "identity.json").read_text())
        key = f"step_{report['updates_per_arm']:06d}"
        marker_path = arm_dir / key / "complete.json"
        marker = json.loads(marker_path.read_text())
        if marker["fingerprint"] != fingerprint(identity) or marker["result"] != result:
            raise ValueError("Final checkpoint identity/result changed")
        # Small evidence is checked now; large checkpoint hashes are checked by
        # the user-run worker immediately before loading the selected policy.
        for name in ("initial_parameters.json", result["training_log"], result["parameter_evidence"]):
            if file_sha256(arm_dir / name) != marker["artifacts"][name]:
                raise ValueError("Training evidence changed: " + name)
        initials.append(json.loads((arm_dir / "initial_parameters.json").read_text()))
        logs.append(read_jsonl(arm_dir / result["training_log"]))
        models[arm] = {"arm_directory": arm_dir.relative_to(root).as_posix(), "key": key,
                       "marker_sha256": file_sha256(marker_path),
                       "checkpoint": (arm_dir / result["checkpoint"] / "pretrained_model").relative_to(root).as_posix(),
                       "parameters": (arm_dir / result["parameter_evidence"]).relative_to(root).as_posix()}
    controls = compare_training_logs(*logs, schedule)
    controls["same_initial_parameter_bytes"] = initials[0] == initials[1]
    if not controls["same_initial_parameter_bytes"] or controls != report["controls"]:
        raise ValueError("Paired training controls changed")
    models["untuned_fp32"] = {"parameters": (directory / MODELS[1] / "initial_parameters.json").relative_to(root).as_posix()}
    return {"comparison_sha256": file_sha256(directory / "comparison.json"), "models": models,
            "updates_per_arm": report["updates_per_arm"], "controls": controls}


def verify_training_checkpoint(root, model):
    directory = Path(root) / model["arm_directory"]
    if file_sha256(directory / model["key"] / "complete.json") != model["marker_sha256"]:
        raise RuntimeError("Selected training checkpoint marker changed")
    journal = BenchmarkJournal(directory, json.loads((directory / "identity.json").read_text()))
    if journal.completed(model["key"]) is None:
        raise RuntimeError("Training checkpoint incomplete")


def compare_models(results, experiment, rows):
    """Only aggregate cases complete under every model and every instruction."""
    if set(results) != set(MODELS):
        raise ValueError("Expected exactly three model result sets")
    indices = {}
    for model in MODELS:
        group_summary(results[model], experiment, rows)  # rejects duplicate/foreign cases
        indices[model] = {(r["initial_state_index"], r["policy_seed"], r["instruction_id"]): r for r in results[model]}
    complete = []
    for state in experiment["initial_states"]:
        for seed in experiment["policy_seeds"]:
            keys = [(state, seed, row["id"]) for row in rows]
            if all(k in indices[m] for m in MODELS for k in keys):
                complete.extend(keys)
    selected = {m: [indices[m][k] for k in complete] for m in MODELS}
    # Require matching physical observations and initial random state, not just
    # matching labels. Repeat wording must also share these group references.
    references = {}
    for key in complete:
        group = key[:2]
        for model in MODELS:
            r = indices[model][key]
            evidence = (r["initial_hashes"], r["initial_rng_hashes"])
            if group in references and evidence != references[group]:
                raise ValueError("Models/conditions have unmatched initial state or RNG")
            references[group] = evidence
            if (r["policy_steps"] != experiment["horizon"] or r["environment_seed"] != experiment["environment_seed"]
                    or r["suite"] != experiment["suite"] or r["task_id"] != experiment["task_id"]):
                raise ValueError("Episode protocol differs across models")
    summaries = {m: group_summary(selected[m], experiment, rows) for m in MODELS}
    pairs = []
    for key in complete:
        if indices[MODELS[0]][key]["condition"] == "original_repeat":
            continue
        record = dict(zip(("initial_state_index", "policy_seed", "instruction_id"), key))
        for metric in ("ever", "final"):
            scores = {m: int(indices[m][key]["goals"]["original_bowl"][metric]) for m in MODELS}
            record.update({m + "_" + metric: v for m, v in scores.items()})
            for arm in MODELS[1:]:
                record[arm + "_minus_untuned_" + metric] = scores[arm] - scores[MODELS[0]]
            record["augmented_minus_original_tuning_" + metric] = scores[MODELS[2]] - scores[MODELS[1]]
        pairs.append(record)
    deltas = {}
    for row in rows:
        if row["condition"] == "original_repeat":
            continue
        subset = [p for p in pairs if p["instruction_id"] == row["id"]]
        fields = [k for k in subset[0] if "_minus_" in k] if subset else []
        deltas[row["id"]] = {"matched_cases": len(subset),
                              **{k: sum(p[k] for p in subset) / len(subset) for k in fields}}
    return {"all_complete": all(s["all_complete"] for s in summaries.values()),
            "completed_matched_groups": len(references), "models": summaries,
            "paired_differences": deltas, "paired_cases": pairs,
            "scope": "Task-0 development evaluation; 280-step fixed horizon, native original goal. No untouched-test or standard LIBERO score claimed.",
            "precision": "All models use FP32 expert storage and identical inference settings; earlier native BF16 pilot is separate."}


def export_comparison(directory, results, experiment, rows):
    report = compare_models(results, experiment, rows)
    atomic_json(Path(directory) / "comparison.json", report)
    pairs = report["paired_cases"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(pairs[0]) if pairs else ["initial_state_index", "policy_seed", "instruction_id"])
    writer.writeheader()
    writer.writerows(pairs)
    path = Path(directory) / "paired_cases.csv.tmp"
    path.write_text(stream.getvalue(), encoding="utf-8", newline="")
    path.replace(Path(directory) / "paired_cases.csv")
    return report
