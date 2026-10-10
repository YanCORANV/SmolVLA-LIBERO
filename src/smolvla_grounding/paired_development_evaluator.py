"""Frozen 60-rollout development evaluation for the two paired training arms."""

from copy import deepcopy
import json
from pathlib import Path
import time

from .adaptation_records import latest_checkpoint
from .benchmark_records import BenchmarkJournal
from .development_evaluator import (
    _execute_dev_episode, load_jsonl, summarize_development_run, validate_development_prompts,
)
from .protocol import atomic_json, file_sha256, run_lock


ARMS = ("paired_bc", "paired_flow_consistency")
PROMPTS = ("original", "paraphrase_dev_01", "paraphrase_dev_02")
STATES = tuple(range(23, 33))
RUN_NAME = "task0_paired_flow_dev_eval"


def build_matrix(root, paired_config):
    matrix = deepcopy(json.loads((root / "configs/development_evaluation/dev_matrix.json").read_text(encoding="utf-8")))
    matrix["description"] = "Frozen paired-flow development selection: 2 arms x 10 states x 3 prompts."
    matrix["paired_training_config"] = "configs/paired_training.json"
    matrix["models"] = list(ARMS)
    matrix["statistical_rules"]["total_rollouts"] = 60
    matrix["model_selection_rule"] = deepcopy(paired_config["frozen_development_gate"])
    validate_scope(root, matrix, paired_config)
    return matrix


def validate_scope(root, matrix, paired_config):
    """Reject any non-development state, prompt, seed, model, task or runtime."""
    if matrix.get("paired_training_config") != "configs/paired_training.json":
        raise RuntimeError("Unexpected paired training configuration")
    if matrix.get("role") != "model_selection_development_evaluation" or matrix.get("suite") != "libero_spatial":
        raise RuntimeError("Unexpected development evaluation role or suite")
    if matrix.get("models") != list(ARMS) or len(matrix.get("tasks", [])) != 1:
        raise RuntimeError("Paired development models/tasks changed")
    task = matrix["tasks"][0]
    if task.get("task_id") != 0 or task.get("initial_states") != list(STATES):
        raise RuntimeError("Only Task 0 States 23-32 are permitted")
    catalog = "configs/development_evaluation/instructions/task0_dev.jsonl"
    if task.get("instruction_catalog") != catalog:
        raise RuntimeError("Only the frozen development prompt catalog is permitted")
    rows = load_jsonl(root / catalog)
    if [row["id"] for row in rows] != list(PROMPTS):
        raise RuntimeError("Non-development prompt requested")
    train_rows = load_jsonl(root / paired_config["training"]["train_catalog"])
    if rows[0]["text"] != train_rows[0]["text"]:
        raise RuntimeError("Development canonical text differs from frozen training catalog")
    validate_development_prompts(root, catalog)
    env, policy = matrix["environment"], matrix["policy"]
    if env.get("environment_seed") != 0 or env.get("episode_length") != 280:
        raise RuntimeError("Development environment seed/horizon changed")
    if policy.get("policy_seed") != 0 or policy.get("n_action_steps") != 1 or policy.get("num_steps") != 10 or policy.get("use_amp") is not False:
        raise RuntimeError("Development policy settings changed")
    if matrix["execution"].get("default_workers") != 1:
        raise RuntimeError("Development worker count changed")
    if matrix["statistical_rules"].get("primary_metric") != "goal_ever" or matrix["statistical_rules"].get("total_rollouts") != 60:
        raise RuntimeError("Development metric or rollout count changed")
    if paired_config["frozen_development_gate"]["evaluation_states"] != list(STATES) or paired_config["frozen_development_gate"]["prompts"] != list(PROMPTS):
        raise RuntimeError("Paired development gate scope changed")
    if matrix["model_selection_rule"] != paired_config["frozen_development_gate"]:
        raise RuntimeError("Development gate changed")
    return rows


def committed_manifest(root, paired_config, training_dir):
    models = {}
    for arm in ARMS:
        arm_dir = training_dir / arm
        identity = json.loads((arm_dir / "identity.json").read_text(encoding="utf-8"))
        if identity["arm"] != arm or identity["paired_config"] != paired_config:
            raise RuntimeError(f"Training identity changed for {arm}")
        journal = BenchmarkJournal(arm_dir, identity)
        result = latest_checkpoint(journal, 1000, 500)
        if result is None or result["step"] != 1000:
            raise RuntimeError(f"Missing valid committed 1000-update checkpoint for {arm}")
        checkpoint = arm_dir / result["checkpoint"]
        parameters = arm_dir / result["parameter_evidence"]
        rel_checkpoint = checkpoint.relative_to(root).as_posix()
        files = ("parameters.json", "pretrained_model/config.json", "pretrained_model/model.safetensors")
        hashes = {name: file_sha256(checkpoint / name) for name in files}
        if parameters != checkpoint / "parameters.json":
            raise RuntimeError("Parameter evidence path differs from committed checkpoint")
        models[arm] = {
            "role": arm,
            "checkpoint": rel_checkpoint,
            "parameters": parameters.relative_to(root).as_posix(),
            "training_updates": 1000,
            "peak_lr": paired_config["training"]["learning_rate"],
            "provenance": training_dir.relative_to(root).as_posix(),
            "artifact_sha256": hashes,
        }
    return {"manifest_version": 1, "evaluated_models": models}


def apply_gate(summary, paired_config):
    results = summary["results"]
    bc, consistency = results["paired_bc"], results["paired_flow_consistency"]
    baseline = paired_config["frozen_development_gate"]["H_baseline_original_language"]
    thresholds = paired_config["frozen_development_gate"]["conditions"]
    if thresholds["1_O_consistency_ge_080"]["threshold"] != 0.8 or thresholds["2_P_consistency_ge_045"]["threshold"] != 0.45 or baseline != 0.5538:
        raise RuntimeError("Frozen gate thresholds changed")
    conditions = {
        "O_consistency_ge_080": consistency["O"] >= 0.8,
        "P_consistency_ge_045": consistency["P"] >= 0.45,
        "H_consistency_gt_max_controls": consistency["H"] > max(baseline, bc["H"]),
    }
    summary["model_selection_gate"] = {
        "status": "PASS" if all(conditions.values()) else "FAIL",
        "conditions": conditions,
        "H_baseline_original_language": baseline,
        "H_paired_bc": bc["H"],
    }
    return summary


def run_paired_development_evaluation():
    root = Path(__file__).resolve().parents[2]
    paired_config = json.loads((root / "configs/paired_training.json").read_text(encoding="utf-8"))
    matrix = build_matrix(root, paired_config)
    rows = validate_scope(root, matrix, paired_config)
    training_dir = root / "outputs/adaptation/runs/task0_paired_flow_training"
    manifest = committed_manifest(root, paired_config, training_dir)
    output_dir = root / matrix["output_root"] / "runs" / RUN_NAME
    baseline_identity = json.loads((root / "outputs/language_pilot/identity.json").read_text(encoding="utf-8"))
    with run_lock(output_dir):
        meta_path = output_dir / "experiment_meta.json"
        meta = {"run_name": RUN_NAME, "matrix_config": matrix, "manifest": manifest}
        if meta_path.exists():
            saved = json.loads(meta_path.read_text(encoding="utf-8"))
            if {k: saved[k] for k in meta} != meta:
                raise RuntimeError("Existing development run has different provenance")
        else:
            atomic_json(meta_path, {**meta, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        count = 0
        for arm in ARMS:
            info = manifest["evaluated_models"][arm]
            for state in STATES:
                for row in rows:
                    count += 1
                    trial = output_dir / arm / "task_00" / f"state_{state:04d}_seed_0000_{row['id']}" / "attempt_001"
                    complete = trial / "complete.json"
                    if complete.exists():
                        saved = json.loads(complete.read_text(encoding="utf-8"))
                        expected = {"status": "COMPLETED", "model": arm, "task_id": 0,
                                    "initial_state_index": state, "instruction_id": row["id"], "policy_steps": 280}
                        if any(saved.get(key) != value for key, value in expected.items()):
                            raise RuntimeError(f"Invalid completed development rollout: {complete}")
                        print(f"[{count:02d}/60] cached {arm} state={state} prompt={row['id']}", flush=True)
                        continue
                    print(f"[{count:02d}/60] running {arm} state={state} prompt={row['id']}", flush=True)
                    trial.mkdir(parents=True, exist_ok=True)
                    _execute_dev_episode(root, matrix, manifest, arm, info, 0, state, row, trial, baseline_identity)
                    if not complete.exists():
                        raise RuntimeError(f"Rollout did not commit complete.json: {trial}")
        for arm in ARMS:
            for state in STATES:
                for row in rows:
                    complete = output_dir / arm / "task_00" / f"state_{state:04d}_seed_0000_{row['id']}" / "attempt_001/complete.json"
                    if not complete.exists():
                        raise RuntimeError(f"Incomplete development evaluation: {complete}")
        summary = apply_gate(summarize_development_run(output_dir), paired_config)
        atomic_json(output_dir / "summary.json", summary)
        print(f"Development gate: {summary['model_selection_gate']['status']}", flush=True)
        return summary