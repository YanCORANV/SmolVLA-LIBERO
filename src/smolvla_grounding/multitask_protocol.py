"""Configuration and full-matrix preflight for multi-task LIBERO evaluations.

This module never constructs or resets an environment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .protocol import file_sha256, fingerprint

SUITE_HORIZONS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def prompt_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError(f"Expected repository-relative path: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError(f"Path leaves repository: {relative}")
    return path


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_catalog(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"Empty instruction catalog: {path}")
    return rows


def _inspect_task_asset(suite_name, task_id):
    """Use the same init-state loader as the actual LeRobot environment."""
    from libero.libero import benchmark
    from lerobot.envs.libero import get_libero_path, get_task_init_states

    registry = benchmark.get_benchmark_dict()
    if suite_name not in registry:
        raise RuntimeError(f"Unknown LIBERO suite: {suite_name}")
    suite = registry[suite_name]()
    if type(task_id) is not int or task_id < 0 or task_id >= suite.n_tasks:
        raise RuntimeError(f"Unknown task ID: {suite_name}/{task_id}")
    task = suite.get_task(task_id)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init = Path(get_libero_path("init_states")) / task.problem_folder / Path(task.init_states_file).name
    if not bddl.is_file() or not init.is_file():
        raise RuntimeError(f"Missing BDDL or init-state asset: {suite_name}/{task_id}")
    count = len(get_task_init_states(suite, task_id))
    return {
        "name": task.name, "language": task.language, "state_count": count,
        "bddl": bddl.resolve(), "init": init.resolve(),
    }


def _resolve_backbone_snapshot(backbone):
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(backbone["repo_id"], "config.json",
                                    revision=backbone["revision"])
    if not isinstance(cached, str):
        raise RuntimeError("Pinned backbone snapshot is unavailable")
    return Path(cached).parent.resolve()


def make_rows(matrix, catalogs):
    rows = []
    for model in matrix["models"]:
        for task in matrix["tasks"]:
            key = (task["suite"], task["task_id"])
            for state in task["states"]:
                for prompt in catalogs[key]:
                    rows.append((model, key[0], key[1], state, prompt["id"]))
    if len(rows) != len(set(rows)):
        raise RuntimeError("Duplicate rollout combination in matrix")
    if len(rows) != matrix["expected_rollouts"]:
        raise RuntimeError(f"Expected {matrix['expected_rollouts']} rollouts, found {len(rows)}")
    return rows


def trial_path(output_dir, row, policy_seed):
    model, suite, task_id, state, prompt = row
    return (output_dir / model / suite / f"task_{task_id:02d}" /
            f"state_{state:04d}_seed_{policy_seed:04d}_{prompt}")


def _valid_completion(trial, row, horizon, identity):
    commits = sorted(trial.glob("attempt_*/complete.json")) if trial.exists() else []
    if len(commits) > 1:
        raise RuntimeError(f"Multiple committed attempts: {trial}")
    if not commits:
        return None
    path = commits[0]
    record = _read_json(path)
    model, suite, task_id, state, prompt = row
    expected = {
        "status": "COMPLETED", "model": model, "suite": suite,
        "task_id": task_id, "state": state, "prompt_id": prompt,
        "policy_steps": horizon, "identity": identity,
    }
    if any(record.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f"Invalid complete.json: {path}")
    if type(record.get("goal_ever")) is not bool or type(record.get("goal_final")) is not bool:
        raise RuntimeError(f"Invalid success fields: {path}")
    artifacts = record.get("artifacts")
    required = {"initial_hashes.json", "initial_state.npz", "trajectory.npz",
                "steps.jsonl", "inference.jsonl", "episode.json"}
    if not isinstance(artifacts, dict) or not required.issubset(artifacts):
        raise RuntimeError(f"Missing committed artifact hashes: {path}")
    for name, expected_hash in artifacts.items():
        artifact = _inside(path.parent, name)
        if not artifact.is_file() or file_sha256(artifact) != expected_hash:
            raise RuntimeError(f"Corrupt committed artifact: {artifact}")
    episode = _read_json(path.parent / "episode.json")
    for key in ("model", "suite", "task_id", "state", "prompt_id",
                "policy_steps", "goal_ever", "goal_final"):
        if episode.get(key) != record.get(key):
            raise RuntimeError(f"Episode/commit mismatch: {path}")
    return path.parent


def new_attempt(trial):
    trial.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        path = trial / f"attempt_{number:03d}"
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            number += 1


def preflight(root, matrix_rel, manifest_rel):
    """Production preflight, also called directly by regression tests."""
    root = Path(root).resolve()
    matrix_path = _inside(root, matrix_rel)
    manifest_path = _inside(root, manifest_rel)
    matrix, manifest = _read_json(matrix_path), _read_json(manifest_path)
    role = matrix.get("role")
    if role not in {"canonical_competence_screen", "multitask_language_robustness"}:
        raise RuntimeError(f"Unsupported multi-task protocol role: {role}")
    if manifest.get("matrix_sha256") != file_sha256(matrix_path):
        raise RuntimeError("Matrix hash mismatch")
    for rel, expected in manifest.get("source_sha256", {}).items():
        if file_sha256(_inside(root, rel)) != expected:
            raise RuntimeError(f"Evaluator source hash mismatch: {rel}")
    if not manifest.get("source_sha256"):
        raise RuntimeError("Missing evaluator source hashes")
    if not matrix.get("models") or not matrix.get("tasks"):
        raise RuntimeError("Multi-task matrix needs at least one model and task")
    if len(matrix["models"]) != len(set(matrix["models"])):
        raise RuntimeError("Duplicate model in matrix")
    if role == "canonical_competence_screen":
        if matrix["models"] != ["untuned_fp32"] or len(matrix["tasks"]) != 6:
            raise RuntimeError("Screen requires one official model and six registered tasks")
    if matrix.get("execution", {}).get("workers") != 1:
        raise RuntimeError("Screen requires worker=1")
    policy = matrix.get("policy", {})
    if policy != {"policy_seed": 0, "n_action_steps": 1, "num_steps": 10, "use_amp": False}:
        raise RuntimeError("Unexpected policy settings")
    env = matrix.get("environment", {})
    expected_env = {"environment_seed": 0, "obs_type": "pixels_agent_pos",
                    "observation_height": 256, "observation_width": 256,
                    "control_freq": 20, "control_mode": "relative",
                    "hard_reset": True, "num_steps_wait": 10}
    if env != expected_env:
        raise RuntimeError("Unexpected environment settings")
    if role == "canonical_competence_screen":
        if matrix.get("eligibility") != {"minimum_successes": 3, "trials_per_task": 5}:
            raise RuntimeError("Screening gate must be frozen at 3/5")
        if matrix.get("expected_rollouts") != 30:
            raise RuntimeError("Screen requires exactly 30 rollouts")
    if matrix.get("primary_success") != "goal_ever":
        raise RuntimeError("Primary success must be goal_ever")

    models = manifest.get("models", {})
    if set(models) != set(matrix["models"]):
        raise RuntimeError("Model registry does not match matrix")
    for model_id in matrix["models"]:
        info = models[model_id]
        if info.get("kind") != "official_smolvla" or not info.get("artifact_sha256"):
            raise RuntimeError(f"Invalid model registration: {model_id}")
        directory = _inside(root, info["path"])
        if not directory.is_dir():
            raise RuntimeError(f"Missing checkpoint: {directory}")
        for rel, expected in info["artifact_sha256"].items():
            if file_sha256(_inside(directory, rel)) != expected:
                raise RuntimeError(f"Checkpoint hash mismatch: {model_id}/{rel}")

    backbone = manifest.get("backbone", {})
    if not backbone.get("artifact_sha256"):
        raise RuntimeError("Missing pinned backbone hashes")
    snapshot = _resolve_backbone_snapshot(backbone)
    for rel, expected in backbone["artifact_sha256"].items():
        relative = Path(rel)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe backbone artifact path: {rel}")
        # Hugging Face snapshot entries legitimately link to sibling blobs/.
        # Their expected content hash still pins the resolved bytes.
        if file_sha256(snapshot / relative) != expected:
            raise RuntimeError(f"Backbone hash mismatch: {rel}")

    catalogs = {}
    seen_tasks = set()
    for task in matrix["tasks"]:
        suite_name, task_id = task["suite"], task["task_id"]
        key = (suite_name, task_id)
        if key in seen_tasks:
            raise RuntimeError(f"Duplicate task: {key}")
        seen_tasks.add(key)
        if suite_name not in SUITE_HORIZONS or task.get("horizon") != SUITE_HORIZONS[suite_name]:
            raise RuntimeError(f"Invalid or missing task horizon: {key}")
        asset = _inspect_task_asset(suite_name, task_id)
        if asset["name"] != task.get("task_name") or asset["language"] != task.get("canonical"):
            raise RuntimeError(f"Official task identity or instruction mismatch: {key}")
        expected_asset = manifest.get("assets", {}).get(f"{suite_name}:{task_id}")
        if not isinstance(expected_asset, dict):
            raise RuntimeError(f"Missing frozen asset record: {key}")
        for kind in ("bddl", "init"):
            if str(asset[kind]) != expected_asset.get(f"{kind}_path"):
                raise RuntimeError(f"Task asset path mismatch: {key}/{kind}")
            if file_sha256(asset[kind]) != expected_asset.get(f"{kind}_sha256"):
                raise RuntimeError(f"Task asset hash mismatch: {key}/{kind}")
        if asset["state_count"] != expected_asset.get("state_count"):
            raise RuntimeError(f"Init-state count mismatch: {key}")
        if any(type(i) is not int or i < 0 or i >= asset["state_count"] for i in task["states"]):
            raise RuntimeError(f"Out-of-range state index: {key}")
        if not task["states"] or len(task["states"]) != len(set(task["states"])):
            raise RuntimeError(f"Empty or duplicate task state set: {key}")
        if role == "canonical_competence_screen" and task["states"] != [0, 1, 2, 3, 4]:
            raise RuntimeError(f"Screen states must be 0–4: {key}")
        catalog_path = _inside(root, task["prompt_catalog"])
        catalog_hash = manifest.get("catalog_sha256", {}).get(task["prompt_catalog"])
        if file_sha256(catalog_path) != catalog_hash:
            raise RuntimeError(f"Prompt catalog hash mismatch: {key}")
        prompts = _read_catalog(catalog_path)
        expected_ids = (["original"] if role == "canonical_competence_screen"
                        else ["original", "paraphrase_01", "paraphrase_02"])
        if [prompt.get("id") for prompt in prompts] != expected_ids:
            raise RuntimeError(f"Wrong prompt IDs or order: {key}")
        if prompts[0].get("text") != task["canonical"] or prompts[0].get("condition") != "original":
            raise RuntimeError(f"Wrong canonical prompt: {key}")
        normalized = [" ".join(prompt["text"].lower().split()) for prompt in prompts]
        if len(normalized) != len(set(normalized)):
            raise RuntimeError(f"Duplicate normalized prompt: {key}")
        for prompt in prompts:
            if not prompt.get("text") or prompt["text"] != prompt["text"].strip():
                raise RuntimeError(f"Invalid prompt text: {key}")
            prompt_key = f"{suite_name}:{task_id}:{prompt['id']}"
            if prompt_sha256(prompt["text"]) != manifest.get("prompt_sha256", {}).get(prompt_key):
                raise RuntimeError(f"Prompt text hash mismatch: {prompt_key}")
        catalogs[key] = prompts

    rows = make_rows(matrix, catalogs)
    output_dir = _inside(root, matrix["output_dir"])
    identity = fingerprint({"matrix_sha256": file_sha256(matrix_path),
                            "manifest_sha256": file_sha256(manifest_path)})
    metadata = output_dir / "experiment_meta.json"
    if output_dir.exists():
        if metadata.exists():
            prior = _read_json(metadata)
            if prior.get("identity") != identity:
                raise RuntimeError("Existing output directory belongs to another protocol")
        elif any(output_dir.iterdir()):
            raise RuntimeError("Output directory collision without experiment metadata")
    for row in rows:
        task = next(t for t in matrix["tasks"] if (t["suite"], t["task_id"]) == row[1:3])
        _valid_completion(trial_path(output_dir, row, policy["policy_seed"]),
                          row, task["horizon"], identity)
    return {"matrix": matrix, "manifest": manifest, "catalogs": catalogs,
            "rows": rows, "output_dir": output_dir, "identity": identity, "snapshot": snapshot}





