"""Amended Task-0 final confirmatory coordinator for 153 serial rollouts."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time

from .adaptation_records import latest_checkpoint, normalized_words
from .benchmark_records import BenchmarkJournal
from .development_evaluator import _execute_dev_episode, load_jsonl
from .final_confirmatory_summary import MODELS, PROMPTS, STATES, summarize_final_run
from .protocol import atomic_json, file_sha256, run_lock


MATRIX_PATH = "configs/final_confirmatory/final_matrix_v4.json"
MANIFEST_PATH = "configs/final_confirmatory/frozen_model_manifest_v4.json"
FREEZE_PATH = "configs/final_confirmatory/freeze_final_protocol_v4.json"
CATALOG_PATH = "configs/final_confirmatory/instructions/task0_final.jsonl"
OUTPUT_ROOT = "outputs/final_confirmatory"
RUN_NAME = "task0_paired_bc_final_confirmatory"
PRIOR_CATALOGS = (
    "configs/instructions/task0_train.jsonl",
    "configs/instructions/task0.jsonl",
    "configs/held_out_evaluation/instructions/task0_heldout.jsonl",
    "configs/development_evaluation/instructions/task0_dev.jsonl",
    "configs/final_confirmatory_reserved/instructions/task0_final_reserved.jsonl",
)


def _prompt_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_scope(root, matrix, manifest):
    """Pure file/config preflight. It does not construct a policy or LIBERO state."""
    root = Path(root)
    if matrix.get("role") != "final_confirmatory_evaluation" or matrix.get("suite") != "libero_spatial":
        raise RuntimeError("Final role or suite changed")
    if matrix.get("manifest_path") != MANIFEST_PATH or matrix.get("output_root") != OUTPUT_ROOT:
        raise RuntimeError("Final manifest or output location changed")
    if matrix.get("run_name") != RUN_NAME or matrix.get("models") != list(MODELS):
        raise RuntimeError("Final run name or model set changed")
    if len(matrix.get("tasks", [])) != 1:
        raise RuntimeError("Final matrix must have exactly one task")
    task = matrix["tasks"][0]
    if task.get("task_id") != 0 or task.get("initial_states") != list(STATES):
        raise RuntimeError("Only Task 0, States 33-49 are permitted")
    if set(task["initial_states"]) & set(range(33)):
        raise RuntimeError("Final states overlap consumed States 0-32")
    if task.get("instruction_catalog") != CATALOG_PATH:
        raise RuntimeError("Only the frozen new final instruction catalog is permitted")
    if file_sha256(root / CATALOG_PATH) != task.get("instruction_catalog_sha256"):
        raise RuntimeError("Final prompt catalog SHA256 mismatch")
    rows = load_jsonl(root / CATALOG_PATH)
    if [row.get("id") for row in rows] != list(PROMPTS):
        raise RuntimeError("Final prompt IDs or order changed")
    if [row.get("condition") for row in rows] != ["original", "paraphrase", "paraphrase"]:
        raise RuntimeError("Final prompt conditions changed")
    canonical = load_jsonl(root / PRIOR_CATALOGS[0])[0]["text"]
    if rows[0]["text"] != canonical:
        raise RuntimeError("Final canonical text differs from frozen training catalog")
    if any(row.get("goal_id") != "task_goal" for row in rows):
        raise RuntimeError("Final prompts do not all refer to the Task-0 goal")
    if len({normalized_words(row["text"]) for row in rows}) != 3:
        raise RuntimeError("Duplicate final prompt after normalization")
    prior = {normalized_words(row["text"]) for rel in PRIOR_CATALOGS
             for row in load_jsonl(root / rel)}
    if any(normalized_words(row["text"]) in prior for row in rows[1:]):
        raise RuntimeError("Final paraphrase overlaps a training, consumed, dev or retired prompt")
    for row in rows:
        if _prompt_sha256(row["text"]) != task["prompt_text_sha256"].get(row["id"]):
            raise RuntimeError(f"Final prompt text SHA256 mismatch: {row['id']}")
    expected_environment = {
        "obs_type": "pixels_agent_pos", "observation_height": 256,
        "observation_width": 256, "control_freq": 20,
        "control_mode": "relative", "hard_reset": True,
        "num_steps_wait": 10, "episode_length": 280,
        "environment_seed": 0,
    }
    expected_policy = {
        "n_action_steps": 1, "num_steps": 10,
        "use_amp": False, "policy_seed": 0,
    }
    if matrix.get("environment") != expected_environment or matrix.get("policy") != expected_policy:
        raise RuntimeError("Final runtime differs from validated LeRobot settings")
    if matrix.get("execution") != {
        "default_workers": 1, "total_rollouts": 153,
        "resume_marker": "complete.json",
    }:
        raise RuntimeError("Final execution count, worker or resume marker changed")
    rules = matrix.get("statistical_rules", {})
    required_rules = {
        "primary_metric": "goal_ever", "secondary_metric": "goal_final",
        "bootstrap_cluster_unit": "state_id", "bootstrap_replicates": 10000,
        "bootstrap_seed": 42, "confidence_level": 0.95,
    }
    if any(rules.get(key) != value for key, value in required_rules.items()):
        raise RuntimeError("Final statistical settings changed")
    if rules.get("headline_comparisons") != [
        "P_paired_bc - P_original_language",
        "P_paired_bc - P_augmented_language",
        "O_paired_bc - O_original_language",
        "O_paired_bc - O_augmented_language",
        "H_paired_bc versus H_original_language and H_augmented_language",
    ]:
        raise RuntimeError("Final headline comparisons changed")
    expected_keys = {(model, state, prompt) for model in MODELS
                     for state in STATES for prompt in PROMPTS}
    actual_keys = [(model, state, row["id"]) for model in matrix["models"]
                   for state in task["initial_states"] for row in rows]
    if len(actual_keys) != 153 or set(actual_keys) != expected_keys:
        raise RuntimeError("Final rollout matrix is not exactly 3 x 17 x 3")
    if set(manifest.get("evaluated_models", {})) != set(MODELS):
        raise RuntimeError("Final frozen manifest contains extra or missing models")
    if manifest.get("model_selection", {}).get("selected") != "paired_bc":
        raise RuntimeError("Selected development model changed")
    if manifest.get("model_selection", {}).get("gate_status") != "FAIL":
        raise RuntimeError("Development gate decision changed")
    return rows


def validate_frozen_artifacts(root, matrix, manifest):
    """Rehash all frozen sources, prompts and checkpoint artifacts before final states."""
    root = Path(root)
    rows = validate_scope(root, matrix, manifest)
    if file_sha256(root / MATRIX_PATH) != manifest["final_matrix_sha256"]:
        raise RuntimeError("Final matrix SHA256 mismatch")
    if file_sha256(root / CATALOG_PATH) != manifest["final_prompt_catalog_sha256"]:
        raise RuntimeError("Final catalog SHA256 mismatch")
    if manifest["final_prompt_text_sha256"] != matrix["tasks"][0]["prompt_text_sha256"]:
        raise RuntimeError("Final prompt hash registry mismatch")
    freeze = root / FREEZE_PATH
    if not freeze.is_file():
        raise RuntimeError("Missing final protocol freeze record")
    frozen = json.loads(freeze.read_text(encoding="utf-8"))
    if frozen.get("version") != 4:
        raise RuntimeError("Final protocol freeze version changed")
    if file_sha256(root / frozen["historical_v3_freeze_path"]) != frozen["historical_v3_freeze_sha256"]:
        raise RuntimeError("Historical v3 final freeze changed")
    training = frozen["training_provenance"]
    historical = training["historical_source_config_sha256"]
    if manifest["training_source_config_sha256"] != historical:
        raise RuntimeError("Historical paired-training hash registry changed")
    paired_freeze = root / training["historical_freeze_path"]
    if file_sha256(paired_freeze) != training["historical_freeze_sha256"]:
        raise RuntimeError("Historical paired-training freeze changed")
    paired_record = json.loads(paired_freeze.read_text(encoding="utf-8"))
    if any(paired_record["file_sha256"].get(rel) != expected for rel, expected in historical.items()):
        raise RuntimeError("Historical paired-training hashes disagree with paired freeze")
    excluded = {"src/smolvla_grounding/benchmark_runtime.py"}
    if (set(training["current_byte_assertions"]) != set(historical) - excluded
            or set(training["archival_only_hashes"]) != excluded
            or training["archival_only_hashes"] != {rel: historical[rel] for rel in excluded}):
        raise RuntimeError("Training provenance scope changed")
    for rel in training["current_byte_assertions"]:
        if file_sha256(root / rel) != historical[rel]:
            raise RuntimeError(f"Training source/config changed: {rel}")
    train_catalog = "configs/instructions/task0_train.jsonl"
    if manifest["training_instruction_catalog_sha256"] != file_sha256(root / train_catalog):
        raise RuntimeError("Training instruction catalog changed")
    schedule = "outputs/adaptation/runs/task0_paired_flow_training/schedule.json"
    if file_sha256(root / schedule) != manifest["training_schedule_sha256"]:
        raise RuntimeError("Training schedule bytes changed")
    from .adaptation_records import schedule_hash
    if schedule_hash(json.loads((root / schedule).read_text(encoding="utf-8"))) != manifest["training_schedule_content_hash"]:
        raise RuntimeError("Training schedule content hash changed")
    audit = "outputs/adaptation/runs/task0_paired_flow_training/paired_control_audit.json"
    if file_sha256(root / audit) != manifest["training_control_audit_sha256"]:
        raise RuntimeError("Paired training control audit changed")
    audit_data = json.loads((root / audit).read_text(encoding="utf-8"))
    if audit_data["all_updates_identical"] is not True or audit_data["updates_compared"] != 1000:
        raise RuntimeError("Paired update controls were not identical")
    baseline = json.loads((root / "configs/development_evaluation/dev_manifest.json").read_text(encoding="utf-8"))["evaluated_models"]
    for model in MODELS:
        info = manifest["evaluated_models"][model]
        checkpoint = root / info["checkpoint"]
        if not checkpoint.is_dir():
            raise RuntimeError(f"Frozen checkpoint missing: {checkpoint}")
        for rel, expected in info["checkpoint_artifact_sha256"].items():
            if file_sha256(checkpoint / rel) != expected:
                raise RuntimeError(f"Frozen checkpoint hash changed: {model}/{rel}")
        if any(info["artifact_sha256"].get(rel) != info["checkpoint_artifact_sha256"][rel]
               for rel in info["artifact_sha256"]):
            raise RuntimeError(f"Runtime checkpoint hashes differ from full freeze: {model}")
        if model in baseline:
            if checkpoint != root / baseline[model]["checkpoint"]:
                raise RuntimeError(f"Baseline checkpoint path changed: {model}")
            for rel, expected in baseline[model]["artifact_sha256"].items():
                if info["checkpoint_artifact_sha256"].get(rel) != expected:
                    raise RuntimeError(f"Historical baseline hash changed: {model}/{rel}")
    paired = manifest["evaluated_models"]["paired_bc"]
    arm_dir = root / "outputs/adaptation/runs/task0_paired_flow_training/paired_bc"
    identity = json.loads((arm_dir / "identity.json").read_text(encoding="utf-8"))
    if file_sha256(arm_dir / "identity.json") != manifest["paired_bc_identity_sha256"]:
        raise RuntimeError("Paired BC identity changed")
    journal = BenchmarkJournal(arm_dir, identity)
    committed = latest_checkpoint(journal, 1000, 500)
    if committed is None or committed["step"] != 1000:
        raise RuntimeError("Paired BC final checkpoint is not validly committed")
    if arm_dir / committed["checkpoint"] != root / paired["checkpoint"]:
        raise RuntimeError("Paired BC checkpoint path differs from journal")
    if committed["schedule_hash"] != manifest["training_schedule_content_hash"]:
        raise RuntimeError("Paired BC committed schedule hash changed")
    commit_path = root / paired["completion_record"]
    if file_sha256(commit_path) != paired["completion_record_sha256"]:
        raise RuntimeError("Paired BC completion record changed")
    for rel, expected in paired["committed_artifact_sha256"].items():
        if file_sha256(arm_dir / rel) != expected:
            raise RuntimeError(f"Paired BC committed artifact changed: {rel}")
    runtime = frozen["evaluation_runtime_provenance"]["file_sha256"]
    if "src/smolvla_grounding/benchmark_runtime.py" not in runtime:
        raise RuntimeError("Final runtime provenance omits benchmark_runtime.py")
    artifacts = frozen["final_protocol_artifacts"]
    if (artifacts["matrix_sha256"] != manifest["final_matrix_sha256"]
            or artifacts["catalog_sha256"] != manifest["final_prompt_catalog_sha256"]
            or artifacts["prompt_text_sha256"] != manifest["final_prompt_text_sha256"]
            or artifacts["checkpoint_artifact_sha256"] != {
                model: manifest["evaluated_models"][model]["checkpoint_artifact_sha256"]
                for model in MODELS
            }):
        raise RuntimeError("Final frozen artifact registry changed")
    if artifacts["manifest_sha256"] != file_sha256(root / MANIFEST_PATH):
        raise RuntimeError("Final v4 manifest SHA256 mismatch")
    validate_active_initial_states(root, matrix, frozen)
    for rel, expected in runtime.items():
        if file_sha256(root / rel) != expected:
            raise RuntimeError(f"Final evaluation runtime changed: {rel}")
    for rel, expected in artifacts["file_sha256"].items():
        if file_sha256(root / rel) != expected:
            raise RuntimeError(f"Final protocol artifact changed: {rel}")
    return rows


def _append_event(path, event, **values):
    record = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "event": event, **values}
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _completed_attempt(trial, model, state, prompt):
    commits = sorted(trial.glob("attempt_*/complete.json")) if trial.exists() else []
    if len(commits) > 1:
        raise RuntimeError(f"More than one committed final rollout: {trial}")
    if not commits:
        return None
    record = json.loads(commits[0].read_text(encoding="utf-8"))
    expected = {"status": "COMPLETED", "model": model, "task_id": 0,
                "initial_state_index": state, "instruction_id": prompt,
                "policy_steps": 280}
    if any(record.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Invalid committed final rollout: {commits[0]}")
    episode = commits[0].parent / "episode.json"
    if not episode.is_file():
        raise RuntimeError(f"Committed rollout is missing episode record: {episode}")
    saved = json.loads(episode.read_text(encoding="utf-8"))
    for key in ("model", "task_id", "initial_state_index", "instruction_id",
                "policy_steps", "goal_ever", "goal_final"):
        if saved.get(key) != record.get(key):
            raise RuntimeError(f"Final episode/commit mismatch: {episode}")
    return commits[0].parent


def _new_attempt(trial):
    trial.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        path = trial / f"attempt_{number:03d}"
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            number += 1



def validate_state_matrix_against_count(matrix, actual_count):
    """Reject any configured index that would wrap in LeRobot reset()."""
    requested = matrix["tasks"][0]["initial_states"]
    if not requested or any(type(index) is not int or index < 0 or index >= actual_count
                            for index in requested):
        raise RuntimeError(f"Final state matrix exceeds active initialization-state count {actual_count}")
    return actual_count


def validate_active_initial_states(root, matrix, frozen):
    """Load the exact source used by LiberoEnv, without constructing/resetting an environment."""
    from libero.libero import benchmark
    from lerobot.envs.libero import get_libero_path, get_task_init_states
    suite = benchmark.get_benchmark_dict()[matrix["suite"]]()
    task = suite.tasks[0]
    source = (Path(get_libero_path("init_states")) / task.problem_folder /
              Path(task.init_states_file).name).resolve()
    expected = frozen["active_initial_state_source"]
    if str(source) != expected["path"] or file_sha256(source) != expected["sha256"]:
        raise RuntimeError("Active Task-0 initialization-state source path or SHA256 changed")
    states = get_task_init_states(suite, 0)
    if tuple(states.shape) != tuple(expected["shape"]):
        raise RuntimeError("Active Task-0 initialization-state array shape changed")
    validate_state_matrix_against_count(matrix, len(states))
    return states


def _prepare_run_metadata(output_dir, expected_meta, frozen):
    """Create v4 metadata only after validating and preserving the 51 v3 commits."""
    v4_path = output_dir / "experiment_meta_v4.json"
    if v4_path.exists():
        current = json.loads(v4_path.read_text(encoding="utf-8"))
        if any(current.get(key) != value for key, value in expected_meta.items()):
            raise RuntimeError("Existing v4 final run has different frozen provenance")
        return v4_path
    v3_path = output_dir / "experiment_meta_v3.json"
    if not v3_path.is_file() or file_sha256(v3_path) != frozen["historical_v3_meta_sha256"]:
        raise RuntimeError("Historical v3 final metadata missing or changed")
    expected_hashes = frozen["historical_completed_artifact_sha256"]
    expected_commits = {output_dir / rel for rel in expected_hashes if rel.endswith("/complete.json")}
    if len(expected_commits) != 51:
        raise RuntimeError("Historical final commit registry is not exactly 51 rollouts")
    actual_commits = set(output_dir.glob("*/task_00/state_*/attempt_*/complete.json"))
    if actual_commits != expected_commits:
        raise RuntimeError("Existing final commits differ from the 51 compatible v3 rollouts")
    for rel, expected in expected_hashes.items():
        if file_sha256(output_dir / rel) != expected:
            raise RuntimeError(f"Historical completed final artifact changed: {rel}")
    for state in range(33, 50):
        for prompt in PROMPTS:
            trial = output_dir / "original_language" / "task_00" / f"state_{state:04d}_seed_0000_{prompt}"
            if _completed_attempt(trial, "original_language", state, prompt) is None:
                raise RuntimeError(f"Missing valid historical completed rollout: {trial}")
    atomic_json(v4_path, {
        **expected_meta,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "historical_v3_meta_sha256": frozen["historical_v3_meta_sha256"],
        "historical_completed_rollouts": 51,
        "amendment": "States 33-52 reduced to valid indices 33-49 before inspecting intermediate success rates",
    })
    return v4_path


def run_final_confirmatory_evaluation():
    root = Path(__file__).resolve().parents[2]
    matrix = json.loads((root / MATRIX_PATH).read_text(encoding="utf-8"))
    manifest = json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8"))
    print("[Final preflight] validating matrix, prompts, sources and checkpoint hashes", flush=True)
    rows = validate_frozen_artifacts(root, matrix, manifest)
    print("[Final preflight] passed: Task 0, States 33-49, three frozen models, three frozen prompts, 153 rollouts", flush=True)
    output_dir = root / OUTPUT_ROOT / "runs" / RUN_NAME
    baseline_identity = json.loads((root / "outputs/language_pilot/identity.json").read_text(encoding="utf-8"))
    with run_lock(output_dir):
        expected_meta = {
            "run_name": RUN_NAME,
            "matrix_config": matrix,
            "manifest": manifest,
            "matrix_sha256": file_sha256(root / MATRIX_PATH),
            "manifest_sha256": file_sha256(root / MANIFEST_PATH),
            "catalog_sha256": file_sha256(root / CATALOG_PATH),
            "freeze_record_sha256": file_sha256(root / FREEZE_PATH),
        }
        frozen = json.loads((root / FREEZE_PATH).read_text(encoding="utf-8"))
        _prepare_run_metadata(output_dir, expected_meta, frozen)
        log = output_dir / "execution.jsonl"
        _append_event(log, "run_started")
        ordinal = 0
        for model in MODELS:
            info = manifest["evaluated_models"][model]
            for state in STATES:
                for row in rows:
                    ordinal += 1
                    prompt = row["id"]
                    trial = output_dir / model / "task_00" / f"state_{state:04d}_seed_0000_{prompt}"
                    completed = _completed_attempt(trial, model, state, prompt)
                    if completed is not None:
                        print(f"[{ordinal:03d}/153] cached {model} state={state} prompt={prompt}", flush=True)
                        _append_event(log, "rollout_cached", ordinal=ordinal, model=model, state=state, prompt=prompt)
                        continue
                    attempt = _new_attempt(trial)
                    print(f"[{ordinal:03d}/153] running {model} state={state} prompt={prompt}", flush=True)
                    _append_event(log, "rollout_started", ordinal=ordinal, model=model,
                                  state=state, prompt=prompt, attempt=attempt.name)
                    _execute_dev_episode(root, matrix, manifest, model, info, 0, state,
                                         row, attempt, deepcopy(baseline_identity))
                    if _completed_attempt(trial, model, state, prompt) != attempt:
                        raise RuntimeError(f"Final rollout did not commit complete.json: {attempt}")
                    _append_event(log, "rollout_completed", ordinal=ordinal, model=model,
                                  state=state, prompt=prompt, attempt=attempt.name)
        validate_frozen_artifacts(root, matrix, manifest)
        report = summarize_final_run(output_dir)
        _append_event(log, "run_finished", completed_rollouts=report["completed_rollouts"])
        print("[Final] 153/153 valid rollouts completed; aggregate report written", flush=True)
        return report