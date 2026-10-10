"""Reusable pinned official runtime for state benchmarks; no upstream model copies."""

from collections import Counter
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

from .protocol import file_sha256

FINAL_CONFIRMATORY_FREEZE_PATH = "configs/final_confirmatory/freeze_final_protocol_v4.json"
from .rollout_records import verify_pairing


FROZEN_CONFIRMATORY_INSTRUCTIONS = {
    "original": "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "original_repeat": "pick up the black bowl between the plate and the ramekin and place it on the plate",
    "paraphrase_heldout_01": "transfer into the plate the black bowl that sits between the plate and the ramekin",
    "paraphrase_heldout_02": "take the black bowl located between the ramekin and the plate, and set it down on the plate",
}
FROZEN_CONFIRMATORY_MODELS = ["untuned_fp32", "original_language", "augmented_language"]
FROZEN_CONFIRMATORY_STATES = set(range(3, 23))  # States 3-22 (20 states)


def verify_confirmatory_pairing(root, config, rows, confirmatory_spec):
    """
    Validate runtime configuration and instruction rows against the frozen confirmatory specification.
    Does NOT compare states/instructions against historical pilot identity.
    Enforces frozen matrix, manifest, task 0, states 3-22, 4 prompt texts, 3 models, seeds, and 280 steps.
    """
    root = Path(root)

    # 1. Resolve matrix and manifest
    matrix = confirmatory_spec.get("matrix")
    if matrix is None:
        matrix_path = root / confirmatory_spec.get("matrix_path", "configs/held_out_evaluation/confirmatory_matrix.json")
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))

    manifest = confirmatory_spec.get("manifest")
    if manifest is None:
        manifest_path = root / matrix.get("manifest_path", "configs/held_out_evaluation/frozen_training_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    model_id = confirmatory_spec.get("model_id") or config.get("model_id")

    # 2. Scope & matrix role
    if matrix.get("role") != "primary_confirmatory_experiment":
        raise RuntimeError(f"Confirmatory matrix role must be 'primary_confirmatory_experiment', got '{matrix.get('role')}'")
    if matrix.get("suite") != "libero_spatial" or config.get("suite") != "libero_spatial":
        raise RuntimeError("Confirmatory suite must be 'libero_spatial'")

    # 3. Task ID: Task 0 only
    task_cfgs = matrix.get("tasks", [])
    if not task_cfgs or task_cfgs[0].get("task_id") != 0:
        raise RuntimeError("Confirmatory matrix must specify Task 0")
    if config.get("task_id") != 0:
        raise RuntimeError(f"Confirmatory runtime task ID must be 0, got {config.get('task_id')}")

    # 4. State IDs: 3-22 only, no development states (0, 1, 2)
    matrix_states = set(task_cfgs[0].get("initial_states", []))
    if matrix_states != FROZEN_CONFIRMATORY_STATES:
        raise RuntimeError(f"Confirmatory matrix states must be exactly states 3-22, got {matrix_states}")

    dev_states = set(manifest.get("development_boundary", {}).get("development_states", [0, 1, 2]))
    init_state = config.get("init_state_index")
    if init_state is None:
        raise RuntimeError("Runtime config missing init_state_index")
    if init_state not in FROZEN_CONFIRMATORY_STATES:
        raise RuntimeError(f"Confirmatory state index {init_state} is not in frozen states 3-22")
    if init_state in dev_states:
        raise RuntimeError(f"Confirmatory state index {init_state} overlaps development states {dev_states}")

    # 5. Exact four language-condition texts
    catalog_rel = task_cfgs[0].get("instruction_catalog")
    if not catalog_rel:
        raise RuntimeError("Confirmatory matrix missing instruction_catalog")
    cat_path = root / catalog_rel
    if not cat_path.exists():
        raise RuntimeError(f"Confirmatory instruction catalog missing: {cat_path}")

    catalog_rows = [json.loads(line) for line in cat_path.read_text(encoding="utf-8").strip().splitlines() if line.strip()]
    catalog_map = {r["id"]: r["text"].strip() for r in catalog_rows}
    if catalog_map != FROZEN_CONFIRMATORY_INSTRUCTIONS:
        raise RuntimeError(f"Instruction catalog differs from frozen confirmatory instructions: {catalog_map}")

    # Verify rows passed to runtime
    if not rows:
        raise RuntimeError("Runtime received empty instruction rows")
    for r in rows:
        r_id = r.get("id")
        if r_id not in FROZEN_CONFIRMATORY_INSTRUCTIONS:
            raise RuntimeError(f"Confirmatory runtime instruction id '{r_id}' not in frozen specification")
        r_text = r.get("text", "").strip()
        if r_text != FROZEN_CONFIRMATORY_INSTRUCTIONS[r_id]:
            raise RuntimeError(f"Confirmatory runtime instruction text for '{r_id}' ('{r_text}') differs from frozen text ('{FROZEN_CONFIRMATORY_INSTRUCTIONS[r_id]}')")

    # 6. Three frozen model arms
    if matrix.get("models") != FROZEN_CONFIRMATORY_MODELS:
        raise RuntimeError(f"Matrix models {matrix.get('models')} != frozen confirmatory models {FROZEN_CONFIRMATORY_MODELS}")
    if model_id:
        if model_id not in FROZEN_CONFIRMATORY_MODELS:
            raise RuntimeError(f"Runtime model '{model_id}' is not one of the frozen models {FROZEN_CONFIRMATORY_MODELS}")
        if model_id not in manifest.get("evaluated_models", {}):
            raise RuntimeError(f"Runtime model '{model_id}' is not registered in manifest evaluated_models")

    # 7. Policy / Environment Seeds: exactly 0
    mat_env = matrix.get("environment", {})
    mat_pol = matrix.get("policy", {})
    if mat_env.get("environment_seed") != 0 or config.get("environment_seed") != 0:
        raise RuntimeError("Environment seed must be strictly 0")
    if mat_pol.get("policy_seed") != 0 or config.get("policy_seed") != 0:
        raise RuntimeError("Policy seed must be strictly 0")

    # 8. Horizon: exactly 280 steps
    if mat_env.get("episode_length") != 280:
        raise RuntimeError(f"Confirmatory matrix episode_length must be 280, got {mat_env.get('episode_length')}")
    cfg_env = config.get("environment", {})
    if cfg_env.get("episode_length") != 280:
        raise RuntimeError(f"Confirmatory runtime episode_length must be 280, got {cfg_env.get('episode_length')}")

    # 9. Relevant Preprocessing/Postprocessing Configuration
    if cfg_env.get("obs_type") != mat_env.get("obs_type") or cfg_env.get("obs_type") != "pixels_agent_pos":
        raise RuntimeError(f"obs_type must be 'pixels_agent_pos', got runtime={cfg_env.get('obs_type')}, matrix={mat_env.get('obs_type')}")
    if cfg_env.get("observation_height") != 256 or cfg_env.get("observation_width") != 256:
        raise RuntimeError("Observation size must be 256x256")
    if cfg_env.get("control_freq") != 20 or cfg_env.get("control_mode") != "relative":
        raise RuntimeError("Control configuration must be 20Hz relative")
    if not cfg_env.get("hard_reset") or cfg_env.get("num_steps_wait") != 10:
        raise RuntimeError("Reset configuration must be hard_reset=True, num_steps_wait=10")

    cfg_pol = config.get("policy", {})
    if cfg_pol.get("n_action_steps") != 1 or cfg_pol.get("num_steps") != 10 or cfg_pol.get("use_amp") is not False:
        raise RuntimeError("Policy configuration must be n_action_steps=1, num_steps=10, use_amp=False")

    if matrix.get("execution", {}).get("default_workers", 1) != 1:
        raise RuntimeError("Execution default_workers must be 1")

    # 10. Load base pairing identity for upstream sources and LIBERO asset hashes
    pairing_out = config.get("pairing_output", "outputs/language_pairing")
    identity_file = root / pairing_out / "identity.json"
    if not identity_file.exists():
        raise RuntimeError(f"Base pairing identity missing: {identity_file}")
    pairing_identity = json.loads(identity_file.read_text(encoding="utf-8"))

    return pairing_identity



def verify_final_confirmatory_pairing(root, config, rows, confirmatory_spec):
    """Validate the exact frozen final matrix without constructing a policy or state."""
    root = Path(root)
    matrix = confirmatory_spec.get("matrix")
    manifest = confirmatory_spec.get("manifest")
    if not isinstance(matrix, dict) or not isinstance(manifest, dict):
        raise RuntimeError("Final confirmatory runtime requires explicit matrix and manifest")
    if matrix.get("role") != "final_confirmatory_evaluation":
        raise RuntimeError("Final confirmatory matrix role must be final_confirmatory_evaluation")

    matrix_path = root / "configs/final_confirmatory/final_matrix_v4.json"
    manifest_path = root / "configs/final_confirmatory/frozen_model_manifest_v4.json"
    historical_freeze = json.loads(
        (root / FINAL_CONFIRMATORY_FREEZE_PATH).read_text(encoding="utf-8")
    )
    frozen_matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    frozen_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matrix_hash = file_sha256(matrix_path)
    manifest_hash = file_sha256(manifest_path)
    if (matrix_hash != historical_freeze["final_protocol_artifacts"]["matrix_sha256"]
            or matrix_hash != frozen_manifest["final_matrix_sha256"]
            or manifest_hash != historical_freeze["final_protocol_artifacts"]["manifest_sha256"]):
        raise RuntimeError("Final frozen matrix or manifest SHA256 mismatch")
    if matrix != frozen_matrix or manifest != frozen_manifest:
        raise RuntimeError("Final runtime matrix or manifest differs from frozen files")

    tasks = matrix.get("tasks", [])
    if (matrix.get("suite") != "libero_spatial"
            or matrix.get("models") != ["original_language", "augmented_language", "paired_bc"]
            or len(tasks) != 1 or tasks[0].get("task_id") != 0
            or tasks[0].get("initial_states") != list(range(33, 50))):
        raise RuntimeError("Final task, models or states differ from frozen scope")
    task = tasks[0]
    if task.get("instruction_catalog") != "configs/final_confirmatory/instructions/task0_final.jsonl":
        raise RuntimeError("Final prompt catalog path changed")
    catalog_path = root / task["instruction_catalog"]
    catalog_hash = file_sha256(catalog_path)
    if (catalog_hash != task.get("instruction_catalog_sha256")
            or catalog_hash != manifest["final_prompt_catalog_sha256"]
            or catalog_hash != historical_freeze["final_protocol_artifacts"]["catalog_sha256"]):
        raise RuntimeError("Final prompt catalog SHA256 mismatch")
    catalog_rows = [json.loads(line) for line in catalog_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
    if [row["id"] for row in catalog_rows] != [
            "original", "paraphrase_final_01", "paraphrase_final_02"]:
        raise RuntimeError("Final prompt IDs changed")
    if not rows or len(rows) != 1 or rows[0] not in catalog_rows:
        raise RuntimeError("Runtime final instruction row differs from frozen catalog")
    for row in catalog_rows:
        actual = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        if (actual != task["prompt_text_sha256"].get(row["id"])
                or actual != manifest["final_prompt_text_sha256"].get(row["id"])):
            raise RuntimeError(f"Final instruction SHA256 mismatch: {row['id']}")

    if config.get("suite") != "libero_spatial" or config.get("task_id") != 0:
        raise RuntimeError("Final runtime must use LIBERO-Spatial Task 0")
    if config.get("init_state_index") not in range(33, 50):
        raise RuntimeError("Final runtime state must be in States 33-49")
    if config.get("environment_seed") != 0 or config.get("policy_seed") != 0:
        raise RuntimeError("Final runtime environment and policy seeds must be 0")
    environment = dict(matrix["environment"])
    policy = dict(matrix["policy"])
    environment.pop("environment_seed")
    if (matrix["environment"].get("episode_length") != 280
            or config.get("environment") != environment
            or config.get("policy") != policy):
        raise RuntimeError("Final runtime environment, policy or 280-step horizon changed")
    if (matrix.get("execution", {}).get("default_workers") != 1
            or matrix.get("execution", {}).get("total_rollouts") != 153
            or matrix.get("statistical_rules", {}).get("primary_metric") != "goal_ever"):
        raise RuntimeError("Final worker, rollout count or primary metric changed")

    model_id = confirmatory_spec.get("model_id")
    if model_id not in matrix["models"]:
        raise RuntimeError("Final runtime model is not in frozen three-model comparison")
    model = manifest["evaluated_models"][model_id]
    for relative, expected in model["artifact_sha256"].items():
        if file_sha256(root / model["checkpoint"] / relative) != expected:
            raise RuntimeError(f"Final checkpoint SHA256 mismatch: {model_id}/{relative}")

    identity_file = root / config.get("pairing_output", "outputs/language_pairing") / "identity.json"
    if not identity_file.is_file():
        raise RuntimeError(f"Base pairing identity missing: {identity_file}")
    return json.loads(identity_file.read_text(encoding="utf-8"))


def verify_scoped_confirmatory_pairing(root, config, rows, confirmatory_spec):
    """Dispatch only the registered development, primary and final matrix scopes."""
    matrix = confirmatory_spec.get("matrix") or {}
    role = matrix.get("role")
    if role == "model_selection_development_evaluation":
        return verify_development_pairing(root, config, rows, confirmatory_spec)
    if role == "final_confirmatory_evaluation":
        return verify_final_confirmatory_pairing(root, config, rows, confirmatory_spec)
    return verify_confirmatory_pairing(root, config, rows, confirmatory_spec)

def verify_development_pairing(root, config, rows, confirmatory_spec):
    """
    Validate runtime configuration and instruction rows against development specification.
    Enforces development matrix, manifest, task 0, states 23-32, 3 dev instructions, 3 models, seeds, and 280 steps.
    """
    root = Path(root)
    matrix = confirmatory_spec.get("matrix")
    if matrix is None:
        matrix_path = root / confirmatory_spec.get("matrix_path", "configs/development_evaluation/dev_matrix.json")
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))

    manifest = confirmatory_spec.get("manifest")
    if manifest is None:
        manifest_path = root / matrix.get("manifest_path", "configs/development_evaluation/dev_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    model_id = confirmatory_spec.get("model_id") or config.get("model_id")

    if matrix.get("role") != "model_selection_development_evaluation":
        raise RuntimeError(f"Development matrix role must be 'model_selection_development_evaluation', got '{matrix.get('role')}'")
    if matrix.get("suite") != "libero_spatial" or config.get("suite") != "libero_spatial":
        raise RuntimeError("Development suite must be 'libero_spatial'")

    task_cfgs = matrix.get("tasks", [])
    if not task_cfgs or task_cfgs[0].get("task_id") != 0:
        raise RuntimeError("Development matrix must specify Task 0")
    if config.get("task_id") != 0:
        raise RuntimeError(f"Development runtime task ID must be 0, got {config.get('task_id')}")

    matrix_states = set(task_cfgs[0].get("initial_states", []))
    allowed_states = set(range(23, 33))
    if matrix_states != allowed_states:
        raise RuntimeError(f"Development matrix states must be exactly states 23-32, got {matrix_states}")

    init_state = config.get("init_state_index")
    if init_state not in allowed_states:
        raise RuntimeError(f"Development state index {init_state} is not in states 23-32")

    expected_models = (["paired_bc", "paired_flow_consistency"]
                       if matrix.get("paired_training_config") == "configs/paired_training.json"
                       else ["original_language", "augmented_language", "anchored_language"])
    if matrix.get("models") != expected_models:
        raise RuntimeError(f"Matrix models {matrix.get('models')} != development models {expected_models}")
    if model_id not in expected_models:
        raise RuntimeError(f"Runtime model '{model_id}' is not one of the development models {expected_models}")

    mat_env = matrix.get("environment", {})
    if mat_env.get("environment_seed") != 0 or config.get("environment_seed") != 0:
        raise RuntimeError("Environment seed must be strictly 0")
    if mat_env.get("episode_length") != 280 or config.get("environment", {}).get("episode_length") != 280:
        raise RuntimeError("Episode length must be strictly 280 steps")

    if matrix.get("execution", {}).get("default_workers", 1) != 1:
        raise RuntimeError("Execution default_workers must be 1")

    pairing_out = config.get("pairing_output", "outputs/language_pairing")
    identity_file = root / pairing_out / "identity.json"
    if not identity_file.exists():
        raise RuntimeError(f"Base pairing identity missing: {identity_file}")
    pairing_identity = json.loads(identity_file.read_text(encoding="utf-8"))

    return pairing_identity


class BenchmarkRuntime:
    def __init__(self, root, baseline, confirmatory_spec=None):
        import torch
        from huggingface_hub import try_to_load_from_cache
        from libero.libero import benchmark, get_libero_path
        from lerobot.envs.libero import get_task_init_states

        self.root, self.baseline = Path(root), baseline
        self.config, self.rows = baseline["config"], baseline["instructions"]
        self.confirmatory_spec = confirmatory_spec or baseline.get("confirmatory_spec")

        for name, expected in baseline["project_sources"].items():
            if file_sha256(self.root / name) != expected:
                raise RuntimeError(f"Audited pilot source changed: {name}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        versions = {name: importlib.metadata.version(name) for name in baseline["packages"]}
        if versions != baseline["packages"]:
            raise RuntimeError("Dependencies differ from the audited pilot")

        if self.confirmatory_spec is not None:
            pairing = verify_scoped_confirmatory_pairing(
                self.root, self.config, self.rows, self.confirmatory_spec
            )
        else:
            _, pairing = verify_pairing(self.root, self.config, self.rows,
                                       self.root / baseline["run_config"]["pairing_report"])
        modules = {**baseline["upstream_sources"], **pairing["upstream_sources"]}
        for name, expected in modules.items():
            if file_sha256(importlib.import_module(name).__file__) != expected:
                raise RuntimeError(f"Upstream implementation changed: {name}")
        self.suite = benchmark.get_benchmark_dict()[self.config["suite"]]()
        task = self.suite.get_task(self.config["task_id"])
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_file = Path(get_libero_path("init_states")) / task.problem_folder / Path(task.init_states_file).name
        if file_sha256(bddl) != pairing["bddl_sha256"] or file_sha256(init_file) != pairing["init_file_sha256"]:
            raise RuntimeError("Task BDDL/initial-state assets changed")
        self.state_count = len(get_task_init_states(self.suite, self.config["task_id"]))
        self.path = self.root / baseline["checkpoint"]["local_dir"]
        cached = try_to_load_from_cache(baseline["backbone"]["repo_id"], "config.json", revision=baseline["backbone"]["revision"])
        if not isinstance(cached, str):
            raise RuntimeError("Validated backbone snapshot is missing from the local cache")
        self.snapshot = Path(cached).parent
        for base, hashes in ((self.path, baseline["policy_hashes"]), (self.snapshot, baseline["backbone_hashes"])):
            for name, expected in hashes.items():
                print(f"  Verify model artifact: {name}", flush=True)
                if file_sha256(base / name) != expected:
                    raise RuntimeError(f"Model artifact changed: {name}")
        backend = {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic,
                   "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
        if backend != baseline["backend"]:
            raise RuntimeError("Keep the audited pilot's backend settings for this comparison")
        driver = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                check=True, capture_output=True, text=True).stdout.strip().splitlines()
        self.provenance = {"packages": versions, "backend": backend, "gpu_driver": driver,
                           "python": platform.python_version(), "kernel": platform.release(),
                           "bddl_sha256": pairing["bddl_sha256"], "init_file_sha256": pairing["init_file_sha256"],
                           "policy_hashes": baseline["policy_hashes"], "backbone": baseline["backbone"],
                           "backbone_hashes": baseline["backbone_hashes"], "upstream_sources": modules}

    def load_policy(self):
        from lerobot.envs.configs import LiberoEnv as LiberoConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.utils.random_utils import set_seed
        from transformers import AutoTokenizer

        policy_config = SmolVLAConfig.from_pretrained(self.path)
        policy_config.device, policy_config.push_to_hub = "cuda", False
        for name, value in self.config["policy"].items():
            setattr(policy_config, name, value)
        policy_config.vlm_model_name = str(self.snapshot)
        if policy_config.n_action_steps != 1 or policy_config.use_amp:
            raise RuntimeError("Expected the audited one-step action horizon and precision settings")
        set_seed(self.config["policy_seed"])
        self.policy = SmolVLAPolicy.from_pretrained(self.path, config=policy_config, strict=True).eval()
        self.pre, self.post = make_pre_post_processors(policy_cfg=policy_config, pretrained_path=str(self.path),
            preprocessor_overrides={"device_processor": {"device": "cuda"}, "tokenizer_processor": {"tokenizer_name": str(self.snapshot)}})
        env = self.config["environment"]
        env_config = LiberoConfig(task=self.config["suite"], task_ids=[self.config["task_id"]],
                                 fps=env["control_freq"], observation_height=env["observation_height"],
                                 observation_width=env["observation_width"])
        self.env_pre, self.env_post = make_env_pre_post_processors(env_config, policy_config)
        tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot), local_files_only=True)
        self.token_ids = {r["id"]: tokenizer(r["text"] + "\n", truncation=False)["input_ids"] for r in self.rows}
        max_tokens = policy_config.tokenizer_max_length
        if any(len(tokens) > max_tokens for tokens in self.token_ids.values()):
            raise RuntimeError("Instruction would be truncated")
        dtypes = Counter()
        for parameter in self.policy.parameters():
            dtypes[str(parameter.dtype)] += parameter.numel()
        return {"strict_checkpoint_load": True, "parameter_counts_by_dtype": dict(dtypes),
                "settings": self.config["policy"], "backbone": self.baseline["backbone"]}
