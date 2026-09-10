"""Reusable pinned official runtime for state benchmarks; no upstream model copies."""

from collections import Counter
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

from .protocol import file_sha256
from .rollout_records import verify_pairing


class BenchmarkRuntime:
    def __init__(self, root, baseline):
        import torch
        from huggingface_hub import try_to_load_from_cache
        from libero.libero import benchmark, get_libero_path
        from lerobot.envs.libero import get_task_init_states

        self.root, self.baseline = Path(root), baseline
        self.config, self.rows = baseline["config"], baseline["instructions"]
        for name, expected in baseline["project_sources"].items():
            if file_sha256(self.root / name) != expected:
                raise RuntimeError(f"Audited pilot source changed: {name}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        versions = {name: importlib.metadata.version(name) for name in baseline["packages"]}
        if versions != baseline["packages"]:
            raise RuntimeError("Dependencies differ from the audited pilot")
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
