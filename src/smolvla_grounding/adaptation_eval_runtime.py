"""Precision-matched adapter around the already audited official rollout runtime."""

from copy import deepcopy
from dataclasses import asdict
import json

from .benchmark_runtime import BenchmarkRuntime, file_sha256
from .adaptation_eval_records import verify_training_checkpoint


class AdaptationEvalRuntime(BenchmarkRuntime):
    def __init__(self, root, baseline, model_id, model, rows, confirmatory_spec=None):
        super().__init__(root, baseline, confirmatory_spec=confirmatory_spec)
        self.original_path = self.path
        self.model_id, self.model, self.rows = model_id, model, rows
        if model_id != "untuned_fp32":
            if self.confirmatory_spec is not None:
                # Confirmatory mode: verify checkpoint artifacts against frozen manifest
                self.path = self.root / model["checkpoint"]
                expected_hashes = model.get("artifact_sha256", {})
                if not expected_hashes:
                    raise RuntimeError(f"Missing artifact_sha256 for confirmatory model {model_id}")
                for rel_name, exp_hash in expected_hashes.items():
                    act_hash = file_sha256(self.path / rel_name)
                    if act_hash != exp_hash:
                        raise RuntimeError(f"Confirmatory checkpoint artifact {rel_name} SHA256 mismatch: {act_hash} != {exp_hash}")
                if (self.path / "pretrained_model").is_dir():
                    self.path = self.path / "pretrained_model"
            else:
                # Historical pilot mode
                print("  Verify committed training checkpoint artifacts", flush=True)
                verify_training_checkpoint(self.root, model)
                self.path = self.root / model["checkpoint"]

    def load_policy(self):
        import torch
        from safetensors.torch import load_file
        from lerobot.envs.configs import LiberoEnv as LiberoConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.utils.random_utils import set_seed
        from transformers import AutoTokenizer
        from .training_policy import FP32ExpertSmolVLAPolicy, parameter_evidence

        # Only optimizer/scheduler fields and loader locations may differ. All
        # architecture, image and action settings must match the official policy.
        def inference_config(path):
            # Upstream saves newly added default fields on reserialization.
            # Resolve both through the same official config class first.
            value = asdict(SmolVLAConfig.from_pretrained(path))
            return {k: v for k, v in value.items() if not k.startswith(("optimizer_", "scheduler_"))
                    and k not in {"device", "push_to_hub", "vlm_model_name", "freeze_vision_encoder", "train_expert_only", "train_state_proj"}}
        if inference_config(self.path) != inference_config(self.original_path):
            raise RuntimeError("Saved policy changes an inference/architecture configuration")

        # Compare complete processor configs after normalizing the tokenizer
        # location, and compare actual normalization tensors rather than filenames.
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            pair = [json.loads((p / name).read_text()) for p in (self.original_path, self.path)]
            normalized = deepcopy(pair)
            for value in normalized:
                for step in value["steps"]:
                    if step["registry_name"] == "tokenizer_processor":
                        step["config"]["tokenizer_name"] = str(self.snapshot)
            if normalized[0] != normalized[1]:
                raise RuntimeError("Processor settings differ from the official checkpoint")
            for a, b in zip(pair[0]["steps"], pair[1]["steps"], strict=True):
                if "state_file" in a:
                    x = load_file(str(self.original_path / a["state_file"]))
                    y = load_file(str(self.path / b["state_file"]))
                    if x.keys() != y.keys() or any(x[k].dtype != y[k].dtype or not torch.equal(x[k], y[k]) for k in x):
                        raise RuntimeError("Checkpoint normalization tensors changed")
        config = SmolVLAConfig.from_pretrained(self.path)
        config.device, config.push_to_hub, config.vlm_model_name = "cuda", False, str(self.snapshot)
        config.freeze_vision_encoder = config.train_expert_only = config.train_state_proj = True
        for name, value in self.config["policy"].items():
            setattr(config, name, value)
        if config.n_action_steps != 1 or config.use_amp:
            raise RuntimeError("Expected audited inference settings")
        set_seed(self.config["policy_seed"])
        self.policy = FP32ExpertSmolVLAPolicy.from_pretrained(self.path, config=config, strict=True).eval()
        actual, counts = parameter_evidence(self.policy)
        if actual != json.loads((self.root / self.model["parameters"]).read_text()):
            raise RuntimeError("Reloaded parameter bytes, precision or frozen status differ from training evidence")
        self.pre, self.post = make_pre_post_processors(config, pretrained_path=str(self.path),
            preprocessor_overrides={"device_processor": {"device": "cuda"},
                                    "tokenizer_processor": {"tokenizer_name": str(self.snapshot)}})
        env = self.config["environment"]
        env_config = LiberoConfig(task=self.config["suite"], task_ids=[self.config["task_id"]],
            fps=env["control_freq"], observation_height=env["observation_height"], observation_width=env["observation_width"])
        self.env_pre, self.env_post = make_env_pre_post_processors(env_config, config)
        tokenizer = AutoTokenizer.from_pretrained(str(self.snapshot), local_files_only=True)
        self.token_ids = {r["id"]: tokenizer(r["text"] + "\n", truncation=False)["input_ids"] for r in self.rows}
        if any(len(ids) > config.tokenizer_max_length for ids in self.token_ids.values()):
            raise RuntimeError("Instruction truncation")
        return {"model": self.model_id, "strict_load": True, "all_parameter_bytes_equal": True,
                "normalization_matches_original": True, "parameter_counts": counts,
                "inference_settings": self.config["policy"], "autocast": False}
