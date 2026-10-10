"""Equivalence tests proving that forward_velocity reproduces the official SmolVLA forward path.

Validates that MSE(v_t, u_t) numerically agrees with official policy.forward()
under identical batch, noise, and time tensors with tight tolerance.
"""

from pathlib import Path
import json
import sys
import torch
import torch.nn.functional as F
import pytest
from huggingface_hub import try_to_load_from_cache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.collate import lerobot_collate_fn

from smolvla_grounding.flow_velocity import forward_velocity
from smolvla_grounding.training_profile_records import verify_ready


@pytest.fixture(scope="module")
def setup_smolvla_env():
    root = Path(__file__).resolve().parents[1]
    ckpt_path = root / "outputs/adaptation/runs/task0_longer_training/original_language/step_001000/attempt_001/pretrained_model"
    if not ckpt_path.exists():
        pytest.skip(f"Source checkpoint not found at {ckpt_path}")

    cached = try_to_load_from_cache(
        "HuggingFaceTB/SmolVLM2-500M-Instruct",
        "config.json",
        revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    )
    if not cached:
        pytest.skip("Backbone snapshot not found in cache")
    snapshot = str(Path(cached).parent)

    cfg = SmolVLAConfig.from_pretrained(ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device
    cfg.push_to_hub = False
    cfg.vlm_model_name = snapshot

    policy = SmolVLAPolicy.from_pretrained(ckpt_path, config=cfg).to(device).eval()
    pre, _ = make_pre_post_processors(
        cfg,
        pretrained_path=str(ckpt_path),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "tokenizer_processor": {"tokenizer_name": snapshot}
        }
    )

    data_config, ready, dataset_root = verify_ready(root, {"data_ready": "outputs/task0_data/ready.json"})
    meta = LeRobotDatasetMetadata(data_config["repo_id"], root=dataset_root, revision=data_config["revision"])
    train_ids = ready["selection"]["split"]["train"]

    dataset = LeRobotDataset(
        data_config["repo_id"],
        root=dataset_root,
        revision=data_config["revision"],
        episodes=train_ids,
        delta_timestamps=resolve_delta_timestamps(cfg, meta),
        return_uint8=False
    )

    return policy, pre, dataset, data_config["original_instruction"], device


def test_velocity_forward_numerical_equivalence(setup_smolvla_env):
    """Proves MSE(v_t, u_t) is bit-for-bit / within float32 tolerance identical to policy.forward()."""
    policy, pre, dataset, orig_instruction, device = setup_smolvla_env

    # Test on multiple distinct demonstration samples
    for sample_idx in [0, 5, 10]:
        sample = dict(dataset[sample_idx])
        sample["task"] = orig_instruction
        pre.reset()
        batch = pre(lerobot_collate_fn([sample]))

        actions = policy.prepare_action(batch)
        noise = policy.model.sample_noise(actions.shape, actions.device)
        time = policy.model.sample_time(actions.shape[0], actions.device)

        # 1. Official forward pass
        with torch.no_grad():
            official_loss, _ = policy.forward(batch, noise=noise, time=time)

        # 2. Custom project-local velocity forward pass
        with torch.no_grad():
            v_t, u_t, _, _ = forward_velocity(policy, batch, noise=noise, time=time)
            custom_loss = F.mse_loss(v_t, u_t)

        diff = torch.abs(official_loss - custom_loss).item()
        assert diff < 1e-6, f"Sample {sample_idx}: loss mismatch: official={official_loss.item():.8f}, custom={custom_loss.item():.8f}, diff={diff:.2e}"
        print(f"Sample {sample_idx}: diff = {diff:.2e} (PASSED)")
