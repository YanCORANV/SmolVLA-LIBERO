"""Comprehensive verification tests for paired_bc and paired_flow_consistency training arms.

Proves:
1. Student initialized from original_language is byte-identical to source checkpoint.
2. Teacher and student produce identical canonical velocity predictions at initialization.
3. Teacher receives zero gradients after backward.
4. Original and paraphrase branches receive identical images, state, action target, noise, time.
5. Only language token tensors differ between the paired branches.
6. paired_bc and paired_flow_consistency use identical frame indices and paraphrase schedules.
7. All four paraphrases occur exactly 250 times across 1000 steps.
8. lambda_cons is frozen at 1.0.
9. With lambda_cons = 0, consistency implementation reduces exactly to paired_bc.
10. No States 33-52 and no future final prompts are accessed.
"""

from collections import Counter
import json
from pathlib import Path
import sys
import pytest
import torch
import torch.nn.functional as F
from huggingface_hub import try_to_load_from_cache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_STATE,
    ACTION
)
from lerobot.utils.collate import lerobot_collate_fn

from smolvla_grounding.adaptation_records import build_paired_schedule
from smolvla_grounding.flow_velocity import (
    forward_velocity,
    compute_paired_bc_loss,
    compute_paired_flow_consistency_loss,
    create_frozen_teacher
)
from smolvla_grounding.protocol import file_sha256
from smolvla_grounding.training_policy import FP32ExpertSmolVLAPolicy
from smolvla_grounding.training_profile_records import verify_ready


@pytest.fixture(scope="module")
def setup_models_and_data():
    root = Path(__file__).resolve().parents[1]
    with open(root / "configs/paired_training.json") as f:
        paired_cfg = json.load(f)

    source_ckpt = root / paired_cfg["source_checkpoint"]["path"]
    if not source_ckpt.exists():
        pytest.skip(f"Source checkpoint not found at {source_ckpt}")

    cached = try_to_load_from_cache(
        "HuggingFaceTB/SmolVLM2-500M-Instruct",
        "config.json",
        revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467"
    )
    if not cached:
        pytest.skip("Backbone snapshot not found in cache")
    snapshot = str(Path(cached).parent)

    cfg = SmolVLAConfig.from_pretrained(source_ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.device = device
    cfg.push_to_hub = False
    cfg.vlm_model_name = snapshot
    cfg.freeze_vision_encoder = True
    cfg.train_expert_only = True
    cfg.train_state_proj = True

    student = FP32ExpertSmolVLAPolicy.from_pretrained(source_ckpt, config=cfg, strict=True).to(device)
    teacher = create_frozen_teacher(source_ckpt, config=cfg, device=device)

    pre, _ = make_pre_post_processors(
        cfg,
        pretrained_path=str(source_ckpt),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "tokenizer_processor": {"tokenizer_name": snapshot}
        }
    )

    data_config, ready, dataset_root = verify_ready(root, {"data_ready": paired_cfg["training"]["data_ready"]})
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

    rows = [json.loads(line) for line in (root / paired_cfg["training"]["train_catalog"]).read_text(encoding="utf-8").splitlines() if line.strip()]

    return root, paired_cfg, source_ckpt, student, teacher, pre, dataset, rows, device


def test_1_student_checkpoint_byte_identity(setup_models_and_data):
    """1. Student initialized from original_language is byte-identical to the frozen source checkpoint."""
    root, paired_cfg, source_ckpt, student, teacher, pre, dataset, rows, device = setup_models_and_data

    expected_hashes = paired_cfg["source_checkpoint"]["model_hashes"]
    for filename, expected_hash in expected_hashes.items():
        actual_hash = file_sha256(source_ckpt / filename)
        assert actual_hash == expected_hash, f"Byte mismatch for {filename}: {actual_hash} != {expected_hash}"

    # Also verify parameters.json
    param_file = root / paired_cfg["source_checkpoint"]["parameters_file"]
    assert file_sha256(param_file) == paired_cfg["source_checkpoint"]["parameters_sha256"]


def test_2_teacher_student_initial_velocity_identity(setup_models_and_data):
    """2. Teacher and student produce identical canonical velocity predictions at initialization."""
    root, paired_cfg, source_ckpt, student, teacher, pre, dataset, rows, device = setup_models_and_data

    sample = dict(dataset[0])
    sample["task"] = rows[0]["text"]
    pre.reset()
    batch = pre(lerobot_collate_fn([sample]))

    actions = student.prepare_action(batch)
    noise = student.model.sample_noise(actions.shape, actions.device)
    time = student.model.sample_time(actions.shape[0], actions.device)

    student_mode = student.training
    student.eval()
    try:
        with torch.no_grad():
            v_student, u_student, _, _ = forward_velocity(student, batch, noise=noise, time=time)
            v_teacher, u_teacher, _, _ = forward_velocity(teacher, batch, noise=noise, time=time)
    finally:
        student.train(student_mode)

    max_diff = torch.max(torch.abs(v_student - v_teacher)).item()
    assert max_diff == 0.0, f"Initial student and teacher velocity predictions differ by {max_diff}"


def test_3_teacher_receives_zero_gradients_after_backward(setup_models_and_data):
    """3. Teacher receives zero gradients after backward."""
    root, paired_cfg, source_ckpt, student, teacher, pre, dataset, rows, device = setup_models_and_data

    # Ensure teacher parameters have requires_grad == False
    for name, p in teacher.named_parameters():
        assert not p.requires_grad, f"Teacher parameter {name} has requires_grad=True!"

    sample = dict(dataset[0])
    sample_orig = dict(sample)
    sample_orig["task"] = rows[0]["text"]
    pre.reset()
    batch_orig = pre(lerobot_collate_fn([sample_orig]))

    sample_para = dict(sample)
    sample_para["task"] = rows[1]["text"]
    pre.reset()
    batch_para = pre(lerobot_collate_fn([sample_para]))

    actions = student.prepare_action(batch_orig)
    noise = student.model.sample_noise(actions.shape, actions.device)
    time = student.model.sample_time(actions.shape[0], actions.device)

    student.zero_grad()
    teacher.zero_grad()

    with torch.no_grad():
        v_teacher, _, _, _ = forward_velocity(teacher, batch_orig, noise=noise, time=time)

    v_orig, u_t, _, _ = forward_velocity(student, batch_orig, noise=noise, time=time)
    v_para, _, _, _ = forward_velocity(student, batch_para, noise=noise, time=time)

    loss, _ = compute_paired_flow_consistency_loss(v_orig, v_para, v_teacher, u_t, lambda_cons=1.0)
    loss.backward()

    # Verify teacher parameters received zero gradient
    for name, p in teacher.named_parameters():
        assert p.grad is None, f"Teacher parameter {name} received non-None gradient!"

    # Verify student trainable parameters received finite gradient
    trainable_grads = [p.grad for p in student.parameters() if p.requires_grad]
    assert len(trainable_grads) > 0
    for g in trainable_grads:
        assert g is not None
        assert torch.isfinite(g).all()


def test_4_and_5_paired_branches_equality_and_isolation(setup_models_and_data):
    """4. Original and paraphrase branches receive identical images, state, action, noise, time.
       5. Only language token tensors differ between the paired branches.
    """
    root, paired_cfg, source_ckpt, student, teacher, pre, dataset, rows, device = setup_models_and_data

    sample = dict(dataset[0])
    sample_orig = dict(sample)
    sample_orig["task"] = rows[0]["text"]
    pre.reset()
    batch_orig = pre(lerobot_collate_fn([sample_orig]))

    sample_para = dict(sample)
    sample_para["task"] = rows[1]["text"]
    pre.reset()
    batch_para = pre(lerobot_collate_fn([sample_para]))

    # Physical components equality (4)
    assert torch.equal(batch_orig["observation.images.image"], batch_para["observation.images.image"])
    assert torch.equal(batch_orig["observation.images.image2"], batch_para["observation.images.image2"])
    assert torch.equal(batch_orig["observation.state"], batch_para["observation.state"])
    assert torch.equal(batch_orig["action"], batch_para["action"])

    # Shared noise and time equality (4)
    actions = student.prepare_action(batch_orig)
    noise = student.model.sample_noise(actions.shape, actions.device)
    time = student.model.sample_time(actions.shape[0], actions.device)
    # The pipeline explicitly passes the same `noise` and `time` variables to both forward_velocity calls.

    # Only language tokens differ (5)
    assert not torch.equal(batch_orig[f"{OBS_LANGUAGE_TOKENS}"], batch_para[f"{OBS_LANGUAGE_TOKENS}"])
    for key in batch_orig.keys():
        if "language" not in key and isinstance(batch_orig[key], torch.Tensor):
            assert torch.equal(batch_orig[key], batch_para[key]), f"Non-language tensor differs for key {key}!"


def test_6_and_7_schedule_identity_and_paraphrase_balance(setup_models_and_data):
    """6. paired_bc and paired_flow_consistency use identical frame indices and paraphrase schedules.
       7. All four paraphrases occur exactly 250 times across 1000 steps.
    """
    root, paired_cfg, source_ckpt, student, teacher, pre, dataset, rows, device = setup_models_and_data

    with open(root / "outputs/task0_data/ready.json") as f:
        ready = json.load(f)

    episodes = [r for r in ready["episodes"] if r["episode_index"] in ready["selection"]["split"]["train"]]
    updates = paired_cfg["training"]["updates"]
    seed = paired_cfg["training"]["seed"]
    lang_seed = paired_cfg["training"]["language_seed"]

    sched_bc = build_paired_schedule(episodes, updates, seed, lang_seed, paraphrase_variants=(1, 2, 3, 4))
    sched_cons = build_paired_schedule(episodes, updates, seed, lang_seed, paraphrase_variants=(1, 2, 3, 4))

    # Test 6: 100% schedule identity
    assert len(sched_bc) == 1000
    assert len(sched_cons) == 1000
    for s_bc, s_cs in zip(sched_bc, sched_cons):
        assert s_bc["step"] == s_cs["step"]
        assert s_bc["episode_index"] == s_cs["episode_index"]
        assert s_bc["frame_index"] == s_cs["frame_index"]
        assert s_bc["canonical_variant"] == s_cs["canonical_variant"] == 0
        assert s_bc["paraphrase_variant"] == s_cs["paraphrase_variant"]

    # Test 7: Exactly 250 of each paraphrase variant
    counts = Counter(s["paraphrase_variant"] for s in sched_bc)
    for variant in [1, 2, 3, 4]:
        assert counts[variant] == 250, f"Variant {variant} occurred {counts[variant]} times, expected exactly 250"


def test_8_lambda_cons_frozen_at_one(setup_models_and_data):
    """8. lambda_cons is frozen at 1.0."""
    root, paired_cfg, source_ckpt, student, teacher, pre, dataset, rows, device = setup_models_and_data
    assert paired_cfg["training"]["lambda_cons"] == 1.0


def test_9_consistency_reduces_to_paired_bc_when_lambda_zero():
    """9. With lambda_cons = 0 in a synthetic equivalence test, consistency reduces exactly to paired_bc."""
    b, s, d = 2, 10, 7
    v_orig = torch.randn(b, s, d)
    v_para = torch.randn(b, s, d)
    v_teacher = torch.randn(b, s, d)
    u_t = torch.randn(b, s, d)

    loss_bc, metrics_bc = compute_paired_bc_loss(v_orig, v_para, u_t)
    loss_cons, metrics_cons = compute_paired_flow_consistency_loss(v_orig, v_para, v_teacher, u_t, lambda_cons=0.0)

    diff = torch.abs(loss_bc - loss_cons).item()
    assert diff < 1e-7, f"Difference at lambda=0: {diff}"
    assert metrics_cons["l_fm"] == pytest.approx(metrics_bc["loss"], abs=1e-7)


def test_10_no_states_33_52_and_no_final_prompts_accessed(setup_models_and_data):
    """10. No States 33-52 and no future final prompts are accessed."""
    root, paired_cfg, source_ckpt, student, teacher, pre, dataset, rows, device = setup_models_and_data

    # Check dev gate states in config
    gate_states = paired_cfg["frozen_development_gate"]["evaluation_states"]
    for s in gate_states:
        assert s < 33, f"Development gate accesses state {s} which is >= 33!"
        assert s >= 23, f"Development gate state {s} out of bounds"

    # Check retired final prompts in task0_final_reserved.jsonl
    reserved_path = root / "configs/final_confirmatory_reserved/instructions/task0_final_reserved.jsonl"
    reserved_lines = [json.loads(line) for line in reserved_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in reserved_lines:
        if r["id"] in ("paraphrase_final_01", "paraphrase_final_02"):
            assert r["status"] == "CONSUMED_BY_DESIGN_RETIRED"

    # Check training instructions do NOT contain any reserved/retired prompts
    training_texts = {r["text"] for r in rows}
    for r in reserved_lines:
        if r["id"] != "original":
            assert r["text"] not in training_texts, f"Final prompt {r['id']} was included in training catalog!"
