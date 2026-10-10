"""Training implementation for paired_bc and paired_flow_consistency arms.

Arm A (paired_bc): Compute-matched paired language training without consistency regularization.
Arm B (paired_flow_consistency): Flow consistency with frozen canonical teacher.
"""

from datetime import datetime, timezone
import csv
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import time

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from huggingface_hub import try_to_load_from_cache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.adaptation_records import build_paired_schedule, checkpoint_steps, latest_checkpoint, read_jsonl, schedule_hash
from smolvla_grounding.benchmark_records import BenchmarkJournal
from smolvla_grounding.flow_velocity import forward_velocity, compute_paired_bc_loss, compute_paired_flow_consistency_loss, create_frozen_teacher, sample_paired_flow_inputs
from smolvla_grounding.protocol import atomic_json, file_sha256, fingerprint, run_lock
from smolvla_grounding.training_policy import FP32ExpertSmolVLAPolicy, parameter_evidence
from smolvla_grounding.training_profile_records import compare_parameter_updates, verify_ready
from smolvla_grounding.diagnostic_records import pack_rng, pack_tensors

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.train_utils import save_training_state, load_training_state
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import get_rng_state, set_seed
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from transformers import AutoTokenizer


def main():
    directory = Path(os.environ["SMOLVLA_ADAPTATION_DIR"])
    arm = os.environ["SMOLVLA_ADAPTATION_ARM"]

    with open(ROOT / "configs/paired_training.json") as f:
        paired_cfg = json.load(f)

    if arm not in paired_cfg["training"]["arms"]:
        raise ValueError(f"Unknown experimental arm {arm}, expected one of {paired_cfg['training']['arms']}")

    experiment_file = directory / "experiment.json"
    if not experiment_file.exists():
        raise RuntimeError(f"Experiment metadata missing: {experiment_file}")
    experiment = json.loads(experiment_file.read_text(encoding="utf-8"))
    training = paired_cfg["training"]
    if experiment["updates"] != training["updates"] or experiment["seed"] != training["seed"]:
        raise RuntimeError("Paired experiment update budget or frame seed differs from the frozen configuration")

    schedule_path = directory / "schedule.json"
    if not schedule_path.exists():
        raise RuntimeError(f"Schedule missing: {schedule_path}")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))

    catalog_path = paired_cfg["training"]["train_catalog"]
    reference = json.loads((ROOT / "outputs/adaptation/runs/task0_longer_training/experiment.json").read_text(encoding="utf-8"))
    if catalog_path != reference["provenance"]["settings"]["training_catalog"]:
        raise RuntimeError("Paired training catalog differs from the original_language/augmented_language catalog")
    catalog_hash = file_sha256(ROOT / catalog_path)
    if catalog_hash != reference["provenance"]["sources"][catalog_path]:
        raise RuntimeError("Frozen historical training-language catalog changed")

    source_ckpt = ROOT / paired_cfg["source_checkpoint"]["path"]
    for filename, expected_hash in paired_cfg["source_checkpoint"]["model_hashes"].items():
        actual_hash = file_sha256(source_ckpt / filename)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Source checkpoint artifact changed: {filename} ({actual_hash} != {expected_hash})")
    if file_sha256(ROOT / paired_cfg["source_checkpoint"]["parameters_file"]) != paired_cfg["source_checkpoint"]["parameters_sha256"]:
        raise RuntimeError("Source checkpoint parameter evidence changed")

    # Lock execution
    arm_dir = directory / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    with run_lock(arm_dir):
        journal = BenchmarkJournal(arm_dir, {
            "experiment": experiment,
            "arm": arm,
            "schedule_hash": schedule_hash(schedule),
            "paired_config": paired_cfg,
            "training_catalog_sha256": catalog_hash,
        })
        last = latest_checkpoint(journal, experiment["updates"], paired_cfg["training"]["checkpoint_every"])
        if last and last["step"] == experiment["updates"]:
            atomic_json(arm_dir / "finished.json", last)
            print("  Final checkpoint verified; no training repeated.", flush=True)
            return
        restored_step = last["step"] if last else 0
        attempt = journal.new_attempt("executions")
        execution = {"passed": False, "start_time": datetime.now(timezone.utc).isoformat(),
                     "arm": arm, "restored_step": restored_step}
        started = time.perf_counter()

        try:
            print(f"[1/4] Verify data and source checkpoint for {arm}", flush=True)
            data_config, ready, dataset_root = verify_ready(ROOT, {"data_ready": paired_cfg["training"]["data_ready"]})
            train_ids = set(ready["selection"]["split"]["train"])
            training_episodes = [row for row in ready["episodes"] if row["episode_index"] in train_ids]
            expected_schedule = build_paired_schedule(
                training_episodes, training["updates"], training["seed"], training["language_seed"],
                tuple(training["paraphrase_variants"]),
            )
            if schedule != expected_schedule:
                raise RuntimeError("Paired frame/language schedule differs from the frozen training configuration")
            rows = read_jsonl(ROOT / catalog_path)
            expected_ids = ["train_original", *(f"train_paraphrase_{i:02d}" for i in range(1, 5))]
            if [row["id"] for row in rows] != expected_ids or rows[0]["text"] != data_config["original_instruction"]:
                raise RuntimeError("Training language IDs or canonical wording differ from the frozen Task-0 catalog")

            cached = try_to_load_from_cache("HuggingFaceTB/SmolVLM2-500M-Instruct", "config.json", revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467")
            snapshot = str(Path(cached).parent)

            # Tokenize rows
            tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
            token_ids = [tokenizer(row["text"] + "\n", truncation=False)["input_ids"] for row in rows]

            # Setup dataset
            config = SmolVLAConfig.from_pretrained(source_ckpt)
            config.device, config.push_to_hub, config.vlm_model_name = "cuda", False, snapshot
            config.freeze_vision_encoder = True
            config.train_expert_only = True
            config.train_state_proj = True
            config.optimizer_lr = paired_cfg["training"]["learning_rate"]
            config.scheduler_warmup_steps = min(20, experiment["updates"] // 10)
            config.scheduler_decay_steps = experiment["updates"]

            meta = LeRobotDatasetMetadata(data_config["repo_id"], root=dataset_root, revision=data_config["revision"])
            dataset = LeRobotDataset(
                data_config["repo_id"], root=dataset_root, revision=data_config["revision"],
                episodes=sorted(train_ids), delta_timestamps=resolve_delta_timestamps(config, meta), return_uint8=False
            )
            ep = dataset.hf_dataset.data.column("episode_index").to_numpy()
            fr = dataset.hf_dataset.data.column("frame_index").to_numpy()
            position = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(ep, fr))}

            print(f"[2/4] Initialize student (and teacher if applicable) from frozen source checkpoint", flush=True)
            set_seed(experiment["seed"])
            student_path = arm_dir / last["checkpoint"] / "pretrained_model" if last else source_ckpt
            student = FP32ExpertSmolVLAPolicy.from_pretrained(student_path, config=config, strict=True)
            pre, post = make_pre_post_processors(
                config, pretrained_path=str(student_path),
                preprocessor_overrides={"device_processor": {"device": "cuda"}, "tokenizer_processor": {"tokenizer_name": snapshot}}
            )

            # Build teacher if Arm B
            teacher = None
            if arm == "paired_flow_consistency":
                teacher = create_frozen_teacher(source_ckpt, config=config, device="cuda")

            optimizer = config.get_optimizer_preset().build([p for p in student.parameters() if p.requires_grad])
            scheduler = config.get_scheduler_preset().build(optimizer, num_training_steps=experiment["updates"])
            accelerator = Accelerator(mixed_precision="bf16", step_scheduler_with_optimizer=False)
            student, optimizer = accelerator.prepare(student, optimizer)
            raw_student = accelerator.unwrap_model(student)
            parameters_now, _ = parameter_evidence(raw_student)

            if last:
                before = json.loads((arm_dir / "initial_parameters.json").read_text(encoding="utf-8"))
                history = read_jsonl(arm_dir / last["training_log"])
                restored_step, optimizer, scheduler = load_training_state(arm_dir / last["checkpoint"], optimizer, scheduler)
                if restored_step != last["step"] or len(history) != restored_step:
                    raise RuntimeError("Committed training state or history disagrees with checkpoint step")
            else:
                before, history, restored_step = parameters_now, [], 0
                initial_path = arm_dir / "initial_parameters.json"
                if initial_path.exists():
                    if json.loads(initial_path.read_text(encoding="utf-8")) != before:
                        raise RuntimeError("Uncommitted earlier attempt has different starting parameters")
                else:
                    atomic_json(initial_path, before)

            def guard_gradient(_opt, _args, _kwargs):
                if teacher is not None:
                    for name, p in teacher.named_parameters():
                        if p.grad is not None:
                            raise RuntimeError(f"Teacher received gradient in {name}")
                for name, p in raw_student.named_parameters():
                    if not p.requires_grad and p.grad is not None:
                        raise RuntimeError(f"Frozen student parameter received gradient: {name}")

            optimizer.optimizer.register_step_pre_hook(guard_gradient)

            print(f"[3/4] Train {arm}: updates {restored_step + 1}..{experiment['updates']}", flush=True)
            save_at = set(checkpoint_steps(experiment["updates"], paired_cfg["training"]["checkpoint_every"]))
            lambda_cons = paired_cfg["training"]["lambda_cons"]
            timings = []

            with (arm_dir / "uncommitted_steps.jsonl").open("a" if restored_step else "w", encoding="utf-8") as stream:
                for planned in schedule[restored_step:]:
                    step = planned["step"]
                    t0 = time.perf_counter()

                    sample_raw = dict(dataset[position[(planned["episode_index"], planned["frame_index"])]])
                    data_hash = fingerprint(pack_tensors({k: v for k, v in sample_raw.items() if isinstance(v, torch.Tensor)}, "sample", {}, {}))

                    # 1. Canonical input
                    sample_orig = dict(sample_raw)
                    sample_orig["task"] = rows[0]["text"]
                    pre.reset()
                    batch_orig = pre(lerobot_collate_fn([sample_orig]))
                    orig_ids = batch_orig[OBS_LANGUAGE_TOKENS][0][batch_orig[OBS_LANGUAGE_ATTENTION_MASK][0].bool()].cpu().tolist()
                    if orig_ids != token_ids[0]:
                        raise RuntimeError("Canonical branch received the wrong instruction tokens")

                    # 2. Paraphrase input
                    sample_para = dict(sample_raw)
                    para_idx = planned["paraphrase_variant"]
                    sample_para["task"] = rows[para_idx]["text"]
                    pre.reset()
                    batch_para = pre(lerobot_collate_fn([sample_para]))
                    para_ids = batch_para[OBS_LANGUAGE_TOKENS][0][batch_para[OBS_LANGUAGE_ATTENTION_MASK][0].bool()].cpu().tolist()
                    if para_ids != token_ids[para_idx]:
                        raise RuntimeError("Paraphrase branch received the wrong instruction tokens")

                    # Verify noise and time sharing
                    actions = raw_student.prepare_action(batch_orig)
                    noise, t_val, flow_seed = sample_paired_flow_inputs(
                        raw_student.model, actions.shape, actions.device, training["seed"], step
                    )
                    action_hash = fingerprint(pack_tensors({"action": actions}, "action", {}, {}))
                    noise_hash = fingerprint(pack_tensors({"noise": noise}, "flow", {}, {}))
                    time_hash = fingerprint(pack_tensors({"time": t_val}, "flow", {}, {}))

                    optimizer.zero_grad()

                    # Student predictions
                    v_orig, u_t, _, _ = forward_velocity(raw_student, batch_orig, noise=noise, time=t_val)
                    v_para, _, _, _ = forward_velocity(raw_student, batch_para, noise=noise, time=t_val)

                    if arm == "paired_bc":
                        loss, loss_metrics = compute_paired_bc_loss(v_orig, v_para, u_t)
                    elif arm == "paired_flow_consistency":
                        with torch.no_grad():
                            v_teacher, _, _, _ = forward_velocity(teacher, batch_orig, noise=noise, time=t_val)
                        loss, loss_metrics = compute_paired_flow_consistency_loss(v_orig, v_para, v_teacher, u_t, lambda_cons=lambda_cons)

                    accelerator.backward(loss)
                    if config.optimizer_grad_clip_norm > 0:
                        accelerator.clip_grad_norm_(student.parameters(), config.optimizer_grad_clip_norm)

                    lr_used = optimizer.param_groups[0]["lr"]
                    optimizer.step()
                    scheduler.step()
                    torch.cuda.synchronize()

                    elapsed = time.perf_counter() - t0
                    timings.append(elapsed)

                    row = {
                        "step": step,
                        "episode_index": planned["episode_index"],
                        "frame_index": planned["frame_index"],
                        "paraphrase_variant": para_idx,
                        "data_hash": data_hash,
                        "action_hash": action_hash,
                        "flow_seed": flow_seed,
                        "noise_hash": noise_hash,
                        "time_hash": time_hash,
                        "lr_used": lr_used,
                        "loss": loss_metrics["loss"],
                        "seconds": elapsed
                    }
                    row.update(loss_metrics)
                    history.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()

                    if step % paired_cfg["training"].get("log_every", 10) == 0 or step == experiment["updates"]:
                        speed = statistics.mean(timings[-20:])
                        remaining = (experiment["updates"] - step) * speed / 60.0
                        print(f"  [{step:04d}/{experiment['updates']}] loss={row['loss']:.5f} ({speed:.2f} s/update, remaining ~{remaining:.1f} min)", flush=True)

                    if step in save_at:
                        print(f"  Saving checkpoint {step}...", flush=True)
                        checkpoint = journal.new_attempt(f"step_{step:06d}")
                        parameters_after, _ = parameter_evidence(raw_student)
                        updates = compare_parameter_updates(before, parameters_after)
                        model_dir = checkpoint / "pretrained_model"
                        raw_student.save_pretrained(model_dir)
                        pre.save_pretrained(model_dir)
                        post.save_pretrained(model_dir)
                        save_training_state(checkpoint, step, optimizer=optimizer, scheduler=scheduler, num_processes=1, batch_size=1)
                        atomic_json(checkpoint / "parameters.json", parameters_after)
                        with (checkpoint / "training_log.jsonl").open("w", encoding="utf-8") as saved_log:
                            for saved in history:
                                saved_log.write(json.dumps(saved, allow_nan=False) + "\n")
                        relative = lambda path: path.relative_to(arm_dir).as_posix()
                        result = {"arm": arm, "step": step, "checkpoint": relative(checkpoint),
                                  "training_log": relative(checkpoint / "training_log.jsonl"),
                                  "parameter_evidence": relative(checkpoint / "parameters.json"),
                                  "schedule_hash": schedule_hash(schedule), "parameter_update": updates}
                        atomic_json(checkpoint / "summary.json", result)
                        artifacts = [arm_dir / "initial_parameters.json", checkpoint / "parameters.json",
                                     checkpoint / "training_log.jsonl", checkpoint / "summary.json",
                                     *sorted(model_dir.iterdir()), *sorted((checkpoint / "training_state").iterdir())]
                        journal.finish(f"step_{step:06d}", result, [path for path in artifacts if path.is_file()])
                        if step == experiment["updates"]:
                            atomic_json(arm_dir / "finished.json", result)
                        print(f"  Checkpoint {step} committed.", flush=True)

            print(f"[4/4] Completed training for {arm}", flush=True)
            execution["passed"] = True

        except BaseException as e:
            execution["error"] = {"type": type(e).__name__, "message": str(e)}
            raise
        finally:
            execution["elapsed_seconds"] = time.perf_counter() - started
            atomic_json(attempt / "execution.json", execution)


if __name__ == "__main__":
    main()
