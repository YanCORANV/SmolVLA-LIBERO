"""One genuine LeRobot optimizer update on a verified training demonstration."""

from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.protocol import atomic_json, file_sha256
from smolvla_grounding.training_profile_records import compare_parameter_updates, verify_ready


def main():
    settings = json.loads((ROOT / "configs/training_profile.json").read_text())
    directory = Path(os.environ["SMOLVLA_TRAINING_PROFILE_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "settings": settings, "stage": "data", "time_utc": datetime.now(timezone.utc).isoformat()}
    started = time.perf_counter()
    try:
        print("[1/5] Verify prepared data and installed training dependencies", flush=True)
        data_config, ready, dataset_root = verify_ready(ROOT, settings)
        packages = ("lerobot", "torch", "accelerate", "datasets", "pyarrow", "safetensors", "transformers")
        report["packages"] = {name: importlib.metadata.version(name) for name in packages}
        report["data_ready_sha256"] = file_sha256(ROOT / settings["data_ready"])
        print("Cold official imports and first Arrow cache construction may take minutes.", flush=True)
        import numpy as np
        import pyarrow as pa
        import torch
        from accelerate import Accelerator
        from huggingface_hub import try_to_load_from_cache
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        from lerobot.datasets.factory import resolve_delta_timestamps
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.scripts.lerobot_train import update_policy
        from lerobot.common.train_utils import save_training_state
        from lerobot.utils.collate import lerobot_collate_fn
        from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
        from lerobot.utils.random_utils import set_seed
        from PIL import Image
        from smolvla_grounding.training_policy import FP32ExpertSmolVLAPolicy, parameter_evidence

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("This profile requires CUDA with BF16 support")
        if (settings["batch_size"] != 1 or settings["optimizer_updates"] != 1
                or settings["mixed_precision"] != "bf16" or settings["trainable_parameter_dtype"] != "float32"):
            raise ValueError("This acceptance program performs exactly one batch-1 update")
        baseline = json.loads((ROOT / settings["source_run"] / "identity.json").read_text())
        for name, expected in baseline["packages"].items():
            if importlib.metadata.version(name) != expected:
                raise RuntimeError("Validated evaluation dependency changed: " + name)
        checkpoint = ROOT / baseline["checkpoint"]["local_dir"]
        cached = try_to_load_from_cache(baseline["backbone"]["repo_id"], "config.json", revision=baseline["backbone"]["revision"])
        if not isinstance(cached, str):
            raise RuntimeError("Pinned backbone cache missing; rerun the existing checkpoint setup")
        snapshot = Path(cached).parent
        for parent, hashes in ((checkpoint, baseline["policy_hashes"]), (snapshot, baseline["backbone_hashes"])):
            for name, expected in hashes.items():
                print("  Verify checkpoint artifact: " + name, flush=True)
                if file_sha256(parent / name) != expected:
                    raise RuntimeError("Checkpoint artifact changed: " + name)
        config = SmolVLAConfig.from_pretrained(checkpoint)
        config.device, config.push_to_hub, config.vlm_model_name = "cuda", False, str(snapshot)
        for name in ("freeze_vision_encoder", "train_expert_only", "train_state_proj"):
            setattr(config, name, settings[name])
        config.optimizer_lr = settings["learning_rate"]
        report["backbone"] = baseline["backbone"]
        report["policy_settings"] = {key: getattr(config, key) for key in ("chunk_size", "n_action_steps", "num_steps", "use_amp", "train_expert_only", "train_state_proj")}

        print("[2/5] Load the official dataset; compare first/tail action chunks with source rows", flush=True)
        report["stage"] = "official_dataset"
        meta = LeRobotDatasetMetadata(data_config["repo_id"], root=dataset_root, revision=data_config["revision"])
        delta = resolve_delta_timestamps(config, meta)
        train_ids = ready["selection"]["split"]["train"]
        dataset = LeRobotDataset(data_config["repo_id"], root=dataset_root, revision=data_config["revision"],
                                 episodes=train_ids, delta_timestamps=delta, return_uint8=False)
        expected_frames = sum(r["frames"] for r in ready["episodes"] if r["episode_index"] in train_ids)
        if len(dataset) != expected_frames:
            raise RuntimeError("Official dataset selected the wrong frame count")
        available = sorted(int(x) for x in dataset.hf_dataset.data.column("episode_index").unique().to_pylist())
        if available != train_ids:
            raise RuntimeError("Official dataloader included wrong or held-out episodes")
        # Explicitly use the first training demonstration, with a separate tail-boundary check.
        episode = train_ids[0]
        indices = dataset.hf_dataset.data.column("episode_index").to_numpy()
        selected = np.flatnonzero(indices == episode)
        length = next(r["frames"] for r in ready["episodes"] if r["episode_index"] == episode)
        if len(selected) != length:
            raise RuntimeError("First training demonstration missing from official loader")
        source_rows = dataset.hf_dataset.data.table.select(["action"]).take(pa.array(selected))
        source_actions = torch.tensor(source_rows.column("action").to_pylist(), dtype=torch.float32)
        samples = [dataset[int(selected[0])], dataset[int(selected[-1])]]
        for sample, offset in zip(samples, (0, length - 1)):
            expected_indices = np.clip(np.arange(config.chunk_size) + offset, 0, length - 1)
            if not torch.equal(sample["action"], source_actions[expected_indices]):
                raise RuntimeError("Official action chunk differs from source actions or crosses episode boundary")
            expected_padding = torch.tensor(np.arange(config.chunk_size) + offset >= length)
            if not torch.equal(sample["action_is_pad"], expected_padding):
                raise RuntimeError("Official action padding mask differs from episode boundary")
            if sample["task"] != data_config["original_instruction"]:
                raise RuntimeError("Official dataset emitted the wrong instruction")
        sample = samples[0]
        arrays = {k: v.numpy() for k, v in sample.items() if isinstance(v, torch.Tensor)}
        np.savez_compressed(directory / "sample.npz", **arrays)
        for key in ("observation.images.image", "observation.images.image2"):
            value = sample[key]
            # A [0] observation delta retains a singleton time axis in the official reader.
            if value.shape == (1, 3, 256, 256):
                value = value[0]
            if value.shape != (3, 256, 256) or not torch.isfinite(value).all() or value.min() < 0 or value.max() > 1:
                raise RuntimeError("Expected official CHW float images in [0,1]")
            Image.fromarray((value.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(directory / (key.split('.')[-1] + ".png"))
        report["dataset"] = {"episodes": train_ids, "frames": len(dataset), "sample_episode": episode,
                             "instruction": sample["task"], "action_shape": list(sample["action"].shape), "delta_timestamps": delta,
                             "first_and_tail_chunk_checks": "passed", "image_orientation": "official dataset pixels unchanged; preview saved"}
        atomic_json(directory / "step_report.json", report)

        print("[3/5] Load frozen-VLM policy, checkpoint processors and official AdamW", flush=True)
        report["stage"] = "model"
        set_seed(settings["seed"])
        policy = FP32ExpertSmolVLAPolicy.from_pretrained(checkpoint, config=config, strict=True)
        if any(p.requires_grad for p in policy.model.vlm_with_expert.vlm.parameters()):
            raise RuntimeError("VLM is not frozen")
        pre, post = make_pre_post_processors(config, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": "cuda"}, "tokenizer_processor": {"tokenizer_name": str(snapshot)}})
        batch = pre(lerobot_collate_fn([sample]))
        if batch["action"].shape != (1, config.chunk_size, 7) or not torch.isfinite(batch["action"]).all():
            raise RuntimeError("Invalid processed action batch")
        # Pre/post action normalization must round-trip using the checkpoint statistics.
        restored = post(batch["action"].clone()).cpu()
        if not torch.allclose(restored, sample["action"].unsqueeze(0), atol=1e-5, rtol=1e-5):
            raise RuntimeError("Checkpoint action normalization round-trip failed")
        before, counts = parameter_evidence(policy)
        report["parameter_counts"] = counts
        print("  Parameter counts: " + json.dumps(counts), flush=True)
        optimizer = config.get_optimizer_preset().build([p for p in policy.parameters() if p.requires_grad])
        accelerator = Accelerator(mixed_precision=settings["mixed_precision"], step_scheduler_with_optimizer=False)
        if accelerator.num_processes != 1 or accelerator.device.type != "cuda":
            raise RuntimeError("Use one CUDA process for this profile")
        policy, optimizer = accelerator.prepare(policy, optimizer)
        raw = accelerator.unwrap_model(policy)
        guards = {}

        def guard_gradient(opt, args, kwargs):
            checked = 0
            for name, parameter in raw.named_parameters():
                if not parameter.requires_grad and parameter.grad is not None:
                    raise RuntimeError("Frozen parameter received a gradient: " + name)
                if parameter.grad is not None:
                    checked += 1
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError("Nonfinite gradient before optimizer step: " + name)
            if checked == 0:
                raise RuntimeError("No gradients reached optimizer")
            guards["finite_gradient_tensors"] = checked

        handle = optimizer.optimizer.register_step_pre_hook(guard_gradient)
        metrics = MetricsTracker(1, len(dataset), len(train_ids),
            {key: AverageMeter(key, ":.5f") for key in ("loss", "grad_norm", "lr", "update_s", "gpu_mem_gb")})
        print("[4/5] Execute one real upstream forward/backward/update with BF16 autocast", flush=True)
        report["stage"] = "optimizer_step"
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        metrics, _ = update_policy(metrics, policy, batch, optimizer, config.optimizer_grad_clip_norm, accelerator)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_start
        handle.remove()
        metrics.step()
        if any(not math.isfinite(float(metrics.to_dict(False)[key])) for key in ("loss", "grad_norm")):
            raise RuntimeError("Nonfinite loss/gradient norm")
        report["update"] = {**metrics.to_dict(False), **guards, "wall_seconds": elapsed,
                            "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                            "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                            "free_device_bytes_after_update": torch.cuda.mem_get_info()[0],
                            "timing_scope": "First update including finite-gradient guard; not steady-state throughput"}
        report["update"]["optimizer_updates"] = 1
        after, _ = parameter_evidence(raw)
        report["parameter_update"] = compare_parameter_updates(before, after)
        atomic_json(directory / "parameters_after.json", after)
        print(f"  Loss {metrics.to_dict(False)['loss']:.6f}; {elapsed:.2f} s; peak reserved {report['update']['torch_peak_reserved_bytes']/2**30:.2f} GiB", flush=True)

        print("[5/5] Save policy, processors, optimizer and RNG with official APIs", flush=True)
        report["stage"] = "checkpoint"
        path = directory / "checkpoint/pretrained_model"
        raw.save_pretrained(path)
        pre.save_pretrained(path)
        post.save_pretrained(path)
        save_training_state(directory / "checkpoint", 1, optimizer=optimizer, num_processes=1, batch_size=1)
        report["artifacts"] = {p.relative_to(directory).as_posix(): file_sha256(p) for p in [
            directory / "sample.npz", directory / "parameters_after.json", *sorted(path.iterdir()),
            *sorted((directory / "checkpoint/training_state").iterdir())] if p.is_file()}
        report["passed"] = True
        print("PASS: one optimizer update saved; fresh-process reload is next.", flush=True)
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if "torch" in locals() and torch.cuda.is_initialized():
            report["cuda_peak_reserved_at_exit_bytes"] = torch.cuda.max_memory_reserved()
        atomic_json(directory / "step_report.json", report)


if __name__ == "__main__":
    main()
