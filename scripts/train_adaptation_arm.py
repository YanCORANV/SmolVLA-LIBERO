"""One paired fine-tuning arm using validated policy and official training APIs."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.adaptation_records import checkpoint_steps, latest_checkpoint, read_jsonl, schedule_hash
from smolvla_grounding.benchmark_records import BenchmarkJournal
from smolvla_grounding.protocol import atomic_json, file_sha256, fingerprint, run_lock
from smolvla_grounding.training_profile_records import compare_parameter_updates, verify_ready


def main():
    directory = Path(os.environ["SMOLVLA_ADAPTATION_DIR"])
    arm = os.environ["SMOLVLA_ADAPTATION_ARM"]
    experiment = json.loads((directory / "experiment.json").read_text())
    settings = experiment["provenance"]["settings"]
    if arm not in settings["arms"]:
        raise ValueError("Unknown experimental arm")
    sources = {**experiment["provenance"]["sources"], **experiment["provenance"]["upstream_sources"]}
    for name, expected in sources.items():
        if file_sha256(ROOT / name) != expected:
            raise RuntimeError("Registered source changed: " + name)
    schedule = json.loads((directory / "schedule.json").read_text())
    profile = json.loads((ROOT / settings["profile_directory"] / "step_report.json").read_text())
    profile_config = json.loads((ROOT / settings["profile_config"]).read_text())
    rows = read_jsonl(ROOT / settings["training_catalog"])
    arm_dir = directory / arm
    with run_lock(arm_dir):
        journal = BenchmarkJournal(arm_dir, {"experiment": experiment, "arm": arm, "schedule_hash": schedule_hash(schedule)})
        last = latest_checkpoint(journal, experiment["updates"], settings["checkpoint_every"])
        if last and last["step"] == experiment["updates"]:
            atomic_json(arm_dir / "finished.json", last)
            print("  Final checkpoint verified; no training repeated.", flush=True)
            return
        attempt = journal.new_attempt("executions")
        execution = {"passed": False, "arm": arm, "stage": "data", "time_utc": datetime.now(timezone.utc).isoformat()}
        started = time.perf_counter()
        try:
            print("[1/4] Verify the accepted data and runtime", flush=True)
            data_config, ready, dataset_root = verify_ready(ROOT, profile_config)
            if file_sha256(ROOT / profile_config["data_ready"]) != profile["data_ready_sha256"]:
                raise RuntimeError("Training data changed since optimizer-step acceptance")
            versions = {name: importlib.metadata.version(name) for name in profile["packages"]}
            if versions != profile["packages"]:
                raise RuntimeError("Training packages differ from the accepted profile")
            import torch
            from accelerate import Accelerator
            from huggingface_hub import try_to_load_from_cache
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
            from lerobot.datasets.factory import resolve_delta_timestamps
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            from lerobot.scripts.lerobot_train import update_policy
            from lerobot.common.train_utils import save_training_state, load_training_state
            from lerobot.utils.collate import lerobot_collate_fn
            from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
            from lerobot.utils.random_utils import get_rng_state, set_seed
            from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
            from transformers import AutoTokenizer
            from smolvla_grounding.training_policy import FP32ExpertSmolVLAPolicy, parameter_evidence
            from smolvla_grounding.diagnostic_records import pack_rng, pack_tensors
            from smolvla_grounding.diagnostic_policy import capture_noise

            baseline = json.loads((ROOT / profile_config["source_run"] / "identity.json").read_text())
            base_path = ROOT / baseline["checkpoint"]["local_dir"]
            cached = try_to_load_from_cache(baseline["backbone"]["repo_id"], "config.json", revision=baseline["backbone"]["revision"])
            if not isinstance(cached, str):
                raise RuntimeError("Pinned backbone snapshot missing")
            snapshot = Path(cached).parent
            for parent, hashes in ((base_path, baseline["policy_hashes"]), (snapshot, baseline["backbone_hashes"])):
                for name, expected in hashes.items():
                    if file_sha256(parent / name) != expected:
                        raise RuntimeError("Original model artifact changed: " + name)
            path = arm_dir / last["checkpoint"] / "pretrained_model" if last else base_path
            config = SmolVLAConfig.from_pretrained(path)
            config.device, config.push_to_hub, config.vlm_model_name = "cuda", False, str(snapshot)
            for name in ("freeze_vision_encoder", "train_expert_only", "train_state_proj"):
                setattr(config, name, profile_config[name])
            config.optimizer_lr = profile_config["learning_rate"]
            config.scheduler_warmup_steps = min(20, experiment["updates"] // 10)
            config.scheduler_decay_steps = experiment["updates"]
            tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
            token_ids = [tokenizer(row["text"] + "\n", truncation=False)["input_ids"] for row in rows]
            if any(len(ids) > config.tokenizer_max_length for ids in token_ids) or len({tuple(x) for x in token_ids}) != len(rows):
                raise RuntimeError("Training wording is truncated or has token collisions")
            print("  Exact training token counts: " + str([len(x) for x in token_ids]), flush=True)
            meta = LeRobotDatasetMetadata(data_config["repo_id"], root=dataset_root, revision=data_config["revision"])
            train_ids = ready["selection"]["split"]["train"]
            dataset = LeRobotDataset(data_config["repo_id"], root=dataset_root, revision=data_config["revision"],
                episodes=train_ids, delta_timestamps=resolve_delta_timestamps(config, meta), return_uint8=False)
            expected_frames = sum(r["frames"] for r in ready["episodes"] if r["episode_index"] in train_ids)
            ep = dataset.hf_dataset.data.column("episode_index").to_numpy()
            fr = dataset.hf_dataset.data.column("frame_index").to_numpy()
            position = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(ep, fr))}
            if sorted(set(int(x) for x in ep)) != train_ids or len(position) != len(dataset) or len(dataset) != expected_frames:
                raise RuntimeError("Official dataset selected wrong, duplicate or held-out frames")

            print("[2/4] Load policy and official optimizer/scheduler", flush=True)
            execution["stage"] = "model"
            set_seed(experiment["seed"])
            policy = FP32ExpertSmolVLAPolicy.from_pretrained(path, config=config, strict=True)
            pre, post = make_pre_post_processors(config, pretrained_path=str(path),
                preprocessor_overrides={"device_processor": {"device": "cuda"}, "tokenizer_processor": {"tokenizer_name": str(snapshot)}})
            optimizer = config.get_optimizer_preset().build([p for p in policy.parameters() if p.requires_grad])
            scheduler = config.get_scheduler_preset().build(optimizer, num_training_steps=experiment["updates"])
            accelerator = Accelerator(mixed_precision=profile_config["mixed_precision"], step_scheduler_with_optimizer=False)
            if accelerator.num_processes != 1 or accelerator.device.type != "cuda":
                raise RuntimeError("Use one CUDA process")
            policy, optimizer = accelerator.prepare(policy, optimizer)
            raw = accelerator.unwrap_model(policy)
            parameters_now, counts = parameter_evidence(raw)
            if last:
                previous_parameters = json.loads((arm_dir / last["parameter_evidence"]).read_text())
                if parameters_now != previous_parameters:
                    raise RuntimeError("Resume policy differs from saved parameter values/dtypes")
                before = json.loads((arm_dir / "initial_parameters.json").read_text())
                history = read_jsonl(arm_dir / last["training_log"])
                if [r["step"] for r in history] != list(range(1, last["step"] + 1)):
                    raise RuntimeError("Committed training history is incomplete")
                restored_step, optimizer, scheduler = load_training_state(arm_dir / last["checkpoint"], optimizer, scheduler)
                if restored_step != last["step"]:
                    raise RuntimeError("Saved optimizer step does not match checkpoint marker")
                print(f"  Restored policy, optimizer, scheduler and RNG at update {restored_step}.", flush=True)
            else:
                before, history, restored_step = parameters_now, [], 0
                initial_path = arm_dir / "initial_parameters.json"
                if initial_path.exists() and json.loads(initial_path.read_text()) != before:
                    raise RuntimeError("Starting parameters changed")
                atomic_json(initial_path, before)
                set_seed(experiment["seed"])
            execution["parameter_counts"] = counts
            execution["resumed_at"] = restored_step
            metrics = MetricsTracker(1, len(dataset), len(train_ids),
                {key: AverageMeter(key, ":.5f") for key in ("loss", "grad_norm", "lr", "update_s", "gpu_mem_gb")},
                initial_step=restored_step)

            def guard_gradient(opt, args, kwargs):
                for name, parameter in raw.named_parameters():
                    if not parameter.requires_grad and parameter.grad is not None:
                        raise RuntimeError("Frozen parameter has gradient: " + name)
                    if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                        raise RuntimeError("Nonfinite gradient before update: " + name)

            handle = optimizer.optimizer.register_step_pre_hook(guard_gradient)
            print(f"[3/4] Train {arm}: updates {restored_step + 1}..{experiment['updates']}", flush=True)
            execution["stage"] = "training"
            timings = []
            save_at = set(checkpoint_steps(experiment["updates"], settings["checkpoint_every"]))
            with (attempt / "uncommitted_steps.jsonl").open("w", encoding="utf-8") as stream:
                for planned in schedule[restored_step:]:
                    step = planned["step"]
                    begin = time.perf_counter()
                    sample = dict(dataset[position[(planned["episode_index"], planned["frame_index"])]] )
                    if sample["task"] != data_config["original_instruction"]:
                        raise RuntimeError("Demonstration task text changed")
                    data_hash = fingerprint(pack_tensors({k: v for k, v in sample.items() if isinstance(v, torch.Tensor)}, "sample", {}, {}))
                    variant = planned["augmented_variant"] if arm == "augmented_language" else 0
                    sample["task"] = rows[variant]["text"]
                    pre.reset()
                    batch = pre(lerobot_collate_fn([sample]))
                    ids = batch[OBS_LANGUAGE_TOKENS][0][batch[OBS_LANGUAGE_ATTENTION_MASK][0].bool()].cpu().tolist()
                    if ids != token_ids[variant]:
                        raise RuntimeError("Wrong training text reached tokenizer")
                    rng_before = fingerprint(pack_rng(get_rng_state(), "rng", {}, {}))
                    lr_used = optimizer.param_groups[0]["lr"]
                    with capture_noise(raw) as noise:
                        metrics, _ = update_policy(metrics, policy, batch, optimizer, config.optimizer_grad_clip_norm, accelerator, scheduler)
                    torch.cuda.synchronize()
                    metrics.step()
                    if len(noise) != 1 or any(not math.isfinite(float(metrics.to_dict(False)[k])) for k in ("loss", "grad_norm")):
                        raise RuntimeError("Invalid loss, gradient norm or noise-call count")
                    elapsed = time.perf_counter() - begin
                    row = {"step": step, "episode_index": planned["episode_index"], "frame_index": planned["frame_index"],
                           "variant_index": variant, "instruction_id": rows[variant]["id"], "data_hash": data_hash,
                           "rng_before": rng_before, "noise_hash": fingerprint(pack_tensors({"noise": noise[0]}, "noise", {}, {})),
                           "lr_used": lr_used, "loss": metrics.to_dict(False)["loss"], "grad_norm": metrics.to_dict(False)["grad_norm"],
                           "seconds": elapsed, "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                    history.append(row)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    timings.append(elapsed)
                    if step % settings["log_every"] == 0 or step == experiment["updates"]:
                        speed = statistics.mean(timings[-20:])
                        remaining = (experiment["updates"] - step) * speed / 60
                        print(f"  [{step}/{experiment['updates']}] loss={row['loss']:.5f}, {speed:.2f} s/update, remaining updates ~{remaining:.1f} min (excludes saves)", flush=True)
                    if step in save_at:
                        print(f"  Save checkpoint {step}: policy, optimizer, scheduler, RNG and committed history", flush=True)
                        checkpoint = journal.new_attempt(f"step_{step:06d}")
                        parameters_after, _ = parameter_evidence(raw)
                        updates = compare_parameter_updates(before, parameters_after)
                        model_dir = checkpoint / "pretrained_model"
                        raw.save_pretrained(model_dir)
                        pre.save_pretrained(model_dir)
                        post.save_pretrained(model_dir)
                        save_training_state(checkpoint, step, optimizer=optimizer, scheduler=scheduler, num_processes=1, batch_size=1)
                        atomic_json(checkpoint / "parameters.json", parameters_after)
                        with (checkpoint / "training_log.jsonl").open("w", encoding="utf-8") as saved_log:
                            for saved in history:
                                saved_log.write(json.dumps(saved, allow_nan=False) + "\n")
                        with (checkpoint / "training_log.csv").open("w", encoding="utf-8", newline="") as saved_csv:
                            writer = csv.DictWriter(saved_csv, fieldnames=list(history[0]))
                            writer.writeheader()
                            writer.writerows(history)
                        relative = lambda p: p.relative_to(arm_dir).as_posix()
                        result = {"arm": arm, "step": step, "checkpoint": relative(checkpoint),
                                  "training_log": relative(checkpoint / "training_log.jsonl"),
                                  "parameter_evidence": relative(checkpoint / "parameters.json"),
                                  "schedule_hash": schedule_hash(schedule), "parameter_update": updates,
                                  "peak_reserved_bytes": max(r["torch_peak_reserved_bytes"] for r in history)}
                        atomic_json(checkpoint / "summary.json", result)
                        artifacts = [arm_dir / "initial_parameters.json", checkpoint / "parameters.json", checkpoint / "training_log.jsonl", checkpoint / "training_log.csv", checkpoint / "summary.json",
                                     *sorted(model_dir.iterdir()), *sorted((checkpoint / "training_state").iterdir())]
                        journal.finish(f"step_{step:06d}", result, [p for p in artifacts if p.is_file()])
                        if step == experiment["updates"]:
                            atomic_json(arm_dir / "finished.json", result)
                        print(f"  Checkpoint {step} committed.", flush=True)
            handle.remove()
            execution["passed"] = True
            print("[4/4] PASS: " + arm + " training completed; rollout evaluation pending.", flush=True)
        except BaseException as exc:
            execution["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            execution["elapsed_seconds"] = time.perf_counter() - started
            execution["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            atomic_json(attempt / "execution.json", execution)


if __name__ == "__main__":
    main()
