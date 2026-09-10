"""Repeat the original task, trace inference, and replay recorded divergent inputs.

Run manually in the validated WSL environment. No training or backend-flag changes.
"""

from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.protocol import Checkpoints, atomic_json, file_sha256, fingerprint, run_lock


def main():
    if platform.system() != "Linux":
        raise RuntimeError("Run in the validated WSL environment")
    if os.environ.get("MUJOCO_GL", "egl") != "egl":
        raise RuntimeError("Expected the validated EGL renderer")
    os.environ["MUJOCO_GL"] = "egl"
    settings = json.loads((ROOT / "configs/repeat_diagnostic.json").read_text())
    source = ROOT / settings["source_run"]
    directory = ROOT / settings["output_dir"]
    baseline = json.loads((source / "identity.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    audit = json.loads((ROOT / settings["source_audit"]).read_text())
    baseline_fingerprint = fingerprint(baseline)
    if not summary["all_trials_complete"] or not audit["artifact_integrity_passed"] or audit["run_fingerprint"] != baseline_fingerprint:
        raise RuntimeError("Expected the audited, completed source pilot")
    rows = [r for r in baseline["instructions"] if r["condition"] in {"original", "original_repeat"}]
    if len(rows) != 2 or rows[0]["text"] != rows[1]["text"]:
        raise RuntimeError("Expected the original and its exact repeat")
    config = baseline["config"]
    with run_lock(directory):
        started = time.perf_counter()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        execution = {"passed": False, "stages_seconds": {}}

        def stage(name):
            execution["stages_seconds"][name] = time.perf_counter() - started
            print(f"  Elapsed {execution['stages_seconds'][name]:.1f} s: {name}", flush=True)

        try:
            print("[1/5] Verify source protocol and import dependencies; cold imports may be slow", flush=True)
            for name, expected in baseline["project_sources"].items():
                if file_sha256(ROOT / name) != expected:
                    raise RuntimeError(f"Source pilot code changed: {name}")
            import av
            import gymnasium as gym
            import numpy as np
            import torch
            from huggingface_hub import try_to_load_from_cache
            from libero.libero import benchmark, get_libero_path
            from lerobot.envs.configs import LiberoEnv as LiberoConfig
            from lerobot.envs.factory import make_env_pre_post_processors
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.scripts.lerobot_eval import rollout
            from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
            from lerobot.utils.io_utils import write_video
            from lerobot.utils.random_utils import set_seed, get_rng_state, set_rng_state, serialize_rng_state
            from smolvla_grounding.diagnostic_policy import TracedPolicy, capture_noise, require_empty_queue
            from smolvla_grounding.diagnostic_records import (
                compare_traces, decode_record, encode_record, pack_rng, pack_tensors, unpack_rng, unpack_tensors,
            )
            from smolvla_grounding.libero_pairing import array_hashes, require_matching_state
            from smolvla_grounding.paired_env import PairedLiberoEnv
            from smolvla_grounding.rollout_records import summarize_goals, verify_pairing
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            for name, expected in baseline["packages"].items():
                if importlib.metadata.version(name) != expected:
                    raise RuntimeError(f"Dependency version changed: {name}")
            for name, expected in baseline["upstream_sources"].items():
                if file_sha256(importlib.import_module(name).__file__) != expected:
                    raise RuntimeError(f"Upstream inference implementation changed: {name}")
            proof, pairing_identity = verify_pairing(ROOT, config, baseline["instructions"], ROOT / baseline["run_config"]["pairing_report"])
            for name, expected in pairing_identity["upstream_sources"].items():
                if file_sha256(importlib.import_module(name).__file__) != expected:
                    raise RuntimeError(f"Upstream environment implementation changed: {name}")
            task_suite = benchmark.get_benchmark_dict()[config["suite"]]()
            task = task_suite.get_task(config["task_id"])
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            init_file = Path(get_libero_path("init_states")) / task.problem_folder / Path(task.init_states_file).name
            if file_sha256(bddl) != pairing_identity["bddl_sha256"] or file_sha256(init_file) != pairing_identity["init_file_sha256"]:
                raise RuntimeError("Task assets changed after pairing validation")
            backend = {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                       "cudnn_benchmark": torch.backends.cudnn.benchmark,
                       "cudnn_deterministic": torch.backends.cudnn.deterministic,
                       "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                       "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}
            if backend != baseline["backend"]:
                raise RuntimeError("Backend settings differ from the pilot; do not silently change the diagnostic")
            stage("imports_and_source_checks")

            print("[2/5] Verify checkpoint and cached backbone; validate lossless trace encoding", flush=True)
            path = ROOT / baseline["checkpoint"]["local_dir"]
            cached = try_to_load_from_cache(baseline["backbone"]["repo_id"], "config.json",
                                            revision=baseline["backbone"]["revision"])
            if not isinstance(cached, str):
                raise RuntimeError("Validated backbone snapshot missing from the local cache")
            snapshot = Path(cached).parent
            for base, hashes in ((path, baseline["policy_hashes"]), (snapshot, baseline["backbone_hashes"])):
                for name, expected in hashes.items():
                    print(f"  Checking {name}", flush=True)
                    if file_sha256(base / name) != expected:
                        raise RuntimeError(f"Model artifact changed: {name}")
            # Exercise exact byte round trips before trusting the diagnostic records.
            metadata, arrays = {}, {}
            example = {"bf16": torch.tensor([0.1, -0.0], dtype=torch.bfloat16),
                       "mask": torch.tensor([True, False]), "scalar": torch.tensor(3, dtype=torch.int64)}
            expected = pack_tensors(example, "example", metadata, arrays)
            expected_rng = pack_rng(get_rng_state(), "rng", metadata, arrays)
            restored_meta, restored_arrays = decode_record(encode_record(metadata, arrays))
            if pack_tensors(unpack_tensors("example", restored_meta, restored_arrays), "example", {}, {}) != expected:
                raise RuntimeError("Tensor trace encoding is not lossless")
            if pack_rng(unpack_rng("rng", restored_meta, restored_arrays), "rng", {}, {}) != expected_rng:
                raise RuntimeError("RNG trace encoding is not lossless")
            stage("weight_checks_and_trace_roundtrip")
            names = ("configs/repeat_diagnostic.json", "scripts/diagnose_repeat.py", "scripts/run_repeat_diagnostic.sh",
                     "src/smolvla_grounding/diagnostic_records.py", "src/smolvla_grounding/diagnostic_policy.py")
            import subprocess
            gpu_driver = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                        check=True, capture_output=True, text=True).stdout.strip().splitlines()
            if gpu_driver != baseline["gpu_driver"]:
                raise RuntimeError("GPU/driver differs from the source pilot")
            identity = {"source_run_fingerprint": baseline_fingerprint, "settings": settings,
                        "source_audit_sha256": file_sha256(ROOT / settings["source_audit"]),
                        "sources": {name: file_sha256(ROOT / name) for name in names}, "backend": backend,
                        "packages": baseline["packages"], "gpu_driver": gpu_driver,
                        "python": platform.python_version(), "kernel": platform.release()}
            checkpoints = Checkpoints(directory, identity)
            complete = {r["id"]: checkpoints.completed(r["id"]) for r in rows}
            replay_done = checkpoints.completed("replay")
            need_model = any(v is None for v in complete.values()) or replay_done is None
            if need_model:
                print("[3/5] Load one official policy and checkpoint processors", flush=True)
                policy_config = SmolVLAConfig.from_pretrained(path)
                policy_config.device, policy_config.push_to_hub = "cuda", False
                for name, value in config["policy"].items():
                    setattr(policy_config, name, value)
                policy_config.vlm_model_name = str(snapshot)
                if policy_config.n_action_steps != 1 or (policy_config.rtc_config and policy_config.rtc_config.enabled):
                    raise RuntimeError("Diagnostic requires the inspected action-only queue and disabled RTC")
                set_seed(config["policy_seed"])
                policy = SmolVLAPolicy.from_pretrained(path, config=policy_config, strict=True).eval()
                pre, post = make_pre_post_processors(policy_cfg=policy_config, pretrained_path=str(path),
                    preprocessor_overrides={"device_processor": {"device": "cuda"},
                                            "tokenizer_processor": {"tokenizer_name": str(snapshot)}})
                env_config = LiberoConfig(task=config["suite"], task_ids=[config["task_id"]])
                env_pre, env_post = make_env_pre_post_processors(env_config, policy_config)
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
                expected_tokens = tokenizer(rows[0]["text"] + "\n", truncation=False)["input_ids"]
                stage("model_and_processors_loaded")
            else:
                print("[3/5] Verified completed diagnostic; model loading unnecessary", flush=True)

            print("[4/5] Record both original-instruction trials, 280 steps each", flush=True)
            results = []
            for index, row in enumerate(rows, 1):
                result = complete[row["id"]]
                if result is None:
                    print(f"  [{index}/2] {row['id']}", flush=True)
                    attempt = checkpoints.new_attempt(row["id"])
                    env = vec = None
                    frames = {"image": [], "image2": []}
                    trial_started = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
                    try:
                        for processor in (pre, post, env_pre, env_post):
                            processor.reset()
                        set_seed(config["environment_seed"])
                        with (attempt / "steps.jsonl").open("w", encoding="utf-8") as step_stream, \
                             (attempt / "inference.jsonl").open("w", encoding="utf-8") as trace_stream, \
                             zipfile.ZipFile(attempt / "inference.zip", "w", compression=zipfile.ZIP_STORED) as archive:
                            env = PairedLiberoEnv(task_suite, config,
                                                 rows[0], row, proof["checks"][0]["hashes"], attempt, step_stream)
                            traced = TracedPolicy(policy, env, archive, trace_stream).eval()
                            vec = gym.vector.SyncVectorEnv([lambda: env])

                            def render(_):
                                for camera in frames:
                                    frames[camera].append(env.latest_pixels[camera][::-1, ::-1].copy())
                                if len(frames["image"]) == 1:
                                    set_seed(config["policy_seed"])
                                    actual = array_hashes({k: v.cpu().numpy() for k, v in serialize_rng_state().items()})
                                    require_matching_state(summary["episodes"][0]["initial_rng_hashes"], actual)

                            def preprocess(observation):
                                if observation["task"] != [row["text"]]:
                                    raise RuntimeError("Unexpected instruction")
                                batch = pre(observation)
                                tokens = batch[OBS_LANGUAGE_TOKENS][0][batch[OBS_LANGUAGE_ATTENTION_MASK][0].bool()]
                                if tokens.cpu().tolist() != expected_tokens:
                                    raise RuntimeError("Policy token mismatch")
                                return batch

                            tensors = rollout(vec, traced, env_pre, env_post, preprocess, post,
                                              seeds=[config["environment_seed"]], render_callback=render)
                            steps = config["environment"]["episode_length"]
                            if len(traced.records) != steps or len(env.records) != steps:
                                raise RuntimeError("Incomplete diagnostic trial")
                            trajectory = {k: v.cpu().numpy()[0] for k, v in tensors.items()}
                            if not np.array_equal(trajectory["action"], np.asarray([r["action"] for r in env.records], dtype=trajectory["action"].dtype)):
                                raise RuntimeError("Action tensor/log mismatch")
                            trajectory["sim_state"] = np.stack(env.sim_states)
                            np.savez_compressed(attempt / "trajectory.npz", **trajectory)
                            goals = summarize_goals([r["goals"] for r in env.records], config["goals"])
                        # Archives are closed before checkpoint completion or replay reads.
                        for camera in frames:
                            video_path = attempt / (camera + ".mp4")
                            write_video(video_path, frames[camera], fps=config["environment"]["control_freq"])
                            with av.open(str(video_path)) as video:
                                if sum(1 for _ in video.decode(video=0)) != steps + 1:
                                    raise RuntimeError("Incomplete diagnostic video")
                        vec.close()
                        env = vec = None
                        result = {"instruction_id": row["id"], "attempt": attempt.relative_to(directory).as_posix(),
                                  "goals": goals, "steps": steps, "trial_seconds": time.perf_counter() - trial_started,
                                  "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                  "trace_archive_bytes": (attempt / "inference.zip").stat().st_size}
                        atomic_json(attempt / "episode.json", result)
                        files = [attempt / n for n in ("steps.jsonl", "inference.jsonl", "inference.zip", "trajectory.npz",
                                 "initial_state.npz", "initial_hashes.json", "image.mp4", "image2.mp4", "episode.json")]
                        checkpoints.finish(row["id"], result, files)
                    except BaseException as exc:
                        atomic_json(attempt / "error.json", {"type": type(exc).__name__, "message": str(exc)})
                        raise
                    finally:
                        if vec is not None:
                            vec.close()
                        elif env is not None:
                            env.close()
                else:
                    print(f"  [{index}/2] {row['id']}: verified saved trial", flush=True)
                results.append(result)
                stage(row["id"] + "_complete")
            traces = [[json.loads(line) for line in (directory / r["attempt"] / "inference.jsonl").read_text().splitlines()]
                      for r in results]
            comparison = compare_traces(*traces)
            atomic_json(directory / "trace_comparison.json", comparison)

            print("[5/5] Replay recorded inputs twice with restored RNG and empty action queues", flush=True)
            focus_steps = {audit["repeat_control"]["first_action_difference_step"],
                           comparison["first_different_step"]["normalized_action"]} - {None}
            if replay_done is None:
                attempt = checkpoints.new_attempt("replay")
                replays = []
                for result in results:
                    with zipfile.ZipFile(directory / result["attempt"] / "inference.zip") as archive:
                        for step in sorted(focus_steps):
                            print(f"  Replay {result['instruction_id']} step {step}", flush=True)
                            metadata, arrays = decode_record(archive.read(f"step_{step:04d}.npz"))
                            observed = []
                            for repetition in (1, 2):
                                policy.reset()
                                require_empty_queue(policy)
                                batch = {**metadata["batch_other"], **unpack_tensors("input", metadata, arrays)}
                                set_rng_state(unpack_rng("rng_before", metadata, arrays))
                                input_hash = pack_tensors({k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}, "input", {}, {})
                                rng_hash = pack_rng(get_rng_state(), "rng_before", {}, {})
                                with torch.inference_mode(), capture_noise(policy) as noise:
                                    action = policy.select_action(batch)
                                if len(noise) != 1 or not torch.isfinite(action).all():
                                    raise RuntimeError("Invalid replay output or sampled noise")
                                output_hash = pack_tensors({"action": action}, "output", {}, {})
                                noise_hash = pack_tensors({"noise": noise[0]}, "noise", {}, {})
                                observed.append({"repetition": repetition,
                                    "input_equal_to_recorded": input_hash == metadata["trace"]["input"],
                                    "rng_equal_to_recorded": rng_hash == metadata["trace"]["rng_before"],
                                    "noise_equal_to_recorded": noise_hash == metadata["trace"]["noise"],
                                    "action_equal_to_recorded": output_hash == metadata["trace"]["normalized_action"],
                                    "action_hash": output_hash, "normalized_action": action.float().cpu().tolist()})
                            if not all(r["input_equal_to_recorded"] and r["rng_equal_to_recorded"] for r in observed):
                                raise RuntimeError("Replay did not restore the exact recorded input/RNG")
                            replays.append({"trial": result["instruction_id"], "step": step, "replays": observed,
                                            "two_replay_actions_equal": observed[0]["action_hash"] == observed[1]["action_hash"]})
                replay_done = {"cases": replays, "scope": "Two replays per recorded input; equality does not prove universal determinism."}
                atomic_json(attempt / "replay.json", replay_done)
                checkpoints.finish("replay", replay_done, [attempt / "replay.json"])
            stage("replays_complete")
            report = {"execution_complete": True, "source_run_fingerprint": baseline_fingerprint,
                      "trials": results, "trace_comparison": comparison, "replay": replay_done,
                      "limitations": ["Tracing adds synchronization and may change timing-dependent behavior.",
                                      "Differences with equal input/noise implicate inference or unrecorded model state, not necessarily a specific CUDA kernel.",
                                      "No backend flags or precision settings were changed."]}
            atomic_json(directory / "diagnostic.json", report)
            print("PASS: repeat diagnostic and recorded-input replays completed.", flush=True)
            print(json.dumps(comparison, indent=2), flush=True)
            print("Saved " + (directory / "diagnostic.json").relative_to(ROOT).as_posix(), flush=True)
            execution["passed"] = True
        except BaseException as exc:
            execution["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            execution["elapsed_seconds"] = time.perf_counter() - started
            atomic_json(directory / f"execution_{stamp}.json", execution)


if __name__ == "__main__":
    main()
