"""Run the six approved task-0 conditions using the official LeRobot rollout.

Use run_language_pilot.sh to also retain console, timing and device-memory logs.
"""

from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.protocol import Checkpoints, atomic_json, file_sha256, load_protocol, run_lock
from smolvla_grounding.rollout_records import export_episode_table, summarize_goals, verify_pairing


def main():
    if platform.system() != "Linux":
        raise RuntimeError("Use the validated WSL/Linux project environment")
    if os.environ.get("MUJOCO_GL", "egl") != "egl":
        raise RuntimeError("Use the previously validated EGL renderer")
    os.environ["MUJOCO_GL"] = "egl"
    config, rows = load_protocol(ROOT)
    run_config = json.loads((ROOT / "configs/language_rollout.json").read_text(encoding="utf-8"))
    directory = ROOT / run_config["output_dir"]
    with run_lock(directory):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        started = time.perf_counter()
        execution = {"started_utc": stamp, "passed": False, "standard_evaluation": False}
        try:
            print("[1/5] Verify passed pairing evidence, import official policy/evaluator", flush=True)
            proof, pairing_identity = verify_pairing(ROOT, config, rows, ROOT / run_config["pairing_report"])
            import av
            import gymnasium as gym
            import numpy as np
            import torch
            from huggingface_hub import try_to_load_from_cache
            from transformers import AutoTokenizer
            from libero.libero import benchmark, get_libero_path
            from lerobot.envs.configs import LiberoEnv as LiberoConfig
            from lerobot.envs.factory import make_env_pre_post_processors
            from lerobot.policies.factory import make_pre_post_processors
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            from lerobot.scripts.lerobot_eval import rollout
            from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
            from lerobot.utils.io_utils import write_video
            from lerobot.utils.random_utils import set_seed, serialize_rng_state
            from smolvla_grounding.libero_pairing import array_hashes, require_matching_state
            from smolvla_grounding.paired_env import PairedLiberoEnv

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            versions = {name: importlib.metadata.version(name) for name in pairing_identity["packages"]}
            if versions != pairing_identity["packages"]:
                raise RuntimeError("Dependencies changed after the passed pairing check")
            for module_name, expected in pairing_identity["upstream_sources"].items():
                if file_sha256(importlib.import_module(module_name).__file__) != expected:
                    raise RuntimeError(f"Pairing dependency source changed: {module_name}")
            checkpoint = json.loads((ROOT / "configs/checkpoint.json").read_text(encoding="utf-8"))
            path = ROOT / checkpoint["local_dir"]
            backbone = pairing_identity["backbone"]
            cached = try_to_load_from_cache(backbone["repo_id"], "config.json", revision=backbone["revision"])
            if not isinstance(cached, str):
                raise RuntimeError("The validated immutable backbone snapshot is missing from this HF cache")
            snapshot = Path(cached).parent
            tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
            if file_sha256(snapshot / "tokenizer.json") != backbone["tokenizer_sha256"]:
                raise RuntimeError("Cached tokenizer changed after validation")
            if file_sha256(snapshot / "tokenizer_config.json") != backbone["tokenizer_config_sha256"]:
                raise RuntimeError("Cached tokenizer configuration changed after validation")
            token_ids = {row["id"]: tokenizer(row["text"] + "\n", truncation=False)["input_ids"] for row in rows}

            print("[2/5] Hash fixed checkpoint/backbone files and record run provenance", flush=True)
            files = [path / name for name in (
                "config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json",
                "policy_preprocessor_step_5_normalizer_processor.safetensors",
                "policy_postprocessor_step_1_unnormalizer_processor.safetensors",
            )]
            model_hashes = {}
            for i, file in enumerate(files, 1):
                print(f"  Policy file [{i}/{len(files)}]: {file.name}", flush=True)
                model_hashes[file.name] = file_sha256(file)
            print("  Backbone weights (one sequential file read; no download)", flush=True)
            backbone_hashes = {name: file_sha256(snapshot / name) for name in
                               ("model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json")}
            if model_hashes["policy_preprocessor.json"] != pairing_identity["checkpoint"]["preprocessor_sha256"]:
                raise RuntimeError("Checkpoint preprocessor changed after pairing validation")
            suite = benchmark.get_benchmark_dict()[config["suite"]]()
            task = suite.get_task(config["task_id"])
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            init_file = Path(get_libero_path("init_states")) / task.problem_folder / Path(task.init_states_file).name
            if file_sha256(bddl) != pairing_identity["bddl_sha256"] or file_sha256(init_file) != pairing_identity["init_file_sha256"]:
                raise RuntimeError("Task BDDL/initial-state assets changed after pairing validation")
            policy_config = SmolVLAConfig.from_pretrained(path)
            policy_config.device = "cuda"
            policy_config.push_to_hub = False
            for name, value in config["policy"].items():
                setattr(policy_config, name, value)
            policy_config.vlm_model_name = str(snapshot)
            if policy_config.use_amp or policy_config.n_action_steps != 1:
                raise RuntimeError("Pilot must retain the approved first-rollout inference settings")
            upstream_modules = (
                "lerobot.scripts.lerobot_eval", "lerobot.policies.smolvla.modeling_smolvla",
                "lerobot.policies.smolvla.smolvlm_with_expert", "lerobot.processor.tokenizer_processor",
                "lerobot.utils.random_utils", "lerobot.utils.io_utils",
            )
            upstream_hashes = {name: file_sha256(importlib.import_module(name).__file__) for name in upstream_modules}
            sources = {name: file_sha256(ROOT / name) for name in (
                "scripts/run_language_pilot.py", "scripts/run_language_pilot.sh",
                "src/smolvla_grounding/paired_env.py", "src/smolvla_grounding/rollout_records.py",
                "src/smolvla_grounding/protocol.py", "src/smolvla_grounding/libero_pairing.py",
            )}
            gpu_driver = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                        check=True, capture_output=True, text=True).stdout.strip().splitlines()
            identity = {"config": config, "run_config": run_config, "instructions": rows,
                        "pairing_fingerprint": proof["fingerprint"], "checkpoint": checkpoint,
                        "policy_hashes": model_hashes, "backbone": backbone, "backbone_hashes": backbone_hashes,
                        "project_sources": sources, "upstream_sources": upstream_hashes,
                        "packages": {**versions, "av": av.__version__, "torchvision": importlib.metadata.version("torchvision")},
                        "python": platform.python_version(), "kernel": platform.release(),
                        "gpu_driver": gpu_driver, "cuda": torch.version.cuda,
                        "backend": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}}
            checkpoints = Checkpoints(directory, identity)
            saved = {r["id"]: checkpoints.completed(r["id"]) for r in rows}
            pending = [r for r in rows if saved[r["id"]] is None]
            execution["fingerprint"] = checkpoints.fingerprint
            execution["new_trials"] = len(pending)

            if pending:
                print("[3/5] Load one official policy with strict weight matching and checkpoint processors", flush=True)
                set_seed(config["policy_seed"])
                policy = SmolVLAPolicy.from_pretrained(path, config=policy_config, strict=True)
                policy.eval()
                preprocessor, postprocessor = make_pre_post_processors(
                    policy_cfg=policy_config, pretrained_path=str(path), preprocessor_overrides={
                        "device_processor": {"device": "cuda"},
                        "tokenizer_processor": {"tokenizer_name": str(snapshot)},
                    })
                env_config = LiberoConfig(task=config["suite"], task_ids=[config["task_id"]],
                                         fps=config["environment"]["control_freq"],
                                         observation_height=config["environment"]["observation_height"],
                                         observation_width=config["environment"]["observation_width"])
                env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_config, policy_config)
                dtypes = Counter()
                for parameter in policy.parameters():
                    dtypes[str(parameter.dtype)] += parameter.numel()
                atomic_json(directory / "model_load.json", {"strict_checkpoint_load": True,
                            "parameter_counts_by_dtype": dict(dtypes), "settings": config["policy"],
                            "backbone": backbone})
            else:
                print("[3/5] All trials already complete; skip model loading", flush=True)

            print("[4/5] Execute serial paired trials with a common 280-step horizon", flush=True)
            results = []
            reference_rng = None
            for index, row in enumerate(rows, 1):
                result = saved[row["id"]]
                if result is None:
                    print(f"  [{index}/{len(rows)}] {row['id']}: {row['text']}", flush=True)
                    attempt = checkpoints.new_attempt(row["id"])
                    env = vec = None
                    frames = {"image": [], "image2": []}
                    initial_rng = {}
                    torch.cuda.reset_peak_memory_stats()
                    trial_started = time.perf_counter()
                    try:
                        for processor in (preprocessor, postprocessor, env_preprocessor, env_postprocessor):
                            processor.reset()
                        set_seed(config["environment_seed"])
                        with (attempt / "steps.jsonl").open("w", encoding="utf-8", buffering=1) as step_stream:
                            env = PairedLiberoEnv(suite, config, rows[0], row, proof["checks"][0]["hashes"], attempt, step_stream)
                            vec = gym.vector.SyncVectorEnv([lambda: env])

                            def render_callback(_vec):
                                for camera in frames:
                                    frames[camera].append(env.latest_pixels[camera][::-1, ::-1].copy())
                                if len(frames["image"]) == 1:
                                    # Seed after physics reset, immediately before the first prediction.
                                    set_seed(config["policy_seed"])
                                    rng = {k: v.cpu().numpy().copy() for k, v in serialize_rng_state().items()}
                                    initial_rng.update(array_hashes(rng))
                                    np.savez_compressed(attempt / "initial_rng.npz", **rng)
                                    if reference_rng is not None:
                                        require_matching_state(reference_rng, initial_rng)

                            def checked_preprocessor(observation):
                                if observation["task"] != [row["text"]]:
                                    raise RuntimeError("Official rollout did not receive the requested instruction")
                                batch = preprocessor(observation)
                                tokens = batch[OBS_LANGUAGE_TOKENS][0]
                                mask = batch[OBS_LANGUAGE_ATTENTION_MASK][0].bool()
                                if tokens[mask].cpu().tolist() != token_ids[row["id"]]:
                                    raise RuntimeError("Policy input tokens differ from the validated instruction")
                                return batch

                            rollout_started = time.perf_counter()
                            tensors = rollout(vec, policy, env_preprocessor, env_postprocessor,
                                              checked_preprocessor, postprocessor,
                                              seeds=[config["environment_seed"]], render_callback=render_callback)
                            torch.cuda.synchronize()
                            rollout_seconds = time.perf_counter() - rollout_started
                            steps = config["environment"]["episode_length"]
                            if len(env.records) != steps or any(len(f) != steps + 1 for f in frames.values()):
                                raise RuntimeError("Incomplete fixed-horizon actions or video frames")
                            arrays = {k: v.cpu().numpy()[0] for k, v in tensors.items()}
                            actions = np.asarray([r["action"] for r in env.records], dtype=arrays["action"].dtype)
                            expected_success = np.asarray([r["goals"][row["goal_id"]] for r in env.records])
                            if not np.array_equal(arrays["action"], actions) or not np.array_equal(arrays["success"], expected_success):
                                raise RuntimeError("Official rollout tensors disagree with the step log")
                            arrays["sim_state"] = np.stack(env.sim_states)
                            for goal_id in config["goals"]:
                                arrays[f"goal_{goal_id}"] = np.asarray([r["goals"][goal_id] for r in env.records])
                            np.savez_compressed(attempt / "trajectory.npz", **arrays)
                            video_info = {}
                            for camera, sequence in frames.items():
                                video_path = attempt / f"{camera}.mp4"
                                write_video(video_path, sequence, fps=config["environment"]["control_freq"])
                                with av.open(str(video_path)) as video:
                                    fps = float(video.streams.video[0].average_rate)
                                    count = sum(1 for _ in video.decode(video=0))
                                if count != steps + 1 or fps != config["environment"]["control_freq"]:
                                    raise RuntimeError("Encoded video frame count or FPS is incorrect")
                                video_info[camera] = {"frames": count, "fps": fps}
                            result = {"instruction_id": row["id"], "condition": row["condition"],
                                      "instruction": row["text"], "goal_id": row["goal_id"], "policy_steps": steps,
                                      "environment_seed": config["environment_seed"], "policy_seed": config["policy_seed"],
                                      "initial_state_index": config["init_state_index"], "initial_hashes": env.initial_hashes,
                                      "initial_rng_hashes": initial_rng,
                                      "goals": summarize_goals([r["goals"] for r in env.records], config["goals"]),
                                      "rollout_seconds": rollout_seconds, "trial_seconds": time.perf_counter() - trial_started,
                                      "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                                      "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                      "trajectory": (attempt / "trajectory.npz").relative_to(directory).as_posix(),
                                      "video_main": (attempt / "image.mp4").relative_to(directory).as_posix(),
                                      "video_wrist": (attempt / "image2.mp4").relative_to(directory).as_posix(),
                                      "video_info": video_info,
                                      "alternative_goal_validation": "provisional; positive cases require video review"}
                        vec.close()
                        vec = env = None
                        atomic_json(attempt / "episode.json", result)
                        artifacts = [attempt / name for name in ("initial_state.npz", "initial_hashes.json", "initial_rng.npz",
                                     "steps.jsonl", "trajectory.npz", "image.mp4", "image2.mp4", "episode.json")]
                        checkpoints.finish(row["id"], result, artifacts)
                        print(f"  Saved {row['id']}: requested goal {result['goals'][row['goal_id']]}", flush=True)
                    except BaseException as exc:
                        atomic_json(attempt / "error.json", {"type": type(exc).__name__, "message": str(exc)})
                        raise
                    finally:
                        if vec is not None:
                            vec.close()
                        elif env is not None:
                            env.close()
                else:
                    print(f"  [{index}/{len(rows)}] {row['id']}: verified completed artifacts", flush=True)
                if reference_rng is None:
                    reference_rng = result["initial_rng_hashes"]
                require_matching_state(reference_rng, result["initial_rng_hashes"])
                results.append(result)
                export_episode_table(directory, results, len(rows))

            print("[5/5] Compare the repeated original and export episode CSV/JSON", flush=True)
            original = next(r for r in results if r["condition"] == "original")
            repeat = next(r for r in results if r["condition"] == "original_repeat")
            with np.load(directory / original["trajectory"], allow_pickle=False) as a, np.load(directory / repeat["trajectory"], allow_pickle=False) as b:
                repeat_control = {"actions_bitwise_equal": bool(np.array_equal(a["action"], b["action"])),
                                  "max_absolute_action_difference": float(np.abs(a["action"] - b["action"]).max()),
                                  "sim_states_bitwise_equal": bool(np.array_equal(a["sim_state"], b["sim_state"])),
                                  "goal_timelines_equal": all(np.array_equal(a[f"goal_{g}"], b[f"goal_{g}"]) for g in config["goals"])}
            export_episode_table(directory, results, len(rows), repeat_control)
            execution["passed"] = True
            execution["repeat_control"] = repeat_control
            print("PASS: all six fixed-horizon trials completed; summary.csv and summary.json saved.", flush=True)
            print("Repeat control: " + json.dumps(repeat_control), flush=True)
            print("Review videos before accepting alternative-goal results. This is a one-state pilot.", flush=True)
        except BaseException as exc:
            execution["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            execution["elapsed_seconds"] = time.perf_counter() - started
            execution["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            atomic_json(directory / f"execution_{stamp}.json", execution)


if __name__ == "__main__":
    main()
