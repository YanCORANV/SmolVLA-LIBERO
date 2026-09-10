"""Validate task-0 instruction tokens, predicates and paired resets before rollouts.

Run manually in the project WSL environment. Imports and renderer startup may be slow
on /mnt/d. No policy weights are loaded and no learned actions are executed here.
"""

from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.protocol import (  # noqa: E402
    Checkpoints, atomic_json, file_sha256, load_protocol, run_lock,
)


def main():
    if platform.system() != "Linux":
        raise RuntimeError("Run in WSL/Linux with the validated project .venv")
    if os.environ.get("MUJOCO_GL", "egl") != "egl":
        raise RuntimeError("Use MUJOCO_GL=egl, as in the validated first rollout")
    os.environ["MUJOCO_GL"] = "egl"
    config, rows = load_protocol(ROOT)
    directory = ROOT / config["pairing_output"]
    with run_lock(directory):
        started = time.perf_counter()
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        report = {"started_utc": stamp, "passed": False, "policy_rollout": False,
                  "scope": "Token checks, initial predicate checks and paired reset identity only"}
        try:
            print("[1/4] Import the official environment and tokenizer (cold imports can take minutes)", flush=True)
            import numpy as np
            import torch
            from PIL import Image
            from huggingface_hub import try_to_load_from_cache
            from transformers import AutoTokenizer
            from libero.libero import benchmark
            from lerobot.envs.libero import LiberoEnv
            from lerobot.utils.random_utils import set_seed
            from smolvla_grounding.libero_pairing import (
                array_hashes, capture_initial_state, check_task, goal_values, require_matching_state,
            )

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; use the environment that passed first-rollout validation")
            checkpoint = json.loads((ROOT / "configs/checkpoint.json").read_text(encoding="utf-8"))
            model_directory = ROOT / checkpoint["local_dir"]
            processor = json.loads((model_directory / "policy_preprocessor.json").read_text(encoding="utf-8"))
            if "smolvla_new_line_processor" not in {s["registry_name"] for s in processor["steps"]}:
                raise RuntimeError("Checkpoint no longer uses the inspected SmolVLA newline processor")
            tokenizer_config = next(s["config"] for s in processor["steps"]
                                    if s["registry_name"] == "tokenizer_processor")
            tokenizer_id = tokenizer_config["tokenizer_name"]
            cached = try_to_load_from_cache(tokenizer_id, "config.json")
            if not isinstance(cached, str):
                raise RuntimeError("Previously downloaded backbone is missing from the HF cache. "
                                   "Use the same WSL user/cache as check_policy.py.")
            snapshot = Path(cached).parent
            if snapshot.parent.name != "snapshots" or not re.fullmatch(r"[0-9a-f]{40}", snapshot.name):
                raise RuntimeError("Cannot identify an immutable cached backbone revision")
            tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
            package_names = ("lerobot", "hf-libero", "torch", "transformers", "mujoco", "robosuite", "numpy", "gymnasium")
            package_versions = {p: importlib.metadata.version(p) for p in package_names}

            print("[2/4] Check all exact instructions for token truncation and token collisions", flush=True)
            tokens = {}
            seen = {}
            for row in rows:
                # The official SmolVLA processor adds a final newline before tokenization.
                ids = tokenizer(row["text"] + "\n", truncation=False)["input_ids"]
                actual_ids = tokenizer(row["text"] + "\n", truncation=tokenizer_config["truncation"],
                                       max_length=tokenizer_config["max_length"])["input_ids"]
                if ids != actual_ids or len(ids) > tokenizer_config["max_length"]:
                    raise RuntimeError(f"Instruction would be truncated: {row['id']}")
                key = tuple(ids)
                if key in seen and row["condition"] != "original_repeat":
                    raise RuntimeError(f"Distinct instructions have identical tokens: {row['id']}")
                seen[key] = row["id"]
                tokens[row["id"]] = {"count": len(ids), "ids": ids}
                print(f"  {row['id']}: {len(ids)}/{tokenizer_config['max_length']} tokens", flush=True)

            suite = benchmark.get_benchmark_dict()[config["suite"]]()
            # Only inspect specific protocol/dependency files, not the filesystem.
            import lerobot.envs.libero as upstream_env
            import libero.libero.envs.env_wrapper as upstream_wrapper
            import libero.libero.envs.predicates.base_predicates as upstream_predicates
            import libero.libero.envs.object_states.base_object_states as upstream_states
            source_hashes = {name: file_sha256(ROOT / name) for name in (
                "scripts/check_language_pairing.py", "src/smolvla_grounding/protocol.py",
                "src/smolvla_grounding/libero_pairing.py", "configs/checkpoint.json",
                "configs/language_pilot.json", config["instruction_catalog"],
            )}
            upstream_hashes = {module.__name__: file_sha256(module.__file__) for module in
                               (upstream_env, upstream_wrapper, upstream_predicates, upstream_states)}
            from libero.libero import get_libero_path
            task = suite.get_task(config["task_id"])
            bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
            init_file = Path(get_libero_path("init_states")) / task.problem_folder / Path(task.init_states_file).name
            identity = {
                "config": config, "instructions": rows, "packages": package_versions,
                "checkpoint": {"repo_id": checkpoint["repo_id"], "revision": checkpoint["revision"],
                               "preprocessor_sha256": file_sha256(model_directory / "policy_preprocessor.json")},
                "backbone": {"repo_id": tokenizer_id, "revision": snapshot.name,
                             "tokenizer_sha256": file_sha256(snapshot / "tokenizer.json"),
                             "tokenizer_config_sha256": file_sha256(snapshot / "tokenizer_config.json")},
                "bddl_sha256": file_sha256(bddl), "init_file_sha256": file_sha256(init_file),
                "project_sources": source_hashes, "upstream_sources": upstream_hashes,
                "python": platform.python_version(), "kernel": platform.release(),
                "gpu": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda,
            }
            checkpoints = Checkpoints(directory, identity)
            atomic_json(directory / "tokens.json", tokens)

            print("[3/4] Reset a fresh official environment for each condition; compare states and both cameras", flush=True)
            reference = None
            results = []
            for index, row in enumerate(rows, 1):
                saved = checkpoints.completed(row["id"])
                if saved is not None:
                    print(f"  [{index}/{len(rows)}] {row['id']}: verified saved check", flush=True)
                    result = saved
                else:
                    print(f"  [{index}/{len(rows)}] {row['id']}: construct, restore state "
                          f"{config['init_state_index']}, settle, capture", flush=True)
                    attempt = checkpoints.new_attempt(row["id"])
                    env = None
                    try:
                        set_seed(config["environment_seed"])
                        env = LiberoEnv(task_suite=suite, task_id=config["task_id"],
                                        task_suite_name=config["suite"], init_states=True,
                                        episode_index=config["init_state_index"], n_envs=1,
                                        **config["environment"])
                        check_task(env, config, rows[0])
                        # This is the field consumed by the official rollout's language input.
                        env.task_description = row["text"]
                        observation, _ = env.reset(seed=config["environment_seed"])
                        parsed = env._env.env.parsed_problem["goal_state"]
                        if [list(p) for p in parsed] != [config["goals"]["original_bowl"]]:
                            raise RuntimeError("Native task goal differs from configured original predicate")
                        arrays = capture_initial_state(env, observation, config["init_state_index"])
                        hashes = array_hashes(arrays)
                        values = goal_values(env, config["goals"])
                        if any(values.values()):
                            raise RuntimeError("A requested goal is already true at the initial state")
                        np.savez_compressed(attempt / "initial_state.npz", **arrays)
                        artifacts = [attempt / "initial_state.npz"]
                        for camera, frame in observation["pixels"].items():
                            path = attempt / f"{camera}.png"
                            Image.fromarray(frame[::-1, ::-1]).save(path)
                            artifacts.append(path)
                        result = {"instruction_id": row["id"], "instruction": env.task_description,
                                  "goal_id": row["goal_id"], "initial_goals": values,
                                  "initial_state_index": config["init_state_index"],
                                  "hashes": hashes, "token_count": tokens[row["id"]]["count"]}
                        atomic_json(attempt / "inspection.json", result)
                        artifacts.append(attempt / "inspection.json")
                        if reference is not None:
                            require_matching_state(reference, hashes)
                        # Publish the completion marker only after the renderer also closes.
                        env.close()
                        env = None
                        checkpoints.finish(row["id"], result, artifacts)
                    except BaseException as exc:
                        atomic_json(attempt / "error.json", {"type": type(exc).__name__, "message": str(exc)})
                        raise
                    finally:
                        if env is not None:
                            env.close()
                if reference is None:
                    reference = result["hashes"]
                else:
                    require_matching_state(reference, result["hashes"])
                results.append(result)

            print("[4/4] Save pairing evidence and remaining acceptance requirements", flush=True)
            report.update({"passed": True, "fingerprint": checkpoints.fingerprint,
                           "conditions_checked": len(results), "checks": results,
                           "backbone": identity["backbone"],
                           "limitations": [
                               "No policy inference or policy-RNG pairing has been validated by this check.",
                               "Exact captured reset arrays match; this does not prove full hidden-state or trajectory determinism.",
                               "Alternative predicates were called only on initial negative states; positive cases remain unvalidated.",
                               "Review camera images for referent ambiguity and alternative-goal feasibility before scoring rollouts.",
                           ]})
            print(f"PASS: {len(results)} conditions have identical captured initial states and observations.", flush=True)
            print("Paired policy rollouts and positive alternative-goal checks are still pending.", flush=True)
        except BaseException as exc:
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            report["elapsed_seconds"] = time.perf_counter() - started
            output = directory / f"pairing_validation_{stamp}.json"
            atomic_json(output, report)
            print("Saved " + output.relative_to(ROOT).as_posix(), flush=True)


if __name__ == "__main__":
    main()
