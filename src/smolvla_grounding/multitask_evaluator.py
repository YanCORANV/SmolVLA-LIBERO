"""Reusable LIBERO evaluator for the preregistered canonical competence screen."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np

from .protocol import atomic_json, file_sha256, run_lock
from .multitask_protocol import _valid_completion, new_attempt, preflight, trial_path

SCREEN_MATRIX = "configs/multitask_evaluation/canonical_screen_matrix.json"
SCREEN_MANIFEST = "configs/multitask_evaluation/canonical_screen_manifest.json"


def _atomic_jsonl(path, rows):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_npz(path, **arrays):
    temporary = path.with_name(path.stem + ".partial" + path.suffix)
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _batch_obs(value):
    if isinstance(value, dict):
        return {key: _batch_obs(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return np.expand_dims(value, axis=0)
    return [value]


def _load_policy(root, plan, task, model_id, prompt_text):
    """Load the official policy using the same LeRobot processors as Task-0."""
    from lerobot.envs.configs import LiberoEnv as LiberoConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.utils.random_utils import set_seed
    from transformers import AutoTokenizer

    matrix, manifest = plan["matrix"], plan["manifest"]
    model_info = manifest["models"][model_id]
    path = root / model_info["path"]
    snapshot = plan["snapshot"]
    config = SmolVLAConfig.from_pretrained(path)
    config.device, config.push_to_hub = "cuda", False
    for key in ("n_action_steps", "num_steps", "use_amp"):
        setattr(config, key, matrix["policy"][key])
    config.vlm_model_name = str(snapshot)
    set_seed(matrix["policy"]["policy_seed"])
    policy = SmolVLAPolicy.from_pretrained(path, config=config, strict=True).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=config, pretrained_path=str(path),
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "tokenizer_processor": {"tokenizer_name": str(snapshot)},
        },
    )
    env_cfg = matrix["environment"]
    libero_cfg = LiberoConfig(
        task=task["suite"], task_ids=[task["task_id"]],
        fps=env_cfg["control_freq"],
        observation_height=env_cfg["observation_height"],
        observation_width=env_cfg["observation_width"],
    )
    env_pre, env_post = make_env_pre_post_processors(libero_cfg, config)
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    token_ids = tokenizer(prompt_text + "\n", truncation=False)["input_ids"]
    if len(token_ids) > config.tokenizer_max_length:
        raise RuntimeError("Instruction would be truncated")
    return policy, pre, post, env_pre, env_post, token_ids


def _execute_episode(root, plan, row, attempt):
    import torch
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION
    from lerobot.utils.io_utils import write_video
    from lerobot.utils.random_utils import get_rng_state, set_seed

    from .diagnostic_policy import capture_noise, require_empty_queue
    from .diagnostic_records import pack_rng, pack_tensors
    from .libero_pairing import array_hashes, capture_initial_state, flatten_observation

    model, suite_name, task_id, state, prompt_id = row
    task = next(t for t in plan["matrix"]["tasks"]
                if (t["suite"], t["task_id"]) == (suite_name, task_id))
    env_cfg, pol_cfg = plan["matrix"]["environment"], plan["matrix"]["policy"]
    horizon = task["horizon"]
    prompt = next((p for p in plan["catalogs"][(suite_name, task_id)]
                   if p["id"] == prompt_id), None)
    if prompt is None:
        raise RuntimeError("Instruction ID absent from frozen catalog")
    set_seed(env_cfg["environment_seed"])
    suite = benchmark.get_benchmark_dict()[suite_name]()
    env = LiberoEnv(
        task_suite=suite, task_id=task_id, task_suite_name=suite_name,
        init_states=True, episode_index=state, n_envs=1,
        obs_type=env_cfg["obs_type"],
        observation_height=env_cfg["observation_height"],
        observation_width=env_cfg["observation_width"],
        control_freq=env_cfg["control_freq"], control_mode=env_cfg["control_mode"],
        hard_reset=env_cfg["hard_reset"], num_steps_wait=env_cfg["num_steps_wait"],
        episode_length=horizon,
    )
    start = time.monotonic()
    try:
        obs, _ = env.reset(seed=env_cfg["environment_seed"])
        initial = capture_initial_state(env, obs, state)
        atomic_json(attempt / "initial_hashes.json", array_hashes(initial))
        _atomic_npz(attempt / "initial_state.npz", **initial)
        policy, pre, post, env_pre, env_post, token_ids = _load_policy(root, plan, task, model, prompt["text"])
        policy.reset()
        set_seed(pol_cfg["policy_seed"])

        actions, rewards, successes, sim_states = [], [], [], []
        eef_before, eef_after = [], []
        steps, inference = [], []
        main_frames, wrist_frames = [], []
        goal_ever, goal_final, first_step = False, False, None
        current = obs
        for step in range(1, horizon + 1):
            pixels = current.get("pixels", {})
            if "image" in pixels:
                main_frames.append(pixels["image"])
            if "image2" in pixels:
                wrist_frames.append(pixels["image2"])
            before = np.asarray(current["robot_state"]["eef"]["pos"], dtype=np.float32).copy()
            observation = preprocess_observation(_batch_obs(current))
            observation["task"] = [prompt["text"]]
            batch = pre(env_pre(observation))
            tokens = batch["observation.language.tokens"][0]
            mask = batch["observation.language.attention_mask"][0].bool()
            if tokens[mask].cpu().tolist() != token_ids:
                raise RuntimeError("Policy tokenizer changed the canonical instruction")
            require_empty_queue(policy)
            rng_before = pack_rng(get_rng_state(), "rng_before", {}, {})
            with capture_noise(policy) as sampled_noise:
                action = policy.select_action(batch)
            if len(sampled_noise) != 1:
                raise RuntimeError("Expected exactly one flow-noise sample")
            require_empty_queue(policy)
            rng_after = pack_rng(get_rng_state(), "rng_after", {}, {})
            actual_action = env_post({ACTION: post(action)})[ACTION]
            action_np = actual_action.squeeze(0).detach().cpu().numpy()
            inference.append({
                "step": step, "raw_pixels": array_hashes(pixels),
                "rng_before": rng_before, "rng_after": rng_after,
                "noise": pack_tensors({"noise": sampled_noise[0]}, "noise", {}, {}),
                "normalized_action": pack_tensors({"action": action}, "output", {}, {}),
            })
            following, reward, _, _, _ = env.step(action_np)
            success = bool(env._env.check_success())
            if success and not goal_ever:
                goal_ever, first_step = True, step
            goal_final = success
            after = np.asarray(following["robot_state"]["eef"]["pos"], dtype=np.float32).copy()
            steps.append({
                "step": step, "instruction_id": prompt_id,
                "action": action_np.tolist(), "reward": float(reward),
                "goal": success, "eef_before": before.tolist(), "eef_after": after.tolist(),
                "robot_state_before": {k: v.tolist() for k, v in flatten_observation(current["robot_state"], "robot_state").items()},
                "robot_state_after": {k: v.tolist() for k, v in flatten_observation(following["robot_state"], "robot_state").items()},
            })
            actions.append(action_np)
            rewards.append(float(reward))
            successes.append(success)
            sim_states.append(env._env.get_sim_state())
            eef_before.append(before)
            eef_after.append(after)
            current = following

        if plan["matrix"]["execution"]["videos"]:
            for name, frames in (("image", main_frames), ("image2", wrist_frames)):
                if frames:
                    temporary = attempt / f"{name}.partial.mp4"
                    write_video(str(temporary), np.stack(frames), fps=env_cfg["control_freq"])
                    temporary.replace(attempt / f"{name}.mp4")
        _atomic_npz(
            attempt / "trajectory.npz", action=np.asarray(actions),
            reward=np.asarray(rewards), success=np.asarray(successes),
            sim_state=np.asarray(sim_states), eef_before=np.asarray(eef_before),
            eef_after=np.asarray(eef_after),
        )
        _atomic_jsonl(attempt / "steps.jsonl", steps)
        _atomic_jsonl(attempt / "inference.jsonl", inference)
        duration = time.monotonic() - start
        episode = {
            "model": model, "suite": suite_name, "task_id": task_id,
            "state": state, "prompt_id": prompt_id, "instruction": prompt["text"],
            "policy_steps": horizon, "goal_ever": goal_ever, "goal_final": goal_final,
            "first_step": first_step, "duration_seconds": duration,
            "screening_only": plan["matrix"]["role"] == "canonical_competence_screen",
        }
        atomic_json(attempt / "episode.json", episode)
        artifacts = {
            path.name: file_sha256(path) for path in attempt.iterdir()
            if path.is_file() and path.name != "complete.json"
        }
        atomic_json(attempt / "complete.json", {
            **episode, "status": "COMPLETED", "identity": plan["identity"],
            "artifacts": artifacts,
        })
    finally:
        env.close()
        if "policy" in locals():
            del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _screen_report(plan):
    matrix = plan["matrix"]
    output = plan["output_dir"]
    per_task = []
    total_runtime = 0.0
    for task in matrix["tasks"]:
        success_count = 0
        durations = []
        for state in task["states"]:
            row = ("untuned_fp32", task["suite"], task["task_id"], state, "original")
            trial = trial_path(output, row, matrix["policy"]["policy_seed"])
            committed = _valid_completion(trial, row, task["horizon"], plan["identity"])
            if committed is None:
                raise RuntimeError(f"Incomplete screen: {trial}")
            record = json.loads((committed / "complete.json").read_text(encoding="utf-8"))
            success_count += int(record["goal_ever"])
            durations.append(record["duration_seconds"])
        total_runtime += sum(durations)
        per_task.append({
            "suite": task["suite"], "task_id": task["task_id"],
            "task_name": task["task_name"], "canonical": task["canonical"],
            "states": task["states"], "successes": success_count, "trials": 5,
            "classification": ("eligible_for_language_robustness" if success_count >= 3
                               else "insufficient_canonical_competence"),
            "duration_seconds": durations,
        })
    return {
        "role": "canonical_competence_screen", "status": "COMPLETED",
        "scientific_evidence": False, "completed_rollouts": 30,
        "eligibility_rule": matrix["eligibility"], "per_task": per_task,
        "total_rollout_seconds": total_runtime,
        "proposed_scientific_rollouts": sum(
            row["classification"] == "eligible_for_language_robustness"
            for row in per_task) * 5 * 3,
    }


def run_canonical_screen(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    print("[Screen preflight] validating the complete 30-rollout matrix", flush=True)
    plan = preflight(root, SCREEN_MATRIX, SCREEN_MANIFEST)
    print("[Screen preflight] passed; no environment has been reset", flush=True)
    output = plan["output_dir"]
    with run_lock(output):
        metadata = output / "experiment_meta.json"
        if not metadata.exists():
            atomic_json(metadata, {
                "identity": plan["identity"], "matrix": plan["matrix"],
                "manifest_sha256": file_sha256(root / SCREEN_MANIFEST),
                "matrix_sha256": file_sha256(root / SCREEN_MATRIX),
            })
        tasks = {(t["suite"], t["task_id"]): t for t in plan["matrix"]["tasks"]}
        for ordinal, row in enumerate(plan["rows"], 1):
            task = tasks[row[1:3]]
            trial = trial_path(output, row, plan["matrix"]["policy"]["policy_seed"])
            if _valid_completion(trial, row, task["horizon"], plan["identity"]):
                print(f"[{ordinal:02d}/30] cached {row[1]}/{row[2]} state={row[3]}", flush=True)
                continue
            attempt = new_attempt(trial)
            print(f"[{ordinal:02d}/30] running {row[1]}/{row[2]} state={row[3]} {attempt.name}", flush=True)
            _execute_episode(root, plan, row, attempt)
            if _valid_completion(trial, row, task["horizon"], plan["identity"]) != attempt:
                raise RuntimeError(f"Rollout did not commit: {attempt}")
            print(f"[{ordinal:02d}/30] completed {row[1]}/{row[2]} state={row[3]}", flush=True)
        report = _screen_report(plan)
        atomic_json(output / "canonical_screen_report.json", report)
        print("[Screen] 30/30 complete; eligibility report committed", flush=True)
        return report



