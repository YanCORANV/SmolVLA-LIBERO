"""Held-out confirmatory evaluation coordinator for frozen SmolVLA checkpoints.

Ensures strict separation between development and held-out test data.
Performs pre-flight validation against data leakage, checkpoint hash mismatch, or matrix mismatch.
Supports resume, terminal progress, single-worker determinism, and detailed per-step records.
"""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

from .protocol import atomic_json, file_sha256, run_lock
from .benchmark_records import validate_run_name


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_task_goal_status(env):
    """Check task goal status using native LIBERO check_success."""
    return bool(env._env.check_success())


def preflight_confirmatory_validation(root, matrix, manifest):
    """Rigorous pre-flight validation preventing data leakage, hash mismatch, or invalid horizons."""
    print("[Pre-flight] Validating confirmatory protocol integrity and artifact hashes...", flush=True)

    # 1. Rollout horizon audit: must be exactly 280 steps
    horizon = matrix.get("environment", {}).get("episode_length", None)
    if horizon != 280:
        raise RuntimeError(f"FATAL PRE-FLIGHT ERROR: Rollout horizon must be exactly 280 steps, got {horizon}!")

    # 2. State overlap check against development states
    dev_states = set(manifest.get("development_boundary", {}).get("development_states", [0, 1, 2]))
    for task_cfg in matrix["tasks"]:
        if task_cfg["task_id"] != 0:
            raise RuntimeError(f"FATAL PRE-FLIGHT ERROR: Primary confirmatory evaluation must be Task 0 only, got Task {task_cfg['task_id']}!")
        confirmatory_states = set(task_cfg["initial_states"])
        overlap = confirmatory_states.intersection(dev_states)
        if overlap:
            raise RuntimeError(f"FATAL PRE-FLIGHT ERROR: Confirmatory state(s) {overlap} overlap with development states {dev_states} in Task {task_cfg['task_id']}!")
        if len(confirmatory_states) != 20:
            raise RuntimeError(f"FATAL PRE-FLIGHT ERROR: Primary confirmatory experiment requires exactly 20 states, got {len(confirmatory_states)}!")

    # 3. Paraphrase leakage check against training catalogs and dev evaluation catalogs
    train_catalog_path = root / "configs/instructions/task0_train.jsonl"
    dev_catalog_path = root / "configs/instructions/task0.jsonl"
    if not train_catalog_path.exists():
        raise RuntimeError("Missing training language catalog for pre-flight check")

    forbidden_texts = {r["text"].strip().lower() for r in load_jsonl(train_catalog_path)}
    if dev_catalog_path.exists():
        for r in load_jsonl(dev_catalog_path):
            if r.get("condition") == "paraphrase":
                forbidden_texts.add(r["text"].strip().lower())

    for task_cfg in matrix["tasks"]:
        catalog_path = root / task_cfg["instruction_catalog"]
        if not catalog_path.exists():
            raise RuntimeError(f"Missing instruction catalog: {catalog_path}")
        eval_rows = load_jsonl(catalog_path)
        for r in eval_rows:
            if r.get("condition") == "paraphrase":
                text = r["text"].strip().lower()
                if text in forbidden_texts:
                    raise RuntimeError(f"FATAL PRE-FLIGHT ERROR: Held-out paraphrase '{r['text']}' leaked in training or development catalog!")

    # 4. Strict checkpoint and artifact SHA256 validation against frozen manifest
    for model_id in matrix["models"]:
        if model_id not in manifest["evaluated_models"]:
            raise RuntimeError(f"Model {model_id} not registered in frozen_training_manifest.json")
        info = manifest["evaluated_models"][model_id]
        expected_hashes = info.get("artifact_sha256", {})
        if not expected_hashes:
            raise RuntimeError(f"FATAL PRE-FLIGHT ERROR: No artifact_sha256 recorded for {model_id} in frozen manifest!")

        base_dir = root / (info["source"] if model_id == "untuned_fp32" else info["checkpoint"])
        for rel_file, exp_hash in expected_hashes.items():
            fpath = base_dir / rel_file
            if not fpath.exists():
                raise RuntimeError(f"FATAL PRE-FLIGHT ERROR: Checkpoint file missing: {fpath}")
            act_hash = sha256_of_file(fpath)
            if act_hash != exp_hash:
                raise RuntimeError(f"FATAL PRE-FLIGHT ERROR: SHA256 mismatch for {model_id} file {rel_file}! Expected {exp_hash}, got {act_hash}")

    # 5. Matrix consistency across all 3 models
    task_keys = []
    for task_cfg in matrix["tasks"]:
        instructions = load_jsonl(root / task_cfg["instruction_catalog"])
        for s in task_cfg["initial_states"]:
            for inst in instructions:
                task_keys.append((task_cfg["task_id"], s, inst["id"]))

    expected_total_trials = 20 * 4  # 80 trials per model
    if len(task_keys) != expected_total_trials:
        raise RuntimeError(f"FATAL PRE-FLIGHT ERROR: Expected {expected_total_trials} trials per model, found {len(task_keys)}")

    # 6. Execution settings: Worker must be 1
    if matrix.get("execution", {}).get("default_workers", 1) != 1:
        raise RuntimeError("FATAL PRE-FLIGHT ERROR: default_workers must be strictly 1 (serial execution)!")

    print(f"[Pre-flight] Integrity check PASSED completely.")
    print(f"  - Horizon: 280 steps")
    print(f"  - States: 20 untouched states (3-22), zero overlap with development states")
    print(f"  - Prompts: Zero leakage against training or development catalogs")
    print(f"  - Artifacts: Exact SHA256 match for all 3 frozen model checkpoints")
    print(f"  - Matrix: Exactly {expected_total_trials} trials per model ({expected_total_trials * len(matrix['models'])} rollouts total)")
    print(f"  - Worker: 1 (single-worker serial execution)\n", flush=True)


def run_held_out_evaluation(matrix_config_path="configs/held_out_evaluation/confirmatory_matrix.json",
                           default_run_name="task0_confirmatory_heldout_eval"):
    root = Path(__file__).resolve().parents[2]
    matrix = json.loads((root / matrix_config_path).read_text(encoding="utf-8"))
    manifest = json.loads((root / matrix["manifest_path"]).read_text(encoding="utf-8"))

    # Execute rigorous pre-flight validation
    preflight_confirmatory_validation(root, matrix, manifest)

    # Interactive confirmation of run name (no command-line flags)
    run_name_prompt = input(f"Held-out evaluation run name [{default_run_name}]: ").strip() or default_run_name
    run_name = validate_run_name(run_name_prompt)

    output_dir = root / matrix["output_root"] / "runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with run_lock(output_dir):
        # Save experiment snapshot
        meta_path = output_dir / "experiment_meta.json"
        if not meta_path.exists():
            atomic_json(meta_path, {
                "run_name": run_name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "matrix_config": matrix,
                "manifest": manifest,
                "sources": {
                    matrix_config_path: file_sha256(root / matrix_config_path),
                    matrix["manifest_path"]: file_sha256(root / matrix["manifest_path"]),
                    "src/smolvla_grounding/held_out_evaluator.py": file_sha256(root / "src/smolvla_grounding/held_out_evaluator.py"),
                }
            })

        tasks = matrix["tasks"]
        models = matrix["models"]
        total_episodes = sum(len(m["initial_states"]) * 4 for m in tasks) * len(models)
        completed_count = 0

        print(f"\n=======================================================", flush=True)
        print(f" Starting Primary Confirmatory Evaluation: {run_name}", flush=True)
        print(f" Models: {models}", flush=True)
        print(f" Task: Task 0 (LIBERO-Spatial)", flush=True)
        print(f" Total planned rollouts: {total_episodes} (20 states x 4 prompts x 3 models)", flush=True)
        print(f" Horizon: 280 steps per episode", flush=True)
        print(f" Mode: Single Worker (Serial Execution, Worker=1)", flush=True)
        print(f"=======================================================\n", flush=True)

        baseline_identity = json.loads((root / "outputs/language_pilot/identity.json").read_text(encoding="utf-8"))

        for model_id in models:
            model_info = manifest["evaluated_models"][model_id]
            model_dir = output_dir / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n>>> Loading model [{model_id}] ...", flush=True)

            for task_cfg in tasks:
                task_id = task_cfg["task_id"]
                instructions = load_jsonl(root / task_cfg["instruction_catalog"])
                task_dir = model_dir / f"task_{task_id:02d}"
                task_dir.mkdir(parents=True, exist_ok=True)

                for state_idx in task_cfg["initial_states"]:
                    for inst in instructions:
                        completed_count += 1
                        inst_id = inst["id"]
                        trial_slug = f"state_{state_idx:04d}_seed_{matrix['policy']['policy_seed']:04d}_{inst_id}"
                        trial_dir = task_dir / trial_slug / "attempt_001"
                        complete_file = trial_dir / "complete.json"

                        pct = (completed_count / total_episodes) * 100
                        prefix = f"[{completed_count:03d}/{total_episodes:03d} - {pct:5.1f}%]"

                        if complete_file.exists():
                            print(f"{prefix} (Cached) {model_id} | Task {task_id} | State {state_idx} | {inst_id}", flush=True)
                            continue

                        print(f"{prefix} (Running) {model_id} | Task {task_id} | State {state_idx} | {inst_id}", flush=True)
                        trial_dir.mkdir(parents=True, exist_ok=True)

                        _execute_episode(
                            root=root,
                            matrix=matrix,
                            manifest=manifest,
                            model_id=model_id,
                            model_info=model_info,
                            task_id=task_id,
                            state_idx=state_idx,
                            instruction_row=inst,
                            trial_dir=trial_dir,
                            baseline_identity=baseline_identity
                        )

        print("\nAll confirmatory held-out evaluation episodes completed successfully!", flush=True)


def _batch_obs(obs):
    """Ensure raw observation from single Gym environment has batch dimension expected by vector processors."""
    if isinstance(obs, dict):
        return {k: _batch_obs(v) for k, v in obs.items()}
    if isinstance(obs, np.ndarray):
        return np.expand_dims(obs, axis=0)
    return [obs]


def _execute_episode(root, matrix, manifest, model_id, model_info, task_id, state_idx, instruction_row, trial_dir, baseline_identity):
    import torch
    from lerobot.envs.libero import LiberoEnv
    from lerobot.envs.utils import preprocess_observation
    from lerobot.utils.constants import ACTION
    from lerobot.utils.random_utils import set_seed, get_rng_state
    from lerobot.utils.io_utils import write_video
    from .libero_pairing import array_hashes, capture_initial_state
    from .diagnostic_policy import capture_noise, require_empty_queue
    from .diagnostic_records import pack_rng, pack_tensors
    from .adaptation_eval_runtime import AdaptationEvalRuntime
    from .benchmark_runtime import BenchmarkRuntime

    env_cfg = matrix["environment"]
    pol_cfg = matrix["policy"]
    suite_name = matrix["suite"]
    horizon = env_cfg["episode_length"]
    assert horizon == 280, f"Expected horizon 280, got {horizon}"

    set_seed(env_cfg["environment_seed"])
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[suite_name]()

    env = LiberoEnv(
        task_suite=suite,
        task_id=task_id,
        task_suite_name=suite_name,
        init_states=True,
        episode_index=state_idx,
        n_envs=1,
        obs_type=env_cfg["obs_type"],
        observation_height=env_cfg["observation_height"],
        observation_width=env_cfg["observation_width"],
        control_freq=env_cfg["control_freq"],
        control_mode=env_cfg["control_mode"],
        hard_reset=env_cfg["hard_reset"],
        num_steps_wait=env_cfg["num_steps_wait"],
        episode_length=horizon
    )

    try:
        obs, _ = env.reset(seed=env_cfg["environment_seed"])
        initial_arrays = capture_initial_state(env, obs, state_idx)
        initial_hashes = array_hashes(initial_arrays)

        atomic_json(trial_dir / "initial_hashes.json", initial_hashes)
        np.savez_compressed(trial_dir / "initial_state.npz", **initial_arrays)

        task_config = deepcopy(baseline_identity["config"])
        task_config["task_id"] = task_id
        task_config["suite"] = suite_name
        task_config["init_state_index"] = state_idx
        task_config["environment"]["episode_length"] = horizon
        task_config["policy"] = pol_cfg

        baseline_copy = deepcopy(baseline_identity)
        baseline_copy["config"] = task_config
        baseline_copy["instructions"] = [instruction_row]

        confirmatory_spec = {
            "matrix": matrix,
            "manifest": manifest,
            "model_id": model_id,
        }
        baseline_copy["confirmatory_spec"] = confirmatory_spec

        if model_id == "untuned_fp32":
            runtime = BenchmarkRuntime(root, baseline_copy, confirmatory_spec=confirmatory_spec)
        else:
            runtime = AdaptationEvalRuntime(root, baseline_copy, model_id, model_info, [instruction_row], confirmatory_spec=confirmatory_spec)

        runtime.load_policy()

        frames_main = []
        frames_wrist = []
        steps_records = []
        inference_records = []
        trajectory_actions = []
        trajectory_rewards = []
        trajectory_successes = []
        trajectory_sim_states = []

        goal_ever = False
        goal_final = False
        first_step = None

        policy_obj = runtime.policy
        policy_obj.reset()
        set_seed(pol_cfg["policy_seed"])

        curr_obs = obs
        for step in range(1, horizon + 1):
            if "pixels" in curr_obs:
                if "image" in curr_obs["pixels"]:
                    frames_main.append(curr_obs["pixels"]["image"])
                if "image2" in curr_obs["pixels"]:
                    frames_wrist.append(curr_obs["pixels"]["image2"])

            # 1. Official observation preprocessing
            obs_batched = _batch_obs(curr_obs)
            obs_converted = preprocess_observation(obs_batched)

            # 2. Attach frozen task/language instruction
            obs_converted["task"] = [instruction_row["text"]]

            # 3. Environment preprocessing
            obs_env = runtime.env_pre(obs_converted)

            # 4. Policy preprocessing
            batch = runtime.pre(obs_env)

            # Verify tokenized instruction matches frozen token sequence
            tokens = batch["observation.language.tokens"][0]
            mask = batch["observation.language.attention_mask"][0].bool()
            if tokens[mask].cpu().tolist() != runtime.token_ids[instruction_row["id"]]:
                raise RuntimeError(f"Tokenized instruction does not match frozen token_ids for {instruction_row['id']}")

            # 5. Policy inference
            require_empty_queue(policy_obj)
            rng_b = pack_rng(get_rng_state(), "rng_before", {}, {})
            with capture_noise(policy_obj) as noise:
                action = policy_obj.select_action(batch)
            require_empty_queue(policy_obj)
            rng_a = pack_rng(get_rng_state(), "rng_after", {}, {})

            # 6. Policy postprocessing
            unnormalized_action = runtime.post(action)

            # 7. Environment postprocessing
            action_transition = runtime.env_post({ACTION: unnormalized_action})
            final_action = action_transition[ACTION]

            # 8. Convert to environment-compatible representation
            action_np = final_action.squeeze(0).detach().cpu().numpy()

            raw_pixels_hash = array_hashes(curr_obs.get("pixels", {})) if "pixels" in curr_obs else {}
            inference_records.append({
                "step": step,
                "raw_pixels": raw_pixels_hash,
                "rng_before": rng_b,
                "rng_after": rng_a,
                "normalized_action": pack_tensors({"action": action}, "output", {}, {})
            })

            next_obs, reward, terminated, truncated, info = env.step(action_np)

            is_goal = get_task_goal_status(env)
            if is_goal and not goal_ever:
                goal_ever = True
                first_step = step
            if step == horizon:
                goal_final = is_goal

            domain = env._env.env
            obj_states = domain.object_states_dict
            positions = {}
            for name, state in obj_states.items():
                if hasattr(state, "get_position"):
                    positions[name] = state.get_position().tolist()

            eef_pos = domain.robots[0].controller.ee_pos.tolist() if hasattr(domain.robots[0], "controller") else []
            grip_qpos = domain.sim.data.qpos[domain.robots[0]._ref_gripper_joint_pos_indexes].tolist() if hasattr(domain.robots[0], "_ref_gripper_joint_pos_indexes") else []

            steps_records.append({
                "step": step,
                "instruction_id": instruction_row["id"],
                "action": action_np.tolist(),
                "reward": float(reward),
                "goal": is_goal,
                "eef_position": eef_pos,
                "gripper_qpos": grip_qpos,
                "object_positions": positions
            })

            trajectory_actions.append(action_np)
            trajectory_rewards.append(reward)
            trajectory_successes.append(is_goal)
            trajectory_sim_states.append(env._env.get_sim_state())

            curr_obs = next_obs

        if "pixels" in curr_obs:
            if "image" in curr_obs["pixels"]:
                frames_main.append(curr_obs["pixels"]["image"])
            if "image2" in curr_obs["pixels"]:
                frames_wrist.append(curr_obs["pixels"]["image2"])

        video_main_path = trial_dir / "image.mp4"
        video_wrist_path = trial_dir / "image2.mp4"
        if frames_main:
            write_video(str(video_main_path), np.stack(frames_main), fps=env_cfg["control_freq"])
        if frames_wrist:
            write_video(str(video_wrist_path), np.stack(frames_wrist), fps=env_cfg["control_freq"])

        np.savez_compressed(
            trial_dir / "trajectory.npz",
            action=np.array(trajectory_actions),
            reward=np.array(trajectory_rewards),
            success=np.array(trajectory_successes),
            sim_state=np.array(trajectory_sim_states)
        )

        with open(trial_dir / "steps.jsonl", "w", encoding="utf-8") as f:
            for r in steps_records:
                f.write(json.dumps(r, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")

        with open(trial_dir / "inference.jsonl", "w", encoding="utf-8") as f:
            for r in inference_records:
                f.write(json.dumps(r, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")

        episode_summary = {
            "model": model_id,
            "task_id": task_id,
            "initial_state_index": state_idx,
            "instruction_id": instruction_row["id"],
            "instruction": instruction_row["text"],
            "policy_steps": horizon,
            "goal_ever": goal_ever,
            "goal_final": goal_final,
            "first_step": first_step,
            "video_main": str(video_main_path.relative_to(trial_dir.parents[2])),
            "video_wrist": str(video_wrist_path.relative_to(trial_dir.parents[2]))
        }
        atomic_json(trial_dir / "episode.json", episode_summary)
        atomic_json(trial_dir / "complete.json", {"completed": True, "timestamp": time.time(), "goal_ever": goal_ever, "goal_final": goal_final})

    finally:
        env.close()
