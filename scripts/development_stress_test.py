"""Full-horizon development-only infrastructure stress test.

Strictly complies with user and protocol rules:
- Uses ONLY LIBERO-Spatial Task 0
- Uses ONLY development State 0 (States 3-22 are strictly forbidden)
- Uses ONLY original development instruction
- Worker = 1
- Horizon = 280 environment steps
- Exercises the exact 9-stage LeRobot observation/action pipeline
- Evaluates sequentially all 3 frozen models: untuned_fp32, original_language, augmented_language
- Does not inspect, interpret, or report task success
- Measures resource usage, memory bounds, latency, and artifact finalization
"""

from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.envs.libero import LiberoEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.utils.constants import ACTION
from lerobot.utils.random_utils import set_seed, get_rng_state
from lerobot.utils.io_utils import write_video
from libero.libero import benchmark

from smolvla_grounding.benchmark_runtime import BenchmarkRuntime
from smolvla_grounding.adaptation_eval_runtime import AdaptationEvalRuntime
from smolvla_grounding.adaptation_eval_records import read_training
from smolvla_grounding.libero_pairing import array_hashes, capture_initial_state
from smolvla_grounding.diagnostic_policy import capture_noise, require_empty_queue
from smolvla_grounding.diagnostic_records import pack_rng, pack_tensors
from smolvla_grounding.protocol import atomic_json


def _batch_obs(obs):
    """Ensure raw observation from single Gym environment has batch dimension expected by vector processors."""
    if isinstance(obs, dict):
        return {k: _batch_obs(v) for k, v in obs.items()}
    if isinstance(obs, np.ndarray):
        return np.expand_dims(obs, axis=0)
    return [obs]


def run_model_stress_test(model_id, baseline_identity, training_info, output_dir):
    process = psutil.Process()
    print(f"\n=======================================================", flush=True)
    print(f"Starting Development Stress Test for Model: {model_id}", flush=True)
    print(f"=======================================================", flush=True)

    # Strict isolation safeguards
    dev_state_idx = 0
    assert dev_state_idx == 0, "Stress test must strictly use development state 0"
    assert dev_state_idx not in range(3, 23), "State 0 must never be in held-out states 3-22"

    horizon = 280
    assert horizon == 280, "Horizon must be strictly 280"

    dev_instruction = baseline_identity["instructions"][0]
    assert dev_instruction["id"] == "original", "Must use original development instruction"

    task_config = deepcopy(baseline_identity["config"])
    task_config["task_id"] = 0
    task_config["suite"] = "libero_spatial"
    assert task_config["init_state_index"] == dev_state_idx
    assert task_config["environment"]["episode_length"] == horizon

    baseline_copy = deepcopy(baseline_identity)
    baseline_copy["config"] = task_config
    baseline_copy["instructions"] = baseline_identity["instructions"]

    # 1. Measure initialization time
    t_init_start = time.perf_counter()

    if model_id == "untuned_fp32":
        runtime = BenchmarkRuntime(ROOT, baseline_copy)
    else:
        model_info = training_info["models"][model_id]
        runtime = AdaptationEvalRuntime(ROOT, baseline_copy, model_id, model_info, baseline_identity["instructions"])

    runtime.load_policy()

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    env_cfg = task_config["environment"]
    env_seed = task_config.get("environment_seed", env_cfg.get("environment_seed", 0))
    set_seed(env_seed)

    env = LiberoEnv(
        task_suite=suite,
        task_id=0,
        task_suite_name="libero_spatial",
        init_states=True,
        episode_index=dev_state_idx,
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

    obs, _ = env.reset(seed=env_seed)
    initial_arrays = capture_initial_state(env, obs, dev_state_idx)
    initial_hashes = array_hashes(initial_arrays)

    trial_dir = output_dir / model_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(trial_dir / "initial_hashes.json", initial_hashes)
    np.savez_compressed(trial_dir / "initial_state.npz", **initial_arrays)

    t_init_end = time.perf_counter()
    init_wall_time = t_init_end - t_init_start
    print(f"[{model_id}] Initialized in {init_wall_time:.2f} s", flush=True)

    # 2. Reset policy & seed
    policy_obj = runtime.policy
    policy_obj.reset()
    set_seed(task_config.get("policy_seed", 0))

    frames_main = []
    frames_wrist = []
    steps_records = []
    inference_records = []
    trajectory_actions = []
    trajectory_rewards = []
    trajectory_sim_states = []

    inference_latencies = []
    memory_samples = []

    torch.cuda.reset_peak_memory_stats()
    curr_obs = obs

    print(f"[{model_id}] Starting 280-step rollout...", flush=True)
    t_rollout_start = time.perf_counter()

    for step in range(1, horizon + 1):
        t_step_start = time.perf_counter()

        # Sanity checks on raw observation
        assert np.all(np.isfinite(curr_obs["pixels"]["image"])), f"Non-finite pixels.image at step {step}"
        assert np.all(np.isfinite(curr_obs["pixels"]["image2"])), f"Non-finite pixels.image2 at step {step}"
        assert np.all(np.isfinite(curr_obs["robot_state"]["eef"]["pos"])), f"Non-finite eef.pos at step {step}"
        assert np.all(np.isfinite(curr_obs["robot_state"]["eef"]["quat"])), f"Non-finite eef.quat at step {step}"

        frames_main.append(curr_obs["pixels"]["image"])
        frames_wrist.append(curr_obs["pixels"]["image2"])

        # Stage 1 & 2: Batch and preprocess observation
        obs_batched = _batch_obs(curr_obs)
        obs_converted = preprocess_observation(obs_batched)

        # Stage 3: Attach frozen language instruction
        obs_converted["task"] = [dev_instruction["text"]]

        # Stage 4: Environment preprocessing
        obs_env = runtime.env_pre(obs_converted)
        assert not torch.isnan(obs_env["observation.state"]).any(), f"NaN in obs_env state at step {step}"
        assert not torch.isinf(obs_env["observation.state"]).any(), f"Inf in obs_env state at step {step}"

        # Stage 5: Policy preprocessing
        batch = runtime.pre(obs_env)
        tokens = batch["observation.language.tokens"][0]
        mask = batch["observation.language.attention_mask"][0].bool()
        token_list = tokens[mask].cpu().tolist()
        assert token_list == runtime.token_ids[dev_instruction["id"]], f"Token mismatch at step {step}"

        # Stage 6: Policy inference with latency capture
        require_empty_queue(policy_obj)
        rng_b = pack_rng(get_rng_state(), "rng_before", {}, {})
        t_inf_start = time.perf_counter()
        with capture_noise(policy_obj) as noise:
            action = policy_obj.select_action(batch)
        t_inf_end = time.perf_counter()
        require_empty_queue(policy_obj)
        rng_a = pack_rng(get_rng_state(), "rng_after", {}, {})

        inf_latency_ms = (t_inf_end - t_inf_start) * 1000.0
        inference_latencies.append(inf_latency_ms)

        assert not torch.isnan(action).any(), f"NaN in policy output action at step {step}"
        assert not torch.isinf(action).any(), f"Inf in policy output action at step {step}"

        # Stage 7: Policy postprocessing
        unnormalized_action = runtime.post(action)

        # Stage 8: Environment postprocessing
        action_transition = runtime.env_post({ACTION: unnormalized_action})
        final_action = action_transition[ACTION]

        # Stage 9: Convert to numpy and step environment
        action_np = final_action.squeeze(0).detach().cpu().numpy()
        assert np.all(np.isfinite(action_np)), f"Non-finite action numpy array at step {step}"

        raw_pixels_hash = array_hashes(curr_obs.get("pixels", {}))
        inference_records.append({
            "step": step,
            "raw_pixels": raw_pixels_hash,
            "rng_before": rng_b,
            "rng_after": rng_a,
            "normalized_action": pack_tensors({"action": action}, "output", {}, {})
        })

        next_obs, reward, terminated, truncated, info = env.step(action_np)

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
            "instruction_id": dev_instruction["id"],
            "action": action_np.tolist(),
            "reward": float(reward),
            "eef_position": eef_pos,
            "gripper_qpos": grip_qpos,
            "object_positions": positions
        })

        trajectory_actions.append(action_np)
        trajectory_rewards.append(reward)
        trajectory_sim_states.append(env._env.get_sim_state())

        curr_obs = next_obs

        # Sample resource usage every 10 steps and at final step
        if step % 10 == 0 or step == horizon:
            gpu_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            gpu_res_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            proc_rss_mb = process.memory_info().rss / (1024 * 1024)
            sys_ram_gb = psutil.virtual_memory().used / (1024 ** 3)
            memory_samples.append({
                "step": step,
                "gpu_alloc_mb": gpu_alloc_mb,
                "gpu_res_mb": gpu_res_mb,
                "proc_rss_mb": proc_rss_mb,
                "sys_ram_gb": sys_ram_gb
            })
            if step % 70 == 0 or step == horizon:
                print(f"  [{model_id}] Step {step:3d}/{horizon}: inf={inf_latency_ms:.1f}ms, GPU={gpu_alloc_mb:.1f}MB, RSS={proc_rss_mb:.1f}MB", flush=True)

    t_rollout_end = time.perf_counter()
    rollout_wall_time = t_rollout_end - t_rollout_start
    print(f"[{model_id}] 280 steps finished in {rollout_wall_time:.2f} s (mean {(rollout_wall_time/horizon)*1000:.1f} ms/step)", flush=True)

    # 3. Finalize video and artifact writers
    print(f"[{model_id}] Finalizing videos and artifacts...", flush=True)
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
        sim_state=np.array(trajectory_sim_states)
    )

    with open(trial_dir / "steps.jsonl", "w", encoding="utf-8") as f:
        for r in steps_records:
            f.write(json.dumps(r, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")

    with open(trial_dir / "inference.jsonl", "w", encoding="utf-8") as f:
        for r in inference_records:
            f.write(json.dumps(r, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o.tolist() if isinstance(o, np.ndarray) else str(o)) + "\n")

    atomic_json(trial_dir / "complete.json", {
        "completed": True,
        "model": model_id,
        "steps": horizon,
        "timestamp": time.time()
    })

    # Measure final artifact sizes
    artifact_files = [
        "initial_hashes.json",
        "initial_state.npz",
        "image.mp4",
        "image2.mp4",
        "trajectory.npz",
        "steps.jsonl",
        "inference.jsonl",
        "complete.json"
    ]
    artifact_sizes = {}
    for fname in artifact_files:
        p = trial_dir / fname
        assert p.exists() and p.stat().st_size > 0, f"Artifact {fname} missing or empty"
        artifact_sizes[fname] = p.stat().st_size

    # Check memory boundedness: compare step 50 and step 280
    samples_by_step = {s["step"]: s for s in memory_samples}
    gpu_growth = samples_by_step[horizon]["gpu_alloc_mb"] - samples_by_step[50]["gpu_alloc_mb"]
    rss_growth = samples_by_step[horizon]["proc_rss_mb"] - samples_by_step[50]["proc_rss_mb"]
    bounded = abs(gpu_growth) < 20.0 and rss_growth < 100.0  # GPU alloc shouldn't grow, RSS shouldn't leak heavily

    peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_proc_rss_mb = max(s["proc_rss_mb"] for s in memory_samples)
    peak_sys_ram_gb = max(s["sys_ram_gb"] for s in memory_samples)

    # 4. Resource release
    print(f"[{model_id}] Closing environment and freeing model resources...", flush=True)
    env.close()
    del env
    del runtime
    del policy_obj
    torch.cuda.empty_cache()
    gc.collect()

    post_gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    post_rss_mb = process.memory_info().rss / (1024 * 1024)
    print(f"[{model_id}] Teardown complete. GPU alloc now: {post_gpu_mb:.1f} MB, RSS: {post_rss_mb:.1f} MB", flush=True)

    latencies_arr = np.array(inference_latencies)
    metrics = {
        "model": model_id,
        "completed_steps": horizon,
        "target_steps": 280,
        "init_wall_time_s": round(init_wall_time, 2),
        "rollout_wall_time_s": round(rollout_wall_time, 2),
        "total_wall_time_s": round(init_wall_time + rollout_wall_time, 2),
        "mean_inference_latency_ms": round(float(np.mean(latencies_arr)), 2),
        "p95_inference_latency_ms": round(float(np.percentile(latencies_arr, 95)), 2),
        "min_inference_latency_ms": round(float(np.min(latencies_arr)), 2),
        "max_inference_latency_ms": round(float(np.max(latencies_arr)), 2),
        "peak_gpu_allocated_mb": round(peak_gpu_mb, 2),
        "peak_process_rss_mb": round(peak_proc_rss_mb, 2),
        "peak_system_ram_gb": round(peak_sys_ram_gb, 2),
        "gpu_alloc_growth_step50_to_280_mb": round(gpu_growth, 2),
        "rss_growth_step50_to_280_mb": round(rss_growth, 2),
        "resource_usage_bounded": bounded,
        "artifact_sizes_bytes": artifact_sizes,
        "memory_samples": memory_samples
    }
    return metrics


def main():
    print("====================================================================", flush=True)
    print("LIBERO Task 0 Infrastructure Stress Test (280 steps x 3 models)", flush=True)
    print("Scope: STRICTLY Development State 0, Original Instruction, Worker = 1", flush=True)
    print("Held-out States 3-22 are STRICTLY NOT executed.", flush=True)
    print("====================================================================", flush=True)

    output_dir = ROOT / "outputs/development_stress_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_identity = json.loads((ROOT / "outputs/language_pilot/identity.json").read_text(encoding="utf-8"))
    training_settings = {
        "training_run": "outputs/adaptation/runs/task0_longer_training",
        "models": ["untuned_fp32", "original_language", "augmented_language"]
    }
    training_info = read_training(ROOT, training_settings)

    models_to_test = ["untuned_fp32", "original_language", "augmented_language"]
    results = {}

    for model_id in models_to_test:
        metrics = run_model_stress_test(
            model_id=model_id,
            baseline_identity=baseline_identity,
            training_info=training_info,
            output_dir=output_dir
        )
        results[model_id] = metrics

    # Save summary report
    summary_path = output_dir / "stress_test_report.json"
    atomic_json(summary_path, {
        "benchmark": "development_stress_test",
        "task_id": 0,
        "state_index": 0,
        "horizon": 280,
        "models": results,
        "timestamp": time.time()
    })
    print(f"\nStress test summary report written to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
