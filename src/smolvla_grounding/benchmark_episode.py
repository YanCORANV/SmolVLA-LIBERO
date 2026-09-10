"""State-pair verification and instrumented episodes through upstream rollout."""

import json
import time

import av
import gymnasium as gym
import numpy as np
from PIL import Image
import torch
from torch import nn

from lerobot.envs.libero import LiberoEnv
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import get_rng_state, serialize_rng_state, set_seed

from .diagnostic_policy import capture_noise, require_empty_queue
from .diagnostic_records import pack_rng, pack_tensors
from .libero_pairing import array_hashes, capture_initial_state, check_task, goal_values, require_matching_state
from .paired_env import PairedLiberoEnv
from .protocol import atomic_json
from .rollout_records import summarize_goals


def verify_reference(runtime, config, attempt):
    reference = None
    artifacts = []
    for repeat in range(2):
        print(f"  State {config['init_state_index']}: fresh reset {repeat + 1}/2", flush=True)
        env = None
        try:
            set_seed(config["environment_seed"])
            env = LiberoEnv(task_suite=runtime.suite, task_id=config["task_id"], task_suite_name=config["suite"],
                            init_states=True, episode_index=config["init_state_index"], n_envs=1, **config["environment"])
            check_task(env, config, runtime.rows[0])
            observation, _ = env.reset(seed=config["environment_seed"])
            parsed = [list(p) for p in env._env.env.parsed_problem["goal_state"]]
            if parsed != [config["goals"]["original_bowl"]]:
                raise RuntimeError("Task goal differs from the approved original predicate")
            arrays = capture_initial_state(env, observation, config["init_state_index"])
            path = attempt / f"reset_{repeat + 1}.npz"
            np.savez_compressed(path, **arrays)
            artifacts.append(path)
            hashes = array_hashes(arrays)
            goals = goal_values(env, config["goals"])
            if any(goals.values()):
                raise RuntimeError("A requested goal is already true at this initial state")
            for camera, pixels in observation["pixels"].items():
                path = attempt / f"reset_{repeat + 1}_{camera}.png"
                Image.fromarray(pixels[::-1, ::-1]).save(path)
                artifacts.append(path)
            if reference is None:
                reference = hashes
            else:
                require_matching_state(reference, hashes)
        finally:
            if env is not None:
                env.close()
    result = {"initial_state_index": config["init_state_index"], "environment_seed": config["environment_seed"],
              "initial_hashes": reference, "initial_goals": goals, "fresh_resets": 2,
              "alternative_goal_validation": "initial negative cases only; positive validation pending"}
    atomic_json(attempt / "reference.json", result)
    return result, artifacts + [attempt / "reference.json"]


class PolicyHashes(nn.Module):
    """Keep per-step input/RNG/noise hashes without archiving full image tensors."""

    def __init__(self, policy, env, stream):
        super().__init__()
        self.policy, self.env, self.stream = policy, env, stream
        self.step = 0

    def reset(self):
        self.policy.reset()
        self.step = 0

    def select_action(self, batch):
        require_empty_queue(self.policy)
        self.step += 1
        record = {"step": self.step, "raw_pixels": array_hashes(self.env.latest_pixels),
                  "input": pack_tensors({k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}, "input", {}, {}),
                  "rng_before": pack_rng(get_rng_state(), "rng_before", {}, {})}
        with capture_noise(self.policy) as noise:
            action = self.policy.select_action(batch)
        if len(noise) != 1 or not torch.isfinite(action).all():
            raise RuntimeError("Unexpected noise calls or nonfinite normalized action")
        require_empty_queue(self.policy)
        record.update(rng_after=pack_rng(get_rng_state(), "rng_after", {}, {}),
                      noise=pack_tensors({"noise": noise[0]}, "noise", {}, {}),
                      normalized_action=pack_tensors({"action": action}, "output", {}, {}))
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.stream.flush()
        return action


def run_episode(runtime, config, row, reference, reference_rng, attempt, directory):
    env = vec = None
    frames = {"image": [], "image2": []}
    initial_rng_hashes = {}
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        for processor in (runtime.pre, runtime.post, runtime.env_pre, runtime.env_post):
            processor.reset()
        set_seed(config["environment_seed"])
        with (attempt / "steps.jsonl").open("w", encoding="utf-8") as steps_log, \
             (attempt / "inference.jsonl").open("w", encoding="utf-8") as inference_log:
            env = PairedLiberoEnv(runtime.suite, config, runtime.rows[0], row, reference["initial_hashes"], attempt, steps_log)
            policy = PolicyHashes(runtime.policy, env, inference_log).eval()
            vec = gym.vector.SyncVectorEnv([lambda: env])

            def render(_):
                for camera in frames:
                    frames[camera].append(env.latest_pixels[camera][::-1, ::-1].copy())
                if len(frames["image"]) == 1:
                    set_seed(config["policy_seed"])
                    rng = {k: v.cpu().numpy().copy() for k, v in serialize_rng_state().items()}
                    initial_rng_hashes.update(array_hashes(rng))
                    np.savez_compressed(attempt / "initial_rng.npz", **rng)
                    if reference_rng is not None:
                        require_matching_state(reference_rng, initial_rng_hashes)

            def preprocess(observation):
                if observation["task"] != [row["text"]]:
                    raise RuntimeError("Wrong instruction reached the policy processor")
                batch = runtime.pre(observation)
                tokens = batch[OBS_LANGUAGE_TOKENS][0][batch[OBS_LANGUAGE_ATTENTION_MASK][0].bool()]
                if tokens.cpu().tolist() != runtime.token_ids[row["id"]]:
                    raise RuntimeError("Policy tokens differ from the validated instruction")
                return batch

            rollout_start = time.perf_counter()
            tensors = rollout(vec, policy, runtime.env_pre, runtime.env_post, preprocess, runtime.post,
                              seeds=[config["environment_seed"]], render_callback=render)
            torch.cuda.synchronize()
            rollout_seconds = time.perf_counter() - rollout_start
            steps = config["environment"]["episode_length"]
            if policy.step != steps or len(env.records) != steps or any(len(f) != steps + 1 for f in frames.values()):
                raise RuntimeError("Incomplete episode, inference log or frames")
            trajectory = {k: v.cpu().numpy()[0] for k, v in tensors.items()}
            if not np.array_equal(trajectory["action"], np.asarray([r["action"] for r in env.records], dtype=trajectory["action"].dtype)):
                raise RuntimeError("Action tensor/log mismatch")
            if not np.array_equal(trajectory["success"], [r["goals"][row["goal_id"]] for r in env.records]):
                raise RuntimeError("Requested goal tensor/log mismatch")
            trajectory["sim_state"] = np.stack(env.sim_states)
            for goal_id in config["goals"]:
                trajectory["goal_" + goal_id] = np.asarray([r["goals"][goal_id] for r in env.records])
            if any(not np.isfinite(v).all() for v in trajectory.values()):
                raise RuntimeError("Nonfinite trajectory")
            np.savez_compressed(attempt / "trajectory.npz", **trajectory)
            goals = summarize_goals([r["goals"] for r in env.records], config["goals"])
            hashes = env.initial_hashes
        video_info = {}
        for camera in frames:
            path = attempt / (camera + ".mp4")
            write_video(path, frames[camera], fps=config["environment"]["control_freq"])
            with av.open(str(path)) as video:
                fps = float(video.streams.video[0].average_rate)
                count = sum(1 for _ in video.decode(video=0))
            if count != steps + 1 or fps != config["environment"]["control_freq"]:
                raise RuntimeError("Video length or frame rate differs from the protocol")
            video_info[camera] = {"frames": count, "fps": fps}
        vec.close()
        env = vec = None
        result = {"suite": config["suite"], "task_id": config["task_id"], "initial_state_index": config["init_state_index"],
                  "environment_seed": config["environment_seed"], "policy_seed": config["policy_seed"],
                  "instruction_id": row["id"], "condition": row["condition"], "instruction": row["text"],
                  "goal_id": row["goal_id"], "policy_steps": steps, "goals": goals,
                  "initial_hashes": hashes, "initial_rng_hashes": initial_rng_hashes,
                  "rollout_seconds": rollout_seconds, "trial_seconds": time.perf_counter() - start,
                  "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "video_main": (attempt / "image.mp4").relative_to(directory).as_posix(),
                  "video_wrist": (attempt / "image2.mp4").relative_to(directory).as_posix(), "video_info": video_info,
                  "trajectory": (attempt / "trajectory.npz").relative_to(directory).as_posix(),
                  "inference_log": (attempt / "inference.jsonl").relative_to(directory).as_posix(),
                  "alternative_goal_validation": "provisional; positive cases require review"}
        atomic_json(attempt / "episode.json", result)
        artifacts = [attempt / n for n in ("steps.jsonl", "inference.jsonl", "trajectory.npz", "initial_rng.npz",
                     "initial_state.npz", "initial_hashes.json", "image.mp4", "image2.mp4", "episode.json")]
        return result, artifacts
    finally:
        if vec is not None:
            vec.close()
        elif env is not None:
            env.close()
