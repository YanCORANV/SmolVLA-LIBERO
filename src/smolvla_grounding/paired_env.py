"""Instrument the official environment; leave physics and predicate code upstream."""

import json
import numpy as np

from lerobot.envs.libero import LiberoEnv

from .libero_pairing import (array_hashes, capture_initial_state, check_task, goal_values,
                             require_matching_state)
from .protocol import atomic_json


class PairedLiberoEnv(LiberoEnv):
    def __init__(self, suite, config, original, row, reference_hashes, attempt, step_stream):
        super().__init__(task_suite=suite, task_id=config["task_id"],
                         task_suite_name=config["suite"], init_states=True,
                         episode_index=config["init_state_index"], n_envs=1, **config["environment"])
        check_task(self, config, original)
        self.config = config
        self.row = row
        self.task_description = row["text"]
        self.reference_hashes = reference_hashes
        self.attempt = attempt
        self.step_stream = step_stream
        self.records = []
        self.sim_states = []
        self.reset_count = 0
        self.metadata = {**self.metadata, "render_fps": self.control_freq}

    def reset(self, seed=None, **kwargs):
        if self.reset_count:
            raise RuntimeError("Unexpected reset during a fixed-state trial")
        self.reset_count += 1
        self.init_state_id = self.config["init_state_index"]
        observation, info = super().reset(seed=seed, **kwargs)
        parsed = self._env.env.parsed_problem["goal_state"]
        if [list(p) for p in parsed] != [self.config["goals"]["original_bowl"]]:
            raise RuntimeError("Original BDDL goal changed")
        arrays = capture_initial_state(self, observation, self.config["init_state_index"])
        self.initial_hashes = array_hashes(arrays)
        np.savez_compressed(self.attempt / "initial_state.npz", **arrays)
        atomic_json(self.attempt / "initial_hashes.json", self.initial_hashes)
        require_matching_state(self.reference_hashes, self.initial_hashes)
        if any(goal_values(self, self.config["goals"]).values()):
            raise RuntimeError("Goal already satisfied at reset")
        self.sim_states.append(arrays["sim/state"])
        self.latest_pixels = observation["pixels"]
        return observation, info

    def step(self, action):
        if action.shape != (7,) or not np.isfinite(action).all():
            raise RuntimeError("Expected a finite seven-dimensional action")
        observation, reward, native_terminated, native_truncated, info = super().step(action)
        values = goal_values(self, self.config["goals"])
        # LIBERO's task done is the original predicate. Robosuite's internal done is
        # its physics horizon; it must stay false while our common 280-step trial runs.
        if self._env.env.done or native_truncated:
            raise RuntimeError("Unexpected underlying simulator termination")
        state = np.asarray(self._env.get_sim_state()).copy()
        if not np.isfinite(state).all():
            raise RuntimeError("Nonfinite simulator state after action")
        self.sim_states.append(state)
        objects = self._env.env.object_states_dict
        names = sorted({name for g in self.config["goals"].values() for name in g[1:]})
        positions = {name: np.asarray(objects[name].get_geom_state()["pos"]).tolist() for name in names}
        record = {"step": len(self.records) + 1, "instruction_id": self.row["id"],
                  "instruction": self.task_description, "action": action.tolist(),
                  "native_reward": float(reward), "native_terminated": bool(native_terminated),
                  "goals": values, "object_positions": positions,
                  "eef_position": np.asarray(observation["robot_state"]["eef"]["pos"]).tolist(),
                  "gripper_qpos": np.asarray(observation["robot_state"]["gripper"]["qpos"]).tolist()}
        self.step_stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.step_stream.flush()
        self.records.append(record)
        self.latest_pixels = observation["pixels"]
        # The official rollout stops at _max_episode_steps. All language conditions
        # receive that same horizon, even when the original task predicate fires.
        info["is_success"] = values[self.row["goal_id"]]
        return observation, reward, False, False, info
