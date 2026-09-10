"""Thin inspection helpers for the installed official LeRobot LIBERO environment.

Private attributes are version-specific. Runtime checks fail on incompatible changes.
No environment physics, predicates or model implementation is copied here.
"""

import hashlib
import json

import numpy as np


def flatten_observation(value, prefix="observation"):
    arrays = {}
    if isinstance(value, dict):
        for key in sorted(value):
            arrays.update(flatten_observation(value[key], f"{prefix}/{key}"))
    else:
        array = np.asarray(value)
        if array.dtype.hasobject or not np.isfinite(array).all():
            raise ValueError(f"Missing or nonfinite observation: {prefix}")
        arrays[prefix] = array.copy()
    return arrays


def array_hashes(arrays):
    hashes = {}
    for name, array in sorted(arrays.items()):
        array = np.ascontiguousarray(array)
        header = json.dumps([array.dtype.str, list(array.shape)]).encode()
        hashes[name] = hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()
    return hashes


def capture_initial_state(env, observation, init_state_index):
    sim = env._env.sim
    arrays = flatten_observation(observation)
    arrays["fixture_state"] = np.asarray(env._init_states[init_state_index]).copy()
    arrays["sim/state"] = np.asarray(env._env.get_sim_state()).copy()
    # get_state omits these mutable dynamics/model values; retain them separately.
    for name in ("ctrl", "act", "qacc_warmstart", "qfrc_applied", "xfrc_applied", "mocap_pos", "mocap_quat"):
        arrays[f"sim/data/{name}"] = np.asarray(getattr(sim.data, name)).copy()
    for name in ("body_pos", "body_quat", "body_mass", "body_inertia", "geom_pos", "geom_quat", "geom_size"):
        arrays[f"sim/model/{name}"] = np.asarray(getattr(sim.model, name)).copy()
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Nonfinite simulator state")
    return arrays


def goal_values(env, goals):
    from libero.libero.envs.predicates import eval_predicate_fn

    domain = env._env.env
    states = domain.object_states_dict
    values = {}
    for goal_id, (predicate, source, destination) in goals.items():
        values[goal_id] = bool(eval_predicate_fn(predicate, states[source], states[destination]))
    if values["original_bowl"] != bool(env._env.check_success()):
        raise RuntimeError("Project original-goal evaluation disagrees with native LIBERO success")
    return values


def check_task(env, config, original):
    if env.task != config["task_name"] or env.task_description != original["text"]:
        raise RuntimeError("Official task identity/instruction differs from the versioned catalog")
    if config["init_state_index"] >= len(env._init_states):
        raise ValueError("Requested initial-state index is out of range (no modulo wrap allowed)")


def require_matching_state(reference, current):
    differences = sorted(k for k in reference.keys() | current.keys()
                         if reference.get(k) != current.get(k))
    if differences:
        raise RuntimeError("Paired reset mismatch in: " + ", ".join(differences))
