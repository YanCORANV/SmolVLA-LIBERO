"""Readiness and parameter-change acceptance checks, without GPU imports."""

import json
from pathlib import Path

from .protocol import file_sha256, fingerprint
from .training_data import split_episodes


def verify_ready(root, settings):
    root = Path(root)
    config = json.loads((root / "configs/task0_dataset.json").read_text())
    ready = json.loads((root / settings["data_ready"]).read_text())
    if not ready["passed"] or ready["selection"]["identity"]["config"] != fingerprint(config):
        raise ValueError("Prepared dataset/config mismatch")
    if ready["selection"]["repo_id"] != config["repo_id"] or ready["selection"]["revision"] != config["revision"]:
        raise ValueError("Prepared dataset identity mismatch")
    records = ready["episodes"]
    ids = [r["episode_index"] for r in records]
    if len(ids) != config["expected_task_episodes"] or len(set(ids)) != len(ids):
        raise ValueError("Wrong or duplicate prepared episode IDs")
    if sorted(ids) != sorted(r["episode_index"] for r in config["episode_file_overrides"]):
        raise ValueError("Prepared episodes differ from the pinned task selection")
    if sum(r["frames"] for r in records) != config["expected_task_frames"]:
        raise ValueError("Prepared frame total mismatch")
    split = ready["selection"]["split"]
    if split != split_episodes(ids, len(split["train"]), config["split_seed"]):
        raise ValueError("Prepared split changed")
    dataset = root / config["local_root"]
    prepared = dataset / "meta/episodes/chunk-000/file-000.parquet"
    if file_sha256(prepared) != ready["prepared_metadata"]["prepared_sha256"]:
        raise ValueError("Prepared metadata changed; do not load stale file pointers")
    for record in config["metadata"]:
        if record["path"] != "meta/episodes/chunk-000/file-000.parquet" and file_sha256(dataset / record["path"]) != record["sha256"]:
            raise ValueError("Dataset metadata changed: " + record["path"])
    for index, record in enumerate(config["data_files"], 1):
        print(f"  Verify prepared shard {index}/{len(config['data_files'])}: {record['path']}", flush=True)
        path = dataset / record["path"]
        if path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
            raise ValueError("Prepared demonstration file changed: " + record["path"])
    return config, ready, dataset


def compare_parameter_updates(before, after):
    if before.keys() != after.keys():
        raise ValueError("Parameter names changed during update")
    changed = []
    for name, previous in before.items():
        current = after[name]
        if any(previous[k] != current[k] for k in ("dtype", "shape", "trainable")):
            raise ValueError("Parameter type/shape/freeze settings changed: " + name)
        if previous["sha256"] != current["sha256"]:
            if not previous["trainable"]:
                raise ValueError("Frozen parameter was updated: " + name)
            changed.append(name)
    if not changed:
        raise ValueError("Optimizer did not change any trainable parameter")
    return {"changed_trainable_tensors": len(changed), "changed_names": changed, "frozen_unchanged": True}
