"""Small metadata checks and deterministic episode splits; no policy imports."""

import hashlib
import json
from pathlib import Path
import urllib.request

from .protocol import file_sha256


def verify_file(path, record):
    path = Path(path)
    return (path.is_file() and path.stat().st_size == record["bytes"]
            and file_sha256(path) == record["sha256"])


def fetch_metadata(config, root):
    """Fetch only four pinned small files, atomically; never fetch demonstrations."""
    root = Path(root)
    for index, record in enumerate(config["metadata"], 1):
        path = root / record["path"]
        print(f"  Metadata {index}/{len(config['metadata'])}: {record['path']}", flush=True)
        if verify_file(path, record):
            continue
        url = f"https://huggingface.co/datasets/{config['repo_id']}/resolve/{config['revision']}/{record['path']}"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read(record["bytes"] + 1)
        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
            raise RuntimeError("Metadata differs from pinned bytes: " + record["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    if "mapping_source" in config:
        source = config["mapping_source"]
        fetch_metadata({"repo_id": source["repo_id"], "revision": source["revision"],
                        "metadata": [source]}, root / "_mapping_reference")


def match_episodes(tasks, episodes, config):
    matches = [r for r in tasks if r["__index_level_0__"] == config["original_instruction"]]
    if len(matches) != 1 or matches[0]["task_index"] != config["dataset_task_index"]:
        raise ValueError("Official original instruction must match exactly one pinned dataset task")
    selected = sorted((dict(r) for r in episodes if r["tasks"] == [config["original_instruction"]]),
                      key=lambda r: r["episode_index"])
    ids = [r["episode_index"] for r in selected]
    if len(set(ids)) != len(ids) or len(ids) != config["expected_task_episodes"]:
        raise ValueError("Wrong episode count or duplicate episode IDs")
    if sum(r["length"] for r in selected) != config["expected_task_frames"]:
        raise ValueError("Wrong selected frame count")
    for row in selected:
        if row["length"] <= 0 or row["dataset_to_index"] - row["dataset_from_index"] != row["length"]:
            raise ValueError("Inconsistent episode frame bounds")
    if "episode_file_overrides" in config:
        overrides = config["episode_file_overrides"]
        if sorted(r["episode_index"] for r in overrides) != ids:
            raise ValueError("File-location corrections must cover exactly the selected episode IDs")
        by_id = {r["episode_index"]: r for r in overrides}
        for row in selected:
            row["original_data_path"] = f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet"
            for field in ("data/chunk_index", "data/file_index"):
                row[field] = by_id[row["episode_index"]][field]
    paths = {f"data/chunk-{r['data/chunk_index']:03d}/file-{r['data/file_index']:03d}.parquet" for r in selected}
    if paths != {r["path"] for r in config["data_files"]}:
        raise ValueError("Pinned shard list does not cover exactly the selected episodes")
    return selected


def inspect_metadata(root, config):
    import pyarrow.parquet as pq

    root = Path(root)
    for record in config["metadata"]:
        if not verify_file(root / record["path"], record):
            raise ValueError("Missing or modified metadata: " + record["path"])
    info = json.loads((root / "meta/info.json").read_text())
    if info["codebase_version"] != "v3.0" or info["fps"] != 10:
        raise ValueError("Unexpected dataset format or declared FPS")
    expected = {"observation.state": ("float32", [8]), "action": ("float32", [7]),
                "observation.images.image": ("image", [256, 256, 3]),
                "observation.images.image2": ("image", [256, 256, 3])}
    for key, (dtype, shape) in expected.items():
        if (info["features"][key]["dtype"], info["features"][key]["shape"]) != (dtype, shape):
            raise ValueError("Unexpected feature: " + key)
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    columns = ["episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
               "dataset_from_index", "dataset_to_index"]
    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet", columns=columns).to_pylist()
    selected = match_episodes(tasks, episodes, config)
    if "mapping_source" in config:
        source = config["mapping_source"]
        path = root / "_mapping_reference" / source["path"]
        if not verify_file(path, source):
            raise ValueError("Missing or modified official file-location reference")
        locations = {r["episode_index"]: r for r in pq.read_table(path, columns=[
            "episode_index", "length", "dataset_from_index", "dataset_to_index",
            "data/chunk_index", "data/file_index"]).to_pylist()}
        for row in selected:
            if any(row[k] != locations[row["episode_index"]][k] for k in (
                    "length", "dataset_from_index", "dataset_to_index", "data/chunk_index", "data/file_index")):
                raise ValueError("Correction disagrees with official episode identity, length, bounds or location")
    return {"repo_id": config["repo_id"], "revision": config["revision"],
            "original_instruction": config["original_instruction"], "dataset_task_index": config["dataset_task_index"],
            "simulator_task_id": config["simulator_task_id"], "episodes": selected,
            "selected_episode_count": len(selected), "selected_frame_count": sum(r["length"] for r in selected),
            "required_download_bytes": sum(r["bytes"] for r in config["data_files"]),
            "dataset_fps": info["fps"], "evaluation_control_hz": 20,
            "file_location_correction": config.get("mapping_source"),
            "images": "embedded in parquet; no separate videos required",
            "checkpoint_training_history": "not established by this dataset match",
            "temporal_alignment": "dataset declares 10 FPS; do not change the 20 Hz evaluation or rescale actions without verifying conversion semantics"}


def validate_saved_selection(selection, config, ids):
    """Preserve the exact split across source-code fixes and the pinned location repair."""
    from .protocol import fingerprint
    if selection["repo_id"] != config["repo_id"] or selection["revision"] != config["revision"]:
        raise ValueError("Saved dataset identity differs")
    allowed = {fingerprint(config)}
    if config.get("supersedes_config_fingerprint"):
        allowed.add(config["supersedes_config_fingerprint"])
    if selection["identity"]["config"] not in allowed:
        raise ValueError("Saved config is not current or the explicitly superseded location config")
    expected = split_episodes(ids, len(selection["split"]["train"]), config["split_seed"])
    if selection["split"] != expected:
        raise ValueError("Saved split was modified")
    return expected


def write_prepared_metadata(root, config, audit):
    """Correct selected file pointers only; keep upstream bytes and all frame data."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .protocol import atomic_json

    root = Path(root)
    relative = "meta/episodes/chunk-000/file-000.parquet"
    path = root / relative
    record = next(r for r in config["metadata"] if r["path"] == relative)
    if not verify_file(path, record):
        raise ValueError("Expected verified upstream episode metadata before applying file-pointer repair")
    original = root / "_original_metadata" / relative
    original.parent.mkdir(parents=True, exist_ok=True)
    if original.exists() and not verify_file(original, record):
        raise ValueError("Preserved original metadata changed")
    if not original.exists():
        original.write_bytes(path.read_bytes())
    table = pq.read_table(path)
    corrected = {r["episode_index"]: r for r in audit["episodes"]}
    ids = table["episode_index"].to_pylist()
    for field in ("data/chunk_index", "data/file_index"):
        values = table[field].to_pylist()
        values = [corrected[ep][field] if ep in corrected else value for ep, value in zip(ids, values)]
        table = table.set_column(table.schema.get_field_index(field), table.schema.field(field),
                                 pa.array(values, type=table[field].type))
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary)
    temporary.replace(path)
    report = {"source_sha256": record["sha256"], "prepared_sha256": file_sha256(path),
              "corrected_episode_ids": sorted(corrected), "mapping_source": config.get("mapping_source"),
              "scope": "Only selected episode data file pointers; frame arrays and image bytes untouched"}
    atomic_json(root / "file_location_repair.json", report)
    return report


def split_episodes(episode_ids, train_count, seed):
    ids = sorted(episode_ids)
    if len(set(ids)) != len(ids) or not 1 <= train_count < len(ids):
        raise ValueError("Choose at least one training and one held-out demonstration; IDs must be unique")
    # Hash ranking avoids version-dependent RNG algorithms and consumes no training RNG.
    ranked = sorted(ids, key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest())
    return {"seed": seed, "train": sorted(ranked[:train_count]), "held_out": sorted(ranked[train_count:]),
            "meaning": "Demonstration split only; does not establish simulator-state or upstream-pretraining disjointness"}
