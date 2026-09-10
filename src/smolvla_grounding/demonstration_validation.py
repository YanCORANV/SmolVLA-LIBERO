"""Validate selected demonstration rows in bounded Arrow batches."""

from io import BytesIO

import numpy as np
from PIL import Image


def validate_episode_rows(rows, metadata, task_index, fps):
    rows = sorted(rows, key=lambda r: r["frame_index"])
    length = metadata["length"]
    if len(rows) != length or [r["frame_index"] for r in rows] != list(range(length)):
        raise ValueError(f"Episode {metadata['episode_index']}: expected {length} frames numbered 0..{length - 1}, "
                         f"found {len(rows)}; check the metadata-to-shard mapping as well as frame integrity")
    if any(r["episode_index"] != metadata["episode_index"] or r["task_index"] != task_index for r in rows):
        raise ValueError("Wrong demonstration episode or task")
    if [r["index"] for r in rows] != list(range(metadata["dataset_from_index"], metadata["dataset_to_index"])):
        raise ValueError("Global frame indices differ from metadata")
    for key, dimension in (("action", 7), ("observation.state", 8)):
        values = np.asarray([r[key] for r in rows], dtype=np.float32)
        if values.shape != (length, dimension) or not np.isfinite(values).all():
            raise ValueError("Invalid numeric demonstration feature: " + key)
    timestamps = np.asarray([r["timestamp"] for r in rows])
    if not np.allclose(timestamps, np.arange(length) / fps, rtol=0, atol=1e-4):
        raise ValueError("Timestamps do not match declared dataset FPS")
    return {"episode_index": metadata["episode_index"], "frames": length,
            "task_index": task_index, "numeric_and_index_checks": "passed"}


def validate_shard(path, metadata, task_index, fps):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    by_id = {r["episode_index"]: r for r in metadata}
    selected = {key: [] for key in by_id}
    image_checks = {key: set() for key in by_id}
    cameras = ("observation.images.image", "observation.images.image2")
    columns = ["episode_index", "frame_index", "index", "task_index", "timestamp", "action", "observation.state", *cameras]
    ids = pa.array(sorted(by_id), type=pa.int64())
    for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=columns):
        table = pa.Table.from_batches([batch])
        table = table.filter(pc.is_in(table["episode_index"], value_set=ids))
        for row in table.to_pylist():
            episode = row["episode_index"]
            sample_indices = {0, by_id[episode]["length"] // 2, by_id[episode]["length"] - 1}
            for camera in cameras:
                cell = row.pop(camera)
                if row["frame_index"] in sample_indices:
                    if not isinstance(cell, dict) or not cell.get("bytes"):
                        raise ValueError("Expected embedded image bytes")
                    with Image.open(BytesIO(cell["bytes"])) as image:
                        image.load()
                        if image.size != (256, 256) or image.mode != "RGB":
                            raise ValueError("Unexpected decoded image shape or channels")
                    image_checks[episode].add((row["frame_index"], camera))
            selected[episode].append(row)
    results = []
    for episode, rows in selected.items():
        if not rows:
            raise ValueError(f"Episode {episode} is absent from {path}; metadata may point to the wrong shard. "
                             "The downloaded file must not be treated as this demonstration.")
        result = validate_episode_rows(rows, by_id[episode], task_index, fps)
        expected = len({0, by_id[episode]["length"] // 2, by_id[episode]["length"] - 1}) * len(cameras)
        if len(image_checks[episode]) != expected:
            raise ValueError("Missing sampled camera frames")
        result["decoded_image_samples"] = expected
        results.append(result)
    return results
