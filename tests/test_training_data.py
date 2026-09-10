"""Data identity, held-out isolation and corrupt-frame acceptance checks."""

from copy import deepcopy
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.training_data import match_episodes, split_episodes, validate_saved_selection, write_prepared_metadata
from smolvla_grounding.protocol import file_sha256, fingerprint
from smolvla_grounding.demonstration_validation import validate_episode_rows, validate_shard


class TrainingDataTests(unittest.TestCase):
    def test_split_is_disjoint_reproducible_and_order_independent(self):
        split = split_episodes(list(range(45)), 36, 7)
        self.assertEqual(split, split_episodes(list(reversed(range(45))), 36, 7))
        self.assertEqual(len(split["train"]), 36)
        self.assertEqual(len(split["held_out"]), 9)
        self.assertFalse(set(split["train"]) & set(split["held_out"]))
        self.assertEqual(sorted(split["train"] + split["held_out"]), list(range(45)))

    def test_empty_holdout_and_duplicate_ids_are_rejected(self):
        for ids, count in [([0, 1], 0), ([0, 1], 2), ([0, 0, 1], 1)]:
            with self.assertRaises(ValueError):
                split_episodes(ids, count, 0)

    def test_pointer_repair_preserves_split_but_unrelated_config_or_changed_split_fails(self):
        config = {"repo_id": "official/data", "revision": "fixed", "split_seed": 7,
                  "supersedes_config_fingerprint": "old-location-config"}
        split = split_episodes(list(range(45)), 36, 7)
        selection = {"repo_id": config["repo_id"], "revision": config["revision"], "split": split,
                     "identity": {"config": "old-location-config", "sources": {"old": "hash"}}}
        self.assertEqual(validate_saved_selection(selection, config, list(range(45))), split)
        unrelated = deepcopy(selection)
        unrelated["identity"]["config"] = "unrelated"
        changed = deepcopy(selection)
        changed["split"]["train"][0] = 12345
        for bad in (unrelated, changed):
            with self.assertRaises(ValueError):
                validate_saved_selection(bad, config, list(range(45)))

    def test_file_corrections_require_exact_episode_coverage(self):
        config = {"original_instruction": "take the bowl", "dataset_task_index": 34,
                  "expected_task_episodes": 1, "expected_task_frames": 2,
                  "data_files": [{"path": "data/chunk-000/file-309.parquet"}],
                  "episode_file_overrides": [{"episode_index": 12, "data/chunk_index": 0, "data/file_index": 309}]}
        tasks = [{"__index_level_0__": "take the bowl", "task_index": 34}]
        rows = [{"tasks": ["take the bowl"], "episode_index": 12, "length": 2,
                 "dataset_from_index": 10, "dataset_to_index": 12, "data/chunk_index": 0, "data/file_index": 55}]
        corrected = match_episodes(tasks, rows, config)
        self.assertEqual(corrected[0]["data/file_index"], 309)
        self.assertEqual(rows[0]["data/file_index"], 55)
        config["episode_file_overrides"] = []
        with self.assertRaises(ValueError):
            match_episodes(tasks, rows, config)

    def test_prepared_metadata_changes_only_selected_file_pointers(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "meta/episodes/chunk-000/file-000.parquet"
            path.parent.mkdir(parents=True)
            original = pa.Table.from_pylist([
                {"episode_index": 12, "data/chunk_index": 0, "data/file_index": 55, "tasks": ["A"], "length": 84},
                {"episode_index": 13, "data/chunk_index": 0, "data/file_index": 56, "tasks": ["B"], "length": 85}])
            pq.write_table(original, path)
            config = {"metadata": [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                                     "sha256": file_sha256(path)}]}
            audit = {"episodes": [{"episode_index": 12, "data/chunk_index": 0, "data/file_index": 309}]}
            result = write_prepared_metadata(root, config, audit)
            updated = pq.read_table(path)
            self.assertEqual(updated["data/file_index"].to_pylist(), [309, 56])
            for key in ("episode_index", "tasks", "length"):
                self.assertTrue(original[key].equals(updated[key]))
            self.assertEqual(file_sha256(root / "_original_metadata" / path.relative_to(root)), result["source_sha256"])

    def test_task_mapping_uses_exact_text_not_simulator_index(self):
        config = {"original_instruction": "take the bowl", "dataset_task_index": 34,
                  "expected_task_episodes": 1, "expected_task_frames": 2,
                  "data_files": [{"path": "data/chunk-000/file-055.parquet"}]}
        tasks = [{"__index_level_0__": "take the bowl", "task_index": 34}]
        episodes = [{"tasks": ["take the bowl"], "episode_index": 12, "length": 2,
                     "dataset_from_index": 10, "dataset_to_index": 12,
                     "data/chunk_index": 0, "data/file_index": 55}]
        self.assertEqual(match_episodes(tasks, episodes, config), episodes)
        for changed in [dict(config, dataset_task_index=0), dict(config, original_instruction="take a bowl")]:
            with self.assertRaises(ValueError):
                match_episodes(tasks, episodes, changed)

    def rows(self):
        return [{"episode_index": 12, "frame_index": i, "index": 100 + i, "task_index": 34,
                 "timestamp": i / 10, "action": [0.] * 7, "observation.state": [0.] * 8} for i in range(3)]

    def metadata(self):
        return {"episode_index": 12, "length": 3, "dataset_from_index": 100, "dataset_to_index": 103}

    def test_missing_duplicate_wrong_task_and_nonfinite_frames_fail(self):
        rows = self.rows()
        self.assertEqual(validate_episode_rows(rows, self.metadata(), 34, 10)["frames"], 3)
        bad_task = deepcopy(rows)
        bad_task[1]["task_index"] = 0
        bad_number = deepcopy(rows)
        bad_number[1]["action"][0] = float("nan")
        bad_time = deepcopy(rows)
        bad_time[1]["timestamp"] = 0.05
        for bad in [rows[:-1], [rows[0], rows[0], rows[2]], bad_task, bad_number, bad_time]:
            with self.assertRaises(ValueError):
                validate_episode_rows(bad, self.metadata(), 34, 10)

    def test_arrow_validation_filters_other_tasks_and_decodes_both_cameras(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image
        stream = BytesIO()
        Image.new("RGB", (256, 256), (20, 40, 60)).save(stream, format="PNG")
        rows = self.rows()
        other = deepcopy(rows[0])
        other.update(episode_index=99, task_index=1)
        rows.append(other)
        for row in rows:
            for camera in ("observation.images.image", "observation.images.image2"):
                row[camera] = {"bytes": stream.getvalue(), "path": None}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shard.parquet"
            pq.write_table(pa.Table.from_pylist(rows), path)
            report = validate_shard(path, [self.metadata()], 34, 10)
            self.assertEqual(len(report), 1)
            self.assertEqual(report[0]["decoded_image_samples"], 6)


if __name__ == "__main__":
    unittest.main()
