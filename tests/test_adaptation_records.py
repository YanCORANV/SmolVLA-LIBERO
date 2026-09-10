"""Regression checks for language leakage, paired sampling and interrupted training."""

from copy import deepcopy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.adaptation_records import (
    build_schedule, checkpoint_steps, compare_training_logs, latest_checkpoint, read_jsonl, validate_training_catalog,
)
from smolvla_grounding.benchmark_records import BenchmarkJournal


class LanguageTests(unittest.TestCase):
    def test_real_catalog_keeps_development_texts_out_of_training(self):
        train = read_jsonl(ROOT / "configs/instructions/task0_train.jsonl")
        development = read_jsonl(ROOT / "configs/instructions/task0.jsonl")
        validate_training_catalog(train, development, development[0]["text"])
        bad = deepcopy(train)
        bad[1]["text"] = development[2]["text"].upper() + "."
        with self.assertRaises(ValueError):
            validate_training_catalog(bad, development, development[0]["text"])
        bad = deepcopy(train)
        bad[1]["goal_id"] = "ramekin"
        with self.assertRaises(ValueError):
            validate_training_catalog(bad, development, development[0]["text"])


class ScheduleTests(unittest.TestCase):
    def episodes(self):
        return [{"episode_index": 10, "frames": 3}, {"episode_index": 42, "frames": 4}]

    def test_reproducible_balanced_language_schedule_does_not_consume_global_rng(self):
        state = random.getstate()
        a = build_schedule(self.episodes(), 20, 7, 9, 5)
        b = build_schedule(self.episodes(), 20, 7, 9, 5)
        self.assertEqual(a, b)
        self.assertEqual(state, random.getstate())
        self.assertEqual({i: sum(r["augmented_variant"] == i for r in a) for i in range(5)}, dict.fromkeys(range(5), 4))
        self.assertTrue(all(r["episode_index"] in (10, 42) for r in a))

    def test_language_choice_cannot_change_frame_schedule(self):
        a = build_schedule(self.episodes(), 20, 7, 9, 5)
        b = build_schedule(self.episodes(), 20, 7, 123, 7)
        self.assertEqual([(r["episode_index"], r["frame_index"]) for r in a],
                         [(r["episode_index"], r["frame_index"]) for r in b])

    def test_incomplete_checkpoint_keeps_last_completed_boundary(self):
        self.assertEqual(checkpoint_steps(250, 100), [100, 200, 250])
        with tempfile.TemporaryDirectory() as directory:
            journal = BenchmarkJournal(directory, {"comparison": "fixed"})
            attempt = journal.new_attempt("step_000100")
            artifact = attempt / "model.txt"
            artifact.write_text("saved")
            journal.finish("step_000100", {"step": 100}, [artifact])
            journal.new_attempt("step_000200")
            self.assertEqual(latest_checkpoint(journal, 250, 100)["step"], 100)
            artifact.write_text("corrupted")
            with self.assertRaises(RuntimeError):
                latest_checkpoint(journal, 250, 100)

    def test_training_comparison_rejects_wrong_data_or_language(self):
        schedule = build_schedule(self.episodes(), 10, 0, 1, 5)
        def logs(augmented):
            return [{**{k: r[k] for k in ("step", "episode_index", "frame_index")},
                     "variant_index": r["augmented_variant"] if augmented else 0,
                     "data_hash": "data", "noise_hash": "noise", "rng_before": "rng", "lr_used": 0.0001} for r in schedule]
        left, right = logs(False), logs(True)
        self.assertEqual(compare_training_logs(left, right, schedule)["different_noise_or_rng_steps"], [])
        right[0]["noise_hash"] = "different"
        self.assertEqual(compare_training_logs(left, right, schedule)["different_noise_or_rng_steps"], [1])
        right[0]["data_hash"] = "wrong frame"
        with self.assertRaises(ValueError):
            compare_training_logs(left, right, schedule)


if __name__ == "__main__":
    unittest.main()
