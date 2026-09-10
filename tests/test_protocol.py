"""Dependency-free regression checks for invalid labels and unsafe continuation."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.protocol import Checkpoints, load_protocol, validate_catalog


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.config, self.rows = load_protocol(ROOT)

    def test_real_catalog_has_six_distinct_trials(self):
        self.assertEqual(len(self.rows), 6)

    def test_paraphrase_cannot_change_goal(self):
        self.rows[2]["goal_id"] = "ramekin"
        with self.assertRaisesRegex(ValueError, "preserve"):
            validate_catalog(self.rows, self.config["goals"])

    def test_changed_instruction_cannot_keep_original_goal(self):
        self.rows[-1]["goal_id"] = "original_bowl"
        with self.assertRaisesRegex(ValueError, "change the goal"):
            validate_catalog(self.rows, self.config["goals"])

    def test_repeat_cannot_silently_change_words(self):
        self.rows[1]["text"] = "put it on the plate"
        with self.assertRaisesRegex(ValueError, "identical"):
            validate_catalog(self.rows, self.config["goals"])

    def test_duplicate_trial_id_rejected(self):
        self.rows.append(deepcopy(self.rows[0]))
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_catalog(self.rows, self.config["goals"])


class CheckpointTests(unittest.TestCase):
    def test_incomplete_attempt_preserved_and_completed_artifacts_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            checkpoints = Checkpoints(directory, {"seed": 0})
            first = checkpoints.new_attempt("original")
            self.assertIsNone(checkpoints.completed("original"))
            second = checkpoints.new_attempt("original")
            self.assertTrue(first.exists())
            self.assertNotEqual(first, second)
            artifact = second / "evidence.json"
            artifact.write_text(json.dumps({"state": [1, 2, 3]}))
            checkpoints.finish("original", {"passed": True}, [artifact])
            with patch("builtins.input", return_value="y"):
                resumed = Checkpoints(directory, {"seed": 0})
            self.assertEqual(resumed.completed("original"), {"passed": True})
            artifact.write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "modified"):
                resumed.completed("original")

    def test_different_provenance_rejected_before_resume_question(self):
        with tempfile.TemporaryDirectory() as temporary:
            Checkpoints(temporary, {"seed": 0})
            with patch("builtins.input") as question:
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    Checkpoints(temporary, {"seed": 1})
                question.assert_not_called()

    def test_declining_resume_preserves_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            Checkpoints(temporary, {"seed": 0})
            identity = Path(temporary) / "identity.json"
            before = identity.read_bytes()
            with patch("builtins.input", return_value="n"):
                with self.assertRaisesRegex(RuntimeError, "preserved"):
                    Checkpoints(temporary, {"seed": 0})
            self.assertEqual(before, identity.read_bytes())


if __name__ == "__main__":
    unittest.main()
