"""Check scoring and evidence contracts without loading GPU or simulator packages."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.protocol import atomic_json, file_sha256, fingerprint
from smolvla_grounding.rollout_records import export_episode_table, summarize_goals, verify_pairing


class GoalSummaryTests(unittest.TestCase):
    def test_transient_original_success_is_not_final_or_changed_goal_success(self):
        timeline = [{"original_bowl": False, "other_bowl": False},
                    {"original_bowl": True, "other_bowl": False},
                    {"original_bowl": False, "other_bowl": False}]
        result = summarize_goals(timeline, ["original_bowl", "other_bowl"])
        self.assertEqual(result["original_bowl"], {"ever": True, "final": False, "first_step": 2})
        self.assertEqual(result["other_bowl"], {"ever": False, "final": False, "first_step": None})

    def test_empty_rollout_is_not_a_failure_sample(self):
        with self.assertRaises(ValueError):
            summarize_goals([], ["original_bowl"])

    def test_csv_keeps_denominators_and_instruction_goals_separate(self):
        goals = {"original_bowl": {"ever": True, "final": False, "first_step": 1},
                 "other_bowl": {"ever": False, "final": False, "first_step": None}}
        results = [{"instruction_id": kind, "condition": kind, "goals": goals,
                    "goal_id": goal} for kind, goal in
                   (("original_repeat", "original_bowl"), ("semantic_change", "other_bowl"))]
        with tempfile.TemporaryDirectory() as temporary:
            export_episode_table(temporary, results, 6)
            with (Path(temporary) / "summary.csv").open(newline="") as stream:
                table = list(csv.DictReader(stream))
            self.assertEqual(table[0]["excluded_from_baseline_denominator"], "True")
            self.assertEqual(table[1]["requested_goal_ever"], "False")
            self.assertEqual(table[1]["original_goal_ever"], "True")
            summary = json.loads((Path(temporary) / "summary.json").read_text())
            self.assertFalse(summary["all_trials_complete"])
            self.assertFalse(summary["standard_evaluation"])


class EvidenceTests(unittest.TestCase):
    def test_changed_source_cannot_reuse_old_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            source.write_text("version = 1\n")
            config = {"pairing_output": "pairing"}
            identity = {"config": config, "instructions": [],
                        "project_sources": {"source.py": file_sha256(source)}}
            atomic_json(root / "pairing/identity.json", identity)
            atomic_json(root / "report.json", {"passed": True, "fingerprint": fingerprint(identity)})
            source.write_text("version = 2\n")
            with self.assertRaisesRegex(RuntimeError, "source changed"):
                verify_pairing(root, config, [], root / "report.json")


if __name__ == "__main__":
    unittest.main()
