"""Selection, checkpoint provenance and paired-denominator regression checks."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.benchmark_records import (
    BenchmarkJournal, group_summary, parse_indices, trial_id, validate_run_name, validate_experiment,
)
from smolvla_grounding.protocol import load_protocol


class SelectionTests(unittest.TestCase):
    def test_selection_is_explicit_unique_and_sorted(self):
        self.assertEqual(parse_indices("2, 0, 1"), [0, 1, 2])
        for invalid in ("0,0", "-1", "0,", "0.5", "", str(2**32)):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_indices(invalid)

    def test_run_name_cannot_escape_output_directory(self):
        self.assertEqual(validate_run_name("task0_states"), "task0_states")
        for invalid in ("../old", "a/b", "a\\b", "C:other", ""):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_run_name(invalid)

    def test_state_and_seed_are_part_of_checkpoint_key(self):
        keys = {trial_id(s, p, "original") for s in (0, 1) for p in (0, 1)}
        self.assertEqual(len(keys), 4)

    def test_fixed_settings_cannot_be_silently_changed_in_saved_experiment(self):
        config, rows = load_protocol(ROOT)
        baseline = {"config": config, "instructions": rows}
        experiment = {"environment_seed": config["environment_seed"], "suite": config["suite"], "task_id": config["task_id"],
                      "horizon": config["environment"]["episode_length"], "instruction_order": [r["id"] for r in rows],
                      "initial_states": [0, 1, 2], "policy_seeds": [0]}
        validate_experiment(experiment, baseline)
        experiment["environment_seed"] = 99
        with self.assertRaises(ValueError):
            validate_experiment(experiment, baseline)

    def test_benchmark_journal_rejects_mixed_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            first = BenchmarkJournal(directory, {"seed": 0})
            second = BenchmarkJournal(directory, {"seed": 0})
            self.assertEqual(first.fingerprint, second.fingerprint)
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                BenchmarkJournal(directory, {"seed": 1})


class SummaryTests(unittest.TestCase):
    def setUp(self):
        _, self.rows = load_protocol(ROOT)
        self.experiment = {"initial_states": [0, 1], "policy_seeds": [0]}

    def group(self, state, original=True, repeat=True, paraphrases=False):
        records = []
        for row in self.rows:
            success = original if row["condition"] == "original" else repeat if row["condition"] == "original_repeat" else paraphrases if row["condition"] == "paraphrase" else False
            goal = {"ever": success, "final": success, "first_step": 1 if success else None}
            records.append({"initial_state_index": state, "policy_seed": 0, "instruction_id": row["id"],
                            "condition": row["condition"], "goal_id": row["goal_id"],
                            "goals": {"original_bowl": deepcopy(goal), row["goal_id"]: deepcopy(goal)}})
        return records

    def test_incomplete_group_is_not_counted_as_failure(self):
        result = group_summary(self.group(0) + self.group(1)[:1], self.experiment, self.rows)
        self.assertEqual(result["completed_episodes"], 7)
        self.assertEqual(result["completed_groups"], 1)
        self.assertEqual(result["per_instruction"]["original"]["episodes"], 1)
        self.assertFalse(result["all_complete"])

    def test_repeat_is_separate_and_paired_gap_uses_complete_groups(self):
        records = self.group(0, original=True, repeat=False, paraphrases=False)
        records += self.group(1, original=False, repeat=False, paraphrases=True)
        result = group_summary(records, self.experiment, self.rows)
        self.assertEqual(result["per_instruction"]["original"]["requested_goal_ever_fraction"], 0.5)
        self.assertEqual(result["per_instruction"]["original_repeat"]["requested_goal_ever_fraction"], 0)
        self.assertEqual(result["mean_paired_paraphrase_minus_original"], 0)
        self.assertEqual(result["repeat_ever_disagreement_groups"], 1)

    def test_no_complete_groups_returns_unavailable_not_zero_rate(self):
        result = group_summary(self.group(0)[:1], self.experiment, self.rows)
        self.assertIsNone(result["per_instruction"]["original"]["requested_goal_ever_fraction"])
        self.assertIsNone(result["mean_paired_paraphrase_minus_original"])

    def test_duplicate_trials_and_wrong_goals_are_rejected(self):
        rows = self.group(0)
        with self.assertRaises(ValueError):
            group_summary(rows + rows[:1], self.experiment, self.rows)
        rows[-1]["goal_id"] = "original_bowl"
        with self.assertRaises(ValueError):
            group_summary(rows, self.experiment, self.rows)


if __name__ == "__main__":
    unittest.main()
