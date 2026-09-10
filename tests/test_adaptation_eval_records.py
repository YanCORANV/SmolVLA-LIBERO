import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smolvla_grounding.adaptation_eval_records import MODELS, compare_models


class AdaptationComparisonTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": n, "condition": c, "goal_id": "original_bowl"} for n, c in (
            ("original", "original"), ("original_repeat", "original_repeat"),
            ("paraphrase_01", "paraphrase"), ("paraphrase_02", "paraphrase"))]
        self.experiment = {"initial_states": [0, 1], "policy_seeds": [0], "horizon": 280,
                           "environment_seed": 0, "suite": "libero_spatial", "task_id": 0}
        self.results = {m: [] for m in MODELS}
        for model in MODELS:
            for state in (0, 1):
                for row in self.rows:
                    success = row["condition"] in {"original", "original_repeat"} or model == "augmented_language"
                    self.results[model].append({"initial_state_index": state, "policy_seed": 0,
                        "instruction_id": row["id"], "condition": row["condition"], "goal_id": "original_bowl",
                        "initial_hashes": {"physics": state}, "initial_rng_hashes": {"rng": 0},
                        "policy_steps": 280, "environment_seed": 0, "suite": "libero_spatial", "task_id": 0,
                        "goals": {"original_bowl": {"ever": success, "final": False}}})

    def test_augmentation_delta_and_repeat_exclusion(self):
        r = compare_models(self.results, self.experiment, self.rows)
        self.assertTrue(r["all_complete"])
        self.assertEqual(len(r["paired_cases"]), 6)
        self.assertEqual(r["paired_differences"]["paraphrase_01"]["augmented_minus_original_tuning_ever"], 1)
        self.assertEqual(r["paired_differences"]["paraphrase_01"]["augmented_minus_original_tuning_final"], 0)
        self.assertNotIn("original_repeat", r["paired_differences"])

    def test_missing_episode_removes_group_from_all_denominators(self):
        self.results["augmented_language"].pop()
        r = compare_models(self.results, self.experiment, self.rows)
        self.assertFalse(r["all_complete"])
        self.assertEqual(r["completed_matched_groups"], 1)
        for m in MODELS:
            self.assertEqual(r["models"][m]["per_instruction"]["original"]["episodes"], 1)

    def test_zero_complete_groups_are_not_failures(self):
        self.results["original_language"] = []
        r = compare_models(self.results, self.experiment, self.rows)
        self.assertEqual(r["paired_cases"], [])
        self.assertIsNone(r["models"]["untuned_fp32"]["per_instruction"]["original"]["requested_goal_ever_fraction"])

    def test_mismatched_physics_rng_or_horizon_rejected(self):
        for field, wrong in (("initial_hashes", {"physics": 99}), ("initial_rng_hashes", {"rng": 9}), ("policy_steps", 279)):
            with self.subTest(field=field):
                results = copy.deepcopy(self.results)
                results["original_language"][0][field] = wrong
                with self.assertRaises(ValueError):
                    compare_models(results, self.experiment, self.rows)

    def test_duplicate_episode_rejected(self):
        self.results["untuned_fp32"].append(self.results["untuned_fp32"][0])
        with self.assertRaises(ValueError):
            compare_models(self.results, self.experiment, self.rows)


if __name__ == "__main__":
    unittest.main()
