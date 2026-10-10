"""Unit and synthetic tests for confirmatory held-out language robustness protocol."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from copy import deepcopy
import json
import unittest
import numpy as np

from smolvla_grounding.held_out_summary import parse_and_aggregate_records, compute_paired_cluster_bootstrap
from smolvla_grounding.held_out_evaluator import preflight_confirmatory_validation


class TestConfirmatoryStatistics(unittest.TestCase):

    def setUp(self):
        self.models = ["untuned_fp32", "original_language", "augmented_language"]
        self.states = list(range(3, 23))  # 20 states
        self.instructions = ["original", "original_repeat", "paraphrase_heldout_01", "paraphrase_heldout_02"]

    def _create_synthetic_episodes(self, custom_rule_fn):
        """Generate 240 synthetic episode records according to a custom rule."""
        episodes = []
        for s in self.states:
            for m in self.models:
                for inst in self.instructions:
                    ever_val, final_val = custom_rule_fn(s, m, inst)
                    episodes.append({
                        "model": m,
                        "task_id": 0,
                        "initial_state_index": s,
                        "instruction_id": inst,
                        "instruction": f"text for {inst}",
                        "policy_steps": 280,
                        "goal_ever": ever_val,
                        "goal_final": final_val,
                        "first_step": 50 if ever_val else None,
                        "video_main": "dummy_main.mp4",
                        "video_wrist": "dummy_wrist.mp4"
                    })
        return episodes

    def test_goal_ever_drives_primary_success(self):
        """Prove that goal_ever, NOT goal_final, determines primary success rates."""
        def rule_touched_then_dropped(s, m, inst):
            return True, False

        episodes = self._create_synthetic_episodes(rule_touched_then_dropped)
        results = parse_and_aggregate_records(episodes, self.models)

        for m in self.models:
            pe = results["primary_point_estimates_goal_ever"][m]
            self.assertEqual(pe["success_rate_original"], 1.0)
            self.assertEqual(pe["success_rate_paraphrase_01"], 1.0)
            self.assertEqual(pe["success_rate_paraphrase_02"], 1.0)
            self.assertEqual(pe["pooled_unseen_paraphrase_success_rate"], 1.0)

            diag = results["secondary_persistence_diagnostics_goal_final"][m]
            self.assertEqual(diag["final_step_success_rate_original"], 0.0)
            self.assertEqual(diag["final_step_pooled_paraphrase_rate"], 0.0)

    def test_repeat_never_in_success_rate_denominator(self):
        """Prove that original_repeat never enters primary success rate denominators."""
        def rule_repeat_diff(s, m, inst):
            if inst == "original":
                return (s == 3), (s == 3)
            elif inst == "original_repeat":
                return True, True
            elif inst in ("paraphrase_heldout_01", "paraphrase_heldout_02"):
                return (s in (3, 4)), (s in (3, 4))
            return False, False

        episodes = self._create_synthetic_episodes(rule_repeat_diff)
        results = parse_and_aggregate_records(episodes, self.models)

        for m in self.models:
            pe = results["primary_point_estimates_goal_ever"][m]
            # Denominator for original must be 20, numerator 1 -> 1/20 = 0.05
            self.assertAlmostEqual(pe["success_rate_original"], 0.05)
            self.assertEqual(pe["success_rate_original_count"], "1/20")

            # Repeat success rate is diagnostic only (20/20 = 1.0)
            self.assertAlmostEqual(pe["repeat_diagnostic_success_rate"], 1.0)
            # Repeat disagreement: 19 states disagreed
            self.assertEqual(pe["repeat_disagreement_count"], 19)
            self.assertAlmostEqual(pe["repeat_disagreement_rate"], 19 / 20)

            # Pooled paraphrase denominator must be 2*20 = 40 (only the 2 paraphrases, repeat is excluded)
            self.assertAlmostEqual(pe["pooled_unseen_paraphrase_success_rate"], 0.10)
            self.assertEqual(pe["pooled_unseen_paraphrase_count"], "4/40")

    def test_paired_bootstrap_resamples_whole_state_clusters(self):
        """Prove that bootstrap resamples states as complete clusters."""
        states_data = {}
        for s in self.states:
            states_data[s] = {}
            for m in self.models:
                states_data[s][m] = {
                    "original": 1 if s == 3 else 0,
                    "original_repeat": 1 if s == 3 else 0,
                    "paraphrase_heldout_01": 1 if s == 3 else 0,
                    "paraphrase_heldout_02": 1 if s == 3 else 0
                }

        ci_results = compute_paired_cluster_bootstrap(states_data, self.models, num_replicates=500, seed=123)

        mean_orig_untuned = ci_results["untuned_fp32_orig"]["mean"]
        mean_orig_aug = ci_results["augmented_language_orig"]["mean"]
        self.assertAlmostEqual(mean_orig_untuned, mean_orig_aug)
        self.assertAlmostEqual(mean_orig_untuned, 1.0 / 20.0, places=2)

    def test_all_three_models_remain_paired_under_bootstrap(self):
        """Prove that paired difference augmented - original is strictly paired per resampled cluster."""
        states_data = {}
        for s in self.states:
            states_data[s] = {
                "untuned_fp32": {"original": 0, "original_repeat": 0, "paraphrase_heldout_01": 0, "paraphrase_heldout_02": 0},
                "original_language": {"original": 0, "original_repeat": 0, "paraphrase_heldout_01": 0, "paraphrase_heldout_02": 0},
                "augmented_language": {"original": 1, "original_repeat": 1, "paraphrase_heldout_01": 1, "paraphrase_heldout_02": 1},
            }

        ci_results = compute_paired_cluster_bootstrap(states_data, self.models, num_replicates=500, seed=99)

        # Difference must be identically 1.0 across all resamples
        self.assertAlmostEqual(ci_results["paired_paraphrase_diff"]["mean"], 1.0)
        self.assertAlmostEqual(ci_results["paired_paraphrase_diff"]["std"], 0.0)
        self.assertEqual(ci_results["paired_paraphrase_diff"]["ci_95"], [1.0, 1.0])

    def test_preflight_assertions(self):
        """Test that preflight_confirmatory_validation catches illegal setups."""
        root = ROOT
        matrix_path = root / "configs/held_out_evaluation/confirmatory_matrix.json"
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        manifest = json.loads((root / matrix["manifest_path"]).read_text(encoding="utf-8"))

        # 1. Valid setup should pass
        preflight_confirmatory_validation(root, matrix, manifest)

        # 2. Horizon != 280 must fail
        bad_matrix_horizon = deepcopy(matrix)
        bad_matrix_horizon["environment"]["episode_length"] = 240
        with self.assertRaises(RuntimeError) as ctx:
            preflight_confirmatory_validation(root, bad_matrix_horizon, manifest)
        self.assertIn("Rollout horizon must be exactly 280", str(ctx.exception))

        # 3. State overlap must fail
        bad_matrix_state = deepcopy(matrix)
        bad_matrix_state["tasks"][0]["initial_states"] = [0, 3, 4]  # includes dev state 0
        with self.assertRaises(RuntimeError) as ctx:
            preflight_confirmatory_validation(root, bad_matrix_state, manifest)
        self.assertIn("overlap with development states", str(ctx.exception))

        # 4. Hash mismatch must fail
        bad_manifest = deepcopy(manifest)
        bad_manifest["evaluated_models"]["untuned_fp32"]["artifact_sha256"]["config.json"] = "deadbeef" * 8
        with self.assertRaises(RuntimeError) as ctx:
            preflight_confirmatory_validation(root, matrix, bad_manifest)
        self.assertIn("SHA256 mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
