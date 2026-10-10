"""Scope and gate regression tests without opening any LIBERO state."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.paired_development_evaluator import (
    ARMS, PROMPTS, STATES, apply_gate, build_matrix, validate_scope,
)


class PairedDevelopmentEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paired = json.loads((ROOT / "configs/paired_training.json").read_text(encoding="utf-8"))

    def test_frozen_60_rollout_scope(self):
        matrix = build_matrix(ROOT, self.paired)
        self.assertEqual(matrix["models"], list(ARMS))
        self.assertEqual(matrix["tasks"][0]["initial_states"], list(STATES))
        self.assertEqual([row["id"] for row in validate_scope(ROOT, matrix, self.paired)], list(PROMPTS))
        self.assertEqual(matrix["statistical_rules"]["total_rollouts"], 60)

    def test_reject_future_state_non_dev_prompt_or_changed_runtime(self):
        matrix = build_matrix(ROOT, self.paired)
        for key, value in (("initial_states", list(range(23, 32)) + [33]),):
            bad = deepcopy(matrix)
            bad["tasks"][0][key] = value
            with self.assertRaises(RuntimeError):
                validate_scope(ROOT, bad, self.paired)
        bad = deepcopy(matrix)
        bad["tasks"][0]["instruction_catalog"] = "configs/instructions/task0_train.jsonl"
        with self.assertRaises(RuntimeError):
            validate_scope(ROOT, bad, self.paired)
        for section, key, value in (("environment", "environment_seed", 1),
                                    ("environment", "episode_length", 100),
                                    ("policy", "policy_seed", 1),
                                    ("execution", "default_workers", 2)):
            bad = deepcopy(matrix)
            bad[section][key] = value
            with self.assertRaises(RuntimeError):
                validate_scope(ROOT, bad, self.paired)

    def test_frozen_gate(self):
        summary = {"results": {
            "paired_bc": {"O": 0.9, "P": 0.4, "H": 2 * 0.9 * 0.4 / 1.3},
            "paired_flow_consistency": {"O": 0.8, "P": 0.5, "H": 2 * 0.8 * 0.5 / 1.3},
        }}
        self.assertEqual(apply_gate(summary, self.paired)["model_selection_gate"]["status"], "PASS")
        failing = deepcopy(summary)
        failing["results"]["paired_flow_consistency"]["O"] = 0.7
        self.assertEqual(apply_gate(failing, self.paired)["model_selection_gate"]["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()