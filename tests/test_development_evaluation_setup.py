"""Unit tests for development model-selection evaluation setup and split guardrails."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.development_evaluator import (
    preflight_development_validation,
    validate_development_prompts,
)


class DevelopmentEvaluationSetupTests(unittest.TestCase):
    def setUp(self):
        self.matrix_path = ROOT / "configs/development_evaluation/dev_matrix.json"
        self.manifest_path = ROOT / "configs/development_evaluation/dev_manifest.json"
        self.matrix = json.loads(self.matrix_path.read_text(encoding="utf-8"))
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def test_strict_states_boundary_guardrails(self):
        # Must strictly be states 23-32
        self.assertEqual(self.matrix["tasks"][0]["initial_states"], list(range(23, 33)))

        # Historical dev state leakage (e.g. state 0)
        bad_matrix_hist = deepcopy(self.matrix)
        bad_matrix_hist["tasks"][0]["initial_states"] = [0] + list(range(24, 33))
        with self.assertRaises(RuntimeError) as ctx:
            preflight_development_validation(ROOT, bad_matrix_hist, self.manifest)
        self.assertIn("Historical dev states", str(ctx.exception))

        # Consumed test state leakage (e.g. state 3)
        bad_matrix_test = deepcopy(self.matrix)
        bad_matrix_test["tasks"][0]["initial_states"] = [3] + list(range(24, 33))
        with self.assertRaises(RuntimeError) as ctx:
            preflight_development_validation(ROOT, bad_matrix_test, self.manifest)
        self.assertIn("Consumed test states", str(ctx.exception))

        # Future confirmatory state leakage (e.g. state 33)
        bad_matrix_future = deepcopy(self.matrix)
        bad_matrix_future["tasks"][0]["initial_states"] = list(range(23, 32)) + [33]
        with self.assertRaises(RuntimeError) as ctx:
            preflight_development_validation(ROOT, bad_matrix_future, self.manifest)
        self.assertIn("Future confirmatory states", str(ctx.exception))

    def test_horizon_and_workers_guardrails(self):
        # Horizon must be 280
        self.assertEqual(self.matrix["environment"]["episode_length"], 280)
        bad_matrix = deepcopy(self.matrix)
        bad_matrix["environment"]["episode_length"] = 100
        with self.assertRaises(RuntimeError):
            preflight_development_validation(ROOT, bad_matrix, self.manifest)

        # Worker must be 1
        self.assertEqual(self.matrix["execution"]["default_workers"], 1)
        bad_matrix2 = deepcopy(self.matrix)
        bad_matrix2["execution"]["default_workers"] = 2
        with self.assertRaises(RuntimeError):
            preflight_development_validation(ROOT, bad_matrix2, self.manifest)

    def test_instructions_catalog(self):
        catalog_rel = self.matrix["tasks"][0]["instruction_catalog"]
        inst_path = ROOT / catalog_rel
        self.assertTrue(inst_path.exists())
        lines = [json.loads(line) for line in inst_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(lines), 3)
        self.assertEqual([r["id"] for r in lines], ["original", "paraphrase_dev_01", "paraphrase_dev_02"])

        # Call prompt validator and ensure zero errors
        validate_development_prompts(ROOT, catalog_rel)

    def test_prompt_validator_negative_guardrails(self):
        import tempfile

        # Case 1: duplicate of a training paraphrase
        with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as tmp:
            tmp.write(json.dumps({"id": "original", "text": "pick up the black bowl between the plate and the ramekin and place it on the plate"}) + "\n")
            tmp.write(json.dumps({"id": "paraphrase_dev_01", "text": "pick up the black bowl that sits between the plate and the ramekin, and put it on the plate"}) + "\n")
            tmp.write(json.dumps({"id": "paraphrase_dev_02", "text": "set onto the plate the black bowl situated between the ramekin and the plate"}) + "\n")
            tmp_path = Path(tmp.name)
        try:
            rel_path = tmp_path.relative_to(ROOT)
        except ValueError:
            # If tempfile is outside ROOT, test directly via tmp_path
            rel_path = tmp_path
        
        with self.assertRaises(RuntimeError) as ctx:
            validate_development_prompts(ROOT, rel_path)
        self.assertIn("duplicate detected", str(ctx.exception).lower())
        tmp_path.unlink()

        # Case 2: missing task-critical noun ('ramekin')
        with tempfile.NamedTemporaryFile("w+", suffix=".jsonl", delete=False) as tmp:
            tmp.write(json.dumps({"id": "original", "text": "pick up the black bowl between the plate and the ramekin and place it on the plate"}) + "\n")
            tmp.write(json.dumps({"id": "paraphrase_dev_01", "text": "grasp the black bowl positioned on the plate and put it onto the plate"}) + "\n")
            tmp.write(json.dumps({"id": "paraphrase_dev_02", "text": "set onto the plate the black bowl situated near the plate"}) + "\n")
            tmp_path = Path(tmp.name)
        try:
            rel_path = tmp_path.relative_to(ROOT)
        except ValueError:
            rel_path = tmp_path
        
        with self.assertRaises(RuntimeError) as ctx:
            validate_development_prompts(ROOT, rel_path)
        self.assertIn("missing task-critical nouns", str(ctx.exception).lower())
        tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
