"""Pure runtime-scope regression tests; no LIBERO state or policy is constructed."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding import benchmark_runtime
from tests.final_v4_fixture import candidate_v4_freeze
from smolvla_grounding.benchmark_runtime import (
    verify_scoped_confirmatory_pairing, verify_final_confirmatory_pairing,
)
from smolvla_grounding.protocol import file_sha256


class FinalRuntimePairingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        baseline = json.loads((ROOT / "outputs/language_pilot/identity.json").read_text(encoding="utf-8"))
        cls.base_config = baseline["config"]
        cls.final_matrix = json.loads((ROOT / "configs/final_confirmatory/final_matrix_v4.json").read_text(encoding="utf-8"))
        cls.final_manifest = json.loads((ROOT / "configs/final_confirmatory/frozen_model_manifest_v4.json").read_text(encoding="utf-8"))
        cls.final_rows = [json.loads(line) for line in
                          (ROOT / "configs/final_confirmatory/instructions/task0_final.jsonl").read_text(encoding="utf-8").splitlines()
                          if line.strip()]
        cls.primary_matrix = json.loads((ROOT / "configs/held_out_evaluation/confirmatory_matrix.json").read_text(encoding="utf-8"))
        cls.primary_manifest = json.loads((ROOT / "configs/held_out_evaluation/frozen_training_manifest.json").read_text(encoding="utf-8"))
        cls.primary_rows = [json.loads(line) for line in
                            (ROOT / "configs/held_out_evaluation/instructions/task0_heldout.jsonl").read_text(encoding="utf-8").splitlines()
                            if line.strip()]
        cls._temporary_freeze = tempfile.TemporaryDirectory()
        candidate_path = Path(cls._temporary_freeze.name) / "candidate_v4.json"
        candidate_path.write_text(json.dumps(candidate_v4_freeze(ROOT, cls.final_manifest)), encoding="utf-8")
        cls._freeze_patch = patch.object(benchmark_runtime, "FINAL_CONFIRMATORY_FREEZE_PATH", candidate_path)
        cls._freeze_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls._freeze_patch.stop()
        cls._temporary_freeze.cleanup()

    def final_fixture(self):
        config = deepcopy(self.base_config)
        config["init_state_index"] = 33  # Only a synthetic integer; no state is loaded.
        config["policy"] = deepcopy(self.final_matrix["policy"])
        spec = {
            "matrix": deepcopy(self.final_matrix),
            "manifest": deepcopy(self.final_manifest),
            "model_id": "original_language",
        }
        return config, deepcopy(self.final_rows[0]), spec

    def primary_fixture(self):
        config = deepcopy(self.base_config)
        config["init_state_index"] = 3
        spec = {
            "matrix": deepcopy(self.primary_matrix),
            "manifest": deepcopy(self.primary_manifest),
            "model_id": "original_language",
        }
        return config, deepcopy(self.primary_rows[0]), spec

    def test_historical_primary_scope_still_passes(self):
        config, row, spec = self.primary_fixture()
        result = verify_scoped_confirmatory_pairing(ROOT, config, [row], spec)
        self.assertIn("bddl_sha256", result)

    def test_exact_final_scope_passes_without_environment_or_policy(self):
        config, row, spec = self.final_fixture()
        result = verify_scoped_confirmatory_pairing(ROOT, config, [row], spec)
        self.assertIn("init_file_sha256", result)

    def test_primary_and_final_roles_cannot_be_substituted(self):
        config, row, spec = self.final_fixture()
        spec["matrix"]["role"] = "primary_confirmatory_experiment"
        with self.assertRaises(RuntimeError):
            verify_scoped_confirmatory_pairing(ROOT, config, [row], spec)
        config, row, spec = self.primary_fixture()
        spec["matrix"]["role"] = "final_confirmatory_evaluation"
        with self.assertRaises(RuntimeError):
            verify_scoped_confirmatory_pairing(ROOT, config, [row], spec)
        config, row, spec = self.final_fixture()
        spec["matrix"]["role"] = "arbitrary_confirmatory_role"
        with self.assertRaises(RuntimeError):
            verify_scoped_confirmatory_pairing(ROOT, config, [row], spec)

    def test_final_state_prompt_model_horizon_seed_worker_and_metric_mutations_fail(self):
        mutations = (
            lambda c, r, s: c.update(init_state_index=32),
            lambda c, r, s: s["matrix"]["tasks"][0].update(initial_states=list(range(32, 52))),
            lambda c, r, s: r.update(text="changed final prompt"),
            lambda c, r, s: s["matrix"]["tasks"][0]["prompt_text_sha256"].update(paraphrase_final_01="bad"),
            lambda c, r, s: s.update(model_id="paired_flow_consistency"),
            lambda c, r, s: s["matrix"].update(models=["original_language", "augmented_language", "paired_flow_consistency"]),
            lambda c, r, s: c["environment"].update(episode_length=279),
            lambda c, r, s: s["matrix"]["environment"].update(episode_length=279),
            lambda c, r, s: c.update(environment_seed=1),
            lambda c, r, s: c.update(policy_seed=1),
            lambda c, r, s: s["matrix"]["execution"].update(default_workers=2),
            lambda c, r, s: s["matrix"]["statistical_rules"].update(primary_metric="goal_final"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                config, row, spec = self.final_fixture()
                mutation(config, row, spec)
                with self.assertRaises(RuntimeError):
                    verify_final_confirmatory_pairing(ROOT, config, [row], spec)

    def test_final_matrix_and_checkpoint_hash_mutations_fail(self):
        config, row, spec = self.final_fixture()
        matrix_path = ROOT / "configs/final_confirmatory/final_matrix_v4.json"
        checkpoint_path = ROOT / spec["manifest"]["evaluated_models"]["original_language"]["checkpoint"]
        def wrong_matrix(path):
            return "0" * 64 if Path(path) == matrix_path else file_sha256(path)
        with patch("smolvla_grounding.benchmark_runtime.file_sha256", side_effect=wrong_matrix):
            with self.assertRaisesRegex(RuntimeError, "matrix or manifest SHA256"):
                verify_final_confirmatory_pairing(ROOT, config, [row], spec)
        def wrong_checkpoint(path):
            return "0" * 64 if Path(path) == checkpoint_path / "pretrained_model/model.safetensors" else file_sha256(path)
        with patch("smolvla_grounding.benchmark_runtime.file_sha256", side_effect=wrong_checkpoint):
            with self.assertRaisesRegex(RuntimeError, "Final checkpoint SHA256 mismatch"):
                verify_final_confirmatory_pairing(ROOT, config, [row], spec)


if __name__ == "__main__":
    unittest.main()