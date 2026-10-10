"""Regression and scope-aware pairing validation tests for confirmatory evaluation."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.rollout_records import verify_pairing
from smolvla_grounding.benchmark_runtime import (
    verify_confirmatory_pairing,
    FROZEN_CONFIRMATORY_INSTRUCTIONS,
    FROZEN_CONFIRMATORY_MODELS,
    FROZEN_CONFIRMATORY_STATES
)


class TestConfirmatoryRuntimePairing(unittest.TestCase):

    def setUp(self):
        self.root = ROOT
        self.pilot_identity = json.loads((self.root / "outputs/language_pilot/identity.json").read_text(encoding="utf-8"))
        self.matrix = json.loads((self.root / "configs/held_out_evaluation/confirmatory_matrix.json").read_text(encoding="utf-8"))
        self.manifest = json.loads((self.root / "configs/held_out_evaluation/frozen_training_manifest.json").read_text(encoding="utf-8"))

        self.confirmatory_spec = {
            "matrix": self.matrix,
            "manifest": self.manifest,
            "model_id": "untuned_fp32"
        }

        # Build a valid confirmatory config based on Task 0 state 3 and held-out paraphrase 01
        self.valid_confirmatory_config = deepcopy(self.pilot_identity["config"])
        self.valid_confirmatory_config["task_id"] = 0
        self.valid_confirmatory_config["suite"] = "libero_spatial"
        self.valid_confirmatory_config["init_state_index"] = 3
        self.valid_confirmatory_config["instruction_catalog"] = "configs/held_out_evaluation/instructions/task0_heldout.jsonl"
        self.valid_confirmatory_config["environment"]["episode_length"] = 280

        self.valid_heldout_instruction = {
            "id": "paraphrase_heldout_01",
            "condition": "paraphrase",
            "text": "transfer into the plate the black bowl that sits between the plate and the ramekin",
            "goal_id": "task_goal"
        }

    def test_reproduce_bug_historical_pilot_mode_fails_on_confirmatory_config(self):
        """Reproduce the bug: historical pilot verify_pairing MUST reject confirmatory config/instruction."""
        report_path = self.root / self.pilot_identity["run_config"]["pairing_report"]
        with self.assertRaisesRegex(RuntimeError, "Pilot configuration/catalog differs from the passed pairing check"):
            verify_pairing(
                self.root,
                self.valid_confirmatory_config,
                [self.valid_heldout_instruction],
                report_path
            )

    def test_pilot_mode_preserves_historical_verification(self):
        """Historical pilot validation succeeds unchanged on original pilot config and instructions."""
        report_path = self.root / self.pilot_identity["run_config"]["pairing_report"]
        report, identity = verify_pairing(
            self.root,
            self.pilot_identity["config"],
            self.pilot_identity["instructions"],
            report_path
        )
        self.assertTrue(report["passed"])
        self.assertEqual(identity["config"]["task_id"], 0)
        self.assertEqual(identity["config"]["init_state_index"], 0)

    def test_confirmatory_pairing_succeeds_on_valid_matrix_and_states(self):
        """Requirement 6: Confirmatory runtime initializes successfully when matching frozen confirmatory matrix."""
        # Test all 4 valid frozen instructions on state 3
        for inst_id, text in FROZEN_CONFIRMATORY_INSTRUCTIONS.items():
            row = {"id": inst_id, "text": text}
            pairing = verify_confirmatory_pairing(
                self.root,
                self.valid_confirmatory_config,
                [row],
                self.confirmatory_spec
            )
            self.assertIn("upstream_sources", pairing)
            self.assertIn("bddl_sha256", pairing)
            self.assertIn("init_file_sha256", pairing)

        # Test across boundaries of states 3-22
        for state_idx in [3, 10, 22]:
            cfg = deepcopy(self.valid_confirmatory_config)
            cfg["init_state_index"] = state_idx
            pairing = verify_confirmatory_pairing(
                self.root,
                cfg,
                [self.valid_heldout_instruction],
                self.confirmatory_spec
            )
            self.assertIsNotNone(pairing)

    def test_negative_cases_reject_invalid_confirmatory_spec(self):
        """Requirement 7: Changing state, instruction, horizon, seeds, etc. must cause failure."""
        row = deepcopy(self.valid_heldout_instruction)

        # 1. State in dev states (0, 1, 2)
        for dev_s in [0, 1, 2]:
            cfg = deepcopy(self.valid_confirmatory_config)
            cfg["init_state_index"] = dev_s
            with self.assertRaisesRegex(RuntimeError, "overlaps development states|not in frozen states"):
                verify_confirmatory_pairing(self.root, cfg, [row], self.confirmatory_spec)

        # 2. State out of bounds
        for bad_s in [-1, 23, 100]:
            cfg = deepcopy(self.valid_confirmatory_config)
            cfg["init_state_index"] = bad_s
            with self.assertRaisesRegex(RuntimeError, "not in frozen states"):
                verify_confirmatory_pairing(self.root, cfg, [row], self.confirmatory_spec)

        # 3. Instruction text mismatch
        cfg = deepcopy(self.valid_confirmatory_config)
        bad_text_row = {"id": "paraphrase_heldout_01", "text": "altered wording that does not match frozen catalog"}
        with self.assertRaisesRegex(RuntimeError, "differs from frozen text"):
            verify_confirmatory_pairing(self.root, cfg, [bad_text_row], self.confirmatory_spec)

        # 4. Unknown instruction ID
        bad_id_row = {"id": "paraphrase_unknown", "text": "transfer into the plate the black bowl that sits between the plate and the ramekin"}
        with self.assertRaisesRegex(RuntimeError, "not in frozen specification"):
            verify_confirmatory_pairing(self.root, cfg, [bad_id_row], self.confirmatory_spec)

        # 5. Horizon != 280
        for bad_h in [240, 300]:
            cfg = deepcopy(self.valid_confirmatory_config)
            cfg["environment"]["episode_length"] = bad_h
            with self.assertRaisesRegex(RuntimeError, "episode_length must be 280"):
                verify_confirmatory_pairing(self.root, cfg, [row], self.confirmatory_spec)

        # 6. Task ID != 0
        cfg = deepcopy(self.valid_confirmatory_config)
        cfg["task_id"] = 1
        with self.assertRaisesRegex(RuntimeError, "task ID must be 0"):
            verify_confirmatory_pairing(self.root, cfg, [row], self.confirmatory_spec)

        # 7. Seeds != 0
        cfg = deepcopy(self.valid_confirmatory_config)
        cfg["environment_seed"] = 1
        with self.assertRaisesRegex(RuntimeError, "Environment seed must be strictly 0"):
            verify_confirmatory_pairing(self.root, cfg, [row], self.confirmatory_spec)

        cfg = deepcopy(self.valid_confirmatory_config)
        cfg["policy_seed"] = 1
        with self.assertRaisesRegex(RuntimeError, "Policy seed must be strictly 0"):
            verify_confirmatory_pairing(self.root, cfg, [row], self.confirmatory_spec)

        # 8. Unregistered / non-frozen model
        spec = deepcopy(self.confirmatory_spec)
        spec["model_id"] = "non_existent_model"
        with self.assertRaisesRegex(RuntimeError, "not one of the frozen models"):
            verify_confirmatory_pairing(self.root, self.valid_confirmatory_config, [row], spec)

        # 9. Preprocessing mismatch (e.g. observation dimensions)
        cfg = deepcopy(self.valid_confirmatory_config)
        cfg["environment"]["observation_height"] = 128
        with self.assertRaisesRegex(RuntimeError, "Observation size must be 256x256"):
            verify_confirmatory_pairing(self.root, cfg, [row], self.confirmatory_spec)

        cfg = deepcopy(self.valid_confirmatory_config)
        cfg["environment"]["control_freq"] = 10
        with self.assertRaisesRegex(RuntimeError, "Control configuration must be 20Hz relative"):
            verify_confirmatory_pairing(self.root, cfg, [row], self.confirmatory_spec)


if __name__ == "__main__":
    unittest.main()
