"""Verification tests for anchored_language training setup."""

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class AnchoredTrainingSetupTests(unittest.TestCase):
    def test_config_and_provenance(self):
        config_path = ROOT / "configs/anchored_training.json"
        self.assertTrue(config_path.exists())
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["arm"], "anchored_language")
        self.assertEqual(config["default_updates"], 1000)
        self.assertEqual(config["default_checkpoint_every"], 500)
        self.assertEqual(config["default_seed"], 0)
        self.assertEqual(config["language_seed"], 20260909)

        # Check reference run exists
        ref_path = ROOT / config["reference_run"]
        self.assertTrue(ref_path.exists())
        ref_exp = json.loads((ref_path / "experiment.json").read_text(encoding="utf-8"))
        self.assertEqual(ref_exp["updates"], 1000)
        self.assertEqual(ref_exp["seed"], 0)

    def test_anchored_schedule_matches_reference_frames(self):
        from smolvla_grounding.adaptation_records import build_anchored_schedule
        config = json.loads((ROOT / "configs/anchored_training.json").read_text(encoding="utf-8"))
        ref_schedule = json.loads((ROOT / config["reference_run"] / "schedule.json").read_text(encoding="utf-8"))
        
        # Load training split
        ready = json.loads((ROOT / "outputs/task0_data/ready.json").read_text(encoding="utf-8"))
        training = [r for r in ready["episodes"] if r["episode_index"] in ready["selection"]["split"]["train"]]
        self.assertEqual(len(training), 36)

        schedule = build_anchored_schedule(training, 1000, 0, 20260909, 5)
        self.assertEqual(len(schedule), 1000)

        # Frame sequence must match task0_longer_training step for step
        ref_frames = [(r["step"], r["episode_index"], r["frame_index"]) for r in ref_schedule[:1000]]
        anc_frames = [(r["step"], r["episode_index"], r["frame_index"]) for r in schedule]
        self.assertEqual(anc_frames, ref_frames)

        # Language distribution: P(0) = 0.50, P(1..4) = 0.125
        counts = {i: sum(r["anchored_variant"] == i for r in schedule) for i in range(5)}
        self.assertEqual(counts[0], 500)
        for i in range(1, 5):
            self.assertEqual(counts[i], 125)


if __name__ == "__main__":
    unittest.main()
