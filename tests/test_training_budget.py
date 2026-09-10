from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_training_budget import longer_schedule
from smolvla_grounding.adaptation_records import build_schedule


class TrainingBudgetTests(unittest.TestCase):
    def test_extension_preserves_sampling_prefix(self):
        episodes = [{"episode_index": 7, "frames": 20}, {"episode_index": 9, "frames": 10}]
        old = build_schedule(episodes, 10, 0, 42, 5)
        new = longer_schedule(episodes, 50, 0, 42, 5, old)
        self.assertEqual(new[:10], old)
        self.assertEqual(len(new), 50)
        self.assertTrue(all(r['episode_index'] in (7, 9) for r in new))

    def test_changed_seed_or_nonextension_rejected(self):
        episodes = [{"episode_index": 7, "frames": 20}]
        old = build_schedule(episodes, 10, 0, 42, 5)
        with self.assertRaises(ValueError):
            longer_schedule(episodes, 10, 0, 42, 5, old)
        with self.assertRaises(RuntimeError):
            longer_schedule(episodes, 50, 1, 42, 5, old)


if __name__ == "__main__":
    unittest.main()
