import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_learning_rate_comparison import require_reference_controls


class LearningRateControlsTests(unittest.TestCase):
    def setUp(self):
        self.previous = [{"step": 1, "episode_index": 1272, "frame_index": 5, "variant_index": 2,
                          "data_hash": "frames", "rng_before": "rng", "noise_hash": "noise", "lr_used": 1e-4}]

    def test_rate_and_outcome_may_change(self):
        current = copy.deepcopy(self.previous)
        current[0].update(lr_used=1e-5, loss=0.2)
        require_reference_controls(self.previous, current)

    def test_unmatched_training_controls_rejected(self):
        for key in ("frame_index", "variant_index", "data_hash", "rng_before", "noise_hash"):
            current = copy.deepcopy(self.previous)
            current[0][key] = "different"
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                require_reference_controls(self.previous, current)

    def test_incomplete_history_rejected(self):
        with self.assertRaises(RuntimeError):
            require_reference_controls(self.previous, [])


if __name__ == "__main__":
    unittest.main()
