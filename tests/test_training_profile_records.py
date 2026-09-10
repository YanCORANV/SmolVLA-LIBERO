"""Reject false training acceptance without importing the GPU runtime."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.training_profile_records import compare_parameter_updates


class ParameterAcceptanceTests(unittest.TestCase):
    def before(self):
        return {"vision.weight": {"sha256": "v", "trainable": False, "dtype": "torch.bfloat16", "shape": [4]},
                "expert.weight": {"sha256": "e", "trainable": True, "dtype": "torch.float32", "shape": [4]}}

    def test_update_requires_changed_trainable_parameters_and_unchanged_frozen_ones(self):
        before = self.before()
        after = deepcopy(before)
        after["expert.weight"]["sha256"] = "updated"
        result = compare_parameter_updates(before, after)
        self.assertEqual(result["changed_trainable_tensors"], 1)
        self.assertTrue(result["frozen_unchanged"])
        after["vision.weight"]["sha256"] = "changed"
        with self.assertRaises(ValueError):
            compare_parameter_updates(before, after)

    def test_no_update_or_silent_precision_change_cannot_pass(self):
        before = self.before()
        with self.assertRaises(ValueError):
            compare_parameter_updates(before, before)
        after = deepcopy(before)
        after["expert.weight"].update(sha256="updated", dtype="torch.bfloat16")
        with self.assertRaises(ValueError):
            compare_parameter_updates(before, after)


if __name__ == "__main__":
    unittest.main()
