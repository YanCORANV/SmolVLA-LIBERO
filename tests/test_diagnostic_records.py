"""Verify lossless archive containers and avoid overinterpreting divergence order."""

from copy import deepcopy
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from smolvla_grounding.diagnostic_records import compare_traces, decode_record, encode_record


class ArchiveTests(unittest.TestCase):
    def test_byte_arrays_and_rng_metadata_survive_compression(self):
        meta = {"input": {"state": {"dtype": "bfloat16", "shape": [1, 2], "device": "cuda:0"}},
                "rng_other": {"python": [3, [1, 2, 3], None], "numpy": ["MT19937", 3, 1, 0.123456789012345]}}
        arrays = {"input/state": np.array([0, 128, 123, 55], dtype=np.uint8),
                  "rng/numpy_keys": np.array([2**32 - 1, 0], dtype=np.uint32)}
        restored_meta, restored_arrays = decode_record(encode_record(meta, arrays))
        self.assertEqual(meta, restored_meta)
        for key in arrays:
            self.assertEqual(arrays[key].dtype, restored_arrays[key].dtype)
            self.assertEqual(arrays[key].tobytes(), restored_arrays[key].tobytes())


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        fields = ("raw_pixels", "input", "rng_before", "noise", "normalized_action", "rng_after")
        self.a = [{"step": i, **{f: {"tensor": "same"} for f in fields}} for i in range(1, 5)]
        self.b = deepcopy(self.a)

    def test_matching_traces_report_no_observed_divergence(self):
        result = compare_traces(self.a, self.b)
        self.assertTrue(all(x is None for x in result["first_different_step"].values()))
        self.assertIsNone(result["evidence_at_first_action_difference"])

    def test_earlier_input_difference_is_not_hidden_by_later_action_difference(self):
        self.b[1]["input"]["tensor"] = "changed"
        self.b[2]["normalized_action"]["tensor"] = "changed"
        result = compare_traces(self.a, self.b)
        self.assertEqual(result["first_different_step"]["input"], 2)
        self.assertEqual(result["first_different_step"]["normalized_action"], 3)

    def test_same_input_and_noise_with_action_difference_is_reported_without_asserting_cause(self):
        self.b[2]["normalized_action"]["tensor"] = "changed"
        result = compare_traces(self.a, self.b)
        self.assertTrue(result["evidence_at_first_action_difference"]["input_equal"])
        self.assertTrue(result["evidence_at_first_action_difference"]["noise_equal"])
        self.assertFalse(result["evidence_at_first_action_difference"]["normalized_action_equal"])

    def test_missing_step_cannot_be_silently_aligned(self):
        self.b[2]["step"] = 9
        with self.assertRaises(ValueError):
            compare_traces(self.a, self.b)


if __name__ == "__main__":
    unittest.main()
