"""Ensure pixel differences use signed arithmetic and the upstream image orientation."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("inspect_pixels", ROOT / "scripts/inspect_repeat_pixels.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PixelTests(unittest.TestCase):
    def test_one_pixel_three_channels_counts_once_and_does_not_wrap(self):
        a = np.zeros((2, 2, 3), dtype=np.uint8)
        b = a.copy()
        a[0, 1] = 255
        result, _ = module.pixel_stats(a, b)
        self.assertEqual(result["changed_pixels"], 1)
        self.assertEqual(result["changed_channels"], 3)
        self.assertEqual(result["max_abs_channel_delta_uint8"], 255)
        self.assertEqual(result["bbox_xyxy_inclusive"], [1, 0, 1, 0])

    def test_identical_images_have_no_bounding_box(self):
        a = np.zeros((2, 2, 3), dtype=np.uint8)
        result, _ = module.pixel_stats(a, a)
        self.assertIsNone(result["bbox_xyxy_inclusive"])

    def test_roundtrip_undoes_180_degree_rotation_for_raw_hash(self):
        pixels = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
        raw_hash = module.array_hashes({"image": pixels[::-1, ::-1]})["image"]
        tensor = pixels.transpose(2, 0, 1)[None].astype(np.float32) / 255
        recovered = module.recover_pixels(tensor, "image", raw_hash)
        np.testing.assert_array_equal(recovered, pixels)
        with self.assertRaisesRegex(RuntimeError, "raw camera hash"):
            module.recover_pixels(tensor, "image", "incorrect")


if __name__ == "__main__":
    unittest.main()
