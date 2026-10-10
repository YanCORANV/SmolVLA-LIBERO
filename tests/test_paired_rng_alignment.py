"""The complete paired schedule and official flow samplers are arm independent."""

from collections import Counter
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.adaptation_records import (
    build_paired_schedule, compare_paired_training_logs, paired_flow_seed,
)
from smolvla_grounding.flow_velocity import sample_paired_flow_inputs


class OfficialShapeSampler:
    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, batch_size, device):
        return (torch.distributions.Beta(1.5, 1.0).sample((batch_size,)) * 0.999 + 0.001).to(device)


class PairedRngAlignmentTests(unittest.TestCase):
    def test_all_1000_updates_match_despite_arm_specific_rng_consumption(self):
        episodes = [{"episode_index": i, "frames": 7 + i % 3} for i in range(36)]
        bc = build_paired_schedule(episodes, 1000, 0, 20260909)
        consistency = build_paired_schedule(episodes, 1000, 0, 20260909)
        self.assertEqual(bc, consistency)
        self.assertEqual(Counter(row["paraphrase_variant"] for row in bc), {1: 250, 2: 250, 3: 250, 4: 250})

        sampler = OfficialShapeSampler()
        torch.manual_seed(123)
        baseline_rng = torch.get_rng_state().clone()
        observed = []
        for arm in ("paired_bc", "paired_flow_consistency"):
            torch.set_rng_state(baseline_rng.clone())
            arm_rows = []
            for row in (bc if arm == "paired_bc" else consistency):
                if arm == "paired_flow_consistency":
                    torch.randn(3)  # Teacher/logging can consume global RNG without shifting flow inputs.
                before = torch.get_rng_state().clone()
                noise, flow_time, seed = sample_paired_flow_inputs(sampler, (1, 4, 7), "cpu", 0, row["step"])
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                self.assertEqual(seed, paired_flow_seed(0, row["step"]))
                arm_rows.append((row["episode_index"], row["frame_index"], row["paraphrase_variant"],
                                 seed, noise.clone(), flow_time.clone()))
            observed.append(arm_rows)
        for left, right in zip(*observed):
            self.assertEqual(left[:4], right[:4])
            self.assertTrue(torch.equal(left[4], right[4]))
            self.assertTrue(torch.equal(left[5], right[5]))

    def test_log_audit_rejects_changed_noise_time_or_controls(self):
        schedule = [{"step": 1, "episode_index": 2, "frame_index": 3, "paraphrase_variant": 1}]
        row = {**schedule[0], "data_hash": "data", "action_hash": "action",
               "flow_seed": paired_flow_seed(0, 1), "noise_hash": "noise",
               "time_hash": "time", "lr_used": 1e-5}
        self.assertTrue(compare_paired_training_logs([row], [row.copy()], schedule, 0)["all_updates_identical"])
        for key in ("episode_index", "frame_index", "paraphrase_variant", "data_hash",
                    "action_hash", "flow_seed", "noise_hash", "time_hash", "lr_used"):
            changed = row.copy()
            changed[key] = "changed"
            with self.assertRaises(ValueError, msg=key):
                compare_paired_training_logs([row], [changed], schedule, 0)


if __name__ == "__main__":
    unittest.main()