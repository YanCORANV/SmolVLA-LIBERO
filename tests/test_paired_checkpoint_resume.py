"""Regression check for the paired trainer's existing journal and state-resume contract."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding.adaptation_records import latest_checkpoint, read_jsonl
from smolvla_grounding.benchmark_records import BenchmarkJournal
from smolvla_grounding.protocol import file_sha256


class PairedCheckpointResumeTests(unittest.TestCase):
    def test_clean_commit_partial_attempt_and_latest_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            identity = {"experiment": {"updates": 1000}, "arm": "paired_bc", "schedule_hash": "fixed"}
            journal = BenchmarkJournal(directory, identity)
            self.assertIsNone(latest_checkpoint(journal, 1000, 500))

            first = journal.new_attempt("step_000500")
            first_artifact = first / "model.safetensors"
            first_artifact.write_bytes(b"committed model")
            first_result = {"step": 500, "checkpoint": first.relative_to(directory).as_posix()}
            journal.finish("step_000500", first_result, [first_artifact])
            self.assertEqual(latest_checkpoint(journal, 1000, 500), first_result)

            partial = journal.new_attempt("step_001000")
            partial_artifact = partial / "model.safetensors"
            partial_artifact.write_bytes(b"incomplete model")
            self.assertIsNone(journal.completed("step_001000"))
            self.assertEqual(latest_checkpoint(journal, 1000, 500), first_result)

            resumed_journal = BenchmarkJournal(directory, identity)
            self.assertEqual(latest_checkpoint(resumed_journal, 1000, 500), first_result)
            completed = resumed_journal.new_attempt("step_001000")
            self.assertEqual(completed.name, "attempt_002")
            completed_artifact = completed / "model.safetensors"
            completed_artifact.write_bytes(b"completed model")
            final_result = {"step": 1000, "checkpoint": completed.relative_to(directory).as_posix()}
            resumed_journal.finish("step_001000", final_result, [completed_artifact])
            self.assertEqual(latest_checkpoint(resumed_journal, 1000, 500), final_result)
            self.assertEqual(partial_artifact.read_bytes(), b"incomplete model")
            self.assertEqual(first_artifact.read_bytes(), b"committed model")

            completed_artifact.write_bytes(b"corrupted model")
            with self.assertRaises(RuntimeError):
                latest_checkpoint(resumed_journal, 1000, 500)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("lerobot"),
                         "PyTorch/LeRobot runtime is required for optimizer-state round trip")
    def test_official_training_state_round_trip(self):
        from copy import deepcopy
        import torch
        from lerobot.common.train_utils import load_training_state, save_training_state

        def initialize():
            model = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2))
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
            return model, optimizer, scheduler

        def update(model, optimizer, scheduler):
            x = torch.randn(4, 3)
            target = torch.randn(4, 2)
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(model(x), target).backward()
            optimizer.step()
            scheduler.step()
            return optimizer.param_groups[0]["lr"]

        def parameters(model):
            return [parameter.detach().clone() for parameter in model.parameters()]

        total_updates, committed_step = 6, 3
        torch.manual_seed(20261002)
        uninterrupted, full_optimizer, full_scheduler = initialize()
        uninterrupted_lrs = []
        for step in range(1, total_updates + 1):
            uninterrupted_lrs.append(update(uninterrupted, full_optimizer, full_scheduler))
            if step == committed_step:
                expected_at_commit = parameters(uninterrupted)
                expected_optimizer = deepcopy(full_optimizer.state_dict())
                expected_scheduler = deepcopy(full_scheduler.state_dict())
        expected_final = parameters(uninterrupted)

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            journal = BenchmarkJournal(directory, {"experiment": "synthetic resume", "arm": "paired_bc"})
            torch.manual_seed(20261002)
            model, optimizer, scheduler = initialize()
            for _ in range(committed_step):
                update(model, optimizer, scheduler)
            for observed, expected in zip(parameters(model), expected_at_commit):
                self.assertTrue(torch.equal(observed, expected))

            checkpoint = journal.new_attempt(f"step_{committed_step:06d}")
            model_file = checkpoint / "model.pt"
            torch.save(model.state_dict(), model_file)
            save_training_state(checkpoint, committed_step, optimizer=optimizer, scheduler=scheduler,
                                num_processes=1, batch_size=1)
            result = {"step": committed_step, "checkpoint": checkpoint.relative_to(directory).as_posix()}
            journal.finish(f"step_{committed_step:06d}", result,
                           [model_file, *sorted((checkpoint / "training_state").iterdir())])

            torch.manual_seed(7)  # Simulate a reconstructed process before loading its saved RNG.
            restored_model, restored_optimizer, restored_scheduler = initialize()
            committed = latest_checkpoint(journal, total_updates, committed_step)
            self.assertEqual(committed["step"], committed_step)
            restored_model.load_state_dict(torch.load(directory / committed["checkpoint"] / "model.pt",
                                                      weights_only=True))
            for observed, expected in zip(parameters(restored_model), expected_at_commit):
                self.assertTrue(torch.equal(observed, expected))

            step, restored_optimizer, restored_scheduler = load_training_state(
                directory / committed["checkpoint"], restored_optimizer, restored_scheduler)
            self.assertEqual(step, committed_step)
            self.assertEqual(restored_scheduler.state_dict(), expected_scheduler)
            self.assertEqual(restored_optimizer.param_groups[0]["lr"], uninterrupted_lrs[committed_step - 1])
            actual_optimizer = restored_optimizer.state_dict()
            for key in expected_optimizer["state"]:
                for name in ("step", "exp_avg", "exp_avg_sq"):
                    self.assertTrue(torch.equal(actual_optimizer["state"][key][name],
                                                expected_optimizer["state"][key][name]))

            next_step = step + 1
            self.assertEqual(next_step, committed_step + 1)
            next_lr = update(restored_model, restored_optimizer, restored_scheduler)
            self.assertEqual(next_lr, uninterrupted_lrs[committed_step])
            self.assertEqual(int(next(iter(restored_optimizer.state.values()))["step"].item()), next_step)
            for _ in range(next_step + 1, total_updates + 1):
                update(restored_model, restored_optimizer, restored_scheduler)
            for observed, expected in zip(parameters(restored_model), expected_final):
                self.assertTrue(torch.allclose(observed, expected, rtol=0, atol=1e-7))


class PairedCatalogTests(unittest.TestCase):
    def test_paired_catalog_matches_historical_training_records(self):
        paired = json.loads((ROOT / "configs/paired_training.json").read_text(encoding="utf-8"))
        historical_run = ROOT / "outputs/adaptation/runs/task0_longer_training"
        historical = json.loads((historical_run / "experiment.json").read_text(encoding="utf-8"))
        catalog = paired["training"]["train_catalog"]
        self.assertEqual(catalog, historical["provenance"]["settings"]["training_catalog"])
        self.assertEqual(file_sha256(ROOT / catalog), historical["provenance"]["sources"][catalog])

        rows = read_jsonl(ROOT / catalog)
        expected_ids = ["train_original", *(f"train_paraphrase_{i:02d}" for i in range(1, 5))]
        self.assertEqual([row["id"] for row in rows], expected_ids)
        task = json.loads((ROOT / "configs/task0_dataset.json").read_text(encoding="utf-8"))
        self.assertEqual(rows[0]["text"], task["original_instruction"])

        for arm in ("original_language", "augmented_language"):
            records = read_jsonl(historical_run / arm / "step_001000/attempt_001/training_log.jsonl")
            self.assertEqual(len(records), 1000)
            self.assertTrue(all(record["instruction_id"] == rows[record["variant_index"]]["id"]
                                for record in records))
            allowed = {record["instruction_id"] for record in records}
            if arm == "original_language":
                self.assertEqual(allowed, {"train_original"})
            else:
                self.assertEqual(allowed, set(expected_ids))


if __name__ == "__main__":
    unittest.main()
