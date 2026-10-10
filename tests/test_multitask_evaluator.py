"""Regression tests for the production multi-task screen preflight and resume."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from smolvla_grounding.multitask_protocol import (
    _valid_completion, make_rows, new_attempt, preflight, trial_path,
)
from smolvla_grounding.protocol import atomic_json, file_sha256


REPO = Path(__file__).resolve().parents[1]
MATRIX_REL = "configs/multitask_evaluation/canonical_screen_matrix.json"
MANIFEST_REL = "configs/multitask_evaluation/canonical_screen_manifest.json"


class MultitaskPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        matrix = json.loads((REPO / MATRIX_REL).read_text(encoding="utf-8"))
        self.matrix = deepcopy(matrix)
        self.matrix["output_dir"] = "outputs/screen"
        for task in self.matrix["tasks"]:
            source = REPO / task["prompt_catalog"]
            target = self.root / task["prompt_catalog"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
        source = self.root / "source.py"
        source.write_text("frozen source\n", encoding="utf-8")
        model = self.root / "checkpoint"
        model.mkdir()
        (model / "model.bin").write_bytes(b"frozen checkpoint")
        snapshot = self.root / "backbone"
        snapshot.mkdir()
        (snapshot / "config.json").write_bytes(b"frozen backbone")
        self.snapshot = snapshot
        asset_dir = self.root / "assets"
        asset_dir.mkdir()
        self.assets = {}
        for task in self.matrix["tasks"]:
            key = f"{task['suite']}:{task['task_id']}"
            bddl = asset_dir / f"{key.replace(':', '_')}.bddl"
            init = asset_dir / f"{key.replace(':', '_')}.pruned_init"
            bddl.write_bytes(b"bddl")
            init.write_bytes(b"states")
            self.assets[key] = {
                "name": task["task_name"], "language": task["canonical"],
                "state_count": 50, "bddl": bddl.resolve(), "init": init.resolve(),
            }
        self._write_protocol()

    def _write_protocol(self):
        matrix_path = self.root / MATRIX_REL
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(matrix_path, self.matrix)
        self.manifest = {
            "matrix_sha256": file_sha256(matrix_path),
            "source_sha256": {"source.py": file_sha256(self.root / "source.py")},
            "models": {
                "untuned_fp32": {
                    "kind": "official_smolvla", "path": "checkpoint",
                    "artifact_sha256": {
                        "model.bin": file_sha256(self.root / "checkpoint/model.bin"),
                    },
                },
            },
            "backbone": {
                "repo_id": "fixture/backbone", "revision": "fixture",
                "artifact_sha256": {
                    "config.json": file_sha256(self.snapshot / "config.json"),
                },
            },
            "assets": {
                key: {
                    "bddl_path": str(asset["bddl"]),
                    "bddl_sha256": file_sha256(asset["bddl"]),
                    "init_path": str(asset["init"]),
                    "init_sha256": file_sha256(asset["init"]),
                    "state_count": asset["state_count"],
                }
                for key, asset in self.assets.items()
            },
            "catalog_sha256": {
                task["prompt_catalog"]: file_sha256(self.root / task["prompt_catalog"])
                for task in self.matrix["tasks"]
            },
            "prompt_sha256": {
                f"{task['suite']}:{task['task_id']}:original":
                hashlib.sha256(task["canonical"].encode()).hexdigest()
                for task in self.matrix["tasks"]
            },
        }
        atomic_json(self.root / MANIFEST_REL, self.manifest)

    def _inspect(self, suite, task_id):
        key = f"{suite}:{task_id}"
        if key not in self.assets:
            raise RuntimeError("Unknown task")
        return self.assets[key]

    def _preflight(self):
        with patch("smolvla_grounding.multitask_protocol._inspect_task_asset",
                   side_effect=self._inspect), patch(
                       "smolvla_grounding.multitask_protocol._resolve_backbone_snapshot",
                       return_value=self.snapshot):
            return preflight(self.root, MATRIX_REL, MANIFEST_REL)

    def test_production_preflight_path_accepts_exact_matrix(self):
        plan = self._preflight()
        self.assertEqual(len(plan["rows"]), 30)
        self.assertFalse(plan["output_dir"].exists())

    def test_backbone_snapshot_symlink_to_blobs_is_accepted(self):
        blob = self.root / "blobs" / "config.json"
        blob.parent.mkdir()
        blob.write_bytes((self.snapshot / "config.json").read_bytes())
        (self.snapshot / "config.json").unlink()
        (self.snapshot / "config.json").symlink_to(blob)
        self._write_protocol()
        self.assertEqual(len(self._preflight()["rows"]), 30)
    def test_out_of_range_state_rejected_before_rollout(self):
        self.matrix["tasks"][0]["states"][-1] = 50
        self._write_protocol()
        with self.assertRaisesRegex(RuntimeError, "Out-of-range"):
            self._preflight()

    def test_unknown_task_rejected(self):
        self.matrix["tasks"][0]["task_id"] = 99
        self._write_protocol()
        with self.assertRaisesRegex(RuntimeError, "Unknown task"):
            self._preflight()

    def test_wrong_prompt_catalog_rejected(self):
        catalog = self.root / self.matrix["tasks"][0]["prompt_catalog"]
        catalog.write_text('{"id":"original","condition":"original","text":"wrong"}\n',
                           encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Prompt catalog hash"):
            self._preflight()

    def test_wrong_checkpoint_rejected(self):
        (self.root / "checkpoint/model.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "Checkpoint hash"):
            self._preflight()

    def test_duplicate_matrix_combination_rejected(self):
        catalogs = {}
        for task in self.matrix["tasks"]:
            catalogs[(task["suite"], task["task_id"])] = [{"id": "original"}]
        self.matrix["tasks"][0]["states"].append(0)
        self.matrix["expected_rollouts"] = 31
        with self.assertRaisesRegex(RuntimeError, "Duplicate rollout"):
            make_rows(self.matrix, catalogs)

    def test_partial_attempt_preserved_and_resume_creates_new_attempt(self):
        plan = self._preflight()
        row = plan["rows"][0]
        task = self.matrix["tasks"][0]
        trial = trial_path(plan["output_dir"], row, 0)
        plan["output_dir"].mkdir(parents=True)
        atomic_json(plan["output_dir"] / "experiment_meta.json", {"identity": plan["identity"]})
        partial = new_attempt(trial)
        (partial / "steps.jsonl").write_text("partial", encoding="utf-8")
        self.assertIsNone(_valid_completion(trial, row, task["horizon"], plan["identity"]))
        fresh = new_attempt(trial)
        self.assertEqual(fresh.name, "attempt_002")
        self.assertEqual((partial / "steps.jsonl").read_text(), "partial")
        atomic_json(fresh / "episode.json", {
            "model": row[0], "suite": row[1], "task_id": row[2],
            "state": row[3], "prompt_id": row[4], "policy_steps": task["horizon"],
            "goal_ever": True, "goal_final": False,
        })
        for name in ("initial_hashes.json", "initial_state.npz", "trajectory.npz",
                     "steps.jsonl", "inference.jsonl"):
            (fresh / name).write_bytes(b"valid fixture")
        artifact_hashes = {
            path.name: file_sha256(path) for path in fresh.iterdir() if path.is_file()
        }
        atomic_json(fresh / "complete.json", {
            "status": "COMPLETED", "model": row[0], "suite": row[1],
            "task_id": row[2], "state": row[3], "prompt_id": row[4],
            "policy_steps": task["horizon"], "goal_ever": True,
            "goal_final": False, "identity": plan["identity"],
            "artifacts": artifact_hashes,
        })
        self.assertEqual(_valid_completion(trial, row, task["horizon"], plan["identity"]), fresh)
        self.assertEqual(self._preflight()["rows"], plan["rows"])

    def test_corrupt_committed_artifact_rejected(self):
        plan = self._preflight()
        row = plan["rows"][0]
        trial = trial_path(plan["output_dir"], row, 0)
        plan["output_dir"].mkdir(parents=True)
        atomic_json(plan["output_dir"] / "experiment_meta.json", {"identity": plan["identity"]})
        attempt = new_attempt(trial)
        atomic_json(attempt / "complete.json", {
            "status": "COMPLETED", "model": row[0], "suite": row[1],
            "task_id": row[2], "state": row[3], "prompt_id": row[4],
            "policy_steps": 280, "goal_ever": False, "goal_final": False,
            "identity": plan["identity"],
            "artifacts": {name: "0" * 64 for name in
                          ("initial_hashes.json", "initial_state.npz", "trajectory.npz",
                           "steps.jsonl", "inference.jsonl", "episode.json")},
        })
        with self.assertRaisesRegex(RuntimeError, "Corrupt committed artifact"):
            self._preflight()

    def test_unrelated_output_directory_rejected(self):
        output = self.root / self.matrix["output_dir"]
        output.mkdir(parents=True)
        (output / "unrelated.txt").write_text("do not overwrite")
        with self.assertRaisesRegex(RuntimeError, "collision"):
            self._preflight()


    def test_frozen_three_of_five_gate_after_complete_screen(self):
        from smolvla_grounding.multitask_evaluator import _screen_report

        plan = self._preflight()
        for task_index, task in enumerate(self.matrix["tasks"]):
            for state in task["states"]:
                row = ("untuned_fp32", task["suite"], task["task_id"], state, "original")
                trial = trial_path(plan["output_dir"], row, 0)
                attempt = new_attempt(trial)
                success = state < (3 if task_index == 0 else 2)
                episode = {
                    "model": row[0], "suite": row[1], "task_id": row[2],
                    "state": row[3], "prompt_id": row[4],
                    "policy_steps": task["horizon"],
                    "goal_ever": success, "goal_final": False,
                    "duration_seconds": 1.0,
                }
                atomic_json(attempt / "episode.json", episode)
                for name in ("initial_hashes.json", "initial_state.npz", "trajectory.npz",
                             "steps.jsonl", "inference.jsonl"):
                    (attempt / name).write_bytes(b"fixture")
                artifacts = {
                    path.name: file_sha256(path) for path in attempt.iterdir() if path.is_file()
                }
                atomic_json(attempt / "complete.json", {
                    **episode, "status": "COMPLETED", "identity": plan["identity"],
                    "artifacts": artifacts,
                })
        report = _screen_report(plan)
        self.assertEqual(report["completed_rollouts"], 30)
        self.assertFalse(report["scientific_evidence"])
        self.assertEqual(report["per_task"][0]["classification"],
                         "eligible_for_language_robustness")
        self.assertTrue(all(row["classification"] == "insufficient_canonical_competence"
                            for row in report["per_task"][1:]))
        self.assertEqual(report["proposed_scientific_rollouts"], 15)
        self.assertEqual(report["total_rollout_seconds"], 30.0)
if __name__ == "__main__":
    unittest.main()





