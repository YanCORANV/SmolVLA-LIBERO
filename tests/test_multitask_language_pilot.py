"""Pilot matrix isolation, pairing audit, and descriptive metric tests."""
from __future__ import annotations

from copy import deepcopy
import csv
import hashlib
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from smolvla_grounding.multitask_pilot_protocol import pilot_preflight, TASKS
from smolvla_grounding.multitask_pilot_evaluator import _paired_reset_audit
from smolvla_grounding.multitask_pilot_analysis import (outcome_transition, regenerate_published,
                                                       task_success)
from smolvla_grounding.multitask_protocol import make_rows

REPO = Path(__file__).resolve().parents[1]


class PilotProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        matrix_path = self.root / "configs/multitask_evaluation/language_pilot_matrix.json"
        matrix_path.parent.mkdir(parents=True)
        self.matrix = json.loads((REPO / "configs/multitask_evaluation/language_pilot_matrix.json").read_text())
        screen = self.root / "results/multitask_canonical_screen/summary.json"
        screen.parent.mkdir(parents=True)
        screen.write_bytes((REPO / "results/multitask_canonical_screen/summary.json").read_bytes())
        self.catalogs = {}
        for task in self.matrix["tasks"]:
            rows = [json.loads(line) for line in (REPO / task["prompt_catalog"]).read_text().splitlines()]
            self.catalogs[(task["suite"], task["task_id"])] = rows
        self.rows = make_rows(self.matrix, self.catalogs)
        self._write()

    def _write(self):
        path = self.root / "configs/multitask_evaluation/language_pilot_matrix.json"
        path.write_text(json.dumps(self.matrix), encoding="utf-8")

    def _check(self):
        with patch("smolvla_grounding.multitask_pilot_protocol.preflight") as generic:
            generic.return_value = {"matrix": self.matrix, "manifest": {"role": "multitask_language_robustness"}, "catalogs": self.catalogs,
                                    "rows": self.rows}
            return pilot_preflight(self.root)

    def test_exact_75_cell_matrix(self):
        plan = self._check()
        self.assertEqual(len(plan["rows"]), 75)
        self.assertEqual({r[1:3] for r in plan["rows"]}, set(TASKS))
        self.assertEqual({r[3] for r in plan["rows"]}, set(range(5, 10)))
        self.assertFalse(any(30 <= r[3] <= 49 for r in plan["rows"]))
        self.assertEqual({r[4] for r in plan["rows"]},
                         {"original", "paraphrase_01", "paraphrase_02"})

    def test_rejects_final_state_and_excluded_task(self):
        self.matrix["tasks"][0]["states"][0] = 30
        self._write()
        with self.assertRaisesRegex(RuntimeError, "states"):
            self._check()
        self.matrix["tasks"][0]["states"][0] = 5
        self.matrix["tasks"][0]["task_id"] = 0
        self._write()
        with self.assertRaisesRegex(RuntimeError, "task set"):
            self._check()

    def test_rejects_role_count_and_prompt_variants(self):
        original = deepcopy(self.matrix)
        for change in (lambda m: m.update(role="canonical_competence_screen"),
                       lambda m: m.update(expected_rollouts=74)):
            self.matrix = deepcopy(original)
            change(self.matrix)
            self._write()
            with self.assertRaises(RuntimeError):
                self._check()
        self.matrix = deepcopy(original)
        self._write()
        self.catalogs["libero_spatial", 1][1]["id"] = "other"
        with self.assertRaisesRegex(RuntimeError, "prompts"):
            self._check()

    def test_paired_initial_and_rng_equality(self):
        task = self.matrix["tasks"][0]
        paths = []
        for prompt in ("original", "paraphrase_01", "paraphrase_02"):
            path = self.root / prompt
            path.mkdir()
            (path / "initial_hashes.json").write_text(json.dumps({"fixture_state": "same"}))
            (path / "inference.jsonl").write_text(json.dumps({"step": 1, "rng_before": {"torch": "same"}, "noise": {"noise": "same"}}) + "\n")
            paths.append(path)
        plan = {"output_dir": self.root, "matrix": self.matrix, "identity": "fixture"}
        with patch("smolvla_grounding.multitask_pilot_evaluator._valid_completion", side_effect=paths):
            _paired_reset_audit(plan, task, 5)
        (paths[1] / "initial_hashes.json").write_text(json.dumps({"fixture_state": "different"}))
        with patch("smolvla_grounding.multitask_pilot_evaluator._valid_completion", side_effect=paths):
            with self.assertRaisesRegex(RuntimeError, "initial states differ"):
                _paired_reset_audit(plan, task, 5)
        (paths[1] / "initial_hashes.json").write_text(json.dumps({"fixture_state": "same"}))
        (paths[2] / "inference.jsonl").write_text(json.dumps({"step": 1, "rng_before": {"torch": "different"}, "noise": {"noise": "same"}}) + "\n")
        with patch("smolvla_grounding.multitask_pilot_evaluator._valid_completion", side_effect=paths):
            with self.assertRaisesRegex(RuntimeError, "RNG differs"):
                _paired_reset_audit(plan, task, 5)
        (paths[2] / "inference.jsonl").write_text(json.dumps({"step": 1, "rng_before": {"torch": "same"}, "noise": {"noise": "different"}}) + "\n")
        with patch("smolvla_grounding.multitask_pilot_evaluator._valid_completion", side_effect=paths):
            with self.assertRaisesRegex(RuntimeError, "flow noise differs"):
                _paired_reset_audit(plan, task, 5)

    def test_paired_success_metrics_keep_all_states(self):
        rows = [{"state": s, "original": s != 9,
                 "paraphrase_01": s in (5, 6), "paraphrase_02": s in (5, 7)}
                for s in range(5, 10)]
        result = task_success(rows)
        self.assertEqual(result["original"]["successes"], 4)
        self.assertEqual(result["pooled_paraphrase"]["successes"], 4)
        self.assertEqual(result["pooled_paraphrase"]["trials"], 10)
        self.assertEqual(result["canonical_success_paraphrase_failure_pairs"], 4)
        self.assertEqual(result["canonical_success_paraphrase_failure_states"], 3)
        with self.assertRaisesRegex(ValueError, "five"):
            task_success(rows[:-1])

    def test_all_outcome_transitions(self):
        self.assertEqual(outcome_transition(True, True), "both_succeed")
        self.assertEqual(outcome_transition(True, False),
                         "canonical_success_paraphrase_failure")
        self.assertEqual(outcome_transition(False, True),
                         "canonical_failure_paraphrase_success")
        self.assertEqual(outcome_transition(False, False), "both_fail")
        with self.assertRaises(ValueError):
            outcome_transition(1, False)


class PublishedPilotTests(unittest.TestCase):
    def _copy_public_fixture(self, destination):
        matrix = json.loads((REPO / "configs/multitask_evaluation/language_pilot_matrix.json")
                            .read_text(encoding="utf-8"))
        paths = [
            "configs/multitask_evaluation/language_pilot_matrix.json",
            "configs/multitask_evaluation/language_pilot_manifest.json",
            *[task["prompt_catalog"] for task in matrix["tasks"]],
            *[f"results/multitask_language_pilot/{name}" for name in
              ("summary.json", "per_rollout.csv", "pair_level.csv",
               "paired_eef_examples.json", "integrity.json")],
        ]
        for rel in paths:
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / rel, target)

    def _rehash(self, destination, name):
        audit_file = destination / "results/multitask_language_pilot/integrity.json"
        audit = json.loads(audit_file.read_text(encoding="utf-8"))
        content = (destination / "results/multitask_language_pilot" / name).read_bytes()
        audit["public_source_sha256"][name] = hashlib.sha256(content).hexdigest()
        audit_file.write_text(json.dumps(audit), encoding="utf-8")

    def test_public_statistics_and_figures_reproduce_without_raw_outputs(self):
        summary = regenerate_published(REPO)
        self.assertEqual(summary["completed_rollouts"], 75)
        self.assertEqual(summary["outcome_transitions"][
            "canonical_success_paraphrase_failure"], 38)

    def test_missing_public_rollout_is_rejected_after_hash_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._copy_public_fixture(root)
            path = root / "results/multitask_language_pilot/per_rollout.csv"
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            self._rehash(root, "per_rollout.csv")
            with self.assertRaisesRegex(RuntimeError, "per-rollout matrix differs"):
                regenerate_published(root)

    def test_missing_eef_is_rejected_after_hash_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._copy_public_fixture(root)
            path = root / "results/multitask_language_pilot/pair_level.csv"
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["eef_step_1_m"] = ""
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            self._rehash(root, "pair_level.csv")
            with self.assertRaisesRegex(RuntimeError, "Missing or invalid published divergence"):
                regenerate_published(root)
