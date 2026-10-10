"""Final protocol tests; no LIBERO simulator or policy is constructed."""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from smolvla_grounding import final_confirmatory_evaluator as final_runtime
from smolvla_grounding.protocol import file_sha256
from smolvla_grounding.final_confirmatory_evaluator import (
    _completed_attempt, _new_attempt, _prepare_run_metadata, validate_scope,
    validate_frozen_artifacts, validate_active_initial_states, validate_state_matrix_against_count,
)
from smolvla_grounding.final_confirmatory_summary import MODELS, PROMPTS, STATES, compute_statistics
from tests.final_v4_fixture import candidate_v4_freeze


class FinalConfirmatoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = json.loads((ROOT / "configs/final_confirmatory/final_matrix_v4.json").read_text(encoding="utf-8"))
        cls.manifest = json.loads((ROOT / "configs/final_confirmatory/frozen_model_manifest_v4.json").read_text(encoding="utf-8"))

    def test_exact_amended_matrix_without_environment_access(self):
        rows = validate_scope(ROOT, self.matrix, self.manifest)
        self.assertEqual([row["id"] for row in rows], list(PROMPTS))
        self.assertEqual(self.matrix["models"], list(MODELS))
        self.assertEqual(self.matrix["tasks"][0]["initial_states"], list(range(33, 50)))
        self.assertEqual(self.matrix["execution"]["total_rollouts"], 153)
        self.assertEqual(STATES, tuple(range(33, 50)))

    def test_reject_wrong_state_model_prompt_and_runtime(self):
        mutations = (
            lambda m: m["tasks"][0].update(initial_states=list(range(33, 51))),
            lambda m: m["tasks"][0].update(initial_states=list(range(32, 49))),
            lambda m: m.update(models=["paired_flow_consistency", "augmented_language", "paired_bc"]),
            lambda m: m["tasks"][0].update(instruction_catalog="configs/final_confirmatory_reserved/instructions/task0_final_reserved.jsonl"),
            lambda m: m["environment"].update(episode_length=100),
            lambda m: m["execution"].update(default_workers=2),
            lambda m: m["statistical_rules"].update(bootstrap_replicates=9999),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                bad = deepcopy(self.matrix)
                mutate(bad)
                with self.assertRaises(RuntimeError):
                    validate_scope(ROOT, bad, self.manifest)

    def test_actual_asset_count_rejects_whole_matrix_before_rollout(self):
        frozen = candidate_v4_freeze(ROOT, self.manifest)
        states = validate_active_initial_states(ROOT, self.matrix, frozen)
        self.assertEqual(tuple(states.shape), (50, 92))
        self.assertEqual(validate_state_matrix_against_count(self.matrix, len(states)), 50)
        bad = deepcopy(self.matrix)
        bad["tasks"][0]["initial_states"] = list(range(33, 51))
        with self.assertRaisesRegex(RuntimeError, "exceeds active initialization-state count"):
            validate_active_initial_states(ROOT, bad, frozen)
        with self.assertRaisesRegex(RuntimeError, "exceeds active initialization-state count"):
            validate_state_matrix_against_count(self.matrix, 49)

    def test_existing_51_fixtures_match_active_source_without_reading_outcomes(self):
        import numpy as np
        frozen = candidate_v4_freeze(ROOT, self.manifest)
        states = validate_active_initial_states(ROOT, self.matrix, frozen)
        base = ROOT / "outputs/final_confirmatory/runs/task0_paired_bc_final_confirmatory/original_language/task_00"
        matched = 0
        for state in STATES:
            for prompt in PROMPTS:
                trial = base / f"state_{state:04d}_seed_0000_{prompt}"
                commits = sorted(trial.glob("attempt_*/complete.json"))
                self.assertEqual(len(commits), 1)
                with np.load(commits[0].parent / "initial_state.npz", allow_pickle=False) as saved:
                    self.assertTrue(np.array_equal(saved["fixture_state"], states[state]))
                matched += 1
        self.assertEqual(matched, 51)

    def test_cluster_bootstrap_preserves_paired_effect_over_17_states(self):
        outcomes = {
            state: {
                "original_language": dict(original=True, paraphrase_final_01=False, paraphrase_final_02=False),
                "augmented_language": dict(original=state % 2 == 0, paraphrase_final_01=True, paraphrase_final_02=False),
                "paired_bc": dict(original=True, paraphrase_final_01=True, paraphrase_final_02=False),
            } for state in STATES
        }
        stats = compute_statistics(outcomes, STATES)
        self.assertEqual(stats["models"]["paired_bc"]["counts"]["pooled_paraphrases"], 17)
        self.assertEqual(stats["models"]["paired_bc"]["rates"]["P"], 0.5)
        self.assertEqual(stats["paired_effects"]["delta_para_original"]["ci95"], [0.5, 0.5])
        del outcomes[33]["paired_bc"]["paraphrase_final_02"]
        with self.assertRaises(ValueError):
            compute_statistics(outcomes, STATES)

    def test_candidate_production_preflight_and_hash_mutations(self):
        candidate = candidate_v4_freeze(ROOT, self.manifest)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            path.write_text(json.dumps(candidate), encoding="utf-8")
            with patch.object(final_runtime, "FREEZE_PATH", path):
                with patch.object(final_runtime, "_execute_dev_episode", side_effect=AssertionError("rollout accessed")):
                    self.assertEqual(len(validate_frozen_artifacts(ROOT, self.matrix, self.manifest)), 3)
                    targets = (
                        ("src/smolvla_grounding/flow_velocity.py", "Training source/config changed"),
                        ("src/smolvla_grounding/benchmark_runtime.py", "Final evaluation runtime changed"),
                        ("configs/final_confirmatory/final_matrix_v4.json", "Final matrix SHA256 mismatch"),
                        ("configs/final_confirmatory/instructions/task0_final.jsonl", "Final prompt catalog SHA256 mismatch"),
                        (self.manifest["evaluated_models"]["paired_bc"]["checkpoint"] + "/pretrained_model/model.safetensors", "Frozen checkpoint hash changed"),
                    )
                    for rel, message in targets:
                        with self.subTest(rel=rel):
                            target = ROOT / rel
                            real_hash = final_runtime.file_sha256
                            def altered(path, target=target, real_hash=real_hash):
                                return "0" * 64 if Path(path) == target else real_hash(path)
                            with patch.object(final_runtime, "file_sha256", side_effect=altered):
                                with self.assertRaisesRegex(RuntimeError, message):
                                    validate_frozen_artifacts(ROOT, self.matrix, self.manifest)

    def test_v4_metadata_preserves_51_commits_and_partial_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            old = run / "experiment_meta_v3.json"
            old.write_text('{"version":3}', encoding="utf-8")
            old_bytes = old.read_bytes()
            hashes = {}
            for state in STATES:
                for prompt in PROMPTS:
                    trial = run / "original_language" / "task_00" / f"state_{state:04d}_seed_0000_{prompt}" / "attempt_001"
                    trial.mkdir(parents=True)
                    marker = {
                        "status": "COMPLETED", "model": "original_language", "task_id": 0,
                        "initial_state_index": state, "instruction_id": prompt, "policy_steps": 280,
                        "goal_ever": False, "goal_final": False,
                    }
                    (trial / "complete.json").write_text(json.dumps(marker), encoding="utf-8")
                    (trial / "episode.json").write_text(json.dumps({k: v for k, v in marker.items() if k != "status"}), encoding="utf-8")
                    (trial / "initial_state.npz").write_bytes(b"synthetic fixture")
                    for name in ("complete.json", "episode.json", "initial_state.npz"):
                        rel = (trial / name).relative_to(run).as_posix()
                        hashes[rel] = file_sha256(trial / name)
            partial = run / "original_language/task_00/state_0050_seed_0000_original/attempt_002"
            partial.mkdir(parents=True)
            frozen = {"historical_v3_meta_sha256": file_sha256(old),
                      "historical_completed_artifact_sha256": hashes}
            expected = {"run_name": "synthetic", "freeze_record_sha256": "v4"}
            active = _prepare_run_metadata(run, expected, frozen)
            self.assertEqual(active.name, "experiment_meta_v4.json")
            self.assertEqual(old.read_bytes(), old_bytes)
            self.assertTrue(partial.is_dir())
            self.assertEqual(_prepare_run_metadata(run, expected, frozen), active)
            self.assertEqual(sorted(p.name for p in partial.parent.iterdir()), ["attempt_002"])
            self.assertEqual(len(list(run.glob("original_language/task_00/state_*/attempt_*/complete.json"))), 51)

    def test_complete_marker_resume_and_partial_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            trial = Path(directory) / "trial"
            partial = _new_attempt(trial)
            self.assertIsNone(_completed_attempt(trial, "paired_bc", 33, "original"))
            second = _new_attempt(trial)
            marker = {
                "status": "COMPLETED", "model": "paired_bc", "task_id": 0,
                "initial_state_index": 33, "instruction_id": "original",
                "policy_steps": 280, "goal_ever": True, "goal_final": True,
            }
            (second / "complete.json").write_text(json.dumps(marker), encoding="utf-8")
            (second / "episode.json").write_text(json.dumps({k: v for k, v in marker.items() if k != "status"}), encoding="utf-8")
            self.assertEqual(_completed_attempt(trial, "paired_bc", 33, "original"), second)
            self.assertTrue(partial.is_dir())


if __name__ == "__main__":
    unittest.main()
