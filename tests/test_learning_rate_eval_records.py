import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smolvla_grounding.benchmark_records import BenchmarkJournal
from smolvla_grounding.learning_rate_eval_records import import_completed, require_same_initial_parameters


class ReferenceImportTests(unittest.TestCase):
    def test_import_and_resume_preserve_artifacts_and_change_journal_identity(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = BenchmarkJournal(root / "old", {"run": "old"})
            destination = BenchmarkJournal(root / "new", {"run": "new"})
            attempt = source.new_attempt("episode")
            (attempt / "video.mp4").write_bytes(b"fixture-video")
            result = {"success": True, "video": "episode/attempt_001/video.mp4"}
            source.finish("episode", result, [attempt / "video.mp4"])
            self.assertEqual(import_completed(source, destination, "episode"), result)
            self.assertEqual(import_completed(source, destination, "episode"), result)
            self.assertEqual((destination.directory / result["video"]).read_bytes(), b"fixture-video")
            self.assertNotEqual(source.fingerprint, destination.fingerprint)
            self.assertEqual(destination.completed("episode"), result)

    def test_corrupted_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = BenchmarkJournal(root / "old", {"run": "old"})
            destination = BenchmarkJournal(root / "new", {"run": "new"})
            attempt = source.new_attempt("episode")
            path = attempt / "data.json"
            path.write_text("old")
            source.finish("episode", {}, [path])
            path.write_text("changed")
            with self.assertRaises(RuntimeError):
                import_completed(source, destination, "episode")
            self.assertIsNone(destination.completed("episode"))

    def test_uncommitted_partial_target_is_replaced_from_verified_source(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = BenchmarkJournal(root / "old", {"run": "old"})
            destination = BenchmarkJournal(root / "new", {"run": "new"})
            a, b = source.new_attempt("episode"), destination.new_attempt("episode")
            (a / "data").write_bytes(b"complete")
            (b / "data").write_bytes(b"partial")
            source.finish("episode", {"complete": True}, [a / "data"])
            import_completed(source, destination, "episode")
            self.assertEqual((b / "data").read_bytes(), b"complete")

    def test_different_starting_precision_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "a.json").write_text(json.dumps({"dtype": "bf16"}))
            (root / "b.json").write_text(json.dumps({"dtype": "fp32"}))
            previous = {"provenance": {"training": {"models": {"untuned_fp32": {"parameters": "a.json"}}}}}
            current = {"models": {"untuned_fp32": {"parameters": "b.json"}}}
            with self.assertRaises(RuntimeError):
                require_same_initial_parameters(root, previous, current)


if __name__ == "__main__":
    unittest.main()
