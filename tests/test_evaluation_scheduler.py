from pathlib import Path
import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from smolvla_grounding.evaluation_scheduler import can_add_worker, ADDITIONAL_GPU_BYTES, ADDITIONAL_RAM_BYTES
from smolvla_grounding.evaluation_scheduler import run_workers


class EvaluationAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.resources = {"available_ram_bytes": ADDITIONAL_RAM_BYTES,
                          "gpu_free_bytes": ADDITIONAL_GPU_BYTES}

    def test_both_budgets_and_ready_required_for_second_worker(self):
        self.assertTrue(can_add_worker(self.resources, 1, 2, True))
        self.assertFalse(can_add_worker(self.resources, 1, 2, False))
        for key in self.resources:
            with self.subTest(resource=key):
                low = dict(self.resources)
                low[key] -= 1
                self.assertFalse(can_add_worker(low, 1, 2, True))

    def test_limit_and_serial_fallback(self):
        self.assertFalse(can_add_worker(self.resources, 1, 1, True))
        self.assertFalse(can_add_worker(self.resources, 2, 2, True))
        # The already accepted single-process execution remains possible when
        # there is not enough memory for two; the scheduler must not deadlock.
        low = {k: 0 for k in self.resources}
        self.assertTrue(can_add_worker(low, 0, 2, False))


@unittest.skipUnless(os.name == "posix", "Process-group checks run in WSL/Linux")
class ProcessSchedulingTests(unittest.TestCase):
    def exercise(self, fail_second):
        snapshot = {"available_ram_bytes": ADDITIONAL_RAM_BYTES, "gpu_free_bytes": ADDITIONAL_GPU_BYTES,
                    "swap_used_bytes": 0, "gpu_utilization_percent": 0}
        sleep = time.sleep
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "scripts").mkdir()
            script = root / "scripts/evaluate_adaptation_model.py"
            script.write_text("import os,time,json,sys\nfrom pathlib import Path\n"
                "m=os.environ['SMOLVLA_ADAPTATION_EVAL_MODEL']\n"
                "p=Path(os.environ['SMOLVLA_ADAPTATION_EVAL_DIR'])/m\np.mkdir(parents=True,exist_ok=True)\n"
                "(p/'model_load.json').write_text('{}')\nprint('policy ready',flush=True)\n"
                + ("time.sleep(2 if m=='first' else 0.2)\nsys.exit(7 if m=='second' else 0)\n" if fail_second
                   else "time.sleep(0.6)\n"))
            directory = root / "evaluation"
            with patch('smolvla_grounding.evaluation_scheduler.resources', return_value=snapshot), \
                 patch('smolvla_grounding.evaluation_scheduler.time.sleep', side_effect=lambda _: sleep(.02)):
                if fail_second:
                    with self.assertRaises(RuntimeError):
                        run_workers(root, directory, ["first", "second"], 2)
                else:
                    run_workers(root, directory, ["first", "second"], 2)
            reports = list((directory / "scheduling").glob('*/summary.json'))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertTrue(report['ever_overlapped'])
            self.assertEqual(report['passed'], not fail_second)
            self.assertTrue(all(w['exit_code'] is not None for w in report['workers'].values()))
            return report

    def test_two_workers_finish_and_report_overlap(self):
        report = self.exercise(False)
        self.assertTrue(all(w['exit_code'] == 0 for w in report['workers'].values()))

    def test_failed_worker_stops_owned_peer_and_saves_report(self):
        report = self.exercise(True)
        self.assertEqual(report['workers']['second']['exit_code'], 7)
        self.assertNotEqual(report['workers']['first']['exit_code'], 0)


if __name__ == "__main__":
    unittest.main()
