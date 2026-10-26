import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("run_validated_local_backtracking_419", HERE / "run_validated_local_backtracking_419.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
owner = runner.load_config_owner()

class ValidatedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg_path = self.root / "config.json"
        self.out = self.root / "fresh-out"
        self.base = {
            "helper": str(owner.LOCAL_HELPER), "helper_sha256": owner.LOCAL_SHA256,
            "helper_entry": owner.LOCAL_ENTRY, "uniform_failure_backtracking": False,
            "state": {}, "trial": "trial.npz", "parent_fit": "fit.json",
            "out": str(self.out), "initial_scale": 1.0,
        }
        self.cfg_path.write_text(json.dumps(self.base))

    def test_valid_local_config_invokes_exact_frozen_runner_with_timeout(self):
        fake = Mock(return_value=subprocess.CompletedProcess([], 0))
        receipt = runner.execute_373(self.cfg_path, timeout_seconds=77, run=fake)
        fake.assert_called_once_with(
            [str(runner.PYTHON), str(runner.RUNNER_PATH), str(self.cfg_path.resolve())],
            check=False, timeout=77)
        self.assertEqual(receipt["status"], "RUNNER_EXITED")
        self.assertEqual(receipt["returncode"], 0)
        self.assertFalse(receipt["timed_out"])
        self.assertEqual(receipt["config_binding"]["mode"], "local")

    def test_418_false_flag_with_global_helper_rejects_without_launch(self):
        bad = dict(self.base, helper=str(owner.GLOBAL_HELPER),
                   helper_sha256=owner.GLOBAL_SHA256,
                   helper_entry=owner.GLOBAL_ENTRY,
                   uniform_failure_backtracking=False)
        self.cfg_path.write_text(json.dumps(bad))
        fake = Mock()
        with self.assertRaisesRegex(ValueError, "local mode requires helper path"):
            runner.execute_373(self.cfg_path, run=fake)
        fake.assert_not_called()

    def test_invalid_output_and_timeout_reject_before_launch(self):
        self.out.mkdir()
        fake = Mock()
        with self.assertRaisesRegex(ValueError, "already exists"):
            runner.execute_373(self.cfg_path, run=fake)
        fake.assert_not_called()
        self.out.rmdir()
        with self.assertRaises(ValueError):
            runner.execute_373(self.cfg_path, timeout_seconds=0, run=fake)
        fake.assert_not_called()

    def test_timeout_is_bounded_and_reported_as_124(self):
        fake = Mock(side_effect=subprocess.TimeoutExpired(["runner"], 9))
        receipt = runner.execute_373(self.cfg_path, timeout_seconds=9, run=fake)
        self.assertEqual(receipt["status"], "RUNNER_TIMEOUT")
        self.assertEqual(receipt["returncode"], 124)
        self.assertTrue(receipt["timed_out"])

    def test_nonzero_child_exit_is_preserved(self):
        fake = Mock(return_value=subprocess.CompletedProcess([], 17))
        receipt = runner.execute_373(self.cfg_path, run=fake)
        self.assertEqual(receipt["status"], "RUNNER_NONZERO_EXIT")
        self.assertEqual(receipt["returncode"], 17)


    def test_actual_424_graded_config_validates_and_injects_context_pins(self):
        actual = Path("/Users/numi/Work/numi-human-connected-surface-069/D/postfit-backtrack-424/run-001/backtrack-config.json")
        cfg = json.loads(actual.read_text())
        cfg["out"] = str(self.root / "graded-out")
        self.cfg_path.write_text(json.dumps(cfg))
        fake = Mock(return_value=subprocess.CompletedProcess([], 0))
        receipt = runner.execute_373(self.cfg_path, timeout_seconds=90, run=fake)
        self.assertEqual(receipt["status"], "RUNNER_EXITED")
        self.assertEqual(receipt["config_binding"]["mode"], "graded_local_423")
        self.assertTrue(receipt["graded_context_environment_injected"])
        kwargs = fake.call_args.kwargs
        self.assertEqual(kwargs["timeout"], 90)
        env = kwargs["env"]
        self.assertEqual(env["NUMI_GRADED_LOCAL_STEP_CONTEXT_423"], cfg["local_step_context_npz"])
        self.assertEqual(env["NUMI_GRADED_LOCAL_STEP_CONTEXT_SHA256_423"], cfg["local_step_context_sha256"])
        self.assertEqual(env["NUMI_GRADED_LOCAL_STEP_KAPPA_423"], "0.1")

    def test_actual_424_context_hash_mismatch_rejects_before_launch(self):
        actual = Path("/Users/numi/Work/numi-human-connected-surface-069/D/postfit-backtrack-424/run-001/backtrack-config.json")
        cfg = json.loads(actual.read_text())
        cfg["out"] = str(self.root / "bad-graded-out")
        cfg["local_step_context_sha256"] = "0" * 64
        self.cfg_path.write_text(json.dumps(cfg))
        fake = Mock()
        with self.assertRaisesRegex(ValueError, "context NPZ path/hash mismatch"):
            runner.execute_373(self.cfg_path, run=fake)
        fake.assert_not_called()

if __name__ == "__main__":
    unittest.main()
