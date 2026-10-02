"""Exercise scientific failure boundaries using explicitly synthetic fixtures."""
import copy
import importlib.util
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("science", ROOT / "python/metalrobo/science.py")
science = importlib.util.module_from_spec(spec)
spec.loader.exec_module(science)


class ScienceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.study = self.root / "study"
        self.plan_file = self.root / "plan.json"
        self.instrument = self.root / "instrument.py"
        self.instrument.write_text("import json,sys\nprint(json.dumps({'value':float(sys.argv[1]),'valid':True,'seed':7}))\n")
        self.plan = {"schema": "numi.science.plan.v1", "question": "Fixture response?", "hypothesis": "Delta is two",
                     "owner": "test fixture", "backend": "Python", "evidence_level": "software",
                     "limitations": "Tests only", "repository": str(ROOT),
                     "model": {"version": "v1", "statement": "Delta=2"},
                     "instrument": {"description": "Fixture reader", "calibration": "Known values"},
                     "design": {"intervention": "value", "controls": "seed", "experimental_unit": "fixture",
                                "allocation": "fixed test order"},
                     "artifacts": [str(self.instrument)], "observable": {"name": "value", "unit": "test", "path": ["value"]},
                     "prediction": {"estimand": "paired_difference_mean", "minimum": 1.9, "maximum": 2.1},
                     "validity": [{"path": ["valid"], "equals": True}], "paired_equal": [["seed"]],
                     "trials": [{"id": arm, "pair": "p", "arm": arm,
                                 "argv": [sys.executable, str(self.instrument), value],
                                 "env": {}, "timeout_seconds": 10}
                                for arm, value in (("control", "1"), ("treatment", "3"))]}

    def tearDown(self):
        self.temp.cleanup()

    def register(self):
        self.plan_file.write_text(json.dumps(self.plan))
        return science.register(self.plan_file, self.study)

    def finish(self):
        self.register()
        science.run_next(self.study)
        science.run_next(self.study)
        return science.analyze(self.study)

    def test_full_loop_and_new_model_lineage(self):
        analysis = self.finish()
        self.assertEqual(analysis["payload"]["verdict"], "supported")
        self.assertEqual(analysis["payload"]["mean_difference"], 2)
        model = {"version": "v2", "statement": "Delta=2 on this fixture only"}
        change = {"model": model, "decision": "retain", "reason": "Observed two",
                  "next_test": "new fixtures", "limitations": "software", "evidence": [analysis["sha256"]]}
        path = self.root / "revision.json"
        path.write_text(json.dumps(change))
        revision = science.revise(self.study, path)
        science.verify(self.study)
        self.plan["model"] = model
        self.plan_file.write_text(json.dumps(self.plan))
        followup = science.register(self.plan_file, self.root / "followup", self.study)
        self.assertEqual(followup["payload"]["parent"]["revision_sha256"], revision["sha256"])
        with self.assertRaises(FileExistsError):
            science.revise(self.study, path)

    def test_contradicted_prediction_is_retained(self):
        self.plan["prediction"].update(minimum=3, maximum=4)
        self.assertEqual(self.finish()["payload"]["verdict"], "contradicted")

    def test_failed_trial_cannot_be_dropped_or_rerun(self):
        self.plan["trials"][1]["argv"] = [sys.executable, "-c", "raise SystemExit(9)"]
        analysis = self.finish()["payload"]
        self.assertEqual(analysis["verdict"], "inconclusive")
        self.assertEqual(analysis["trials"][1]["status"], "invalid")
        with self.assertRaises(science.Invalid):
            science.run_next(self.study)

    def test_timeout_retained_with_output(self):
        self.plan["trials"][0]["argv"] = [sys.executable, "-c", "import time; print('begun',flush=True); time.sleep(10)"]
        self.plan["trials"][0]["timeout_seconds"] = 0.1
        self.register()
        receipt = science.run_next(self.study)
        self.assertEqual(receipt["payload"]["failure"], "TimeoutExpired")
        self.assertIn("begun", (self.study / "trials/control/output/stdout.json").read_text())

    def test_missing_trial_prevents_final_analysis(self):
        self.register()
        science.run_next(self.study)
        with self.assertRaises(science.Invalid):
            science.analyze(self.study)

    def test_fault_stop_preserves_missing_trials_as_inconclusive(self):
        self.register()
        science.run_next(self.study)
        science.stop(self.study, "instrument fault requires a replacement study")
        with self.assertRaises(science.Invalid):
            science.run_next(self.study)
        analysis = science.analyze(self.study)["payload"]
        self.assertEqual(analysis["verdict"], "inconclusive")
        self.assertEqual(analysis["trials"][1]["status"], "missing")
        science.verify(self.study)

    def test_unfinished_trial_never_restarted(self):
        self.register()
        (self.study / "trials/control").mkdir()
        with self.assertRaises((science.Invalid, OSError)):
            science.run_next(self.study)

    def test_bound_instrument_drift_prevents_launch(self):
        self.register()
        self.instrument.write_text("raise SystemExit(0)")
        with self.assertRaises(science.Invalid):
            science.run_next(self.study)
        self.assertFalse((self.study / "trials/control").exists())

    def test_raw_evidence_tampering_is_detected(self):
        self.finish()
        (self.study / "trials/control/output/stdout.json").write_text('{"value": 0}')
        with self.assertRaises(science.Invalid):
            science.verify(self.study)

    def test_analysis_recomputed_even_if_resealed(self):
        analysis = self.finish()
        analysis["payload"]["verdict"] = "contradicted"
        analysis["sha256"] = science.digest(analysis["payload"])
        (self.study / "analysis.json").write_text(json.dumps(analysis))
        with self.assertRaises(science.Invalid):
            science.verify(self.study)

    def test_nan_boolean_and_missing_observables_are_invalid(self):
        for value in ("float('nan')", "True", "None"):
            with self.subTest(value=value):
                self.study = self.root / ("invalid" + str(len(list(self.root.iterdir()))))
                self.plan["trials"][1]["argv"] = [sys.executable, "-c",
                    "import json; print(json.dumps({'value':" + value + ",'valid':True,'seed':7}))"]
                self.assertEqual(self.finish()["payload"]["verdict"], "inconclusive")

    def test_unmatched_control_cannot_support_prediction(self):
        self.plan["trials"][1]["argv"] = [sys.executable, "-c", "print('{\"value\":3,\"valid\":true,\"seed\":8}')"]
        analysis = self.finish()["payload"]
        self.assertEqual(analysis["verdict"], "inconclusive")
        self.assertIn("paired control mismatch", str(analysis["issues"]))

    def test_duplicate_and_traversal_ids_rejected(self):
        for identity in ("control", "../escape"):
            self.plan["trials"][1]["id"] = identity
            with self.assertRaises(science.Invalid):
                self.register()

    def test_no_shell_interpolation(self):
        output = self.root / "injected"
        self.plan["trials"][0]["argv"] = [sys.executable, "-c", "import sys; print(sys.argv[1])", "$(touch " + str(output) + ")"]
        self.register()
        science.run_next(self.study)
        self.assertFalse(output.exists())

    def test_concurrent_mutation_is_rejected(self):
        self.register()
        with science.locked(self.study):
            command = subprocess.run([sys.executable, str(ROOT / "python/metalrobo/science.py"),
                                      "run", str(self.study)], capture_output=True, text=True)
        self.assertEqual(command.returncode, 2)
        self.assertIn("in use", command.stderr)

    def test_native_reader_calibration(self):
        result = subprocess.run([sys.executable, str(ROOT / "examples/science/neuron_weight_instrument.py"), "--calibrate"],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["rejected_invalid_inputs"], 5)

    def test_installed_layout_dispatches_without_source_tree(self):
        prefix = self.root / "installed"
        for directory in ("bin", "libexec/numi", "share/numi/python/metalrobo"):
            (prefix / directory).mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "tools/numi", prefix / "bin/numi")
        shutil.copy2(ROOT / "numi/commands/science", prefix / "libexec/numi/science")
        shutil.copy2(ROOT / "python/metalrobo/science.py", prefix / "share/numi/python/metalrobo/science.py")
        result = subprocess.run([str(prefix / "bin/numi"), "science", "--help"], cwd=self.root,
                                env={**os.environ, "NUMI_LAB_ROOT": str(prefix)}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("register", result.stdout)

    def test_stale_parent_model_cannot_be_called_followup(self):
        self.test_full_loop_and_new_model_lineage()
        self.plan["model"] = {"version": "wrong", "statement": "uncited replacement"}
        self.plan_file.write_text(json.dumps(self.plan))
        with self.assertRaises(science.Invalid):
            science.register(self.plan_file, self.root / "bad-followup", self.study)


if __name__ == "__main__":
    unittest.main()
