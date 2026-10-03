"""Exercise scientific failure boundaries using explicitly synthetic fixtures."""
import copy
import importlib.util
import json
from pathlib import Path
import os
import shutil
import subprocess
import signal
import tarfile
import time
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("science", ROOT / "python/metalrobo/science_notebook.py")
science = importlib.util.module_from_spec(spec)
spec.loader.exec_module(science)


class ScienceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.study = self.root / "study"
        self.plan_file = self.root / "plan.json"
        self.instrument = self.root / "instrument.py"
        self.instrument.write_text("import json,sys\nprint(json.dumps({'value':float(sys.argv[1]),'valid':True,'seed':int(sys.argv[2])}))\n")
        self.model_file = self.root / "model.json"
        self.calibration_file = self.root / "calibration.json"
        self.predictor = self.root / "predictor.py"
        self.predictor.write_text("import json,sys,hashlib\np=sys.argv[1]\nm=json.load(open(p))\nprint(json.dumps({'schema':'numi.science.prediction.v1','model_sha256':hashlib.sha256(open(p,'rb').read()).hexdigest(),'prediction':m['prediction']}))\n")
        self.plan = {"schema": "numi.science.plan.v2", "purpose": "exploration", "question": "Fixture response?", "hypothesis": "Delta is two",
                     "owner": "test fixture", "backend": "Python", "evidence_level": "software",
                     "limitations": "Tests only", "repository": str(ROOT),
                     "model": {"version": "v1", "statement": "Delta=2", "training_units": []},
                     "model_file": str(self.model_file),
                     "predictor": {"argv": [sys.executable, str(self.predictor), str(self.model_file)], "env": {}, "timeout_seconds": 10},
                     "instrument": {"description": "Fixture reader", "calibration": str(self.calibration_file), "artifacts": [str(self.instrument)]},
                     "design": {"intervention": "value", "controls": "seed", "experimental_unit": "fixture",
                                "allocation": "fixed test order", "unit_paths": {"seed": ["seed"]}},
                     "artifacts": [str(self.instrument), str(self.calibration_file), str(self.model_file), str(self.predictor)], "observable": {"name": "value", "unit": "test", "path": ["value"]},
                     "prediction": {"estimand": "paired_difference_mean", "minimum": 1.9, "maximum": 2.1},
                     "validity": [{"path": ["valid"], "equals": True}], "paired_equal": [["seed"]],
                     "trials": [{"id": arm, "pair": "p", "arm": arm,
                                 "argv": [sys.executable, str(self.instrument), value, "7"],
                                 "unit": {"seed": 7}, "env": {}, "timeout_seconds": 10}
                                for arm, value in (("control", "1"), ("treatment", "3"))]}

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self):
        self.plan["model"]["prediction"] = copy.deepcopy(self.plan["prediction"])
        self.model_file.write_text(json.dumps(self.plan["model"]))
        self.calibration_file.write_text(json.dumps({"schema": "numi.science.calibration.v1", "status": "passed",
            "checks": [{"id": "software_fixture_only", "passed": True}], "scope": "synthetic parser fixture",
            "bindings": {str(self.instrument): science.file_hash(self.instrument)}}))
        self.plan_file.write_text(json.dumps(self.plan))

    def register(self):
        self.prepare()
        return science.register(self.plan_file, self.study)

    def revision(self, analysis):
        model = copy.deepcopy(self.plan["model"])
        model.update(version="v2", training_units=[{"seed": 7}])
        model_file = self.root / "revised-model.json"
        model_file.write_text(json.dumps(model))
        change = {"model": model, "model_file": str(model_file), "decision": "retain", "reason": "Observed two",
                  "next_test": "unused seed", "limitations": "software", "evidence": [analysis["sha256"]]}
        path = self.root / "revision.json"
        path.write_text(json.dumps(change))
        return path, model

    def finish(self):
        self.register()
        science.run_next(self.study)
        science.run_next(self.study)
        return science.analyze(self.study)

    def test_full_loop_and_new_model_lineage(self):
        analysis = self.finish()
        self.assertEqual(analysis["payload"]["verdict"], "supported")
        self.assertEqual(analysis["payload"]["mean_difference"], 2)
        path, model = self.revision(analysis)
        revision = science.revise(self.study, path)
        science.verify(self.study)
        self.plan["model"] = model
        self.plan["purpose"] = "confirmation"
        for trial in self.plan["trials"]:
            trial["unit"]["seed"] = 8
            trial["argv"][-1] = "8"
        self.prepare()
        followup = science.register(self.plan_file, self.root / "followup", self.study)
        self.assertEqual(followup["payload"]["parent"]["revision_sha256"], revision["sha256"])
        with self.assertRaises(science.Invalid):
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
        self.assertIn("TimeoutExpired", receipt["payload"]["failure"])
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
        self.assertIn("observed experimental unit", str(analysis["trials"]))

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
            command = subprocess.run([sys.executable, str(ROOT / "python/metalrobo/science_notebook.py"),
                                      "run", str(self.study)], capture_output=True, text=True)
        self.assertEqual(command.returncode, 2)
        self.assertIn("in use", command.stderr)

    def test_native_reader_calibration(self):
        result = subprocess.run([sys.executable, str(ROOT / "examples/science/neuron_weight_instrument.py"), "--calibrate"],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["rejected_invalid_inputs"], 10)

    def test_installed_layout_dispatches_without_source_tree(self):
        prefix = self.root / "installed"
        for directory in ("bin", "libexec/numi", "share/numi/python/metalrobo"):
            (prefix / directory).mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "tools/numi", prefix / "bin/numi")
        shutil.copy2(ROOT / "numi/commands/science", prefix / "libexec/numi/science")
        shutil.copy2(ROOT / "python/metalrobo/science_notebook.py", prefix / "share/numi/python/metalrobo/science_notebook.py")
        shutil.copy2(ROOT / "python/metalrobo/science.py", prefix / "share/numi/python/metalrobo/science.py")
        shutil.copytree(ROOT / "examples/science", prefix / "share/numi/examples/science", ignore=shutil.ignore_patterns('evidence', '__pycache__'))
        (prefix / 'share/numi/docs').mkdir()
        shutil.copy2(ROOT / 'docs/SCIENTIFIC_WORKFLOW.md', prefix / 'share/numi/docs/SCIENTIFIC_WORKFLOW.md')
        result = subprocess.run([str(prefix / "bin/numi"), "science", "--help"], cwd=self.root,
                                env={**os.environ, "NUMI_LAB_ROOT": str(prefix)}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("register", result.stdout)
        self.assertIn(str(prefix.resolve() / 'share/numi/docs/SCIENTIFIC_WORKFLOW.md'), ''.join(result.stdout.split()))

    def test_registered_legacy_work_can_finish_without_artifact_drift(self):
        spec = importlib.util.spec_from_file_location('legacy', ROOT / 'python/metalrobo/science.py')
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        self.plan['schema'] = 'numi.science.plan.v1'
        self.prepare()
        registration = legacy.register(self.plan_file, self.study)
        legacy.run_next(self.study)
        expected = registration['payload']['artifacts'][str(ROOT / 'python/metalrobo/science.py')]
        for command in ('run', 'analyze'):
            result = subprocess.run([sys.executable, str(ROOT / 'python/metalrobo/science_notebook.py'), command, str(self.study)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(science.file_hash(ROOT / 'python/metalrobo/science.py'), expected)
        self.assertEqual(science.unseal(self.study / 'analysis.json')['payload']['verdict'], 'supported')
        science.verify(self.study)

    def test_stale_parent_model_cannot_be_called_followup(self):
        self.test_full_loop_and_new_model_lineage()
        self.plan["model"] = {"version": "wrong", "statement": "uncited replacement"}
        self.plan_file.write_text(json.dumps(self.plan))
        with self.assertRaises(science.Invalid):
            science.register(self.plan_file, self.root / "bad-followup", self.study)

    def test_missing_or_stale_calibration_rejected_before_execution(self):
        self.prepare()
        self.calibration_file.unlink()
        with self.assertRaises(science.Invalid):
            science.register(self.plan_file, self.study)
        self.prepare()
        self.instrument.write_text('print(0)')
        with self.assertRaises(science.Invalid):
            science.register(self.plan_file, self.study)
        self.assertFalse(self.study.exists())

    def test_executable_prediction_must_match_model_and_declared_bounds(self):
        self.prepare()
        self.plan['prediction']['minimum'] = 0
        self.plan_file.write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(science.Invalid, 'executable model'):
            science.register(self.plan_file, self.study)
        self.assertFalse((self.study / 'registration.json').exists())
        self.assertTrue((self.study / 'prediction/output/stdout.json').exists())

    def test_deleted_trial_directory_never_becomes_a_fresh_trial(self):
        self.register()
        science.run_next(self.study)
        shutil.rmtree(self.study / 'trials/control')
        with self.assertRaisesRegex(science.Invalid, 'lost its directory'):
            science.run_next(self.study)

    def test_reused_training_seed_cannot_be_confirmation(self):
        analysis = self.finish()
        path, model = self.revision(analysis)
        science.revise(self.study, path)
        self.plan['model'] = model
        self.plan['purpose'] = 'confirmation'
        for trial in self.plan['trials']:
            trial['pair'] = 'renamed'
        self.prepare()
        with self.assertRaisesRegex(science.Invalid, 'reuses'):
            science.register(self.plan_file, self.root / 'followup', self.study)

    def test_revised_model_cannot_forget_exposed_conditions(self):
        analysis = self.finish()
        path, model = self.revision(analysis)
        change = science.read(path)
        change['model']['training_units'] = []
        path.write_text(json.dumps(change))
        with self.assertRaisesRegex(science.Invalid, 'retain all'):
            science.revise(self.study, path)

    def test_revised_parameters_require_matching_model_artifact(self):
        analysis = self.finish()
        path, model = self.revision(analysis)
        change = science.read(path)
        change['model']['prediction']['minimum'] = 0
        change['decision'] = 'revise'
        path.write_text(json.dumps(change))
        with self.assertRaisesRegex(science.Invalid, 'content differs'):
            science.revise(self.study, path)

    def test_child_retains_parent_but_detects_corrupt_snapshot(self):
        self.test_full_loop_and_new_model_lineage()
        child = self.root / 'followup'
        shutil.rmtree(self.study)
        science.verify(child)
        parent_revision = child / 'lineage/parent/revision.json'
        parent_revision.write_text('{}')
        with self.assertRaises((science.Invalid, KeyError)):
            science.verify(child)

    def test_archive_verifies_without_original_inputs_or_study(self):
        self.finish()
        archive = self.root / 'portable'
        science.archive(self.study, archive)
        shutil.rmtree(self.study)
        for path in (self.instrument, self.model_file, self.calibration_file, self.predictor):
            path.unlink()
        self.assertEqual(science.status(archive)['recorded_trials'], 2)
        obj = next((archive / 'objects').iterdir())
        obj.write_bytes(b'corrupt')
        with self.assertRaisesRegex(science.Invalid, 'snapshot'):
            science.verify(archive)

    def test_unsafe_output_is_retained_with_failed_terminal_receipt(self):
        for code in ("import os; os.symlink('/does/not/exist','link')", "import os; os.mkfifo('pipe')"):
            with self.subTest(code=code):
                self.study = self.root / ('unsafe' + str(len(list(self.root.iterdir()))))
                self.plan['trials'][0]['argv'] = [sys.executable, '-c', code]
                self.register()
                receipt = science.run_next(self.study)
                self.assertIn('symlink or special file', receipt['payload']['failure'])
                science.verify(self.study)
                science.stop(self.study, 'unsafe output')
                self.assertEqual(science.analyze(self.study)['payload']['verdict'], 'inconclusive')

    def test_successful_leader_cannot_leave_unsealed_writers(self):
        self.plan['trials'][0]['argv'] = [sys.executable, '-c',
            "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])"]
        self.register()
        receipt = science.run_next(self.study)
        self.assertIn('descendants', receipt['payload']['failure'])
        process = science.unseal(self.study / 'trials/control/process.json')['payload']
        self.assertEqual(science.group_members(process['pgid']), [])
        science.verify(self.study)

    def start_interruptible_trial(self):
        self.plan['trials'][0]['argv'] = [sys.executable, '-c',
            "import time; print('partial observation',flush=True); time.sleep(2)"]
        self.register()
        runner = subprocess.Popen([sys.executable, str(ROOT / 'python/metalrobo/science_notebook.py'), 'run', str(self.study)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        process_file = self.study / 'trials/control/process.json'
        deadline = time.monotonic() + 10
        while not process_file.exists() and runner.poll() is None and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertTrue(process_file.exists())
        return runner

    def test_sigterm_retains_partial_output_and_closes_process(self):
        runner = self.start_interruptible_trial()
        runner.terminate()
        stdout, stderr = runner.communicate(timeout=10)
        self.assertEqual(runner.returncode, 1, stderr)
        receipt = json.loads(stdout)
        self.assertIn('TrialInterrupted', receipt['payload']['failure'])
        science.verify(self.study)

    def test_sigkill_recovery_refuses_live_worker_and_never_retries(self):
        runner = self.start_interruptible_trial()
        runner.kill()
        runner.communicate(timeout=10)
        with self.assertRaisesRegex(science.Invalid, 'still alive'):
            science.recover(self.study, 'killed test runner')
        info = science.unseal(self.study / 'trials/control/process.json')['payload']
        deadline = time.monotonic() + 10
        while science.group_members(info['pgid']) and time.monotonic() < deadline:
            time.sleep(.05)
        receipt = science.recover(self.study, 'killed test runner')
        self.assertIn('recovered interrupted', receipt['payload']['failure'])
        science.stop(self.study, 'stop after interruption')
        self.assertEqual(science.analyze(self.study)['payload']['verdict'], 'inconclusive')
        with self.assertRaises(science.Invalid):
            science.run_next(self.study)

    def test_resealed_environment_change_is_rejected(self):
        self.register()
        science.run_next(self.study)
        path = self.study / 'trials/control/receipt.json'
        payload = science.unseal(path)['payload']
        payload['environment']['NEW'] = 'unregistered'
        path.unlink()
        science.seal(path, payload)
        with self.assertRaisesRegex(science.Invalid, 'environment'):
            science.verify(self.study)

    def test_executable_model_fit_changes_its_next_prediction(self):
        program = ROOT / 'examples/science/weight_effect_model.py'
        model_spec = importlib.util.spec_from_file_location('weight_model', program)
        model_module = importlib.util.module_from_spec(model_spec)
        model_spec.loader.exec_module(model_module)
        self.plan['model'] = model_module.initial_model()
        self.plan['prediction'] = model_module.prediction(self.plan['model'])
        self.plan['predictor']['argv'] = [sys.executable, str(program), 'predict', '--model', str(self.model_file)]
        self.plan['artifacts'].append(str(program))
        original = copy.deepcopy(self.plan['trials'])
        self.plan['trials'] = []
        for seed in (7, 8, 9):
            for template in original:
                trial = copy.deepcopy(template)
                trial.update(id=str(seed) + '-' + trial['arm'], pair=str(seed), unit={'seed': seed})
                trial['argv'][-1] = str(seed)
                self.plan['trials'].append(trial)
        self.register()
        for trial in self.plan['trials']:
            science.run_next(self.study)
        self.assertEqual(science.analyze(self.study)['payload']['verdict'], 'contradicted')
        model_path, revision_path = self.root / 'fit.json', self.root / 'fit-revision.json'
        subprocess.run([sys.executable, str(program), 'fit', '--study', str(self.study), '--output', str(model_path),
                        '--revision', str(revision_path)], check=True, capture_output=True)
        result = subprocess.run([sys.executable, str(program), 'predict', '--model', str(model_path)],
                                check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(result.stdout)['prediction'],
                         {'estimand': 'paired_difference_mean', 'minimum': 2, 'maximum': 2})
        science.revise(self.study, revision_path)
        science.verify(self.study)

    def test_calibration_conditions_cannot_be_test_units(self):
        self.prepare()
        report = science.read(self.calibration_file)
        report['observed_units'] = [{'seed': 7}]
        self.calibration_file.write_text(json.dumps(report))
        with self.assertRaisesRegex(science.Invalid, 'calibration conditions'):
            science.register(self.plan_file, self.study)

    def test_legacy_native_evidence_remains_readable(self):
        for name in ('discovery', 'confirmation', 'recovery-check'):
            study = ROOT / 'examples/science/evidence/20261003' / name / 'study'
            result = science.status(study)
            self.assertIn('legacy', result['guarantees'])
            legacy_copy = self.root / name
            shutil.copytree(study, legacy_copy)
            with self.assertRaises(science.Invalid):
                science.run_next(legacy_copy)

    def test_published_native_archive_verifies_all_parent_evidence(self):
        directory = ROOT / 'examples/science/evidence'
        summary = science.read(directory / '20261003-v2-summary.json')
        source = directory / summary['archive']['file']
        self.assertEqual(science.file_hash(source), summary['archive']['sha256'])
        with tarfile.open(source) as archive:
            for member in archive.getmembers():
                self.assertTrue(member.isfile() or member.isdir())
                self.assertIn(self.root.resolve(), (self.root / member.name).resolve().parents)
            archive.extractall(self.root)
        study = self.root / 'final-confirmation'
        result = science.status(study)
        self.assertEqual(result['state'], 'revision')
        self.assertEqual(result['recorded_trials'], 6)
        self.assertEqual(result['registration_sha256'], summary['studies'][-1]['registration_sha256'])
        for item in reversed(summary['studies']):
            analysis = science.unseal(study / 'analysis.json')
            self.assertEqual(analysis['sha256'], item['analysis_sha256'])
            self.assertEqual(analysis['payload']['verdict'], item['verdict'])
            study = study / 'lineage/parent'


if __name__ == "__main__":
    unittest.main()
