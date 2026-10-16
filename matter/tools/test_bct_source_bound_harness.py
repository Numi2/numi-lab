#!/usr/bin/env python3
"""Synthetic-only end-to-end check for pre-run BCT binding and packaging."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
RUNNER = TOOLS / "cardboard_source_bound_run.py"
SUPPORT = TOOLS / "cardboard_source_bound_run_support.py"
BINDING = TOOLS / "cardboard_calibration_case_binding.py"
ENVIRONMENT = TOOLS / "cardboard_solver_environment.py"
PRODUCER_TEST_PATH = TOOLS / "test_bct_capacity_history_producer.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


producer_test = load_module(PRODUCER_TEST_PATH, "bct_producer_fixture_for_harness")
producer = producer_test.producer
ensemble = producer_test.ensemble


class SourceBoundHarnessTests(unittest.TestCase):
    def _fake_source(self, root: Path) -> tuple[Path, Path]:
        tools = root / "matter" / "tools"
        tools.mkdir(parents=True)
        for source in (RUNNER, SUPPORT, BINDING, ENVIRONMENT, TOOLS / "bct_capacity_history_producer.py",
                       TOOLS / "cardboard_calibration_ensemble.py"):
            shutil.copy2(source, tools / source.name)
        (root / "matter" / "materials").mkdir(parents=True)
        (root / "matter" / "examples" / "cardboard" / "source-data").mkdir(parents=True)
        (root / "matter" / "CMakeLists.txt").write_text("# synthetic harness fixture\n")
        runtime = root / "runtime"
        runtime.mkdir()
        return root, runtime

    def _native_script(self, root: Path, template_output: Path, *,
                       mutate_manifest: bool = False,
                       mutate_qualification: bool = False) -> Path:
        script = root / "fake_native.py"
        script.write_text(
            "#!/usr/bin/env python3\n"
            "import argparse, json, shutil\n"
            "from pathlib import Path\n"
            "p=argparse.ArgumentParser()\n"
            "p.add_argument('--shipping-box', action='store_true')\n"
            "p.add_argument('--shipping-score-depth-ratio', required=True)\n"
            "p.add_argument('--shipping-capacity-qualification', required=True)\n"
            "p.add_argument('--output', required=True)\n"
            "a=p.parse_args()\n"
            f"shutil.copytree({str(template_output)!r}, Path(a.output))\n"
            "Path(a.output).parent.joinpath('fake-native-ran.txt').write_text('ran\\n')\n"
            + ("m=Path(a.output)/'manifest.json'; d=json.loads(m.read_text()); "
               "d['shipping_score_command']['selected_depth_ratio_of_caliper']=0.3; "
               "m.write_text(json.dumps(d, sort_keys=True)+'\\n')\n" if mutate_manifest else "")
            + ("Path(a.shipping_capacity_qualification).write_text('tampered after launch\\n')\n"
               if mutate_qualification else "")
        )
        script.chmod(0o755)
        return script

    def _run_harness(self, base: Path, run: Path, plan: Path, binding: Path,
                     qualification: Path, *, argv_value: str = "0.25",
                     mutate_manifest: bool = False,
                     mutate_qualification: bool = False) -> tuple[Path, subprocess.CompletedProcess[str]]:
        source, runtime = self._fake_source(base / "fake-checkout")
        script = self._native_script(source, run / "output", mutate_manifest=mutate_manifest,
                                     mutate_qualification=mutate_qualification)
        case = base / ("harness-mutated" if mutate_manifest else
                       "harness-input-mutated" if mutate_qualification else "harness-run")
        command = [
            sys.executable, str(source / "matter/tools/cardboard_source_bound_run.py"),
            "--case", str(case), "--question", "synthetic harness integration test",
            "--prediction", "candidate parameter maps to the executed manifest",
            "--source-snapshot", str(source), "--runtime-dir", str(runtime),
            "--ensemble-plan", str(plan), "--case-binding", str(binding),
            "--timeout", "30", "--", str(script), "--shipping-box",
            "--shipping-score-depth-ratio", argv_value,
            "--shipping-capacity-qualification", str(qualification),
            "--output", str(case / "output"),
        ]
        result = subprocess.run(command, text=True, capture_output=True, timeout=60)
        return case, result

    def _fixture(self, base: Path):
        run, plan, binding, candidate, condition, capacity_binding = producer_test.make_run(base)
        return run, plan, binding, candidate, condition, capacity_binding

    def test_harness_to_packager_to_evaluator_positive(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            template, plan, binding, candidate, condition, _ = self._fixture(base)
            qualification = template.parent / "shipping-capacity-qualification.v2"
            run, result = self._run_harness(base, template, plan, binding, qualification)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            prediction_path, receipt_path = run / "prediction.json", run / "receipt.json"
            prediction = json.loads(prediction_path.read_text())
            receipt = json.loads(receipt_path.read_text())
            environment = json.loads((run / "native-environment.json").read_text())
            self.assertEqual(prediction["native_protocol_id"], producer.PROTOCOL_ID)
            self.assertEqual(prediction["calibration_case_binding"]["candidate_sha256"],
                             candidate["candidate_sha256"])
            self.assertEqual(receipt["calibration_parameter_validation"]["status"], "passed")
            self.assertTrue(environment["launch"]["native_controls_scrubbed"])
            self.assertEqual(environment["contract_sha256"],
                             prediction["effective_native_environment_contract_sha256"])
            self.assertEqual(environment["prediction_sha256"],
                             __import__("hashlib").sha256(prediction_path.read_bytes()).hexdigest())
            self.assertEqual(receipt["calibration_parameter_validation"]["manifest_sha256"],
                             producer.sha256(run / "output/manifest.json"))
            raw_receipt_sha = producer.sha256(receipt_path)
            raw_manifest_sha = producer.sha256(run / "output/manifest.json")
            package = plan.parent / "native_histories" / candidate["candidate_id"] / condition["condition_id"]
            producer.finalize_run(run, plan, binding, package,
                                  producer_test.CATALOG, producer_test.DATA_DIR)
            self.assertEqual(raw_receipt_sha, producer.sha256(receipt_path), "raw runner receipt was mutated")
            self.assertEqual(raw_manifest_sha, producer.sha256(run / "output/manifest.json"),
                             "raw native manifest was mutated")
            history = json.loads((package / "history.json").read_text())
            certificate = json.loads((package / "certificate.json").read_text())
            manifest = json.loads((package / "native_input_manifest.json").read_text())
            packaged_receipt = json.loads((package / "run_receipt.json").read_text())
            self.assertEqual(history["native_protocol_id"], producer.PROTOCOL_ID)
            self.assertEqual(manifest["pre_run_calibration_execution"]["case_binding_sha256"],
                             producer.sha256(binding))
            ensemble._verify_bct_capacity_evidence(history, certificate, manifest,
                                                   packaged_receipt, package)
            expected = {
                "candidate_id": candidate["candidate_id"],
                "candidate_sha256": candidate["candidate_sha256"],
                "condition_id": condition["condition_id"],
                "specimen_id": condition["specimen_id"],
                "native_protocol_id": condition["native_protocol_id"],
                "source_data_catalog_sha256": json.loads(plan.read_text())["source_data_catalog_sha256"],
                "source_dataset_sha256": condition["source_dataset_sha256"],
                "source_asset_sha256": condition["source_asset_sha256"],
                "dataset_id": condition["dataset_id"],
                "target_recipe_id": "literature2009_cross_source_recipe",
                "parameters": candidate["parameters"],
            }
            record = json.loads((package / "plan-record.json").read_text())
            ensemble._verify_certificate(record, plan.parent, expected)

    def test_raw_manifest_mutation_is_rejected_by_harness_and_packager(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            template, plan, binding, candidate, condition, _ = self._fixture(base)
            qualification = template.parent / "shipping-capacity-qualification.v2"
            run, result = self._run_harness(base, template, plan, binding, qualification,
                                            mutate_manifest=True)
            self.assertEqual(result.returncode, 3, result.stderr + result.stdout)
            prediction = json.loads((run / "prediction.json").read_text())
            receipt = json.loads((run / "receipt.json").read_text())
            self.assertEqual(receipt["exit_code"], 0, "native process exit must remain distinct")
            self.assertEqual(receipt["calibration_parameter_validation"]["status"], "failed")
            self.assertIn("differs from candidate", receipt["calibration_parameter_validation"]["error"])
            before = producer.sha256(run / "output/box-native-result.json")
            package = plan.parent / "rejected-package"
            with self.assertRaises(producer.ProducerError):
                producer.finalize_run(run, plan, binding, package,
                                      producer_test.CATALOG, producer_test.DATA_DIR)
            self.assertFalse(package.exists())
            self.assertEqual(before, producer.sha256(run / "output/box-native-result.json"))

    def test_external_binding_mutation_during_native_run_is_retained_and_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            template, plan, binding, candidate, condition, _ = self._fixture(base)
            qualification = template.parent / "shipping-capacity-qualification.v2"
            run, result = self._run_harness(base, template, plan, binding, qualification,
                                            mutate_qualification=True)
            self.assertEqual(result.returncode, 3, result.stderr + result.stdout)
            prediction = json.loads((run / "prediction.json").read_text())
            receipt = json.loads((run / "receipt.json").read_text())
            self.assertFalse(receipt["source_inputs_unchanged"])
            copied_sha = prediction["inputs_sha256"]["external/" + qualification.name]
            self.assertNotEqual(copied_sha, producer.sha256(qualification))
            package = plan.parent / "external-mutation-rejected"
            with self.assertRaises(producer.ProducerError):
                producer.finalize_run(run, plan, binding, package,
                                      producer_test.CATALOG, producer_test.DATA_DIR)
            self.assertFalse(package.exists())

    def test_argv_candidate_mismatch_is_rejected_before_native_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            template, plan, binding, candidate, condition, _ = self._fixture(base)
            qualification = template.parent / "shipping-capacity-qualification.v2"
            run, result = self._run_harness(base, template, plan, binding, qualification,
                                            argv_value="0.30")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((run / "fake-native-ran.txt").exists())
            self.assertFalse((run / "output" / "manifest.json").exists())
            failure = json.loads((run / "receipt.json").read_text())
            self.assertFalse(failure["execution_started"])

    def test_prepared_binding_mutation_after_run_cannot_be_repackaged(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            template, plan, binding, candidate, condition, _ = self._fixture(base)
            qualification = template.parent / "shipping-capacity-qualification.v2"
            run, result = self._run_harness(base, template, plan, binding, qualification)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            original = binding.read_text()
            binding.write_text(original + " ")
            package = plan.parent / "must-reject-mutated-case-binding"
            with self.assertRaisesRegex(producer.ProducerError, "digest|hash|pre-run"):
                producer.finalize_run(run, plan, binding, package,
                                      producer_test.CATALOG, producer_test.DATA_DIR)
            self.assertFalse(package.exists())
            self.assertEqual((run / "receipt.json").is_file(), True)

    def test_nonexistent_candidate_manifest_mapping_is_rejected_after_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            template, plan, binding, candidate, condition, _ = self._fixture(base)
            body = json.loads(binding.read_text())
            body["parameter_bindings"]["score_depth_ratio"]["pointer"] = "/unrelated/value"
            binding.write_text(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
            # The schema-valid pointer is provisionally sealed before the run;
            # the raw executed manifest must resolve it to the candidate value.
            qualification = template.parent / "shipping-capacity-qualification.v2"
            run, result = self._run_harness(base, template, plan, binding, qualification)
            self.assertEqual(result.returncode, 3, result.stderr + result.stdout)
            self.assertTrue((run / "fake-native-ran.txt").exists())
            failure = json.loads((run / "receipt.json").read_text())
            self.assertTrue(failure["execution_started"])
            self.assertEqual(failure["calibration_parameter_validation"]["status"], "failed")
            self.assertIn("JSON pointer does not resolve",
                          failure["calibration_parameter_validation"]["error"])
            with self.assertRaises(producer.ProducerError):
                producer.finalize_run(run, plan, binding, plan.parent / "bad-pointer-package",
                                      producer_test.CATALOG, producer_test.DATA_DIR)

    def test_duplicate_manifest_mapping_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            template, plan, binding, candidate, condition, _ = self._fixture(base)
            body = json.loads(binding.read_text())
            body["candidate"]["parameters"]["second_parameter"] = {
                "value": 0.25, "unit": "dimensionless", "family": "tooling", "fit": True}
            candidate_body = {key: value for key, value in body["candidate"].items()
                              if key != "candidate_sha256"}
            body["candidate"]["candidate_sha256"] = producer_test.producer.canonical_sha256(candidate_body)
            body["parameter_bindings"]["second_parameter"] = dict(
                body["parameter_bindings"]["score_depth_ratio"])
            # Keep this internally plan-consistent so the uniqueness guard,
            # rather than a stale candidate hash, is the rejecting layer.
            plan_data = json.loads(plan.read_text())
            plan_data["candidates"][0] = body["candidate"]
            write = json.dumps(plan_data, sort_keys=True, separators=(",", ":")) + "\n"
            plan.write_text(write)
            body["plan_sha256"] = producer_test.producer.sha256(plan)
            binding.write_text(json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
            qualification = template.parent / "shipping-capacity-qualification.v2"
            run, result = self._run_harness(base, template, plan, binding, qualification)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((run / "fake-native-ran.txt").exists())
            failure = json.loads((run / "receipt.json").read_text())
            self.assertFalse(failure["execution_started"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
