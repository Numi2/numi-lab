from __future__ import annotations

import unittest
import os
import tempfile
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cardboard_solver_environment as env


SOURCE_ROOT = Path(os.environ.get("CARDBOARD_TEST_SOURCE_ROOT", Path(__file__).resolve().parents[2]))


class EffectiveEnvironmentTests(unittest.TestCase):
    def test_registry_matches_checkout_and_detects_unregistered_control(self):
        env.validate_source_registry(SOURCE_ROOT)
        self.assertEqual(env.source_key_registry(SOURCE_ROOT), env.NATIVE_ENV_KEYS)
        self.assertEqual(len(env.NATIVE_ENV_KEYS), 51)
        self.assertIn("NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR", env.NATIVE_ENV_KEYS)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("matter/src/runtime.mm", "matter/tools/cardboard_probe.mm"):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(SOURCE_ROOT / relative, target)
            with (root / "matter/src/runtime.mm").open("a") as stream:
                stream.write('\nconst char *unregistered = "NM_FEM_UNREGISTERED_TEST_CONTROL";\n')
            with self.assertRaisesRegex(ValueError, "unlisted=.*NM_FEM_UNREGISTERED_TEST_CONTROL"):
                env.validate_source_registry(root)
        contract = env.make_contract()
        self.assertTrue(all(state == {"set": False, "value": None}
                            for state in contract["effective"].values()))
        contract = env.make_contract()
        self.assertEqual(set(contract["effective"]), set(env.NATIVE_ENV_KEYS))
        self.assertTrue(all(state == {"set": False, "value": None}
                            for state in contract["effective"].values()))

    def test_contact_control_binding_and_inherited_scrubbing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("matter/src/runtime.mm", "matter/tools/cardboard_probe.mm"):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(SOURCE_ROOT / relative, target)
            with (root / "matter/src/runtime.mm").open("a") as stream:
                stream.write('\nconst char *contact = std::getenv("NM_CONTACT_UNREGISTERED_TEST_CONTROL");\n')
            with self.assertRaisesRegex(ValueError, "unlisted=.*NM_CONTACT_UNREGISTERED_TEST_CONTROL"):
                env.validate_source_registry(root)
        absent = env.make_contract()
        selected = env.make_contract({"NM_CONTACT_NODE_INCIDENCE_SORT": "1"})
        self.assertNotEqual(absent["sha256"], selected["sha256"])
        self.assertEqual(absent["effective"]["NM_CONTACT_NODE_INCIDENCE_SORT"],
                         {"set": False, "value": None})
        parent = {"PATH": "/bin", "NM_CONTACT_NODE_INCIDENCE_SORT": "1",
                  "NM_CONTACT_UNREGISTERED_LEAK": "private-value"}
        child, removed = env.child_environment(parent, absent)
        self.assertEqual(child, {"PATH": "/bin"})
        self.assertEqual(removed, ["NM_CONTACT_NODE_INCIDENCE_SORT",
                                   "NM_CONTACT_UNREGISTERED_LEAK"])
        record = env.make_run_record(absent, producer="source-bound-native-runner",
                                     removed_inherited_keys=removed,
                                     prediction_sha256="0" * 64)
        env.validate_run_record(record)
        self.assertNotIn("private-value", json.dumps(record))
        selected_child, _ = env.child_environment(parent, selected)
        self.assertEqual(selected_child["NM_CONTACT_NODE_INCIDENCE_SORT"], "1")
        for value in ("", "0", "true", "2"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "exactly 1"):
                env.make_contract({"NM_CONTACT_NODE_INCIDENCE_SORT": value})
        with self.assertRaisesRegex(ValueError, "unknown native environment key"):
            env.make_contract({"NM_CONTACT_UNKNOWN_CONTROL": "1"})
        historical = env.make_contract()
        historical["keys"].remove("NM_CONTACT_NODE_INCIDENCE_SORT")
        del historical["effective"]["NM_CONTACT_NODE_INCIDENCE_SORT"]
        historical["sha256"] = env._contract_digest(historical)
        with self.assertRaisesRegex(ValueError, "registry differs"):
            env.validate_contract(historical)

    def test_absent_empty_and_explicit_values_are_distinct(self):
        absent = env.make_contract()
        empty = env.make_contract({"NM_FEM_TRUE_RESIDUAL_ROOT": ""})
        selected = env.make_contract({"NM_FEM_TRUE_RESIDUAL_ROOT": "all"})
        self.assertNotEqual(absent["sha256"], empty["sha256"])
        self.assertNotEqual(empty["sha256"], selected["sha256"])
        self.assertEqual(empty["effective"]["NM_FEM_TRUE_RESIDUAL_ROOT"], {"set": True, "value": ""})

    def test_inherited_controls_are_scrubbed_but_unrelated_environment_is_kept(self):
        contract = env.make_contract({"NM_FEM_BASE_CGLS_ITERATIONS": "64"})
        parent = {"PATH": "/bin", "NM_FEM_BASE_CGLS_ITERATIONS": "8",
                  "NM_FEM_UNREGISTERED_LEAK": "secret-value",
                  "NUMI_CARDBOARD_STEP_TIMING": "1", "USER": "tester"}
        child, removed = env.child_environment(parent, contract)
        self.assertEqual(child["NM_FEM_BASE_CGLS_ITERATIONS"], "64")
        self.assertNotIn("NM_FEM_UNREGISTERED_LEAK", child)
        self.assertNotIn("NUMI_CARDBOARD_STEP_TIMING", child)
        self.assertEqual(child["PATH"], "/bin")
        self.assertEqual(child["USER"], "tester")
        self.assertEqual(removed, ["NM_FEM_BASE_CGLS_ITERATIONS", "NM_FEM_UNREGISTERED_LEAK",
                                   "NUMI_CARDBOARD_STEP_TIMING"])
        self.assertNotIn("secret-value", str(env.make_run_record(
            contract, producer="test", removed_inherited_keys=removed,
            prediction_sha256="0" * 64)))

    def test_rejects_malformed_duplicate_unknown_and_capture_paths(self):
        for entries in (("NM_FEM_TRUE_RESIDUAL_ROOT",),
                        ("NM_FEM_TRUE_RESIDUAL_ROOT=1", "NM_FEM_TRUE_RESIDUAL_ROOT=2"),
                        ("NM_FEM_NOT_IN_SOURCE=1",),
                        ("NM_FEM_OPERATOR_FD_CAPTURE_PATH=/tmp/capture",),
                        ("NM_FEM_CONTACT_TRACE_ROOT=/tmp/trace",),
                        ("bad-key=1",)):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                env.parse_overrides(entries)

    def test_trace_root_is_numeric_control_selector_not_output_path(self):
        # Newton trace is numeric-only; Armijo roots support the source-defined `all` token.
        numeric = env.make_contract({"NM_FEM_CONTACT_TRACE_ROOT": "48"})
        every = env.make_contract({"NM_FEM_RESIDUAL_ARMIJO_ROOT": "all"})
        self.assertEqual(numeric["effective"]["NM_FEM_CONTACT_TRACE_ROOT"],
                         {"set": True, "value": "48"})
        self.assertEqual(every["effective"]["NM_FEM_RESIDUAL_ARMIJO_ROOT"],
                         {"set": True, "value": "all"})
        with self.assertRaisesRegex(ValueError, "numeric control-step selector"):
            env.make_contract({"NM_FEM_CONTACT_TRACE_ROOT": "/tmp/trace"})
        with self.assertRaisesRegex(ValueError, "uint32"):
            env.make_contract({"NM_FEM_NEWTON_TRACE_ROOT": str(1 << 32)})

    def test_diagnostic_capture_prefix_is_case_local_fresh_and_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / "case"
            case.mkdir()
            key = "NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH"
            prefixes = env.create_diagnostic_capture_prefixes(case, [key])
            prefix = Path(prefixes[key])
            contract = env.make_diagnostic_contract(
                {"NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR": "3:0:7:1:4:0"},
                case, prefixes)
            self.assertEqual(contract["effective"][key], {"set": True, "value": str(prefix)})
            with self.assertRaisesRegex(ValueError, "unsupported effective native environment contract"):
                env.validate_contract(contract)
            child, _ = env.child_environment({}, contract)
            self.assertEqual(child[key], str(prefix))
            (prefix.parent / "capture.consumer.bin").write_bytes(b"binary")
            (prefix.parent / "capture.consumer.json").write_bytes(b"{}\n")
            outputs = env.validate_diagnostic_outputs(case, prefixes)
            self.assertEqual(outputs["status"], "validated")
            self.assertEqual(outputs["file_count"], 2)

            external = Path(temporary) / "outside.json"
            external.write_text("{}\n")
            linked = prefix.parent / "capture.untrusted.json"
            linked.symlink_to(external)
            with self.assertRaisesRegex(ValueError, "not a new regular unlinked file"):
                env.validate_diagnostic_outputs(case, prefixes)
            linked.unlink()

            linked.hardlink_to(prefix.parent / "capture.consumer.bin")
            with self.assertRaisesRegex(ValueError, "not a new regular unlinked file"):
                env.validate_diagnostic_outputs(case, prefixes)
            linked.unlink()

            oversized = prefix.parent / "capture.oversized"
            with oversized.open("wb") as stream:
                stream.truncate(env.MAX_DIAGNOSTIC_CAPTURE_FILE_BYTES + 1)
            with self.assertRaisesRegex(ValueError, "byte cap exceeded"):
                env.validate_diagnostic_outputs(case, prefixes)
            oversized.unlink()
            with self.assertRaises(FileExistsError):
                env.create_diagnostic_capture_prefixes(case, [key])

    def test_diagnostic_paths_reject_traversal_symlinks_and_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "case"
            case.mkdir()
            with self.assertRaisesRegex(ValueError, "not a registered"):
                env.create_diagnostic_capture_prefixes(case, ["../outside"])
            outside = root / "outside"
            outside.mkdir()
            diagnostics = case / "diagnostics"
            diagnostics.symlink_to(outside, target_is_directory=True)
            with self.assertRaises((FileExistsError, ValueError)):
                env.create_diagnostic_capture_prefixes(case, ["NM_FEM_OPERATOR_FD_CAPTURE_PATH"])

            safe_case = root / "safe-case"
            safe_case.mkdir()
            key = "NM_FEM_OPERATOR_FD_CAPTURE_PATH"
            prefixes = env.create_diagnostic_capture_prefixes(safe_case, [key])
            contract = env.make_diagnostic_contract({}, safe_case, prefixes)
            real_diagnostics = safe_case / "diagnostics-real"
            (safe_case / "diagnostics").rename(real_diagnostics)
            (safe_case / "diagnostics").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "output root is missing, aliased, or symlinked"):
                env.validate_diagnostic_contract(contract)

    def test_diagnostic_records_are_not_qualification_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            case = Path(temporary) / "case"
            case.mkdir()
            prefixes = env.create_diagnostic_capture_prefixes(
                case, ["NM_FEM_OPERATOR_FD_CAPTURE_PATH"])
            contract = env.make_diagnostic_contract(
                {"NM_FEM_OPERATOR_FD_SELECTOR": "1:0"}, case, prefixes)
            record = env.make_diagnostic_run_record(
                contract, removed_inherited_keys=[], prediction_sha256="a" * 64)
            self.assertFalse(env.is_qualification_run_record(record))
            with self.assertRaisesRegex(ValueError, "unsupported effective native environment contract"):
                env.make_run_record(contract, producer="source-bound-native-runner",
                                    removed_inherited_keys=[], prediction_sha256="a" * 64)

    def test_source_bound_runner_materializes_and_validates_case_local_capture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            for relative in ("matter/tools", "matter/materials", "matter/src",
                             "matter/include", "matter/examples/cardboard/source-data"):
                (source / relative).mkdir(parents=True, exist_ok=True)
            (source / "matter/CMakeLists.txt").write_text("# diagnostic runner fixture\n")
            runtime = root / "runtime"
            runtime.mkdir()
            case = root / "case"
            native = root / "fake-native.sh"
            native.write_text(
                "#!/bin/sh\n"
                'printf "diagnostic-ok\\n" > "${NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH}.json"\n')
            native.chmod(0o755)
            key = "NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH"
            runner = Path(__file__).with_name("cardboard_source_bound_run.py")
            worker_dir = runner.resolve().parents[2] / "build-cardboard-native-worker"
            worker_existed = worker_dir.exists()
            command = [
                sys.executable, str(runner), "--case", str(case),
                "--question", "diagnostic capture path integration fixture",
                "--prediction", "one bounded case-local output file",
                "--timeout", "10", "--source-snapshot", str(source),
                "--runtime-dir", str(runtime), "--diagnostic-run",
                "--diagnostic-capture", key,
                "--env", "NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR=0:0:7:1:4:0",
                "--", str(native),
            ]
            try:
                completed = subprocess.run(command, capture_output=True, text=True,
                                           check=False, timeout=30)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                receipt = json.loads((case / "receipt.json").read_text())
                self.assertEqual(receipt["exit_code"], 0)
                self.assertEqual(receipt["run_mode"], "diagnostic")
                capture = case / "diagnostics/capture-regional-consumer-capture/capture.json"
                self.assertEqual(capture.read_text(), "diagnostic-ok\n")
                output_record = json.loads((case / "diagnostic-captures.json").read_text())
                self.assertEqual(output_record["status"], "validated")
                self.assertEqual(output_record["file_count"], 1)
                record, _ = env.read_run_record([case / "native-environment.json"])
                self.assertFalse(env.is_qualification_run_record(record))
                prediction_raw = (case / "prediction.json").read_bytes()
                env.validate_prediction_binding(record, prediction_raw)
            finally:
                if not worker_existed and worker_dir.exists():
                    lock = worker_dir / "worker.lock"
                    if lock.is_file():
                        lock.unlink()
                    try:
                        worker_dir.rmdir()
                    except OSError:
                        pass

    def test_rejects_tampered_or_ambiguous_contract_and_run_record(self):
        contract = env.make_contract()
        corrupted = dict(contract)
        corrupted["effective"] = dict(contract["effective"])
        corrupted["effective"]["NM_FEM_GRAPH_PARALLEL_LU_SEARCH"] = {"set": True, "value": "1"}
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            env.validate_contract(corrupted)
        record = env.make_run_record(contract, producer="source-bound-native-runner", removed_inherited_keys=[],
                                     prediction_sha256="a" * 64)
        record["launch"] = {"native_controls_scrubbed": False, "removed_inherited_key_names": []}
        with self.assertRaisesRegex(ValueError, "does not prove"):
            env.validate_run_record(record)

    def test_environment_sidecar_creation_is_exclusive(self):
        record = env.make_run_record(env.make_contract(), producer="source-bound-native-runner",
                                     removed_inherited_keys=[], prediction_sha256="a" * 64)
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "native-environment.json"
            env.write_run_record_new(sidecar, record)
            first = sidecar.read_bytes()
            with self.assertRaises(FileExistsError):
                env.write_run_record_new(sidecar, record)
            self.assertEqual(sidecar.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
