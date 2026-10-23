#!/usr/bin/env python3
"""Runner integrity checks. None of these tests qualifies cardboard mechanics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace

import cardboard_opencourant as runner


class RunnerIntegrity(unittest.TestCase):
    def test_native_converged_flag_does_not_hide_force_fallback(self):
        deck = "#RADIOSS ENGINE\n/IMPL/NONL/01\n20 2 1e-6 1e-6\n"
        listing = ("  23 Y 1.949E-06 9.564E-07 7.353E-14 C\n"
                   "  10 Y 6.842D-07 1.311D-05 1.000D-13 C\n"
                   "  50 Y 1.833E-06 2.884E-06 6.765E-13\n")
        audit = runner.implicit_force_audit(deck, listing)
        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["accepted_rows"], 2)
        self.assertEqual(audit["exceedance_count"], 1)
        self.assertAlmostEqual(audit["maximum_accepted_relative_force_residual"], 1.311e-5)
        self.assertEqual(runner.implicit_force_audit(deck, "")["status"], "unresolved")
        self.assertEqual(runner.implicit_force_audit("/IMPL/NONL/01\n20 1 1e-6", listing)["status"], "unresolved")

    def test_timeout_kills_child_when_wrapper_exits_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Reproduce the observed time-wrapper/MPI failure: the wrapper
            # obeys TERM, but its child ignores TERM and keeps writing.
            child = (
                "import os,signal,time,pathlib; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "pathlib.Path('child.pid').write_text(str(os.getpid())); "
                "f=open('heartbeat','a',buffering=1); "
                "exec('while True:\\n f.write(\"alive\\\\n\"); time.sleep(.01)')"
            )
            wrapper = (
                "import subprocess,sys,time,pathlib; "
                f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
                "exec('while not pathlib.Path(\"child.pid\").exists(): time.sleep(.01)'); "
                "p.wait()"
            )
            result = runner.execute([sys.executable, "-c", wrapper], root,
                                    dict(os.environ), 0.5, "timeout")
            self.assertTrue(result["timed_out"])
            self.assertTrue((root / "child.pid").is_file())
            size = (root / "heartbeat").stat().st_size
            time.sleep(0.1)
            self.assertEqual((root / "heartbeat").stat().st_size, size)

    def test_type43_profile_binds_patch_receipt_and_engine_identity(self):
        import copy
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / 'runtime/OpenCourant'
            setup = root / 'setup'
            (runtime/'exec').mkdir(parents=True)
            setup.mkdir()
            for name in ('starter_linuxa64_gf', 'engine_type43'):
                (runtime/'exec'/name).write_bytes(b'non-executable test identity')
            patch = Path(__file__).resolve().parents[1] / 'examples/cardboard/opencourant/adhesive-fix-20261023/type43-implicit.patch'
            (setup/'type43-implicit.patch').write_bytes(patch.read_bytes())
            receipt = {'source_commit': runner.SOURCE,
                       'patch_sha256': runner.TYPE43_PATCH_SHA256,
                       'engine_sha256': runner.sha(runtime/'exec/engine_type43')}
            runner.write(setup/'build-receipt.json', receipt)
            identity = {'source_commit': runner.SOURCE, 'package_sha256': runner.PACKAGE_SHA256,
                        'profile': 'arm64-mumps-type43',
                        'files': {'exec/engine_type43': receipt['engine_sha256']},
                        'build': {'source_commit': runner.SOURCE,
                                  'patch_sha256': runner.TYPE43_PATCH_SHA256,
                                  'receipt_sha256': runner.sha(setup/'build-receipt.json')}}
            self.assertEqual(runner.runtime_binaries(runtime, identity, identity['profile'])['engine'].name,
                             'engine_type43')
            wrong = copy.deepcopy(identity)
            wrong['files']['exec/engine_type43'] = 'another engine'
            with self.assertRaises(ValueError):
                runner.runtime_binaries(runtime, wrong, identity['profile'])
            with self.assertRaises(ValueError):
                runner.runtime_binaries(runtime, identity, 'arm64-mumps-build')
            (setup/'type43-implicit.patch').write_bytes(patch.read_bytes()+b'changed patch')
            with self.assertRaises(ValueError):
                runner.runtime_binaries(runtime, identity, identity['profile'])

    def test_paper_predictor_profile_binds_combined_patch_receipt_and_engine(self):
        import copy
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / 'runtime/OpenCourant'
            setup = root / 'setup'
            (runtime/'exec').mkdir(parents=True)
            setup.mkdir()
            (runtime/'exec'/'starter_linuxa64_gf').write_bytes(b'starter identity')
            engine = runtime/'exec'/'engine_cardboard'
            engine.write_bytes(b'paper predictor candidate engine')

            evidence = (Path(__file__).resolve().parents[1] /
                        'examples/cardboard/opencourant/paper-predictor-20261023')
            patch = evidence/'paper-implicit.patch'
            spec = json.loads((evidence/'patch-identity.json').read_text())
            self.assertEqual(runner.sha(patch), runner.PAPER_PATCH_SHA256)
            (setup/'paper-implicit.patch').write_bytes(patch.read_bytes())
            runner.write(setup/'patch-identity.json', spec)

            receipt = {'schema': 'cardboard.opencourant.paper.build.v1',
                       'runtime_profile': 'arm64-mumps-paper',
                       'source_commit': runner.SOURCE,
                       'patch_sha256': runner.PAPER_PATCH_SHA256,
                       'parent_adhesive_patch_sha256': runner.TYPE43_PATCH_SHA256,
                       'changed_sources': spec['files'],
                       'engine_filename': 'engine_cardboard',
                       'engine_sha256': runner.sha(engine)}
            runner.write(setup/'build-receipt.json', receipt)
            identity = {'source_commit': runner.SOURCE,
                        'package_sha256': runner.PACKAGE_SHA256,
                        'profile': 'arm64-mumps-paper',
                        'files': {'exec/engine_cardboard': runner.sha(engine)},
                        'build': {'source_commit': runner.SOURCE,
                                  'patch_sha256': runner.PAPER_PATCH_SHA256,
                                  'parent_adhesive_patch_sha256': runner.TYPE43_PATCH_SHA256,
                                  'receipt_sha256': runner.sha(setup/'build-receipt.json')}}
            binaries = runner.runtime_binaries(runtime, identity, identity['profile'])
            self.assertEqual(binaries['engine'].name, 'engine_cardboard')

            wrong = copy.deepcopy(identity)
            wrong['build']['parent_adhesive_patch_sha256'] = 'wrong parent patch'
            with self.assertRaises(ValueError):
                runner.runtime_binaries(runtime, wrong, identity['profile'])
            wrong = copy.deepcopy(receipt)
            wrong['engine_sha256'] = 'wrong engine'
            runner.write(setup/'build-receipt.json', wrong)
            with self.assertRaises(ValueError):
                runner.runtime_binaries(runtime, identity, identity['profile'])
            (setup/'paper-implicit.patch').write_bytes(patch.read_bytes()+b'tampered')
            with self.assertRaises(ValueError):
                runner.runtime_binaries(runtime, identity, identity['profile'])

    def test_paper_profile_and_starter_only_options_are_available(self):
        result = subprocess.run([sys.executable, str(Path(runner.__file__).resolve()),
                                 'run', '--help'], capture_output=True, text=True, check=True)
        self.assertIn('arm64-mumps-paper', result.stdout)
        self.assertIn('--starter-only', result.stdout)
        builder = (Path(__file__).resolve().parent /
                   'build_cardboard_opencourant_type43.py')
        result = subprocess.run([sys.executable, str(builder), '--help'],
                                capture_output=True, text=True, check=True)
        self.assertIn('arm64-mumps-paper', result.stdout)

    def test_mumps_profile_requires_unchanged_source_and_bound_build_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime" / "OpenCourant"
            (runtime / "exec").mkdir(parents=True)
            (root / "setup").mkdir()
            for name in ("starter_linuxa64_gf", "engine_linuxa64_gf_ompi"):
                (runtime / "exec" / name).write_text("test binary identity")
            receipt = root / "setup" / "build-receipt.json"
            receipt.write_text("bound receipt")
            identity = {"source_commit": runner.SOURCE, "package_sha256": runner.PACKAGE_SHA256,
                        "profile": "arm64-mumps-build", "build": {
                            "source_commit": runner.SOURCE, "tracked_source_diff": "",
                            "receipt_sha256": runner.sha(receipt)}}
            binaries = runner.runtime_binaries(runtime, identity, "arm64-mumps-build")
            self.assertEqual(binaries["engine"].name, "engine_linuxa64_gf_ompi")
            with self.assertRaises(ValueError):
                runner.runtime_binaries(runtime, identity, "arm64-package")
            identity["build"]["tracked_source_diff"] = "changed mechanics"
            with self.assertRaises(ValueError):
                runner.runtime_binaries(runtime, identity, "arm64-mumps-build")
            identity["build"]["tracked_source_diff"] = ""
            receipt.write_text("modified receipt")
            with self.assertRaises(ValueError):
                runner.runtime_binaries(runtime, identity, "arm64-mumps-build")

    def test_resume_requires_immediately_preceding_checkpoint(self):
        engines = ["box_0001.rad", "box_0002.rad", "box_0003.rad"]
        parent = {"completed_engine": engines[0]}
        runner.continuation(engines, parent, engines[1])
        for requested in (engines[0], engines[2]):
            with self.assertRaises(ValueError):
                runner.continuation(engines, parent, requested)

    def test_native_error_cannot_pass_from_normal_text_elsewhere(self):
        state = runner.native_termination("NORMAL TERMINATION\nERROR ID : 574\nERROR TERMINATION")
        self.assertTrue(state["error_termination"])
        self.assertFalse(runner.native_termination("Process exited 0")["normal_termination"])

    def test_input_cannot_escape_case_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "case").mkdir()
            (root / "outside").write_text("not a case input")
            with self.assertRaises(ValueError):
                runner.local_file(root / "case", "../outside")

    def test_inventory_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "real").write_text("a")
            (root / "link").symlink_to("real")
            with self.assertRaises(ValueError):
                runner.inventory(root)

    def test_timeout_retains_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = runner.execute([sys.executable, "-c", "import time; time.sleep(10)"],
                                    Path(tmp), dict(os.environ), .05, "timeout")
            self.assertTrue(result["timed_out"])
            self.assertNotEqual(result["exit_code"], 0)
            self.assertTrue((Path(tmp) / "timeout.process.json").is_file())

    def test_collect_detects_changed_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "solver.stdout"
            output.write_text("original observation")
            runner.write(root / "run.json", {"execution_complete": True,
                         "outputs": {output.name: {"sha256": runner.sha(output)}}})
            output.write_text("changed observation")
            code = runner.collect(SimpleNamespace(receipt=root / "run.json"))
            self.assertEqual(code, 2)
            report = json.loads((root / "collection.json").read_text())
            self.assertFalse(report["physical_qualification"])
            self.assertFalse(report["output_identity_valid"])


if __name__ == "__main__":
    unittest.main()
