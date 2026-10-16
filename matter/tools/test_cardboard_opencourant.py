#!/usr/bin/env python3
"""Runner integrity checks. None of these tests qualifies cardboard mechanics."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import cardboard_opencourant as runner


class RunnerIntegrity(unittest.TestCase):
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
