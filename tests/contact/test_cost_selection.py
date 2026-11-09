#!/usr/bin/env python3
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from cost_selection import select_largest_certified


class SelectionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.arm = "ipc_area"

    def add(self, name, h, passed, compressed=False):
        prefix = self.root / ("ladder-" + name)
        steps = round(.01 / h)
        record = {"command": ["probe", "shader", str(h), str(steps), "2", self.arm,
                              "0.0002", "physical", "contact", "reference"],
                  "all_gates_pass": passed}
        Path(str(prefix) + "-execution.json").write_text(json.dumps(record))
        state = {"arm": self.arm, "h": h, "accepted_steps": steps, "rejected_steps": 0, "traces": []}
        path = Path(str(prefix) + (".json.gz" if compressed else ".json"))
        if compressed:
            with gzip.open(path, "wt") as f:
                json.dump(state, f)
        else:
            path.write_text(json.dumps(state))
        Path(str(prefix) + "-analysis.json").write_text(json.dumps({"passes": passed}))
        return prefix

    def select(self):
        return select_largest_certified(self.root, "ladder-*", self.arm, lambda e: None)

    def test_selects_largest_pass_even_if_nonmonotonic(self):
        self.add("fine", .000001, True, compressed=True)
        self.add("middle", .000002, False)
        self.add("coarse", .000004, True)
        selected, seen = self.select()
        self.assertEqual(selected, "ladder-coarse")
        self.assertEqual([x[2] for x in seen], [True, False, True])

    def test_incomplete_attempt_is_not_a_failed_accuracy_test(self):
        self.add("fine", .000001, True)
        (self.root / "ladder-interrupted.json.gz").write_bytes(b"partial")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.select()

    def test_stopped_attempt_is_inconclusive(self):
        prefix = self.add("fine", .000001, True)
        execution = Path(str(prefix) + "-execution.json")
        record = json.loads(execution.read_text())
        record["stopped_reason"] = "free-space limit"
        execution.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.select()

    def test_mismatched_fixture_and_false_certificate_rejected(self):
        prefix = self.add("fine", .000001, True)
        analysis = Path(str(prefix) + "-analysis.json")
        analysis.write_text(json.dumps({"passes": False}))
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.select()
        analysis.write_text(json.dumps({"passes": True}))
        execution = Path(str(prefix) + "-execution.json")
        record = json.loads(execution.read_text())
        record["command"][6] = "0.0003"
        execution.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "fixture"):
            self.select()


if __name__ == "__main__":
    unittest.main()
