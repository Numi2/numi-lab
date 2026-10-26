import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import continue_surface_round546 as owner

def put_json(path, value):
    Path(path).write_text(json.dumps(value))

class Tests(unittest.TestCase):
    def test_empty_point_archive_shapes(self):
        arrays = owner.empty_point_archive()
        self.assertEqual(arrays["query_rows"].shape, (0,))
        self.assertEqual(arrays["target_world_f32_m"].shape, (0, 3))
        self.assertEqual(arrays["target_world_f32_m"].dtype, np.float32)
        self.assertEqual(arrays["proposal_direction_world_unit"].dtype, np.float64)
        self.assertEqual(arrays["target_exactly_outside_all185"].shape, (0,))

    def test_empty_point_target_report_is_candidate_bound_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            candidate = root / "candidate.npz"
            candidate.write_bytes(b"candidate")
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            point_summary = root / "point-summary.json"
            put_json(point_summary, {"candidate_sha256": digest,
                "complete_surface_audit": True, "exact_PLC_surface_pairs": 0,
                "inside_node_bank_pairs": 0})
            depth = root / "depth.json"
            put_json(depth, {"candidate_sha256": digest,
                "inside_row_count": 0, "outside_row_count": 3})
            npz, report_path = owner.write_empty_point_targets(
                root / "point-targets", candidate, point_summary, depth)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["inputs"]["candidate_sha256"], digest)
            self.assertEqual(report["status"], "PASS_EMPTY_POINT_TARGETS_ZERO_CONTAINMENT")
            with np.load(npz, allow_pickle=False) as z:
                self.assertEqual(z["query_rows"].shape, (0,))
                self.assertEqual(z["target_world_f32_m"].shape, (0, 3))
            bad_summary = root / "bad-summary.json"
            put_json(bad_summary, {"candidate_sha256": digest,
                "complete_surface_audit": True, "exact_PLC_surface_pairs": 0,
                "inside_node_bank_pairs": 1})
            with self.assertRaisesRegex(ValueError, "zero-contained"):
                owner.write_empty_point_targets(
                    root / "bad-point-targets", candidate, bad_summary, depth)

    def test_matched_inputs_updates_only_input_records(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            template = root / "template.json"
            source_pins = {"owner": {"path": "/owner.py", "sha256": "a" * 64}}
            roles = ("candidate", "audit159", "face_target_npz", "point_target_npz")
            put_json(template, {"schema": "numi.human.matched_inputs_component_fit_475.v1",
                "inputs": {role: {"path": "", "sha256": ""} for role in roles},
                "source_pins": source_pins})
            paths = {}
            for role in roles:
                p = root / (role + ".dat")
                p.write_text(role)
                paths[role] = p
            result_path = root / "matched.json"
            result = owner.matched_inputs_from_template(template, result_path, paths)
            self.assertEqual(result["source_pins"], source_pins)
            self.assertEqual(set(result["inputs"]), set(paths))
            for role, path in paths.items():
                self.assertEqual(result["inputs"][role]["sha256"],
                    hashlib.sha256(path.read_bytes()).hexdigest())

    def test_cli_replacement_rejects_missing_and_duplicate_flags(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            owner.replace_cli_value(["--candidate", "x"], "--audit159", "y")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            owner.replace_cli_value(["--candidate", "x", "--candidate", "z"], "--candidate", "y")

if __name__ == "__main__":
    unittest.main(verbosity=2)
