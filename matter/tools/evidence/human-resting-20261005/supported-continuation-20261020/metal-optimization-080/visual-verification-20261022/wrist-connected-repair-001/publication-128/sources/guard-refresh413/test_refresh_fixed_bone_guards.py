import importlib.util
import unittest
from pathlib import Path

P = Path(__file__).with_name("refresh_fixed_bone_guards.py")
spec = importlib.util.spec_from_file_location("refresh_fixed_bone_guards_tested", P)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GuardIdentityTests(unittest.TestCase):
    def test_resolved_pairs_are_added_and_current_pairs_filtered(self):
        prev = {(1, 10, 100, 5), (1, 10, 101, 6), (2, 20, 200, 7)}
        curr = {(1, 10, 100, 5), (3, 30, 300, 9)}
        prior = {(4, 40, 400, 2), (1, 10, 100, 5)}
        got = module.compute_guard_identity_sets(prev, curr, prior)
        self.assertEqual(got["introduced"], {(3, 30, 300, 9)})
        self.assertEqual(got["resolved"], {(1, 10, 101, 6), (2, 20, 200, 7)})
        self.assertEqual(got["removed_current"], {(1, 10, 100, 5)})
        self.assertEqual(got["retained"], {(4, 40, 400, 2), (1, 10, 101, 6), (2, 20, 200, 7)})
        self.assertTrue(got["resolved"].issubset(got["retained"]))
        self.assertTrue(got["current"].isdisjoint(got["retained"]))

    def test_no_change_preserves_prior_guards_without_inventing_delta(self):
        cur = {(1, 10, 2, 3)}
        prior = {(1, 11, 4, 5)}
        got = module.compute_guard_identity_sets(cur, cur, prior)
        self.assertFalse(got["introduced"])
        self.assertFalse(got["resolved"])
        self.assertEqual(got["retained"], prior)

    def test_current_filter_removes_prior_and_resolved_identity(self):
        previous = {(1, 1, 1, 1), (1, 1, 2, 2)}
        current = {(1, 1, 2, 2)}
        prior = {(1, 1, 2, 2), (2, 2, 3, 3)}
        got = module.compute_guard_identity_sets(previous, current, prior)
        self.assertEqual(got["removed_current"], {(1, 1, 2, 2)})
        self.assertEqual(got["retained"], {(1, 1, 1, 1), (2, 2, 3, 3)})

    def test_report_binding_rejects_wrong_candidate_hash(self):
        report = {"complete": True, "bank_count": 185, "expected_bank_count": 185,
                  "candidate_sha256": "wrong", "surface_point_count": 2,
                  "surface_face_count": 1,
                  "banks": [{"body": i, "stable_bone_id": i+100} for i in range(185)]}
        with self.assertRaisesRegex(ValueError, "does not bind"):
            module.validate_report_binding(report, "expected", [0, 1], [[0, 1, 0]])

    def test_report_binding_rejects_missing_or_duplicate_bank_inventory(self):
        points = [[0, 0, 0]] * 2
        faces = [[0, 1, 0]]
        bankrows = [{"body": i, "stable_bone_id": i + 100} for i in range(185)]
        for bad in (bankrows[:-1], bankrows[:-1] + [bankrows[0]]):
            report = {"complete": True, "bank_count": 185, "expected_bank_count": 185,
                      "candidate_sha256": "pin", "surface_point_count": 2,
                      "surface_face_count": 1, "banks": bad}
            with self.subTest(banks=len(bad)), self.assertRaisesRegex(ValueError, "unique typed banks"):
                module.validate_report_binding(report, "pin", points, faces)

    def test_face_topology_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "topology changed"):
            module.require_same_topology([[0, 1, 2]], [[0, 2, 1]], [[0, 2, 1]])

    def test_current_reference_face_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "037 reference"):
            module.require_same_topology([[0, 1, 2]], [[0, 1, 2]], [[0, 2, 1]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
