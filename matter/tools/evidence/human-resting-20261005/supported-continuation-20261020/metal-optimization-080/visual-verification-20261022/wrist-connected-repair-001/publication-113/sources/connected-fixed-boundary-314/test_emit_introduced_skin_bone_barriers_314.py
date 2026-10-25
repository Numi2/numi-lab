from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "emit_introduced_skin_bone_barriers_314.py"
spec = importlib.util.spec_from_file_location("barriers314", SOURCE)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class Barrier314Tests(unittest.TestCase):
    def setUp(self):
        self.keys = ((10, 2), (20, 7))
        self.script = "a" * 64
        self.predicate = "b" * 64
        self.candidate = "c" * 64

    def report(self, pairs_by_key, candidate=None, complete=True):
        banks = []
        for body, bone in self.keys:
            pairs = [list(p) for p in pairs_by_key.get((body, bone), [])]
            banks.append({
                "body": body,
                "stable_bone_id": bone,
                "skin_faces": 3,
                "bank_faces": 4,
                "skin_aabb_active": 1 if pairs else 0,
                "bank_aabb_active": 1 if pairs else 0,
                "result": ({"count": len(pairs), "triangle_pairs": pairs} if pairs else None),
            })
        return {
            "status": "COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT" if complete else "BOUNDED_PARTIAL_AUDIT",
            "complete": complete,
            "candidate_sha256": candidate or self.candidate,
            "script_sha256": self.script,
            "exact_predicate_sha256": self.predicate,
            "surface_point_count": 5,
            "surface_face_count": 3,
            "bank_count": 2,
            "expected_bank_count": 2,
            "banks": banks,
        }

    def validate(self, report):
        return mod.validate_audit_report(
            report,
            candidate_sha256=self.candidate,
            point_count=5,
            face_count=3,
            expected_bank_keys=self.keys,
            expected_script_sha256=self.script,
            expected_predicate_sha256=self.predicate,
        )

    def test_union_minus_base_is_typed_deterministic(self):
        base = self.validate(self.report({(10, 2): [(0, 99)]}))
        trial_a = self.validate(self.report({(10, 2): [(0, 99), (1, 101)]}))
        trial_b = self.validate(self.report({(20, 7): [(2, 5)]}))
        self.assertEqual(
            mod.introduced_pairs(base, (trial_a, trial_b)),
            ((10, 2, 1, 101), (20, 7, 2, 5)),
        )

    def test_nonorthonormal_affine_roundtrip_and_mm(self):
        local = np.array([[.01, -.02, .03], [.04, .01, -.01], [0., .02, .05]])
        matrix = np.array([[2.0, .3, -.1], [.2, 1.4, .25], [0.1, -.2, .8]])
        offset = np.array([.003, -.004, .002])
        position = np.array([.2, -.1, .6])
        world_mm = mod.world_triangle_mm(local, matrix, offset, position)
        recovered = (world_mm / 1000.0 - position) @ matrix.T - offset
        np.testing.assert_allclose(recovered, local, rtol=0, atol=5e-17)

    def test_stable_triangle_id_not_row_index(self):
        local = np.array([
            [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
            [[5., 0., 0.], [6., 0., 0.], [5., 1., 0.]],
        ])
        ids = np.array([[10, 2, 0, 9001], [10, 2, 0, 77]], dtype=np.int64)
        lookup = mod.typed_triangle_lookup(local, ids)
        arrays = mod.make_barrier_arrays(
            {(10, 2, 1, 77)},
            lookup,
            {10: (np.eye(3), np.zeros(3), np.array([.1, .2, .3]))},
        )
        self.assertEqual(arrays["bank_triangle_stable_id"].tolist(), [77])
        np.testing.assert_allclose(arrays["fixed_triangles_world_mm"][0, 0], [5100., 200., 300.])

    def test_empty_barrier_arrays_keep_shapes(self):
        arrays = mod.make_barrier_arrays((), {}, {})
        self.assertEqual(arrays["skin_face_rows"].shape, (0,))
        self.assertEqual(arrays["fixed_triangles_world_mm"].shape, (0, 3, 3))

    def test_incomplete_or_wrong_hash_rejected(self):
        with self.assertRaisesRegex(ValueError, "not complete"):
            self.validate(self.report({}, complete=False))
        with self.assertRaisesRegex(ValueError, "candidate SHA"):
            self.validate(self.report({}, candidate="d" * 64))

    def test_missing_bank_rejected(self):
        report = self.report({})
        report["banks"] = report["banks"][:1]
        report["bank_count"] = 1
        with self.assertRaisesRegex(ValueError, "bank count"):
            self.validate(report)

    def test_wrong_predicate_owner_rejected(self):
        report = self.report({})
        report["exact_predicate_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "predicate SHA"):
            self.validate(report)

    def test_missing_stable_triangle_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing typed stable triangle"):
            mod.make_barrier_arrays(
                {(10, 2, 0, 1234)},
                {(10, 2, 99): np.zeros((3, 3))},
                {10: (np.eye(3), np.zeros(3), np.zeros(3))},
            )

    def test_face_array_identity_gate(self):
        base = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.int32)
        reordered = base[[1, 0, 2]]
        self.assertFalse(mod.same_face_array(base, reordered))
        self.assertFalse(mod.same_face_array(base, base.astype(np.int64)))
        self.assertTrue(mod.same_face_array(base, base.copy()))


if __name__ == "__main__":
    unittest.main()
