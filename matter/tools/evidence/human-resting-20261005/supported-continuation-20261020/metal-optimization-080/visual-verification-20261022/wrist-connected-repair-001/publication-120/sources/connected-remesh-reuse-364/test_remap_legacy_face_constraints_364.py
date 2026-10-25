import unittest
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remap_legacy_face_constraints_364 import filter_constraint_rows


class ConstraintRowFilterTests(unittest.TestCase):
    def setUp(self):
        self.base = np.array([
            [0, 1, 2],
            [1, 3, 2],
            [3, 4, 2],
            [4, 5, 2],
        ], dtype=np.int64)
        self.remeshed = self.base.copy()
        self.remeshed[1] = [1, 4, 3]
        self.pairs = np.array([[0, 2], [1, 3], [2, 3]], dtype=np.int64)
        self.bone = np.array([1, 3], dtype=np.int64)

    def test_only_rows_with_identical_face_connectivity_survive(self):
        out = filter_constraint_rows(self.base, self.base, self.remeshed, [1], self.pairs, self.bone)
        np.testing.assert_array_equal(out["changed_face_rows"], [1])
        np.testing.assert_array_equal(out["retained_self_pairs"], [[0, 2], [2, 3]])
        np.testing.assert_array_equal(out["stale_self_pairs"], [[1, 3]])
        np.testing.assert_array_equal(out["retained_bone_mask"], [False, True])
        np.testing.assert_array_equal(out["stale_bone_mask"], [True, False])

    def test_provenance_change_also_invalidates_constraint_row(self):
        out = filter_constraint_rows(
            self.base, self.base, self.base, [], self.pairs, self.bone,
            metadata_changed_rows={3},
        )
        np.testing.assert_array_equal(out["metadata_changed_face_rows"], [3])
        np.testing.assert_array_equal(out["retained_self_pairs"], [[0, 2]])
        np.testing.assert_array_equal(out["stale_self_pairs"], [[1, 3], [2, 3]])
        np.testing.assert_array_equal(out["retained_bone_mask"], [True, False])

    def test_legacy_base_must_match_remesh_input_rows(self):
        wrong = self.base.copy()
        wrong[0] = [0, 2, 1]
        with self.assertRaisesRegex(ValueError, "legacy constraints do not match"):
            filter_constraint_rows(wrong, self.base, self.remeshed, [1], self.pairs, self.bone)

    def test_changed_rows_must_match_exact_diff_and_report(self):
        with self.assertRaisesRegex(ValueError, "flip report changed rows"):
            filter_constraint_rows(self.base, self.base, self.remeshed, [], self.pairs, self.bone)

    def test_face_row_domain_must_not_change(self):
        with self.assertRaisesRegex(ValueError, "face row domains differ"):
            filter_constraint_rows(self.base, self.base, self.remeshed[:-1], [1], self.pairs, self.bone)

    def test_invalid_constraint_face_id_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "self-pair face row out of range"):
            filter_constraint_rows(
                self.base, self.base, self.remeshed, [1],
                np.array([[0, 99]], dtype=np.int64), self.bone,
            )


if __name__ == "__main__":
    unittest.main()

