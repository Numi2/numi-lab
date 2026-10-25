import importlib.util
import pathlib
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SOURCE = HERE / "human_skin_tetra_refinement.py"
SPEC = importlib.util.spec_from_file_location("refine_f64_test_owner", SOURCE)
OWNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = OWNER
SPEC.loader.exec_module(OWNER)


class CanonicalFloat64ReferenceTests(unittest.TestCase):
    def test_f64_midpoint_closure_and_prefix_preservation(self):
        ref = np.asarray([[0.1, 0.2, 0.3], [1.1, 0.2, 0.3],
                          [0.2, 1.2, 0.3], [0.3, 0.4, 1.4]], dtype=np.float64)
        current = ref.astype(np.float32)
        tets = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        result = OWNER.refine_tetra_edges_f64_reference(
            ref, current, tets, [(0, 1)],
            source_vertex_indptr=np.arange(5, dtype=np.int64),
            source_vertex_indices=np.asarray([10, 11, 12, 13], dtype=np.int64),
            source_vertex_weights=np.ones(4, dtype=np.float64),
            material_ids=np.asarray([2], dtype=np.int32),
            parent_cell_ids=np.asarray([91], dtype=np.int64),
        )
        self.assertTrue(np.array_equal(result["canonical_reference_points_f64"][:4], ref))
        self.assertTrue(np.array_equal(result["current_points_f32"][:4], current))
        self.assertTrue(np.array_equal(result["reference_points_f32"][:4], ref.astype(np.float32)))
        np.testing.assert_array_equal(result["canonical_reference_points_f64"][4], (ref[0] + ref[1]) * 0.5)
        np.testing.assert_array_equal(result["current_points_f32"][4], ((current[0].astype(np.float64) + current[1].astype(np.float64)) * 0.5).astype(np.float32))
        self.assertLessEqual(result["report"]["maximum_canonical_f64_parent_reference_volume_relative_drift"], 1e-14)
        self.assertLessEqual(result["report"]["maximum_raw_canonical_f64_parent_fraction_sum_error"], 1e-14)
        self.assertEqual(result["material_ids"].tolist(), [2, 2])
        self.assertEqual(result["child_parent_cell_ids"].tolist(), [91, 91])
        self.assertEqual(result["source_vertex_ancestry_indptr"].tolist()[:5], [0, 1, 2, 3, 4])
        self.assertTrue(result["report"]["legacy_f32_reference_volume_drift_is_diagnostic_only"])
        self.assertFalse(result["report"]["physical_mass_transfer_admission"])

    def test_emitted_f32_quality_failure_is_not_hidden_by_f64_closure(self):
        h = 0.00005
        ref = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                          [0.5, 0.5, h]], dtype=np.float64)
        current = ref.copy()
        current[3, 2] = 0.0003
        current = current.astype(np.float32)
        tets = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        _, input_q = OWNER._volumes_and_quality(ref, tets)
        self.assertGreater(float(input_q.min()), 1e-5)
        with self.assertRaisesRegex(ValueError, "emitted F32 reference quality failed"):
            OWNER.refine_tetra_edges_f64_reference(ref, current, tets, [(0, 1)])

    def test_bad_reference_dtype_fails_closed(self):
        ref = np.asarray([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Float64"):
            OWNER.refine_tetra_edges_f64_reference(ref, ref.copy(), np.asarray([[0, 1, 2, 3]]), [(0, 1)])


if __name__ == "__main__":
    unittest.main()
