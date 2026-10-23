import struct
import unittest

from pose_diagnostic_gate import classify_kinematic_pose, equality_residual_summary


class PoseDiagnosticGateTests(unittest.TestCase):
    def test_zero_bone_pairs_do_not_override_skin_crossings(self):
        result = classify_kinematic_pose(
            source_probe_accepted=False,
            full_surface_self_pairs=155,
            incremental_roi_self_pairs=0,
            roi_vs_rest_skin_pairs=0,
            roi_vs_bone_pairs=0,
            bilateral_bone_surface_pairs=0,
            skin_to_bone_pairs=None,
        )
        self.assertEqual(result["status"], "rejected_source_self_intersection")
        self.assertFalse(result["kinematic_geometry_clear"])
        self.assertFalse(result["physical_motion_qualified"])

    def test_rejected_source_probe_is_not_fabricated_self_intersection(self):
        result = classify_kinematic_pose(
            source_probe_accepted=False,
            full_surface_self_pairs=0,
            incremental_roi_self_pairs=0,
            roi_vs_rest_skin_pairs=0,
            roi_vs_bone_pairs=0,
            bilateral_bone_surface_pairs=0,
            skin_to_bone_pairs=0,
        )
        self.assertEqual(result["status"], "rejected_source_probe_non_self_intersection")
        self.assertFalse(result["source_surface_clear"])
        self.assertFalse(result["kinematic_geometry_clear"])
        self.assertFalse(result["physical_motion_qualified"])

    def test_rejected_probe_without_pair_measurement_stays_unclassified(self):
        result = classify_kinematic_pose(
            source_probe_accepted=False,
            full_surface_self_pairs=None,
            incremental_roi_self_pairs=None,
            roi_vs_rest_skin_pairs=None,
            roi_vs_bone_pairs=None,
            bilateral_bone_surface_pairs=None,
            skin_to_bone_pairs=None,
        )
        self.assertEqual(result["status"], "rejected_source_probe_unclassified")
        self.assertFalse(result["source_surface_clear"])
        self.assertFalse(result["kinematic_geometry_clear"])

    def test_missing_skin_bone_measurement_is_not_clearance(self):
        result = classify_kinematic_pose(
            source_probe_accepted=True,
            full_surface_self_pairs=0,
            incremental_roi_self_pairs=0,
            roi_vs_rest_skin_pairs=0,
            roi_vs_bone_pairs=0,
            bilateral_bone_surface_pairs=0,
            skin_to_bone_pairs=None,
        )
        self.assertEqual(result["status"], "incomplete_contact_geometry_measurements")
        self.assertFalse(result["kinematic_geometry_clear"])
        self.assertFalse(result["physical_motion_qualified"])

    def test_static_zero_pair_geometry_is_not_physical_motion_qualification(self):
        result = classify_kinematic_pose(
            source_probe_accepted=True,
            full_surface_self_pairs=0,
            incremental_roi_self_pairs=0,
            roi_vs_rest_skin_pairs=0,
            roi_vs_bone_pairs=0,
            bilateral_bone_surface_pairs=0,
            skin_to_bone_pairs=0,
        )
        self.assertEqual(result["status"], "kinematic_geometry_clear_only")
        self.assertTrue(result["kinematic_geometry_clear"])
        self.assertFalse(result["physical_motion_qualified"])

    def _nheq_payload(self, *, master_v=105, coefficient=0.010506):
        floats = [0.0, 0.0, coefficient, 0.0247615, -1.31647,
                  0.716337, -0.138302, 0.0] + [0.0] * 12
        row = struct.pack("<4I20f", 114, 113, 106, master_v, *floats)
        header = struct.pack("<8s10I32s", b"NHEQ1\0\0\0", 1, 129, 128,
                             1, 96, 1, 0, 0, 0, 0, bytes(32))
        return header + row

    def test_nheq_rejects_nonfinite_polynomial_coefficient(self):
        payload = self._nheq_payload(coefficient=float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite coefficients"):
            equality_residual_summary([0.0] * 129, payload)

    def test_nheq_rejects_nonfinite_unused_record_parameter(self):
        floats = [0.0] * 20
        floats[18] = float("inf")
        row = struct.pack("<4I20f", 114, 113, 106, 105, *floats)
        header = struct.pack("<8s10I32s", b"NHEQ1\0\0\0", 1, 129, 128,
                             1, 96, 1, 0, 0, 0, 0, bytes(32))
        with self.assertRaisesRegex(ValueError, "nonfinite coefficients"):
            equality_residual_summary([0.0] * 129, header + row)

    def test_nheq_rejects_out_of_range_master_velocity_index(self):
        payload = self._nheq_payload(master_v=128)
        with self.assertRaisesRegex(ValueError, "master index out of range"):
            equality_residual_summary([0.0] * 129, payload)

    def test_nheq_requires_consistent_fixed_master_sentinels(self):
        floats = [0.0] * 20
        row = struct.pack("<4I20f", 114, 113, 0xFFFFFFFF, 105, *floats)
        header = struct.pack("<8s10I32s", b"NHEQ1\0\0\0", 1, 129, 128,
                             1, 96, 1, 0, 0, 0, 0, bytes(32))
        with self.assertRaisesRegex(ValueError, "master index out of range"):
            equality_residual_summary([0.0] * 129, header + row)

    def test_equality_projection_checks_dependent_closure_value(self):
        # One coupled equality: q_dep = .010506 + .0247615*x
        #                         -1.31647*x^2 + .716337*x^3 - .138302*x^4.
        q = [0.0] * 129
        q[106] = 2.0
        q[114] = -1.6879868805408478
        floats = [0.0, 0.0, 0.010506, 0.0247615, -1.31647,
                  0.716337, -0.138302, 0.0] + [0.0] * 12
        row = struct.pack("<4I20f", 114, 113, 106, 105, *floats)
        header = struct.pack("<8s10I32s", b"NHEQ1\0\0\0", 1, 129, 128,
                             1, 96, 1, 0, 0, 0, 0, bytes(32))
        result = equality_residual_summary(q, header + row)
        self.assertEqual(result["constraint_count"], 1)
        self.assertLess(result["max_absolute_residual"], 1e-7)


if __name__ == "__main__":
    unittest.main()
