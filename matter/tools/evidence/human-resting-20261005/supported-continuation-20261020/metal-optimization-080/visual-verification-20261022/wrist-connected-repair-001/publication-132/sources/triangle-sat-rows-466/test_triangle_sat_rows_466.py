from pathlib import Path
import importlib.util
import sys
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("triangle_sat_rows_466", HERE / "triangle_sat_rows_466.py")
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)
OLD_PATH = HERE.parent / "fixed-bone-outward-plane-458" / "fixed_bone_outward_plane_458.py"
OLD_SPEC = importlib.util.spec_from_file_location("fixed_bone_outward_plane_458_pinned", OLD_PATH)
old = importlib.util.module_from_spec(OLD_SPEC)
OLD_SPEC.loader.exec_module(old)


class TriangleSatRows466Tests(unittest.TestCase):
    def setUp(self):
        # Verified outward bone face is +x; its thin y extent makes a lateral
        # separating direction much cheaper than the bone-face-only row.
        self.bone = np.array([[0., 0., 0.], [0., .1, 0.], [0., 0., 10.]])
        self.skin = np.array([[-1., .05, 1.], [1., .05, 1.], [0., .05, 2.]])
        self.identity = (0, 43, 16, 133)
        self.pair = dict(face_row=0, body=43, bone_id=16, triangle_id=133)

    def test_default_plane_and_builder_preserve_458_rows(self):
        bone = np.array([[0., -2., -2.], [0., 2., -2.], [0., 0., 2.]])
        skin = np.array([[.3, -1., -1.], [-.4, 1., -1.], [-.1, 0., 1.]])
        a = old.outward_plane_for_pair(skin, bone)
        b = m.outward_plane_for_pair(skin, bone)
        self.assertEqual(set(a), set(b))
        for key in ("normal_world_unit", "plane_point_mm", "corner_signed_distance_mm", "corner_lower_mm"):
            np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(a["float32_ulp_margin_mm"], b["float32_ulp_margin_mm"])
        points = skin.copy()
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        support = np.arange(3, dtype=np.int64)
        bones = {(43, 16, 133): bone}
        x = old.build_fixed_bone_outward_plane_rows(points, faces, support, [], [self.pair], bones)
        y = m.build_fixed_bone_outward_plane_rows(points, faces, support, [], [self.pair], bones)
        self.assertEqual(set(x), set(y))
        np.testing.assert_array_equal(x["G"].toarray(), y["G"].toarray())
        np.testing.assert_array_equal(x["lower"], y["lower"])
        np.testing.assert_array_equal(x["C"].toarray(), y["C"].toarray())
        np.testing.assert_array_equal(x["h"], y["h"])
        np.testing.assert_array_equal(x["weights"], y["weights"])

    def test_sat_axis_chooses_cheaper_lateral_candidate_and_rows_separate_whole_pair(self):
        face_only = m.outward_plane_for_pair(self.skin, self.bone)
        sat = m.triangle_sat_plane_for_pair(self.skin, self.bone)
        self.assertEqual(sat["axis_strategy"], "triangle_sat_minimum_max_deficit")
        self.assertEqual(sat["axis_source"], "edge_cross")
        self.assertGreaterEqual(sat["axis_outward_dot"], 0.0)
        self.assertEqual(float(np.linalg.norm(sat["normal_world_unit"])), 1.0)
        face_cost = max(float(np.max(self.bone @ face_only["normal_world_unit"])
                              - np.min(self.skin @ face_only["normal_world_unit"])
                              + face_only["float32_ulp_margin_mm"]), 0.0)
        self.assertLess(sat["axis_cost_mm"], face_cost)
        ulp_skin = m._float32_ulp_bound(self.skin)
        ulp_bone = m._float32_ulp_bound(self.bone)
        costs = []
        for candidate in m.triangle_sat_axis_candidates(self.skin, self.bone):
            axis = candidate["normal_world_unit"]
            margin = (np.max(np.sum(np.abs(axis)[None, :] * ulp_bone, axis=1))
                      + np.max(np.sum(np.abs(axis)[None, :] * ulp_skin, axis=1)))
            costs.append(max(float(np.max(self.bone @ axis) - np.min(self.skin @ axis) + margin), 0.0))
        self.assertEqual(sat["axis_cost_mm"], min(costs))
        # Each of the three inequalities enforces its corner beyond the full
        # fixed triangle's maximum projection plus the Float32 margin.
        n = sat["normal_world_unit"]
        displacement = sat["corner_lower_mm"][:, None] * n[None, :]
        moved = self.skin + displacement
        bone_max = float(np.max(self.bone @ n))
        self.assertGreaterEqual(float(np.min(moved @ n)),
                                bone_max + sat["float32_ulp_margin_mm"] - 1e-12)

        result = m.build_fixed_bone_outward_plane_rows(
            self.skin, np.array([[0, 1, 2]]), np.arange(3), [], [self.pair],
            {(43, 16, 133): self.bone}, separating_axis_pairs={self.identity})
        self.assertEqual(result["G"].shape, (3, 9))
        self.assertEqual(result["diagnostics"]["triangle_sat_axis_pair_count"], 1)
        d = np.zeros((3, 3))
        for row, meta in enumerate(result["guard_metadata"]):
            d[meta["corner_index"]] = result["lower"][row] * np.asarray(meta["normal_world_unit"])
        np.testing.assert_array_less(result["lower"] - 1e-12,
                                     result["G"] @ d.reshape(-1))
        moved_by_rows = self.skin + d
        n = np.asarray(result["guard_metadata"][0]["normal_world_unit"])
        fixed_max = float(np.max(self.bone @ n))
        margin = result["guard_metadata"][0]["float32_ulp_margin_mm"]
        self.assertGreaterEqual(float(np.min(moved_by_rows @ n)), fixed_max + margin - 1e-12)

    def test_both_tangent_signs_are_allowed_and_axis_choice_is_deterministic(self):
        candidates = m.triangle_sat_axis_candidates(self.skin, self.bone)
        self.assertTrue(candidates)
        for c in candidates:
            self.assertGreaterEqual(c["axis_outward_dot"], 0.0)
        tangent = [c for c in candidates if c["axis_outward_dot"] == 0.0]
        self.assertTrue(any(c["axis_sign"] == 1 for c in tangent))
        self.assertTrue(any(c["axis_sign"] == -1 for c in tangent))
        a = m.triangle_sat_plane_for_pair(self.skin, self.bone)
        b = m.triangle_sat_plane_for_pair(self.skin, self.bone)
        self.assertEqual((a["axis_source"], a["axis_index"], a["axis_sign"], a["axis_cost_mm"]),
                         (b["axis_source"], b["axis_index"], b["axis_sign"], b["axis_cost_mm"]))
        np.testing.assert_array_equal(a["normal_world_unit"], b["normal_world_unit"])

    def test_sat_selection_is_per_pair_and_unknown_identity_fails_closed(self):
        second_bone = self.bone + np.array([0., 20., 0.])
        second_pair = dict(face_row=0, body=44, bone_id=17, triangle_id=134)
        out = m.build_fixed_bone_outward_plane_rows(
            self.skin, np.array([[0, 1, 2]]), np.arange(3), [],
            [self.pair, second_pair],
            {(43, 16, 133): self.bone, (44, 17, 134): second_bone},
            separating_axis_pairs={self.identity})
        kinds = [out["guard_metadata"][0]["axis_strategy"],
                 out["guard_metadata"][3]["axis_strategy"]]
        self.assertEqual(kinds, ["triangle_sat_minimum_max_deficit", "bone_face_458_compatible"])
        with self.assertRaisesRegex(ValueError, "absent from pairs"):
            m.build_fixed_bone_outward_plane_rows(
                self.skin, np.array([[0, 1, 2]]), np.arange(3), [], [self.pair],
                {(43, 16, 133): self.bone}, separating_axis_pairs={(1, 1, 1, 1)})


if __name__ == "__main__":
    unittest.main()
