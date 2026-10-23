#!/usr/bin/env python3
import importlib.util
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("baseline_guard", Path(__file__).resolve().parents[1] / "tools/check_human_skin_layer_baseline.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {"outer_world_m": np.zeros((4, 3)), "skin_inner_world_m": np.ones((4, 3)),
                         "faces": np.array([[0, 1, 2], [1, 2, 3]]), "source_vertex_ids": np.arange(4),
                         "donor_body_indices": np.array([3, 4, 4, 5])}
        self.candidate = {k: v.copy() for k, v in self.baseline.items()}

    def test_only_declared_rows_change(self):
        self.candidate["outer_world_m"][1, 1] = .003
        self.assertTrue(guard.check_arrays(self.baseline, self.candidate, [1])["passed"])

    def test_stale_inner_layer_rejected(self):
        self.candidate["skin_inner_world_m"][2, 2] += 1e-12
        result = guard.check_arrays(self.baseline, self.candidate, [1])
        self.assertFalse(result["passed"])
        self.assertEqual(result["arrays"]["skin_inner_world_m"]["first_rows"], [2])

    def test_signed_zero_is_not_bit_exact(self):
        self.candidate["outer_world_m"][2, 0] = -0.0
        self.assertFalse(guard.check_arrays(self.baseline, self.candidate, [1])["passed"])

    def test_topology_and_identity_immutable_even_on_edited_row(self):
        for name in ["faces", "source_vertex_ids", "donor_body_indices"]:
            with self.subTest(name=name):
                candidate = {k: v.copy() for k, v in self.baseline.items()}
                candidate[name].flat[1] += 1
                self.assertFalse(guard.check_arrays(self.baseline, candidate, [1])["passed"])

    def test_nonfinite_rejected_inside_patch(self):
        self.candidate["outer_world_m"][1, 0] = np.nan
        with self.assertRaises(guard.BaselineError):
            guard.check_arrays(self.baseline, self.candidate, [1])

    def test_refinement_requires_different_ancestry_path(self):
        self.candidate["outer_world_m"] = np.zeros((5, 3))
        with self.assertRaises(guard.BaselineError):
            guard.check_arrays(self.baseline, self.candidate, [1])

    def test_invalid_allowlist_and_missing_arrays(self):
        for rows in [[], [1, 1], [-1], [4]]:
            with self.assertRaises(guard.BaselineError):
                guard.check_arrays(self.baseline, self.candidate, rows)
        del self.candidate["skin_inner_world_m"]
        with self.assertRaises(guard.BaselineError):
            guard.check_arrays(self.baseline, self.candidate, [1])

    def test_native_float_fields_round_before_double_deltas(self):
        frame = "WholeBodySkinDonorFrameV1{3u,{1.00000006f,0.f,0.f},{1.f,0.f,0.f,0.f,1.f,0.f,0.f,0.f,1.f},{1.00000006f,0.f,0.f},{0.f,0.f,1.f}}"
        names = ["SkinMechanicalWorldDeltas", "SkinMechanicalDonorLocalDeltas", "SkinInnerWorld", "SkinInnerLocal", "FatDeepWorld", "FatDeepLocal"]
        text = "frames={" + frame + "};\n"
        for name in names:
            text += "kWholeBody" + name + "={{std::array<double,3>{0.000000001,0.,0.}}};\n"
        for name in ["SkinThicknessM", "FatThicknessM"]:
            text += "kWholeBody" + name + "={0.001};\n"
        text += "kWholeBodySkinSourceVertices={4u};\nkWholeBodySkinSourceFaces={0u};\nkWholeBodySkinFaces={Face{0u,0u,0u}};\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.hpp"
            path.write_text(text)
            out = guard.read_baseline(path, hashlib.sha256(path.read_bytes()).hexdigest())
        expected = float(np.float32(1.00000006)) + .000000001
        self.assertEqual(out["outer_world_m"][0, 0], expected)
        self.assertEqual(out["outer_donor_local_m"][0, 0], expected)
        self.assertNotEqual(expected, 1.00000006 + .000000001)

    def test_wrong_native_hash_rejected_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.hpp"
            path.write_text("source identity alone is insufficient")
            with self.assertRaisesRegex(guard.BaselineError, "SHA-256"):
                guard.read_baseline(path, "0" * 64)


if __name__ == "__main__":
    unittest.main()
