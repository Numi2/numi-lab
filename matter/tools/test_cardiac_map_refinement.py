"""Admission tests for source-bound corrections to the existing cardiac map."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from cardiac_geometry_binding import _load_ventricular_wall_map_refinement, _wall_closure_from_native_parameters, _pack_refined_wall_map


class CardiacMapRefinementTest(unittest.TestCase):
    def descriptor(self, version=2):
        row = {"vertex": 0, "rv_coefficient_m": [.001, 0, 0],
               "lv_coefficient_m": [0, .001, 0], "rv_delta_m": [.001, 0, 0],
               "lv_delta_m": [0, .001, 0]}
        values = row["rv_coefficient_m"] + row["lv_coefficient_m"]
        if version == 2:
            row.update(closure_coefficient=[.1, 0, 0], closure_delta=[.1, 0, 0])
            values += row["closure_coefficient"]
        data = {"schema": f"numi.human.cardiac.ventricular_wall_map_refinement.v{version}",
                "method": f"joint_selected_state_affine_harmonic_map_conditioning_v{version}",
                "interpretation": "inferred_reference_registration_not_measured_subject_geometry",
                "source_payload_sha256": "a"*64, "arrangement_candidate_sha256": "b"*64,
                "vertex_count": 4, "q_domain": {"q_rv": [-.1, .1], "q_lv": [-.1, .1]},
                "corrections": [row], "max_closure_shift_m": .0001,
                "max_displacement_bound_m": .001302 if version == 2 else .000302,
                "checked_phases": [{"step": i, "exact_self_intersection_pairs": 0,
                                    "exact_ra_intersection_pairs": 0, "exact_la_intersection_pairs": 0}
                                   for i in range(8)]}
        for key in ("base_map_binary_sha256", "corrected_map_binary_sha256",
                    "candidate_coefficients_sha256", "phase_audit_report_sha256"):
            data[key] = "c"*64
        data["correction_records_sha256"] = hashlib.sha256(
            struct.pack("<I" + str(len(values)) + "f", 0, *values)).hexdigest()
        if version == 2:
            data["closure_domain_m"] = [-.01, .01]
        return data

    def load(self, data, owners=None):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/"refinement.json"
            path.write_text(json.dumps(data))
            return _load_ventricular_wall_map_refinement(
                path, source_payload_sha256="a"*64, candidate_sha256="b"*64,
                points=np.zeros((4, 3)), faces=np.array([[0, 1, 2]]),
                owners=owners or ["ID23_wall"])

    def test_v1_compatibility_and_v2_closure_bound(self):
        first = self.load(self.descriptor(1))
        second = self.load(self.descriptor(2))
        self.assertAlmostEqual(first["validated_max_displacement_bound_m"], .000301, places=9)
        self.assertAlmostEqual(second["validated_max_displacement_bound_m"], .001301, places=9)

    def test_changed_closure_bytes_require_matching_digest(self):
        data = self.descriptor()
        data["corrections"][0]["closure_coefficient"][0] += .001
        with self.assertRaisesRegex(ValueError, "canonical source-bound digest"):
            self.load(data)

    def test_closure_correction_cannot_understate_displacement(self):
        data = self.descriptor()
        data["max_displacement_bound_m"] = .000302
        with self.assertRaisesRegex(ValueError, "understated"):
            self.load(data)

    def test_full_solver_bracket_is_required(self):
        data = self.descriptor()
        data["closure_domain_m"] = [-.005, .005]
        with self.assertRaisesRegex(ValueError, "full existing"):
            self.load(data)

    def test_owned_lumen_boundary_cannot_be_conditioned(self):
        with self.assertRaisesRegex(ValueError, "owned lumen boundary"):
            self.load(self.descriptor(), ["ID319_RV_lumen"])

    def test_distinct_clear_geometry_states_are_required(self):
        for mutation in ("duplicate", "self", "ra", "la"):
            with self.subTest(mutation=mutation):
                data = self.descriptor()
                if mutation == "duplicate":
                    data["checked_phases"][1]["step"] = 0
                else:
                    key = {"self": "exact_self_intersection_pairs",
                           "ra": "exact_ra_intersection_pairs", "la": "exact_la_intersection_pairs"}[mutation]
                    data["checked_phases"][0][key] = 1
                with self.assertRaises(ValueError):
                    self.load(data)

    def test_source_identity_and_nonfinite_closure_are_rejected(self):
        for mutation in ("source", "closure"):
            with self.subTest(mutation=mutation):
                data = self.descriptor()
                if mutation == "source":
                    data["source_payload_sha256"] = "d"*64
                else:
                    data["corrections"][0]["closure_delta"][0] = float("nan")
                with self.assertRaises(ValueError):
                    self.load(data)

    def test_v1_does_not_accept_v2_method(self):
        data = self.descriptor(1)
        data["method"] = self.descriptor(2)["method"]
        with self.assertRaisesRegex(ValueError, "method/provenance"):
            self.load(data)


class SparseMapPackingTest(unittest.TestCase):
    def test_signed_zero_is_a_byte_change_and_reserved_lanes_are_preserved(self):
        base = np.zeros((2, 12), dtype=np.float32)
        base[0, 0] = -0.0
        base[1, 7] = -0.0
        field = np.zeros((2, 3, 3), dtype=np.float32)
        changed, packed = _pack_refined_wall_map(base, field, np.array([False, True]))
        self.assertEqual(changed.tolist(), [0])
        self.assertEqual(packed[0, 0].view(np.uint32), np.float32(0).view(np.uint32))
        self.assertEqual(packed[1].tobytes(), base[1].tobytes())

    def test_signed_zero_change_on_lumen_owned_vertex_is_rejected(self):
        base = np.zeros((1, 12), dtype=np.float32)
        base[0, 0] = -0.0
        field = np.zeros((1, 3, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "lumen-owned"):
            _pack_refined_wall_map(base, field, np.array([True]))


class NativePolynomialAdmissionTest(unittest.TestCase):
    def parameters(self):
        p = np.zeros(32, dtype=np.float32)
        p[0], p[10] = .00008, .0001
        p[20:24] = [.4, .2, .01, .0001]
        p[24:28] = [-.01, .01, 1e-5, 0]
        return p

    def test_linear_reference_has_unique_two_mm_closure(self):
        closure = _wall_closure_from_native_parameters(self.parameters(), np.float32(0), np.float32(0))
        self.assertAlmostEqual(float(closure), .002, places=8)

    def test_nonmonotone_or_unbracketed_polynomial_is_rejected(self):
        for mutation in ("derivative", "target"):
            with self.subTest(mutation=mutation):
                p = self.parameters()
                p[10 if mutation == "derivative" else 23] = -.0001 if mutation == "derivative" else .1
                with self.assertRaisesRegex(ValueError, "monotone bracket"):
                    _wall_closure_from_native_parameters(p, np.float32(0), np.float32(0))


if __name__ == "__main__":
    unittest.main()
