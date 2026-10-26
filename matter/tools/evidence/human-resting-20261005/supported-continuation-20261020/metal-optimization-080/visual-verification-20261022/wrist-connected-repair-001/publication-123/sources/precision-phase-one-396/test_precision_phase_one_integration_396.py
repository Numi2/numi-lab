import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("phase_one_integration_396_test", HERE / "precision_phase_one_integration_396.py")
phase = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(phase)


def sha_bytes(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(value):
    return phase.sha256_array(np.asarray(value))


class PrecisionPhaseOneIntegrationTests(unittest.TestCase):
    def fixture(self, directory):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
        base_rows = [
            {"nodes": np.array([0, 1]), "coefficients": np.array([[1., 0., 0.], [-1., 0., 0.]]), "lower": -0.2, "label": "0:1:A"},
            {"nodes": np.array([1, 2]), "coefficients": np.array([[0., 1., 0.], [0., -1., 0.]]), "lower": -0.1, "label": "0:1:B"},
            {"nodes": np.array([1, 3]), "coefficients": np.array([[1., 0., 0.], [-1., 0., 0.]]), "lower": -0.3, "label": "1:2:C"},
        ]
        pair_records = [
            {"faces": [0, 1], "kind": "common_vertex_wedge"},
            {"faces": [1, 2], "kind": "moving_sat_midplane"},
        ]
        report_rows = [
            {"row_index": 0, "label": "0:1:A", "delta_rhs_mm": -0.2, "pair_face_rows": [0, 1], "pair_kind": "common_vertex_wedge", "preserved_pair_margin_mm": 0.01, "rounding_bound_within_axiswise_interval_mm": 0.02},
            {"row_index": 1, "label": "0:1:B", "delta_rhs_mm": -0.1, "pair_face_rows": [0, 1], "pair_kind": "common_vertex_wedge", "preserved_pair_margin_mm": 0.2, "rounding_bound_within_axiswise_interval_mm": 0.001},
            {"row_index": 2, "label": "1:2:C", "delta_rhs_mm": -0.3, "pair_face_rows": [1, 2], "pair_kind": "moving_sat_midplane", "preserved_pair_margin_mm": 0.01, "rounding_bound_within_axiswise_interval_mm": 0.001},
        ]
        payload = Path(directory) / "rows.npz"
        ptr = np.array([0, 2, 4, 6], dtype=np.int64)
        nodes = np.concatenate([r["nodes"] for r in base_rows]).astype(np.int64)
        coeff = np.concatenate([r["coefficients"] for r in base_rows]).astype(np.float64)
        np.savez_compressed(
            payload, row_indptr=ptr, node_rows=nodes, coefficients_mm=coeff,
            lower_mm=np.array([r["lower"] for r in base_rows], dtype=np.float64),
            labels=np.array([r["label"] for r in base_rows]),
            pair_faces=np.array([[0, 1], [1, 2]], dtype=np.int64),
            pair_kind=np.array(["common_vertex_wedge", "moving_sat_midplane"]),
        )
        report = {
            "diagnostic_only": True,
            "candidate_sha256": "a" * 64,
            "seed_pairs_sha256": "b" * 64,
            "owner_sha256": "c" * 64,
            "candidate_points_sha256": array_sha(points),
            "candidate_faces_sha256": array_sha(faces),
            "points_shape": list(points.shape), "faces_shape": list(faces.shape),
            "points_dtype": "float32", "selected_pair_count": 2, "margin_fraction": 0.25,
            "rows_payload_sha256": sha_bytes(payload),
            "summary": {"row_count": 3}, "rows": report_rows,
        }
        return points, faces, base_rows, pair_records, report, payload

    def validate(self, fixture, **overrides):
        points, faces, rows, records, report, payload = fixture
        kwargs = dict(
            points_world_m_f32=points, faces=faces,
            base_candidate_sha256="a" * 64, seed_pairs_sha256="b" * 64,
            pair_owner_sha256="c" * 64, margin_fraction=0.25,
            report=report, report_sha256="d" * 64, payload_path=payload,
            base_rows=rows, pair_records=records, safety_factor=4.0,
            cert_tolerance_mm=2e-7,
        )
        kwargs.update(overrides)
        return phase.validate_and_build_phase_rows(**kwargs)

    def test_appends_only_needed_factor4_rows_preserving_base(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = self.fixture(td)
            base_rows = fixture[2]
            cloned, receipt = self.validate(fixture)
            self.assertEqual(receipt["phase_one_source_row_indices"], [0])
            self.assertEqual(len(cloned), 1)
            self.assertAlmostEqual(cloned[0]["lower"], -0.2 + (4.0 * (0.02 + 2e-7) - 0.01), places=14)
            self.assertEqual(cloned[0]["nodes"].tolist(), base_rows[0]["nodes"].tolist())
            self.assertEqual(cloned[0]["label"], "0:1:A:F32_PHASE1")
            self.assertTrue(receipt["original_self_rows_preserved_as_exact_prefix"])
            self.assertFalse(receipt["native_admission"])
            self.assertEqual(base_rows[0]["lower"], -0.2)
            self.assertEqual(base_rows[0]["label"], "0:1:A")

    def test_rejects_candidate_seed_owner_or_payload_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = self.fixture(td)
            with self.assertRaisesRegex(ValueError, "candidate_sha256"):
                self.validate(fixture, base_candidate_sha256="e" * 64)
            with self.assertRaisesRegex(ValueError, "seed_pairs_sha256"):
                self.validate(fixture, seed_pairs_sha256="e" * 64)
            with self.assertRaisesRegex(ValueError, "owner_sha256"):
                self.validate(fixture, pair_owner_sha256="e" * 64)
            changed = list(fixture)
            changed[4] = dict(fixture[4])
            changed[4]["rows_payload_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "payload bytes"):
                self.validate(tuple(changed))

    def test_rejects_payload_row_or_precision_report_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = self.fixture(td)
            rows = list(fixture[2])
            rows[0] = dict(rows[0], lower=-0.25)
            with self.assertRaisesRegex(ValueError, "RHS"):
                self.validate((fixture[0], fixture[1], rows, *fixture[3:]))
            report = dict(fixture[4])
            report["candidate_points_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "candidate_points_sha256"):
                self.validate((fixture[0], fixture[1], fixture[2], fixture[3], report, fixture[5]))

    def test_emitted_float32_rows_require_strengthened_residual(self):
        base = np.zeros((2, 3), dtype=np.float32)
        row = {"nodes": np.array([0]), "coefficients": np.array([[1., 0., 0.]]), "lower": 0.1, "label": "0:1:x"}
        pass_points = base.copy()
        pass_points[0, 0] = np.float32(0.0001)
        passed = phase.evaluate_emitted_self_rows(base, pass_points, [row], tolerance_mm=2e-5)
        self.assertTrue(passed["pass"], passed)
        fail_points = base.copy()
        failed = phase.evaluate_emitted_self_rows(base, fail_points, [row], tolerance_mm=2e-5)
        self.assertFalse(failed["pass"])
        self.assertEqual(failed["failed_row_indices_first_64"], [0])

    def test_run_assembly_attaches_clearance_omission_after_phase_receipt_load(self):
        base = [{"label": "original-hard-row"}]
        phase_row = {"label": "duplicate-hard-row"}
        calls = []

        def load_phase_rows():
            calls.append("loaded")
            return [phase_row], {
                "phase_one_source_row_indices": [0],
                "classification": "CONSTRUCTION_PRECISION_PHASE_ONE",
            }

        assembled, receipt, source_rows = phase.assemble_phase_one_self_rows(
            base, enabled=True, omitted_soft_clearance_rows=17, load_rows=load_phase_rows
        )
        self.assertEqual(calls, ["loaded"])
        self.assertIs(assembled[0], base[0])
        self.assertIs(assembled[1], phase_row)
        self.assertEqual(receipt["soft_clearance_rows_omitted"], 17)
        self.assertEqual(receipt["soft_clearance_target"], "empty; phase-I objective is zero displacement")
        self.assertEqual(receipt["base_self_row_count"], 1)
        self.assertEqual(receipt["duplicate_self_row_start"], 1)
        self.assertEqual(receipt["duplicate_self_row_stop"], 2)
        np.testing.assert_array_equal(source_rows, np.array([0], dtype=np.int64))

    def test_run_assembly_default_path_preserves_rows_without_loader(self):
        base = [{"label": "original-hard-row"}]
        assembled, receipt, source_rows = phase.assemble_phase_one_self_rows(
            base, enabled=False
        )
        self.assertIs(assembled[0], base[0])
        self.assertIsNone(receipt)
        self.assertEqual(source_rows.shape, (0,))

    def test_invalid_safety_or_tolerance_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            fixture = self.fixture(td)
            with self.assertRaisesRegex(ValueError, "safety factor"):
                self.validate(fixture, safety_factor=0.5)
            with self.assertRaisesRegex(ValueError, "certificate tolerance"):
                self.validate(fixture, cert_tolerance_mm=1e-8)


if __name__ == "__main__":
    unittest.main()

