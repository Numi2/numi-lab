import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

OWNER = Path("/Users/n/Work/numi-human-connected-surface-069/D/connected-local-witness-proposal-365/propose_joint_clearance_witnesses_365.py")
HELPER = Path(__file__).with_name("precision_phase_one_392.py")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


phase = load_module("precision_phase_one_392_test", HELPER)
owner = load_module("proposal_365_phase_one_test", OWNER)


def report_for(rhs, margins, bounds):
    rows = []
    for i, (b, margin, bound) in enumerate(zip(rhs, margins, bounds)):
        rows.append({
            "row_index": i,
            "delta_rhs_mm": float(b),
            "preserved_pair_margin_mm": float(margin),
            "rounding_bound_within_axiswise_interval_mm": float(bound),
            "label": f"100:{200+i}:A{i}",
        })
    return {
        "diagnostic_only": True,
        "candidate_sha256": "a" * 64,
        "rows_payload_sha256": "b" * 64,
        "summary": {"row_count": len(rows)},
        "rows": rows,
    }


class PrecisionPhaseOneTests(unittest.TestCase):
    def test_appends_only_needed_rows_and_preserves_exact_prefix(self):
        A = sparse.csr_matrix([[1., 0.], [0., 1.], [1., 1.]])
        b = np.array([-0.4, -0.2, 0.0])
        report = report_for(b, [0.9, 0.3, 0.3], [0.2, 0.6, 0.25])
        A2, b2, receipt = phase.build_precision_phase_one_system(
            A, b, report, expected_candidate_sha256="a" * 64,
            base_labels=[r["label"] for r in report["rows"]],
        )
        self.assertEqual(A2.shape, (4, 2))
        self.assertEqual(b2.shape, (4,))
        self.assertEqual((A2[:3] != A).nnz, 0)
        np.testing.assert_array_equal(b2[:3], b)
        self.assertEqual(receipt["phase_one_source_row_indices"], [1])
        self.assertAlmostEqual(b2[3], -0.2 + (0.6 + 2e-7 - 0.3), places=14)
        self.assertTrue(receipt["original_rows_preserved_as_exact_prefix"])
        self.assertFalse(receipt["native_admission"])

    def test_empty_precision_increment_preserves_system(self):
        A = sparse.eye(2, format="csr")
        b = np.array([-1., -2.])
        report = report_for(b, [0.5, 0.9], [0.1, 0.2])
        A2, b2, receipt = phase.build_precision_phase_one_system(A, b, report)
        self.assertEqual(A2.shape, A.shape)
        self.assertEqual((A2 != A).nnz, 0)
        np.testing.assert_array_equal(b2, b)
        self.assertEqual(receipt["phase_one_duplicate_row_count"], 0)

    def test_stale_rhs_hash_and_incomplete_row_reports_fail_closed(self):
        A = sparse.eye(2, format="csr")
        b = np.array([-1., -2.])
        report = report_for(b, [1., 1.], [0., 0.])
        with self.assertRaisesRegex(ValueError, "candidate hash"):
            phase.build_precision_phase_one_system(A, b, report, expected_candidate_sha256="c" * 64)
        report2 = report_for(b, [1., 1.], [0., 0.])
        report2["rows"][0]["delta_rhs_mm"] = -0.5
        with self.assertRaisesRegex(ValueError, "RHS"):
            phase.build_precision_phase_one_system(A, b, report2)
        report3 = report_for(b, [1., 1.], [0., 0.])
        report3["rows"].pop()
        report3["summary"]["row_count"] = 1
        with self.assertRaisesRegex(ValueError, "cover every"):
            phase.build_precision_phase_one_system(A, b, report3)

    def test_duplicate_bad_and_under_certified_inputs_fail(self):
        A = sparse.eye(2, format="csr")
        b = np.array([-1., -2.])
        report = report_for(b, [0.1, 0.1], [0.2, 0.2])
        report["rows"][1]["row_index"] = 0
        with self.assertRaisesRegex(ValueError, "unique"):
            phase.build_precision_phase_one_system(A, b, report)
        report = report_for(b, [0.1, 0.1], [0.2, 0.2])
        report["rows"][0]["rounding_bound_within_axiswise_interval_mm"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid"):
            phase.build_precision_phase_one_system(A, b, report)
        report = report_for(b, [0.1, 0.1], [0.2, 0.2])
        with self.assertRaisesRegex(ValueError, "cert_tolerance"):
            phase.build_precision_phase_one_system(A, b, report, cert_tolerance_mm=0.0)

    def test_zero_row_cannot_be_strengthened(self):
        A = sparse.csr_matrix([[0.]])
        b = np.array([0.])
        report = report_for(b, [0.1], [0.2])
        with self.assertRaisesRegex(ValueError, "zero hard row"):
            phase.build_precision_phase_one_system(A, b, report)

    def test_existing_basis_solver_respects_fixed_anchor_and_strengthened_row(self):
        A = sparse.csr_matrix([[1., 0.], [0., 1.]])
        b = np.array([-1., -0.2])
        report = report_for(b, [1.0, 0.3], [0.1, 0.6])
        # First output coordinate is an immutable anchor; second is the only DOF.
        M = sparse.csr_matrix([[0.], [1.]])
        result = phase.solve_precision_phase_one(
            owner.solve_sparse_l1_clearance_in_basis,
            A, b, report,
            np.array([0., -1.]), np.array([0., 1.]), M,
            time_limit_s=5.0,
            expected_candidate_sha256="a" * 64,
        )
        self.assertTrue(result["success"], result)
        d = result["delta"]
        self.assertEqual(float(d[0]), 0.0)
        self.assertGreaterEqual(float(d[1]), 0.1)
        self.assertGreaterEqual(result["precision_phase_one"]["minimum_hard_residual_mm"], -2e-7)
        self.assertEqual(result["precision_phase_one"]["classification"], "CONSTRUCTION_PRECISION_PHASE_ONE")
        self.assertTrue(result["precision_phase_one"]["exact_emitted_f32_reaudit_required"])


if __name__ == "__main__":
    unittest.main()

