import unittest
from pathlib import Path

import numpy as np

from check_recycle_basis import (
    orthogonalize,
    orthogonalize_masked_candidate,
    orthonormalize_projected_basis,
)


class RecycleBasisGateTests(unittest.TestCase):
    def setUp(self):
        self.base = np.eye(4, dtype=np.float64)[:3]

    def test_independent_finite_direction_is_normalized_and_orthogonal(self):
        value, ratio, accepted = orthogonalize(np.array([1.0, 2.0, 3.0, 1.0]), self.base)
        self.assertTrue(accepted)
        self.assertAlmostEqual(float(np.linalg.norm(value)), 1.0, places=14)
        self.assertLess(float(np.max(np.abs(self.base @ value))), 1.0e-14)
        self.assertGreater(ratio, 0.0)

    def test_zero_and_dependent_vectors_are_rejected(self):
        _, _, zero = orthogonalize(np.zeros(4), self.base)
        _, _, dependent = orthogonalize(np.array([1.0, 0.0, 0.0, 0.0]), self.base)
        self.assertFalse(zero)
        self.assertFalse(dependent)

    def test_nonfinite_vector_is_rejected(self):
        _, _, accepted = orthogonalize(np.array([np.inf, 0.0, 0.0, 0.0]), self.base)
        self.assertFalse(accepted)


class DynamicMaskMgsTests(unittest.TestCase):
    def test_masked_basis_is_reorthogonalized_before_candidate_projection(self):
        basis = np.array([
            [1.0, 1.0, 0.0, 0.0],
            [1.0, -1.0, 1.0, 1.0],
        ], dtype=np.float64)
        basis[0] /= np.linalg.norm(basis[0])
        basis[1] /= np.linalg.norm(basis[1])
        basis[:, 1] = 0.0
        self.assertGreater(abs(float(np.dot(basis[0], basis[1]))), 0.1)
        q, valid, rank, ambiguous = orthonormalize_projected_basis(basis)
        self.assertTrue(valid)
        self.assertFalse(ambiguous)
        self.assertEqual(rank, 2)
        active = q[np.linalg.norm(q, axis=1) > 0.0]
        self.assertLess(float(np.max(np.abs(active @ active.T - np.eye(rank)))), 1.0e-14)

        candidate = np.array([0.2, -0.1, 0.7, 0.3], dtype=np.float64)
        projected, _, accepted, diagnostic = orthogonalize_masked_candidate(
            candidate, basis)
        q_reference, _ = np.linalg.qr(basis.T, mode="reduced")
        residual = candidate - q_reference @ (q_reference.T @ candidate)
        expected = residual / np.linalg.norm(residual)
        self.assertTrue(accepted)
        self.assertLess(float(np.linalg.norm(projected - expected)), 2.0e-12)
        self.assertLessEqual(diagnostic["max_normalized_source_correlation"], 1.0e-4)

    def test_masked_dependent_and_rank_ambiguous_basis_fail_closed(self):
        exact_dependent = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        q, valid, rank, ambiguous = orthonormalize_projected_basis(exact_dependent)
        self.assertTrue(valid)
        self.assertEqual(rank, 1)
        self.assertFalse(ambiguous)
        _, _, accepted, diagnostic = orthogonalize_masked_candidate(
            np.array([1.0, 0.0, 0.0]), exact_dependent)
        self.assertFalse(accepted)
        self.assertEqual(diagnostic["reason"], "dependent_candidate")

        near_dependent = np.array([[1.0, 0.0, 0.0], [1.0, 1.0e-6, 0.0]])
        _, valid, _, ambiguous = orthonormalize_projected_basis(near_dependent)
        self.assertFalse(valid)
        self.assertTrue(ambiguous)
        _, _, accepted, diagnostic = orthogonalize_masked_candidate(
            np.array([0.0, 1.0, 0.0]), near_dependent)
        self.assertFalse(accepted)
        self.assertEqual(diagnostic["reason"], "rank_ambiguous")

    def test_environment_major_workspace_offsets_are_disjoint(self):
        environments, modes, nodes = 2, 7, 11
        per_environment = modes * nodes
        offsets = [environment * per_environment for environment in range(environments)]
        self.assertEqual(offsets, [0, per_environment])
        self.assertEqual(offsets[0] + per_environment, offsets[1])
        self.assertEqual(offsets[-1] + per_environment, environments * modes * nodes)


class OverlayWiringTests(unittest.TestCase):
    def test_reset_select_capture_order_and_registration(self):
        root = Path(__file__).parent / "overlay/matter/src"
        runtime = (root / "runtime.mm").read_text()
        kernels = (root / "metal/fgmres.metalinc").read_text()
        for name in (
            "nm_fgmres_reset_regional_recycle",
            "nm_fgmres_capture_regional_recycle",
            "nm_fgmres_regional_select_recycle_mode",
        ):
            self.assertIn(f'"{name}"', runtime)
            self.assertIn(f"kernel void {name}", kernels)
        reset = runtime.index('"nm_fgmres_reset_regional_recycle"')
        newton_loop = runtime.index("for (std::uint32_t nonlinearIteration = 0u;")
        project = runtime.index('"nm_fgmres_regional_project_basis"')
        select = runtime.index('"nm_fgmres_regional_select_recycle_mode"')
        fem_apply = runtime.index('dispatchThreads("nm_fem_apply_solution"')
        capture = runtime.index('dispatchThreads("nm_fgmres_capture_regional_recycle"')
        self.assertLess(reset, newton_loop)
        self.assertLess(project, select)
        self.assertLess(fem_apply, capture)

    def test_flag_is_exact_opt_in_and_fingerprint_bound(self):
        root = Path(__file__).parent / "overlay/matter/src"
        runtime = (root / "runtime.mm").read_text()
        kernels = (root / "metal/fgmres.metalinc").read_text()
        self.assertIn('"NM_FEM_REGIONAL_ACCEPTED_CORRECTION_RECYCLE"', runtime)
        self.assertIn('std::strcmp(acceptedCorrectionRecycle, "1") != 0', runtime)
        self.assertIn("candidate->femRegionalAcceptedCorrectionRecycleEnabled\n        );", runtime)
        self.assertIn("femRegionalRecycleOrthogonalBasis", runtime)
        self.assertIn("environmentModeNodeCount", runtime)
        self.assertIn("accepted-correction recycle E*M*N allocation overflows size_t", runtime)
        self.assertIn("device float4* orthogonalBasis [[buffer(9)]]", kernels)
        self.assertIn("sourceBasisFinite", kernels)


if __name__ == "__main__":
    unittest.main()
