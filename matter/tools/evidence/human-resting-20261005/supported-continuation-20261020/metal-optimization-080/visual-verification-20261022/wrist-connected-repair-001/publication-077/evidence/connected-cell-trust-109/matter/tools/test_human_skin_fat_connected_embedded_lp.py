import unittest

import numpy as np
from scipy import sparse

from human_skin_fat_connected_repair import (
    cell_values_gradients,
    coupled_clearance_step,
)
from human_skin_fat_embedded_surface import MappedSkinSurface
from human_skin_fat_surface_separation import build_pair_inequalities


class ConnectedEmbeddedLPTests(unittest.TestCase):
    def test_exact_float32_detail_fold_backtracks_while_coarse_cell_stays_positive(self):
        X = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=np.float32,
        )
        T = np.array([[0, 1, 2, 3]], dtype=np.int64)
        detail = np.array(
            [[0, 0, 0], [.125, 0, 0], [0, .125, 0], [.25, .25, .125]],
            dtype=np.float32,
        )
        W = sparse.csr_matrix(
            ([1., 1., 1.], ([0, 1, 2], [0, 1, 2])), shape=(4, 4)
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        surface = MappedSkinSurface(X, detail, faces, W)
        contact = sparse.csr_matrix(([1.], ([0], [3 * 1 + 2])), shape=(1, 12))
        targets = np.array([.5])

        common = dict(
            step_limit_mm=500.,
            q_min=1e-5,
            J_bounds=(.35, 2.5),
            max_incremental_gradient=None,
            construction_origin=X,
            cumulative_displacement_weight=.01,
            time_limit_s=10.,
        )
        unguarded, plain_report = coupled_clearance_step(
            X, X, T, np.arange(4), contact, targets, **common
        )
        J_plain, _, q_plain, _ = cell_values_gradients(unguarded, X, T)
        self.assertTrue(plain_report["lp_success"])
        self.assertGreater(float(J_plain.min()), .35)
        self.assertGreater(float(q_plain.min()), 1e-5)
        self.assertFalse(surface.passes(unguarded))

        guarded, report = coupled_clearance_step(
            X, X, T, np.arange(4), contact, targets,
            embedded_surface=surface, **common
        )
        J_guarded, _, q_guarded, _ = cell_values_gradients(guarded, X, T)
        self.assertTrue(report["lp_success"])
        self.assertTrue(report["step_applied"])
        self.assertTrue(surface.passes(guarded))
        self.assertTrue(report["mapped_skin_geometry_pass"])
        self.assertGreater(float(J_guarded.min()), .35)
        self.assertGreater(float(q_guarded.min()), 1e-5)
        self.assertGreater(guarded[1, 2], X[1, 2])
        self.assertLess(max(c["line_fraction"] for c in report["components"]), 1.)

    def test_surface_separator_rows_remain_hard_in_connected_lp(self):
        X = np.array(
            [
                [-1., 0., 0.], [-1., 0., 1.], [-1., 1., 0.], [-2., 1., 1.],
                [.05, 0., 0.], [.05, 1., 0.], [.05, 0., 1.], [1.05, 1., 1.],
            ],
            dtype=np.float32,
        )
        T = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.int64)
        faces_a = np.array([0, 1, 2], dtype=np.int64)
        faces_b = np.array([4, 5, 6], dtype=np.int64)
        W = sparse.eye(8, format="csr")
        crossing = X.copy()
        crossing[[4, 5, 6], 0] = -1.1
        L, lt, provenance = build_pair_inequalities(
            faces_a, faces_b, X, W, X, X,
            candidate_detail_positions=crossing, pair_id="hard-lp-regression",
        )
        self.assertGreater(L.shape[0], 0)
        self.assertGreater(provenance["candidate_margin_violation_m"], 0.)
        self.assertGreaterEqual(float(np.min(L @ X.astype(float).ravel() - lt)), -1e-12)

        contact = sparse.csr_matrix(([-1.], ([0], [3 * 4])), shape=(1, 24))
        target = np.array([1.1])
        candidate, report = coupled_clearance_step(
            X, X, T, np.array([4, 5, 6, 7]), contact, target,
            lower_linear_matrix=L,
            lower_linear_targets=lt,
            step_limit_mm=1200.,
            q_min=1e-5,
            J_bounds=(.35, 2.5),
            max_incremental_gradient=None,
            construction_origin=X,
            cumulative_displacement_weight=.01,
            time_limit_s=10.,
        )
        self.assertTrue(report["lp_success"])
        self.assertTrue(report["step_applied"])
        self.assertGreaterEqual(
            float(np.min(L @ candidate.astype(float).ravel() - lt)), -1e-9
        )
        # The contact request is infeasible with the separator: its slack must
        # stay positive instead of softening the caller's mapped-surface rows.
        contact_deficit = float(np.max(target - contact @ candidate.astype(float).ravel()))
        self.assertGreater(contact_deficit, .5)
        J, _, q, _ = cell_values_gradients(candidate, X, T)
        self.assertGreater(float(J.min()), .35)
        self.assertGreater(float(q.min()), 1e-5)


if __name__ == "__main__":
    unittest.main()
