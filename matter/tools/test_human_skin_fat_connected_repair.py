import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy import sparse


MODULE_PATH = Path(__file__).with_name("human_skin_fat_connected_repair.py")
spec = importlib.util.spec_from_file_location("human_skin_fat_connected_repair_review", MODULE_PATH)
repair = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(repair)


def positive_tets(points, tets):
    out = np.asarray(tets, dtype=np.int64).copy()
    for row in range(len(out)):
        p = np.asarray(points, dtype=np.float64)[out[row]]
        det = np.linalg.det(np.stack((p[1] - p[0], p[2] - p[0], p[3] - p[0]), axis=1))
        if det < 0:
            out[row, 0], out[row, 1] = out[row, 1], out[row, 0]
        if det == 0:
            raise ValueError("degenerate test tetrahedron")
    return out


class ConnectedVolumeRepairTests(unittest.TestCase):
    def test_cell_j_gradients_match_central_finite_differences(self):
        reference = np.asarray([
            [0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.1, 0.9, 0.0], [0.2, 0.3, 1.2],
        ], dtype=np.float64)
        points = reference.copy()
        points[1] += [0.03, 0.02, -0.01]
        points[2] += [-0.02, 0.01, 0.04]
        points[3] += [0.01, -0.03, 0.02]
        tets = np.asarray([[0, 1, 2, 3]], dtype=np.int64)

        J, G, q, Q = repair.cell_values_gradients(points, reference, tets)
        h = 2e-7
        fd_j = np.zeros((4, 3))
        fd_q = np.zeros((4, 3))
        for node in range(4):
            for axis in range(3):
                xp = points.copy(); xp[node, axis] += h
                xm = points.copy(); xm[node, axis] -= h
                jp = repair.cell_values_gradients(xp, reference, tets)[0][0]
                jm = repair.cell_values_gradients(xm, reference, tets)[0][0]
                qp = repair.cell_values_gradients(xp, reference, tets)[2][0]
                qm = repair.cell_values_gradients(xm, reference, tets)[2][0]
                fd_j[node, axis] = (jp - jm) / (2 * h)
                fd_q[node, axis] = (qp - qm) / (2 * h)
        ref_det = np.linalg.det(np.stack((reference[1]-reference[0], reference[2]-reference[0], reference[3]-reference[0]), axis=1))
        np.testing.assert_allclose(G[0], fd_j, rtol=2e-6, atol=2e-8)
        np.testing.assert_allclose(Q[0], fd_q, rtol=2e-5, atol=2e-8)

    def test_sparse_cell_jacobian_couples_shared_tet_nodes(self):
        points = np.asarray([
            [0, 0, 0], [1.2, .05, 0], [.1, .95, .03], [.02, .1, 1.15], [-.07, .14, -1.23],
        ], dtype=np.float64)
        tets = positive_tets(points, np.asarray([[0, 1, 2, 3], [0, 2, 1, 4]]))
        free = np.asarray([0, 1, 2], dtype=np.int64)
        J, G, q, Q = repair.cell_values_gradients(points, points, tets)
        jmat = repair.sparse_cell_jacobian(G / J[:, None, None], tets, free, len(points))
        qmat = repair.sparse_cell_jacobian(Q, tets, free, len(points))
        direction = np.asarray([.3, -.2, .4, -.1, .5, .2, .4, .1, -.3])
        h = 1e-7
        moved = points.copy()
        moved[free] += h * direction.reshape(-1, 3)
        j2, _, q2, _ = repair.cell_values_gradients(moved, points, tets)
        np.testing.assert_allclose((j2 - J) / h, jmat @ direction, rtol=3e-5, atol=2e-7)
        np.testing.assert_allclose((q2 - q) / h, qmat @ direction, rtol=5e-5, atol=2e-7)
        # Each tetra contributes derivatives for all shared free vertices.
        self.assertEqual(jmat.shape, (2, 9))
        self.assertGreater(jmat.nnz, 0)

    def test_backtracking_keeps_fixed_nodes_and_shared_cells_valid(self):
        points = np.asarray([
            [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1],
        ], dtype=np.float32)
        tets = positive_tets(points, np.asarray([[0, 1, 2, 3], [0, 2, 1, 4]]))
        free = np.asarray([0, 1, 2], dtype=np.int64)
        n3 = 3 * len(points)
        contact = np.zeros((2, n3), dtype=np.float64)
        contact[0, 0] = 1.0
        contact[1, 3] = 1.0
        target = contact @ points.astype(np.float64).ravel() + np.asarray([2e-4, -2e-4])
        layer = np.zeros((1, n3), dtype=np.float64)
        layer[0, 7] = 1.0  # node 2, y coordinate
        layer_target = np.asarray([float(points[2, 1]) + 1e-4])

        result, report = repair.coupled_clearance_step(
            points, points.astype(np.float64), tets, free, sparse.csr_matrix(contact), target,
            step_limit_mm=.5, q_min=1e-5, J_bounds=(.35, 2.5), time_limit_s=5.,
            lower_linear_matrix=sparse.csr_matrix(layer), lower_linear_targets=layer_target,
        )
        self.assertTrue(report["lp_success"], report)
        self.assertTrue(report["step_applied"], report)
        self.assertEqual(len(report["components"]), 1)  # shared-cell graph couples the free nodes
        np.testing.assert_array_equal(result[3:].view(np.uint32), points[3:].view(np.uint32))
        self.assertGreaterEqual(float((layer @ result.astype(np.float64).ravel())[0]), layer_target[0] - 1e-9)
        self.assertLess(float(np.maximum(target - contact @ result.astype(np.float64).ravel(), 0).sum()),
                        float(np.maximum(target - contact @ points.astype(np.float64).ravel(), 0).sum()))
        jj, _, qq, _ = repair.cell_values_gradients(result, points.astype(np.float64), tets)
        self.assertGreaterEqual(float(qq.min()), 1e-5)
        self.assertGreaterEqual(float(jj.min()), .35)
        self.assertLessEqual(float(jj.max()), 2.5)


if __name__ == "__main__":
    unittest.main()
