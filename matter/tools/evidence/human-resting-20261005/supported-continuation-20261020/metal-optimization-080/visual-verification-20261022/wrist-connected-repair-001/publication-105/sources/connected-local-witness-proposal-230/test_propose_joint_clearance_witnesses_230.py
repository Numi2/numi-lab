"""Focused tests for 230."""
import importlib.util
import unittest
import numpy as np
from pathlib import Path

SOURCE = Path(__file__).with_name("propose_joint_clearance_witnesses_230.py")

spec = importlib.util.spec_from_file_location("proposal230", SOURCE)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {SOURCE}")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Proposal230Tests(unittest.TestCase):
    def test_existing_lexicographic_sparse_clearance_and_geometry_guards(self):
        module.self_test()

    def test_edge_gradient_rows_are_unique_node_major_and_componentwise(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]])
        faces = np.array([[0, 1, 2], [1, 0, 2]], dtype=np.int64)
        A, b, meta = module.build_edge_gradient_constraints(points, faces, np.array([0, 1]), 0.25)
        self.assertEqual(meta["active_edges"], 3)
        self.assertEqual(A.shape, (18, 6))
        feasible = np.array([0., 0., 0., 0.2, 0., 0.])
        violating = np.array([0., 0., 0., 0.3, 0., 0.])
        self.assertGreaterEqual(float(np.min(A @ feasible - b)), -1e-12)
        self.assertLess(float(np.min(A @ violating - b)), -1e-12)
        target = np.array([0., 0., 0., 0.8, 0., 0.])
        projected = module.solve_sparse_l1(A, b, target, -np.ones(6), np.ones(6), 5.0)
        self.assertTrue(projected["success"], projected)
        displacement = projected["delta"].reshape(2, 3)
        self.assertLessEqual(float(np.max(np.abs(displacement[0] - displacement[1]))), 0.25 + 1e-8)

    def test_edge_gradient_rejects_bad_domains_and_cap(self):
        points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.]])
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        for support, cap in ((np.array([0, 0]), .25), (np.array([9]), .25),
                             (np.array([0]), 0.0), (np.array([0]), float("nan"))):
            with self.assertRaises(ValueError):
                module.build_edge_gradient_constraints(points, faces, support, cap)
        degenerate = points.copy()
        degenerate[1] = degenerate[0]
        with self.assertRaises(ValueError):
            module.build_edge_gradient_constraints(degenerate, faces, np.array([0]), .25)


if __name__ == "__main__":
    unittest.main()
                                 