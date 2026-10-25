import unittest
import numpy as np
from scipy import sparse
import human_skin_fat_connected_repair as repair
class SparseConnected(unittest.TestCase):
    def fixture(self):
        X=np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01]],np.float32)
        T=np.array([[0,1,2,3]],np.int64);B=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]],np.int64)
        C=sparse.csr_matrix(([1.],([0],[0])),shape=(1,12))
        return X,T,B,C
    def test_gradient_translation_and_affine(self):
        X,T,B,C=self.fixture();D=repair.sparse_displacement_gradient(X,T,np.arange(4))
        np.testing.assert_allclose(D@np.ones(12),0,atol=1e-14)
        gradient=np.array([[.02,.03,.04],[.05,.06,.07],[.08,.09,.1]])
        delta=1000*X.astype(float)@gradient.T
        np.testing.assert_allclose((D@delta.ravel()).reshape(3,3),gradient,rtol=1e-12,atol=1e-12)
    def test_sparse_direction_preserves_hard_gates(self):
        X,T,B,C=self.fixture()
        Y,r=repair.coupled_clearance_least_squares_step(X,X,T,np.arange(4),C,np.array([.0002]),
            construction_origin=X,boundary_faces=B)
        self.assertTrue(r['step_applied'])
        self.assertLess(r['contact_deficit_l1_mm'],r['starting_contact_deficit_l1_mm'])
        self.assertGreaterEqual(r['emitted_q_min'],1e-5)
        self.assertGreaterEqual(r['boundary_area_ratio_min'],.5)
        self.assertGreaterEqual(r['boundary_normal_cosine_min'],.5)
        self.assertEqual(r['accepted_physical_steps'],0)
    def test_outside_nodes_are_bitwise_anchored(self):
        X,T,B,C=self.fixture()
        Y,r=repair.coupled_clearance_least_squares_step(X,X,T,np.array([0]),C,np.array([.0002]),
            construction_origin=X,boundary_faces=B)
        np.testing.assert_array_equal(Y[1:].view(np.uint32),X[1:].view(np.uint32))
    def test_empty_or_satisfied_work(self):
        X,T,B,C=self.fixture()
        Y,r=repair.coupled_clearance_least_squares_step(X,X,T,np.arange(4),C,np.array([-.001]),
            construction_origin=X,boundary_faces=B)
        self.assertFalse(r['step_applied'])
        np.testing.assert_array_equal(Y.view(np.uint32),X.view(np.uint32))
        with self.assertRaises(ValueError):
            repair.coupled_clearance_least_squares_step(X,X,T,np.arange(4),sparse.csr_matrix((0,12)),np.zeros(0),
                construction_origin=X,boundary_faces=B)
    def test_invalid_bound_rejected(self):
        X,T,B,C=self.fixture()
        for bad in [float('nan'),-1,0]:
            with self.assertRaises(ValueError):
                repair.coupled_clearance_least_squares_step(X,X,T,np.arange(4),C,np.array([.001]),
                    construction_origin=X,boundary_faces=B,gradient_penalty_mm=bad)
    def test_active_set_enforces_connected_local_bounds(self):
        X,T,B,C=self.fixture()
        Y,r=repair.coupled_clearance_least_squares_step(X,X,T,np.arange(4),C,np.array([.004]),
            construction_origin=X,boundary_faces=B,active_set_passes=4,max_incremental_gradient=.15)
        self.assertTrue(r['step_applied'])
        self.assertTrue(r['active_set_history'])
        for row in r['active_set_history']:
            self.assertLessEqual(row['merit_after'],row['merit_before']+1e-8)
        self.assertLessEqual(np.abs(Y.astype(float)-X).max()*1000,2.0001)
        self.assertGreaterEqual(r['emitted_J_min'],.35)
        self.assertLess(r['contact_deficit_l1_mm'],4.)
    def test_invalid_method_and_active_budget(self):
        X,T,B,C=self.fixture()
        with self.assertRaises(ValueError):
            repair.coupled_clearance_step(X,X,T,np.arange(4),C,np.array([.001]),linear_method='unknown')
        for budget in [-1,13,True,1.5]:
            with self.assertRaises(ValueError):
                repair.coupled_clearance_least_squares_step(X,X,T,np.arange(4),C,np.array([.001]),
                    construction_origin=X,boundary_faces=B,active_set_passes=budget)

if __name__=='__main__':unittest.main()

