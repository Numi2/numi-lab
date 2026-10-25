import unittest
import numpy as np
from scipy import sparse
import human_skin_fat_connected_repair as repair

class ConnectedSurface(unittest.TestCase):
    def fixture(self):
        X=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)*.01
        T=np.array([[0,1,2,3]],np.int64)
        B=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]],np.int64)
        return X,T,B
    def test_boundary_gradient(self):
        X,T,B=self.fixture()
        Y=X.astype(float)+np.array([[.0001,0,0],[0,.0002,0],[0,0,.0001],[.0001,.0001,0]])
        v,G,a,c=repair.boundary_face_values_gradients(Y,X,B)
        for node in range(4):
            for k in range(3):
                h=1e-8
                plus=Y.copy();minus=Y.copy();plus[node,k]+=h;minus[node,k]-=h
                fd=(repair.boundary_face_values_gradients(plus,X,B)[0]-repair.boundary_face_values_gradients(minus,X,B)[0])/(2*h)
                exact=np.zeros(len(B))
                for row,face in enumerate(B):
                    if node in face:exact[row]=G[row,list(face).index(node),k]
                np.testing.assert_allclose(fd,exact,rtol=1e-7,atol=1e-7)
    def test_rejects_invalid_faces(self):
        X,T,B=self.fixture()
        for bad in [np.array([[0.,1.,2.]]),np.array([[0,0,2]]),np.array([[0,1,99]])]:
            with self.assertRaises(ValueError):repair.boundary_face_values_gradients(X,X,bad)
        with self.assertRaises(ValueError):repair.boundary_face_values_gradients(X,X,np.empty((0,3),int))
    def test_cumulative_anchor_and_surface(self):
        O,T,B=self.fixture();X=(O+np.float32(.0004)).astype(np.float32)
        # The contact requires only node 1's x coordinate to remain at least
        # 0.1 mm beyond its origin, not to preserve a pre-existing global drift.
        C=sparse.csr_matrix(([1.],([0],[3])),shape=(1,12))
        Y,r=repair.coupled_clearance_step(X,O,T,np.arange(4),C,np.array([float(O[1,0])+.0005]),
            construction_origin=O,boundary_faces=B,cumulative_displacement_weight=.05,
            step_limit_mm=1.,max_incremental_gradient=.2)
        self.assertTrue(r['step_applied'])
        self.assertLess(r['contact_deficit_l1_mm'],1e-5)
        self.assertLess(r['cumulative_displacement_l1_mm'],np.abs(X.astype(float)-O).sum()*1000)
        self.assertGreaterEqual(r['boundary_normal_cosine_min'],.25)
        self.assertGreaterEqual(r['boundary_area_ratio_min'],.25)
    def test_rejects_origin_free_surface_gate(self):
        X,T,B=self.fixture();C=sparse.csr_matrix(([1.],([0],[0])),shape=(1,12))
        with self.assertRaises(ValueError):
            repair.coupled_clearance_step(X,X,T,np.arange(4),C,np.array([.001]),boundary_faces=B)
    def test_rejects_positive_volume_rotated_surface(self):
        O,T,B=self.fixture();X=O.copy();X[:,[0,1]]=-X[:,[0,1]]
        self.assertGreater(repair.cell_values_gradients(X,O,T)[0].min(),.99)
        C=sparse.csr_matrix(([1.],([0],[0])),shape=(1,12))
        with self.assertRaises(ValueError):
            repair.coupled_clearance_step(X,O,T,np.arange(4),C,np.array([.001]),construction_origin=O,boundary_faces=B)
    def test_anchor_does_not_change_legacy_default(self):
        X,T,B=self.fixture();C=sparse.csr_matrix(([1.],([0],[0])),shape=(1,12))
        a,ra=repair.coupled_clearance_step(X,X,T,np.arange(4),C,np.array([.0002]))
        b,rb=repair.coupled_clearance_step(X,X,T,np.arange(4),C,np.array([.0002]),construction_origin=X)
        np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
if __name__=='__main__':unittest.main()
