import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
import human_skin_fat_connected_repair as repair


class ReferenceDisplacementTests(unittest.TestCase):
    def fixture(self, extra_node=False):
        X=np.array([[0,0,0],[.01,0,0],[0,.01,0],[0,0,.01]],dtype=np.float32)
        T=np.array([[0,1,2,3]],dtype=np.int64)
        B=np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]],dtype=np.int64)
        if extra_node:
            X=np.vstack([X,np.array([[.002,.002,.002]],dtype=np.float32)])
        return X,T,B

    def lp(self,X,T,B,target,**kw):
        C=sparse.csr_matrix(([1.],([0],[3])),shape=(1,3*len(X)))
        return repair.coupled_clearance_step(X,X,T,np.arange(len(X)),C,[target],
            construction_origin=X,boundary_faces=B,step_limit_mm=2.,
            max_incremental_gradient=None,time_limit_s=5.,**kw)

    def test_envelope_is_cumulative_from_immutable_origin_not_increment(self):
        O,T,B=self.fixture()
        X=(O.astype(np.float64)+np.array([.0004,0,0])).astype(np.float32)
        C=sparse.csr_matrix(([1.],([0],[3])),shape=(1,12))
        target=float(O[1,0])+.0006
        Y,r=repair.coupled_clearance_step(X,O,T,np.arange(4),C,[target],
            construction_origin=O,boundary_faces=B,step_limit_mm=2.,
            max_incremental_gradient=None,time_limit_s=5.,
            max_reference_displacement_m=np.full(4,.0005))
        self.assertTrue(r['step_applied'])
        self.assertTrue(r['reference_displacement_envelope_final_float32_pass'])
        self.assertEqual(r['reference_displacement_envelope_shape'],'axis_aligned_componentwise_box')
        self.assertLessEqual(np.max(np.abs(Y.astype(float)-O)),.0005)
        self.assertGreater(np.max(np.abs(Y.astype(float)-X)),0.)
        # Incremental motion could be 2 mm, but cumulative origin motion is capped at .5 mm.
        self.assertLessEqual(abs(float(Y[1,0])-float(X[1,0])),.00010001)

    def test_anisotropic_node_weights_are_used_by_l1_objective(self):
        X,T,B=self.fixture()
        weights=np.arange(1,13,dtype=float).reshape(4,3)
        C=sparse.csr_matrix(([1.],([0],[3])),shape=(1,12))
        with patch.object(repair,'linprog',wraps=repair.linprog) as wrapped:
            repair.coupled_clearance_step(X,X,T,np.arange(4),C,[.0101],
                construction_origin=X,boundary_faces=B,max_incremental_gradient=None,
                cumulative_displacement_weight=.002,reference_l1_weights=weights,time_limit_s=5.)
        objective=wrapped.call_args.args[0]
        np.testing.assert_array_equal(objective[14:],.002*weights.ravel())

    def test_default_l1_weight_parity(self):
        X,T,B=self.fixture();C=sparse.csr_matrix(([1.],([0],[3])),shape=(1,12))
        a,ra=repair.coupled_clearance_step(X,X,T,np.arange(4),C,[.0102],
            construction_origin=X,boundary_faces=B,max_incremental_gradient=None,time_limit_s=5.)
        b,rb=repair.coupled_clearance_step(X,X,T,np.arange(4),C,[.0102],
            construction_origin=X,boundary_faces=B,max_incremental_gradient=None,time_limit_s=5.,
            reference_l1_weights=np.ones((4,3)))
        np.testing.assert_array_equal(a.view(np.uint32),b.view(np.uint32))
        self.assertEqual(ra['lp_success'],rb['lp_success'])
        self.assertEqual(ra['contact_deficit_l1_mm'],rb['contact_deficit_l1_mm'])

    def test_generated_node_requires_full_node_domain(self):
        X,T,B=self.fixture(extra_node=True)
        C=sparse.csr_matrix(([1.],([0],[12])),shape=(1,15))
        Y,r=repair.coupled_clearance_step(X,X,T,np.arange(5),C,[.002],
            construction_origin=X,boundary_faces=B,max_incremental_gradient=None,
            reference_l1_weights=np.ones(5),max_reference_displacement_m=np.full(5,.001),time_limit_s=5.)
        self.assertTrue(r['reference_displacement_envelope_final_float32_pass'])
        self.assertLessEqual(np.max(np.abs(Y.astype(float)-X)),.001)
        for values,name in [(np.ones(4),'max_reference_displacement_m'),(np.ones(4),'reference_l1_weights')]:
            with self.assertRaisesRegex(ValueError,'generated nodes'):
                self.lp(X,T,B,.002,**{name:values})

    def test_invalid_envelope_and_weight_bounds_fail_closed(self):
        X,T,B=self.fixture();C=sparse.csr_matrix(([1.],([0],[3])),shape=(1,12))
        common=dict(construction_origin=X,boundary_faces=B,max_incremental_gradient=None,time_limit_s=5.)
        for bad in [np.zeros(4),np.full(4,np.nan),np.full((4,2),.001),np.full(4,np.inf)]:
            with self.assertRaises(ValueError):
                repair.coupled_clearance_step(X,X,T,np.arange(4),C,[.0101],**common,max_reference_displacement_m=bad)
        with self.assertRaisesRegex(ValueError,'requires an immutable'):
            repair.coupled_clearance_step(X,X,T,np.arange(4),C,[.0101],boundary_faces=B,
                max_incremental_gradient=None,max_reference_displacement_m=np.full(4,.001))
        with self.assertRaisesRegex(ValueError,'reference_l1_weights requires'):
            repair.coupled_clearance_step(X,X,T,np.arange(4),C,[.0101],boundary_faces=B,
                max_incremental_gradient=None,reference_l1_weights=np.ones(4))
        with self.assertRaisesRegex(ValueError,'outside its immutable-origin'):
            repair.coupled_clearance_step((X+np.float32(.001)).astype(np.float32),X,T,np.arange(4),C,[.0101],
                **common,max_reference_displacement_m=np.full(4,.0005))
        bad_labels=np.array([0,0,-1,0],dtype=np.int64)
        with self.assertRaisesRegex(ValueError,'domain labels'):
            repair.coupled_clearance_step(X,X,T,np.arange(4),C,[.0101],**common,
                max_reference_displacement_m=np.full(4,.001),backtracking_domain_labels=bad_labels)
        for bad in [np.zeros(4),np.full(4,np.nan),np.ones(3),np.full((4,2),1.)]:
            with self.assertRaises(ValueError):
                repair.coupled_clearance_step(X,X,T,np.arange(4),C,[.0101],**common,reference_l1_weights=bad)

    def test_exact_float32_gate_detects_rounding_outside_box(self):
        origin=np.array([[1.,0.,0.]])
        emitted=np.array([[np.nextafter(np.float32(1.),np.float32(np.inf)),0.,0.]],dtype=np.float32)
        cap=np.array([[1.1e-7,1.,1.]])
        ok,slack=repair._reference_envelope_status(emitted,origin,cap)
        self.assertFalse(ok)
        self.assertLess(slack,0.)

    def test_least_squares_backtracking_respects_reference_box(self):
        X,T,B=self.fixture();C=sparse.csr_matrix(([1.],([0],[3])),shape=(1,12))
        Y,r=repair.coupled_clearance_least_squares_step(X,X,T,np.arange(4),C,[.0102],
            construction_origin=X,boundary_faces=B,step_limit_mm=2.,max_incremental_gradient=.3,
            max_reference_displacement_m=np.full(4,.0001),iteration_limit=50)
        self.assertTrue(r['reference_displacement_envelope_final_float32_pass'])
        self.assertLessEqual(np.max(np.abs(Y.astype(float)-X)),.0001)


if __name__=='__main__':
    unittest.main()
