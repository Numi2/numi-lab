import unittest
import numpy as np
from scipy import sparse
from human_skin_fat_embedded_surface import MappedSkinSurface
from human_skin_fat_connected_repair import coupled_clearance_step, coupled_clearance_least_squares_step

class MappedSkinTests(unittest.TestCase):
    def setUp(self):
        self.X=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
        self.T=np.array([[0,1,2,3]])
        self.B=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
        self.D=np.array([[0,0,0],[.125,0,0],[0,.125,0],[.25,.25,.125]],dtype=np.float32)
        self.W=sparse.csr_matrix(np.array([[1.,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]]))
        self.F=np.array([[0,1,2],[1,3,2]])

    def test_sparse_chain_rule_and_inactive_vertex(self):
        surf=MappedSkinSurface(self.X,self.D,self.F,self.W)
        _,J=surf.projected_area_linearization(self.X,np.arange(4))
        step=2.**-12
        for i in range(12):
            a=self.X.copy();b=self.X.copy();a.flat[i]+=step;b.flat[i]-=step
            numerical=(surf.metrics(a)[0]-surf.metrics(b)[0])/(2*step)
            np.testing.assert_allclose(J[:,i].toarray().ravel(),numerical,rtol=1e-7,atol=1e-7)
        Y=self.X+np.float32(.25)
        np.testing.assert_array_equal(surf.positions(Y)[3],self.D[3])
        np.testing.assert_allclose(surf.positions(Y)[:3],self.D[:3]+.25)

    def test_mapped_true_area_and_cosine_random_direction_chain_rule(self):
        surf=MappedSkinSurface(self.X,self.D,self.F,self.W)
        free=np.array([2,1],dtype=np.int64)
        ids,area,area_j,cosine,cosine_j=surf.area_cosine_linearization(self.X,free)
        self.assertEqual(ids.tolist(),[0,1])
        rng=np.random.default_rng(812)
        direction=rng.normal(size=(len(free),3))
        direction/=np.linalg.norm(direction)
        delta=direction.reshape(-1)
        eps=2e-3
        plus=self.X.copy();minus=self.X.copy()
        plus[free]=(plus[free].astype(float)+eps*direction).astype(np.float32)
        minus[free]=(minus[free].astype(float)-eps*direction).astype(np.float32)
        ap,cp=surf.metrics(plus)[2:]
        am,cm=surf.metrics(minus)[2:]
        numeric_area=(ap-am)/(2*eps)
        numeric_cos=(cp-cm)/(2*eps)
        np.testing.assert_allclose(area_j@delta,numeric_area,rtol=2e-3,atol=2e-3)
        np.testing.assert_allclose(cosine_j@delta,numeric_cos,rtol=2e-3,atol=2e-3)

    def test_linearization_filters_faces_without_free_support(self):
        faces=np.vstack((self.F,np.array([[0,1,3]])))
        surf=MappedSkinSurface(self.X,self.D,faces,self.W)
        free=np.array([2],dtype=np.int64)
        ids=surf.linearization_face_ids(free)
        self.assertEqual(ids.tolist(),[0,1])
        _,area_j=surf.projected_area_linearization(self.X,free,face_ids=ids)
        self.assertEqual(area_j.shape,(2,3))
        all_value,all_j=surf.projected_area_linearization(self.X,free)
        self.assertEqual(all_value.shape,(3,))
        self.assertEqual(all_j.shape,(3,3))

    def test_free_inner_node_with_no_mapped_face_support(self):
        surf=MappedSkinSurface(self.X,self.D,self.F,self.W)
        free=np.array([3],dtype=np.int64)
        ids=surf.linearization_face_ids(free)
        self.assertEqual(ids.tolist(),[])
        projected,pj=surf.projected_area_linearization(self.X,free,face_ids=ids)
        ids2,area,aj,cosine,cj=surf.area_cosine_linearization(self.X,free,face_ids=ids)
        self.assertEqual(projected.shape,(0,))
        self.assertEqual(pj.shape,(0,3))
        self.assertEqual(ids2.tolist(),[])
        self.assertEqual(area.shape,(0,));self.assertEqual(cosine.shape,(0,))
        self.assertEqual(aj.shape,(0,3));self.assertEqual(cj.shape,(0,3))
        C=sparse.csr_matrix(([1.],([0],[11])),shape=(1,12))
        out,report=coupled_clearance_step(
            self.X,self.X,self.T,free,C,[1.1],construction_origin=self.X,
            step_limit_mm=100.,max_incremental_gradient=.5,
            cumulative_displacement_weight=.01,time_limit_s=10.,
            embedded_surface=surf)
        self.assertTrue(report['lp_success'],report)
        self.assertEqual(report['mapped_skin_linearization_face_count'],0)
        self.assertEqual(report['mapped_skin_linearization_face_ids'],[])
        self.assertTrue(surf.passes(out))
        self.assertGreater(out[3,2],self.X[3,2])

    def test_positive_coarse_cell_can_fold_detailed_skin(self):
        surf=MappedSkinSurface(self.X,self.D,self.F[:1],self.W)
        Y=self.X.copy();Y[1,0]=.75
        self.assertGreater(np.linalg.det((Y[1:]-Y[0]).T),0)
        self.assertFalse(surf.passes(Y))
        self.assertEqual(surf.report(Y)['mapped_skin_nonpositive_normal_faces'],1)

    def test_mapped_cosine_rows_limit_lp_direction(self):
        X=np.array([[0,0,0],[1,0,0],[0,1,.3],[0,0,1]],dtype=np.float32)
        T=np.array([[0,1,2,3]],dtype=np.int64)
        detail=X[:3].copy()
        W=sparse.csr_matrix((np.ones(3), (np.arange(3),np.arange(3))),shape=(3,4))
        surf=MappedSkinSurface(X,detail,np.array([[0,1,2]],dtype=np.int64),W,
                               area_bounds=(.5,2.),normal_cosine_min=.9)
        C=sparse.csr_matrix(([1.],([0],[8])),shape=(1,12))
        out,report=coupled_clearance_step(
            X,X,T,np.arange(4),C,[.8],construction_origin=X,
            step_limit_mm=1200.,q_min=1e-5,J_bounds=(.05,5.),
            time_limit_s=10.,max_incremental_gradient=2.,
            cumulative_displacement_weight=.01,embedded_surface=surf)
        self.assertTrue(report['lp_success'],report)
        self.assertTrue(report['step_applied'],report)
        self.assertEqual(report['mapped_skin_linearization_face_ids'],[0])
        self.assertGreater(out[2,2],X[2,2])
        self.assertGreaterEqual(report['lp_linearized_mapped_skin_normal_cosine_min'],.9-1e-8)
        self.assertGreaterEqual(report['lp_linearized_mapped_skin_area_ratio_min'],.5-1e-8)
        self.assertTrue(surf.passes(out))
        self.assertGreaterEqual(surf.report(out)['mapped_skin_normal_cosine_min'],.9)
        ls_out,ls_report=coupled_clearance_least_squares_step(
            X,X,T,np.arange(4),C,[.8],construction_origin=X,
            boundary_faces=np.array([[0,1,3]],dtype=np.int64),
            boundary_area_bounds=(.5,2.),boundary_normal_cosine_min=.5,
            step_limit_mm=1200.,q_min=1e-5,J_bounds=(.05,5.),
            cumulative_displacement_weight=.01,gradient_penalty_mm=.001,
            max_incremental_gradient=2.,iteration_limit=100,
            active_set_passes=1,active_set_penalty=1000.,embedded_surface=surf)
        self.assertTrue(ls_report['step_applied'],ls_report)
        self.assertEqual(ls_report['mapped_skin_linearization_face_ids'],[0])
        self.assertTrue(surf.passes(ls_out))

    def test_reject_invalid_map_and_geometry(self):
        for W in (self.W*.5,-self.W,sparse.eye(4)):
            if W.shape==(4,4) and (W!=sparse.eye(4)).nnz==0:continue
            with self.assertRaises(ValueError):MappedSkinSurface(self.X,self.D,self.F,W)
        with self.assertRaises(ValueError):MappedSkinSurface(self.X,self.D.astype(float),self.F,self.W)
        with self.assertRaises(ValueError):MappedSkinSurface(self.X,self.D,np.array([[0,0,1]]),self.W)
        with self.assertRaises(ValueError):MappedSkinSurface(self.X,self.D,np.array([[0,1,2]]),sparse.csr_matrix((3,4)))

    def test_connected_solver_gates_detail_without_repairing_render_output(self):
        surf=MappedSkinSurface(self.X,self.D,self.F[:1],self.W)
        C=sparse.csr_matrix(([-1.],([0],[3])),shape=(1,12))
        args=dict(construction_origin=self.X,boundary_faces=self.B,step_limit_mm=250.,
                  max_incremental_gradient=.5,gradient_penalty_mm=.001,
                  cumulative_displacement_weight=1e-6,iteration_limit=100)
        plain,r=coupled_clearance_least_squares_step(self.X,self.X,self.T,np.arange(4),C,[-.75],**args)
        self.assertTrue(r['step_applied'])
        self.assertFalse(surf.passes(plain))
        safe,r=coupled_clearance_least_squares_step(self.X,self.X,self.T,np.arange(4),C,[-.75],embedded_surface=surf,**args)
        self.assertTrue(r['step_applied']);self.assertTrue(surf.passes(safe))
        self.assertTrue(r['mapped_skin_geometry_pass'])
        self.assertFalse(r['mapped_skin_native_shader_parity_qualified'])
        self.assertIn('mapped_skin_linearization_face_count',r)
        self.assertEqual(r['mapped_skin_linearization_face_count'],0)
        # The only returned repair is the mechanical mesh. Detailed coordinates
        # remain a deterministic evaluation of that very same coarse state.
        self.assertGreater(safe[1,0],plain[1,0])
        with self.assertRaisesRegex(ValueError,'mapped detailed-skin'):
            coupled_clearance_least_squares_step(plain,self.X,self.T,np.arange(4),C,[-.7],embedded_surface=surf,**args)

    def test_detail_face_couples_disconnected_volume_components(self):
        X=np.vstack((self.X,self.X+np.array([2.,0,0],np.float32)))
        T=np.vstack((self.T,self.T+4));B=np.vstack((self.B,self.B+4))
        W=sparse.csr_matrix(([1.,1.,1.],([0,1,2],[0,4,6])),shape=(3,8))
        surf=MappedSkinSurface(X,X[[0,4,6]],np.array([[0,1,2]]),W)
        C=sparse.csr_matrix(([1.,1.],([0,1],[0,12])),shape=(2,24))
        kwargs=dict(construction_origin=X,boundary_faces=B,step_limit_mm=20.)
        _,plain=coupled_clearance_least_squares_step(X,X,T,np.arange(8),C,[.01,2.01],**kwargs)
        _,mapped=coupled_clearance_least_squares_step(X,X,T,np.arange(8),C,[.01,2.01],embedded_surface=surf,**kwargs)
        self.assertEqual(len(plain['components']),2)
        self.assertEqual(len(mapped['components']),1)

if __name__=='__main__':unittest.main()
