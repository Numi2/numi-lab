import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import refine_conforming_017 as r

class RefinementTests(unittest.TestCase):
    def test_single_triangle_red_split_area_orientation_and_barycentric(self):
        X=np.array([[0,0,0],[2,0,0],[0,2,0]],dtype=np.float32)
        F=np.array([[0,1,2]],dtype=np.int32)
        q=r.build_refinement(X,F,np.array([0]))
        self.assertEqual(q['points_world_m'].shape,(7,3))
        self.assertEqual(q['source_O_faces'].shape,(6,3))
        self.assertEqual(q['selected_edges'].shape,(3,2))
        self.assertEqual(q['affected_parent_face_rows'].tolist(),[0])
        self.assertTrue(np.all(q['parent_face_rows']==0))
        self.assertTrue(np.all(q['parent_face_barycentric']>=0))
        self.assertTrue(np.allclose(q['parent_face_barycentric'].sum(axis=2),1))
        self.assertLess(np.max(np.abs(q['new_face_area_sum_m2']-q['old_face_area_m2'])),1e-12)
        self.assertTrue(np.all(q['refinement_node_parent_edge_weights'][3:6]==0.5))
        self.assertTrue(np.all(q['refinement_node_parent_edge_weights'][6]==-1))
        for face,bary in zip(q['source_O_faces'],q['parent_face_barycentric']):
            reconstructed=bary@X[F[0]].astype(np.float64)
            actual=q['points_world_m'][face].astype(np.float64)
            self.assertTrue(np.allclose(reconstructed,actual,atol=1e-7))

    def test_shared_edge_midpoint_and_conforming_neighbor_fan(self):
        X=np.array([[0,0,0],[2,0,0],[0,2,0],[2,-2,0]],dtype=np.float32)
        F=np.array([[0,1,2],[1,0,3]],dtype=np.int32)
        q=r.build_refinement(X,F,np.array([0]))
        edge_to_mid={tuple(e):int(m) for e,m in zip(q['selected_edges'],q['edge_midpoint_rows'])}
        m=edge_to_mid[(0,1)]
        self.assertEqual(len(q['affected_parent_face_rows']),2)
        inc=r._edge_incidence(q['source_O_faces'])
        self.assertEqual(len(inc[r._edge_key(0,m)]),2)
        self.assertEqual(len(inc[r._edge_key(m,1)]),2)
        self.assertEqual(np.count_nonzero(q['parent_face_rows']==0),6)
        self.assertEqual(np.count_nonzero(q['parent_face_rows']==1),4)
        self.assertEqual(q['boundary_components_before'],q['boundary_components_after'])

    def test_two_selected_edges_in_neighbor_use_five_triangle_fan(self):
        X=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[2,1,0]],dtype=np.float32)
        F=np.array([[0,1,2],[2,1,3],[1,4,3]],dtype=np.int32)
        q=r.build_refinement(X,F,np.array([0,2]))
        counts=np.bincount(q['parent_face_rows'],minlength=len(F))
        self.assertEqual(counts.tolist(),[6,5,6])
        self.assertEqual(q['boundary_components_before'],q['boundary_components_after'])
        self.assertTrue(np.allclose(q['new_face_area_sum_m2'],q['old_face_area_m2'],rtol=0,atol=1e-12))

    def test_extension_preserves_typed_prefix_and_remaps_face_ancestry(self):
        X=np.array([[0,0,0],[2,0,0],[0,2,0],[2,-2,0]],dtype=np.float32)
        F=np.array([[0,1,2],[1,0,3]],dtype=np.int32)
        src={'points_world_m':X,'source_O_faces':F,'node_kind':np.array([0,1,2,5],np.int8),
             'node_source_row':np.array([10,11,12,-1],np.int32),'node_parent_coarse_edge_parameter':np.array([-1.,.2,.3,-1.]),
             'face_kind':np.array([1,2],np.int8),'parent_coarse_face_row':np.array([7,-1],np.int32),
             'face_source_ancestry_indptr':np.array([0,2,3],np.int64),'face_source_ancestry_face_rows':np.array([70,71,80],np.int32)}
        q=r.build_refinement(X,F,np.array([0])); out=r.extend_candidate_arrays(src,q,'a'*64,'b'*64)
        self.assertTrue(np.array_equal(out['node_kind'][:4],src['node_kind']))
        self.assertTrue(np.all(out['node_kind'][4:]==-1))
        self.assertTrue(np.array_equal(out['parent_coarse_face_row'],src['parent_coarse_face_row'][q['parent_face_rows']]))
        self.assertEqual(len(out['face_source_ancestry_indptr']),len(out['source_O_faces'])+1)
        for row,p in enumerate(q['parent_face_rows']):
            old=src['face_source_ancestry_face_rows'][src['face_source_ancestry_indptr'][p]:src['face_source_ancestry_indptr'][p+1]]
            got=out['face_source_ancestry_face_rows'][out['face_source_ancestry_indptr'][row]:out['face_source_ancestry_indptr'][row+1]]
            self.assertTrue(np.array_equal(old,got))
        self.assertEqual(len(out['source_prolongation_indptr']),len(out['points_world_m'])+1)
        self.assertEqual(out['source_prolongation_indptr'][-1],len(out['source_prolongation_node_rows']))

    def test_midpoint_float32_collapse_fails_closed(self):
        a=np.float32(1.0); b=np.nextafter(a,np.float32(np.inf))
        X=np.array([[a,0,0],[b,0,0],[1,1,0]],dtype=np.float32)
        F=np.array([[0,1,2]],dtype=np.int32)
        with self.assertRaisesRegex(ValueError,'midpoint collapses'):
            r.build_refinement(X,F,np.array([0]))

    def test_nonmanifold_or_invalid_crossing_fails_closed(self):
        X=np.array([[0,0,0],[1,0,0],[0,1,0],[0,-1,0],[0,0,1]],dtype=np.float32)
        F=np.array([[0,1,2],[1,0,3],[0,1,4]],dtype=np.int32)
        with self.assertRaisesRegex(ValueError,'nonmanifold'):
            r.build_refinement(X,F,np.array([0]))
        with self.assertRaisesRegex(ValueError,'invalid crossed'):
            r.build_refinement(X[:3],np.array([[0,1,2]],np.int32),np.array([2]))

    def test_exact_self_predicate_package_loader_fixture(self):
        X=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=np.float32)
        F=np.array([[0,1,2]],dtype=np.int32)
        report=r.exact_self_audit(X,F)
        self.assertTrue(report['complete_surface_audit'])
        self.assertEqual(len(report['triangle_pairs']),0)

    def test_crossing_report_requires_complete_diagnostic(self):
        report={'complete':True,'geometry_or_native_admission':False,'banks':[{'result':{'count':1,'triangle_pairs':[[0,8]]}}]}
        rows,n=r.crossing_face_rows(report,2)
        self.assertEqual(rows.tolist(),[0]);self.assertEqual(n,1)
        report['complete']=False
        with self.assertRaisesRegex(ValueError,'complete'):
            r.crossing_face_rows(report,2)

if __name__=='__main__':unittest.main(verbosity=2)
