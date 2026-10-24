import importlib.util
from pathlib import Path
import unittest
import numpy as np

SCRIPT = Path(__file__).with_name('refine_internal_o_edges_403.py')
spec = importlib.util.spec_from_file_location('edge_support_403_test_module', SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

class InternalOEdgeSupportTests(unittest.TestCase):
    def setUp(self):
        # Three-tet ring around the interior edge AB. A and B are physical O
        # nodes, but AB is not an edge of any exterior shell face.
        pts = np.asarray([
            [0,0,0], [0,0,1],
            [1,0,.5], [-.5,.8660254,.5], [-.5,-.8660254,.5],
        ], dtype=np.float32)
        tets = np.asarray([[0,1,2,3],[0,1,3,4],[0,1,4,2]], dtype=np.int64)
        self.points = pts
        self.tets = np.asarray([module.orient_positive(pts, t) for t in tets], dtype=np.int64)
        self.shell = module.oriented_boundary_faces(self.tets)
        self.outer = np.asarray([0,1], dtype=np.int64)
        self.edges, _ = module.find_internal_outer_edges(self.tets, self.outer, self.shell)

    def test_detects_internal_o_edge_and_splits_conformingly(self):
        self.assertEqual(self.edges, [(0,1)])
        out_x, out_t, parent, edge_rows = module.split_edges_globally(self.points, self.tets, self.edges)
        self.assertEqual(len(out_x), len(self.points)+1)
        self.assertEqual(len(out_t), len(self.tets)+3)
        self.assertEqual(parent.shape, (len(out_t),))
        np.testing.assert_array_equal(edge_rows, np.asarray(self.edges))
        np.testing.assert_array_equal(out_x[:len(self.points)], self.points)
        self.assertFalse(any(0 in tet and 1 in tet for tet in out_t))
        self.assertTrue(np.all(module.tetra_metrics(out_x, out_t)[0] > 0))
        parity = module.compare_oriented_boundary(self.shell, module.oriented_boundary_faces(out_t))
        self.assertTrue(parity['exact_unoriented_set'])
        self.assertEqual(parity['oriented_mismatch_count'], 0)
        self.assertEqual(sorted(np.bincount(parent, minlength=len(self.tets)).tolist()), [2,2,2])

    def test_parent_fractions_are_conservative_and_raw_drift_is_separate(self):
        vols = np.asarray([.4,.6000000000002,2.,3.])
        parents = np.asarray([0,0,1,1])
        frac, totals, closure = module.conservative_child_fractions(vols, parents, 2)
        np.testing.assert_allclose(closure, [1,1], atol=2e-15, rtol=0)
        np.testing.assert_allclose(frac, [.4/1.0000000000002,.6000000000002/1.0000000000002,.4,.6], atol=2e-15, rtol=0)
        np.testing.assert_allclose(totals, [1.0000000000002,5.], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            module.conservative_child_fractions(np.asarray([1.,0.]), np.asarray([0,1]), 2)

    def test_midpoint_source_ancestry_is_endpoint_average_and_prefix_exact(self):
        ptr = np.asarray([0,1,2,3], dtype=np.int64)
        ids = np.asarray([10,11,12], dtype=np.int64)
        weights = np.ones(3, dtype=np.float64)
        edges = np.asarray([[0,1],[1,2]], dtype=np.int64)
        p2, i2, w2, diag = module.append_midpoint_source_csr(ptr, ids, weights, edges)
        np.testing.assert_array_equal(p2[:len(ptr)], ptr)
        np.testing.assert_array_equal(i2[:len(ids)], ids)
        np.testing.assert_array_equal(w2[:len(weights)], weights)
        self.assertEqual(p2.tolist(), [0,1,2,3,5,7])
        self.assertEqual(i2.tolist(), [10,11,12,10,11,11,12])
        np.testing.assert_allclose(w2[3:], [.5,.5,.5,.5])
        self.assertEqual(diag['new_rows'], 2)
        self.assertEqual(diag['max_abs_weight_sum_error'], 0.)


    def test_stellar_split_removes_internal_all_o_face_without_moving_boundary(self):
        pts=np.asarray([[0,0,0],[1,0,0],[0,1,0],[.2,.2,1],[.2,.2,-1]],dtype=np.float32)
        raw=np.asarray([[0,1,2,3],[0,2,1,4]],dtype=np.int64)
        tets=np.asarray([module.orient_positive(pts,t) for t in raw],dtype=np.int64)
        shell=module.oriented_boundary_faces(tets)
        o=np.asarray([0,1,2],dtype=np.int64)
        faces,summary=module.find_internal_all_o_faces(tets,o)
        self.assertEqual(faces,[(0,1,2)])
        self.assertEqual(summary['internal_all_o_face_count'],1)
        x2,t2,parent,centers=module.split_internal_o_faces_globally(pts,tets,np.arange(len(tets)),faces)
        self.assertEqual(len(centers),1)
        self.assertEqual(len(t2),6)
        self.assertEqual(sorted(np.bincount(parent).tolist()),[3,3])
        self.assertFalse(any(set(face).issubset(set(tet)) for tet in t2 for face in faces))
        parity=module.compare_oriented_boundary(shell,module.oriented_boundary_faces(t2))
        self.assertTrue(parity['exact_unoriented_set'])
        self.assertEqual(parity['oriented_mismatch_count'],0)
        support=module.internal_o_edge_and_cell_counts(t2,o,shell)
        self.assertEqual(support['remaining_internal_all_o_face_count'],0)
        self.assertEqual(support['remaining_all_o_tet_count'],0)
        ptr=np.arange(6,dtype=np.int64)
        ids=np.arange(10,15,dtype=np.int64)
        weights=np.ones(5,dtype=np.float64)
        p3,i3,w3,_=module.append_weighted_source_csr(ptr,ids,weights,np.asarray(faces),np.full((1,3),1/3))
        self.assertEqual(p3[-1]-p3[-2],3)
        np.testing.assert_array_equal(i3[-3:],[10,11,12])
        np.testing.assert_allclose(w3[-3:],[1/3]*3)


    def test_stellar_split_removes_internal_all_o_face_and_keeps_boundary(self):
        pts=np.asarray([[0,0,0],[1,0,0],[0,1,0],[.2,.2,1],[.2,.2,-1]],dtype=np.float32)
        raw=np.asarray([[0,1,2,3],[0,2,1,4]],dtype=np.int64)
        tets=np.asarray([module.orient_positive(pts,t) for t in raw],dtype=np.int64)
        shell=module.oriented_boundary_faces(tets)
        outer=np.asarray([0,1,2],dtype=np.int64)
        faces,summary=module.find_internal_all_o_faces(tets,outer)
        self.assertEqual(faces,[(0,1,2)])
        self.assertEqual(summary['internal_all_o_face_count'],1)
        out_x,out_t,parent,centroids=module.split_internal_o_faces_globally(pts,tets,np.arange(2),faces)
        self.assertEqual(len(centroids),1)
        self.assertEqual(len(out_t),6)
        np.testing.assert_array_equal(np.bincount(parent),[3,3])
        self.assertFalse(any(all(v in tet for v in faces[0]) for tet in out_t))
        parity=module.compare_oriented_boundary(shell,module.oriented_boundary_faces(out_t))
        self.assertTrue(parity['exact_unoriented_set'])
        self.assertEqual(parity['oriented_mismatch_count'],0)
        support=module.internal_o_edge_and_cell_counts(out_t,outer,shell)
        self.assertEqual(support['remaining_internal_all_o_face_count'],0)
        self.assertEqual(support['remaining_all_o_tet_count'],0)
        ptr=np.arange(6,dtype=np.int64); ids=np.arange(10,15,dtype=np.int64); w=np.ones(5)
        p2,i2,w2,_=module.append_weighted_source_csr(ptr,ids,w,np.asarray(faces),np.full((1,3),1/3))
        self.assertEqual(p2[-1]-p2[-2],3)
        np.testing.assert_array_equal(i2[-3:],[10,11,12])
        np.testing.assert_allclose(w2[-3:],[1/3]*3)

    def test_rejects_shared_faces_with_equal_orientation(self):
        # Duplicate cell and a distinct cell on the same side both have
        # count-two shared faces; neither is an admissible interior interface.
        with self.assertRaisesRegex(ValueError, 'equal orientation'):
            module.oriented_boundary_faces(np.asarray([[0,1,2,3],[0,1,2,3]]))
        with self.assertRaisesRegex(ValueError, 'equal orientation'):
            module.oriented_boundary_faces(np.asarray([[0,1,2,3],[0,1,2,4]]))

    def test_accepts_opposite_internal_face_orientations(self):
        faces = module.oriented_boundary_faces(np.asarray([[0,1,2,3],[0,2,1,4]]))
        self.assertEqual(faces.shape, (6,3))

    def test_rejects_missing_marked_edges(self):
        with self.assertRaisesRegex(ValueError, 'absent'):
            module.split_edges_globally(self.points, self.tets, [(0,5)])

if __name__ == '__main__':
    unittest.main()
