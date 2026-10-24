import unittest
from itertools import combinations
import numpy as np
from human_skin_p1_material_volume import refine_interior_support

def positive(x,t):
    t=np.array(t,dtype=np.int64)
    for z in t:
        a=x[z].astype(float)
        if np.linalg.det((a[1:]-a[0]).T)<0:z[2],z[3]=z[3],z[2]
    return t

class InteriorSupportTests(unittest.TestCase):
    def bipyramid(self):
        x=np.array([[0,0,-1],[0,0,1],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]],np.float32)
        t=positive(x,[[0,1,2,3],[0,1,3,4],[0,1,4,5],[0,1,5,2]])
        return x,t

    def check_ancestry(self,x,result):
        a=result
        np.testing.assert_array_equal(a['points'][:len(x)],x)
        self.assertEqual(len(a['node_source_indptr']),len(a['points'])+1)
        for i in range(len(a['points'])):
            lo,hi=a['node_source_indptr'][i:i+2]
            ids=a['node_source_indices'][lo:hi];w=a['node_source_weights'][lo:hi]
            self.assertAlmostEqual(float(w.sum()),1.)
            np.testing.assert_allclose(w@x[ids].astype(float),a['points'][i],atol=1e-7)
        self.assertFalse(a['report']['physical_admission'])
        self.assertGreater(a['report']['minimum_float32_quality'],0.)
        self.assertLess(a['report']['maximum_source_cell_relative_volume_change'],1e-6)

    def test_all_outer_node_interior_cell_gains_free_support(self):
        x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
        r=refine_interior_support(x,np.array([[0,1,2,3]]),selected_tetrahedra=[0])
        self.assertEqual(r['report']['added_cell_nodes'],1)
        self.assertEqual(len(r['tetrahedra']),4)
        self.check_ancestry(x,r)

    def test_shared_interior_face_is_subdivided_consistently(self):
        x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,-1]],np.float32)
        t=positive(x,[[0,1,2,3],[0,1,2,4]])
        r=refine_interior_support(x,t,selected_faces=[[2,0,1]])
        self.assertEqual(r['report']['added_face_nodes'],1)
        self.assertEqual(r['report']['added_cell_nodes'],2)
        self.assertEqual(len(r['tetrahedra']),12)
        self.check_ancestry(x,r)

    def test_internal_edge_fan_has_no_hanging_faces(self):
        x,t=self.bipyramid()
        r=refine_interior_support(x,t,selected_edges=[[1,0]])
        self.assertEqual(r['report']['added_edge_nodes'],1)
        self.assertEqual(r['report']['added_face_nodes'],4)
        self.assertEqual(r['report']['added_cell_nodes'],4)
        self.assertEqual(len(r['tetrahedra']),40)
        self.check_ancestry(x,r)
        self.assertGreater(r['report']['minimum_float32_quality'],1e-5)

    def test_unselected_cells_are_exact_and_prefix_preserved(self):
        x,t=self.bipyramid()
        r=refine_interior_support(x,t,selected_tetrahedra=[0])
        for row in range(1,4):
            np.testing.assert_array_equal(r['tetrahedra'][r['source_tetrahedron_rows']==row],t[row:row+1])
        self.check_ancestry(x,r)

    def test_empty_selection_is_identity(self):
        x,t=self.bipyramid();r=refine_interior_support(x,t)
        np.testing.assert_array_equal(r['points'],x)
        np.testing.assert_array_equal(r['tetrahedra'],t)
        np.testing.assert_array_equal(r['source_tetrahedron_rows'],np.arange(4))
        self.check_ancestry(x,r)

    def test_physical_boundary_edge_and_face_are_rejected(self):
        x,t=self.bipyramid()
        for kw in [dict(selected_edges=[[0,2]]),dict(selected_faces=[[0,2,3]])]:
            with self.subTest(kw=kw),self.assertRaisesRegex(ValueError,'boundary support'):
                refine_interior_support(x,t,**kw)

    def test_order_duplicate_and_orientation_independent_selection(self):
        x,t=self.bipyramid()
        a=refine_interior_support(x,t,selected_edges=[[0,1]],selected_faces=[[0,1,3]])
        b=refine_interior_support(x,t,selected_edges=[[1,0],[0,1]],selected_faces=[[3,1,0],[0,3,1]])
        for k in ['points','tetrahedra','source_tetrahedron_rows','node_source_indptr','node_source_indices','node_source_weights']:
            np.testing.assert_array_equal(a[k],b[k])

    def test_float32_failure_retains_exact_parent_and_child_witness(self):
        from human_skin_p1_material_volume import InteriorSupportRefinementError
        # An interior centroid along a one-ULP source dimension rounds onto
        # a source face: source is positive, but the attempted split is not.
        x=np.array([[1,0,0],[np.nextafter(np.float32(1),np.float32(2)),0,0],
                    [1,1,0],[1,0,1]],np.float32)
        with self.assertRaises(InteriorSupportRefinementError) as caught:
            refine_interior_support(x,np.array([[0,1,2,3]]),selected_tetrahedra=[0])
        d=caught.exception.diagnostics
        self.assertFalse(d['physical_admission'])
        self.assertGreater(len(d['failed_child_rows']),0)
        self.assertEqual(set(d['failed_source_tetrahedron_rows']),{0})
        self.assertEqual(d['first_witnesses'][0]['determinant_m3'],0.)

    def test_invalid_features_and_source_rejected(self):
        x,t=self.bipyramid()
        for kw in [dict(selected_edges=[[1,1]]),dict(selected_edges=[[2,4]]),
                   dict(selected_edges=[[0.,1.]]),dict(selected_faces=[[0,2,4]]),
                   dict(selected_tetrahedra=[4]),dict(selected_tetrahedra=[-.5])]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):
                refine_interior_support(x,t,**kw)
        bad=t.copy();bad[0,2:]=bad[0,2:][::-1]
        with self.assertRaisesRegex(ValueError,'positive'):refine_interior_support(x,bad)
        with self.assertRaisesRegex(ValueError,'duplicate'):refine_interior_support(x,np.concatenate([t,t[:1]]))

if __name__=='__main__':unittest.main()
