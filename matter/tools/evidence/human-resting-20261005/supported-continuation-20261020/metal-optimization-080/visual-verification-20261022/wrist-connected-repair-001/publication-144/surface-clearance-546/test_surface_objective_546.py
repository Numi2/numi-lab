import unittest
import numpy as np
from scipy import sparse
from surface_objective_546 import retain_surface_rows546, select_outside_depth_rows546

class Tests(unittest.TestCase):
    def test_empty_point_keeps_every_surface_row_and_weights(self):
        C0=sparse.csr_matrix([[1.,0.],[0.,1.]])
        h0=np.array([.2,.4]); w0=np.array([.25,.75])
        meta=[{"kind":"face","pair_index":3},{"kind":"face","pair_index":4}]
        def build(*args): return C0,h0,w0,meta
        point={"point_rows":np.empty(0,np.int64)}
        C,h,w,m,rec=retain_surface_rows546(build,np.zeros((3,3)),[[0,1,2]],np.arange(3),[],point,{},None)
        self.assertEqual((C!=C0).nnz,0); self.assertTrue(np.array_equal(h,h0)); self.assertTrue(np.array_equal(w,w0))
        self.assertEqual([x["pair_index"] for x in m],[3,4]); self.assertEqual(rec["surface_weight_mass_preserved"],1.)
    def test_nonempty_point_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"empty point-target"):
            retain_surface_rows546(lambda *a:None,np.zeros((1,3)),[],[],[],{"point_rows":np.array([0])},{},None)
    def test_empty_surface_objective_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"no positive"):
            retain_surface_rows546(lambda *a:(sparse.csr_matrix((0,3)),[],[],[]),np.zeros((1,3)),[],[],[],{"point_rows":[]},{},None)
    def test_outside_depth_rows_only_and_typed_ids(self):
        nodes,normals,lower,ids,rec=select_outside_depth_rows546([2,3,4],[[1,0,0]]*3,[-.1,0,.2],[[1,10],[2,20],[3,30]])
        self.assertEqual(nodes.tolist(),[2]); self.assertEqual(lower.tolist(),[-.1]); self.assertEqual(ids.tolist(),[[1,10]])
        self.assertEqual(rec["omitted_inside_monotonicity_rows"],2)
    def test_no_outside_depth_rows_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"at least one outside"):
            select_outside_depth_rows546([2],[[1,0,0]],[0.])

if __name__ == "__main__": unittest.main(verbosity=2)
