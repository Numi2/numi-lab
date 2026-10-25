import unittest
import numpy as np
from local_witness_backtracking import reduce_witnessed_step_scales


class LocalWitnessBacktrackingTests(unittest.TestCase):
    def test_bad_face_reduces_only_incident_nodes(self):
        faces=np.array([[0,1,2],[2,3,4]],np.int64)
        scales=np.array([.8,.6,.4,.9,.7,.3],np.float64)
        out,count,report=reduce_witnessed_step_scales(faces,scales,[0])
        np.testing.assert_array_equal(out[:3],scales[:3]*.5)
        np.testing.assert_array_equal(out[3:],scales[3:])
        self.assertEqual(report['unaffected_scales_bitwise_unchanged'],True)
        self.assertEqual(count.tolist(),[1,1,1,0,0,0])

    def test_self_pair_reduces_both_faces_and_anchor_is_fixed(self):
        faces=np.array([[0,1,2],[3,4,5],[5,6,7]],np.int64)
        scales=np.ones(8,np.float64)
        out,count,report=reduce_witnessed_step_scales(faces,scales, [0], [[1,2]], anchor_rows=[4])
        np.testing.assert_array_equal(out[[0,1,2,3,5,6,7]],.5)
        self.assertEqual(out[4],0.0)
        self.assertEqual(count[[0,1,2,3,5,6,7]].tolist(),[1]*7)
        self.assertEqual(count[4],1)

    def test_bounded_halvings_then_exact_zero_fallback(self):
        f=np.array([[0,1,2]],np.int64); s=np.ones(3); counts=np.zeros(3,np.int64)
        for expected in (.5,.25):
            s,counts,_=reduce_witnessed_step_scales(f,s,[0],halving_counts=counts,max_halvings=2)
            np.testing.assert_array_equal(s,np.full(3,expected))
        s,counts,r=reduce_witnessed_step_scales(f,s,[0],halving_counts=counts,max_halvings=2)
        np.testing.assert_array_equal(s,np.zeros(3))
        self.assertEqual(r['exact_zero_fallback_node_rows'],[0,1,2])

    def test_no_witnesses_empty_or_duplicate_witnesses_rejected(self):
        f=np.array([[0,1,2],[2,3,4]],np.int64); s=np.ones(5)
        for kwargs in ({'bad_face_rows':[]},{'bad_face_rows':[0,0]},{'bad_face_rows':[0],'self_pair_face_rows':[[0,1],[1,0]]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                bad=kwargs.pop('bad_face_rows') if 'bad_face_rows' in kwargs else [0]
                reduce_witnessed_step_scales(f,s,bad,**kwargs)

    def test_bad_dimensions_indices_and_nonfinite_scales_rejected(self):
        f=np.array([[0,1,2]],np.int64); s=np.ones(3)
        cases=[(np.array([[0,1]]),s,[0],{}),(f,s,[1],{}),(f,np.array([1.,np.nan,1.]),[0],{}),(f,np.array([1.,1.,1.2]),[0],{}),(f,s,[0.0],{}),(f,s,[0],{'anchor_rows':[3]}),(f,s,[0],{'halving_counts':[0,0]})]
        for ff,ss,bad,kw in cases:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                reduce_witnessed_step_scales(ff,ss,bad,**kw)

    def test_runner_wrapper_halves_local_rows_and_zeroes_at_threshold(self):
        from local_witness_backtracking import reduce_witness_node_scales
        f=np.array([[0,1,2],[2,3,4]],np.int64);s=np.ones(5)
        for _ in range(7):
            s,report=reduce_witness_node_scales(f,s,[0],np.empty((0,2),np.int64),zero_below=1/128)
        np.testing.assert_array_equal(s[:3],np.zeros(3))
        np.testing.assert_array_equal(s[3:],np.ones(2))
        self.assertEqual(report['threshold_zero_fallback_node_rows'],[0,1,2])

    def test_empty_pair_list_is_accepted_when_face_witness_exists(self):
        f=np.array([[0,1,2]],np.int64);s=np.ones(3)
        out,_,_=reduce_witnessed_step_scales(f,s,[0],[])
        np.testing.assert_array_equal(out,np.full(3,.5))

    def test_self_pair_only_witness_is_valid(self):
        f=np.array([[0,1,2],[3,4,5]],np.int64);s=np.ones(6)
        out,_,_=reduce_witnessed_step_scales(f,s,[],[[0,1]])
        np.testing.assert_array_equal(out,np.full(6,.5))

    def test_unaffected_nodes_remain_exact_after_prior_rounds(self):
        f=np.array([[0,1,2],[2,3,4]],np.int64);s=np.array([.5,.4,.3,.8,.7,.6])
        c=np.array([2,2,2,0,0,0],np.int64)
        out,c2,_=reduce_witnessed_step_scales(f,s,[0],halving_counts=c,max_halvings=4)
        np.testing.assert_array_equal(out[3:],s[3:]);np.testing.assert_array_equal(c2[3:],c[3:])
        self.assertTrue(np.all(out[:3]==s[:3]*.5))

if __name__=='__main__':unittest.main()
