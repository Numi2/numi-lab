import unittest
import numpy as np
from human_skin_p1_obstacles import solve_exclusion_field as solve

class JointFieldTests(unittest.TestCase):
    T=np.array([[0,1,2,3]],dtype=np.int64)
    def test_joint_weighted_minimum_and_polygon(self):
        p=np.zeros(4)
        q,r=solve(p,self.T,[(0,np.array([[.9,.1,0,0],[.5,.5,0,0]]))],[],target=1,maximum_value=3)
        self.assertTrue(r["pass"])
        self.assertAlmostEqual(q[0]+q[1],2)
        self.assertTrue(np.allclose(p,0))
    def test_uses_shared_high_weight_node_not_all_vertices(self):
        q,r=solve(np.zeros(4),self.T,[(0,np.array([[.9,.1,0,0]]))],[],target=1,maximum_value=3)
        self.assertTrue(r["pass"])
        self.assertAlmostEqual(q.sum(),1/.9)
        self.assertEqual(q[1],0)
    def test_fixed_boundary_conflict_no_partial_result(self):
        q,r=solve(np.zeros(4),self.T,[(0,np.array([[1.,0,0,0]]))],[0],target=1,maximum_value=3)
        self.assertIsNone(q);self.assertFalse(r["pass"])
    def test_cap_conflict(self):
        q,r=solve(np.zeros(4),self.T,[(0,np.array([[.01,.99,0,0]]))],[1],target=1,maximum_value=3)
        self.assertIsNone(q);self.assertFalse(r["pass"])
    def test_fixed_nodes_and_all_polygon_vertices(self):
        B=np.array([[.25,.25,.25,.25],[0,.5,.5,0]])
        q,r=solve(np.zeros(4),self.T,[(0,B)],[0],target=1,maximum_value=3)
        self.assertTrue(r["pass"]);self.assertEqual(q[0],0)
        self.assertTrue(np.all(B@q>=1-1e-8))
    def test_already_satisfied_no_change(self):
        p=np.ones(4)*2
        q,r=solve(p,self.T,[(0,np.array([[.25]*4]))],[0,1,2,3],target=1,maximum_value=3)
        self.assertTrue(r["pass"]);np.testing.assert_array_equal(p,q)
    def test_order_invariant_objective(self):
        rows=[(0,np.array([[.9,.1,0,0]])),(0,np.array([[0,.1,.9,0]]))]
        a,ar=solve(np.zeros(4),self.T,rows,[],target=1,maximum_value=3)
        b,br=solve(np.zeros(4),self.T,list(reversed(rows)),[],target=1,maximum_value=3)
        self.assertTrue(ar["pass"] and br["pass"])
        self.assertAlmostEqual(ar["objective"],br["objective"])
    def test_negative_barycentric_rejected(self):
        with self.assertRaises(ValueError):
            solve(np.zeros(4),self.T,[(0,np.array([[-.1,1.1,0,0]]))],[],target=1,maximum_value=3)
    def test_invalid_limits_rejected(self):
        with self.assertRaises(ValueError):
            solve(np.zeros(4),self.T,[],[],target=1,maximum_value=3,time_limit=-1)
    def test_nonfinite_costs_rejected(self):
        with self.assertRaises(ValueError):
            solve(np.zeros(4),self.T,[],[],target=1,maximum_value=3,costs=[1,1,float("nan"),1])
if __name__=="__main__":unittest.main()
