import unittest
import numpy as np
from human_skin_p1_obstacles import triangle_tet_barycentric_polygon as clip, raise_exclusion_field as raise_field

class ExclusionTests(unittest.TestCase):
    def setUp(self):
        self.X=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
        self.T=np.array([[0,1,2,3]])
    def check_polygon(self,P):
        B,C=clip(self.X,P)
        self.assertTrue(np.all(B >= -2e-12))
        np.testing.assert_allclose(B@self.X,C@P,atol=2e-12)
        np.testing.assert_allclose(B.sum(axis=1),1,atol=2e-12)
        return B,C
    def test_inside(self):
        P=np.array([[.1,.1,.1],[.2,.1,.1],[.1,.2,.1]])
        B,C=self.check_polygon(P);self.assertEqual(len(B),3)
    def test_outside(self):
        B,C=clip(self.X,np.array([[2.,0,0],[2.,1,0],[2.,0,1]]));self.assertEqual(B.shape,(0,4))
    def test_crossing_without_inside_triangle_vertices(self):
        P=np.array([[-1.,.2,.2],[2.,.2,.2],[.2,2.,.2]])
        B,C=self.check_polygon(P);self.assertGreaterEqual(len(B),3)
    def test_shared_face(self):
        B,C=self.check_polygon(self.X[[0,1,2]]);self.assertGreaterEqual(len(B),3)
    def test_shared_vertex(self):
        B,C=self.check_polygon(np.array([[0.,0,0],[-1.,0,0],[0.,-1,0]]));self.assertGreater(len(B),0)
    def test_affine_parity(self):
        P=np.array([[-1.,.2,.2],[2.,.2,.2],[.2,2.,.2]])
        B,C=clip(self.X,P)
        A=np.array([[2.,.3,.2],[.1,3.,.4],[.2,.5,1.]])
        t=np.array([3.,-4.,5.])
        D,E=clip(self.X@A.T+t,P@A.T+t)
        self.assertEqual(len(B),len(D))
        np.testing.assert_allclose(np.sort(B,axis=0),np.sort(D,axis=0),atol=2e-14)
    def test_degenerate_parent_rejected(self):
        X=self.X.copy();X[3]=X[2]
        with self.assertRaises(ValueError):clip(X,self.X[:3])
    def test_nonfinite_rejected(self):
        X=self.X.copy();X[0,0]=np.nan
        with self.assertRaises(ValueError):clip(X,self.X[:3])
    def test_raise_enforces_entire_polygon(self):
        P=np.array([[-1.,.2,.2],[2.,.2,.2],[.2,2.,.2]])
        B,C=clip(self.X,P)
        p,r=raise_field(np.array([0.,.2,.3,.4]),self.T,[(0,B)],np.array([0]))
        self.assertTrue(r["pass"]);self.assertEqual(p[0],0)
        self.assertTrue(np.all(B@p >= 1.-1e-12))
        w=np.arange(1,len(B)+1,dtype=float);w/=w.sum()
        self.assertGreaterEqual(float((w@B)@p),1.-1e-12)
    def test_fixed_conflict_explicit(self):
        p,r=raise_field(np.zeros(4),self.T,[(0,np.eye(4))],np.arange(4))
        self.assertFalse(r["pass"]);self.assertEqual(len(r["conflicts"]),4)
        np.testing.assert_array_equal(p,np.zeros(4))
    def test_empty_constraints(self):
        p,r=raise_field(np.zeros(4),self.T,[],np.array([],dtype=int))
        self.assertTrue(r["pass"]);self.assertIsNone(r["minimum_obstacle_value"])
    def test_monotonic_multiple_tets(self):
        T=np.array([[0,1,2,3],[1,2,3,4]])
        B=np.array([[.2,.2,.3,.3]])
        p,r=raise_field(np.zeros(5),T,[(0,B),(1,B)],np.array([0]))
        self.assertTrue(r["pass"]);self.assertTrue(np.all(p>=0));self.assertEqual(p[0],0)
    def test_cap_conflict_not_silenced(self):
        p,r=raise_field(np.zeros(4),self.T,[(0,np.array([[.999999, .000001,0,0]]))],np.array([0,2,3]),maximum_value=2.)
        self.assertFalse(r["pass"]);self.assertEqual(r["conflicts"][0]["reason"],"construction field cap")
    def test_bad_bary_rejected(self):
        with self.assertRaises(ValueError):raise_field(np.zeros(4),self.T,[(0,np.array([[-.1,.4,.4,.3]]))],np.array([0]))
    def test_no_input_mutation(self):
        p0=np.zeros(4);B=np.array([[.1,.2,.3,.4]])
        p,r=raise_field(p0,self.T,[(0,B)],np.array([0]))
        np.testing.assert_array_equal(p0,np.zeros(4));self.assertTrue(r["pass"])

if __name__=="__main__":unittest.main(verbosity=2)
