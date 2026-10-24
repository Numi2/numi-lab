import importlib.util
from pathlib import Path
import unittest
import numpy as np
path=Path(__file__).resolve().parents[1]/"tools/human_skin_contact_planes.py"
spec=importlib.util.spec_from_file_location("planes280",path)
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

class NormalTransportTests(unittest.TestCase):
    def test_rotation_and_uniform_scale(self):
        m=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        n=np.array([1.,2.,3.]);expected=m@n;expected/=np.linalg.norm(expected)
        np.testing.assert_allclose(p.transport_contact_normal(n,5*m),expected,atol=1e-15)
    def test_shear_preserves_plane_tangents_and_side(self):
        m=np.array([[2.,3.,0.],[0.,.5,1.],[0.,0.,4.]])
        n=np.array([1.,2.,-3.]);nn=p.transport_contact_normal(n,m)
        for tangent in (np.array([2.,-1.,0.]),np.array([3.,0.,1.])):
            self.assertLess(abs(float(nn@(m@tangent))),2e-15)
        self.assertGreater(float(nn@(m@n)),0.)
        self.assertGreater(np.linalg.norm(nn-(m@n)/np.linalg.norm(m@n)),.1)
    def test_extreme_common_units_do_not_overflow(self):
        m=np.array([[2.,.5,0.],[0.,3.,1.],[0.,0.,4.]])
        n=np.array([1.,2.,3.]);expected=p.transport_contact_normal(n,m)
        for scale in (1e-250,1e250):
            np.testing.assert_allclose(p.transport_contact_normal(scale*n,scale*m),expected,atol=1e-15)
    def test_invalid_inputs_fail_closed(self):
        for n,m in (([0,0,0],np.eye(3)),([1,0,0],np.zeros((3,3))),([1,0,0],np.diag([1,0,1])),([1,np.nan,0],np.eye(3)),([1,0,0],np.full((3,3),np.inf)),([1,0],np.eye(3))):
            with self.assertRaises(ValueError):p.transport_contact_normal(n,m)

if __name__=="__main__":unittest.main()
