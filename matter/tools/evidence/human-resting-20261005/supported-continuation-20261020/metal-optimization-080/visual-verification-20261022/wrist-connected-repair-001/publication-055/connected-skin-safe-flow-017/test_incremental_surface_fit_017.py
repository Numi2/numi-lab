import unittest
import numpy as np
from incremental_surface_fit_017 import surface_step_quality
class SurfaceStepQualityTests(unittest.TestCase):
    def test_large_total_rotation_in_small_valid_steps(self):
        f=np.array([[0,1,2]]);x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        original=x.copy();initial_area=np.array([1.])
        for angle in np.deg2rad(np.arange(20,181,20)):
            c,s=np.cos(angle),np.sin(angle)
            R=np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])
            y=original@R.T
            ok,cos,ratio=surface_step_quality(x,y,f,initial_area)
            self.assertTrue(ok);self.assertAlmostEqual(float(ratio[0]),1.)
            x=y
        self.assertLess(np.dot(np.cross(*(x[1:]-x[0])),np.cross(*(original[1:]-original[0]))),0)
    def test_collapsed_face_fails(self):
        x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]);f=np.array([[0,1,2]])
        y=x.copy();y[2]=y[1]
        self.assertFalse(surface_step_quality(x,y,f,np.array([1.]))[0])
    def test_nonfinite_fails(self):
        x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]]);f=np.array([[0,1,2]])
        y=x.copy();y[1,0]=np.nan
        self.assertFalse(surface_step_quality(x,y,f,np.array([1.]))[0])
if __name__=='__main__':unittest.main()
