import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec = importlib.util.spec_from_file_location("geometry_cut", Path(__file__).resolve().parents[1] / "tools/human_skin_patch_geometry.py")
geom = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geom)

class ExactCutPlaneTests(unittest.TestCase):
    def test_near_plane_side_faces_are_not_deleted(self):
        # Regression: the old 50um plus normal test deleted these tissue faces.
        x = np.array([[0,0,-.19728248],[.001,0,-.19728248],[0,.001,-.19728248],
                      [0,0,-.19726455],[.001,0,-.19726455],[0,.001,-.19726455]],np.float32)
        actual = geom.exact_float32_planar_face_mask(x, np.array([[0,1,2],[3,4,5]]), axis=2, coordinate=-.19728248)
        np.testing.assert_array_equal(actual,[True,False])

    def test_emission_not_float64_distance_controls_membership(self):
        c = np.float32(.125)
        x = np.array([[0,0,float(c)+1e-10],[1,0,float(c)-1e-10],[0,1,float(c)]])
        self.assertTrue(geom.exact_float32_planar_face_mask(x,np.array([[0,1,2]]),axis=2,coordinate=float(c)+1e-10)[0])
        x[2,2] = np.nextafter(c,np.float32(np.inf))
        self.assertFalse(geom.exact_float32_planar_face_mask(x,np.array([[0,1,2]]),axis=2,coordinate=c)[0])

    def test_axis_and_empty_surface(self):
        x = np.array([[2,0,0],[2,1,0],[2,0,1]])
        self.assertTrue(geom.exact_float32_planar_face_mask(x,np.array([[0,1,2]]),axis=0,coordinate=2)[0])
        self.assertEqual(geom.exact_float32_planar_face_mask(x,np.empty((0,3),np.int64),axis=0,coordinate=2).shape,(0,))

    def test_bad_inputs_and_emitted_degeneracy_fail(self):
        x = np.array([[0,0,0],[1,0,0],[0,1,0]],float)
        cases = [(x,np.array([[0,1,3]]),2,0),(x,np.array([[-1,1,2]]),2,0),
                 (x,np.array([[0.,1.,2.]]),2,0),(x,np.array([[0,1,2]]),True,0),
                 (x,np.array([[0,1,2]]),2,np.inf),(x,np.array([[0,1,2]]),2,[0]),
                 (x*1e39,np.array([[0,1,2]]),2,0),
                 (np.zeros((3,3)),np.array([[0,1,2]]),2,0)]
        for v,f,a,c in cases:
            with self.subTest(axis=a,plane=c), self.assertRaises(ValueError):
                geom.exact_float32_planar_face_mask(v,f,axis=a,coordinate=c)




class ExactCapTopologyTests(unittest.TestCase):
    def setUp(self):
        self.x=np.array([[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1]],np.float32)
        self.f=np.array([[5,1,2],[5,2,3],[5,3,4],[5,4,1],
                         [0,2,1],[0,3,2],[0,4,3],[0,1,4]])

    def test_opened_cap_is_a_single_disk_with_original_face_ids(self):
        opened,mask,top=geom.remove_exact_planar_cap(self.x,self.f,axis=2,coordinate=0)
        np.testing.assert_array_equal(opened,self.f[:4])
        np.testing.assert_array_equal(mask,[False]*4+[True]*4)
        self.assertEqual(top["genus"],0)
        self.assertEqual(len(top["boundary_loops"]),1)
        self.assertEqual(top["euler_characteristic"],1)

    def test_nonplanar_or_open_input_is_not_repaired_by_tolerance(self):
        bad=self.x.copy();bad[0,2]=1e-5
        with self.assertRaisesRegex(ValueError,"no exact"):
            geom.remove_exact_planar_cap(bad,self.f,axis=2,coordinate=0)
        with self.assertRaisesRegex(ValueError,"closed manifold"):
            geom.remove_exact_planar_cap(self.x,self.f[:-1],axis=2,coordinate=0)

    def test_requested_topology_is_mandatory(self):
        with self.assertRaisesRegex(ValueError,"unexpected"):
            geom.remove_exact_planar_cap(self.x,self.f,axis=2,coordinate=0,expected_genus=1)
        with self.assertRaisesRegex(ValueError,"genus"):
            geom.remove_exact_planar_cap(self.x,self.f,axis=2,coordinate=0,expected_genus=True)
