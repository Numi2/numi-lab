import unittest,importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).with_name('finite_witness_fit_014.py')
spec=importlib.util.spec_from_file_location('fit014',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class FiniteWitnessTest(unittest.TestCase):
 def test_barycentric_affine_reconstruction(self):
  t=np.array([[.003,.002,.005],[.007,.002,.005],[.003,.008,.005]])
  w=np.array([.2,.3,.5]);np.testing.assert_allclose(m.bary(w@t,t),w,atol=1e-14)
 def test_finite_witness_rows_keep_bary_and_sign(self):
  x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
  f=np.array([[0,1,2]])
  # A tetrahedral closed bank provides positive outward winding.
  v=np.array([[0.,0.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
  bf=np.array([[0,2,1],[0,1,3],[0,3,2],[1,2,3]])
  z={'bank_ids':np.array([0]),'bank_local_face_ids':np.array([1]),'physical_O_face_ids':np.array([0]),
   'exact_hit_offsets':np.array([0,1]),'exact_hit_points_m':np.array([[.25,0.,0.]])}
  A,b,src,sign=m.contact_rows(x,f,z,{'b_vertices':v,'b_faces':bf},[{'key':'b'}])
  self.assertEqual(A.shape[1],9);self.assertAlmostEqual(b[0],.02)
  n=np.cross(v[bf[1,1]]-v[bf[1,0]],v[bf[1,2]]-v[bf[1,0]]);n*=sign[0]/np.linalg.norm(n)
  np.testing.assert_allclose(A[0].toarray().reshape(3,3),np.array([.75,.25,0.])[:,None]*n)
 def test_lexicographic_fit_retains_unused_nodes(self):
  x=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[3.,3.,3.]])
  f=np.array([[0,1,2]])
  from scipy import sparse
  A=sparse.csr_matrix(([1.],([0],[2])),shape=(1,12))
  d,ids,info,_=m.fit(x,f,A,np.array([.1]))
  self.assertLess(info['contact_slack_sum'],1e-6);self.assertGreaterEqual(d[0,2],.0001-1e-9)
  np.testing.assert_array_equal(d[3],np.zeros(3));self.assertNotIn(3,ids)
if __name__=='__main__':unittest.main()

