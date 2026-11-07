import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from metalrobo.crow_retinal_normalization import fit_rgb_normalization
class RetinalNormalizationTest(unittest.TestCase):
 def test_population_moments_and_floor(self):
  x=np.tile(np.array([0.,.2,.4,1.],np.float32)[:,None],(1,1152));x[:,1]=.7
  mean,inverse=fit_rgb_normalization(x,np.ones(4),.05)
  expected=np.mean(x.astype('f8'),axis=0);std=np.sqrt(np.mean((x.astype('f8')-expected)**2,axis=0))
  np.testing.assert_array_equal(mean,expected.astype('f4'));np.testing.assert_array_equal(inverse,(1/np.maximum(std,.05)).astype('f4'))
  self.assertEqual(inverse[1],20.)
 def test_invalid_frames_have_no_statistical_influence(self):
  x=np.zeros((4,1152),'f4');x[1]=.3;x[3]=.7
  expected=fit_rgb_normalization(x[[1,3]],np.ones(2),.05)
  x[0]=.99;x[2]=.99
  actual=fit_rgb_normalization(x,np.array([0,1,0,1]),.05)
  for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
 def test_malformed_input_rejects(self):
  x=np.zeros((2,1152),'f4');v=np.ones(2)
  for floor in [True,0.,.001,1.01,float('nan'),float('inf'),'x']:
   with self.subTest(floor=floor),self.assertRaises(ValueError):fit_rgb_normalization(x,v,floor)
  bad=np.copy(x);bad[0,0]=float('nan')
  for a,b in [(bad,v),(x[:,:-1],v),(x,np.array([0,0])),(x,np.array([1,2])),(x,np.ones(3)),(x-1,v),(x+2,v)]:
   with self.assertRaises(ValueError):fit_rgb_normalization(a,b,.05)
if __name__=='__main__':unittest.main()
