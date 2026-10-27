import unittest,numpy as np
from changed_triangle_masks572 import changed_triangle_masks
class Tests(unittest.TestCase):
 def test_cyclic_and_order_preserve(self):
  a=np.array([[0,1,2],[0,2,3]])
  x,y=changed_triangle_masks(a,np.array([[2,3,0],[1,2,0]]))
  self.assertFalse(x.any());self.assertFalse(y.any())
 def test_unaffected_not_new(self):
  a=np.array([[0,1,2],[0,2,3],[3,4,5]])
  b=np.array([[0,1,3],[1,2,3],[4,5,3]])
  x,y=changed_triangle_masks(a,b)
  self.assertEqual(x.tolist(),[True,True,False]);self.assertEqual(y.tolist(),[True,True,False])
 def test_orientation_reversal_is_new(self):
  a=np.array([[0,1,2]]);x,y=changed_triangle_masks(a,a[:,::-1]);self.assertTrue(x.all());self.assertTrue(y.all())
 def test_invalid_rejected(self):
  for a,b in [(np.array([[0,0,2]]),np.array([[0,1,2]])),(np.array([[0,1,2],[0,1,2]]),np.array([[0,1,2],[0,1,3]]))]:
   with self.assertRaises(ValueError):changed_triangle_masks(a,b)
if __name__=="__main__":unittest.main()

