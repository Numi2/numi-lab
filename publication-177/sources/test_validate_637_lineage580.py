import unittest
import numpy as np
from validate_637_lineage580 import derive_singleton_barycentric

class LineageTests(unittest.TestCase):
    def test_singleton_q_rows_map_to_parent_triangle_order(self):
        ptr=np.array([0,1,3,4],dtype=np.int64)
        idx=np.array([4,4,8,9],dtype=np.int64)
        w=np.array([1.0,0.5,0.5,1.0],dtype=np.float64)
        got=derive_singleton_barycentric([0,1,2],[4,8,9],ptr,idx,w)
        np.testing.assert_array_equal(got,np.array([[1,0,0],[.5,.5,0],[0,0,1]],dtype=np.float64))
    def test_support_outside_single_parent_fails_closed(self):
        ptr=np.array([0,1,2,3],dtype=np.int64); idx=np.array([4,10,9],dtype=np.int64); w=np.ones(3)
        with self.assertRaisesRegex(ValueError,"outside"):
            derive_singleton_barycentric([0,1,2],[4,8,9],ptr,idx,w)
    def test_nonunit_affine_row_fails_closed(self):
        ptr=np.array([0,1,3,4],dtype=np.int64); idx=np.array([4,4,8,9],dtype=np.int64)
        w=np.array([1.0,.2,.2,1.0])
        with self.assertRaisesRegex(ValueError,"preserve affine weights"):
            derive_singleton_barycentric([0,1,2],[4,8,9],ptr,idx,w)
if __name__=="__main__": unittest.main()

