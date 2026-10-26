import sys
from pathlib import Path
import unittest
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import current_parent_normal_helper555 as m


class CurrentParentNormalTests(unittest.TestCase):
    def test_centroid_uses_orientation_signed_current_parent_face(self):
        pp=np.array([[0,0,0],[2,0,0],[0,2,0]],np.float32)
        cp=np.vstack([pp,np.array([[2/3,2/3,0]],np.float32)])
        F=np.array([[0,1,2]],np.int64)
        n,prov=m.parent_normal_for_node(3,parent_points=pp,parent_faces=F,child_points=cp,
            node_kind=np.array([0,0,0,2],np.int8),parent_face_indptr=np.array([0,0,0,0,1]),
            parent_face_rows=np.array([0]),parent_face_barycentric=np.array([[1/3,1/3,1/3]]),
            outward_sign=-1)
        np.testing.assert_allclose(n,[0,0,-1],atol=1e-14)
        self.assertEqual(prov["method"],"immediate-parent-face-area-weighted-oriented-normal")

    def test_edge_midpoint_is_area_weighted_two_face_bisector(self):
        pp=np.array([[0,0,0],[2,0,0],[0,1,0],[0,0,4]],np.float32)
        cp=np.vstack([pp,np.array([[1,0,0]],np.float32)])
        F=np.array([[0,1,2],[1,0,3]],np.int64)
        n,prov=m.parent_normal_for_node(4,parent_points=pp,parent_faces=F,child_points=cp,
            node_kind=np.array([0,0,0,0,1],np.int8),parent_face_indptr=np.array([0,0,0,0,0,2]),
            parent_face_rows=np.array([0,1]),
            parent_face_barycentric=np.array([[.5,.5,0],[.5,.5,0]]),outward_sign=-1)
        expected=np.array([0,-4,-1],float);expected/=np.linalg.norm(expected)
        np.testing.assert_allclose(n,expected,atol=1e-14)
        self.assertEqual(prov["parent_face_rows"],[0,1])

    def test_prefix_and_unknown_kind_reject(self):
        pp=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
        cp=np.vstack([pp,np.array([[1/3,1/3,0]],np.float32)])
        F=np.array([[0,1,2]],np.int64); kinds=np.array([0,0,0,2],np.int8)
        kwargs=dict(parent_points=pp,parent_faces=F,child_points=cp,node_kind=kinds,
            parent_face_indptr=np.array([0,0,0,0,1]),parent_face_rows=np.array([0]),
            parent_face_barycentric=np.array([[1/3,1/3,1/3]]),outward_sign=1)
        with self.assertRaisesRegex(ValueError,"appended"):
            m.parent_normal_field([2],old_prefix_count=3,**{k:v for k,v in kwargs.items() if k!="outward_sign"},outward_sign=1)
        with self.assertRaisesRegex(ValueError,"not an appended"):
            m.parent_normal_for_node(1,**kwargs)

    def test_wrong_barycentric_point_rejected(self):
        pp=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
        cp=np.vstack([pp,np.array([[.4,.3,0]],np.float32)])
        with self.assertRaisesRegex(ValueError,"does not match"):
            m.parent_normal_for_node(3,parent_points=pp,parent_faces=np.array([[0,1,2]]),
                child_points=cp,node_kind=np.array([0,0,0,2]),parent_face_indptr=np.array([0,0,0,0,1]),
                parent_face_rows=np.array([0]),parent_face_barycentric=np.array([[1/3,1/3,1/3]]),outward_sign=1)

    def test_duplicate_or_degenerate_ancestry_rejected(self):
        pp=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32)
        cp=np.vstack([pp,np.array([[.5,.5,0]],np.float32)])
        F=np.array([[0,1,2],[0,1,3]],np.int64)
        args=dict(parent_points=pp,parent_faces=F,child_points=cp,node_kind=np.array([0,0,0,0,1]),
            parent_face_indptr=np.array([0,0,0,0,0,2]),parent_face_rows=np.array([0,0]),
            parent_face_barycentric=np.array([[.5,.5,0],[.5,.5,0]]),outward_sign=1)
        with self.assertRaisesRegex(ValueError,"duplicate"):
            m.parent_normal_for_node(4,**args)


if __name__=="__main__":unittest.main(verbosity=2)

