import numpy as np,pytest
from human_skin_fat_registered_surface_anchors import registered_surface_anchors as _registered_surface_anchors
P=np.array([[0,0,0],[.0171,0,0],[0,1,0]],np.float32);F=np.array([[0,1,2]])
def registered_surface_anchors(P,F,C,rows):return _registered_surface_anchors(P,F,C,rows,detail_source_vertex_ids=np.arange(len(P)),direct_tolerance_m=1e-6)
def test_registered_id_overrides_unrelated_onehot_position():
    # A one-hot W row could point to detail vertex 0; its rest offset makes
    # vertex 1 the actual registered geometric anchor. W is deliberately absent.
    C=P[[1]].copy()
    Q,R=registered_surface_anchors(P,F,C,[{'coarse_row':0,'source_direct_id':1}])
    np.testing.assert_array_equal(Q,P[[1]].astype(float))
    assert R[0]['coincidence_residual_m']==0
def test_indirect_anchor_preserves_registered_offset():
    C=np.array([[0,0,.001]],np.float32)
    row={'coarse_row':0,'source_direct_id':4294967295,
         'source_face_v0':0,'source_face_v1':1,'source_face_v2':2,
         'source_bary0':1.,'source_bary1':0.,'source_bary2':0.}
    Q,R=registered_surface_anchors(P,F,C,[row])
    np.testing.assert_array_equal(Q,np.zeros((1,3)))
    assert R[0]['coarse_to_source_offset_m'][2]==-float(C[0,2])
def test_direct_mismatch_rejects_without_reselection():
    with pytest.raises(ValueError,match='differs'):
        registered_surface_anchors(P,F,P[[0]].copy(),[{'coarse_row':0,'source_direct_id':1}])
def test_invalid_direct_does_not_fall_back():
    with pytest.raises(ValueError,match='no proximity fallback'):
        registered_surface_anchors(P,F,P[[0]].copy(),[{'coarse_row':0,'source_direct_id':999}])
def test_invalid_face_and_duplicate_row_reject():
    row={'coarse_row':0,'source_direct_id':4294967295,'source_face_v0':0,'source_face_v1':1,'source_face_v2':1,'source_bary0':.5,'source_bary1':.5,'source_bary2':0.}
    with pytest.raises(ValueError,match='source face'):registered_surface_anchors(P,F,P[[0]].copy(),[row])
    with pytest.raises(ValueError,match='duplicate'):registered_surface_anchors(P,F,P[[0,1]].copy(),[{'coarse_row':0,'source_direct_id':0}]*2)

def test_source_ids_are_not_array_rows():
    Q,R=_registered_surface_anchors(P,F,P[[1]].copy(),[{"coarse_row":0,"source_direct_id":123}],detail_source_vertex_ids=np.array([999,123,456]))
    np.testing.assert_array_equal(Q,P[[1]].astype(float))
    assert R[0]["source_vertex_ids"]==[123] and R[0]["detail_vertex_rows"]==[1]

def test_unsigned_face_indices():
    Q,_=registered_surface_anchors(P,F.astype(np.uint32),P[[0]].copy(),[{"coarse_row":0,"source_direct_id":0}])
    np.testing.assert_array_equal(Q,P[[0]].astype(float))

def test_registered_direct_offset_is_preserved_when_coincidence_not_requested():
    Q,R=_registered_surface_anchors(P,F,P[[0]].copy(),[{"coarse_row":0,"source_direct_id":1}],detail_source_vertex_ids=np.arange(len(P)))
    np.testing.assert_array_equal(Q,P[[1]].astype(float))
    assert R[0]["coincidence_residual_m"]>0.017

import unittest
import human_skin_fat_registered_surface_anchors as mod
class ResolverTests(unittest.TestCase):
    def test_reordered_typed_mapping(self):
        got=mod.resolve_typed_coarse_anchors(np.array([99,41]),np.array([0,2,0,3,0]),np.array([41,99,99,41,7]))
        np.testing.assert_array_equal(got,[2,0])
    def test_noncoarse_collision_is_ignored(self):
        got=mod.resolve_typed_coarse_anchors(np.array([41]),np.array([2,0,3]),np.array([41,41,41]))
        np.testing.assert_array_equal(got,[1])
    def test_missing_is_rejected(self):
        with self.assertRaisesRegex(mod.AnchorResolutionError,"missing"):
            mod.resolve_typed_coarse_anchors(np.array([9]),np.array([0,2]),np.array([8,9]))
    def test_ambiguous_is_rejected(self):
        with self.assertRaisesRegex(mod.AnchorResolutionError,"ambiguous"):
            mod.resolve_typed_coarse_anchors(np.array([9]),np.array([0,0]),np.array([9,9]))
    def test_duplicate_anchor_ids_rejected(self):
        with self.assertRaisesRegex(mod.AnchorResolutionError,"duplicated"):
            mod.resolve_typed_coarse_anchors(np.array([9,9]),np.array([0]),np.array([9]))
    def test_alignment_rejected(self):
        with self.assertRaisesRegex(mod.AnchorResolutionError,"aligned"):
            mod.resolve_typed_coarse_anchors(np.array([9]),np.array([0,0]),np.array([9]))

def test_typed_anchor_negative_ids_rejected():
    with pytest.raises(ValueError,match="nonnegative"):
        mod.resolve_typed_coarse_anchors(np.array([-1]),np.array([0]),np.array([-1]))
def test_typed_anchor_overflow_rejected():
    with pytest.raises(ValueError,match="64-bit"):
        mod.resolve_typed_coarse_anchors(np.array([2**63],dtype=np.uint64),np.array([0]),np.array([1]))
