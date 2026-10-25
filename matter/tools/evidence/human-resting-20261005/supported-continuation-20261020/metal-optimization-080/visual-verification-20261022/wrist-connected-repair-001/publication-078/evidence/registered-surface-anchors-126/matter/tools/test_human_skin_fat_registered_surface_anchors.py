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
