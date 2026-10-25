import numpy as np
import pytest
from swept_self_pair_preseed_350 import swept_face_aabb_candidates, classify_pair_rows, canonical_pairs


def test_swept_interval_catches_intermediate_contact_and_union_is_deterministic():
    base=np.asarray([[0,0,0],[1,0,0],[0,1,0], [0.2,0.2,1],[0.8,0.2,1],[0.2,0.8,1]],np.float32)
    faces=np.asarray([[0,1,2],[3,4,5]],np.int64)
    end=base.copy();end[3:,2]=-1
    a,meta=swept_face_aabb_candidates(base,faces,[end],scale=1,padding_m=0)
    b,_=swept_face_aabb_candidates(base,faces,[end],scale=1,padding_m=0)
    assert np.array_equal(a,np.array([[0,1]],np.int64))
    assert np.array_equal(a,b)
    assert meta['swept_aabb_pair_count']==1


def test_scaled_raw_displacement_expands_box():
    base=np.asarray([[0,0,0],[1,0,0],[0,1,0], [0.2,0.2,1],[0.8,0.2,1],[0.2,0.8,1]],np.float32)
    faces=np.asarray([[0,1,2],[3,4,5]],np.int64)
    trial=base.copy();trial[3:,2]=.4
    at1,_=swept_face_aabb_candidates(base,faces,[trial],scale=1,padding_m=0)
    at2,_=swept_face_aabb_candidates(base,faces,[trial],scale=2,padding_m=0)
    assert at1.shape==(0,2)
    assert np.array_equal(at2,np.array([[0,1]],np.int64))


def test_shared_edge_excluded_but_explicit_seed_retained():
    p=np.asarray([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],np.float32)
    f=np.asarray([[0,1,2],[1,3,2]],np.int64)
    candidates,meta=swept_face_aabb_candidates(p,f,[p],scale=1,padding_m=0)
    assert candidates.shape==(0,2)
    retained,_=swept_face_aabb_candidates(p,f,[p],scale=1,padding_m=0,include_pairs=[[1,0]])
    assert np.array_equal(retained,np.array([[0,1]],np.int64))
    assert meta['excluded_shared_vertex_threshold']==2


def test_seed_pairs_union_and_deduplicate():
    p=np.asarray([[0,0,0],[1,0,0],[0,1,0], [0,0,1],[1,0,1],[0,1,1]],np.float32)
    f=np.asarray([[0,1,2],[3,4,5]],np.int64)
    pairs,_=swept_face_aabb_candidates(p,f,[p],include_pairs=[[1,0],[0,1]])
    assert np.array_equal(pairs,np.array([[0,1]],np.int64))


def test_rejected_pair_rows_are_returned_explicitly_not_dropped():
    class Owner:
        @staticmethod
        def build_pair_rows(points_mm,faces,pairs):
            if tuple(pairs[0])==(0,1): raise ValueError('fixture nonseparable')
            return [{'lower':0.0}],[{'kind':'eligible'}]
    p=np.asarray([[0,0,0],[1,0,0],[0,1,0], [0,0,1],[1,0,1],[0,1,1], [0,0,2],[1,0,2],[0,1,2]],np.float32)
    f=np.asarray([[0,1,2],[3,4,5],[6,7,8]],np.int64)
    result=classify_pair_rows(p,f,np.asarray([[0,1],[0,2]],np.int64),Owner)
    assert result['eligible_pairs']==[(0,2)]
    assert result['rejected_pairs']==[{'face_rows':(0,1),'exception_type':'ValueError','reason':'fixture nonseparable'}]
    assert result['rejected_pair_count']==1
    assert result['rejected_pairs_retained_in_broadphase_output'] is True


def test_bad_mesh_pair_and_trial_inputs_fail_closed():
    p=np.zeros((3,3),np.float32);f=np.array([[0,1,3]],np.int64)
    with pytest.raises(ValueError): swept_face_aabb_candidates(p,f,[p])
    f=np.array([[0,1,2]],np.int64)
    with pytest.raises(ValueError): swept_face_aabb_candidates(p,f,[p.astype(np.float64)])
    with pytest.raises(ValueError): canonical_pairs([[0,0]],1)
    with pytest.raises(ValueError): canonical_pairs([[0,2]],2)


def test_owner_timeout_marks_unvisited_pairs_unclassified():
    class Owner:
        @staticmethod
        def build_pair_rows(points_mm,faces,pairs):
            raise AssertionError('expired budget must not call owner')
    p=np.asarray([[0,0,0],[1,0,0],[0,1,0], [0,0,1],[1,0,1],[0,1,1], [0,0,2],[1,0,2],[0,1,2]],np.float32)
    f=np.asarray([[0,1,2],[3,4,5],[6,7,8]],np.int64)
    result=classify_pair_rows(p,f,np.asarray([[0,1],[0,2]],np.int64),Owner,max_seconds=1e-12)
    assert result['eligible_pair_count']==0
    assert result['unclassified_pairs']==[(0,1),(0,2)]
    assert result['classification_completed'] is False
