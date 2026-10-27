import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import relocate_midpoint_targets586_v2 as m


def test_only_new_inside_midpoints_are_selected_and_old_targets_ignored():
    mids=np.asarray([6,7],np.int64)
    rows=np.asarray([1,6,3,7],np.int64)
    targets=np.asarray([[1,0,0],[6,1,0],[3,0,0],[7,0,1]],np.float32)
    selected,T=m.select_midpoint_targets(mids,rows,targets,np.asarray([0,2],np.int32),8)
    assert selected.tolist()==[6,7] and np.array_equal(T,targets[[1,3]])
    X=np.arange(24,dtype=np.float32).reshape(8,3); Y=m.relocate_points(X,selected,T)
    assert np.array_equal(Y[selected],T) and Y[:6].tobytes()==X[:6].tobytes()
    assert Y[[0,2]].tobytes()==X[[0,2]].tobytes()


def test_no_new_inside_targets_fails_instead_of_noop():
    with pytest.raises(ValueError,match='no appended midpoint'):
        m.select_midpoint_targets(np.asarray([5,6]),np.asarray([1,2]),np.zeros((2,3),np.float32),np.asarray([0]),7)


def test_anchor_duplicate_and_out_of_range_targets_fail_closed():
    with pytest.raises(ValueError,match='typed anchor'):
        m.select_midpoint_targets(np.asarray([5]),np.asarray([5]),np.zeros((1,3),np.float32),np.asarray([5]),6)
    with pytest.raises(ValueError,match='unique integers'):
        m.select_midpoint_targets(np.asarray([4]),np.asarray([4,4]),np.zeros((2,3),np.float32),np.asarray([],np.int64),5)
    with pytest.raises(ValueError,match='outside point domain'):
        m.select_midpoint_targets(np.asarray([5]),np.asarray([5]),np.zeros((1,3),np.float32),np.asarray([],np.int64),5)


def test_source_alignment_diagnostics_retain_nonpositive_hemisphere_witness():
    d=m.source_alignment_diagnostic([0,0,0],[0,0,-1],[0,0,1],[0,0,-1])
    assert d['source_normal_dot_proposal_direction']==-1.0
    assert d['source_normal_dot_applied_direction']==-1.0
    assert d['nonpositive_source_applied_alignment'] is True
    assert d['source_hemisphere_requirement_applied'] is False


def test_refinement_validation_pins_suffix_lineage_QP_and037_reference(tmp_path):
    parent_path=tmp_path/'parent.npz'; candidate_path=tmp_path/'refined.npz'; report_path=tmp_path/'refinement.json'
    X=np.asarray([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],np.float32)
    F=np.asarray([[0,1,2],[1,3,2]],np.int32)
    qptr=np.arange(5,dtype=np.int64); qrows=np.arange(4,dtype=np.int64); qw=np.ones(4,np.float64)
    parent={'points_world_m':X,'source_O_faces':F,'source_prolongation_indptr':qptr,
      'source_prolongation_node_rows':qrows,'source_prolongation_weights':qw,
      'source_prolongation_semantics':np.asarray(['identity Q'],dtype='<U10'),
      'area_reference_037_points_world_m_f64':X.astype(np.float64),
      'area_reference_037_source_O_faces':F.astype(np.int64),
      'area_reference_037_semantics':np.asarray(['037 ref'],dtype='<U8'),
      'area_reference_037_input_sha256':np.asarray(['a'*64],dtype='<U64'),
      'typed_coarse_anchor_output_rows_018':np.asarray([0],np.int32)}
    with parent_path.open('wb') as f: np.savez_compressed(f,**parent)
    E=np.asarray([[1,2]],np.int64); mid=((X[1].astype(np.float64)+X[2].astype(np.float64))*.5).astype(np.float32)
    candidate=dict(parent)
    candidate['points_world_m']=np.concatenate((X,mid[None]),axis=0)
    candidate['source_prolongation_indptr']=np.asarray([0,1,2,3,4,6],np.int64)
    candidate['source_prolongation_node_rows']=np.asarray([0,1,2,3,1,2],np.int64)
    candidate['source_prolongation_weights']=np.asarray([1,1,1,1,.5,.5],np.float64)
    candidate['refinement_node_parent_edge_rows']=np.asarray([[-1,-1]]*4+[[1,2]],np.int64)
    candidate['refinement_node_parent_edge_weights']=np.asarray([[-1,-1]]*4+[[.5,.5]],np.float64)
    candidate['refinement_node_kind']=np.asarray([0,0,0,0,1],np.int8)
    candidate['area_reference_037_points_world_m_f64']=np.concatenate((parent['area_reference_037_points_world_m_f64'],mid.astype(np.float64)[None]),axis=0)
    for suffix,value in (('indptr',qptr),('node_rows',qrows),('weights',qw),('semantics',parent['source_prolongation_semantics'])):
        candidate['prior_input_parent__source_prolongation_'+suffix]=value.copy()
    with candidate_path.open('wb') as f: np.savez_compressed(f,**candidate)
    rows=np.asarray([4],dtype='<i8')
    report={'status':'REFINEMENT_CONSTRUCTION_READY','input_candidate_path':str(parent_path.resolve()),
      'input_candidate_sha256':m.sha(parent_path),'output_candidate_path':str(candidate_path.resolve()),
      'output_candidate_sha256':m.sha(candidate_path),'input_points':4,'parent_prefix_node_count':4,
      'old_node_count':4,'new_node_count':5,'edge_midpoint_rows':[4],'edge_midpoint_rows_count':1,
      'edge_midpoint_rows_sha256':m.sha_i64(rows),'edge_midpoint_parent_edges_sha256':m.sha_i64(E),
      'selected_edge_count':1,'parent_prefix_points_bytes_sha256':m.raw_sha(X),
      'added_nodes_are_only_edge_midpoints':True,'added_face_centroids':0,
      'old_point_prefix_bitwise_unchanged':True,'existing_node_coordinates_moved':False,
      'geometry_moved':False,'physical_steps':0,'area_reference_uses_same_Q':True,
      'area_reference_faces_match_candidate_exactly':True,
      'edge_only_refinement585_sha256':m.REFINER585_SHA,'pinned_546_adapter_sha256':m.ADAPTER546_SHA,
      'pinned_037_refiner_sha256':m.P037_SHA,'pinned_084_owner_sha256':m.P084_SHA}
    report_path.write_text(json.dumps(report))
    a,b=m.loadnpz(parent_path),m.loadnpz(candidate_path)
    assert m.validate_refinement(parent_path,candidate_path,report_path,a,b,report).tolist()==[4]
    report['edge_midpoint_rows_sha256']='0'*64
    with pytest.raises(ValueError,match='midpoint rows do not uniquely cover'):
        m.validate_refinement(parent_path,candidate_path,report_path,a,b,report)


def test_current_q_midpoint_must_equal_composed_parent_rows_and_rejects_tamper():
    parent={
      'source_prolongation_indptr':np.asarray([0,2,3],np.int64),
      'source_prolongation_node_rows':np.asarray([0,1,1],np.int64),
      'source_prolongation_weights':np.asarray([.25,.75,1.0],np.float64),
    }
    child={
      'points_world_m':np.zeros((3,3),np.float32),
      'source_prolongation_indptr':np.asarray([0,2,3,5],np.int64),
      'source_prolongation_node_rows':np.asarray([0,1,1,0,1],np.int64),
      'source_prolongation_weights':np.asarray([.25,.75,1.0,.125,.875],np.float64),
    }
    edges=np.asarray([[0,1]],np.int64); weights=np.asarray([[.5,.5]],np.float64)
    result=m.validate_current_source_prolongation(parent,child,2,np.asarray([2]),edges,weights)
    assert result['status']=='CURRENT_Q_EQUALS_EDGE_MIDPOINT_COMPOSITION'
    child['source_prolongation_weights'][-2:]=[.2,.8]
    with pytest.raises(ValueError,match='weights differ from edge-parent Q composition'):
        m.validate_current_source_prolongation(parent,child,2,np.asarray([2]),edges,weights)
