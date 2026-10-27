import numpy as np
import pytest
import run_mixed_target_selection591 as m

def test_selection_preserves_row_order_and_explicit_origin():
    rows=np.array([11,12]); proof={'selected_rows':[
      {'row':12,'chosen_mode':'source-outward','all185_endpoint_certified':True,'actual_displacement_dot_source_outward':1.,'actual_displacement_dot_signed_current_outward':.2,'actual_displacement_dot_depth_covector':None,'chosen_move_mm':2.,'source_distance_mm':3.},
      {'row':11,'chosen_mode':'depth-maximizing','all185_endpoint_certified':True,'actual_displacement_dot_source_outward':.05,'actual_displacement_dot_signed_current_outward':.6,'actual_displacement_dot_depth_covector':.7,'chosen_move_mm':1.,'source_distance_mm':4.}]}
    s=np.array([[1,0,0],[2,0,0]],np.float32);d=np.array([[0,1,0],[0,2,0]],np.float32)
    got,origins=m.select_targets(rows,proof,s,d)
    assert origins==['depth-maximizing','source-outward']
    assert np.array_equal(got,np.array([[0,1,0],[2,0,0]],np.float32))

def test_rejects_nonpositive_actual_current_dot():
    q={'row':11,'chosen_mode':'depth-maximizing','all185_endpoint_certified':True,'actual_displacement_dot_source_outward':.5,'actual_displacement_dot_signed_current_outward':-.01,'actual_displacement_dot_depth_covector':.8,'chosen_move_mm':1.,'source_distance_mm':2.}
    with pytest.raises(ValueError,match='positive actual outward dots'):
        m.select_targets(np.array([11]),{'selected_rows':[q]},np.ones((1,3),np.float32),np.ones((1,3),np.float32))

def test_rejects_unbounded_or_uncertified_target():
    q={'row':11,'chosen_mode':'source-outward','all185_endpoint_certified':True,'actual_displacement_dot_source_outward':1.,'actual_displacement_dot_signed_current_outward':1.,'actual_displacement_dot_depth_covector':None,'chosen_move_mm':4.1,'source_distance_mm':1.}
    with pytest.raises(ValueError,match='trust/source cap'):
        m.select_targets(np.array([11]),{'selected_rows':[q]},np.ones((1,3),np.float32),np.ones((1,3),np.float32))
    q['chosen_move_mm']=1.;q['all185_endpoint_certified']=False
    with pytest.raises(ValueError,match='all185'):
        m.select_targets(np.array([11]),{'selected_rows':[q]},np.ones((1,3),np.float32),np.ones((1,3),np.float32))

def test_replacement_preserves_other_arrays_and_unselected_points():
    p=np.arange(18,dtype=np.float32).reshape(6,3);w=np.array([1,2,3],np.int64)
    rows=np.array([2,5]);t=np.array([[100,101,102],[200,201,202]],np.float32)
    out=m.replace_points({'points_world_m':p,'source_prolongation_weights':w},rows,t)
    keep=np.ones(6,bool);keep[rows]=False
    assert out['points_world_m'][rows].tobytes()==t.tobytes()
    assert out['points_world_m'][keep].tobytes()==p[keep].tobytes()
    assert out['source_prolongation_weights'].tobytes()==w.tobytes()

def test_replacement_rejects_anchor_and_duplicate_rows():
    p=np.zeros((4,3),np.float32);t=np.ones((2,3),np.float32)
    with pytest.raises(ValueError,match='anchored'):
        m.replace_points({'points_world_m':p},np.array([1,2]),t,anchors=[2])
    with pytest.raises(ValueError,match='invalid candidate/target arrays'):
        m.replace_points({'points_world_m':p},np.array([1,1]),t)
