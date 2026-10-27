import numpy as np
import pytest
import run_source_outward_targets589 as m

def test_source_outward_cli_is_the_only_enabled_direction():
    assert m.require_source_outward(['--direction-mode','source-outward','--candidate','x'])=='source-outward'
    with pytest.raises(ValueError,match='requires --direction-mode source-outward'):
        m.require_source_outward(['--direction-mode','direct-depth'])
    with pytest.raises(ValueError,match='requires --direction-mode source-outward'):
        m.require_source_outward(['--direction-mode','depth-compatible'])

def test_current_normal_uses_pinned_coarse_sign_and_agrees_with_registered_outward():
    points=np.asarray([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    faces=np.asarray([[0,1,2]],np.int64)
    got=m.current_source_alignment(points,faces,np.asarray([0]),{0:np.asarray([0,0,-1.])},-1)
    assert got['status']=='PASS'
    assert got['rows'][0]['current_outward_normal_world_unit']==[0.,0.,-1.]
    assert got['rows'][0]['current_source_outward_dot']==1.0

def test_opposed_current_source_retains_witness_and_fails_gate():
    points=np.asarray([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    faces=np.asarray([[0,1,2]],np.int64)
    got=m.current_source_alignment(points,faces,np.asarray([0]),{0:np.asarray([0,0,1.])},-1)
    assert got['status']=='CURRENT_SOURCE_OUTWARD_DISAGREEMENT'
    assert got['witnesses'][0]['query_row']==0
    assert got['witnesses'][0]['current_source_outward_dot']==-1.0

def test_current_normal_cancellation_is_not_replaced_by_an_arbitrary_normal():
    points=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,-1,0]],np.float32)
    faces=np.asarray([[0,1,2],[0,1,3]],np.int64)
    got=m.current_source_alignment(points,faces,np.asarray([0]),{0:np.asarray([0,0,-1.])},-1)
    assert got['status']=='CURRENT_SOURCE_OUTWARD_DISAGREEMENT'
    assert got['witnesses'][0]['status']=='CURRENT_NORMAL_CANCELLED_OR_UNRESOLVED'

def test_target_report_requires_source_mode_and_complete_185_endpoints():
    report={'direction_mode':'source-outward','physical_steps':0,'all_bank_ray_results':1,'passing_targets':1,
      'targets':[{'query_row':7,'status':'PASS_FIRST_UNION_EXIT','target_exactly_outside_all185':True,'exact_endpoint_bank_count':185}]}
    assert m.validate_545_source_outward_report(report)['all_passing_endpoints_exactly_outside_185']
    bad={**report,'direction_mode':'direct-depth'}
    with pytest.raises(ValueError,match='not source-outward'):
        m.validate_545_source_outward_report(bad)
    bad={**report,'targets':[{'query_row':7,'status':'PASS_FIRST_UNION_EXIT','target_exactly_outside_all185':True,'exact_endpoint_bank_count':184}]}
    with pytest.raises(ValueError,match='lacks complete all185'):
        m.validate_545_source_outward_report(bad)
