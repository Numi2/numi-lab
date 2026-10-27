import numpy as np
import pytest
import run_depthmax_targets589_v2 as m

def test_cli_requires_depth_maximizing_with_pinned_005_and_357_rows():
    argv=['--direction-mode','depth-maximizing','--minimum-depth-cosine','.05','--point-depth-rows-npz','rows.npz','--point-depth-rows-report','rows.json']
    assert m.require_depth_maximizing(argv)=='depth-maximizing'
    with pytest.raises(ValueError,match='requires --direction-mode depth-maximizing'):
        m.require_depth_maximizing(['--direction-mode','source-outward','--minimum-depth-cosine','.05','--point-depth-rows-npz','a','--point-depth-rows-report','b'])
    with pytest.raises(ValueError,match='pins minimum-depth-cosine'):
        m.require_depth_maximizing(['--direction-mode','depth-maximizing','--minimum-depth-cosine','.1','--point-depth-rows-npz','a','--point-depth-rows-report','b'])
    with pytest.raises(ValueError,match='requires both pinned 357'):
        m.require_depth_maximizing(['--direction-mode','depth-maximizing','--minimum-depth-cosine','.05'])

def _fixture():
    report={'direction_mode':'depth-maximizing','physical_steps':0,'minimum_depth_cosine':.05,
      'all_bank_ray_results':1,'passing_targets':1,
      'targets':[{'query_row':4,'status':'PASS_FIRST_UNION_EXIT','target_exactly_outside_all185':True,
        'exact_endpoint_bank_count':185,'end_outside_all185_by_event_parity':True,
        'proposal_direction_world_unit':[0.05,(1-.05**2)**.5,0],
        'direction_compatibility':{'selected_source_hemisphere_cosine':.05,'selected_depth_cosine':.05},
        'selection':{'target_exactly_outside_all_banks':True,'classified_bank_count':185}}]}
    a={'query_rows':np.asarray([4]),'source_surface_normal_world_unit':np.asarray([[1.,0,0]]),
       'proposal_direction_world_unit':np.asarray([[.05,(1-.05**2)**.5,0]]),
       'depth_covector_world_unit':np.asarray([[.05,(1-.05**2)**.5,0]]),
       'pass_mask':np.asarray([True]),'target_exactly_outside_all185':np.asarray([True])}
    return report,a

def test_depthmax_report_requires_source_and_depth_cosines_plus_all185():
    r,a=_fixture(); result=m.validate_545_depth_maximizing_report(r,a)
    assert result['passing_targets']==1 and result['all_passing_endpoints_exactly_outside_185']
    r,a=_fixture(); a['depth_covector_world_unit'][0]=[-1,0,0]
    with pytest.raises(ValueError,match='violates pinned source/depth cosine floors'):
        m.validate_545_depth_maximizing_report(r,a)

def test_depthmax_report_rejects_wrong_mode_or_missing_bank_certificate():
    r,a=_fixture(); r['direction_mode']='direct-depth'
    with pytest.raises(ValueError,match='not depth-maximizing'):
        m.validate_545_depth_maximizing_report(r,a)
    r,a=_fixture();r['targets'][0]['exact_endpoint_bank_count']=184
    with pytest.raises(ValueError,match='complete all185'):
        m.validate_545_depth_maximizing_report(r,a)

def test_current_source_sign_gate_retained_for_depthmax_adapter():
    points=np.asarray([[0,0,0],[1,0,0],[0,1,0]],np.float32);faces=np.asarray([[0,1,2]],np.int64)
    passed=m.current_source_alignment(points,faces,np.asarray([0]),{0:np.asarray([0,0,-1.])},-1)
    failed=m.current_source_alignment(points,faces,np.asarray([0]),{0:np.asarray([0,0,1.])},-1)
    assert passed['status']=='PASS' and failed['witnesses'][0]['status']=='CURRENT_SOURCE_OUTWARD_DISAGREEMENT'
