from pathlib import Path
import importlib.util
import numpy as np
import pytest

D=Path('/Users/numi/Work/numi-human-connected-surface-069')
H=D/'barycentric-connected-clearance-170'
POLICY_PATH=H/'no-progress-backtracker-373/backtracking_policy_373.py'
GLOBAL_PATH=H/'global_step_backtracking_275.py'

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

policy=load(POLICY_PATH,'policy373')
global_helper=load(GLOBAL_PATH,'global275')

F=np.array([[0,1,2],[1,2,3]],dtype=np.int64)
S=np.array([0.,.8,.6,.4])
A=np.array([0],dtype=np.int64)

@pytest.mark.parametrize('kind,bad,pairs,point',[
    ('area',[0],[],None),
    ('self',[],[[0,1]],None),
    ('point',[],[],[1]),
    ('surface',[1],[],None),
])
def test_uniform_half_applies_to_all_exact_failure_kinds(kind,bad,pairs,point):
    new,receipt=policy.update_failure_scales(
        faces=F,scales=S,bad_faces=bad,self_pairs=pairs,anchors=A,
        fixed_prefix_nodes=1,zero_below=1/128,failure_kind=kind,
        uniform_failure_backtracking=True,local_reducer=global_helper.reduce_global_step_scales,
        global_reducer=global_helper.reduce_global_step_scales,
        point_witness_faces=point)
    assert np.array_equal(new,np.array([0.,.4,.3,.2]))
    assert receipt['policy']=='UNIFORM_FAILURE_BACKTRACKING'
    assert receipt['failure_kind']==kind
    assert receipt['native_admission'] is False
    assert receipt['accepted_physical_steps']==0
    assert receipt['geometry_certified'] is False
    assert receipt['changed_nodes']==3
    assert np.all(new<=S)
    assert new[0]==0

def test_uniform_policy_routes_point_witness_faces_not_fake_face():
    seen={}
    def reducer(faces,scales,bad,pairs,**kwargs):
        seen['bad']=np.asarray(bad).copy()
        return global_helper.reduce_global_step_scales(faces,scales,bad,pairs,**kwargs)
    new,rec=policy.update_failure_scales(
        faces=F,scales=S,bad_faces=[],self_pairs=[],anchors=A,fixed_prefix_nodes=1,
        zero_below=1/128,failure_kind='point',uniform_failure_backtracking=True,
        local_reducer=reducer,global_reducer=reducer,point_witness_faces=[1])
    assert np.array_equal(seen['bad'],[1])
    assert rec['failure_witness_face_rows']==[1]
    assert new[1]==pytest.approx(.4)

def test_local_policy_preserves_local_reducer_selection():
    called=[]
    def local(faces,scales,bad,pairs,**kwargs):
        called.append('local')
        result=np.asarray(scales).copy()
        result[1]*=.25
        return result,{'status':'LOCAL_FIXTURE'}
    def global_reducer(*a,**k):
        raise AssertionError('global reducer used when opt-in is false')
    new,rec=policy.update_failure_scales(
        faces=F,scales=S,bad_faces=[0],self_pairs=[],anchors=A,fixed_prefix_nodes=1,
        zero_below=1/128,failure_kind='area',uniform_failure_backtracking=False,
        local_reducer=local,global_reducer=global_reducer)
    assert called==['local']
    assert rec['policy']=='LOCAL_WITNESS_BACKTRACKING'
    assert new[1]==pytest.approx(.2)

def test_zero_scales_reports_no_progress_and_caps():
    zero=np.zeros(4)
    new,rec=policy.update_failure_scales(
        faces=F,scales=zero,bad_faces=[0],self_pairs=[],anchors=A,fixed_prefix_nodes=1,
        zero_below=1/128,failure_kind='area',uniform_failure_backtracking=True,
        local_reducer=global_helper.reduce_global_step_scales,
        global_reducer=global_helper.reduce_global_step_scales)
    assert np.array_equal(new,zero)
    assert rec['status']=='NO_CONSTRUCTION_SCALE_PROGRESS'
    assert rec['unchanged_scales'] is True

def test_missing_point_witness_and_out_of_range_witness_fail_closed():
    args=dict(faces=F,scales=S,bad_faces=[],self_pairs=[],anchors=A,fixed_prefix_nodes=1,
              zero_below=1/128,failure_kind='point',uniform_failure_backtracking=True,
              local_reducer=global_helper.reduce_global_step_scales,
              global_reducer=global_helper.reduce_global_step_scales)
    with pytest.raises(ValueError,match='actual incident-face'):
        policy.update_failure_scales(**args,point_witness_faces=[])
    with pytest.raises(ValueError,match='point-witness face'):
        policy.update_failure_scales(**args,point_witness_faces=[99])
    with pytest.raises(ValueError,match='exact witnesses'):
        policy.update_failure_scales(**{**args,'failure_kind':'surface'},point_witness_faces=None)
