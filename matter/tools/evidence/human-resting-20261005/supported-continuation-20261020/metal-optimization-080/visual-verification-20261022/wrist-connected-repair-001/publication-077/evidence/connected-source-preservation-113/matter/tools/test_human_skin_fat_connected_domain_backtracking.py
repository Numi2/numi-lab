import numpy as np
import pytest
from scipy import sparse
from human_skin_fat_connected_repair import coupled_clearance_step

class Gate:
    def __init__(self, x, blocked=()):
        self.coarse_origin=np.asarray(x,dtype=np.float32).copy()
        self.coupling_support=sparse.csr_matrix((0,len(x)))
        self.max_dihedral_change_degrees=None
        self.blocked=tuple(blocked); self.calls=0
    def passes(self, y):
        self.calls+=1; y=np.asarray(y,dtype=np.float32)
        return all(np.array_equal(y[n].view(np.uint32),self.coarse_origin[n].view(np.uint32)) for n in self.blocked)
    def linearization_face_ids(self, free): return np.empty(0,dtype=np.int64)
    def report(self, y): return {'mapped_skin_geometry_pass':self.passes(y)}

def solve(x,t,f,c,target,**kw):
    return coupled_clearance_step(x,x.astype(float),t,f,c,target,step_limit_mm=50.,q_min=1e-5,J_bounds=(.35,2.5),
        time_limit_s=10.,max_incremental_gradient=None,construction_origin=x,cumulative_displacement_weight=.01,**kw)

def fixture():
    x=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[3,0,0],[4,0,0],[3,1,0],[3,0,1]],dtype=np.float32)
    t=np.asarray([[0,1,2,3],[4,5,6,7]],dtype=np.int64); f=np.asarray([3,7],dtype=np.int64)
    c=sparse.csr_matrix(([1.,1.],([0,0],[11,23])),shape=(1,24))
    return x,t,f,c,np.asarray([2.1]),np.asarray([0,0,0,0,1,1,1,1],dtype=np.int32)

def test_blocked_domain_does_not_scale_distant_domain():
    x,t,f,c,target,labels=fixture()
    _,legacy=solve(x,t,f,c,target,embedded_surface=Gate(x,(3,)))
    assert legacy['lp_success'] and not legacy['step_applied']
    assert legacy['components'][0]['attempts'][0]['first_failed_gate']=='mapped_skin'
    y,r=solve(x,t,f,c,target,embedded_surface=Gate(x,(3,)),backtracking_domain_labels=labels)
    assert r['construction_backtracking_domain_fallback_accepted']
    ds=r['construction_backtracking_domain_attempts']
    assert ds[0]['domain_id']==0 and not ds[0]['applied']
    assert ds[1]['domain_id']==1 and ds[1]['applied']
    np.testing.assert_array_equal(y[3].view(np.uint32),x[3].view(np.uint32))
    assert y[7,2]>x[7,2]
    assert r['construction_backtracking_global_final_hard_gates_pass']
    assert r['construction_backtracking_global_final_contact_deficit_reduced']

def test_shared_tet_and_mapped_crease_reject_local_trial():
    x=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
    t=np.asarray([[0,1,2,3]],dtype=np.int64); f=np.asarray([1,2],dtype=np.int64)
    c=sparse.csr_matrix(([1.,1.],([0,0],[3,6])),shape=(1,12)); target=np.asarray([1.2])
    class Crease(Gate):
        def passes(self,y):
            self.calls+=1; y=np.asarray(y,dtype=np.float32)
            dx=y[1,0]-self.coarse_origin[1,0]; dy=y[2,0]-self.coarse_origin[2,0]
            return np.array_equal(y[1:3].view(np.uint32),self.coarse_origin[1:3].view(np.uint32))
    gate=Crease(x)
    y,r=solve(x,t,f,c,target,embedded_surface=gate,backtracking_domain_labels=np.asarray([0,0,1,1]))
    assert r['construction_backtracking_domain_fallback_attempted'] and not r['construction_backtracking_domain_fallback_accepted']
    assert not r['step_applied']; np.testing.assert_array_equal(y.view(np.uint32),x.view(np.uint32))
    assert r['construction_backtracking_global_final_hard_gates_pass']
    assert not r['construction_backtracking_global_final_contact_deficit_reduced']
    assert all(a['first_failed_gate']=='mapped_skin' for d in r['construction_backtracking_domain_attempts'] for a in d['attempts'])
    assert gate.calls>1

@pytest.mark.parametrize('labels',[[0,0],np.zeros(8),np.zeros(8,dtype=bool),np.asarray([0,0,0,0,0,0,0,-1])])
def test_invalid_labels_fail_closed(labels):
    x,t,f,c,target,_=fixture()
    with pytest.raises(ValueError,match='domain labels'): solve(x,t,f,c,target,backtracking_domain_labels=labels)

def test_default_and_single_domain_label_path_preserve_legacy_output():
    x,t,f,c,target,_=fixture(); ya,a=solve(x,t,f,c,target); yb,b=solve(x,t,f,c,target,backtracking_domain_labels=np.zeros(len(x),dtype=np.int64))
    for c in b['components']:
        c.pop('initial_full_step_first_failed_gate',None)
    for k in ('construction_backtracking_domain_labels_enabled','construction_backtracking_domain_label_count','construction_backtracking_partition_is_not_physical_ownership','construction_backtracking_global_final_hard_gates_pass','construction_backtracking_global_final_contact_deficit_reduced','construction_backtracking_global_final_contact_deficit_l1_mm','construction_backtracking_global_final_hard_linear_min_slack_m','construction_backtracking_full_component_alpha1_failed','construction_backtracking_domain_fallback_attempted','construction_backtracking_domain_fallback_candidate_valid','construction_backtracking_domain_fallback_accepted','construction_backtracking_selected_path','construction_backtracking_domain_attempts'): b.pop(k,None)
    np.testing.assert_array_equal(ya.view(np.uint32),yb.view(np.uint32))
    for c in b['components']:
        for k in ('initial_full_step_first_failed_gate','selected_backtracking_path','domain_fallback_attempted','domain_fallback_accepted','domain_attempts'): c.pop(k,None)
    assert a['lp_success']==b['lp_success']
    assert a['components']==b['components']
    assert a['contact_deficit_l1_mm']==b['contact_deficit_l1_mm']
    assert a['step_applied']==b['step_applied']
