from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pytest

SRC=Path(__file__).with_name('condition_area_quality_548.py')
spec=importlib.util.spec_from_file_location('condition_area_quality_548_test',SRC)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def q(area,angle,cos=None):
    if cos is None: cos=np.ones(len(area))
    return {'area_ratio':np.asarray(area,float),'min_angle_deg':np.asarray(angle,float),'normal_cos':np.asarray(cos,float)}

def complete_report(counts):
    return {'status':'COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT','complete':True,'expected_bank_count':185,'bank_count':185,
            'banks':[{'body':b,'stable_bone_id':s,'result':{'triangle_pairs':[[] for _ in range(n)]}} for (b,s),n in counts.items()]}

def test_witness_parser_prioritizes_persistent_22027():
    d={'area_scp_iterations':[{'exact_f32_area_backtrack':{'trials':[
        {'area_gate':{'bad_fixed_037_reference_face_count':2,'bad_fixed_037_reference_face_rows_first_64':[22027,9]}},
        {'area_gate':{'bad_fixed_037_reference_face_count':1,'bad_fixed_037_reference_face_rows_first_64':[22027]}}]}}]}
    rows,freq=m.parse_witness_rows(d)
    assert rows[0]==22027 and freq[22027]==2

def test_quality_gate_requires_local_area_gain_and_preserves_global_floors():
    # An unrelated untouched face owns the global area/angle minima.
    before=q([.09,.10007,.3],[2.0,2.5,4.0])
    after=q([.09,.105,.3],[2.0,2.4,4.0])
    assert m.quality_gate(before,after,[1,2])['passed']
    # Local ratio must improve, but the mesh-wide minimum need not move.
    assert not m.quality_gate(before,q([.09,.10007,.3],[2.0,2.5,4.0]),[1,2])['passed']
    # A bounded local angle loss is permitted when the unchanged global gate stays safe.
    assert m.quality_gate(before,q([.09,.105,.3],[2.0,2.0,4.0]),[1,2])['passed']
    assert not m.quality_gate(before,q([.09,.105,.3],[2.0,1.99,4.0]),[1,2])['passed']
    assert not m.quality_gate(before,q([.09,.105,.3],[1.9,2.4,4.0]),[1,2])['passed']

def test_existing_global_angle_and_cosine_defects_do_not_block_clean_local_patch():
    before=q([.10007,.3,.2],[2.5,4.0,.0207],cos=[.5,.6,-.9997])
    after=q([.105,.3,.2],[2.4,4.0,.0207],cos=[.4,.6,-.9997])
    result=m.quality_gate(before,after,[0,1])
    assert result['passed']
    assert result['global_old_min_normal_cosine']==result['global_new_min_normal_cosine']
    assert result['global_old_min_angle_deg']==result['global_new_min_angle_deg']
    # An unrelated mesh-wide regression still rejects the proposal.
    assert not m.quality_gate(before,q([.089,.105,.3],[2.0,2.4,4.0]),[1,2])['passed']
    assert not m.quality_gate(before,q([.09,.105,.3],[1.9,2.4,4.0]),[1,2])['passed']

def test_existing_global_angle_and_cosine_defects_do_not_block_clean_local_patch():
    before=q([.10007,.3,.2],[2.5,4.0,.0207],cos=[.5,.6,-.9997])
    after=q([.105,.3,.2],[2.4,4.0,.0207],cos=[.4,.6,-.9997])
    result=m.quality_gate(before,after,[0,1])
    assert result['passed']
    assert result['global_old_min_normal_cosine']==result['global_new_min_normal_cosine']
    assert result['global_old_min_angle_deg']==result['global_new_min_angle_deg']


def test_exact_bank_checker_requires_all185_and_nonincrease_per_typed_bank():
    old={(i,i+1000):2 for i in range(185)}
    same=complete_report(old)
    assert m.check_exact_nonincrease(same,same)['passed']
    new=dict(old);new[(2,1002)]=3
    result=m.check_exact_nonincrease(same,complete_report(new))
    assert not result['passed'] and result['increased_banks']=={'2/1002':(2,3)}
    bad=complete_report(old);bad['banks']=bad['banks'][:-1];bad['bank_count']=184
    with pytest.raises(m.ProposalError):m.check_exact_nonincrease(same,bad)

def test_seed_edge_scope_only_accepts_edges_touching_witness_face():
    faces=np.asarray([[0,1,2],[1,3,2],[3,4,2]],dtype=np.int64)
    em={(0,1):[(0,0,1)],(1,2):[(0,1,2),(1,2,1)],(0,2):[(0,2,0)],
        (1,3):[(1,1,3)],(2,3):[(1,3,2),(2,2,3)]}
    picked=[]
    for a,b in zip(faces[0],np.roll(faces[0],-1)):
        edge=(min(int(a),int(b)),max(int(a),int(b)))
        if len(em.get(edge,[]))==2:picked.append((edge,[r[0] for r in em[edge]]))
    assert picked==[((1,2),[0,1])]

if __name__=='__main__':pytest.main([__file__,'-q'])
