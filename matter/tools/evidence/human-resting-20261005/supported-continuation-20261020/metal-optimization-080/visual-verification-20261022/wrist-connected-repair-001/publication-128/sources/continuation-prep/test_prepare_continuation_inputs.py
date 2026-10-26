import importlib.util, json
from pathlib import Path
import numpy as np

SOURCE=Path('/Users/numi/Work/numi-human-connected-surface-069/D/prepare_connected_continuation_inputs.py')
spec=importlib.util.spec_from_file_location('prep_continuation_test',SOURCE)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

class Z:
    def __init__(self,**kw): self.__dict__.update(kw); self.files=list(kw)
    def __getitem__(self,k): return self.__dict__[k]

def fixture159(null_inactive=False):
    banks=[]
    for i in range(185):
        result={'count':0,'triangle_pairs':[]}
        row={'body':i//20,'stable_bone_id':i,'skin_aabb_active':1,'bank_aabb_active':1,'result':result}
        if null_inactive and i==0:
            row.update(skin_aabb_active=0,bank_aabb_active=0,result=None)
        banks.append(row)
    return {'complete':True,'bank_count':185,'expected_bank_count':185,'banks':banks}

def test_count_null_result_allowed_only_for_inactive_bank():
    assert mod.count_surface_pairs(fixture159(True))==0
    x=fixture159(True); x['banks'][0]['bank_aabb_active']=1
    import pytest
    with pytest.raises(ValueError): mod.count_surface_pairs(x)

def test_count_requires_all_unique_185_typed_banks():
    import pytest
    x=fixture159(); x['banks'].pop()
    with pytest.raises(ValueError): mod.count_surface_pairs(x)
    x=fixture159(); x['banks'][1]['stable_bone_id']=x['banks'][0]['stable_bone_id']; x['banks'][1]['body']=x['banks'][0]['body']
    with pytest.raises(ValueError): mod.count_surface_pairs(x)

def test_same_topology_allows_only_registered_coordinate_derived_fields():
    a=Z(source_O_faces=np.array([[0,1,2]]),points_world_m=np.zeros((3,3)),
        node_parent_coarse_edge_parameter=np.zeros(3),refinement_face_parent_barycentric=np.zeros((1,3,3)),
        face_source_ancestry_indptr=np.array([0,1]),face_source_ancestry_face_rows=np.array([7]))
    b=Z(source_O_faces=np.array([[0,1,2]]),points_world_m=np.ones((3,3)),
        node_parent_coarse_edge_parameter=np.ones(3),refinement_face_parent_barycentric=np.ones((1,3,3)),
        face_source_ancestry_indptr=np.array([0,1]),face_source_ancestry_face_rows=np.array([7]))
    mod.same_topology_identity(a,b,'fixture')
    b.face_source_ancestry_face_rows[0]=8
    import pytest
    with pytest.raises(ValueError): mod.same_topology_identity(a,b,'fixture')

def test_pair_merge_is_sorted_unique_and_retains_prior():
    prior=np.array([[4,9],[1,3],[3,8]])
    failed=[np.array([[1,3],[2,7],[9,4]])]
    before,merged=mod.merge_pair_sets(prior,failed,10)
    assert before.tolist()==[[1,3],[3,8],[4,9]]
    assert merged.tolist()==[[1,3],[2,7],[3,8],[4,9]]

def test_pair_merge_rejects_invalid_rows():
    import pytest
    with pytest.raises(ValueError): mod.merge_pair_sets(np.array([[1,1]]),[],5)
    with pytest.raises(ValueError): mod.merge_pair_sets(np.array([[1,8]]),[],5)

def test_json_writer_handles_numpy_scalars(tmp_path):
    p=tmp_path/'receipt.json'
    mod.json_write(p,{'count':np.int64(3),'score':np.float64(.25),'flag':np.bool_(True),'rows':np.array([1,2])})
    assert json.loads(p.read_text())=={'count':3,'flag':True,'rows':[1,2],'score':.25}


def test_failed_manifest_410_schema_normalizes_and_pins(tmp_path):
    p=tmp_path/'failed.json'
    obj={'trials':[{'candidate':'c.npz','candidate_sha256':'c'*64,'point_summary':'s.json','point_summary_sha256':'s'*64,'self_pairs':'p.npy','self_pairs_sha256':'p'*64}]}
    p.write_text(json.dumps(obj))
    got=mod.load_failed_manifest(p,mod.sha(p))
    assert got==[{'candidate_path':'c.npz','candidate_sha256':'c'*64,'audit130_path':'s.json','audit130_sha256':'s'*64,'pairs_path':'p.npy','pairs_sha256':'p'*64}]
    import pytest
    with pytest.raises(ValueError): mod.load_failed_manifest(p,'0'*64)
