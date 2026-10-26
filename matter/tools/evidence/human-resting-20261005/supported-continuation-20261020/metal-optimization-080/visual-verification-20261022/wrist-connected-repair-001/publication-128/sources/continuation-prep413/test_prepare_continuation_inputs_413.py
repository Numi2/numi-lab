import importlib.util, json, hashlib
from pathlib import Path
import numpy as np

SOURCE=Path('/Users/numi/Work/numi-human-connected-surface-069/D/prepare_connected_continuation_inputs_413.py')
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

def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_npy(path, arr):
    with Path(path).open('wb') as f: np.save(f,arr,allow_pickle=False)

def union_fixture(tmp_path, monkeypatch, *, pair_hash_alias='union_pair_ids_sha256'):
    candidate=tmp_path/'candidate.npz'; candidate.write_bytes(b'candidate-bytes')
    a130=tmp_path/'a130.json'; a130.write_text('{}')
    a159=tmp_path/'a159.json'; a159.write_text('{}')
    owner=tmp_path/'owner.py'; owner.write_text('owner')
    monkeypatch.setattr(mod,'OWNER026',owner)
    monkeypatch.setitem(mod.PINS,'owner026',sha_file(owner))
    pairs_path=tmp_path/'pair-ids.npy'
    pairs=np.array([[0,1],[1,2]],dtype=np.int64); write_npy(pairs_path,pairs)
    payload=tmp_path/'rows.npz'
    np.savez(payload,
        row_indptr=np.array([0,1,2],dtype=np.int64),
        row_node_ids=np.array([0,1],dtype=np.int32),
        row_coefficients=np.array([[1.,0.,0.],[0.,1.,0.]],dtype=np.float64),
        row_lower_mm=np.array([-0.1,-0.2],dtype=np.float64),
        row_pair_ids=pairs.astype(np.int32))
    report={
      'schema':mod.PROSPECTIVE_UNION_SCHEMA,'status':'COMPLETE',
      'all_union_pair_ids_classified':True,'candidate_path':str(candidate),
      'candidate_sha256':sha_file(candidate),
      'candidate_exact_self_audit_path':str(a130),'candidate_exact_self_audit_sha256':sha_file(a130),
      'candidate_exact_self_pair_count':0,
      'candidate_surface_audit_path':str(a159),'candidate_surface_audit_sha256':sha_file(a159),
      'candidate_surface_audit_complete':True,'candidate_surface_bank_count':185,
      'candidate_surface_pair_count':1,
      'union_pair_count':2,'compatible_pair_count':2,'processed_pair_count':2,
      'incompatible_pair_count':0,'unprocessed_pair_count':0,'seed_ids_dropped':False,
      'positive_rhs_within_tolerance_count':0,'physical_steps':0,'geometry_admitted':False,
      'zero_feasibility_tolerance_mm':2e-8,'max_baseline_rhs_mm':-0.1,'emitted_row_count':2,
      'row_owner_path':str(owner),'row_owner_sha256':sha_file(owner),
      'row_payload_path':str(payload),'row_payload_sha256':sha_file(payload),
      'union_pair_ids_path':str(pairs_path),'union_pair_ids_sha256':sha_file(pairs_path)}
    if pair_hash_alias=='output_pairs_sha256':
        report['output_pairs_path']=report.pop('union_pair_ids_path')
        report['output_pairs_sha256']=report.pop('union_pair_ids_sha256')
    rp=tmp_path/'union-report.json'; rp.write_text(json.dumps(report))
    return {'report_path':rp,'report_sha256':sha_file(rp),'pairs_path':pairs_path,
      'pairs_sha256':sha_file(pairs_path),'candidate':candidate,'a130':a130,'a159':a159,
      'owner':owner,'pairs':pairs,'report':report}

def load_union_fixture(f):
    return mod.load_prior_seed_bundle(f['report_path'],f['report_sha256'],f['pairs_path'],f['pairs_sha256'],
      current_candidate_path=f['candidate'],current_candidate_sha256=sha_file(f['candidate']),
      audit130_path=f['a130'],audit130_sha256=sha_file(f['a130']),
      audit159_path=f['a159'],audit159_sha256=sha_file(f['a159']),
      current_surface_pair_count=1,face_count=4)

def test_complete_prospective_union_schema_and_payload_are_accepted(tmp_path,monkeypatch):
    f=union_fixture(tmp_path,monkeypatch)
    got=load_union_fixture(f)
    assert got['schema']==mod.PROSPECTIVE_UNION_SCHEMA
    assert got['validation_mode']=='complete-prospective-union'
    assert np.array_equal(got['pairs'],f['pairs'])

def test_prospective_union_accepts_legacy_output_pairs_hash_alias(tmp_path,monkeypatch):
    f=union_fixture(tmp_path,monkeypatch,pair_hash_alias='output_pairs_sha256')
    got=load_union_fixture(f)
    assert got['validation_mode']=='complete-prospective-union'
    assert np.array_equal(got['pairs'],f['pairs'])

def test_prospective_union_rejects_incomplete_or_unprocessed_report(tmp_path,monkeypatch):
    import pytest
    f=union_fixture(tmp_path,monkeypatch)
    f['report']['unprocessed_pair_count']=1
    f['report_path'].write_text(json.dumps(f['report']))
    f['report_sha256']=sha_file(f['report_path'])
    with pytest.raises(ValueError,match='incomplete|incompatible or unprocessed'):
        load_union_fixture(f)

def test_prospective_union_rejects_unknown_schema(tmp_path,monkeypatch):
    import pytest
    f=union_fixture(tmp_path,monkeypatch)
    f['report']['schema']='numi.human.prospective-self-guards.unknown.v9'
    f['report_path'].write_text(json.dumps(f['report']))
    f['report_sha256']=sha_file(f['report_path'])
    with pytest.raises(ValueError,match='unknown prior seed schema'):
        load_union_fixture(f)

def test_prospective_union_rejects_candidate_or_pair_hash_binding_mismatch(tmp_path,monkeypatch):
    import pytest
    f=union_fixture(tmp_path,monkeypatch)
    f['report']['candidate_sha256']='0'*64
    f['report_path'].write_text(json.dumps(f['report']))
    f['report_sha256']=sha_file(f['report_path'])
    with pytest.raises(ValueError,match='candidate binding'):
        load_union_fixture(f)
    second=tmp_path/'second'; second.mkdir()
    f=union_fixture(second,monkeypatch)
    f['report']['union_pair_ids_sha256']='f'*64
    f['report_path'].write_text(json.dumps(f['report']))
    f['report_sha256']=sha_file(f['report_path'])
    with pytest.raises(ValueError,match='pair file hash'):
        load_union_fixture(f)

def test_prospective_union_rejects_nonzero_or_report_only_zero_step_claim(tmp_path,monkeypatch):
    import pytest
    f=union_fixture(tmp_path,monkeypatch)
    f['report']['max_baseline_rhs_mm']=1e-5
    f['report_path'].write_text(json.dumps(f['report']))
    f['report_sha256']=sha_file(f['report_path'])
    with pytest.raises(ValueError,match='zero-step feasibility'):
        load_union_fixture(f)

def test_legacy_seed_report_remains_accepted(tmp_path,monkeypatch):
    candidate=tmp_path/'old-candidate.npz'; candidate.write_bytes(b'old-candidate')
    pairs_path=tmp_path/'legacy-pairs.npy'
    pairs=np.array([[0,1],[1,2]],dtype=np.int64); write_npy(pairs_path,pairs)
    report={'schema':'numi.human.continuation-self-seeds.v1',
      'status':'PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY','candidate_path':str(candidate),
      'candidate_sha256':sha_file(candidate),'output_pairs_path':str(pairs_path),
      'output_pairs_sha256':sha_file(pairs_path),'merged_unique_pair_count':2,
      'zero_step_feasible':True,'zero_step_max_rhs_mm':-1e-5}
    rp=tmp_path/'legacy-report.json'; rp.write_text(json.dumps(report))
    got=mod.load_prior_seed_bundle(rp,sha_file(rp),pairs_path,sha_file(pairs_path),
      current_candidate_path=tmp_path/'next-candidate.npz',current_candidate_sha256='a'*64,
      audit130_path=tmp_path/'a130.json',audit130_sha256='b'*64,
      audit159_path=tmp_path/'a159.json',audit159_sha256='c'*64,
      current_surface_pair_count=1,face_count=4)
    assert got['validation_mode']=='legacy-output-pairs'
    assert np.array_equal(got['pairs'],pairs)

