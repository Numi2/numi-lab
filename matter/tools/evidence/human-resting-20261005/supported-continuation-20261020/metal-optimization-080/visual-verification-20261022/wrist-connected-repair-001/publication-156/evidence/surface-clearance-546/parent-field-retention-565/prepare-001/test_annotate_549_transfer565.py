from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('annotate_549_transfer565_test',HERE/'annotate_549_transfer565.py')
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_json(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def fixture(tmp_path):
    base=tmp_path/'base.npz';target=tmp_path/'target.npz';out=tmp_path/'out.npz'
    np.savez_compressed(base,points_world_m=np.array([[0,0,0],[1,0,0]],dtype=np.float32),
                        source_O_faces=np.array([[0,1,0]],dtype=np.int32),
                        typed_coarse_anchor_output_rows_018=np.array([0],dtype=np.int32))
    np.savez_compressed(target,points_world_m=np.array([[0,0,0],[1,0.001,0]],dtype=np.float32),
                        source_O_faces=np.array([[0,1,0]],dtype=np.int32),
                        typed_coarse_anchor_output_rows_018=np.array([0],dtype=np.int32))
    np.savez_compressed(out,points_world_m=np.array([[0,0,0],[1,0.0005,0]],dtype=np.float32),
                        source_O_faces=np.array([[0,1,0]],dtype=np.int32),
                        typed_coarse_anchor_output_rows_018=np.array([0],dtype=np.int32))
    descriptor={
      'schema':'numi.human.generic_coordinate_transfer_for_549.v1',
      'status':'PREPARED_INPUT_DESCRIPTOR_ONLY_UNADMITTED',
      'source_kind':'transferred coordinate correction from saved 031 proposal applied to 6ce baseline',
      'not_a_raw_lp_candidate':True,'raw_lp_feasibility_claim':False,
      'base_candidate':str(base),'base_sha256':sha(base),
      'coordinate_target_candidate':str(target),'coordinate_target_sha256':sha(target),
      'free_nodes':[1],
      '549_compatibility':{'backtracker_sha256':mod.EXPECTED_549_SHA},
    }
    desc=tmp_path/'descriptor.json';write_json(desc,descriptor)
    raw={
      'source_kind':descriptor['source_kind'],'raw_lp_feasibility_claim':False,
      'base_candidate':str(base),'base_sha256':sha(base),
      'candidate_path':str(target),'candidate_sha256':sha(target),'free_nodes':[1],
    }
    rawdir=tmp_path/'raw-dir';rawdir.mkdir();rawpath=rawdir/'report.json'
    import shutil
    shutil.copyfile(target,rawdir/'candidate.npz')
    write_json(rawpath,raw)
    report={
      'status':'CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED','base_candidate':str(base),
      'base_sha256':sha(base),'candidate_path':str(out),'candidate_sha256':sha(out),
      'free_nodes':[1],'scope':'offline retained raw LP direction, independent nonlinear-stencil component alpha; unchanged actual545 gates',
      'full_final_gate':{'area_pass':True,'maximum_step_mm':0.5},
      'native_admission':False,'accepted_physical_steps':0,
    }
    reportpath=tmp_path/'549-report.json';write_json(reportpath,report)
    original=reportpath.read_bytes()
    return desc,rawpath,reportpath,original,out

def test_annotation_preserves_original_report_and_relabels_source_only(tmp_path):
    desc,rawpath,reportpath,original,out=fixture(tmp_path)
    result=mod.annotate(reportpath,rawpath,desc)
    annotated=json.loads(reportpath.read_text())
    backup=reportpath.with_name('549-report.pre-565-annotation.json')
    receipt=reportpath.with_name('549-report.source-annotation-receipt.json')
    assert backup.read_bytes()==original
    assert annotated['original_549_scope_label']['value'].startswith('offline retained raw LP')
    assert 'transferred coordinate correction' in annotated['scope']
    assert annotated['source_method_annotation_565']['raw_lp_feasibility_claim'] is False
    assert annotated['full_final_gate']=={'area_pass':True,'maximum_step_mm':0.5}
    assert result['candidate_sha256']==sha(out)
    assert json.loads(receipt.read_text())['full_545_exact_check_required'] is True

def test_rejects_lp_claim_or_mismatched_source_kind(tmp_path):
    desc,rawpath,reportpath,_,_=fixture(tmp_path)
    raw=json.loads(rawpath.read_text());raw['raw_lp_feasibility_claim']=True;write_json(rawpath,raw)
    with np.testing.assert_raises(ValueError):mod.annotate(reportpath,rawpath,desc)
    raw['raw_lp_feasibility_claim']=False;raw['source_kind']='raw LP output';write_json(rawpath,raw)
    with np.testing.assert_raises(ValueError):mod.annotate(reportpath,rawpath,desc)


def test_rejects_base_target_or_free_rows_mismatch(tmp_path):
    desc,rawpath,reportpath,_,_=fixture(tmp_path)
    raw=json.loads(rawpath.read_text());raw['free_nodes']=[];write_json(rawpath,raw)
    with np.testing.assert_raises(ValueError):mod.annotate(reportpath,rawpath,desc)
