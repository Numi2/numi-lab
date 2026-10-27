#!/usr/bin/env python3
"""Separate source-and-depth proposal policy; current-normal mismatch remains diagnostic."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
OWNER589=ROOT/'D/surface-clearance-546/source-outward-targets-589/run_depthmax_targets589_v2.py'
OWNER589_SHA='cd64f5280dec578ef8441aa9e44b9c01d44ba703edbd3fde39167ed471c71200'
BUILDER=ROOT/'D/contained-clearance-545/build_direct_depth_targets545.py'
BUILDER_SHA='2864d9547d2715c61c9c27b17bcfd11fac250197c47f4fd4870f8f6898cecb60'
EXT=ROOT/'D/source-outward-proposal-500/build_outward_targets_multi_source_546.py'
EXT_SHA='dd61f0fb87f12c132166515a1e058e144d493a835cadbb2a47e6a2c8574e1e68'
OWNER500=ROOT/'D/source-outward-proposal-500/build_outward_nearest_union_targets_500.py'
OWNER500_SHA='454501303db7c6420ff3744729c62c723c208aaf9348a53534137c62b7e6afbf'
BASE589=ROOT/'D/surface-clearance-546/source-outward-targets-589/run_source_outward_targets589.py'
BASE589_SHA='11f8a4a4b906294e9832173b63601f438c9c4bf1f57e488e8fd95f398f0038ec'
NORMAL588=ROOT/'D/surface-clearance-546/source-normal-audit-588/audit_midpoint_normals588.py'
NORMAL588_SHA='a6f7b8c1cd333e117f9a8f6f31bb9cc40f8bf5569c6f01ea7cc2074a99c3b703'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def loadmod(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(f'cannot load {path}')
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def validate_current_normal_diagnostic(audit,candidate_sha,expected_rows):
    if audit.get('candidate_sha256')!=candidate_sha: raise ValueError('current-normal diagnostic is bound to another candidate')
    rows=audit.get('rows');n=int(expected_rows)
    if not isinstance(rows,list) or len(rows)!=n or audit.get('target_row_count')!=n: raise ValueError('current-normal diagnostic row coverage mismatch')
    q=[int(r['query_row']) for r in rows]
    if len(set(q))!=n: raise ValueError('current-normal diagnostic has duplicate query rows')
    passed=sum(r.get('status')=='PASS' for r in rows)
    disagreed=sum(r.get('status')=='CURRENT_SOURCE_OUTWARD_DISAGREEMENT' for r in rows)
    unresolved=n-passed-disagreed
    if passed!=audit.get('agreement_count') or disagreed!=audit.get('disagreement_count') or unresolved: raise ValueError('current-normal diagnostic accounting is inconsistent')
    if disagreed and audit.get('status')!='CURRENT_SOURCE_OUTWARD_DISAGREEMENT': raise ValueError('disagreement audit status mismatch')
    if not disagreed and audit.get('status')!='PASS': raise ValueError('agreement audit status mismatch')
    dots=[float(r['current_source_outward_dot']) for r in rows if 'current_source_outward_dot' in r]
    return {'status':audit['status'],'rows':n,'agreement_count':passed,'disagreement_count':disagreed,'minimum_dot':min(dots) if dots else None}
def validate_row_identity(target_rows,diagnostic_rows):
    tq=np.asarray(target_rows);dq=np.asarray(diagnostic_rows)
    if tq.ndim!=1 or dq.ndim!=1 or len(np.unique(tq))!=len(tq) or len(np.unique(dq))!=len(dq): raise ValueError('target/diagnostic row identities must be unique vectors')
    if not np.array_equal(np.sort(tq.astype(np.int64)),np.sort(dq.astype(np.int64))): raise ValueError('target rows do not match current-normal diagnostic rows')
    return True
def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    ap=argparse.ArgumentParser(add_help=False);ap.add_argument('--diagnostic-current-source-audit',type=Path,required=True)
    extra,builder_argv=ap.parse_known_args(argv)
    helper=loadmod(OWNER589,'depthmax_target_validation589_for606')
    for path,expected,label in ((OWNER589,OWNER589_SHA,'589 validator'),(BUILDER,BUILDER_SHA,'545 builder'),(EXT,EXT_SHA,'546 extension'),(OWNER500,OWNER500_SHA,'500 source-normal owner'),(BASE589,BASE589_SHA,'589 baseline adapter'),(NORMAL588,NORMAL588_SHA,'588 diagnostic owner')):
        if sha(path)!=expected: raise ValueError(f'pinned {label} changed')
    helper.require_depth_maximizing(builder_argv)
    pa=argparse.ArgumentParser(add_help=False)
    pa.add_argument('--candidate',type=Path,required=True);pa.add_argument('--point-summary',type=Path,required=True)
    pa.add_argument('--query-dir',type=Path,required=True);pa.add_argument('--output-dir',type=Path,required=True)
    pa.add_argument('--point-depth-rows-npz',type=Path,required=True);pa.add_argument('--point-depth-rows-report',type=Path,required=True)
    cli,_=pa.parse_known_args(builder_argv)
    candidate=cli.candidate.resolve();out=cli.output_dir.resolve();diag_path=extra.diagnostic_current_source_audit.resolve()
    if out.exists(): raise FileExistsError(out)
    summary=json.loads(cli.point_summary.read_text());candidate_sha=sha(candidate)
    if candidate_sha!=summary.get('candidate_sha256'): raise ValueError('candidate/point-summary hash mismatch')
    expected_rows=int(summary.get('inside_nodes',-1))
    if expected_rows<=0: raise ValueError('point summary has no inside rows')
    depth_report=json.loads(cli.point_depth_rows_report.read_text())
    if depth_report.get('candidate_sha256')!=candidate_sha or depth_report.get('status')!='PASS_POINT_DEPTH_ROWS_BASELINE_FEASIBLE':
        raise ValueError('413 point-depth report is not matched and baseline-feasible')
    if sha(cli.point_depth_rows_npz)!=depth_report.get('output_npz_sha256') or int(depth_report.get('inside_row_count',-1))!=expected_rows:
        raise ValueError('413 point-depth rows NPZ/report mismatch')
    diag=json.loads(diag_path.read_text());diagnostic=validate_current_normal_diagnostic(diag,candidate_sha,expected_rows)
    diag_rows=np.asarray([int(r['query_row']) for r in diag['rows']],dtype=np.int64)
    extension=loadmod(EXT,'multi_parent_source_normals_546_for_606')
    if sha(extension.OWNER500)!=OWNER500_SHA or extension.OWNER500_SHA!=OWNER500_SHA: raise ValueError('546 extension owner pin mismatch')
    builder=loadmod(BUILDER,'direct_depth_target_builder545_for_606')
    face_provenance={};original_loadmod=builder.loadmod
    def load_with_multisource(path,name):
        module=original_loadmod(path,name)
        if Path(path).resolve()==Path(builder.SOURCE303).resolve(): helper.install_source303_extension(module,extension,face_provenance)
        return module
    builder.loadmod=load_with_multisource
    builder.main(builder_argv)
    report_path=out/'nearest-union-exits.json';npz_path=out/'nearest-union-exits.npz'
    if not report_path.is_file() or not npz_path.is_file(): raise RuntimeError('545 did not emit complete target report and NPZ')
    report=json.loads(report_path.read_text());archive=np.load(npz_path,allow_pickle=False)
    target_validation=helper.validate_545_depth_maximizing_report(report,archive)
    q=np.asarray(archive['query_rows'],dtype=np.int64);validate_row_identity(q,diag_rows)
    source=np.asarray(archive['source_surface_normal_world_unit'],dtype=np.float64);depth=np.asarray(archive['depth_covector_world_unit'],dtype=np.float64);proposal=np.asarray(archive['proposal_direction_world_unit'],dtype=np.float64)
    source=source/np.linalg.norm(source,axis=1)[:,None];depth=depth/np.linalg.norm(depth,axis=1)[:,None];proposal=proposal/np.linalg.norm(proposal,axis=1)[:,None]
    source_dots=np.einsum('ij,ij->i',source,proposal);depth_dots=np.einsum('ij,ij->i',depth,proposal)
    if len(q)!=expected_rows or np.min(source_dots)<.05-1e-10 or np.min(depth_dots)<.05-1e-10: raise ValueError('source/depth cosine floor or row coverage failed')
    serial=json.dumps(face_provenance,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    report['proposal_policy_606']={'status':'PASS_SOURCE_AND_DEPTH_TARGETS_ONLY','direction_mode':'depth-maximizing','minimum_source_cosine':.05,'minimum_depth_cosine':.05,
      'current_source_normal_agreement_required':False,'current_source_normal_agreement_claimed':False,
      'current_source_normal_disagreement_diagnostic':{'path':str(diag_path),'sha256':sha(diag_path),**diagnostic},
      'source_depth_target_count':int(len(q)),'actual_minimum_source_dot':float(source_dots.min()),'actual_minimum_depth_dot':float(depth_dots.min()),
      'full_185_bank_first_union_exit_validation':target_validation,'geometry_admitted':False,'physical_steps':0,'native_admission':False}
    report['multi_parent_source_normal_extension']={'adapter_path':str(Path(__file__).resolve()),'adapter_sha256':sha(__file__),'pinned_545_builder_path':str(BUILDER),'pinned_545_builder_sha256':BUILDER_SHA,'pinned_546_extension_path':str(EXT),'pinned_546_extension_sha256':EXT_SHA,
      'pinned_500_owner_path':str(OWNER500),'pinned_500_owner_sha256':OWNER500_SHA,'single_parent_303_function_preserved':True,
      'multi_parent_face_rows':sorted(face_provenance),'multi_parent_face_count':len(face_provenance),'multi_parent_provenance_sha256':hashlib.sha256(serial).hexdigest(),
      'exact_185_bank_ray_and_endpoint_owner_unchanged':True}
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':report.get('status'),'proposal_policy_606':report['proposal_policy_606'],'report_path':str(report_path),'report_sha256':sha(report_path),'target_npz_path':str(npz_path),'target_npz_sha256':sha(npz_path)},indent=2))
if __name__=='__main__': main()
