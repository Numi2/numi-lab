#!/usr/bin/env python3
"""Corrected emitted-F32 direction receipt for source-depth proposal 606; no rerays."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
OWNER589=ROOT/'D/surface-clearance-546/source-outward-targets-589/run_depthmax_targets589_v2.py'
OWNER589_SHA='cd64f5280dec578ef8441aa9e44b9c01d44ba703edbd3fde39167ed471c71200'
CANDIDATE_SHA='16a7f33f4b3d3ea7cf61bf55d01a1dca00fe9a32ee2cb7c4fa5f930531bb674e'
V1_REPORT_SHA='57ddfe11e0221803f99a050baa7eee9fd599994996cc7760236c30f7a2a33ecb'
TARGET_NPZ_SHA='c8fcb14c6f1a280a2f264567896a98e2111f717f6fc761aa206125a656f18a58'
ADAPTER606_SHA='e43cb5672f98f0b39636ed2484d481f61d9a80932846732e0d677107d2b4a1e0'
BASE=ROOT/'D/surface-clearance-546/connected-refinement-603/baseline-599-affine-filtered-002/candidate.npz'
V1=ROOT/'D/surface-clearance-546/source-depth-targets-606/targets-603-filtered-001/nearest-union-exits.json'
NPZ=ROOT/'D/surface-clearance-546/source-depth-targets-606/targets-603-filtered-001/nearest-union-exits.npz'
OUT=ROOT/'D/surface-clearance-546/source-depth-targets-606/targets-603-filtered-001/nearest-union-exits.emitted-f32-v2.json'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def loadmod(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if spec is None or spec.loader is None: raise ImportError(str(path))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def unit_rows(a,label):
    x=np.asarray(a,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=3 or not np.isfinite(x).all(): raise ValueError(label+' must be finite Nx3')
    n=np.linalg.norm(x,axis=1)
    if np.any(n<=0): raise ValueError(label+' has zero vectors')
    return x/n[:,None]
def emitted_direction_audit(base_points,target_f32,nominal,source,depth):
    b=np.asarray(base_points);t=np.asarray(target_f32)
    if b.dtype!=np.float32 or t.dtype!=np.float32 or b.shape!=t.shape or b.ndim!=2 or b.shape[1]!=3: raise ValueError('base and target must be matching Float32 Nx3 arrays')
    delta=t.astype(np.float64)-b.astype(np.float64)
    lengths=np.linalg.norm(delta,axis=1)
    if np.any(lengths<=0): raise ValueError('emitted F32 target has zero displacement')
    actual=delta/lengths[:,None]
    nom=unit_rows(nominal,'nominal directions');src=unit_rows(source,'source normals');dep=unit_rows(depth,'depth covectors')
    if not(len(actual)==len(nom)==len(src)==len(dep)): raise ValueError('direction array count mismatch')
    ns=np.einsum('ij,ij->i',nom,src);nd=np.einsum('ij,ij->i',nom,dep)
    ars=np.einsum('ij,ij->i',actual,src);ard=np.einsum('ij,ij->i',actual,dep)
    if np.any(ars<=0) or np.any(ard<=0): raise ValueError('emitted F32 directions are not positive against both source and depth vectors')
    stats=lambda x:{'minimum':float(x.min()),'count_below_0_05':int(np.sum(x<.05)),'count_nonpositive':int(np.sum(x<=0))}
    return {'algorithm':'normalize(Float64(target_world_f32_m) - Float64(candidate.points_world_m[query_rows]))',
      'nominal_proposal_direction':{'source_dot':stats(ns),'depth_dot':stats(nd),'source_depth_floor':.05},
      'emitted_float32_displacement_direction':{'source_dot':stats(ars),'depth_dot':stats(ard),'all_source_dots_positive':bool(np.all(ars>0)),'all_depth_dots_positive':bool(np.all(ard>0)),'actual_0_05_floor_claimed':False},
      'displacement_mm':{'minimum':float(lengths.min()*1000.0),'maximum':float(lengths.max()*1000.0)}}
def validate_report_target_emission(report,archive,base_points):
    q=np.asarray(archive['query_rows'])
    t=np.asarray(archive['target_world_f32_m'])
    if q.ndim!=1 or q.dtype.kind not in 'iu' or len(np.unique(q))!=len(q): raise ValueError('target query rows invalid')
    base=np.asarray(base_points)
    if np.any(q<0) or np.any(q>=len(base)): raise ValueError('target row outside candidate points')
    records=report.get('targets')
    if not isinstance(records,list) or len(records)!=len(q): raise ValueError('report target rows incomplete')
    for i,(row,rec) in enumerate(zip(q,records)):
        if int(rec.get('query_row',-1))!=int(row): raise ValueError('report/NPZ query row order mismatch')
        if not np.array_equal(np.asarray(rec.get('target_world_f32_m'),dtype=np.float32),t[i]): raise ValueError('report/NPZ emitted target mismatch')
        if not np.array_equal(np.asarray(rec.get('start_world_f32_m'),dtype=np.float32),base[int(row)]): raise ValueError('report/base start mismatch')
    return q,t
def build_report(base_path=BASE,v1_path=V1,npz_path=NPZ,out_path=OUT):
    for p,expected,label in ((OWNER589,OWNER589_SHA,'589 validator'),(base_path,CANDIDATE_SHA,'candidate'),(v1_path,V1_REPORT_SHA,'v1 report'),(npz_path,TARGET_NPZ_SHA,'target NPZ')):
        if sha(p)!=expected: raise ValueError(label+' hash mismatch')
    if sha(ROOT/'D/surface-clearance-546/source-depth-targets-606/build_source_depth_targets606.py')!=ADAPTER606_SHA: raise ValueError('606 adapter hash mismatch')
    if out_path.exists(): raise FileExistsError(out_path)
    report=json.loads(v1_path.read_text());z=np.load(npz_path,allow_pickle=False);base_npz=np.load(base_path,allow_pickle=False)
    helper=loadmod(OWNER589,'depthmax_target_validation589_for606_v2')
    validation=helper.validate_545_depth_maximizing_report(report,z)
    q,t=validate_report_target_emission(report,z,base_npz['points_world_m'])
    stats=emitted_direction_audit(base_npz['points_world_m'][q],t,z['proposal_direction_world_unit'],z['source_surface_normal_world_unit'],z['depth_covector_world_unit'])
    report['proposal_policy_606']={'status':'PASS_SOURCE_AND_DEPTH_NOMINAL_TARGETS_ONLY','direction_mode':'depth-maximizing',
      'nominal_minimum_source_cosine':stats['nominal_proposal_direction']['source_dot']['minimum'],
      'nominal_minimum_depth_cosine':stats['nominal_proposal_direction']['depth_dot']['minimum'],
      'nominal_source_depth_floor':.05,'emitted_float32_direction_audit':stats['emitted_float32_displacement_direction'],
      'emitted_float32_displacement_mm':stats['displacement_mm'],
      'current_source_normal_agreement_required':False,'current_source_normal_agreement_claimed':False,
      'current_source_normal_disagreement_diagnostic':report.get('proposal_policy_606',{}).get('current_source_normal_disagreement_diagnostic'),
      'all185_first_union_exit_validation':validation,'geometry_admitted':False,'physical_steps':0,'native_admission':False}
    report['emitted_direction_correction_606_v2']={'classification':'arithmetic correction of direction metadata only; no target arrays or ray certificates changed',
      'parent_v1_report_path':str(v1_path),'parent_v1_report_sha256':sha(v1_path),'target_npz_path':str(npz_path),'target_npz_sha256':sha(npz_path),
      'candidate_path':str(base_path),'candidate_sha256':sha(base_path),'correction_helper_path':str(Path(__file__).resolve()),'correction_helper_sha256':sha(__file__),
      'source_606_adapter_sha256':ADAPTER606_SHA}
    out_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return {'output_path':str(out_path),'output_sha256':sha(out_path),'target_npz_sha256':sha(npz_path),'candidate_sha256':sha(base_path),
      'target_count':len(q),'direction_audit':stats,'endpoint_validation':validation}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=OUT);ns=ap.parse_args()
    print(json.dumps(build_report(out_path=ns.output),indent=2))
if __name__=='__main__': main()
