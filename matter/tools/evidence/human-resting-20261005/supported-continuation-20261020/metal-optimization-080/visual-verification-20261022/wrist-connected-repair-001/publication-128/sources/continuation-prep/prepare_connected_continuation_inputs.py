#!/usr/bin/env python3
"""Prepare same-topology connected-clearance inputs from pinned artifacts.
The tool emits construction targets and regenerated 026 pair identities only;
it does not fit geometry or grant physical/native admission.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
import numpy as np

D = Path('/Users/numi/Work/numi-human-connected-surface-069')
PYTHON = Path('/Users/n/numi-human-prep-venv-20261005/bin/python')
SLOT = Path('/tmp/numi-human-geometry-slot.py')
BUILDER335 = D/'barycentric-connected-clearance-170/source-outward-proposal-300/build_outward_nearest_union_targets_335.py'
BUILDER390 = D/'D/coherent-point-targets-390/build_coherent_point_targets_390.py'
OWNER026 = D/'barycentric-connected-clearance-170/connected-local-continuation-189-run001/round-005/proposal/round-001/fit/self_separation_026.py'
DEPTH357 = D/'D/point-depth-barriers-357/point_depth_barriers_357.py'
PINS = {
    'builder335': '1c3468b6e6285afed77e8ae7ca463abb797f6a22d562ccf878734c39161947c9',
    'builder390': '15be65c3f3863998a2fb709e70e22a20402ab1580e84960aeb1a3f2301c70b00',
    'owner026': 'c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3',
    'depth357': 'dc8078515dd12b2edc62c0bc75759a7cb9472c9914e4704ecd41e04c4d5378e1',
}
COORDINATE_DERIVED = {'points_world_m','node_parent_coarse_edge_parameter','refinement_face_parent_barycentric'}

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def json_load(path): return json.loads(Path(path).read_text())
def json_ready(obj):
    if isinstance(obj, np.generic): return obj.item()
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, Path): return str(obj)
    if isinstance(obj, dict): return {str(k):json_ready(v) for k,v in obj.items()}
    if isinstance(obj, (list,tuple)): return [json_ready(v) for v in obj]
    return obj
def json_write(path,obj):
    with Path(path).open('x') as f: json.dump(json_ready(obj),f,indent=2,sort_keys=True); f.write('\n')
def npy_write(path,a):
    with Path(path).open('xb') as f: np.save(f,np.asarray(a),allow_pickle=False)

def require_sha(path, expected, label):
    p=Path(path)
    if not p.is_file(): raise FileNotFoundError(f'{label} missing: {p}')
    got=sha(p)
    if got!=expected: raise ValueError(f'{label} SHA mismatch: expected {expected}, got {got}')
    return got

def count_surface_pairs(report):
    if not isinstance(report,dict) or report.get('complete') is not True:
        raise ValueError('159 report incomplete')
    banks=report.get('banks')
    if not isinstance(banks,list) or len(banks)!=185 or int(report.get('bank_count',-1))!=185 or int(report.get('expected_bank_count',-1))!=185:
        raise ValueError('159 must report exactly 185 typed banks')
    typed=set(); total=0
    for b in banks:
        key=(int(b['body']),int(b['stable_bone_id']))
        if key in typed: raise ValueError('duplicate typed bank in 159 report')
        typed.add(key); result=b.get('result')
        if result is None:
            if int(b.get('skin_aabb_active',-1))!=0 or int(b.get('bank_aabb_active',-1))!=0:
                raise ValueError('null result on active 159 bank')
            continue
        rows=result.get('triangle_pairs'); n=int(result.get('count',-1))
        if not isinstance(rows,list) or len(rows)!=n or n<0: raise ValueError('159 explicit count/rows mismatch')
        total+=n
    return total

def same_topology_identity(reference,current,label):
    if set(reference.files)!=set(current.files): raise ValueError(f'{label}: NPZ key set differs')
    for key in reference.files:
        if reference[key].shape!=current[key].shape or reference[key].dtype!=current[key].dtype: raise ValueError(f'{label}: dtype/shape changed for {key}')
        if key not in COORDINATE_DERIVED and np.ascontiguousarray(reference[key]).tobytes()!=np.ascontiguousarray(current[key]).tobytes():
            raise ValueError(f'{label}: non-coordinate/topology identity field changed: {key}')
    if reference['source_O_faces'].dtype!=current['source_O_faces'].dtype or np.ascontiguousarray(reference['source_O_faces']).tobytes()!=np.ascontiguousarray(current['source_O_faces']).tobytes():
        raise ValueError(f'{label}: source_O_faces differ')

def merge_pair_sets(prior, extras, face_count):
    arrays=[np.asarray(prior,dtype=np.int64)] + [np.asarray(x,dtype=np.int64) for x in extras]
    for a in arrays:
        if a.ndim!=2 or a.shape[1]!=2 or np.any(a<0) or np.any(a>=face_count) or np.any(a[:,0]==a[:,1]):
            raise ValueError('malformed or out-of-range self-pair identity rows')
    before=np.unique(np.sort(arrays[0],axis=1),axis=0)
    merged=np.unique(np.sort(np.concatenate(arrays,axis=0),axis=1),axis=0)
    if not set(map(tuple,before.tolist())).issubset(set(map(tuple,merged.tolist()))):
        raise ValueError('prior self-pair identities were not preserved')
    return before,merged

def load_failed_manifest(path, expected_sha256):
    require_sha(path,expected_sha256,'failed-trials manifest')
    obj=json_load(path)
    items=obj.get('trials') if isinstance(obj,dict) else None
    if items is None and isinstance(obj,dict): items=obj.get('failed_trials')
    if not isinstance(items,list): raise ValueError('failed-trials manifest malformed')
    normalized=[]
    for row in items:
        if not isinstance(row,dict): raise ValueError('failed-trial entry must be an object')
        normalized.append({
            'candidate_path':row.get('candidate_path',row.get('candidate')),
            'candidate_sha256':row.get('candidate_sha256'),
            'audit130_path':row.get('audit130_path',row.get('point_summary')),
            'audit130_sha256':row.get('audit130_sha256',row.get('point_summary_sha256')) ,
            'pairs_path':row.get('pairs_path',row.get('self_pairs')),
            'pairs_sha256':row.get('pairs_sha256',row.get('self_pairs_sha256')),
        })
    return normalized

def run_geometry_subprocess(args, timeout=300):
    cmd=[str(PYTHON),str(SLOT),str(PYTHON),*map(str,args)]
    cp=subprocess.run(cmd,text=True,capture_output=True,timeout=timeout)
    if cp.returncode:
        raise RuntimeError(f'geometry subprocess failed ({cp.returncode})\nCMD={cmd}\nSTDOUT={cp.stdout[-4000:]}\nSTDERR={cp.stderr[-4000:]}')
    return {'argv':cmd,'stdout':cp.stdout,'stderr':cp.stderr,'returncode':cp.returncode}

def prepare(args):
    for label,path,pin in [('335 owner',BUILDER335,PINS['builder335']),('390 owner',BUILDER390,PINS['builder390']),('026 owner',OWNER026,PINS['owner026']),('357 owner',DEPTH357,PINS['depth357'])]:
        require_sha(path,pin,label)
    candidate=Path(args.candidate); require_sha(candidate,args.candidate_sha256,'candidate')
    a130p=Path(args.audit130); require_sha(a130p,args.audit130_sha256,'130 summary')
    a159p=Path(args.audit159); require_sha(a159p,args.audit159_sha256,'159 report')
    a130=json_load(a130p); a159=json_load(a159p)
    if a130.get('candidate_sha256')!=args.candidate_sha256: raise ValueError('130 summary candidate mismatch')
    if a130.get('complete_surface_audit') is not True: raise ValueError('130 point/self audit is not marked complete')
    if int(a130.get('exact_PLC_surface_pairs',-1))!=0: raise ValueError('selected candidate exact self audit is not zero')
    if a159.get('candidate_sha256')!=args.candidate_sha256: raise ValueError('159 report candidate mismatch')
    surface_pairs=count_surface_pairs(a159)
    cur=np.load(candidate,allow_pickle=False)
    if cur['points_world_m'].ndim!=2 or cur['points_world_m'].shape[1]!=3 or cur['source_O_faces'].ndim!=2 or cur['source_O_faces'].shape[1]!=3: raise ValueError('candidate geometry arrays malformed')
    # Verify a matched query receipt from the 130 directory.
    query_dir=Path(args.query_dir)
    query_receipt=query_dir/'query-receipt.json'
    if not query_receipt.is_file(): raise FileNotFoundError(f'query receipt missing: {query_receipt}')
    receipt_obj=json_load(query_receipt)
    qpoints=query_dir/'points.csv'; qpairs=query_dir/'pairs.csv'; qnodes=query_dir/'nodes.csv'
    if receipt_obj.get('physical_steps')!=0 or receipt_obj.get('identity_domain')!='mechanics_node_row': raise ValueError('query receipt physical-step or row-identity contract mismatch')
    if int(receipt_obj.get('query_rows',-1))!=len(cur['points_world_m']): raise ValueError('query receipt row count differs from candidate')
    if receipt_obj.get('points_sha256')!=sha(qpoints) or receipt_obj.get('pairs_sha256')!=sha(qpairs) or receipt_obj.get('nodes_sha256')!=sha(qnodes): raise ValueError('query receipt CSV hashes do not match files')
    # Validate 357 bytes/report through the exact contract consumed by owner 390.
    dnpz=Path(args.point_depth_npz); drep=Path(args.point_depth_report)
    require_sha(dnpz,args.point_depth_npz_sha256,'357 NPZ')
    require_sha(drep,args.point_depth_report_sha256,'357 report')
    depth=json_load(drep)
    if depth.get('owner_sha256')!=PINS['depth357'] or depth.get('builder_sha256')!=PINS['depth357']:
        raise ValueError('357 owner/builder pin mismatch')
    point_bytes_sha=hashlib.sha256(np.ascontiguousarray(cur['points_world_m']).tobytes()).hexdigest()
    if depth.get('candidate_sha256')!=args.candidate_sha256 or depth.get('rows_npz_sha256')!=args.point_depth_npz_sha256:
        raise ValueError('357 candidate/NPZ binding mismatch')
    if depth.get('candidate_points_sha256')!=point_bytes_sha or depth.get('baseline_points_sha256')!=point_bytes_sha:
        raise ValueError('357 Float32 point bytes differ from current candidate')
    if depth.get('query_receipt_sha256')!=sha(query_receipt) or depth.get('query_points_csv_sha256')!=receipt_obj.get('points_sha256') or depth.get('query_pairs_csv_sha256')!=receipt_obj.get('pairs_sha256') or depth.get('query_nodes_csv_sha256')!=receipt_obj.get('nodes_sha256'):
        raise ValueError('357 table is not bound to the exact matched query receipt/CSV set')
    if depth.get('status')!='PASS_POINT_DEPTH_ROWS_BASELINE_FEASIBLE' or depth.get('bank_count')!=185 or depth.get('complete_bank_accounting_verified') is not True:
        raise ValueError('357 table is not complete/baseline feasible')
    # Validate the previous seed table and its topology source.
    prior_report_path=Path(args.prior_seed_report); require_sha(prior_report_path,args.prior_seed_report_sha256,'prior seed report')
    prior_pairs_path=Path(args.prior_seed_pairs); require_sha(prior_pairs_path,args.prior_seed_pairs_sha256,'prior seed pairs')
    prior_report=json_load(prior_report_path)
    if prior_report.get('output_pairs_sha256')!=args.prior_seed_pairs_sha256: raise ValueError('prior seed report does not bind pair file')
    prior_candidate_path=Path(prior_report['candidate_path'])
    if sha(prior_candidate_path)!=prior_report.get('candidate_sha256'): raise ValueError('prior seed candidate pin mismatch')
    prior=np.load(prior_pairs_path,allow_pickle=False); prior_candidate=np.load(prior_candidate_path,allow_pickle=False)
    same_topology_identity(prior_candidate,cur,'prior seed candidate')
    if prior.ndim!=2 or prior.shape[1]!=2 or np.any(prior<0) or np.any(prior>=len(cur['source_O_faces'])): raise ValueError('prior seed rows malformed/out of range')
    failures=load_failed_manifest(args.failed_trials_manifest,args.failed_trials_manifest_sha256)
    extra=[]; failure_receipts=[]
    for item in failures:
        for k in ('candidate_path','candidate_sha256','audit130_path','audit130_sha256','pairs_path','pairs_sha256'):
            if k not in item: raise ValueError(f'failed trial manifest missing {k}')
        fcp=Path(item['candidate_path']); fsp=Path(item['audit130_path']); fpp=Path(item['pairs_path'])
        require_sha(fcp,item['candidate_sha256'],'failed candidate')
        require_sha(fsp,item['audit130_sha256'],'failed audit')
        require_sha(fpp,item['pairs_sha256'],'failed pair rows')
        fc=np.load(fcp,allow_pickle=False); fs=json_load(fsp); fp=np.load(fpp,allow_pickle=False)
        same_topology_identity(cur,fc,'failed trial topology')
        if fs.get('candidate_sha256')!=item['candidate_sha256'] or int(fs.get('exact_PLC_surface_pairs',-1))!=len(fp):
            raise ValueError('failed trial audit does not bind pair rows')
        if fp.ndim!=2 or fp.shape[1]!=2 or np.any(fp<0) or np.any(fp>=len(cur['source_O_faces'])) or np.any(fp[:,0]>=fp[:,1]):
            raise ValueError('failed trial pair rows malformed')
        extra.append(fp.astype(np.int64,copy=False))
        failure_receipts.append({**item,'pair_count':int(len(fp)),'topology_verified':True})
    before,merged=merge_pair_sets(prior,extra,len(cur['source_O_faces']))
    # Rebuild all owner 026 coefficients at the exact current F32 coordinates.
    spec=importlib.util.spec_from_file_location('self_separation_026_continuation',OWNER026)
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    rows,records=module.build_pair_rows(cur['points_world_m'].astype(np.float64)*1000.0,cur['source_O_faces'],merged,margin_fraction=0.25)
    max_rhs=max((float(r['lower']) for r in rows),default=0.0)
    if max_rhs>2e-8: raise ValueError(f'zero-step regenerated 026 rows infeasible: {max_rhs} mm')
    out=Path(args.output_root)
    out.mkdir(parents=True,exist_ok=False)
    seed_dir=out/'self-seeds'; seed_dir.mkdir()
    seed_path=seed_dir/'merged-self-pair-seeds.npy'; npy_write(seed_path,merged)
    seed_report={
      'schema':'numi.human.continuation-self-seeds.v1','status':'PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY',
      'candidate_path':str(candidate),'candidate_sha256':args.candidate_sha256,
      'prior_seed_report_path':str(prior_report_path),'prior_seed_report_sha256':args.prior_seed_report_sha256,
      'failed_trials_manifest_path':str(Path(args.failed_trials_manifest)),'failed_trials_manifest_sha256':args.failed_trials_manifest_sha256,
      'prior_seed_pairs_path':str(prior_pairs_path),'prior_seed_pairs_sha256':args.prior_seed_pairs_sha256,
      'failed_trials_manifest_path':str(Path(args.failed_trials_manifest)),'failed_trials_manifest_sha256':args.failed_trials_manifest_sha256,
      'prior_pair_count':int(len(before)),'failed_trial_inputs':failure_receipts,
      'added_raw_pair_rows':int(sum(len(x) for x in extra)),'new_unique_pair_count':int(len(merged)-len(before)),
      'merged_unique_pair_count':int(len(merged)),'pair_owner_path':str(OWNER026),'pair_owner_sha256':PINS['owner026'],
      'rebuilt_linear_row_count':int(len(rows)),'rebuilt_record_count':int(len(records)),'margin_fraction':0.25,
      'zero_step_max_rhs_mm':max_rhs,'zero_step_feasible':bool(max_rhs<=2e-8),
      'output_pairs_path':str(seed_path),'output_pairs_sha256':sha(seed_path),
      'limitations':['Construction constraints only; exact self audit remains mandatory.','No physical/native/material admission is claimed.']}
    seed_report_path=seed_dir/'report.json'; json_write(seed_report_path,seed_report); seed_report['report_sha256']=sha(seed_report_path)
    # Build 335 first-union proposals against the matched 159 and query receipt.
    out335=out/'surface-targets-335'
    run335=run_geometry_subprocess([BUILDER335,'--candidate',candidate,'--surface-report',a159p,'--query-receipt',query_receipt,'--output-dir',out335,'--clearance-m',str(args.clearance_m),'--direction-mode','outward-nearest'])
    r335p=out335/'report.json'; n335p=out335/'targets.npz'; r335=json_load(r335p)
    if r335.get('matched_159_complete') is not True or int(r335.get('banks',-1))!=185 or int(r335.get('passing_targets',-1))!=int(r335.get('crossing_pairs',-2)) or int(r335.get('crossing_pairs',-1))!=surface_pairs:
        raise ValueError('335 output is not complete/matched to all 185 current crossings')
    if r335.get('inputs',{}).get('candidate_sha256')!=args.candidate_sha256: raise ValueError('335 output candidate mismatch')
    if r335.get('inputs',{}).get('surface_report_sha256')!=args.audit159_sha256: raise ValueError('335 output surface report mismatch')
    # Build 390 depth-maximizing targets, with exact 185-bank endpoint validation.
    out390=out/'point-targets-390'
    run390=run_geometry_subprocess([BUILDER390,'--candidate',candidate,'--point-summary',a130p,'--query-dir',query_dir,'--output-dir',out390,'--clearance-m',str(args.clearance_m),'--direction-mode','depth-maximizing','--point-depth-rows-npz',dnpz,'--point-depth-rows-report',drep,'--minimum-depth-cosine',str(args.minimum_depth_cosine)])
    r390p=out390/'nearest-union-exits.json'; n390p=out390/'nearest-union-exits.npz'; r390=json_load(r390p)
    inside_count=int(depth.get('inside_row_count',-1))
    if r390.get('status')!='PASS_FIRST_UNION_EXIT_TARGETS_ONLY' or int(r390.get('passing_targets',-1))!=inside_count or int(r390.get('query_rows',-1))!=inside_count:
        raise ValueError('390 target result incomplete or differs from matched 357 inside rows')
    overall={
      'schema':'numi.human.connected-continuation-input-preparation.v1','status':'PASS_CONSTRUCTION_INPUTS_ONLY',
      'candidate_path':str(candidate),'candidate_sha256':args.candidate_sha256,
      'audit130_path':str(a130p),'audit130_sha256':args.audit130_sha256,'audit159_path':str(a159p),'audit159_sha256':args.audit159_sha256,
      'query_receipt_path':str(query_receipt),'query_receipt_sha256':sha(query_receipt),'surface_pair_count':int(surface_pairs),
      'point_depth_npz_path':str(dnpz),'point_depth_npz_sha256':args.point_depth_npz_sha256,'point_depth_report_path':str(drep),'point_depth_report_sha256':args.point_depth_report_sha256,
      'source_path':str(Path(__file__).resolve()),'source_sha256':sha(Path(__file__).resolve()),
      'source_pins':{k:{'path':str(v),'sha256':sha(v)} for k,v in [('335',BUILDER335),('390',BUILDER390),('026',OWNER026),('357',DEPTH357)]},
      'self_seeds':{'path':str(seed_path),'sha256':sha(seed_path),'report_path':str(seed_report_path),'report_sha256':sha(seed_report_path),'pair_count':int(len(merged)),'new_unique_pair_count':int(len(merged)-len(before)),'zero_step_max_rhs_mm':max_rhs},
      '335':{'report_path':str(r335p),'report_sha256':sha(r335p),'npz_path':str(n335p),'npz_sha256':sha(n335p),'passing_targets':int(r335['passing_targets']),'surface_pairs':int(surface_pairs),'classification':r335.get('classification')},
      '390':{'report_path':str(r390p),'report_sha256':sha(r390p),'npz_path':str(n390p),'npz_sha256':sha(n390p),'passing_targets':int(r390['passing_targets']),'depth_rows':inside_count,'classification':r390.get('classification')},
      'construction_only':True,'accepted_physical_steps':0,'native_admission':False}
    top=out/'preparation-report.json'; json_write(top,overall)
    print(json.dumps(overall,indent=2,sort_keys=True))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','audit130','audit159','prior-seed-report','prior-seed-pairs','failed-trials-manifest','point-depth-npz','point-depth-report','output-root'):
        ap.add_argument('--'+name,required=True)
    for name in ('candidate-sha256','audit130-sha256','audit159-sha256','prior-seed-report-sha256','prior-seed-pairs-sha256','failed-trials-manifest-sha256','point-depth-npz-sha256','point-depth-report-sha256'):
        ap.add_argument('--'+name,required=True)
    ap.add_argument('--query-dir',required=True)
    ap.add_argument('--clearance-m',type=float,default=0.00025)
    ap.add_argument('--minimum-depth-cosine',type=float,default=0.05)
    args=ap.parse_args()
    if not (0.0<args.clearance_m<=0.001): raise ValueError('construction clearance outside approved bounded range')
    if not (0.0<args.minimum_depth_cosine<1.0): raise ValueError('invalid minimum depth cosine')
    prepare(args)
if __name__=='__main__': main()
