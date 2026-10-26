#!/usr/bin/env python3
"""Fail-closed adapter from one frozen 396 construction proposal to exact 373 audits."""
from pathlib import Path
import argparse, hashlib, json, subprocess
import numpy as np

D=Path('/Users/numi/Work/numi-human-connected-surface-069')
H=D/'barycentric-connected-clearance-170'
BASE=D/'barycentric-connected-clearance-170/combined-clearance-394/run-001/candidate.npz'
STATE=D/'barycentric-connected-clearance-170/combined-clearance-394/run-001/selected-state-003/selected-result.json'
POINT=D/'barycentric-connected-clearance-170/combined-clearance-394/run-001/audit-130/summary.json'
SURFACE=D/'barycentric-connected-clearance-170/combined-clearance-394/run-001/audit-159/report.json'
FIT_OWNER=D/'D/precision-phase-one-396/propose_joint_clearance_witnesses_396.py'
BACKTRACKER=H/'no-progress-backtracker-373/run_local_witness_backtracking_373.py'
POLICY=H/'no-progress-backtracker-373/backtracking_policy_373.py'
GLOBAL_HELPER=H/'global_step_backtracking_275.py'
PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
PINS={
 'base_candidate':'251b7e0a8c39d2e1e3ef57b2654bf4912cfb3e62bc9cc38fb9ee4d7a9a52e8c3',
 'state':'3eff4846352bc8090cf52ff9c6f754bc184dec1cd738d23077971723fb30fabd',
 'point':'37fde05865aebd9037ab0840e50371157eb1f08c61929e643223115a4a351e0a',
 'surface':'a16543bc13f8fb35eec62e0f8cad6dc3b3fe587cb92366c93f33874dda8b6774',
 'fit_owner':'705ebe465df1e9d186b465e498cdb8c3f693ad46690c48cb1f4794fc478ea03a',
 'backtracker':'a91b410c979701a29ab32f376637db566abb745e066b629c000f6a37fea13212',
 'policy':'94d1c929745a5db95569bc58f96acf2b82579998a6a408d3f12ddf181366ab19',
 'global_helper':'18891b1f700592563e139edfc8bd1361d230ff6fd5af2ae5ee036c9625c1826a',
}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,obj): Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def verify_pins():
 paths={'base_candidate':BASE,'state':STATE,'point':POINT,'surface':SURFACE,'fit_owner':FIT_OWNER,
        'backtracker':BACKTRACKER,'policy':POLICY,'global_helper':GLOBAL_HELPER}
 actual={k:sha(v) for k,v in paths.items()}
 for k,v in actual.items():
  if v!=PINS[k]: raise ValueError(f'pinned {k} source/input changed: {v}')
 return actual

def verify_baseline():
 state=read(STATE); point=read(POINT); surface=read(SURFACE)
 if state.get('candidate_sha256')!=PINS['base_candidate'] or Path(state.get('candidate','')).resolve()!=BASE.resolve(): raise ValueError('selected-state-003 candidate mismatch')
 if state.get('native_admission') is not False or state.get('accepted_physical_steps')!=0: raise ValueError('baseline must remain construction-only')
 if state.get('status')!='AUDITED_CONSTRUCTION_STATE_SURFACE_COMPARISON_UNQUALIFIED': raise ValueError('unexpected state-003 classification')
 if state.get('surface_pair_comparison',{}).get('classification')!='PARTIAL_IDENTITY_SAFE_SURFACE_SUBCOMPARISON': raise ValueError('topology-changed comparison is not expressly marked unqualified')
 if point.get('candidate_sha256')!=PINS['base_candidate'] or point.get('exact_PLC_surface_pairs')!=0: raise ValueError('baseline exact self receipt mismatch')
 if surface.get('candidate_sha256')!=PINS['base_candidate'] or surface.get('status')!='COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT' or surface.get('bank_count')!=185: raise ValueError('baseline 185-bank surface receipt incomplete/mismatched')
 return state

def validate_fit(trial, report):
 if report.get('status')!='CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED': raise ValueError('fit did not emit a construction proposal')
 if report.get('candidate_admission')!='BLOCKED_PENDING_EXACT_GLOBAL_SELF_AND_OTHER_GATES': raise ValueError('fit report does not preserve pending exact gates')
 if Path(report.get('candidate_path','')).resolve()!=trial.resolve() or report.get('candidate_sha256')!=sha(trial): raise ValueError('fit candidate path/hash does not match report')
 if report.get('native_admission') is not False or report.get('accepted_physical_steps')!=0: raise ValueError('fit is not construction-only')
 if report.get('exact_global_self_audit')!='NOT_RUN_REQUIRED' or report.get('all_185_bank_point_query')!='NOT_RUN' or report.get('full_surface_bone_audit')!='NOT_RUN': raise ValueError('fit report claims post-fit audit already ran')
 inputs=report.get('inputs',{})
 if Path(inputs.get('base_candidate','')).resolve()!=BASE.resolve() or inputs.get('base_candidate_sha256')!=PINS['base_candidate']: raise ValueError('fit base candidate provenance mismatch')
 if Path(inputs.get('area_reference','')).resolve()!=BASE.resolve() or inputs.get('area_reference_sha256')!=PINS['base_candidate']: raise ValueError('fit area-reference provenance mismatch')
 state=read(STATE)
 if inputs.get('self_pairs_sha256')!=state['identity_safe_self_pair_seed']['sha256']: raise ValueError('fit did not use selected-state-003 exact-safe self-pair seed bytes')
 support_report=report.get('support',{})
 if support_report.get('fixed_prefix_nodes')!=0: raise ValueError('fit used unexpected fixed-prefix semantics')
 emitted=report.get('emitted_f32_checks',{})
 if emitted.get('strict_f32_emission_gate',{}).get('pass') is not True: raise ValueError('fit strict emitted-F32 gate did not pass')
 if emitted.get('full_mesh_area_gate',{}).get('pass') is not True: raise ValueError('fit full-mesh current/037 area gates did not pass')
 if emitted.get('maximum_node_step_mm',float('inf'))>1.000001 or report.get('maximum_source_displacement_mm',float('inf'))>12.000001: raise ValueError('fit step/source cap exceeded')
 solver=report.get('solver',{})
 if solver.get('fallback_used'):
  cert=solver.get('stage1_primal_certificate',{})
  if cert.get('status')!='CERTIFIED_STAGE1_PRIMAL': raise ValueError('stage-1 fallback lacks its certified-primal receipt')
 with np.load(BASE,allow_pickle=False) as b, np.load(trial,allow_pickle=False) as t:
  if not np.array_equal(b['source_O_faces'],t['source_O_faces']): raise ValueError('fit changed topology')
  if b['points_world_m'].dtype!=np.float32 or t['points_world_m'].dtype!=np.float32 or t['points_world_m'].shape!=b['points_world_m'].shape: raise ValueError('fit point arrays are not matched F32')
  for k in b.files:
   if k=='points_world_m': continue
   if k not in t.files or np.asarray(b[k]).dtype!=np.asarray(t[k]).dtype or np.asarray(b[k]).shape!=np.asarray(t[k]).shape or np.ascontiguousarray(b[k]).tobytes()!=np.ascontiguousarray(t[k]).tobytes(): raise ValueError(f'fit altered or omitted baseline non-coordinate field {k}')
  key='local_witness_cluster_222_support_rows'
  if key not in t.files: raise ValueError('fit candidate lacks explicit 222 support rows')
  support=np.asarray(t[key]); n=len(b['points_world_m'])
  if support.ndim!=1 or not np.issubdtype(support.dtype,np.integer) or np.any(support<0) or np.any(support>=n) or len(np.unique(support))!=len(support): raise ValueError('fit support row set malformed')
  if support_report.get('support_nodes')!=len(support): raise ValueError('fit report/candidate support size mismatch')
  changed=np.flatnonzero(np.any(t['points_world_m'].view(np.uint32)!=b['points_world_m'].view(np.uint32),axis=1))
  if np.any(~np.isin(changed,support)): raise ValueError('fit moved nodes outside LP support')
  anchors=np.asarray(b['typed_coarse_anchor_output_rows_018'],dtype=np.int64)
  if not np.array_equal(t['points_world_m'][anchors].view(np.uint32),b['points_world_m'][anchors].view(np.uint32)): raise ValueError('fit changed typed anchors')
  return {'support_rows':support.tolist(),'support_count':len(support),'changed_rows_count':len(changed),'changed_rows_outside_support':0,'anchor_count':len(anchors),'anchors_bitwise_unchanged':True,'topology_rows':len(b['source_O_faces']),'node_count':n}

def self_test():
 base=np.zeros((4,3),np.float32); support=np.array([2,3]); anchors=np.array([0])
 trial=base.copy(); trial[2,1]=np.float32(1)
 changed=np.flatnonzero(np.any(trial.view(np.uint32)!=base.view(np.uint32),axis=1))
 assert np.all(np.isin(changed,support)) and np.array_equal(trial[anchors].view(np.uint32),base[anchors].view(np.uint32))
 bad=base.copy(); bad[1,0]=np.float32(1)
 assert np.any(~np.isin(np.flatnonzero(np.any(bad.view(np.uint32)!=base.view(np.uint32),axis=1)),support))
 print('postfit-373 self-test PASS')

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fit-dir',default=str(H/'normal-clearance-396/run-003')); ap.add_argument('--out',default=str(H/'postfit-backtrack-400/run-001')); ap.add_argument('--validate-only',action='store_true'); ap.add_argument('--self-test',action='store_true'); a=ap.parse_args()
 if a.self_test: self_test(); return
 actual=verify_pins(); verify_baseline()
 fit_dir=Path(a.fit_dir); report_path=fit_dir/'report.json'; trial=fit_dir/'candidate.npz'
 if not report_path.is_file() or not trial.is_file(): raise FileNotFoundError('fit candidate/report not complete yet')
 fit_report=read(report_path); fit_hash=sha(report_path); fit_candidate_hash=sha(trial)
 scope=validate_fit(trial,fit_report)
 summary={'status':'FIT_OUTPUT_READY_FOR_EXACT_373_BACKTRACK','baseline_candidate_sha256':PINS['base_candidate'],'baseline_state_sha256':PINS['state'],'baseline_point_summary_sha256':PINS['point'],'baseline_surface_report_sha256':PINS['surface'],'fit_report_sha256':fit_hash,'fit_candidate_sha256':fit_candidate_hash,'fit_owner_sha256':actual['fit_owner'],'support_validation':scope,'native_admission':False,'accepted_physical_steps':0,'topology_comparison_qualification':'UNQUALIFIED_DUE_TO_CHANGED_TOPOLOGY'}
 if a.validate_only: print(json.dumps(summary,indent=2)); return
 out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
 cfg={'state':{'candidate':str(BASE),'point_summary':str(POINT),'surface_report':str(SURFACE),'parent_fit':str(report_path)},
  'trial':str(trial),'out':str(out/'backtrack-373'),'parent_fit':str(report_path),
  'helper':str(GLOBAL_HELPER),'helper_sha256':PINS['global_helper'],'helper_entry':'reduce_global_step_scales',
  'initial_scale':1.0,'zero_below':1.0/64.0,'maximum_exact_audits':8,'maximum_construction_passes':32,
  'fixed_prefix_nodes':0,'fit_support_rows':scope['support_rows'],'uniform_failure_backtracking':True,
  'area_reference_npz':str(BASE),'area_reference_sha256':PINS['base_candidate'],
  'area_reference_points_key':'area_reference_037_points_world_m_f64','area_reference_faces_key':'area_reference_037_source_O_faces',
  'fit_report_sha256':fit_hash,'fit_candidate_sha256':fit_candidate_hash}
 cfgpath=out/'backtrack-config.json'; write(cfgpath,cfg); write(out/'adapter-preflight.json',summary)
 cmd=[PY,str(BACKTRACKER),str(cfgpath)]
 with (out/'backtracker.log').open('x') as log: result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 write(out/'execution.json',{'command':cmd,'returncode':result.returncode,'fit_report_sha256':fit_hash,'fit_candidate_sha256':fit_candidate_hash,'backtracker_sha256':PINS['backtracker'],'native_admission':False,'accepted_physical_steps':0})
 if result.returncode: raise SystemExit(result.returncode)
 selected=out/'backtrack-373/selected-result.json'
 if selected.exists():
  s=read(selected); sp=Path(s['point_summary']); sr=Path(s['surface_report'])
  if s.get('native_admission') is not False or s.get('accepted_physical_steps')!=0: raise ValueError('373 selected result violates construction-only contract')
  write(out/'postcheck-receipt.json',{'status':s.get('status'),'selected_candidate':s.get('candidate'),'selected_candidate_sha256':s.get('candidate_sha256'),'point_summary_sha256':sha(sp),'surface_report_sha256':sha(sr),'native_admission':False,'accepted_physical_steps':0,'topology_comparison_qualification':'UNQUALIFIED_DUE_TO_CHANGED_TOPOLOGY'})
 print(json.dumps({'status':'POSTCHECKS_COMPLETE','out':str(out),'returncode':result.returncode,'selected_exists':selected.exists(),'native_admission':False,'accepted_physical_steps':0}))

if __name__=='__main__': main()
