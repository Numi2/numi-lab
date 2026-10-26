#!/usr/bin/env python3
"""Fail-closed adapter from the frozen 401 proposal and verified 403 cumulative seed union to exact 373 audits."""
from pathlib import Path
import argparse, hashlib, json, subprocess, os
import numpy as np

D=Path('/Users/numi/Work/numi-human-connected-surface-069')
H=D/'barycentric-connected-clearance-170'
BASE=D/'barycentric-connected-clearance-170/combined-clearance-394/run-001/candidate.npz'
STATE=D/'barycentric-connected-clearance-170/combined-clearance-394/run-001/selected-state-003/selected-result.json'
POINT=D/'barycentric-connected-clearance-170/combined-clearance-394/run-001/audit-130/summary.json'
SURFACE=D/'barycentric-connected-clearance-170/combined-clearance-394/run-001/audit-159/report.json'
FIT_OWNER=D/'D/retained-gain-continuation-401/propose_joint_clearance_witnesses_401.py'
SEED_REPORT=D/'D/self-witness-refresh-403/report-403-adapter.json'
SEED_PAIRS=D/'D/self-witness-refresh-403/merged-self-pair-seeds.npy'
PARENT_SEED_REPORT=D/'D/self-witness-refresh-402/report.json'
PARENT_SEED_PAIRS=D/'D/self-witness-refresh-402/merged-self-pair-seeds.npy'
PAIR_OWNER=D/'barycentric-connected-clearance-170/connected-local-continuation-189-run001/round-005/proposal/round-001/fit/self_separation_026.py'
BACKTRACKER=H/'no-progress-backtracker-373/run_local_witness_backtracking_373.py'
POLICY=H/'no-progress-backtracker-373/backtracking_policy_373.py'
GLOBAL_HELPER=H/'global_step_backtracking_275.py'
PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
PINS={
 'base_candidate':'251b7e0a8c39d2e1e3ef57b2654bf4912cfb3e62bc9cc38fb9ee4d7a9a52e8c3',
 'state':'3eff4846352bc8090cf52ff9c6f754bc184dec1cd738d23077971723fb30fabd',
 'point':'37fde05865aebd9037ab0840e50371157eb1f08c61929e643223115a4a351e0a',
 'surface':'a16543bc13f8fb35eec62e0f8cad6dc3b3fe587cb92366c93f33874dda8b6774',
 'fit_owner':'2f1eda3f0d46d0f5497a1a04075b5dc2c3b37afe906d0389ff636fca8abc5869',
 'self_pair_seed_union':'528a72d9c48adbebdca960b37fbd1e124448ae80bf43cb6642f4389aacc209f4',
 'self_pair_seed_report':'328c81381e5c5bbc74d420fdb1c39e0e0f536664d43342acd1e967f203a9ea4f',
 'parent_seed_union':'5886a19ba21f7c5143ebaa83797b2c89392bf5335a9c7936132fb7a479bd8b4f',
 'parent_seed_report':'e370f1766dbf99dd4fa782ba1753e54bc14711416d8d36b732df086ce052a15d',
 'pair_owner':'c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3',
 'backtracker':'a91b410c979701a29ab32f376637db566abb745e066b629c000f6a37fea13212',
 'policy':'94d1c929745a5db95569bc58f96acf2b82579998a6a408d3f12ddf181366ab19',
 'global_helper':'18891b1f700592563e139edfc8bd1361d230ff6fd5af2ae5ee036c9625c1826a',
}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,obj): Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')

def verify_pins():
 paths={'base_candidate':BASE,'state':STATE,'point':POINT,'surface':SURFACE,'fit_owner':FIT_OWNER,
        'self_pair_seed_union':SEED_PAIRS,'self_pair_seed_report':SEED_REPORT,'pair_owner':PAIR_OWNER,
        'parent_seed_union':PARENT_SEED_PAIRS,'parent_seed_report':PARENT_SEED_REPORT,
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



def _canonical_pair_array(value, name):
 arr=np.asarray(value)
 if arr.ndim!=2 or arr.shape[1]!=2 or not np.issubdtype(arr.dtype,np.integer):
  raise ValueError(f"{name} must be an integer Nx2 face-row array")
 arr=np.asarray(arr,dtype=np.int64)
 if np.any(arr[:,0]>=arr[:,1]): raise ValueError(f"{name} contains noncanonical/self pairs")
 if len(np.unique(arr,axis=0))!=len(arr): raise ValueError(f"{name} contains duplicate pairs")
 return arr

def _pair_set(arr):
 return set(map(tuple,np.asarray(arr,dtype=np.int64).tolist()))

def validate_pair_union(merged, old, sources):
 merged=_canonical_pair_array(merged,"merged seed set")
 old=_canonical_pair_array(old,"baseline seed set")
 source_arrays=[_canonical_pair_array(x,f"witness source {i}") for i,x in enumerate(sources)]
 merged_set=_pair_set(merged); old_set=_pair_set(old)
 if not old_set.issubset(merged_set): raise ValueError("merged seeds omit baseline-retained pairs")
 source_sets=[_pair_set(x) for x in source_arrays]
 for i,source_set in enumerate(source_sets):
  if not source_set.issubset(merged_set): raise ValueError(f"merged seeds omit pass witness pairs from source {i+1}")
 expected=old_set.union(*source_sets)
 if merged_set!=expected: raise ValueError("merged seeds differ from exact old-plus-witness union")
 if not np.array_equal(merged,np.asarray(sorted(expected),dtype=np.int64).reshape(-1,2)):
  raise ValueError("merged seed order is not deterministic lexicographic order")
 return {"old_pair_count":len(old),"source_pair_counts":[len(x) for x in source_arrays],
         "merged_pair_count":len(merged),"new_unique_pair_count":len(expected-old_set),
         "overlap_with_old_count":sum(len(x & old_set) for x in source_sets)}

def verify_seed_bundle(state=None):
 report=read(SEED_REPORT)
 if sha(SEED_REPORT)!=PINS["self_pair_seed_report"]: raise ValueError("403 seed-union adapter report SHA mismatch")
 if sha(SEED_PAIRS)!=PINS["self_pair_seed_union"]: raise ValueError("403 seed-union bytes SHA mismatch")
 if sha(PARENT_SEED_REPORT)!=PINS["parent_seed_report"] or sha(PARENT_SEED_PAIRS)!=PINS["parent_seed_union"]:
  raise ValueError("402 parent seed bundle pins changed")
 parent=read(PARENT_SEED_REPORT)
 if parent.get("candidate_sha256")!=PINS["base_candidate"] or parent.get("output_pairs_sha256")!=PINS["parent_seed_union"] or int(parent.get("merged_pair_count",-1))!=17643:
  raise ValueError("403 parent is not the validated 402 seed bundle")
 if report.get("candidate_npz")!=str(BASE) or report.get("candidate_sha256")!=PINS["base_candidate"]:
  raise ValueError("403 seed-union baseline candidate pin mismatch")
 if report.get("output_pairs_npz")!=str(SEED_PAIRS) or report.get("output_pairs_sha256")!=PINS["self_pair_seed_union"]:
  raise ValueError("403 report does not pin merged pair file")
 if report.get("pair_owner")!=str(PAIR_OWNER) or report.get("pair_owner_sha256")!=PINS["pair_owner"]:
  raise ValueError("403 report pair-row owner pin mismatch")
 if report.get("parent_report_path")!=str(PARENT_SEED_REPORT) or report.get("parent_report_sha256")!=PINS["parent_seed_report"]:
  raise ValueError("403 does not cite the frozen 402 parent report")
 if report.get("parent_pairs_path")!=str(PARENT_SEED_PAIRS) or report.get("parent_pairs_sha256")!=PINS["parent_seed_union"]:
  raise ValueError("403 does not cite the frozen 402 parent pairs")
 if int(report.get("added_unique_pairs_vs_parent",-1))!=189:
  raise ValueError("403 does not contain the complete 189 new witness set")
 if report.get("native_admission") is not False or report.get("accepted_physical_steps")!=0:
  raise ValueError("403 seed refresh is not construction-only")
 old_path=Path(report.get("old_seeds", ""))
 expected_old=H/"combined-clearance-394/run-001/selected-state-003/retained-self-pair-seeds.npy"
 if old_path.resolve()!=expected_old.resolve(): raise ValueError("403 union does not trace to selected-state-003 baseline seeds")
 if sha(old_path)!=report.get("old_seeds_sha256"): raise ValueError("403 original seed source hash mismatch")
 if state is None: state=read(STATE)
 old_receipt=state.get("identity_safe_self_pair_seed",{})
 if old_receipt.get("sha256")!=report.get("old_seeds_sha256"): raise ValueError("403 original seeds differ from baseline selected-state evidence")
 old=np.load(old_path,allow_pickle=False); prior=np.load(PARENT_SEED_PAIRS,allow_pickle=False); merged=np.load(SEED_PAIRS,allow_pickle=False)
 if int(report.get("old_pair_count",-1))!=len(old) or len(old)!=16527: raise ValueError("403 original seed count mismatch")
 if int(report.get("merged_pair_count",-1))!=len(merged) or len(merged)!=17832: raise ValueError("403 merged pair count mismatch")
 if int(report.get("new_pair_count",-1))!=len(merged)-len(old) or int(report.get("new_pair_count",-1))!=1305:
  raise ValueError("403 cumulative added-pair count mismatch")
 sources=report.get("sources")
 if not isinstance(sources,list) or len(sources)!=14: raise ValueError("403 must include all 14 complete 400/401 pass witnesses")
 source_arrays=[]; source_proofs=[]
 with np.load(BASE,allow_pickle=False) as base:
  base_faces=np.asarray(base["source_O_faces"]); base_points=np.asarray(base["points_world_m"])
  for i,rec in enumerate(sources,1):
   family="400" if i<=7 else "401"; pass_id=i if i<=7 else i-7
   if rec.get("family")!=family or int(rec.get("pass",-1))!=pass_id: raise ValueError(f"403 source family/pass ordering mismatch at {i}")
   pair_path=Path(rec.get("pairs_path",""))
   expected_root=H/f"postfit-backtrack-{family}/run-001/backtrack-373"
   expected_path=expected_root/f"pass-{pass_id:03d}/audit-130/surface-self-pairs.npy"
   if pair_path.resolve()!=expected_path.resolve(): raise ValueError(f"403 witness path mismatch at source {i}")
   if sha(pair_path)!=rec.get("pairs_sha256"): raise ValueError(f"403 witness {i} pair hash mismatch")
   summary_path=Path(rec.get("summary_path","")); summary=read(summary_path)
   candidate=Path(rec.get("candidate_path","")); candidate_hash=sha(candidate)
   if summary_path.resolve()!=(pair_path.parent/"summary.json").resolve() or candidate.resolve()!=(pair_path.parent.parent/"candidate.npz").resolve():
    raise ValueError(f"403 witness {i} audit/candidate path mismatch")
   if rec.get("candidate_sha256")!=candidate_hash or summary.get("candidate_sha256")!=candidate_hash:
    raise ValueError(f"403 witness {i} trial candidate hash mismatch")
   if rec.get("summary_sha256")!=sha(summary_path): raise ValueError(f"403 witness {i} summary hash mismatch")
   if summary.get("complete_surface_audit") is not True or summary.get("status")!="STOPPED_AT_SURFACE_GATE":
    raise ValueError(f"403 witness {i} is not a complete exact self-pair audit")
   if summary.get("points")!=len(base_points) or summary.get("faces")!=len(base_faces): raise ValueError(f"403 witness {i} topology dimensions mismatch")
   if summary.get("surface_predicate_sha256")!="f0c1d81bffb185565b174f293b67da01926db1dfa6f6b80cb5629828ccf08b6d":
    raise ValueError(f"403 witness {i} exact predicate pin mismatch")
   pairs=_canonical_pair_array(np.load(pair_path,allow_pickle=False),f"403 source {i} pairs")
   if len(pairs)!=int(rec.get("pair_count",-1)) or len(pairs)!=int(summary.get("exact_PLC_surface_pairs",-1)):
    raise ValueError(f"403 witness {i} pair count mismatch")
   with np.load(candidate,allow_pickle=False) as trial:
    if not np.array_equal(np.asarray(trial["source_O_faces"]),base_faces): raise ValueError(f"403 witness {i} topology differs from baseline")
    if np.asarray(trial["points_world_m"]).shape!=base_points.shape: raise ValueError(f"403 witness {i} point dimension mismatch")
   source_arrays.append(pairs)
   source_proofs.append({"family":family,"pass":pass_id,"pairs_path":str(pair_path),"pairs_sha256":sha(pair_path),
     "pair_count":len(pairs),"summary_path":str(summary_path),"summary_sha256":sha(summary_path),
     "candidate_path":str(candidate),"candidate_sha256":candidate_hash,"complete_surface_audit":True})
 counts=validate_pair_union(merged,old,source_arrays)
 if not np.array_equal(prior,np.asarray(sorted(set(map(tuple,prior.tolist()))),dtype=np.int64).reshape(-1,2)):
  raise ValueError("402 parent pair order is not canonical")
 prior_set=_pair_set(prior); merged_set=_pair_set(merged)
 if not prior_set.issubset(merged_set) or len(merged_set-prior_set)!=189:
  raise ValueError("403 does not add exactly the 189 pairs beyond 402")
 if counts["merged_pair_count"]!=17832 or counts["new_unique_pair_count"]!=1305:
  raise ValueError("403 seed union differs from 394 plus all fourteen source arrays")
 if int(report.get("rebuilt_row_count",-1))!=100558 or int(report.get("rebuilt_026_row_count",-1))!=100558:
  raise ValueError("403 rebuilt 026 row count mismatch")
 if not np.isfinite(float(report.get("baseline_max_rhs_mm",np.nan))) or float(report["baseline_max_rhs_mm"])>2e-8:
  raise ValueError("403 026 rows are not zero-step feasible at unchanged 394 baseline")
 if int(report.get("baseline_positive_rhs_count",-1))!=0:
  raise ValueError("403 026 report has positive baseline rows")
 return {"report_path":str(SEED_REPORT),"report_sha256":PINS["self_pair_seed_report"],
         "pairs_path":str(SEED_PAIRS),"pairs_sha256":PINS["self_pair_seed_union"],
         **counts,"new_unique_pairs_vs_402":189,"parent_pair_count":len(prior),
         "rebuilt_026_row_count":100558,"rebuilt_026_max_rhs_mm":float(report["baseline_max_rhs_mm"]),
         "source_proofs":source_proofs}

def validate_fit(trial, report, seed_bundle):
 if report.get('status')!='CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED': raise ValueError('fit did not emit a construction proposal')
 if report.get('candidate_admission')!='BLOCKED_PENDING_EXACT_GLOBAL_SELF_AND_OTHER_GATES': raise ValueError('fit report does not preserve pending exact gates')
 if Path(report.get('candidate_path','')).resolve()!=trial.resolve() or report.get('candidate_sha256')!=sha(trial): raise ValueError('fit candidate path/hash does not match report')
 if report.get('native_admission') is not False or report.get('accepted_physical_steps')!=0: raise ValueError('fit is not construction-only')
 if report.get('exact_global_self_audit')!='NOT_RUN_REQUIRED' or report.get('all_185_bank_point_query')!='NOT_RUN' or report.get('full_surface_bone_audit')!='NOT_RUN': raise ValueError('fit report claims post-fit audit already ran')
 inputs=report.get('inputs',{})
 if Path(inputs.get('base_candidate','')).resolve()!=BASE.resolve() or inputs.get('base_candidate_sha256')!=PINS['base_candidate']: raise ValueError('fit base candidate provenance mismatch')
 if Path(inputs.get('area_reference','')).resolve()!=BASE.resolve() or inputs.get('area_reference_sha256')!=PINS['base_candidate']: raise ValueError('fit area-reference provenance mismatch')
 if Path(inputs.get('self_pairs','')).resolve()!=SEED_PAIRS.resolve() or inputs.get('self_pairs_sha256')!=seed_bundle['pairs_sha256']: raise ValueError('fit did not use the verified 402 seven-pass self-pair seed union')
 if Path(inputs.get('pair_owner','')).resolve()!=PAIR_OWNER.resolve() or inputs.get('pair_owner_sha256')!=PINS['pair_owner']: raise ValueError('fit pair-row owner provenance mismatch')
 if report.get('support',{}).get('threatened_pairs')!=seed_bundle['merged_pair_count']: raise ValueError('fit omitted merged self-pair constraints')
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


def guard_not_geometry_slot_wrapped(parent_command: str) -> None:
 if 'numi-human-geometry-slot.py' in parent_command:
  raise RuntimeError('Do not wrap this adapter in numi-human-geometry-slot.py: its backtracker launches exact audit subprocesses that acquire the slot themselves.')

def assert_not_geometry_slot_wrapped() -> None:
 try:
  parent=subprocess.run(['ps','-p',str(os.getppid()),'-o','command='],check=True,capture_output=True,text=True).stdout.strip()
 except Exception as exc:
  raise RuntimeError(f'cannot verify geometry-slot parent; invoke adapter directly: {exc}') from exc
 guard_not_geometry_slot_wrapped(parent)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--fit-dir',default=str(H/'retained-gain-continuation-401/run-001')); ap.add_argument('--out',default=str(H/'postfit-backtrack-401/run-001')); ap.add_argument('--validate-only',action='store_true'); ap.add_argument('--self-test',action='store_true'); a=ap.parse_args()
 if a.self_test: self_test(); return
 assert_not_geometry_slot_wrapped()
 actual=verify_pins(); state=verify_baseline(); seed_bundle=verify_seed_bundle(state)
 fit_dir=Path(a.fit_dir); report_path=fit_dir/'report.json'; trial=fit_dir/'candidate.npz'
 if not report_path.is_file() or not trial.is_file(): raise FileNotFoundError('fit candidate/report not complete yet')
 fit_report=read(report_path); fit_hash=sha(report_path); fit_candidate_hash=sha(trial)
 scope=validate_fit(trial,fit_report,seed_bundle)
 summary={'status':'FIT_OUTPUT_READY_FOR_EXACT_373_BACKTRACK','baseline_candidate_sha256':PINS['base_candidate'],'baseline_state_sha256':PINS['state'],'baseline_point_summary_sha256':PINS['point'],'baseline_surface_report_sha256':PINS['surface'],'fit_report_sha256':fit_hash,'fit_candidate_sha256':fit_candidate_hash,'fit_owner_sha256':actual['fit_owner'],'support_validation':scope,'verified_402_self_pair_seed_bundle':seed_bundle,'native_admission':False,'accepted_physical_steps':0,'topology_comparison_qualification':'UNQUALIFIED_DUE_TO_CHANGED_TOPOLOGY'}
 if a.validate_only: print(json.dumps(summary,indent=2)); return
 out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
 cfg={'state':{'candidate':str(BASE),'point_summary':str(POINT),'surface_report':str(SURFACE),'parent_fit':str(report_path)},
  'trial':str(trial),'out':str(out/'backtrack-373'),'parent_fit':str(report_path),
  'helper':str(GLOBAL_HELPER),'helper_sha256':PINS['global_helper'],'helper_entry':'reduce_global_step_scales',
  'initial_scale':1.0,'zero_below':1.0/64.0,'maximum_exact_audits':8,'maximum_construction_passes':32,
  'fixed_prefix_nodes':0,'fit_support_rows':scope['support_rows'],'uniform_failure_backtracking':True,
  'area_reference_npz':str(BASE),'area_reference_sha256':PINS['base_candidate'],
  'area_reference_points_key':'area_reference_037_points_world_m_f64','area_reference_faces_key':'area_reference_037_source_O_faces',
  'fit_report_sha256':fit_hash,'fit_candidate_sha256':fit_candidate_hash,
  'self_pair_seed_path':str(SEED_PAIRS),'self_pair_seed_sha256':seed_bundle['pairs_sha256'],
  'self_pair_seed_report_sha256':seed_bundle['report_sha256'],'self_pair_seed_count':seed_bundle['merged_pair_count']}
 cfgpath=out/'backtrack-config.json'; write(cfgpath,cfg); write(out/'adapter-preflight.json',summary)
 cmd=[PY,str(BACKTRACKER),str(cfgpath)]
 with (out/'backtracker.log').open('x') as log: result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 write(out/'execution.json',{'command':cmd,'returncode':result.returncode,'fit_report_sha256':fit_hash,'fit_candidate_sha256':fit_candidate_hash,'self_pair_seed_sha256':seed_bundle['pairs_sha256'],'backtracker_sha256':PINS['backtracker'],'native_admission':False,'accepted_physical_steps':0})
 if result.returncode: raise SystemExit(result.returncode)
 selected=out/'backtrack-373/selected-result.json'
 if selected.exists():
  s=read(selected); sp=Path(s['point_summary']); sr=Path(s['surface_report'])
  if s.get('native_admission') is not False or s.get('accepted_physical_steps')!=0: raise ValueError('373 selected result violates construction-only contract')
  write(out/'postcheck-receipt.json',{'status':s.get('status'),'selected_candidate':s.get('candidate'),'selected_candidate_sha256':s.get('candidate_sha256'),'point_summary_sha256':sha(sp),'surface_report_sha256':sha(sr),'native_admission':False,'accepted_physical_steps':0,'topology_comparison_qualification':'UNQUALIFIED_DUE_TO_CHANGED_TOPOLOGY'})
 print(json.dumps({'status':'POSTCHECKS_COMPLETE','out':str(out),'returncode':result.returncode,'selected_exists':selected.exists(),'native_admission':False,'accepted_physical_steps':0}))

if __name__=='__main__': main()
