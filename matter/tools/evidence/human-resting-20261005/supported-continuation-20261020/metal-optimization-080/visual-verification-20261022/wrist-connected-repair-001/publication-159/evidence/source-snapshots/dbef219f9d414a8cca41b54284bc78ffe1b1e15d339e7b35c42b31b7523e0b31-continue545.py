"""Continue a connected contained-point construction with existing exact owners.

No runtime physics or geometry acceptance rule is replaced. Each round refreshes
candidate-bound targets/guards, retains historical self witnesses, and runs the
existing fit and exact witness backtracker. All intermediate failures remain.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,subprocess,sys,time
import numpy as np
D=Path('/Users/numi/Work/numi-human-connected-surface-069');W=Path(__file__).parent
PY='/Users/n/numi-human-prep-venv-20261005/bin/python'
SLOT='/tmp/numi-human-geometry-slot.py'
C0=D/'D/focused-clearance-541/constraints/topology-007-c1e02'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,j):Path(p).write_text(json.dumps(j,indent=2,sort_keys=True,allow_nan=False)+'\n')
def run(argv,stem,slot=False):
 argv=list(map(str,argv));actual=[PY,SLOT]+argv if slot else argv
 pins={str(p):sha(p) for p in [Path(argv[1]),Path(__file__)] if p.is_file()}
 write(str(stem)+'.launch.json',dict(argv=actual,source_pins=pins))
 start=time.monotonic()
 with Path(str(stem)+'.log').open('x')as log:r=subprocess.run(actual,stdout=log,stderr=subprocess.STDOUT)
 write(str(stem)+'.execution.json',dict(returncode=r.returncode,seconds=time.monotonic()-start,pins_unchanged=all(sha(k)==v for k,v in pins.items())))
 if r.returncode:raise RuntimeError(f'failed {stem}: '+Path(str(stem)+'.log').read_text()[-3000:])
 if any(sha(k)!=v for k,v in pins.items()):raise RuntimeError('source changed')
def loadmod(p):
 sp=importlib.util.spec_from_file_location('seed026_cont545',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--full-point-fit',action='store_true');ap.add_argument('--point-direction-mode',choices=['depth-maximizing','direct-depth','source-outward'],default='depth-maximizing');ap.add_argument('--point-margin-m',type=float,default=.00005);ap.add_argument('--retained-gain',type=float,default=.95);ap.add_argument('--point-weight-mode',choices=['constant-per-mm','equal-per-point'],default='constant-per-mm');ap.add_argument('--state',required=True,type=Path);ap.add_argument('--round-dir',required=True,type=Path);ap.add_argument('--prior-context',type=Path);ap.add_argument('--prior-trial',type=Path);ap.add_argument('--prior-trial-audit130',type=Path);a=ap.parse_args()
 O=a.round_dir;O.mkdir(parents=True,exist_ok=False);st=read(a.state)
 for k in ('candidate','point_summary','surface_report'):
  if sha(st[k])!=st[k+'_sha256']:raise ValueError('state mismatch '+k)
 base=Path(st['candidate']);q=Path(st['point_summary']).parent/'query'
 before=read(st['point_summary']);assert before['exact_PLC_surface_pairs']==0 and before['complete_surface_audit']
 assert before['inside_node_bank_pairs']>0
 prior=read(a.prior_context)if a.prior_context else dict(candidate=read(W/'initial-state.json')['candidate'],seed_pairs_path=str(C0/'seed-refresh-001/merged-self-pair-seeds.npy'),guard_npz=str(C0/'fixed-guard-refresh-001/run-002/fixed-guards/fixed-skin-bone-guards.npz'),guard_report=str(C0/'fixed-guard-refresh-001/run-002/fixed-guards/report.json'))
 trial=a.prior_trial or W/'fit-001/candidate.npz';ta=a.prior_trial_audit130 or W/'check-001-diagnostic/audit-130/summary.json'
 # Refresh fixed guards and saved failed-trial witnesses on current coordinates.
 args=[PY,D/'D/focused-clearance-541/constraints/refresh_guards_541.py','--candidate',base,'--audit130',st['point_summary'],'--audit159',st['surface_report'],'--query-dir',q,'--ancestry-parent-candidate',base,'--seed-source-candidate',trial,'--seed-source-audit130',ta,'--seed-source-pairs',ta.parent/'surface-self-pairs.npy','--prior-guard-candidate',prior['candidate'],'--prior-guard-report',prior['guard_report'],'--prior-guard-npz',prior['guard_npz'],'--output-dir',O/'constraints','--expected-candidate-sha256',sha(base),'--expected-audit130-sha256',sha(st['point_summary']),'--expected-audit159-sha256',sha(st['surface_report'])]
 run(args,O/'refresh',True)
 rr=read(O/'constraints/refresh-report.json')
 # Retain full historical seeds, not only the newest failed trial.
 z=np.load(base,allow_pickle=False);pz=np.load(prior['candidate'],allow_pickle=False)
 assert np.array_equal(z['source_O_faces'],pz['source_O_faces']),'topology changed: exact identity remap required'
 pairs=np.unique(np.vstack([np.load(prior['seed_pairs_path']),np.load(rr['seed_pairs_path'])]),axis=0)
 seed=O/'constraints/merged-self-pair-seeds.npy';np.save(seed,pairs)
 ownerp=D/'publication-101/evidence/204/source/fit/self_separation_026.py'
 assert sha(ownerp)=='c67daf7c9fd2292e32ae88dba215a74350e1cbf83c669bc299a64b86d12e99f3'
 rows,_=loadmod(ownerp).build_pair_rows(z['points_world_m'].astype(float)*1000,z['source_O_faces'],pairs,margin_fraction=.25)
 maxrhs=max(float(r['lower'])for r in rows);assert maxrhs<=2e-8
 seed_report=O/'constraints/seed-preflight.json'
 write(seed_report,dict(schema='numi.human.continuation-self-seeds.v1',status='PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY',candidate_sha256=sha(base),zero_step_feasible=True,zero_step_max_rhs_mm=maxrhs,rebuilt_linear_row_count=len(rows),output_pairs_path=str(seed),output_pairs_sha256=sha(seed),merged_unique_pair_count=len(pairs),pair_owner_sha256=sha(ownerp),prior_seed_path=prior['seed_pairs_path'],prior_seed_sha256=sha(prior['seed_pairs_path']),new_trial_seed_path=rr['seed_pairs_path'],new_trial_seed_sha256=sha(rr['seed_pairs_path']),native_admission=False,accepted_physical_steps=0))
 ctx=dict(candidate=str(base),seed_pairs_path=str(seed),seed_report=str(seed_report),guard_npz=rr['fixed_guard_npz_path'],guard_report=rr['fixed_guard_report_path'])
 write(O/'constraint-context.json',ctx)
 # Existing depth and target builders bind all emissions to this exact state.
 args=read(W/'depth-argv.json')
 changes={'--candidate':base,'--point-summary':st['point_summary'],'--surface-report':st['surface_report'],'--query-dir':q,'--output-dir':O/'depth','--expected-candidate-sha256':sha(base),'--expected-point-summary-sha256':sha(st['point_summary']),'--expected-surface-report-sha256':sha(st['surface_report']),'--expected-query-receipt-sha256':sha(q/'query-receipt.json')}
 for k,v in changes.items():args[args.index(k)+1]=str(v)
 run(args,O/'depth',True)
 run([PY,(W/'build_direct_depth_targets545.py' if a.point_direction_mode=='direct-depth' else D/'D/coherent-point-targets-390/build_coherent_point_targets_390.py'),'--candidate',base,'--point-summary',st['point_summary'],'--query-dir',q,'--output-dir',O/'point-targets','--clearance-m',str(a.point_margin_m),'--direction-mode',a.point_direction_mode,'--point-depth-rows-npz',O/'depth/point-depth-rows.npz','--point-depth-rows-report',O/'depth/365-compatible-report.json','--minimum-depth-cosine','.05'],O/'points',True)
 run([PY,D/'D/source-outward-proposal-500/build_outward_nearest_union_targets_500.py','--candidate',base,'--surface-report',st['surface_report'],'--query-receipt',q/'query-receipt.json','--output-dir',O/'surface-targets','--clearance-m','.00005','--direction-mode','outward-nearest'],O/'surfaces',True)
 mi=read(W/'matched-inputs-001.json');ip={'candidate':base,'audit159':Path(st['surface_report']),'face_target_npz':O/'surface-targets/targets.npz','point_target_npz':O/'point-targets/nearest-union-exits.npz'}
 mi['inputs']={k:dict(path=str(v),sha256=sha(v))for k,v in ip.items()};write(O/'matched-inputs.json',mi)
 if a.prepare_only:return
 args=read(W/'fit-001-argv.json')
 changes={'--base-candidate':base,'--raw-trial':base,'--self-pairs':seed,'--area-reference-npz':base,'--point-target-npz':ip['point_target_npz'],'--point-target-report':O/'point-targets/nearest-union-exits.json','--face-target-npz':ip['face_target_npz'],'--face-target-report':O/'surface-targets/report.json','--point-depth-rows-npz':O/'depth/point-depth-rows.npz','--point-depth-rows-report':O/'depth/365-compatible-report.json','--fixed-boundary-npz':ctx['guard_npz'],'--fixed-boundary-report':ctx['guard_report'],'--output':O/'fit','--matched-inputs-json-490':O/'matched-inputs.json','--audit159-490':st['surface_report'],'--retained-gain-fraction':str(a.retained_gain)}
 for k,v in changes.items():args[args.index(k)+1]=str(v)
 if a.full_point_fit:
  args[1]=str(W/'run_full_coordinate_point_fit545.py')
  for flag in ['--fine-point-controls','--fine-surface-controls']:
   if flag in args:args.remove(flag)
  for flag,value in [('--trust-mm','8'),('--source-cap-mm','20'),('--area-scp-iterations','4'),('--lp-method','highs-ipm')]:args[args.index(flag)+1]=value
 else:args += ['--point-weight-mode-545',a.point_weight_mode]
 run(args,O/'fit',True)
 if a.full_point_fit:return
 # Existing 525 orchestration extended only for explicit current seed/guard context.
 run([PY,W/'run_checked_batch_context545.py',a.state,O/'fit',O/'constraint-context.json',O/'backtrack'],O/'backtrack')
 selected=read(O/'backtrack/backtrack-373/selected-result.json')
 run([PY,W/'run_candidate_checks545.py','--candidate',selected['candidate'],'--baseline-state',a.state,'--output-dir',O/'arithmetic','--arithmetic-only'],O/'arithmetic')
 new={k:selected[k]for k in ('candidate','point_summary','surface_report')}
 new.update({k+'_sha256':sha(v)for k,v in list(new.items())});new.update(native_admission=False,accepted_physical_steps=0)
 write(O/'state.json',new)
 after=read(new['point_summary']);surface=read(new['surface_report']);sp=sum((b.get('result')or{}).get('count',0)for b in surface['banks'])
 write(O/'progress.json',dict(before_inside=before['inside_node_bank_pairs'],after_inside=after['inside_node_bank_pairs'],surface_pairs=sp,self_pairs=after['exact_PLC_surface_pairs'],candidate_sha256=new['candidate_sha256'],state=str(O/'state.json'),native_admission=False,accepted_physical_steps=0))
 print(json.dumps(read(O/'progress.json')),flush=True)
if __name__=='__main__':main()
