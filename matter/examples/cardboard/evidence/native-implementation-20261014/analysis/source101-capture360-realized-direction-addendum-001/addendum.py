#!/usr/bin/env python3
"""Compare v3 FP64 tangent against nominal and endpoint-realized directions."""
from __future__ import annotations
import hashlib, importlib.util, json, math, pathlib, signal, sys, time, warnings
import numpy as np
BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
HERE=pathlib.Path(__file__).resolve().parent
CAP=BASE/'solver/reset-augmentation-first-failure-capture-360/operator-capture'
ROWS=BASE/'analysis/source101-capture360-v3-uniform-1728-006/output-006/all1728_v3_rows.jsonl'
REPORT=BASE/'analysis/source101-capture360-v3-uniform-1728-006/output-006/report.json'
REF=BASE/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
HELP=BASE/'analysis/source101-capture360-directional-fd-001/compare_directional_fd.py'
OUT=HERE/'output-001'
EXPECTED={'capture':'548dce37fbe5a28a609dd6bc45318926072c878774ea0caffa3237d925cebb33',
 'rows':'7378794b10863ff2df78b3d5e4d60e6ebdf26abce6791867bfe6d511a9ab0782',
 'report':'dd6bb021cbd0e9b51425c19dd635ad31fc85a660bff667eb7b4a3e2ce39c690a',
 'reference':'c17010564a98fe63da0c8b88be81313685a65cfbd57ff82292e797cab5a0fdf3',
 'direction_helper':'742b0920c4f543e284e41ed18f8dbc78253915a31f39b3ef8d32ff7dec01b5e8'}
TARGET=1e-7
DEADLINE_SECONDS=55.0
CHUNK=48
class BoundReached(Exception): pass
def alarm(signum,frame): raise BoundReached('55 second CPU bound reached')
def sha(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def parse_int(t):return -0.0 if t=='-0' else int(t)
def f32(x):return np.asarray(x,dtype=np.float32).astype(np.float64)
def pair(obj,key):return f32(obj[key])+f32(obj.get(key+'_low',np.zeros_like(obj[key])))
def norm(x):return np.linalg.norm(np.asarray(x).reshape(len(x),-1),axis=1)
def rel(actual,reference):
 a=np.asarray(actual).reshape(len(actual),-1);b=np.asarray(reference).reshape(len(reference),-1)
 dn=np.linalg.norm(a-b,axis=1);bn=np.linalg.norm(b,axis=1)
 return np.divide(dn,bn,out=np.full_like(dn,np.nan),where=bn>0)
def branch_key(b):return (b.get('multiplier_sign'),b.get('yield_sign'),b.get('fb_selection'))
def summarize(rows,key):
 x=np.asarray([r[key] for r in rows],dtype=float);finite=x[np.isfinite(x)]
 return {'finite_count':int(len(finite)),'max':float(finite.max()) if len(finite) else None,
  'max_tet_id':int(rows[int(np.nanargmax(x))]['tet_id']) if len(finite) else None,
  'median':float(np.median(finite)) if len(finite) else None,
  'p95':float(np.percentile(finite,95)) if len(finite) else None,
  'le_1e-6_count':int(np.sum(finite<=1e-6)),'gt_1e-6_count':int(np.sum(finite>1e-6))}
def main():
 if any(OUT.iterdir()):raise SystemExit('refusing to overwrite nonempty output directory')
 paths={'capture':CAP,'rows':ROWS,'report':REPORT,'reference':REF,'direction_helper':HELP}
 hashes={k:sha(v) for k,v in paths.items()}
 for k,v in EXPECTED.items():
  if hashes[k]!=v:raise RuntimeError(f'{k} SHA mismatch: {hashes[k]} != {v}')
 cap=[json.loads(s,parse_int=parse_int) for s in CAP.read_text().splitlines() if s.strip()]
 head=cap[0]
 if (head.get('record'),head.get('root'),head.get('newton_iteration'),head.get('world_fingerprint_hex'))!=('header',2,23,'6de3475316b7e8b9'):raise RuntimeError('capture identity mismatch')
 if cap[-1].get('record')!='integrity' or cap[-1].get('production_inputs_bitwise_unchanged') is not True:raise RuntimeError('capture integrity missing')
 tets={int(r['tet_id']):r for r in cap if r.get('record')=='tetrahedron' and r.get('environment')==0}
 if len(tets)!=1920:raise RuntimeError('capture tet domain mismatch')
 rows=[json.loads(s,parse_int=parse_int) for s in ROWS.read_text().splitlines() if s.strip()]
 if len(rows)!=1728 or len({int(r['tet_id']) for r in rows})!=1728:raise RuntimeError('v3 row domain mismatch')
 if set(map(int,(r['tet_id'] for r in rows))) != set(t for t in tets if t in range(864) or 960<=t<1824):raise RuntimeError('capture/v3 selected-domain mismatch')
 ref=load(REF,'realized_direction_bilinear15_reference');helper=load(HELP,'realized_direction_input_helper')
 ptab=helper.f32(head['parameters_by_environment'][0])
 dt_env=float(np.float32(head['dt_seconds_by_environment'][0]))
 # Flatten source records in deterministic sorted row order; material params are exact captured FP32 widened to FP64.
 data=[]
 for row in sorted(rows,key=lambda x:int(x['tet_id'])):
  tid=int(row['tet_id']);native=tets[tid];center=native['center'];mat=int(native['material_index'])
  p=f32(ptab[mat*28:(mat+1)*28]);old=f32(center['accepted_state']);F=pair(center,'F_local').reshape(3,3)
  h=f32(center['F_direction_local_per_s']).reshape(3,3);dt=float(np.float32(center['dt_seconds_bulk_modulus_kind_frame_active'][0]))
  if dt!=dt_env:raise RuntimeError(f'tet{tid}: dt mismatch')
  if row['continuation_v3']['center_projection']['converged'] is not True:raise RuntimeError(f'tet{tid}: center projection not admitted')
  case=next((c for c in row['continuation_v3']['epsilon_cases'] if c['target_deltaF_frobenius']==TARGET),None)
  if case is None:raise RuntimeError(f'tet{tid}: common step absent')
  if case['center_and_both_endpoints_v3_converged'] is not True:raise RuntimeError(f'tet{tid}: endpoint projection not admitted')
  if case['branch_signature_stable'] is not True:raise RuntimeError(f'tet{tid}: recorded branch predicates unstable')
  Fp=np.asarray(case['plus']['F_endpoint'],dtype=np.float64).reshape(3,3);Fm=np.asarray(case['minus']['F_endpoint'],dtype=np.float64).reshape(3,3)
  eps=float(case['epsilon_scalar']);z=np.asarray(row['continuation_v3']['center_candidate_state_f64'],dtype=np.float64)
  Hreal=(Fp-Fm)/(2*eps*dt)
  fd=np.asarray(case['central_fd_action_Pa_per_s'],dtype=np.float64).reshape(3,3)
  hnom=np.asarray(row['continuation_v3']['center_tangent_action_Pa_per_s'],dtype=np.float64).reshape(3,3)
  data.append((row,p,F,old,z,h,Hreal,dt,eps,Fp,Fm,fd,hnom,case))
 signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,DEADLINE_SECONDS)
 start=time.monotonic();result_rows=[];warnings_seen=[];status='completed';err=None
 try:
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always')
   for i in range(0,len(data),CHUNK):
    batch=data[i:i+CHUNK]
    P=np.stack([d[1] for d in batch]);F=np.stack([d[2] for d in batch]);old=np.stack([d[3] for d in batch]);z=np.stack([d[4] for d in batch]);H=np.stack([d[6] for d in batch])
    realized=np.asarray(ref.tangent(P,F,old,z,H),dtype=np.float64)
    for j,d in enumerate(batch):
     row,p,ff,oldv,zv,h,hr,dt,eps,Fp,Fm,fd,nom,case=d
     r_act=float(rel(realized[j:j+1],fd[None])[0]);r_nom=float(rel(nom[None],fd[None])[0])
     hdiff=float(np.linalg.norm(hr-h)/np.linalg.norm(h)) if np.linalg.norm(h)>0 else None
     actdiff=float(np.linalg.norm(realized[j]-nom)/np.linalg.norm(nom)) if np.linalg.norm(nom)>0 else None
     mid=(Fp+Fm)*0.5-ff
     plus=np.linalg.norm(Fp-ff);minus=np.linalg.norm(Fm-ff)
     result_rows.append({'tet_id':int(row['tet_id']),'material_index':int(tets[int(row['tet_id'])]['material_index']),
      'target_deltaF_frobenius':TARGET,'epsilon_scalar':eps,'center_and_endpoints_v3_converged':case['center_and_both_endpoints_v3_converged'],
      'captured_branch_signature_stable':case['branch_signature_stable'],'center_branch':row['continuation_v3']['center_branch'],
      'plus_branch':case['plus']['continuation']['final_metrics']['branch'],'minus_branch':case['minus']['continuation']['final_metrics']['branch'],
      'actual_deltaF_plus_frobenius':float(plus),'actual_deltaF_minus_frobenius':float(minus),
      'midpoint_shift_frobenius':float(np.linalg.norm(mid)),
      'realized_rate_direction_vs_captured_relative_l2':hdiff,
      'realized_tangent_action_vs_nominal_relative_l2':actdiff,
      'fd_vs_nominal_tangent_relative_l2':r_nom,'fd_vs_realized_tangent_relative_l2':r_act,
      'within_1e-6_nominal':bool(r_nom<=1e-6),'within_1e-6_realized':bool(r_act<=1e-6)})
    warnings_seen=[f'{w.category.__name__}: {w.message} ({w.filename}:{w.lineno})' for w in caught]
 except BoundReached as e:
  status='timed_out_partial';err=str(e)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
 elapsed=time.monotonic()-start
 result_rows.sort(key=lambda x:x['tet_id'])
 (OUT/'realized_direction_rows.jsonl').write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in result_rows))
 selected=next((x for x in result_rows if x['tet_id']==1186),None)
 report={'schema':'numi.cardboard.source101-realized-direction-addendum.v1','status':status,
  'classification':'CPU-only diagnostic reusing sealed v3 center/endpoints; no new projection roots, native execution, or physical qualification.',
  'source_bindings':{k:{'path':str(v),'sha256':hashes[k]} for k,v in paths.items()},
  'capture_identity':{'root':2,'newton_iteration':23,'world_fingerprint_hex':head['world_fingerprint_hex'],'environment':0},
  'selection':{'rows_expected':1728,'rows_completed':len(result_rows),'target_deltaF_frobenius':TARGET,'accepted_history':'captured FP32 accepted 15-state, fixed across center/endpoints','center_candidate':'sealed v3 converged FP64 center projection','tangent_reference':'independent bilinear15 FP64 implicit tangent at v3 center, applied once to nominal captured H and once to H_realized=(Fplus-Fminus)/(2*epsilon*dt).'},
  'metrics':{'fd_vs_nominal_tangent':summarize(result_rows,'fd_vs_nominal_tangent_relative_l2') if result_rows else None,
   'fd_vs_realized_tangent':summarize(result_rows,'fd_vs_realized_tangent_relative_l2') if result_rows else None,
   'realized_rate_direction_vs_captured':summarize(result_rows,'realized_rate_direction_vs_captured_relative_l2') if result_rows else None,
   'realized_tangent_action_vs_nominal':summarize(result_rows,'realized_tangent_action_vs_nominal_relative_l2') if result_rows else None,
   'projection_and_recorded_branch_stable_count':sum(x['center_and_endpoints_v3_converged'] and x['captured_branch_signature_stable'] for x in result_rows),
   'nominal_strict_count':sum(x['within_1e-6_nominal'] for x in result_rows),'realized_strict_count':sum(x['within_1e-6_realized'] for x in result_rows),
   'tet1186':selected},
  'warnings':warnings_seen,'elapsed_seconds':elapsed,'timeout_error':err,
  'limits':['The reported center/endpoints and nominal common-step gate come from sealed package006.','Changing tangent input direction to measured endpoint direction is a diagnostic decomposition, not a corrected validation gate.','No universal arithmetic noise floor is estimated; sub-micro-Pa cancellation at this point cannot be generalized.','The common 1e-7 nominal-direction all-row gate remains as reported in package006; this addendum does not promote any alternate gate.']}
 (OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':status,'rows':len(result_rows),'elapsed_seconds':elapsed,'report':str(OUT/'report.json'),'tet1186':selected,'nominal':report['metrics']['fd_vs_nominal_tangent'],'realized':report['metrics']['fd_vs_realized_tangent'],'warnings':len(warnings_seen)},indent=2,allow_nan=False))
if __name__=='__main__':main()
