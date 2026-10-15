#!/usr/bin/env python3
"""Isolated FP64 diagnostic solve for source125/native424 plus endpoint tet 1691.
Uses the existing independent 15-state equations and accepted history. No native
execution, source mutation, or relaxation of the published residual/correction gates.
"""
import datetime, hashlib, importlib.util, json, math, pathlib, sys, warnings
import numpy as np
from scipy.optimize import least_squares

BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
PKG=BASE/'analysis/source125-medium1691-independent-endpoint-004'
SOURCE=BASE/'snapshots/source-status5-vtccd-diagnostics-125'
CAP=BASE/'solver/two-stage-first-failure-rich-fd-419/operator-capture'
SWEEP=BASE/'analysis/native-residual-fd-sweep-result-424'
REPORT003=BASE/'analysis/source125-kinematics-and-native424-force-audit-003/report.json'
SCRIPT003=BASE/'analysis/source125-kinematics-and-native424-force-audit-003/audit.py'
REFPATH=BASE/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
FDHELP=BASE/'analysis/source101-capture360-directional-fd-001/compare_directional_fd.py'
E2=BASE/'analysis/operator-fd-e2-adapter-001'
TARGET_TET=1691
EPS=float(np.float32(1/16384))


def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()

def load(path,name):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def norm_inf(x): return float(np.max(np.abs(np.asarray(x,dtype=np.float64))))
def vecnorm(x): return math.sqrt(math.fsum(float(q)*float(q) for q in np.asarray(x).ravel()))

def solve_checked(ref,p,F,old,z0,max_updates=128,max_backtracks=64):
 """Newton with original residual merit and stricter unchanged final gates."""
 z=np.asarray(z0,dtype=np.float64).copy(); S=np.array([1e4]*6+[1.0]*6+[1e4]*3,dtype=np.float64)
 history=[]
 for it in range(max_updates+1):
  r=ref.residual(p,F,old,z); rn=norm_inf(r*S)
  J=ref.jacobian(p,F,old,z)
  try: corr=np.linalg.solve(J,-r)
  except Exception as e:
   history.append({'iteration':it,'scaled_residual':rn,'event':'jacobian_solve_failed','error':f'{type(e).__name__}: {e}'})
   return {'ok':False,'reason':'jacobian_solve_failed','z':z,'history':history}
  scale=np.maximum(np.maximum(np.abs(old),np.abs(z)),float(p[26]))
  cn=norm_inf(corr/scale)
  b=branch(ref,p,old,z)
  row={'iteration':it,'scaled_residual':rn,'dynamic_scaled_correction':cn,'branch':b}
  if rn<=2e-10 and cn<=2e-8:
   row['event']='strict_gates_pass'; history.append(row)
   return {'ok':True,'z':z,'history':history,'scaled_residual':rn,'scaled_correction':cn}
  if it==max_updates:
   row['event']='update_budget_exhausted'; history.append(row)
   return {'ok':False,'reason':'diagnostic_update_budget_exhausted','z':z,'history':history,'scaled_residual':rn,'scaled_correction':cn}
  accepted=False; alpha=1.0; trials=[]
  for bt in range(max_backtracks+1):
   cand=z+alpha*corr
   rc=ref.residual(p,F,old,cand); nc=norm_inf(rc*S)
   trials.append({'backtrack':bt,'alpha':alpha,'scaled_residual':nc,
                  'finite':bool(np.isfinite(nc)),
                  'branch':branch(ref,p,old,cand) if np.isfinite(nc) else None})
   if np.isfinite(nc) and nc<rn:
    z=cand; accepted=True; row['accepted_alpha']=alpha; row['accepted_scaled_residual']=nc; row['accepted_branch']=trials[-1]['branch']; break
   alpha*=0.5
  row['trials']=trials
  if not accepted:
   row['event']='line_search_stalled'; history.append(row)
   return {'ok':False,'reason':'diagnostic_line_search_stalled','z':z,'history':history,'scaled_residual':rn,'scaled_correction':cn}
  row['event']='update_accepted'; history.append(row)
 return {'ok':False,'reason':'unexpected_return','z':z,'history':history}

def branch(ref,p,old,z):
 q,_=ref.hill(p[None,:],z[None,6:12]); q=float(q[0])
 radius=1.0+p[17]*p[19]*p[21]*z[13]/p[10]+p[18]*p[20]*p[22]*z[14]/p[10]
 return {'plastic_multiplier_increment':float(z[12]-old[12]),
         'lambda_active':bool(z[12]>old[12]),
         'yield_gap':float(radius-q),
         'fb_denominator':float(math.sqrt((z[12]-old[12])**2+(radius-q)**2+p[26]**2)+(z[12]-old[12])+(radius-q))}

def main():
 report003=json.loads(REPORT003.read_text())
 assert report003['status']=='diagnostic_inconclusive_one_endpoint_projection_failed'
 per=load(SCRIPT003,'audit003').load_module if False else None
 # Reproduce exact input extraction using source-transcription helpers frozen in audit003.
 a=load(SCRIPT003,'audit003')
 ref=load(REFPATH,'ref15_1691')
 records=[json.loads(line) for line in CAP.read_text().splitlines() if line.strip()]
 h=records[0]; tets=sorted((r for r in records if r.get('record')=='tetrahedron' and r.get('environment')==0),key=lambda r:int(r['tet_id']))
 nodes=sorted((r for r in records if r.get('record')=='node' and r.get('environment')==0),key=lambda r:int(r['node_id']))
 assert len(tets)==1920 and len(nodes)==882 and int(tets[TARGET_TET]['material_index'])==1
 invrest=np.asarray([a.f32(t['inverse_rest_rows_per_m']) for t in tets],dtype=np.float64)
 frame32=np.asarray([a.frame_from_xyzw(t['material_frame_quaternion_xyzw']) for t in tets],dtype=np.float32)
 active=np.asarray([int(a.f32([t['center']['dt_seconds_bulk_modulus_kind_frame_active'][3]])[0])==1 for t in tets])
 headerp=a.f32(h['parameters_by_environment'][0]); mat={int(x['index']):x for x in h['materials']}
 t=tets[TARGET_TET]; m=mat[int(t['material_index'])]; p=a.f32(headerp[int(m['parameter_offset']):int(m['parameter_offset'])+len(m['parameter_names'])])
 old=a.f32(t['center']['accepted_state'])
 # center/plus/minus exact FP32 nodal endpoints from source-bound sweep 424.
 def load_nodes(stem):
  arr=np.fromfile(SWEEP/'result'/f'{stem}-nodes.bin',dtype='<f4')
  assert arr.size==2*882*5*4
  return arr.reshape(2,882,5,4)[0]
 center_nodes=load_nodes('sample-0-0'); plus_nodes=load_nodes('sample-6-0'); minus_nodes=load_nodes('sample-6-1')
 Fhi,Flo=a.source_deformation_words(plus_nodes,tets,invrest,frame32,active)
 Fplus=Fhi[TARGET_TET].astype(np.float64)+Flo[TARGET_TET].astype(np.float64)
 Fchi,Fclo=a.source_deformation_words(center_nodes,tets,invrest,frame32,active)
 Fcenter=Fchi[TARGET_TET].astype(np.float64)+Fclo[TARGET_TET].astype(np.float64)
 Fmhi,Fmlo=a.source_deformation_words(minus_nodes,tets,invrest,frame32,active)
 Fminus=Fmhi[TARGET_TET].astype(np.float64)+Fmlo[TARGET_TET].astype(np.float64)
 assert np.all(np.isfinite(Fplus))
 # Standard start matches reference projector's elastic stress initialization.
 z_el=old.copy(); z_el[6:12]=ref.stiffness(p[None,:])[0]@(ref.strain(Fplus[None,:,:])[0]-old[:6])
 standard=solve_checked(ref,p,Fplus,old,z_el,max_updates=128,max_backtracks=64)
 # Center and minus independently solved with the same accepted history. Use these as
 # continuation anchors, not as alternate history or altered constitutive equations.
 zc0,_,_=ref.project(p[None,:],Fcenter[None,:,:],old[None,:],maxiter=128)
 zm0,_,_=ref.project(p[None,:],Fminus[None,:,:],old[None,:],maxiter=128)
 center_anchor=solve_checked(ref,p,Fcenter,old,zc0[0],max_updates=32,max_backtracks=64)
 minus_anchor=solve_checked(ref,p,Fminus,old,zm0[0],max_updates=32,max_backtracks=64)
 # Fixed-F continuation from the converged minus solution toward exact plus F;
 # only Newton initialization/path changes, equations/history/gates do not.
 def continue_path(anchor,steps):
  z=anchor['z'].copy(); path=[]
  start=Fminus.copy()
  for k in range(1,steps+1):
   tau=k/steps; Fi=start+(Fplus-start)*tau
   rr=solve_checked(ref,p,Fi,old,z,max_updates=32,max_backtracks=64)
   path.append({'step':k,'fraction':tau,'result':{k:v for k,v in rr.items() if k not in ('z','history')},'iterations':len(rr['history']),
                'last_state':rr['z'].tolist(),'last_branch':branch(ref,p,old,rr['z'])})
   if not rr['ok']:
    return {'ok':False,'steps_requested':steps,'steps_completed':k,'path':path,'z':rr['z']}
   z=rr['z']
  # verify target exactly the native-source-transcribed plus matrix
  final=solve_checked(ref,p,Fplus,old,z,max_updates=32,max_backtracks=64)
  return {'ok':final['ok'],'steps_requested':steps,'steps_completed':steps,'path':path,
          'final':{k:v for k,v in final.items() if k not in ('z','history')},'final_state':final['z'].tolist(),
          'final_history':final['history'],'z':final['z']}
 paths=[]
 for steps in (8,32,128):
  outcome=continue_path(minus_anchor,steps)
  paths.append({k:v for k,v in outcome.items() if k!='z'})
  if outcome['ok']:
   candidate=outcome['z']; break
 else: candidate=None
 # Derivative gate and stress/branch are assessed only if unchanged strict gates pass.
 final_extra=None
 if candidate is not None:
  residual=ref.residual(p[None,:],Fplus[None,:,:],old[None,:],candidate[None,:])[0]
  corr=np.linalg.solve(ref.jacobian(p[None,:],Fplus[None,:,:],old[None,:],candidate[None,:])[0],-residual)
  S=np.array([1e4]*6+[1.0]*6+[1e4]*3)
  dynamic=np.maximum(np.maximum(np.abs(old),np.abs(candidate)),p[26])
  st=ref.stress(p[None,:],Fplus[None,:,:],candidate[None,:])[0]
  final_extra={'state':candidate.tolist(),'scaled_residual_inf':norm_inf(residual*S),
               'dynamic_scaled_correction_inf':norm_inf(corr[0]/dynamic),
               'strict_residual_gate':norm_inf(residual*S)<=2e-10,
               'strict_correction_gate':norm_inf(corr[0]/dynamic)<=2e-8,
               'stress_Pa':st.tolist(),'branch':branch(ref,p,old,candidate)}
 # Retain row failure and solver attempt details from unchanged 003.
 original_failure={'reason':'RuntimeError: FP64 return map stalled iteration1 rows[0] norm[90.81492362]',
                   'source':'audit003/report.json and compressed per-tet row 1691',
                   '003_row':next(json.loads(x) for x in __import__('gzip').open(BASE/'analysis/source125-kinematics-and-native424-force-audit-003/per-paper-tet.jsonl.gz','rt') if json.loads(x)['tet_id']==TARGET_TET)}
 payload={'schema':'numi.cardboard.source125-medium1691-fp64-endpoint-diagnostic.v1',
  'status':'continuation_recovered_strict_endpoint' if candidate is not None else 'unresolved_independent_endpoint',
  'scope':'single source125/native424 plus endpoint, medium tet 1691; independent CPU only; fixed accepted history',
  'bindings':{'source_snapshot':str(SOURCE),'source_runtime_mm_sha256':sha(SOURCE/'matter/src/runtime.mm'),
    'source_fem_metalinc_sha256':sha(SOURCE/'matter/src/metal/fem.metalinc'),
    'capture419_sha256':sha(CAP),'native424_receipt_sha256':sha(SWEEP/'receipt.json'),
    'native424_audit_sha256':sha(BASE/'analysis/native424-residual-sweep-audit-001/report.json'),
    'audit003_report_sha256':sha(REPORT003),'audit003_script_sha256':sha(SCRIPT003),
    'reference_sha256':sha(REFPATH),'endpoint_nodes_sha256':sha(SWEEP/'result/sample-6-0-nodes.bin'),
    'center_nodes_sha256':sha(SWEEP/'result/sample-0-0-nodes.bin'),'minus_nodes_sha256':sha(SWEEP/'result/sample-6-1-nodes.bin')},
  'input':{'tet_id':TARGET_TET,'material_index':int(t['material_index']),'material':m['name'],
    'parameters_fp32_promoted':p.tolist(),'accepted_history_fp32_promoted':old.tolist(),
    'F_center_source_transcribed':Fcenter.tolist(),'F_minus_source_transcribed':Fminus.tolist(),
    'F_plus_source_transcribed':Fplus.tolist(),'epsilon':EPS,
    'F_plus_high_words':Fhi[TARGET_TET].tolist(),'F_plus_low_words':Flo[TARGET_TET].tolist()},
  'original_failure':original_failure,
  'standard_plus_newton':{k:v for k,v in standard.items() if k not in ('z',)},
  'center_anchor':{k:v for k,v in center_anchor.items() if k not in ('z','history')},
  'minus_anchor':{k:v for k,v in minus_anchor.items() if k not in ('z','history')},
  'continuation_attempts':paths,'recovered_endpoint':final_extra,
  'solver_contract':{'equations':'unchanged bilinear15_reference residual/jacobian/stress',
    'history':'same captured accepted state for all endpoints and continuation fractions',
    'residual_gate':'max(abs(residual)*[1e4x6,1x6,1e4x3]) <= 2e-10',
    'dynamic_correction_gate':'max(abs(J^-1 r)/max(abs(old),abs(z),p26)) <= 2e-8',
    'line_search':'strict decrease of unchanged scaled infinity residual; diagnostic up to64 backtracks',
    'iteration_budget':'up to128 Newton updates for direct attempt; 32 per continuation step; only diagnostic algorithm/budget changed'},
  'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (PKG/'result.json').write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':payload['status'],'standard':{k:v for k,v in payload['standard_plus_newton'].items() if k!='history'},
   'center_anchor':payload['center_anchor'],'minus_anchor':payload['minus_anchor'],
   'continuations':[{'steps_requested':x['steps_requested'],'steps_completed':x['steps_completed'],'ok':x['ok']} for x in paths],
   'endpoint': None if final_extra is None else {k:v for k,v in final_extra.items() if k not in ('state','stress_Pa')},
   'result_path':str(PKG/'result.json')},indent=2,allow_nan=False))

if __name__=='__main__': main()
