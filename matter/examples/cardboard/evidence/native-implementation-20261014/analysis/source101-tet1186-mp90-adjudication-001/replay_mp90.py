#!/usr/bin/env python3
"""MP90 adjudication of tet1186 center and two captured-direction endpoint pairs."""
from __future__ import annotations
import csv,hashlib,importlib.util,json,math,pathlib,signal,struct,sys,time,warnings
import numpy as np
import mpmath as mp
BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014')
HERE=pathlib.Path(__file__).resolve().parent
CAP=BASE/'solver/reset-augmentation-first-failure-capture-360/operator-capture'
HELPER=BASE/'analysis/medium-tangent-state-sensitivity-1634-002/mp90_endpoint_fd.py'
MP_SOURCE=BASE/'analysis/medium-tangent-state-sensitivity-1634-001/replay.py'
MP_RUNNER=BASE/'analysis/source101-capture360-selected-mp90-002/replay_selected.py'
FP_HELPER=BASE/'analysis/source101-capture360-directional-fd-001/compare_directional_fd.py'
FP64=BASE/'analysis/bilinear15-fp64-replay-267/bilinear15_reference.py'
V3_PACKAGE=BASE/'analysis/source101-capture360-v3-uniform-1728-006'
V3_ROW=V3_PACKAGE/'output-006/all1728_v3_rows.jsonl'
V3_REPORT=V3_PACKAGE/'output-006/report.json'
V3_MANIFEST=V3_PACKAGE/'manifest.json'
EPS_SWEEP=BASE/'analysis/source101-tet1186-epsilon-diagnostic-001/output-001/row1186.json'
OUT=HERE/'output-002'
DPS=90
TARGETS=(1e-7,3e-8)
EXPECTED={
 'capture':'548dce37fbe5a28a609dd6bc45318926072c878774ea0caffa3237d925cebb33',
 'mp90_helper':'042cd63f87206f4cc8698e16d7ec75348637d1b219a0d0f7fc8b148f0d585913',
 'mp90_source':'3214549746cb872fbf089f59e96ce961b239bae3a724f5da7bbeed9f8b0497f0',
 'mp90_runner':'aa73eda519a30807f00b741d3b6cc9b73fb8434e9082fbf39955739572f5c089',
 'direction_helper':'742b0920c4f543e284e41ed18f8dbc78253915a31f39b3ef8d32ff7dec01b5e8',
 'fp64_reference':'c17010564a98fe63da0c8b88be81313685a65cfbd57ff82292e797cab5a0fdf3',
 'v3_rows':'7378794b10863ff2df78b3d5e4d60e6ebdf26abce6791867bfe6d511a9ab0782',
 'v3_epsilon_sweep':'e143b710d07bc62c308680c4ac4f6ddc7f41121a8362131ab71813f448707dae',
}
progress=[]
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path)
 m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def parse_int(t): return -0.0 if t=='-0' else int(t)
def flat32(x): return [float(np.float32(v)) for v in np.asarray(x).reshape(-1)]
def mpf(x):
 if isinstance(x,mp.mpf): return x
 n,d=float(x).as_integer_ratio();return mp.mpf(n)/d
def mp_relabs(a,b):
 av=list(a);bv=list(b);d=[x-y for x,y in zip(av,bv)]
 an=mp.sqrt(sum(abs(x)**2 for x in d));bn=mp.sqrt(sum(abs(x)**2 for x in bv))
 return {'absolute_l2':mp.nstr(an,24),'reference_l2':mp.nstr(bn,24),'relative_l2':mp.nstr(an/bn if bn else mp.nan,24)}
def np_relabs(a,b):
 a=np.asarray(a,float).reshape(-1);b=np.asarray(b,float).reshape(-1);den=float(np.linalg.norm(b))
 return {'absolute_l2':float(np.linalg.norm(a-b)),'relative_l2':float(np.linalg.norm(a-b)/den) if den else None}
def mp_to_np(v): return np.asarray([float(x) for x in v],float)
def append_progress(item):
 progress.append(item)
 with (OUT/'progress.jsonl').open('a') as f:
  f.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n');f.flush()
def alarm_handler(signum,frame): raise TimeoutError('hard CPU wall bound 115 seconds reached')
def main():
 if OUT.exists(): raise RuntimeError(f'refusing to overwrite {OUT}')
 OUT.mkdir(parents=True)
 paths={'capture':CAP,'mp90_helper':HELPER,'mp90_source':MP_SOURCE,'mp90_runner':MP_RUNNER,'direction_helper':FP_HELPER,'fp64_reference':FP64,'v3_rows':V3_ROW,'v3_report':V3_REPORT,'v3_manifest':V3_MANIFEST,'v3_epsilon_sweep':EPS_SWEEP}
 hashes={k:sha(v) for k,v in paths.items()}
 for k,h in EXPECTED.items():
  if hashes[k]!=h: raise RuntimeError(f'{k} hash mismatch {hashes[k]} != {h}')
 # Validate selected uniform-v3 row and report before using its candidate endpoints.
 v3_report=json.loads(V3_REPORT.read_text())
 if v3_report['summary']['uniform_v3_all1728']['row_count']!=1728: raise RuntimeError('v3 report row count mismatch')
 v3=None
 for line in V3_ROW.open():
  row=json.loads(line)
  if int(row['tet_id'])==1186: v3=row;break
 if v3 is None: raise RuntimeError('tet1186 absent from sealed uniform v3 result')
 sweep=json.loads(EPS_SWEEP.read_text())['row']
 if int(sweep['tet_id'])!=1186: raise RuntimeError('tet1186 epsilon sweep mismatch')
 mp.mp.dps=DPS
 runner=load(MP_RUNNER,'mp90_selected_runner_contract')
 helper=load(HELPER,'mp90_equation_helper')
 fphelper=load(FP_HELPER,'fp64_input_helper')
 records=[json.loads(line,parse_int=parse_int) for line in CAP.read_text().splitlines() if line.strip()]
 header=records[0]
 if (header.get('record'),header.get('root'),header.get('newton_iteration'),header.get('world_fingerprint_hex'))!=('header',2,23,'6de3475316b7e8b9'):
  raise RuntimeError('capture identity mismatch')
 if records[-1].get('record')!='integrity' or records[-1].get('production_inputs_bitwise_unchanged') is not True:
  raise RuntimeError('capture integrity missing')
 matches=[r for r in records if r.get('record')=='tetrahedron' and r.get('environment')==0 and r.get('tet_id')==1186]
 if len(matches)!=1: raise RuntimeError('capture tet1186 identity mismatch')
 native=matches[0]; center=native['center']
 if native['material_index']!=0 or native['material_state_count']!=15: raise RuntimeError('unexpected material layout')
 p=flat32(header['parameters_by_environment'][0][0:28]);old=flat32(center['accepted_state'])
 fh,fl,F=runner.pair_words(center,'F_local')
 H=np.asarray(flat32(center['F_direction_local_per_s']),float).reshape(3,3)
 dt=float(np.float32(center['dt_seconds_bulk_modulus_kind_frame_active'][0]))
 F=np.asarray(F,float).reshape(3,3);zhi,zlo,zf=runner.pair_words(center,'projected_state')
 zseed=np.asarray(zf,float)
 pmp,Fmp,oldmp,_,residual,stress=helper.make_equations(p,F.reshape(-1).tolist(),old)
 signal.signal(signal.SIGALRM,alarm_handler);signal.setitimer(signal.ITIMER_REAL,115.0)
 started=time.monotonic();status='completed';error=None
 try:
  # Use the existing MP90 projection contract: exact float32-captured inputs widened
  # by rational conversion, 90-digit equation/tangent evaluation and 1e-70 root gates.
  zc,dc=helper.project_endpoint(residual,pmp,oldmp,zseed.tolist(),mp.mpf('1e-70'),max_iter=48)
  Pc=mp.matrix([stress(*list(zc))[i] for i in range(9)])
  C=helper.refmod.mp_tangent_matrix(pmp,Fmp,oldmp,zc)
  hmp=mp.matrix([mpf(x) for x in H.reshape(-1)])
  tangent_nominal=C*hmp
  append_progress({'kind':'center','converged':dc['converged'],'iterations':dc['iterations'],'residual':dc['terminal_scaled_residual'],'absolute_correction':dc['terminal_newton_correction_inf'],'branch':dc['terminal_branch']})
  results=[]
  for target in TARGETS:
   eps_float=target/float(np.linalg.norm(dt*H))
   eps=mpf(eps_float)
   source_row=v3 if target==1e-7 else sweep
   v3case=next(c for c in source_row['continuation_v3']['epsilon_cases'] if c['target_deltaF_frobenius']==target)
   if abs(float(v3case['epsilon_scalar'])-eps_float)>1e-20*max(1.0,abs(eps_float)):
    raise RuntimeError(f'v3 epsilon mismatch for target {target}: {v3case["epsilon_scalar"]} vs {eps_float}')
   Fp=np.asarray(v3case['plus']['F_endpoint'],float).reshape(3,3)
   Fm=np.asarray(v3case['minus']['F_endpoint'],float).reshape(3,3)
   # Reconstruct endpoint equations from exactly the serialized binary64 F values;
   # their origin is the same captured float32 high+low words and epsilon scalar.
   pp,pF,po,pe,rp,sp=helper.make_equations(p,Fp.reshape(-1).tolist(),old)
   pm_,mF,mo,me,rm,sm=helper.make_equations(p,Fm.reshape(-1).tolist(),old)
   zp,dp=helper.project_endpoint(rp,pp,po,zc,mp.mpf('1e-70'),max_iter=48)
   Pp=mp.matrix([sp(*list(zp))[i] for i in range(9)])
   append_progress({'kind':'endpoint','target':target,'sign':'plus','converged':dp['converged'],'iterations':dp['iterations'],'residual':dp['terminal_scaled_residual'],'absolute_correction':dp['terminal_newton_correction_inf'],'branch':dp['terminal_branch']})
   zm,dm=helper.project_endpoint(rm,pm_,mo,zc,mp.mpf('1e-70'),max_iter=48)
   Pm=mp.matrix([sm(*list(zm))[i] for i in range(9)])
   append_progress({'kind':'endpoint','target':target,'sign':'minus','converged':dm['converged'],'iterations':dm['iterations'],'residual':dm['terminal_scaled_residual'],'absolute_correction':dm['terminal_newton_correction_inf'],'branch':dm['terminal_branch']})
   denom=2*eps*mpf(dt)
   realized=mp.matrix([(mpf(Fp.reshape(-1)[i])-mpf(Fm.reshape(-1)[i]))/denom for i in range(9)])
   tangent_realized=C*realized
   fd=(Pp-Pm)/denom
   v3fd=np.asarray(v3case['central_fd_action_Pa_per_s'],float).reshape(-1)
   v3tan=np.asarray(source_row['continuation_v3']['center_tangent_action_Pa_per_s'],float).reshape(-1)
   row={'target_deltaF_frobenius':target,'epsilon_scalar':eps_float,
    'center_projection':{'converged':dc['converged'],'iterations':dc['iterations'],'scaled_residual':dc['terminal_scaled_residual'],'absolute_correction':dc['terminal_newton_correction_inf'],'branch':dc['terminal_branch']},
    'plus_projection':{'converged':dp['converged'],'iterations':dp['iterations'],'scaled_residual':dp['terminal_scaled_residual'],'absolute_correction':dp['terminal_newton_correction_inf'],'branch':dp['terminal_branch']},
    'minus_projection':{'converged':dm['converged'],'iterations':dm['iterations'],'scaled_residual':dm['terminal_scaled_residual'],'absolute_correction':dm['terminal_newton_correction_inf'],'branch':dm['terminal_branch']},
    'recorded_v3_branch_stable':v3case['branch_signature_stable'],
    'mp90_fd_vs_mp90_matched_tangent':runner.mp_relabs(fd,tangent_realized),
    'mp90_fd_vs_mp90_nominal_tangent':runner.mp_relabs(fd,tangent_nominal),
    'mp90_fd_vs_v3_fd':np_relabs(mp_to_np(fd),v3fd),
    'mp90_tangent_action_vs_v3_tangent_action':np_relabs(mp_to_np(tangent_nominal),v3tan),
    'mp90_center_state_vs_v3_center_state':np_relabs(mp_to_np(zc),np.asarray(v3['continuation_v3']['center_candidate_state_f64'],float)),
    'mp90_plus_state_vs_v3_plus_state':np_relabs(mp_to_np(zp),np.asarray(v3case['plus']['candidate_state_f64'],float)),
    'mp90_minus_state_vs_v3_minus_state':np_relabs(mp_to_np(zm),np.asarray(v3case['minus']['candidate_state_f64'],float)),
    'mp90_center_stress_Pa':[[mp.nstr(Pc[3*i+j],25) for j in range(3)] for i in range(3)],
    'mp90_plus_stress_Pa':[[mp.nstr(Pp[3*i+j],25) for j in range(3)] for i in range(3)],
    'mp90_minus_stress_Pa':[[mp.nstr(Pm[3*i+j],25) for j in range(3)] for i in range(3)],
    'mp90_fd_action_Pa_per_s':[[mp.nstr(fd[3*i+j],25) for j in range(3)] for i in range(3)],
    'mp90_tangent_action_matched_direction_Pa_per_s':[[mp.nstr(tangent_realized[3*i+j],25) for j in range(3)] for i in range(3)],
    'v3_center_fd_relative_error':v3case['central_fd_vs_fp64_tangent']['relative_l2'],
    'v3_target_projection_gates':v3case['center_and_both_endpoints_v3_converged']}
   results.append(row)
  status='completed'
 except TimeoutError as e:
  status='timed_out';error=str(e)
 finally:
  signal.setitimer(signal.ITIMER_REAL,0)
 elapsed=time.monotonic()-started
 result={'schema':'numi.cardboard.source101-tet1186-mp90-adjudication.v1','status':status,
  'classification':'Bounded independent MP90 constitutive replay for one captured tet/direction; no native or physical qualification.',
  'input_bindings':{k:{'path':str(v),'sha256':hashes[k]} for k,v in paths.items()},
  'capture_identity':{'root':2,'newton_iteration':23,'world_fingerprint_hex':header['world_fingerprint_hex'],'environment':0,'tet_id':1186,'material_index':0},
  'precision':{'mpmath_dps':DPS,'root_scaled_residual_and_absolute_correction_tolerance':'1e-70','max_newton_iterations':48,'equations':'existing independent MP90 source-equation helper; no physical-law changes'},
  'inputs':'Captured float32 parameters, accepted 15-state history, F high+low words, captured dt and direction; endpoint F values are the exact binary64 endpoints recorded by sealed uniform v3 result. Accepted history remains unchanged.',
  'targets':list(TARGETS),'v3_comparison':'sealed package006 provides target 1e-7; sealed single-row v3 epsilon diagnostic provides target 3e-8. Both share the same capture and accepted history.',
  'progress':progress,'results':results,'warnings':[],'elapsed_seconds':elapsed,'timeout_error':error,
  'limits':['Only tet1186 and two captured-direction perturbation magnitudes are replayed.','No universal FP64 noise floor, full-row gate, native FP32 claim, or solver qualification is inferred.']}
 (OUT/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'status':status,'result':str(OUT/'result.json'),'elapsed_seconds':elapsed,'progress_count':len(progress),
   'targets':[{'target':x['target_deltaF_frobenius'],'mp90_fd_vs_tangent':x['mp90_fd_vs_mp90_matched_tangent']['relative_l2'],'mp90_vs_v3_fd':x['mp90_fd_vs_v3_fd']['relative_l2'],'mp90_vs_v3_tangent':x['mp90_tangent_action_vs_v3_tangent_action']['relative_l2'],'projections':[x[k]['converged'] for k in ['center_projection','plus_projection','minus_projection']]} for x in results]},indent=2,allow_nan=False))
if __name__=='__main__': main()
