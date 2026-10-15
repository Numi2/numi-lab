#!/usr/bin/env python3
"""Decompose tet1186 nominal-vs-realized tangent direction and stress roundoff."""
import csv,hashlib,importlib.util,json,pathlib,sys
import numpy as np,mpmath as mp
BASE=pathlib.Path('/Users/home/cardboard-implementation-20261014');HERE=pathlib.Path(__file__).resolve().parent
CAP=BASE/'solver/reset-augmentation-first-failure-capture-360/operator-capture'
MPH=BASE/'analysis/medium-tangent-state-sensitivity-1634-002/mp90_endpoint_fd.py'
MPR=BASE/'analysis/source101-capture360-selected-mp90-002/replay_selected.py'
HELP=BASE/'analysis/source101-capture360-directional-fd-001/compare_directional_fd.py'
V3=BASE/'analysis/source101-capture360-v3-uniform-1728-006/output-006/all1728_v3_rows.jsonl'
SW=BASE/'analysis/source101-tet1186-epsilon-diagnostic-001/output-001/row1186.json'
MPRESULT=HERE/'output-002/result.json';OUT=HERE/'output-003'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def load(p,n):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
def flat32(x):return [float(np.float32(v)) for v in np.asarray(x).reshape(-1)]
def mpf(x):
 n,d=float(x).as_integer_ratio();return mp.mpf(n)/d
def nrm(v):return mp.sqrt(sum(abs(x)**2 for x in v))
def metric(a,b):
 d=[x-y for x,y in zip(a,b)];na=nrm(d);nb=nrm(b)
 return {'absolute_l2':mp.nstr(na,24),'reference_l2':mp.nstr(nb,24),'relative_l2':mp.nstr(na/nb if nb else mp.nan,24)}
def main():
 if OUT.exists():raise RuntimeError('refusing overwrite')
 OUT.mkdir()
 paths={'capture':CAP,'mp90_helper':MPH,'mp90_runner':MPR,'direction_helper':HELP,'v3_rows':V3,'single_row_sweep':SW,'mp90_result':MPRESULT}
 hashes={k:sha(v) for k,v in paths.items()}
 mp.mp.dps=90
 runner=load(MPR,'roundoff_runner');helper=load(MPH,'roundoff_helper');fd=load(HELP,'roundoff_fp32')
 cap=[json.loads(x,parse_int=lambda t:-0.0 if t=='-0' else int(t)) for x in CAP.read_text().splitlines() if x.strip()]
 header=cap[0];native=next(r for r in cap if r.get('record')=='tetrahedron' and r.get('environment')==0 and int(r.get('tet_id',-1))==1186);c=native['center']
 p=flat32(header['parameters_by_environment'][0][0:28]);old=flat32(c['accepted_state'])
 fh,fl,F=runner.pair_words(c,'F_local');H=np.asarray(flat32(c['F_direction_local_per_s']),float).reshape(3,3);dt=float(np.float32(c['dt_seconds_bulk_modulus_kind_frame_active'][0]))
 zhi,zlo,zf=runner.pair_words(c,'projected_state')
 pmp,Fmp,oldmp,_,residual,stress=helper.make_equations(p,F,old)
 zc,diag=helper.project_endpoint(residual,pmp,oldmp,zf,mp.mpf('1e-70'),max_iter=48)
 C=helper.refmod.mp_tangent_matrix(pmp,Fmp,oldmp,zc);hmp=mp.matrix([mpf(x) for x in H.reshape(-1)]);nom=C*hmp
 rowv3=next(json.loads(line) for line in V3.open() if int(json.loads(line)['tet_id'])==1186)
 sweeprow=json.loads(SW.read_text())['row']
 mpresult=json.loads(MPRESULT.read_text());out=[]
 for target,source in [(1e-7,rowv3),(3e-8,sweeprow)]:
  cased=next(x for x in source['continuation_v3']['epsilon_cases'] if x['target_deltaF_frobenius']==target)
  Fp=np.asarray(cased['plus']['F_endpoint'],float).reshape(-1);Fm=np.asarray(cased['minus']['F_endpoint'],float).reshape(-1)
  eps=mpf(cased['epsilon_scalar']);dtm=mpf(dt)
  realized=mp.matrix([(mpf(Fp[i])-mpf(Fm[i]))/(2*eps*dtm) for i in range(9)])
  actual=C*realized
  v3act=next(x for x in mpresult['results'] if x['target_deltaF_frobenius']==target)
  mpplus=np.array([[float(q) for q in row] for row in v3act['mp90_plus_stress_Pa']])
  mpminus=np.array([[float(q) for q in row] for row in v3act['mp90_minus_stress_Pa']])
  v3plus=np.asarray(cased['plus']['stress_Pa'],float);v3minus=np.asarray(cased['minus']['stress_Pa'],float)
  out.append({'target_deltaF_frobenius':target,'epsilon_scalar':float(cased['epsilon_scalar']),
   'realized_rate_direction_vs_captured_direction':metric(realized,hmp),
   'tangent_action_actual_realized_vs_nominal':metric(actual,nom),
   'mp90_fd_vs_mp90_tangent_realized':v3act['mp90_fd_vs_mp90_matched_tangent']['relative_l2'],
   'mp90_fd_vs_mp90_tangent_nominal':v3act['mp90_fd_vs_mp90_nominal_tangent']['relative_l2'],
   'fp64_endpoint_stress_vs_mp90':{'plus_abs_Pa':float(np.linalg.norm(v3plus-mpplus)),'plus_relative':float(np.linalg.norm(v3plus-mpplus)/np.linalg.norm(mpplus)),'minus_abs_Pa':float(np.linalg.norm(v3minus-mpminus)),'minus_relative':float(np.linalg.norm(v3minus-mpminus)/np.linalg.norm(mpminus))},
   'fp64_central_fd_vs_mp90_fd':v3act['mp90_fd_vs_v3_fd'],
   'v3_central_fd_vs_nominal_tangent':cased['central_fd_vs_fp64_tangent']['relative_l2'],
   'v3_projection_state_vs_mp90_root':{'center':v3act['mp90_center_state_vs_v3_center_state'],'plus':v3act['mp90_plus_state_vs_v3_plus_state'],'minus':v3act['mp90_minus_state_vs_v3_minus_state'],
    'mp90_center_scaled_residual':diag['terminal_scaled_residual'],'mp90_center_absolute_correction':diag['terminal_newton_correction_inf']},
   'yield_gap_center_mp90':diag['terminal_branch']['yield_gap'],'yield_gap_plus_mp90':v3act['plus_projection']['branch']['yield_gap'],'yield_gap_minus_mp90':v3act['minus_projection']['branch']['yield_gap'],
   'projection_converged_center_plus_minus':all(v3act[k]['converged'] for k in ['center_projection','plus_projection','minus_projection'])})
 result={'schema':'numi.cardboard.source101-tet1186-mp90-direction-roundoff-analysis.v1','classification':'CPU-only explanatory decomposition for one captured point/direction; no universal noise-floor or full validation claim.','inputs':{k:{'path':str(v),'sha256':hashes[k]} for k,v in paths.items()},'mpmath_dps':90,'accepted_history_fixed':True,'projection_gate':'MP90 scaled residual and absolute correction <=1e-70; max48 iterations','results':out,
  'interpretation':'Separate the tangent evaluated on the nominal captured rate direction from the direction actually represented by the FP64 endpoint pair. MP90 FD is compared with both. The small per-endpoint stress discrepancy is shown in Pa as well as its derivative-amplified relative effect. No global tolerance is changed.'}
 (OUT/'direction-roundoff.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'output':str(OUT/'direction-roundoff.json'),'results':out},indent=2,allow_nan=False))
if __name__=='__main__':main()
