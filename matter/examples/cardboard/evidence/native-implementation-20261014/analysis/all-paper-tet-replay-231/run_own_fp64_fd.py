import json, hashlib, subprocess
from pathlib import Path
import numpy as np
BASE=Path('/Users/home/cardboard-implementation-20261014')
CAP=BASE/'solver/graph-schwarz-first-control-final-newton-capture-216/operator-capture'
RUN=CAP.parent; OUT=BASE/'analysis/all-paper-tet-replay-231/output'
REPLAY=BASE/'analysis/smooth-target-pair-audit-225/build/cardboard_solver_pair_replay_realized'
LINER=RUN/'inputs/source/matter/materials/hajali2009_liner_hill_ideal.nmatter'; MEDIUM=RUN/'inputs/source/matter/materials/hajali2009_medium_hill_ideal.nmatter'
def f32(a): return np.asarray(a,dtype=np.float32).reshape(-1)
def fmt(a): return ' '.join(format(float(x),'.17g') for x in a)
recs=[json.loads(x) for x in CAP.read_text().splitlines() if x.strip()]
h=recs[0]; materials={int(m['index']):m for m in h['materials']}; rows=[]; ids=[]; zero=[]
for t in recs[1:-1]:
 if t.get('record')!='tetrahedron' or int(t['material_index']) not in (0,1): continue
 env=int(t['environment']); kind='liner' if int(t['material_index'])==0 else 'medium'; c=t['center']
 block=f32(h['parameters_by_environment'][env]); off=int(materials[int(t['material_index'])]['parameter_offset']); pars=block[off:off+20]
 fhi=f32(c['F_local']); flo=f32(c.get('F_local_low',[0]*9)); d=f32(c['F_direction_local_per_s'])
 acc=f32(c['accepted_state']); chi=f32(c['projected_state']); clo=f32(c.get('projected_state_low',[0]*13))
 key=f"e{env}_t{int(t['tet_id'])}_s0"; ids.append(key)
 if np.max(np.abs(d))==0: zero.append(key)
 rows.append(' '.join([key,kind,fmt(fhi),fmt(flo),fmt(d),fmt(acc),fmt(chi),fmt(clo),fmt(pars)]))
inp=OUT/'fp64-own-fd-input.txt'; out=OUT/'fp64-own-fd-output.jsonl'; err=OUT/'fp64-own-fd-stderr.txt'
inp.write_text('\n'.join(rows)+'\n')
with inp.open() as fi, out.open('w') as fo, err.open('w') as fe:
 cp=subprocess.run([str(REPLAY),str(LINER),str(MEDIUM),'--derivative-sweep'],stdin=fi,stdout=fo,stderr=fe,check=False)
if cp.returncode not in (0,2): raise SystemExit(f'replay exit {cp.returncode}: {err.read_text()[:1500]}')
observed=[]; n=0
for line in out.open():
 if not line.strip():continue
 r=json.loads(line); n+=1; sw=r.get('FP64_derivative_sweep',{}); rowsw=sw.get('rows',[])
 passes=[x for x in rowsw if x.get('valid') and x.get('smooth_branch_eligible') and x.get('relative_error') is not None and x['relative_error']<=1e-6]
 pairpass=any(rowsw[i].get('relative_error',1)>1e-6 and rowsw[i+1].get('relative_error',1)>1e-6 for i in range(len(rowsw)-1)) if False else False
 adjacent=sw.get('verdict')=='two_adjacent_smooth_refinements_pass'
 eligible=[x for x in rowsw if x.get('valid') and x.get('smooth_branch_eligible') and x.get('relative_error') is not None]
 best=min((x['relative_error'] for x in eligible),default=None)
 observed.append({'key':r['key'],'valid':r.get('valid'),'center_q':r.get('elastic_trial_q'),'branch':('transition_unresolved' if r.get('elastic_trial_q') is not None and abs(r['elastic_trial_q']-1)<=1e-6 else ('plastic' if r.get('elastic_trial_q',0)>1 else 'elastic')),
  'direction_scale_per_s':sw.get('direction_scale_per_s'),'sweep_verdict':sw.get('verdict'),'smooth_rows':len(eligible),'best_smooth_relative_error':best,'best_smooth_epsilon_s':next((x['epsilon_seconds'] for x in eligible if x['relative_error']==best),None),'adjacent_1e-6_pass':adjacent,
  'per_epsilon':rowsw})
if n!=len(rows): raise SystemExit(f'row count {n} != {len(rows)}')
summary={'schema':'numi.cardboard.all_paper_tet_fp64_fd_sweep.v1','capture':str(CAP),'capture_sha256':hashlib.sha256(CAP.read_bytes()).hexdigest(),'replay_binary_sha256':hashlib.sha256(REPLAY.read_bytes()).hexdigest(),'replay_cpp_sha256':hashlib.sha256((BASE/'analysis/smooth-target-pair-audit-228/cardboard_solver_pair_replay_realized.cpp').read_bytes()).hexdigest(),
 'rows_expected':len(rows),'rows_observed':n,'zero_direction_keys':zero,'gate':'1e-6 relative centered finite difference with independently projected FP64 endpoint stresses, same accepted state and branch; two adjacent eligible refinements required for a pass','counts':{'adjacent_pass':sum(x['adjacent_1e-6_pass'] for x in observed),'no_adjacent_pass':sum(not x['adjacent_1e-6_pass'] for x in observed),'zero_direction':len(zero),'smooth_center':sum(x['branch'] in ('elastic','plastic') and x['direction_scale_per_s'] not in (None,0) for x in observed)},'rows':observed}
(OUT/'fp64-own-fd-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))
