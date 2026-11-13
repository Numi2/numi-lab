from pathlib import Path
import gzip,json,math
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts')
A=R/'crow-v15-low-wing-com-audit-20261113'
S=R/'crow-v15-low-wing-effort-screen-20261113'
com=json.loads((A/'analysis.json').read_text());out=[]
for arm in ('push','hold'):
 aero={}
 with gzip.open(S/(arm+'-wing-20261126-rebound-low')/'run.stderr.gz','rt') as f:
  for l in f:
   z=l.split()
   if z and z[0]=='CROW_AERO_FORCE':
    t,k,e=map(int,z[1:4]);v=list(map(float,z[5:]));assert len(v)==12 and (e,t,k) not in aero and all(map(math.isfinite,v));aero[e,t,k]=v[2]
 assert len(aero)==8*28*16
 for row in com['rows']:
  if row['arm']!=arm:continue
  e=row['lane']; impul=sum(aero[e,t,k]*.00125 for t in range(348,366) for k in range(16))
  out.append({'arm':arm,'lane':e,'model_aero_vertical_impulse_ns':impul,'other_residual_after_aero_ns':row['unattributed_momentum_residual_ns']-impul,'unattributed_residual_before_aero_ns':row['unattributed_momentum_residual_ns']})
(A/'aero-momentum.json').write_text(json.dumps({'scope':'Posthoc integration of recorded model aerodynamic force using assumed .00125s physics substep, and difference with independently reconstructed endpoint COM/contact/gravity balance. Force logger timing and applied-force equivalence require independent validation; residual is not a complete solver energy or momentum audit.','rows':out},indent=2)+'\n')
for x in out:
 if x['lane']>=2:print(x['arm'],x['lane'],'aero',round(x['model_aero_vertical_impulse_ns'],4),'remaining',round(x['other_residual_after_aero_ns'],5))
