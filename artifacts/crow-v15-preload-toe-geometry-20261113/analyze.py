from pathlib import Path
import gzip,json,os,subprocess,hashlib
R=Path('/Users/n/Developer/numi-crow-rl-20261107');S=R/'artifacts/crow-v15-front-toe-preload-reach-20261113/hold-wing-20261126-rebound-reach';A=R/'artifacts/crow-v15-preload-toe-geometry-20261113';A.mkdir(exist_ok=True);kin=R/'build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics';states={}
with gzip.open(S/'run.stderr.gz','rt') as f:
 for l in f:
  if l.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(l.split(' ',1)[1]);
   if x['step']==347 and x['lane']>=2:states[x['lane']]=x
assert len(states)==6
variants=[-.20,-.10,-.05,0.,.05,.10,.20];inputs=[];keys={};i=0
for e,x in sorted(states.items()):
 for delta in variants:
  q=x['q'].copy()
  for qi in (23,25,27):q[qi]+=delta
  keys[i]=(e,delta);inputs.append(f'{i} 347 '+' '.join(map(str,q+x['v'])));i+=1
env={**os.environ,'DYLD_LIBRARY_PATH':str(R/'build/crow-v15-hip-roll-journey-integration-20261111')};p=subprocess.run([str(kin)],input='\n'.join(inputs)+'\n',text=True,capture_output=True,env=env,check=True)
names={f'crow_left_front_toe_{j}_{part}' for j in (1,2,3) for part in ('proximal','distal')};b={}
for l in p.stdout.splitlines():
 z=l.split()
 if z and z[0]=='body' and z[3] in names:b[int(z[1]),z[3]]=list(map(float,z[4:]))
assert len(b)==len(keys)*len(names)
def box(v):
 x,y,z,w=v[3:7];n=(2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y));return v[2]-sum(abs(a)*c for a,c in zip(n,(.009,.0025,.0092)))
rows=[]
for i,(e,delta) in keys.items():
 vals={name:box(b[i,name]) for name in names}
 rows.append({'lane':e,'delta_proximal_rad':delta,'left_front_group_min_clearance_m':min(vals.values()),'left_middle_group_min_clearance_m':min(v for k,v in vals.items() if 'toe_2_' in k),'clearance_by_body_m':vals})
(A/'analysis.json').write_text(json.dumps({'scope':'Posthoc frozen-pose FK finite difference at step 347 on six used development lanes; varies only three left proximal toe coordinates, not a dynamically controlled trajectory. Box dimensions and ground plane are model estimates. Baseline FK is consistent with previously replay-validated same states. Joint limits, penetrations, actuator dynamics and physical contact require separate validation.','rows':rows},indent=2)+'\n')
for e in range(2,8):print(e,[(r['delta_proximal_rad'],round(1000*r['left_middle_group_min_clearance_m'],2)) for r in rows if r['lane']==e])
