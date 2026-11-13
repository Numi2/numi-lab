from pathlib import Path
import gzip,json,math,os,subprocess,hashlib
R=Path('/Users/n/Developer/numi-crow-rl-20261107'); S=R/'artifacts/crow-v15-front-toe-preload-reach-20261113';A=R/'artifacts/crow-v15-reach-geometry-audit-20261113';A.mkdir(exist_ok=True);kin=R/'build/crow-v15-hip-roll-walk-launch-matched-20261111/kinematics'
def box(v,half):
 x,y,z,w=v[3:7];n=(2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y));return v[2]-sum(abs(a)*b for a,b in zip(n,half))
assert abs(box([0,0,.01,0,0,0,1],(.009,.0025,.0092))-.0008)<1e-12
names={f'crow_{side}_{name}' for side in ('left','right') for name in ('foot','front_toe_2_proximal','front_toe_2_distal','rear_toe_proximal','rear_toe_distal')};times=(340,345,347,350,355);out={}
for arm in ('push','hold'):
 p=S/(arm+'-wing-20261126-rebound-reach');states={}
 with gzip.open(p/'run.stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);key=(x['lane'],x['step']);
    if key[0]>=2 and key[1] in times: assert key not in states;states[key]=x
 assert len(states)==6*len(times)
 ordered=sorted(states);req=''.join(f'{e} {t} '+' '.join(map(str,states[e,t]['q']+states[e,t]['v']))+'\n' for e,t in ordered)
 env={**os.environ,'DYLD_LIBRARY_PATH':str(R/'build/crow-v15-hip-roll-journey-integration-20261111')};run=subprocess.run([str(kin)],input=req,text=True,capture_output=True,env=env,check=True);b={}
 for l in run.stdout.splitlines():
  z=l.split()
  if z and z[0]=='body' and z[3] in names:b[int(z[1]),int(z[2]),z[3]]=list(map(float,z[4:]))
 assert len(b)==len(ordered)*len(names)
 with gzip.open(p/'replay.json.gz','rt') as f:rp=json.load(f)['payload']
 frames={fr['step']+1:fr for fr in rp['frames']};err=0.;rot=0.
 for t in times:
  fr=frames[t]
  for name in names:
   i=rp['body_names'].index(name)*13;raw=fr['body_states'][i:i+7];v=b[5,t,name];err=max(err,math.dist(raw[:3],v[:3]));rot=max(rot,min(math.dist(raw[3:7],v[3:7]),math.dist(raw[3:7],[-u for u in v[3:7]])))
 if not (err<1e-5 and rot<1e-5):raise SystemExit(f'FK and replay mismatch: arm {arm}, position {err}, quaternion {rot}')
 rows=[]
 for e,t in ordered:
  for side in ('left','right'):
   get=lambda n:b[e,t,f'crow_{side}_{n}']
   front=min(box(get('front_toe_2_'+s),(.009,.0025,.0092)) for s in ('proximal','distal'))
   rear=min(box(get('rear_toe_'+s),(.009,.0025,.0092)) for s in ('proximal','distal'))
   foot=box(get('foot'),(.010,.01725,.0092));rows.append({'lane':e,'step':t,'side':side,'front_clearance_m':front,'rear_clearance_m':rear,'sole_clearance_m':foot,'ankle_q':states[e,t]['q'][16 if side=='left' else 19]})
 out[arm]={'replay_position_max_error_m':err,'replay_quaternion_max_error':rot,'rows':rows}
print('validated max errors',[(k,v['replay_position_max_error_m'],v['replay_quaternion_max_error']) for k,v in out.items()])
(A/'analysis.json').write_text(json.dumps({'scope':'Posthoc reconstructed collision-box clearance at selected control endpoints, ground z=0; contact signal and q samples may be staggered. Forward kinematics checked against selected native lane-5 replay frames. Not exact solver separation or biological geometry.','data':out},indent=2)+'\n')
for arm,d in out.items():
 for e in range(2,8):
  rr=[r for r in d['rows'] if r['lane']==e and r['step']==347 and r['side']=='left'][0]
  print(arm,e,'step347 ankle',round(rr['ankle_q'],4),'front mm',round(1000*rr['front_clearance_m'],3),'rear mm',round(1000*rr['rear_clearance_m'],3),'sole mm',round(1000*rr['sole_clearance_m'],3))
