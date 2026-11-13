from pathlib import Path
from collections import defaultdict
import gzip,json,math,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-mirrored-hip-timing-20261113';B=R/'artifacts/crow-v15-mirrored-leg-isolation-20261113';L=R/'build/crow-v15-mirrored-leg-isolation-20261113';screen=json.loads((A/'screen.json').read_text());states={};targets={};imp=defaultdict(float);count=defaultdict(int);support=defaultdict(float);reference={};ref_target={}
for arm in ('push','hold'):
 for ref,base,tag in ((False,A,'timing'),(True,B,'mirrored')):
  p=base/(arm+'-wing-20261126-rebound-'+tag)/'run.stderr.gz'
  with gzip.open(p,'rt') as f:
   for line in f:
    if line.startswith('CROW_SUPPORT_SAMPLE '):
     x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step']
     if e<=7 and t<=400:(reference if ref else states)[arm,e,t]=x
     continue
    z=line.split()
    if not z:continue
    if z[0]=='CROW_DRIVE_TARGET':
     t,e=map(int,z[1:3]);
     if 335<=t<=400:(ref_target if ref else targets)[arm,e,t]=list(map(float,z[3:]))
    if ref or z[0] not in ('CROW_GROUND_CONTACT','CROW_TWO_TOE_SUBSTEP'):continue
    t,k,e=map(int,z[1:4]);
    if not (348<=t<=400 and e>=2):continue
    if z[0]=='CROW_TWO_TOE_SUBSTEP': support[arm,e,t]+=sum(map(float,z[8:10]));count[arm,e,t]+=1;continue
    a,b=map(int,z[4:6]);n=list(map(float,z[12:15]));u=list(map(float,z[15:18]));j=list(map(float,z[20:23]));v=[n[1]*u[2]-n[2]*u[1],n[2]*u[0]-n[0]*u[2],n[0]*u[1]-n[1]*u[0]];assert (a==35)!=(b==35);s=-1 if b==35 else 1;imp[arm,e,t]+=s*(j[0]*n[0]+j[1]*u[0]+j[2]*v[0])
parity=[]
for arm in ('push','hold'):
 for e in range(4):
  bad=[t for t in range(1,401) if (arm,e,t) not in states or (arm,e,t) not in reference or any(states[arm,e,t][k]!=reference[arm,e,t][k] for k in ('q','v','policy_action','contacts'))];tb=[t for t in range(335,401) if targets.get((arm,e,t))!=ref_target.get((arm,e,t))];parity.append({'arm':arm,'lane':e,'state_mismatches':bad,'target_mismatches':tb})
inputs=[];keys=[]
for row in screen['rows']:
 if row['arm']!='push' or row['release_tick'] is None:continue
 e=row['lane'];t=row['release_tick']
 for arm in ('push','hold'):
  x=states[arm,e,t];keys.append((arm,e,t));inputs.append(str(len(keys)-1)+' 0 '+' '.join(map(str,x['q']+x['v'])))
p=subprocess.run([str(L/'kinematics')],input='\n'.join(inputs)+'\n',text=True,capture_output=True,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(L)});com={}
for line in p.stdout.splitlines():
 z=line.split();assert z[0]=='C';v=list(map(float,z[3:]));assert len(v)==7 and all(map(math.isfinite,v));com[keys[int(z[1])]]=v
assert len(com)==len(keys)
rows=[]
for row in screen['rows']:
 if row['arm']!='push':continue
 e=row['lane'];t=row['release_tick'];first=next((r['first_target_difference'] for r in screen['target_divergence'] if r['lane']==e),None)
 if t is None:rows.append({'lane':e,'release':None,'passed':False});continue
 assert all(count[a,e,u]==16 for a in ('push','hold') for u in range(first,t+1));cv=com['push',e,t];hv=com['hold',e,t];ground_delta=sum(imp['push',e,u]-imp['hold',e,u] for u in range(first,t+1));supported=support['hold',e,t]>1e-7
 checks={'previous_screen':row['screen_pass'],'hold_supported_at_push_release':supported,'horizontal_com_velocity':-.02<=cv[4]<=1.,'minimum_foot_impulse_delta':row['push_minus_hold_foot_normal_impulse_371_to_release_ns']>.1,'reference_parity':all(not p['state_mismatches'] and not p['target_mismatches'] for p in parity)}
 rows.append({'lane':e,'release':t,'push_com_vx':cv[4],'hold_com_vx':hv[4],'ground_x_impulse_difference_ns':ground_delta,'checks':checks,'passed':all(checks.values())})
out={'reference_parity':parity,'rows':rows,'pairs':{str(e)+'-'+str(e+1):all(next(r for r in rows if r['lane']==j)['passed'] for j in (e,e+1)) for e in (2,4,6)},'scope':'Wing-held used-seed screen only; no qualified takeoff or flight.'};(A/'horizontal-gate.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
