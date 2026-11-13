from pathlib import Path
from collections import defaultdict
import gzip,json,math,os,subprocess
R=Path.cwd(); A=R/'artifacts/crow-v15-contact-impulse-diagnosis-20261113'; L=R/'build/crow-v15-mirrored-leg-isolation-20261113'
source=R/'artifacts/crow-v15-mirrored-launch-phase-20261113'; states={}; ground=defaultdict(lambda:[0.,0.,0]); aero=defaultdict(dict)
for arm in ('push','hold'):
 p=source/(arm+'-wing-20261126-rebound-phase')/'run.stderr.gz'
 with gzip.open(p,'rt') as f:
  for line in f:
   if line.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(line.split(' ',1)[1]);e=x['lane'];t=x['step']
    if e in (2,3) and t in (374,375,376):states[arm,e,t]=x
    continue
   z=line.split()
   if not z or z[0] not in ('CROW_GROUND_CONTACT','CROW_AERO_FORCE'):continue
   t,k,e=map(int,z[1:4])
   if e not in (2,3) or t!=375:continue
   if z[0]=='CROW_AERO_FORCE':
    assert k not in aero[arm,e];aero[arm,e][k]=[float(z[5]),float(z[7])];continue
   a,b=map(int,z[4:6]);n=list(map(float,z[12:15]));tan=list(map(float,z[15:18]));j=list(map(float,z[20:23]));bit=[n[1]*tan[2]-n[2]*tan[1],n[2]*tan[0]-n[0]*tan[2],n[0]*tan[1]-n[1]*tan[0]]
   assert (a==35) != (b==35);sign=-1 if b==35 else 1
   ground[arm,e][0]+=sign*(j[0]*n[0]+j[1]*tan[0]+j[2]*bit[0]);ground[arm,e][1]+=sign*(j[0]*n[2]+j[1]*tan[2]+j[2]*bit[2]);ground[arm,e][2]+=1
assert len(states)==12
keys=list(states);inp=[]
for i,key in enumerate(keys):
 x=states[key];inp.append(str(i)+' 0 '+' '.join(map(str,x['q']+x['v'])))
p=subprocess.run([str(L/'kinematics')],input='\n'.join(inp)+'\n',text=True,capture_output=True,check=True,env={**os.environ,'DYLD_LIBRARY_PATH':str(L)});com={}
for line in p.stdout.splitlines():
 z=line.split();assert z[0]=='C';v=list(map(float,z[3:]));assert len(v)==7 and all(map(math.isfinite,v));com[keys[int(z[1])]]=v
assert len(com)==len(keys)
rows=[]
for arm in ('push','hold'):
 for e in (2,3):
  assert set(aero[arm,e])==set(range(16));before=com[arm,e,374];after=com[arm,e,375];mass=after[0];assert mass>0 and abs(mass-before[0])<1e-8
  da=[sum(v[j] for v in aero[arm,e].values())*.00125 for j in (0,1)];obs=[mass*(after[4]-before[4]),mass*(after[6]-before[6])];pred=[ground[arm,e][0]+da[0],ground[arm,e][1]+da[1]-mass*9.81*.02]
  rows.append({'arm':arm,'lane':e,'mass_kg':mass,'com_velocity_before_xz':[before[4],before[6]],'com_velocity_after_xz':[after[4],after[6]],'observed_com_momentum_change_xz_ns':obs,'ground_impulse_xz_ns':ground[arm,e][:2],'ground_contact_records':ground[arm,e][2],'aerodynamic_impulse_xz_ns':da,'gravity_impulse_z_ns':-mass*9.81*.02,'predicted_momentum_change_xz_ns':pred,'residual_xz_ns':[obs[j]-pred[j] for j in range(2)]})
out={'source':'Retained launch phase push and hold first episodes; tick 374 to 375','rows':rows,'scope':'Impulse reconstructed from logged contact basis and modeled external aerodynamic force; residual is reported, not assumed zero. This does not identify a solver defect or establish biological realism.'}
(A/'force-balance.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
