from pathlib import Path
import gzip,json,collections,math
p=Path('artifacts/crow-v15-rear-toe-touchdown-screen-20261112');paths={'candidate':p/'candidate-20261127-rebound','prior':Path('artifacts/crow-v15-walk-paired-phase-trial-20261112/candidate-20261127-rebound')};out={}
for n,path in paths.items():
 st={};sub={};caps=collections.Counter();errs=[]
 with gzip.open(path/'run.stderr.gz','rt') as f:
  for s in f:
   if 'CROW_SUBSTEP_ERROR' in s:errs.append(s.strip())
   if s.startswith('CROW_SUPPORT_SAMPLE '):
    z=json.loads(s.split(' ',1)[1]);st[z['lane'],z['step']]=z
   if s.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=s.split();t,k,e=map(int,z[1:4]);sub[e,t,k]=list(map(float,z[6:]));
    if int(z[4])>int(z[5]):errs.append('overflow')
   if s.startswith('CROW_TOE_V_SUBSTEP '):
    z=s.split();t,k,e=map(int,z[1:4]);
    if 101<=t<=300 and max(map(abs,map(float,z[4:])))>=11.99:caps[e]+=1
 assert len(st)==2408 and len(sub)==38528
 rows=[]
 for e in range(8):
  def count(lo,hi):return sum(sub[e,t,k][2]<=1e-7 and sub[e,t,k][3]<=1e-7 for t in range(lo,hi+1) for k in range(16))
  def bad(lo,hi):return sum(any(sub[e,t,k][4]>1e-7 for k in range(16)) for t in range(lo,hi+1))
  z=[st[e,t]['q'][2] for t in range(101,301)];rows.append({'lane':e,'preparation_unloaded_substeps':count(101,200),'walking_unloaded_substeps':count(201,300),'preparation_nonfoot_intervals':bad(101,200),'walking_nonfoot_intervals':bad(201,300),'toe_speed_cap_substeps':caps[e],'root_height_min_m':min(z),'root_height_max_m':max(z)})
 out[n]={'errors':errs,'rows':rows,'first_episode':json.loads((path/'run.stdout').read_text())['first_episode_navigation']}
out['scope']='Native recorded normal impulse; 2–4 s preparation and 4–6 s walk. Zero foot normal means airborne or loss of accepted normal impulse; not automatically a fall. Ground-contact classification, q and toe cap sampled; unobserved link forces and biological validation remain outside scope.'
(p/'whole-preparation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
