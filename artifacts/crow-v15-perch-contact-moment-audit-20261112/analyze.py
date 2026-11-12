from pathlib import Path
import gzip,json,math
from collections import defaultdict
R=Path('artifacts/crow-v15-early-wing-sequence-screen-20261112')
P=R/'autonomous-20261126';A=Path('artifacts/crow-v15-perch-contact-moment-audit-20261112')
d=json.loads((P/'run.stdout').read_text())['first_episode_navigation'];ends=d['steps_by_environment'];state={};contacts=defaultdict(list)
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);e,t=x['lane'],x['step']
   if 3000<=t<=ends[e]:
    a,b,c,w=x['q'][3:7];state[e,t]=(x['q'][:3],math.asin(max(-1,min(1,2*(w*b-c*a)))),x['v'][4])
  elif line.startswith('CROW_SCENE_CONTACT '):
   z=line.split();t,k,e,ba,bb=map(int,z[1:6])
   if 3000<=t<=ends[e] and 40 in (ba,bb):
    n,t1,t2=map(float,z[6:9]);point=list(map(float,z[9:12]));normal=list(map(float,z[13:16]));tan=list(map(float,z[16:19]));cross=[normal[1]*tan[2]-normal[2]*tan[1],normal[2]*tan[0]-normal[0]*tan[2],normal[0]*tan[1]-normal[1]*tan[0]]
    sign=1 if ba==40 else -1
    impulse=[sign*(n*normal[i]+t1*tan[i]+t2*cross[i]) for i in range(3)]
    contacts[e,t].append((point,impulse,bb if ba==40 else ba))
rows=[]
for e,end in enumerate(ends):
 events=[]
 for threshold in (.1,.2,.4,.8):
  t=next((t for t in range(3000,end+1) if (e,t) in state and abs(state[e,t][1])>=threshold),None)
  if t is None:continue
  samples=[]
  for u in range(max(3000,t-12),min(end,t+12)+1):
   if (e,u) not in state:continue
   root,pitch,rate=state[e,u];moment=0.;net=[0.,0.,0.]
   for point,impulse,body in contacts[e,u]:
    dx=point[0]-root[0];dz=point[2]-root[2]
    moment+=dz*impulse[0]-dx*impulse[2]
    for i in range(3):net[i]+=impulse[i]
   samples.append({'step':u,'pitch_rad':pitch,'pitch_rate_world_y':rate,'perch_impulse_Ns':net,'perch_pitch_impulse_about_endpoint_root_Nms':moment,'contact_count':len(contacts[e,u])})
  events.append({'threshold_rad':threshold,'crossing_step':t,'samples':samples})
 rows.append({'lane':e,'end':end,'events':events})
out={'rows':rows,'limits':'Perch contacts only, relative to root endpoint position, not center of mass; contact points and impulses occur at individual substeps but root reference is one endpoint. The sign follows the existing scene-contact convention. This is not total angular impulse, joint torque, muscle effort, ground force, or proof of causal stabilization; other external actuation and angular momentum are omitted. Previously used development run.'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:
 print('lane',r['lane'],[(x['threshold_rad'],x['crossing_step'],round(sum(s['perch_pitch_impulse_about_endpoint_root_Nms'] for s in x['samples']),6),round(sum(s['perch_impulse_Ns'][2] for s in x['samples']),5)) for x in r['events']])
