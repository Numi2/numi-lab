from pathlib import Path
import collections,gzip,json,math
a=Path('artifacts/crow-v15-overlap-full-sequence-screen-20261112/autonomous-20261126')
d=json.loads((a/'run.stdout').read_text());first=d['first_episode_navigation'];ends=first['steps_by_environment']
sums=collections.defaultdict(lambda:[0.]*8);coverage=collections.Counter();velocities={};errors=[];overflow=[]
with gzip.open(a/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_TWO_TOE_SUBSTEP '):
   z=line.split();t,k,e=map(int,z[1:4])
   if t<=ends[e]+1:
    coverage[e,t,k]+=1
    if int(z[4])>int(z[5]):overflow.append([e,t,k,int(z[4]),int(z[5])])
  elif line.startswith('CROW_TOE_V_SUBSTEP '):
   z=line.split();t,k,e=map(int,z[1:4])
   if t<=ends[e]+1:velocities[e,t,k]=max(abs(float(v)) for v in z[4:])
  elif line.startswith('CROW_SCENE_CONTACT '):
   z=line.split();t,k,e,ba,bb=map(int,z[1:6])
   if t>ends[e]+1 or 40 not in (ba,bb):continue
   body=bb if ba==40 else ba;sign=1. if ba==40 else -1.
   normal,tan1,tan2=map(float,z[6:9]);nx,ny,nz=map(float,z[13:16]);tx,ty,tz=map(float,z[16:19])
   vz=nx*ty-ny*tx
   upward=sign*(nz*normal+tz*tan1+vz*tan2)
   if body==11 or 19<=body<=26:side=0
   elif body==15 or 27<=body<=34:side=1
   else:side=-1
   o=sums[e,t]
   if side<0:o[6]+=normal;continue
   o[side]+=upward
   if body>=19:
    digit=((body-19)%8)//2
    o[2+2*side+(1 if digit==3 else 0)]+=sign*nx*normal
   o[7]+=normal
  elif 'CROW_SUBSTEP_ERROR' in line:errors.append(line.strip())
def longest(seq):
 best=0;run=0
 for v in seq:run=run+1 if v else 0;best=max(best,run)
 return best
rows=[];nom=.45006*9.81*.02
for e,end in enumerate(ends):
 vals=[sums[e,t] for t in range(1,end+1)]
 support=[x[0]+x[1]>=.5*nom and x[6]<=1e-7 for x in vals]
 grip=[ok and x[2]>1e-6 and x[3]<-1e-6 and x[4]>1e-6 and x[5]<-1e-6 for ok,x in zip(support,vals)]
 rows.append({'lane':e,'first_episode_steps':end,'physical_failure':first['physical_failure_by_environment'][e],'waypoints':first['maximum_waypoints_by_environment'][e],
 'first_perch_foot_impulse_step':next((i+1 for i,x in enumerate(vals) if x[7]>1e-7),None),'max_consecutive_half_weight_nonbody_intervals':longest(support),'max_consecutive_half_weight_bilateral_opposed_intervals':longest(grip),'last_25_half_weight_bilateral_opposed':len(grip)>=25 and all(grip[-25:]),
 'nonfoot_perch_impulse_intervals':sum(x[6]>1e-7 for x in vals),'missing_or_duplicate_substeps':sum(coverage[e,t,k]!=1 for t in range(1,end+1) for k in range(16)),'toe_speed_limit_substeps':sum(v>=11.99 for (lane,t,k),v in velocities.items() if lane==e and t<=end)})
out={'rows':rows,'substep_errors':errors[-10:],'contact_capacity_overflows':overflow,'limitations':'Contact impulses sampled at physics-substep endpoints; opposite normal components do not alone establish biological grip. Zero contact messages require complete substep coverage and no overflow. First-episode step indexing needs alignment checks; q/vel stability and geometry clearance require independent evaluation.'}
(a/'perch-analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
