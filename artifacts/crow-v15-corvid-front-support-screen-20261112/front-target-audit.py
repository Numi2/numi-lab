from pathlib import Path
import gzip,json,statistics
p=Path('artifacts/crow-v15-corvid-front-support-screen-20261112');old=Path('artifacts/crow-v15-own-support-hallux-release-screen-20261112/candidate-20261127-rebound');new=p/'candidate-20261127-rebound'
def load(path):
 q={};targets={}
 with gzip.open(path/'run.stderr.gz','rt') as f:
  for l in f:
   z=l.split()
   if not z:continue
   if z[0]=='CROW_Q_SUBSTEP':
    t,k,e=map(int,z[1:4]);
    if k==0:q[e,t]=list(map(float,z[4:]))
   elif z[0]=='CROW_DRIVE_TARGET':
    t,e=map(int,z[1:3]);targets[e,t]=list(map(float,z[3:]))
 assert len(q)==len(targets)==800
 return q,targets
a,at=load(new);b,bt=load(old);rows=[]
for e in range(8):
 samples=[]
 for t in range(201,301):
  for s in range(2):
   for j in range(3):
    dof=22+8*s+2*j;target=at[e,t][dof];
    if target>.001:samples.append((target,a[e,t][dof+1]-b[e,t][dof+1]))
 rows.append({'lane':e,'target_positive_samples':len(samples),'max_target_rad':max((x[0] for x in samples),default=None),'median_actual_joint_difference_vs_prior_rad':statistics.median(x[1] for x in samples) if samples else None})
out={'rows':rows,'scope':'Substep-zero actual joint angle and target comparisons at matched lane and time with prior body/controller. Trajectories may diverge; target is not applied torque and angles are not biological kinematics.'};(p/'front-target-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
