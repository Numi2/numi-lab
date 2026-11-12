from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-uniform-hip-second-seed-20261112');paths=[Path('artifacts/crow-v15-uniform-hip-pitch-walk-launch-20261112/candidate-20261127-rebound-hip/run.stderr.gz'),A/'candidate-20261126-rebound-hip/run.stderr.gz'];a=[]
for p in paths:
 d={}
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1])
    if x['step'] in (1,200):d[x['lane'],x['step']]=x
    if len(d)==16:break
 assert len(d)==16;a.append(d)
rows=[]
for t in (1,200):
 for e in range(8):
  x,y=a[0][e,t],a[1][e,t];dot=abs(sum(q*r for q,r in zip(x['q'][3:7],y['q'][3:7])));rows.append({'step':t,'lane':e,'root_position_distance_m':math.dist(x['q'][:3],y['q'][:3]),'quaternion_angular_distance_rad':2*math.acos(max(0,min(1,dot))),'max_joint_coordinate_difference_rad':max(abs(q-r) for q,r in zip(x['q'][7:],y['q'][7:])),'max_generalized_velocity_difference':max(abs(q-r) for q,r in zip(x['v'],y['v']))})
out={'scope':'Endpoint state difference between two already-used seeds, matched lane indices, not an IID biological sampling model; generalized velocity components have mixed linear and angular units.','rows':rows};(A/'initial-state-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
for t in (1,200):
 q=[x for x in rows if x['step']==t];print(t,'root distance range',min(x['root_position_distance_m'] for x in q),max(x['root_position_distance_m'] for x in q),'joint difference range',min(x['max_joint_coordinate_difference_rad'] for x in q),max(x['max_joint_coordinate_difference_rad'] for x in q))
