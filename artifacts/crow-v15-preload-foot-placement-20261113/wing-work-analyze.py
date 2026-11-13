from pathlib import Path
import gzip,json,math
R=Path('/Users/n/Developer/numi-crow-rl-20261107/artifacts');A=R/'crow-v15-preload-foot-placement-20261113';A.mkdir(exist_ok=True);p=R/'crow-v15-preload-foot-placement-20261113/push-wing-20261126-rebound-preload/run.stderr.gz';d={e:[] for e in range(2,8)}
with gzip.open(p,'rt') as f:
 for line in f:
  z=line.split()
  if not z or z[0]!='CROW_WING_EFFORT':continue
  t,k,e=map(int,z[1:4]);v=list(map(float,z[4:]));assert len(v)==14
  if e not in d:continue
  for wing,j in enumerate((0,7)):
   effort,target,q0,q1,v0,vpre,v1=v[j:j+7];assert all(map(math.isfinite,v[j:j+7]));d[e].append((t,k,wing,effort*(q1-q0),effort*(v0+v1)*.5*.00125))
rows=[]
for e,x in d.items():
 assert len(x)==28*16*2
 exact=[y[3] for y in x];trap=[y[4] for y in x]
 rows.append({'lane':e,'working_effort_times_joint_displacement_net':sum(exact),'working_effort_times_joint_displacement_positive':sum(max(0,y) for y in exact),'working_effort_times_joint_displacement_negative_magnitude':sum(max(0,-y) for y in exact),'trapezoid_velocity_net':sum(trap),'trapezoid_velocity_positive':sum(max(0,y) for y in trap),'trapezoid_velocity_negative_magnitude':sum(max(0,-y) for y in trap),'n_samples':len(x)})
(A/'wing-work-analysis.json').write_text(json.dumps({'source':'low effort push-first trial, steps 348–375, both flap DOFs, seed 20261126, selected post hoc','scope':'Working-effort times accepted angular displacement and an alternate trapezoid velocity approximation. Observer buffer timing, dissipative and coupled effects are not independently isolated; do not identify as full muscle work or use as biological validation. Difference between integrators is itself material.','rows':rows},indent=2)+'\n')
for x in rows:print(x['lane'],round(x['working_effort_times_joint_displacement_positive'],3),round(x['trapezoid_velocity_positive'],3),'net',round(x['working_effort_times_joint_displacement_net'],3))
