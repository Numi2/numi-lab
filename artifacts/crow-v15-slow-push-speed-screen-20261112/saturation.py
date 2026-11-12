from pathlib import Path
import collections,gzip,json,math
A=Path('artifacts/crow-v15-slow-push-speed-screen-20261112');P=A/'candidate-20261127-rebound-flexed';mp=Path('artifacts/crow-v15-drive-work-observer-20261112/dof-map.txt');mapping={}
for line in mp.read_text().splitlines():
 z=line.split();mapping[int(z[0])]={'name':z[1],'limit_rad_s':float(z[-1])}
meta=json.loads((P/'run.stdout').read_text())['first_episode_navigation'];ends=meta['steps_by_environment'];groups=collections.defaultdict(lambda:collections.defaultdict(lambda:{'maximum_abs_rad_s':0.,'limit_hits':0}));total=collections.Counter();nonfoot=collections.Counter();over=[];errors=[]
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_DRIVE_WORK '):
   z=line.split();t,k,e,j,flags,qi=map(int,z[1:7]);v0,v1=map(float,z[8:10]);
   if t>ends[e]:continue
   lim=mapping[j]['limit_rad_s'];assert math.isfinite(v0) and math.isfinite(v1)
   if lim>0 and flags & 8:
    item=groups[e][j];item['maximum_abs_rad_s']=max(item['maximum_abs_rad_s'],abs(v0),abs(v1));item['limit_hits']+=int(abs(v1)>=lim-1e-4)
   total[e]+=1
  elif line.startswith('CROW_TWO_TOE_SUBSTEP '):
   z=line.split();t,k,e=map(int,z[1:4]);
   if t>ends[e]:continue
   if 340<=t<=380:
    nonfoot[e]+=int(float(z[10])>1e-7)
    if int(z[4])>int(z[5]):over.append([e,t,k])
  elif line.startswith('CROW_SUBSTEP_ERROR'):errors.append(line.strip())
rows=[]
for e in range(8):
 rows.append({'lane':e,'target_ramp_s':[.04,.08,.16,.32][e//2],'first_episode_steps':ends[e],'drive_records':total[e],'nonfoot_ground_substeps':nonfoot[e],'joints':[{'index':j,**mapping[j],**value} for j,value in sorted(groups[e].items()) if value['limit_hits'] or j in (13,14,15,16,17,18,38,39)],'total_joint_limit_hits':sum(x['limit_hits'] for x in groups[e].values())})
out={'rows':rows,'contact_overflows':over,'substep_errors':errors,'scope':'All velocity-limited joints, substeps 340–380 (6.8–7.6 s nominal); start/end velocities sampled, hit defined by endpoint within 0.0001 rad/s of authored speed limit. Other windows, impulsive clipping, biological realism, leg work and full mission remain unqualified.'};(A/'joint-speed-saturation.json').write_text(json.dumps(out,indent=2)+'\n')
for x in rows:print(x['lane'],'ramp',x['target_ramp_s'],'records',x['drive_records'],'caps',x['total_joint_limit_hits'],'nonfoot',x['nonfoot_ground_substeps'],'affected',[(j['name'],j['limit_hits'],round(j['maximum_abs_rad_s'],3)) for j in x['joints'] if j['limit_hits']])
