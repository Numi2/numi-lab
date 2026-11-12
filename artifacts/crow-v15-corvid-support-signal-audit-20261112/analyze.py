from pathlib import Path
import gzip,json,collections
p=Path('artifacts/crow-v15-corvid-support-signal-audit-20261112');paths={'front':Path('artifacts/crow-v15-corvid-front-support-screen-20261112/candidate-20261127-rebound'),'combined':Path('artifacts/crow-v15-corvid-front-stance-combination-20261112/candidate-20261127-rebound')};out={};eps=1e-7
for label,path in paths.items():
 sub={};targets={};flags={}
 with gzip.open(path/'run.stderr.gz','rt') as f:
  for line in f:
   z=line.split()
   if not z:continue
   if z[0]=='CROW_TWO_TOE_SUBSTEP':
    t,k,e=map(int,z[1:4]);
    if 201<=t<=300:sub[e,t,k]=list(map(float,z[6:]))
   elif z[0]=='CROW_DRIVE_TARGET':
    t,e=map(int,z[1:3]);targets[e,t]=list(map(float,z[3:]))
   elif z[0]=='CROW_PHASE_SAMPLE':
    t,e=map(int,z[1:3]);flags[e,t]=int(z[3])
 assert len(sub)==12800 and len(targets)==800 and all((e,t) in flags for e in range(8) for t in range(200,301))
 rows=[];counts=collections.Counter();lanes=[collections.Counter() for _ in range(8)];worst=0
 for e in range(8):
  for t in range(201,301):
   for side in range(2):
    front=sum(sum(sub[e,t,k][5+8*side:11+8*side]) for k in range(16));rear=sum(sum(sub[e,t,k][11+8*side:13+8*side]) for k in range(16));total=sum(sub[e,t,k][2+side] for k in range(16));other=sum(sub[e,t,k][3-side] for k in range(16));sole=total-front-rear;worst=min(worst,sole);target=max(targets[e,t][22+8*side+2*j] for j in range(3));flag=bool(flags[e,t-1] & (1<<(20+side)));active=target>.001
    row={'lane':e,'step':t,'side':side,'captured_front_target_rad':target,'previous_own_flag':flag,'total_normal_impulse':total,'front_normal_impulse':front,'rear_normal_impulse':rear,'sole_normal_estimate':sole,'opposite_total_normal_impulse':other}
    if active:
     counts['active']+=1;lanes[e]['active']+=1
     categories={'prior_own_flag':flag,'no_sole_toe_loaded':sole<=eps and front+rear>eps,'no_sole_front_loaded':sole<=eps and front>eps,'flag_no_sole_toe_loaded':flag and sole<=eps and front+rear>eps,'flag_no_sole_toe_loaded_opposite_unloaded':flag and sole<=eps and front+rear>eps and other<=eps}
     for k,v in categories.items():
      if v:counts[k]+=1;lanes[e][k]+=1
     rows.append(row)
 out[label]={'counts':dict(counts),'lane_counts':[{'lane':e,**v} for e,v in enumerate(lanes)],'most_negative_rounding_residual':worst,'active_intervals':rows}
out['scope']='Prior sampled support flag, substep-zero front target, and interval-summed normals are temporally related observations, not proof of what the controller saw when applying the gate. Sole estimated by total foot/toe group minus front/rear toe normals; the values are impulse sums, not continuous pressure, friction force or a causal intervention. Used development seed only.';(p/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for name in paths:
 print(name,out[name]['counts'],'residual',out[name]['most_negative_rounding_residual']);print(out[name]['lane_counts'])
