from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-upward-trigger-longflight-20261114'); P=A/'turn-20261126-rebound-longflight'
S={(x['lane'],x['step']):x for x in json.load(gzip.open(A/'turn-reference.json.gz','rt'))};R={(x['lane'],x['step']):x for x in json.load(gzip.open(A/'straight-reference.json.gz','rt'))};times=[600,700,750,800,825,850,875,900,910,916,917,918,919,925,950];targets={};ground=[]
with gzip.open(P/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_DRIVE_TARGET '):
   v=line.split(); t,e=map(int,v[1:3]);
   if e==7 and t in times: targets[t]=list(map(float,v[3:]))
  elif line.startswith('CROW_GROUND_CONTACT '):
   v=line.split();t,sub,e=map(int,v[1:4]);
   if e==7 and 377<=t<=918: ground.append(v[1:])
def euler(q):
 x,y,z,w=q[3:7]
 return [math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),math.asin(max(-1,min(1,2*(w*y-z*x)))),math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))]
rows=[]
for t in times:
 x=S[7,t];y=R[7,t];rows.append({'observer_step':t,'state_epoch_caveat':'on replay lane 5 this state is preceding accepted frame; candidate timing inferred','position_m':x['q'][:3],'rpy_rad':euler(x['q']),'velocity_first6':x['v'][:6],'matched_straight_position_m':y['q'][:3],'matched_straight_rpy_rad':euler(y['q']),'policy_action':x['policy_action'],'drive_targets':targets.get(t)})
replay=json.load(gzip.open(P/'replay.json.gz','rt'))['payload']; names=replay['body_names']; first=ground[0] if ground else None
out={'scope':'used seed 20261126, right-turn lane 7 failure; not a navigation layout test','first_logged_ground_contact':{'step':int(first[0]),'substep':int(first[1]),'lane':int(first[2]),'bodyA_index':int(first[3]),'bodyA_name':names[int(first[3])],'bodyB_index':int(first[4]),'bodyB_name':names[int(first[4])],'pair_key':first[5],'feature_key':first[6],'point_xyz':list(map(float,first[7:10])),'normal_impulse':float(first[19])} if first else None,'rows':rows}
(A/'turn-first-ground.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
