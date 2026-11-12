from pathlib import Path
import gzip,json,collections,math
A=Path('artifacts/crow-v15-walking-forward-impulse-audit-20261112');C=A/'candidate-20261127-rebound';R=Path('artifacts/crow-v15-sole-support-swing-screen-20261112/candidate-20261127-rebound')
def read(path,forward):
 state={};sub={};horizontal={}
 with gzip.open(path/'run.stderr.gz','rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    z=json.loads(l.split(' ',1)[1]);state[z['lane'],z['step']]=z
   elif l.startswith('CROW_TWO_TOE_SUBSTEP '):
    z=l.split();t,k,e=map(int,z[1:4]);sub[e,t,k]=z[4:]
   elif forward and l.startswith('CROW_FORWARD_IMPULSE '):
    z=l.split();t,k,e=map(int,z[1:4]);horizontal[e,t,k]=list(map(float,z[4:]));assert len(horizontal[e,t,k])==6
 assert len(state)==2408 and len(sub)==38528
 if forward:assert len(horizontal)==12800
 return state,sub,horizontal
c=read(C,True);r=read(R,False);ep=json.loads((C/'run.stdout').read_text())['first_episode_navigation'];assert ep['steps_by_environment']==[300]*8 and not any(ep['physical_failure_by_environment'])
parity={'endpoint_mismatches':sum(c[0][x]!=r[0][x] for x in c[0]),'substep_mismatches':sum(c[1][x]!=r[1][x] for x in c[1])};assert not any(parity.values()),parity
lanes=[]
for e in range(8):
 data=[c[2][e,t,k] for t in range(201,301) for k in range(16)];categories={}
 for i,name in enumerate(('left_sole','left_front','left_rear','right_sole','right_front','right_rear')):
  v=[row[i] for row in data];categories[name]={'net_Ns':sum(v),'positive_Ns':sum(x for x in v if x>0),'negative_Ns':sum(x for x in v if x<0)}
 total=[sum(row) for row in data];start=c[0][e,200];end=c[0][e,300];lanes.append({'lane':e,'root_x_displacement_m':end['q'][0]-start['q'][0],'start_root_x_velocity_m_s':start['v'][0],'end_root_x_velocity_m_s':end['v'][0],'ground_world_x_net_Ns':sum(total),'ground_world_x_positive_Ns':sum(x for x in total if x>0),'ground_world_x_negative_Ns':sum(x for x in total if x<0),'components':categories})
out={'parity':parity,'first_episode_steps':ep['steps_by_environment'],'lanes':lanes,'scope':'Native accepted ground impulses, 1.25 ms steps and 4-6s only, in world x (not per-sample body-forward projection). Positive/negative totals are sums of signed group or total substeps, respectively, not individual contact work. Changes in whole-body momentum also include other external forces. Root velocity is not COM velocity; no direct friction or energy measurement, and no biological validation.'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
