from pathlib import Path
import gzip,json,collections,math
R=Path.cwd();A=R/'artifacts/crow-v15-toe-flight-screen-20261110';all=[]
for seed in ('20261120','20261121'):
 d=json.loads((A/(seed+'-analysis.json')).read_text());lanes={x['lane']:x for x in d['lanes']};first={}
 with gzip.open(A/(seed+'.stderr.gz'),'rt') as f:
  for l in f:
   if not l.startswith('CROW_CONTACT_TRACE '):continue
   x=json.loads(l.split(' ',1)[1]);lane=x['lane'];v=lanes[lane]
   if lane in first or not v['physical_failure'] or x['step']>v['steps']+1:continue
   cs=[c for c in x['contacts'] if c['normal_impulse']>0]
   if cs:
    q=x['pre_q'];vx=x['pre_v'];a,b,c,w=q[3:7];pitch=math.asin(max(-1,min(1,2*(w*b-c*a))));roll=math.atan2(2*(w*a+b*c),1-2*(a*a+b*b));yaw=math.atan2(2*(w*c+a*b),1-2*(b*b+c*c));first[lane]={'lane':lane,'step':x['step'],'collider_a':cs[0]['a_name'],'collider_b':cs[0]['b_name'],'normal_impulse_Ns':cs[0]['normal_impulse'],'root_position_m':q[:3],'root_velocity':vx[:6],'roll_pitch_yaw_rad':[roll,pitch,yaw],'policy_action':x['policy_action'],'action_limit':'zero diagnostic policy outputs, not the flight controller\'s actual actuator commands'}
 pairs=collections.Counter(x['collider_a']+' / '+x['collider_b'] for x in first.values());all.append({'seed':seed,'recorded_first_contacts':len(first),'physical_failures':sum(x['physical_failure'] for x in lanes.values()),'pairs':dict(pairs),'contacts':list(first.values())})
(A/'contact-analysis.json').write_text(json.dumps(all,indent=2)+'\n');print([(x['seed'],x['recorded_first_contacts'],x['pairs']) for x in all])
