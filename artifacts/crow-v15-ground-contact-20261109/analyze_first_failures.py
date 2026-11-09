from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-ground-contact-20261109')
for stem in ('full-handoff-reserved-final-i','full-handoff-reserved-final-j'):
 data=json.loads((A/(stem+'-summary.json')).read_text());first=data['first_episode_navigation'];limits=first['steps_by_environment'];out=[];seen=set()
 with gzip.open(A/(stem+'.stderr.gz'),'rt') as f:
  for line in f:
   if not line.startswith('CROW_CONTACT_TRACE '):continue
   x=json.loads(line.split(' ',1)[1]);lane=x['lane']
   if lane in seen or x['step']>limits[lane]+1:continue
   seen.add(lane);q=x['pre_q'];v=x['pre_v'];qx,qy,qz,qw=q[3:7];roll=math.atan2(2*(qw*qx+qy*qz),1-2*(qx*qx+qy*qy));pitch=math.asin(max(-1,min(1,2*(qw*qy-qz*qx))));yaw=math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz));out.append({'lane':lane,'trace_step':x['step'],'reported_first_episode_steps':limits[lane],'position_m':q[:3],'roll_pitch_yaw_rad':[roll,pitch,yaw],'linear_velocity_m_s':v[:3],'angular_velocity_rad_s':v[3:6],'policy_output_not_physical_applied_action':x.get('policy_action'),'contacts':x['contacts']})
 (A/(stem+'-first-failures.json')).write_text(json.dumps(out,indent=2)+'\n');print(stem,[(x['lane'],x['trace_step'],[(c['a_name'],c['b_name']) for c in x['contacts']]) for x in out])
