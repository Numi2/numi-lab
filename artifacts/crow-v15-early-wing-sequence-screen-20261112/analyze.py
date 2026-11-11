from pathlib import Path
import gzip,json,math
a=Path('artifacts/crow-v15-early-wing-sequence-screen-20261112/autonomous-20261126')
d=json.loads((a/'run.stdout').read_text());first=d['first_episode_navigation'];states={}
with gzip.open(a/'run.stderr.gz','rt') as f:
 for line in f:
  if line.startswith('CROW_SUPPORT_SAMPLE '):
   x=json.loads(line.split(' ',1)[1]);states[x['lane'],x['step']]=x
def brief(x):
 if not x:return None
 q=x['q'];v=x['v'];a,b,c,w=q[3:7]
 roll=math.atan2(2*(w*a+b*c),1-2*(a*a+b*b))
 pitch=math.asin(max(-1.,min(1.,2*(w*b-c*a))))
 yaw=math.atan2(2*(w*c+a*b),1-2*(b*b+c*c))
 return {'observer_step':x['step'],'position_m':q[:3],'attitude_roll_pitch_yaw_rad':[roll,pitch,yaw],'linear_velocity_m_s':v[:3],'angular_velocity_rad_s':v[3:6],'policy_action':x['policy_action']}
rows=[]
for e in range(8):
 end=first['steps_by_environment'][e];contacts=[]
 for t in range(356,end+2):
  x=states.get((e,t))
  if not x:continue
  hits=[c for c in x['contacts'] if c['impulses'][0]>1e-7 and 'locomotion_ground' not in (c['a_name'],c['b_name'])]
  if hits:contacts.append((t,hits))
 last=states.get((e,end))
 ct,hits=contacts[0] if contacts else (None,[])
 rows.append({'lane':e,'first_episode_steps':end,'physical_failure':first['physical_failure_by_environment'][e],'max_waypoints':first['maximum_waypoints_by_environment'][e],'first_observed_non_ground_contact':{'observer_step':ct,'pairs':[[c['a_name'],c['b_name']] for c in hits],'impulses': [c['impulses'] for c in hits], 'contact_state':brief(states.get((e,ct))),'previous_state':brief(states.get((e,ct-1)))} if ct else None,'last_observed_first_episode_state':brief(last)})
out={'teacher':d['birdflow_journey_teacher'],'action_source':d['action_source'],'host_requested_resets':d['host_requested_resets'],'failed_environment_steps':d['failed_environment_steps'],'rows':rows,'limitations':'Contact is endpoint observation, not complete substep coverage; ordering at a fatal step is not established. Policy actions are recorded actor outputs and not the applied actuator targets. Telemetry can include subsequent episodes; analysis stops at first episode boundary. Not a final-layout result.'}
(a/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:
 c=r['first_observed_non_ground_contact']
 print(r['lane'],'end',r['first_episode_steps'],'failure',r['physical_failure'],'waypoints',r['max_waypoints'],'first contact',None if not c else (c['observer_step'],c['pairs']),'last pos',r['last_observed_first_episode_state']['position_m'] if r['last_observed_first_episode_state'] else None)
