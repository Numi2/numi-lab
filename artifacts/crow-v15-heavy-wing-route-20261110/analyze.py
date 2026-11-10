from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-heavy-wing-route-20261110');out={}
for p in sorted(A.glob('diagnostic*.stdout')):
 name=p.name[:-7];d=json.loads(p.read_text());e=d['first_episode_navigation'];end=e['steps_by_environment'][0];contact=None;clear=[];samples=[]
 with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
  for l in f:
   if l.startswith('CROW_CONTACT_TRACE '):
    x=json.loads(l.split(' ',1)[1]);
    if contact is None and x['lane']==0 and x['step']<=end+1:
     q=x['pre_q'];v=x['pre_v'];cs=[c for c in x['contacts'] if 'crow_course_' in c['a_name'] or 'crow_course_' in c['b_name'] or 'locomotion_ground' in (c['a_name'],c['b_name'])];contact={'logged_step':x['step'],'position_m':q[:3],'quaternion_xyzw':q[3:7],'velocity_mps':v[:3],'angular_velocity_rad_s':v[3:6],'contacts':cs,'note':'policy_action is a latent, not the applied low-level controller action'}
   elif l.startswith('CROW_CLEARANCE_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);
    if x['lane']==0 and x['end_step']<=end+1:clear.append(x)
   elif l.startswith('CROW_FLIGHT_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);
    if x['lane']==0 and x['step']<=end:samples.append({'step':x['step'],'position_m':x['q'][:3],'velocity_mps':x['v'][:3]})
 out[name]={'first_episode':e,'solver_failed_steps':d['failed_environment_steps'],'first_fatal_contact':contact,'minimum_sampled_clearance':min(clear,key=lambda x:x['minimum_m']) if clear else None,'samples':samples,'scope':'old policy weights rebound to a changed body and control contract; prior development layouts, not autonomous qualification'}
(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for n,r in out.items():print(n,'waypoints',r['first_episode']['maximum_waypoints_by_environment'][0],'end',r['first_episode']['steps_by_environment'][0],'physical',r['first_episode']['physical_failure_by_environment'][0],'clearance',r['minimum_sampled_clearance']['minimum_m'] if r['minimum_sampled_clearance'] else None)
