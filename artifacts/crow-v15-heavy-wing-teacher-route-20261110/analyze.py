from pathlib import Path
import gzip,json
A=Path('artifacts/crow-v15-heavy-wing-teacher-route-20261110');report={}
for course in ['reserved-final-j','reserved-final-i']:
 name='development-'+course;d=json.loads((A/(name+'.stdout')).read_text());e=d['first_episode_navigation'];contacts={};clear={}
 with gzip.open(A/(name+'.stderr.gz'),'rt') as f:
  for l in f:
   if l.startswith('CROW_CONTACT_TRACE '):
    x=json.loads(l.split(' ',1)[1]);lane=x['lane'];
    if str(lane) not in contacts and x['step']<=e['steps_by_environment'][lane]+1:
     q=x['pre_q'];v=x['pre_v'];cs=[c for c in x['contacts'] if 'crow_course_' in c['a_name'] or 'crow_course_' in c['b_name'] or 'locomotion_ground' in (c['a_name'],c['b_name'])];contacts[str(lane)]={'step':x['step'],'position_m':q[:3],'orientation_xyzw':q[3:7],'velocity_mps':v[:3],'angular_velocity_rad_s':v[3:6],'contacts':cs}
   elif l.startswith('CROW_CLEARANCE_SAMPLE '):
    x=json.loads(l.split(' ',1)[1]);lane=x['lane'];
    if x['end_step']<=e['steps_by_environment'][lane]+1 and (str(lane) not in clear or x['minimum_m']<clear[str(lane)]['minimum_m']):clear[str(lane)]=x
 report[course]={'first_episode':e,'failed_environment_steps':d['failed_environment_steps'],'first_contacts':contacts,'minimum_sampled_clearances':clear,'teacher_enabled':True}
(A/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
for c,x in report.items():print(c,'waypoints',x['first_episode']['maximum_waypoints_by_environment'],'physical failures',x['first_episode']['physical_failure_count'],'solver failed steps',x['failed_environment_steps'])
