from pathlib import Path
import gzip,json,math
A=Path('artifacts/crow-v15-v1-teacher-landing-approach-20261111');B=Path('artifacts/crow-v15-v1-teacher-launch-parity-20261111');j=json.loads((A/'route.stdout').read_text())
def read(p):
 z={};clear=[];fatal=None
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('CROW_SUPPORT_SAMPLE '):
    d=json.loads(l.split(' ',1)[1]);z[d['step']]=d
   elif l.startswith('CROW_CLEARANCE_SAMPLE '):clear.append(json.loads(l.split(' ',1)[1]))
   elif l.startswith('CROW_CONTACT_TRACE ') and fatal is None:fatal=json.loads(l.split(' ',1)[1])
 return z,clear,fatal
a,clear,fatal=read(A/'route.stderr.gz');b,_,_=read(B/'route.stderr.gz');diff=[t for t in range(1,1306) if t not in a or t not in b or any(a[t][k]!=b[t][k] for k in ('q','v'))];timeline=[]
for t in (1305,1350,1400,1450,1500,1520,1540,1550,1553,1554):
 if t in a:
  d=a[t];q=d['q'];v=d['v'];timeline.append({'step':t,'position_m':q[:3],'orientation_xyzw':q[3:7],'tilt_rad':math.acos(max(-1,min(1,1-2*(q[3]**2+q[4]**2)))),'velocity_m_s':v[:3],'angular_velocity_rad_s':v[3:6]})
out={'first_episode':j['first_episode_navigation'],'failed_environment_steps':j['failed_environment_steps'],'foot_support_step_count':j['foot_support_step_count'],'height_or_tilt_termination_count':j['height_or_tilt_termination_count'],'q_v_differences_through_1305':diff,'first_fatal_obstacle_contact':fatal,'clearance_samples':clear,'first_episode_timeline':timeline,'scope':'privileged development J landing diagnostic; post-failure reset excluded'};(A/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
