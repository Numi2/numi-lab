from pathlib import Path
import gzip,json,math
R=Path.cwd();A=R/'artifacts/crow-v15-contact-slip-20261109';O=R/'artifacts/crow-v15-ground-contact-20261109';reports=[]
for course in ('reserved-final-i','reserved-final-j'):
 name='boundary-handoff-'+course;data=json.loads((O/(name+'-summary.json')).read_text());per=[{} for _ in range(2)]
 for line in gzip.open(O/(name+'.stderr.gz'),'rt'):
  if not line.startswith('CROW_SUPPORT_SAMPLE '):continue
  s=json.loads(line.split(' ',1)[1]);step=s['step']
  if not 2977<=step<=3001:continue
  up=0.;foot=False;other=False
  for c in s['contacts']:
   a,b=c['a_name'],c['b_name']
   if 'crow_course_perch' not in (a,b) or c['impulses'][0]<=0:continue
   n=b if a=='crow_course_perch' else a
   if n in ('crow_left_foot','crow_right_foot'):foot=True;up+=(-1 if a==n else 1)*c['normal'][2]*c['impulses'][0]
   else:other=True
  v=s['v'];per[s['lane']][step]=bool(foot and not other and up>0 and math.hypot(v[0],v[1])<=.12 and abs(v[2])<=.12)
 first=data['first_episode_navigation'];rows=[]
 for lane in range(2):
  count=sum(per[lane].get(step,False) for step in range(2977,3002));passed=count==25 and first['steps_by_environment'][lane]==3000 and first['maximum_waypoints_by_environment'][lane]==5 and not first['physical_failure_by_environment'][lane] and data['failed_environment_steps']==0;rows.append({'lane':lane,'terminal_replay_frames':list(range(2976,3001)),'accepted_supported_frames':count,'first_episode_gate':passed})
 reports.append({'course':course,'lanes':rows})
report={'trace_step_offset':1,'results':reports,'scope':'previously exposed development courses; flat perch support not grip'};(A/'terminal-support-aligned.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
