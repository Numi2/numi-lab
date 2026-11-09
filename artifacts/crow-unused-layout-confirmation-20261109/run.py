from pathlib import Path
import gzip,hashlib,json,math,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-unused-layout-confirmation-20261109';B=R/'build/crow-unused-layout-confirmation-20261109';plan=json.loads((A/'plan.json').read_text())
course=os.environ['CROW_FINAL_COURSE'];assert course in plan['layouts']
seed=plan['layouts'][course];O=A/course;O.mkdir(exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
for rel,digest in plan['bound_sha256'].items():assert sha(R/rel)==digest,('frozen input changed',rel)
cmd=json.loads((R/'artifacts/crow-stage3-fresh-eval-20261109/baseline/training/results.json').read_text())['argv'].copy()
for key,value in [('--birdflow-navigation-course',course),('--seed',str(seed)),('--state-trace',str(B/(course+'-lane1.tsv')))]:cmd[cmd.index(key)+1]=value
assert '--no-scheduled-resets' in cmd and '--birdflow-journey-teacher' not in cmd and int(cmd[cmd.index('--envs')+1])==16 and int(cmd[cmd.index('--steps')+1])==3000
env={**os.environ,'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_CLEARANCE_TRACE':'1','DYLD_LIBRARY_PATH':str(R/'build/crow-mixed-native-observer-20261109'),'PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=600)
(O/'rollout.stdout').write_text(p.stdout)
with (O/'rollout.stderr.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
if p.returncode:raise RuntimeError('final trial failed, retained outputs: '+p.stderr[-5000:])
d=json.loads(p.stdout);first=d['first_episode_navigation']
support=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_SUPPORT_SAMPLE ')]
contacts=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_CONTACT_TRACE ')]
samples=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_FLIGHT_SAMPLE ')]
clearances=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_CLEARANCE_SAMPLE ')]
rows=[]
for lane in range(16):
 end=first['steps_by_environment'][lane];wp=first['maximum_waypoints_by_environment'][lane];physical=first['physical_failure_by_environment'][lane];valid=[]
 for z in support:
  if z['lane']!=lane or z['step']>end:continue
  feet=[c for c in z['contacts'] if 'foot' in c['a_name'] or 'foot' in c['b_name']]
  other=[c for c in z['contacts'] if not ('foot' in c['a_name'] or 'foot' in c['b_name'])]
  up=sum((-1 if 'foot' in c['a_name'] else 1)*c['normal'][2]*c['impulses'][0] for c in feet)
  if feet and not other and up>0 and math.hypot(*z['v'][:2])<=.12 and abs(z['v'][2])<=.12:valid.append(z['step'])
 terminal=0
 for t in reversed(valid):
  if t==end-terminal:terminal+=1
  else:break
 matches=[z for z in clearances if z['lane']==lane and z['end_step'] in (end,end+1)]
 assert matches,(lane,end)
 minimum=next((z for z in matches if z['end_step']==end),matches[0])
 assert math.isfinite(minimum['minimum_m']) and minimum['minimum_m']>=0
 finish=terminal>=25 and not physical and wp>=5 and end==3000 and d['failed_environment_steps']==0
 fatal=[z for z in contacts if z['lane']==lane and z['step']<=end+1]
 recorded=[z for z in samples if z['lane']==lane and z['step']<=end]
 last=recorded[-1] if recorded else None
 failure=None if finish else {'kind':'physical_contact' if physical else 'waypoint_timeout' if wp<5 else 'unsupported_landing','contact_witness':fatal[0] if fatal else None,'last_sample':last}
 rows.append({'lane':lane,'steps':end,'waypoints_reached':wp,'physical_failure':physical,'terminal_support_steps':terminal,'supported_finish':finish,'first_failure':failure,'minimum_obstacle_clearance':minimum})
report={'layout':course,'seed':seed,'full_start_count':16,'teacher_enabled':False,'scheduled_resets_enabled':False,'first_episode_only':True,'argv':cmd,'failed_environment_steps':d['failed_environment_steps'],'first_episode':first,'lanes':rows,'waypoint_completions':sum(x['waypoints_reached']>=5 for x in rows),'supported_finishes':sum(x['supported_finish'] for x in rows),'physical_failures':sum(x['physical_failure'] for x in rows),'minimum_sampled_clearance_m':min(x['minimum_obstacle_clearance']['minimum_m'] for x in rows),'meets_layout_gate':sum(x['supported_finish'] for x in rows)>=15 and d['failed_environment_steps']==0,'clearance_scope':'minimum over accepted 20 ms control-step endpoints, all bird collision boxes versus four gate/slalom boxes; not continuous-time and excludes landing perch'}
(O/'results.json').write_text(json.dumps(report,indent=2)+'\n')
print(course,'supported',report['supported_finishes'],'of 16, waypoint completions',report['waypoint_completions'],'physical failures',report['physical_failures'],'min sampled clearance',report['minimum_sampled_clearance_m'],'gate',report['meets_layout_gate'],flush=True)
