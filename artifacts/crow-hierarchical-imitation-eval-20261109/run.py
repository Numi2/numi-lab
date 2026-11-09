from pathlib import Path
import gzip,hashlib,json,math,os,subprocess
R=Path.cwd();A=R/'artifacts/crow-hierarchical-imitation-eval-20261109';B=R/'build/crow-hierarchical-imitation-eval-20261109';B.mkdir(parents=True,exist_ok=True)
mode='actor';course=os.environ['CROW_DEV_COURSE'];seed=os.environ['CROW_START_SEED'];O=A/course;O.mkdir(exist_ok=True)
cmd=json.loads((R/'artifacts/crow-latched-joint-dev-20261109/training/route-results.json').read_text())['argv']
for k,v in [('--metallib',str(R/'build/crow-hierarchical-contract-20261109/actor.metallib')),('--policy-pack',str(R/'build/crow-hierarchical-imitation-20261109/initial-imitation.policypack')),('--state-trace',str(B/(course+'-trace.tsv')))]:cmd[cmd.index(k)+1]=v
cmd[cmd.index('--birdflow-navigation-course')+1]=course
cmd[cmd.index('--seed')+1]=seed
env={**os.environ,'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','DYLD_LIBRARY_PATH':str(R/'build/crow-hierarchical-bounded-20261109'),'PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=480)
(O/'rollout.stdout').write_text(p.stdout)
with (O/'rollout.stderr.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
if p.returncode:raise RuntimeError(p.stderr[-5000:])
data=json.loads(p.stdout);first=data['first_episode_navigation']
support=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_SUPPORT_SAMPLE ')]
rows=[]
for lane in range(16):
 end=first['steps_by_environment'][lane];valid=[]
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
 rows.append({'lane':lane,'steps':end,'waypoints':first['maximum_waypoints_by_environment'][lane],'physical_failure':first['physical_failure_by_environment'][lane],'terminal_support_steps':terminal,'supported_finish':terminal>=25 and not first['physical_failure_by_environment'][lane] and first['maximum_waypoints_by_environment'][lane]>=5})
contacts=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_CONTACT_TRACE ')]
for row in rows:
 matches=[z for z in contacts if z['lane']==row['lane'] and z['step']<=row['steps']+1]
 row['first_contact_candidate']=matches[0] if matches else None
r={'course':course,'seed':seed,'mode':mode,'teacher_sampling':False,'privileged_oracle':mode=='oracle','argv':cmd,'failed_environment_steps':data['failed_environment_steps'],'first_episode':first,'lanes':rows,'supported_finishes':sum(x['supported_finish'] for x in rows),'physical_failures':sum(x['physical_failure'] for x in rows),'scope':'oracle parity diagnostic' if mode=='oracle' else 'teacher-off trained actor; development layouts only'}
(O/'results.json').write_text(json.dumps(r,indent=2)+'\n');print(course,'supported',r['supported_finishes'],'physical failures',r['physical_failures'],'waypoints',[x['waypoints'] for x in rows],flush=True)
