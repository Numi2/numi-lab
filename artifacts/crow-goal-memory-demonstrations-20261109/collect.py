from pathlib import Path
import gzip,hashlib,json,math,os,shutil,subprocess
R=Path.cwd();A=R/'artifacts/crow-goal-memory-demonstrations-20261109';B=R/'build/crow-goal-memory-demonstrations-20261109';B.mkdir(parents=True,exist_ok=True)
course=os.environ['CROW_DEV_COURSE'];seed=os.environ['CROW_START_SEED'];O=A/course;O.mkdir(exist_ok=True)
lib=R/'build/crow-hierarchical-goal-memory-20261109/libmetalrobo.dylib'
cmd=json.loads((R/'artifacts/crow-hierarchical-contract-20261109/oracle/results.json').read_text())['argv']
pack=B/(course+'.rollout')
for k,v in [('--metallib',str(R/'build/crow-hierarchical-goal-memory-20261109/actor.metallib')),('--policy-pack',str(R/'build/crow-hierarchical-goal-memory-20261109/hierarchical-initial.policypack')),('--envs','8'),('--seed',seed),('--birdflow-navigation-course',course),('--state-trace',str(B/(course+'.tsv')))]:cmd[cmd.index(k)+1]=v
cmd+=['--birdflow-journey-teacher','--birdflow-journey-student-authority','0','--rollout-pack',str(pack)]
env={**os.environ,'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','DYLD_LIBRARY_PATH':str(lib.parent),'PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=600)
(O/'rollout.stdout').write_text(p.stdout)
with (O/'rollout.stderr.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
if p.returncode:raise RuntimeError(p.stderr[-5000:])
data=json.loads(p.stdout);first=data['first_episode_navigation']
support=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_SUPPORT_SAMPLE ')]
rows=[]
for lane in range(8):
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
 ok=(terminal>=25 and end==3000 and not first['physical_failure_by_environment'][lane] and first['maximum_waypoints_by_environment'][lane]>=5 and data['failed_environment_steps']==0)
 rows.append({'lane':lane,'steps':end,'waypoints':first['maximum_waypoints_by_environment'][lane],'physical_failure':first['physical_failure_by_environment'][lane],'terminal_support_steps':terminal,'eligible':ok})
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
rawhash=sha(pack);compressed=O/'rollout.bin.gz'
with pack.open('rb') as f, compressed.open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,compresslevel=6) as z:shutil.copyfileobj(f,z,1024*1024)
h=hashlib.sha256()
with gzip.open(compressed,'rb') as z:
 for block in iter(lambda:z.read(1024*1024),b''):h.update(block)
if h.hexdigest()!=rawhash:raise RuntimeError('compressed pack did not round-trip')
report={'course':course,'seed':seed,'argv':cmd,'raw_pack_bytes':pack.stat().st_size,'raw_pack_sha256':rawhash,'compressed_pack_bytes':compressed.stat().st_size,'compressed_pack_sha256':sha(compressed),'round_trip_verified':True,'frozen_controller_sha256':sha(R/'src/metal/CrowFlightControllerV1.metal'),'shader_sha256':sha(R/'build/crow-hierarchical-goal-memory-20261109/actor.metallib'),'library_sha256':sha(lib),'policy_sha256':sha(R/'build/crow-hierarchical-goal-memory-20261109/hierarchical-initial.policypack'),'failed_environment_steps':data['failed_environment_steps'],'lanes':rows,'eligible_lanes':sum(x['eligible'] for x in rows),'sampling':'teacher on, student authority zero; first episode only'}
(O/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(course,'eligible',report['eligible_lanes'],'raw MB',round(report['raw_pack_bytes']/1e6,1),'compressed MB',round(report['compressed_pack_bytes']/1e6,1),flush=True)
