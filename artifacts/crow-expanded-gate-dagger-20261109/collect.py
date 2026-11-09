from pathlib import Path
import gzip,hashlib,json,math,os,shutil,subprocess
R=Path.cwd();A=R/'artifacts/crow-expanded-gate-dagger-20261109';B=R/'build/crow-expanded-gate-dagger-20261109';B.mkdir(parents=True,exist_ok=True)
course=os.environ['CROW_DEV_COURSE'];seed=os.environ['CROW_START_SEED'];batch=os.environ['CROW_BATCH'];O=A/batch;O.mkdir(exist_ok=True)
lib=R/'build/crow-mixed-native-observer-20261109/libmetalrobo.dylib'
cmd=json.loads((R/'artifacts/crow-hierarchical-contract-20261109/oracle/results.json').read_text())['argv']
pack=B/(batch+'.rollout')
for k,v in [('--metallib',str(R/'build/crow-expanded-gate-dagger-20261109/recovery.metallib')),('--policy-pack',str(R/'build/crow-gate-terminal-training-20261109/landing-actor.policypack')),('--envs','16'),('--seed',seed),('--birdflow-navigation-course',course),('--state-trace',str(B/(course+'.tsv')))]:cmd[cmd.index(k)+1]=v
cmd[cmd.index('--birdflow-journey-variant')+1]='v15-hierarchical-navigation'
cmd+=['--base-policy-pack',str(R/'build/crow-stage2-ablation-expert-20261109/combined-expert.policypack')]
cmd+=['--birdflow-journey-teacher','--birdflow-journey-student-authority','1','--rollout-pack',str(pack)]
env={**os.environ,'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','DYLD_LIBRARY_PATH':str(lib.parent),'PYTHONDONTWRITEBYTECODE':'1'}
control=cmd.copy()
for option in ['--rollout-pack','--birdflow-journey-student-authority']:
 i=control.index(option);del control[i:i+2]
control.remove('--birdflow-journey-teacher')
control_env=env.copy();control_env.pop('CROW_SUPPORT_TRACE',None)
cp=subprocess.run(control,cwd=R,env=control_env,text=True,capture_output=True,timeout=600)
if cp.returncode:raise RuntimeError(cp.stderr[-5000:])
(O/'student-control.stdout').write_text(cp.stdout)
with (O/'student-control.stderr.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(cp.stderr.encode())
p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=600)
(O/'rollout.stdout').write_text(p.stdout)
with (O/'rollout.stderr.gz').open('wb') as f:
 with gzip.GzipFile(fileobj=f,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
if p.returncode:raise RuntimeError(p.stderr[-5000:])
def samples(stderr):
 return {(x['lane'],x['step']):x for line in stderr.splitlines() if line.startswith('CROW_FLIGHT_SAMPLE ') for x in [json.loads(line.split(' ',1)[1])]}
student=samples(cp.stderr); mixed=samples(p.stderr)
control_first=json.loads(cp.stdout)['first_episode_navigation']
compared=[];mismatches=[]
for k,x in student.items():
 lane,t=k; takeover=1000+50*(lane%4)
 if t<=takeover and t<=control_first['steps_by_environment'][lane]:
  compared.append(k)
  if k not in mixed or x['q']!=mixed[k]['q'] or x['v']!=mixed[k]['v']:mismatches.append(k)
if mismatches or not compared:raise RuntimeError('student prefix does not match teacher-off control: '+repr(mismatches[:5]))
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
report={'course':course,'seed':seed,'argv':cmd,'raw_pack_bytes':pack.stat().st_size,'raw_pack_sha256':rawhash,'compressed_pack_bytes':compressed.stat().st_size,'compressed_pack_sha256':sha(compressed),'round_trip_verified':True,'frozen_controller_sha256':sha(R/'src/metal/CrowFlightControllerV1.metal'),'shader_sha256':sha(R/'build/crow-expanded-gate-dagger-20261109/recovery.metallib'),'library_sha256':sha(lib),'policy_sha256':sha(R/'build/crow-gate-terminal-training-20261109/landing-actor.policypack'),'base_policy_sha256':sha(R/'build/crow-stage2-ablation-expert-20261109/combined-expert.policypack'),'failed_environment_steps':data['failed_environment_steps'],'lanes':rows,'eligible_lanes':sum(x['eligible'] for x in rows),'sampling':'teacher labels on; full student route and landing until lane-specific physical recovery; first episode only','student_takeover_steps':[1000+50*(i%4) for i in range(16)],'student_prefix_compared_samples':len(compared),'student_prefix_mismatches':len(mismatches)}
(O/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
print(course,'eligible',report['eligible_lanes'],'raw MB',round(report['raw_pack_bytes']/1e6,1),'compressed MB',round(report['compressed_pack_bytes']/1e6,1),flush=True)
