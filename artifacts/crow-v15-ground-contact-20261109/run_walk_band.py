from pathlib import Path
import gzip, hashlib, json, math, os, subprocess
R=Path.cwd();A=R/'artifacts/crow-v15-ground-contact-20261109';B=R/'build/crow-v15-ground-contact-20261109'
base=json.loads((A/'bindings.json').read_text())['argv']
for key,value in [('--minimum-difficulty-band','1'),('--maximum-difficulty-band','1'),('--state-trace',str(B/'walk-band.tsv'))]:base[base.index(key)+1]=value
base+=['--crow-replay-pack',str(B/'walk-band.json')]
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(A/'walk-band-bindings.json').write_text(json.dumps({'argv':base,'plan_sha256':h(A/'walk-band-plan.md'),'observer_library_sha256':h(B/'libmetalrobo.dylib'),'source_binding':'bindings.json'},indent=2)+'\n')
env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'}
p=subprocess.run(base,cwd=R,env=env,text=True,capture_output=True,timeout=240)
(A/'walk-band.stdout').write_text(p.stdout)
with (A/'walk-band.stderr.gz').open('wb') as raw:
 with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:z.write(p.stderr.encode())
if p.returncode:raise RuntimeError('walk-band rollout failed: '+p.stderr[-3000:])
raw=(B/'walk-band.json').read_bytes()
with (A/'walk-band.replay.json.gz').open('wb') as out:
 with gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as z:z.write(raw)
fs=json.loads(raw)['payload']['frames'];d=json.loads(p.stdout)
samples=[json.loads(x.split(' ',1)[1]) for x in p.stderr.splitlines() if x.startswith('CROW_SUPPORT_SAMPLE ')]
result=[]
for lane in range(2):
 loads={}
 for s in samples:
  if s['lane']!=lane:continue
  flags=loads.setdefault(s['step'],set())
  for c in s['contacts']:
   a,b=c['a_name'],c['b_name']
   if 'locomotion_ground' not in (a,b):continue
   isA=a in ('crow_left_foot','crow_right_foot');isB=b in ('crow_left_foot','crow_right_foot')
   if not(isA or isB):continue
   if (-1 if isA else 1)*c['normal'][2]*c['impulses'][0]>0:flags.add(a if isA else b)
 left='crow_left_foot';right='crow_right_foot'
 row={'lane':lane,'left_loaded_samples':sum(left in x for x in loads.values()),'right_loaded_samples':sum(right in x for x in loads.values()),'left_only_samples':sum(x=={left} for x in loads.values()),'right_only_samples':sum(x=={right} for x in loads.values())}
 # Replay pack contains selected lane 1 only; lane 0 must be assessed separately.
 if lane==1:
  a,b=fs[0]['q'],fs[-1]['q'];row['net_planar_displacement_m']=math.hypot(b[0]-a[0],b[1]-a[1])
 result.append(row)
out={'rows':result,'failed_environment_steps':d.get('failed_environment_steps'),'first_episode_navigation':d.get('first_episode_navigation'),'lane0_displacement':'not observed: replay export selected lane 1','scope':'exploratory 350 accepted endpoint samples; cannot pass bilateral lane gate without lane-0 trajectory'}
(A/'walk-band-summary.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
