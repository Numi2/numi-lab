from pathlib import Path
import gzip, hashlib, json, math, os, subprocess, sys
R=Path.cwd(); A=R/'artifacts/crow-v15-ground-contact-20261109'; B=R/'build/crow-v15-ground-contact-20261109'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
 return h.hexdigest()
base=json.loads((R/'artifacts/crow-unused-layout-confirmation-round4-20261109/reserved-final-i/results.json').read_text())['argv']
for key,value in [('--envs','2'),('--steps','350'),('--state-trace',str(B/'trace.tsv'))]:
 base[base.index(key)+1]=value
inputs=[Path(base[0]),R/'src/metal/CrowFlightControllerV1.metal',R/'src/metal/LocomotionTask.metal',Path(base[base.index('--metallib')+1]),Path(base[base.index('--policy-pack')+1]),Path(base[base.index('--base-policy-pack')+1]),B/'libmetalrobo.dylib',A/'plan.md',A/'observer.patch',A/'run_pair.py']
(A/'bindings.json').write_text(json.dumps({'argv':base,'sha256':{str(p.relative_to(R)):sha(p) for p in inputs},'arms':['observer_off','observer_on'],'lane':1},indent=2)+'\n')
frames={}
for arm in ['observer_off','observer_on']:
 cmd=base.copy();cmd[cmd.index('--state-trace')+1]=str(B/(arm+'.tsv'));cmd+=['--crow-replay-pack',str(B/(arm+'.json'))]
 env={**os.environ,'DYLD_LIBRARY_PATH':str(B),'PYTHONDONTWRITEBYTECODE':'1'}
 env.pop('CROW_CONTACT_TRACE',None);env.pop('CROW_SUPPORT_TRACE',None)
 if arm=='observer_on':env.update(CROW_CONTACT_TRACE='1',CROW_SUPPORT_TRACE='1')
 p=subprocess.run(cmd,cwd=R,env=env,text=True,capture_output=True,timeout=240)
 (A/(arm+'.stdout')).write_text(p.stdout)
 with (A/(arm+'.stderr.gz')).open('wb') as raw:
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as f:f.write(p.stderr.encode())
 if p.returncode: raise RuntimeError(arm+' failed: '+p.stderr[-3000:])
 raw=(B/(arm+'.json')).read_bytes()
 with (A/(arm+'.replay.json.gz')).open('wb') as out:
  with gzip.GzipFile(fileobj=out,mode='wb',mtime=0) as z:z.write(raw)
 frames[arm]=json.loads(raw)['payload']['frames']
 trace=p.stderr.splitlines()
 if arm=='observer_on': samples=[json.loads(x.split(' ',1)[1]) for x in trace if x.startswith('CROW_SUPPORT_SAMPLE ')]
if len(frames['observer_off'])!=350 or len(frames['observer_on'])!=350:raise RuntimeError('incomplete replay')
equal=frames['observer_off']==frames['observer_on']
(A/'parity.json').write_text(json.dumps({'complete_replay_frames_equal':equal,'frame_count':350,'scope':'same build, same two lanes, observer enablement only'},indent=2)+'\n')
if not equal:raise RuntimeError('observer changed replay; contact interpretation invalid')
fs=frames['observer_on']; ground=[]
for s in samples:
 if s['lane']!=1:continue
 for c in s['contacts']:
  a,b=c['a_name'],c['b_name']
  if 'locomotion_ground' not in (a,b):continue
  footA=a in ('crow_left_foot','crow_right_foot');footB=b in ('crow_left_foot','crow_right_foot')
  if not (footA or footB):continue
  up=(-1 if footA else 1)*c['normal'][2]*c['impulses'][0]
  ground.append({'step':s['step'],'foot':a if footA else b,'upward_impulse':up,'normal_impulse':c['impulses'][0],'separation_m':c['separation']})
if not ground:raise RuntimeError('no ground contacts observed; instrument invalid')
loaded=[x for x in ground if x['upward_impulse']>0]
if not loaded:raise RuntimeError('no positive foot impulse observed; instrument invalid')
pre=fs[:100]; symmetry=sum(abs(f['q'][14]-f['q'][17])<.01 for f in pre)/len(pre)
p0,p1=pre[0]['q'],pre[-1]['q']; distance=math.hypot(p1[0]-p0[0],p1[1]-p0[1])
report={'observer_parity':equal,'lane':1,'frames':len(fs),'prelaunch_hip_symmetry_fraction':symmetry,'prelaunch_planar_net_displacement_m':distance,'prediction_met':symmetry>=.9 and distance<.03,'ground_contact_rows':len(ground),'positive_ground_impulse_rows':len(loaded),'first_positive_ground_step':min(x['step'] for x in loaded),'last_positive_ground_step':max(x['step'] for x in loaded),'maximum_reported_upward_impulse_ns':max(x['upward_impulse'] for x in loaded),'scope':'endpoint-published contact impulses; not a time-integrated push-off force or gait qualification'}
(A/'ground-contacts.json').write_text(json.dumps(ground,indent=2)+'\n')
(A/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
