from pathlib import Path
import gzip,hashlib,json,os,subprocess,sys,threading
R=Path.cwd();A=R/'artifacts/crow-v15-moderate-full-journey-dev-20261112';B=R/'build/crow-v15-moderate-full-journey-dev-20261112';M=R/'build/crow-v15-matched-binary-gait-screen-20261112';J=R/'artifacts/crow-v15-hip-roll-journey-integration-20261111';L=R/'build/crow-v15-stance-contact-velocity-audit-20261112';name=sys.argv[1];assert name in ('sham','candidate');seed=int(sys.argv[2]) if len(sys.argv)>2 else 20261126;assert seed in (20261126,20261127);tag=sys.argv[3] if len(sys.argv)>3 else '';substeps=int(sys.argv[4]) if len(sys.argv)>4 else 16;assert substeps == 16, "release threshold is calibrated only for this 16-substep diagnostic";assert all(c.isalnum() or c=='-' for c in tag);D=A/(name+'-'+str(seed)+(('-'+tag) if tag else ''));D.mkdir(exist_ok=False);cmd=json.loads((J/'rebound-command.json').read_text());
for k,v in [('--seed',str(seed)),('--steps','5000'),('--physics-substeps',str(substeps)),('--metallib',str(B/str(R/'build/crow-v15-moderate-walk-flight-integration-20261112/flexed.metallib'))),('--state-trace',str(B/(D.name+'-trace.tsv'))),('--crow-replay-pack',str(B/(D.name+'-replay.json')))]:cmd[cmd.index(k)+1]=v
if substeps==32:
 for option,label in (('--policy-pack','landing'),('--base-policy-pack','route')):
  candidate=B/('phys32-'+label+'.policypack')
  if candidate.exists():cmd[cmd.index(option)+1]=str(candidate)
cmd[cmd.index('--base-policy-pack')+1]=str(R/'build/crow-v15-localized-route-student-imitation-20261111/route-imitation.policypack')
cmd[cmd.index("--state-trace-environment")+1]="5"  # retain the prior release lane
cmd.remove('--birdflow-journey-teacher')
idx=cmd.index('--birdflow-journey-student-authority'); del cmd[idx:idx+2]
if name=='candidate' and tag.startswith('rebound'):
 for option,label in (('--policy-pack','landing'),('--base-policy-pack','route')):
  cmd[cmd.index(option)+1]=str(M/(label+'-candidate.policypack'))
cmd[0]=str(R/'build/crow-v15-replay-writer-roundtrip-20261111/metalrobo_task_rollout')
(D/'command.json').write_text(json.dumps(cmd,indent=2)+'\n');h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(D/'inputs.json').write_text(json.dumps({'host_sha256':h(L/'libmetalrobo.dylib'),'metallib_sha256':h(Path(cmd[cmd.index('--metallib')+1])),'frozen_flight_sha256':h(R/'artifacts/crow-v15-flight-feedback-freeze-20261110/CrowFlightControllerResearchV2.metal'),'landing_policy_sha256':h(Path(cmd[cmd.index('--policy-pack')+1])),'route_policy_sha256':h(Path(cmd[cmd.index('--base-policy-pack')+1]))},indent=2)+'\n');env={**os.environ,'DYLD_LIBRARY_PATH':str(L),'CROW_SUBSTEP_TRACE':'1','CROW_CONTACT_TRACE':'1','CROW_SUPPORT_TRACE':'1','CROW_EARLY_FLIGHT_TRACE':'1','PYTHONDONTWRITEBYTECODE':'1'};env.pop('CROW_EXPERIMENT_GAIT_SMOOTH',None);env.update({'CROW_EXPERIMENT_GAIT_SMOOTH':'1'} if name=='candidate' else {});p=subprocess.Popen(cmd,cwd=R,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
class SegmentedWriter:
 def __init__(self,base,limit=32*1024*1024):
  self.base=base;self.limit=limit;self.parts=[];self.total=hashlib.sha256();self.count=0;self.f=None;self.part_hash=None;self.part_size=0
 def _open(self):
  name=self.base.name+f'.part{len(self.parts):03d}';self.path=self.base.parent/name;self.f=self.path.open('xb');self.part_hash=hashlib.sha256();self.part_size=0
 def _finish(self):
  self.f.close();self.parts.append({'name':self.path.name,'bytes':self.part_size,'sha256':self.part_hash.hexdigest()});self.f=None
 def write(self,data):
  data=memoryview(data);size=len(data);self.total.update(data);self.count+=size
  while data:
   if self.f is None:self._open()
   n=min(len(data),self.limit-self.part_size);self.f.write(data[:n]);self.part_hash.update(data[:n]);self.part_size+=n;data=data[n:]
   if self.part_size==self.limit:self._finish()
  return size
 def flush(self):
  if self.f:self.f.flush()
 def close(self):
  if self.f:self._finish()
  (self.base.parent/'stderr-parts.json').write_text(json.dumps({'bytes':self.count,'compressed_sha256':self.total.hexdigest(),'parts':self.parts},indent=2)+'\n')
def pump_impl(src,path,compress):
 if compress:
  raw=SegmentedWriter(path)
  with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
   for b in iter(lambda:src.read(1<<16),b''):z.write(b)
  raw.close()
 else:
  with path.open('wb') as raw:
   for b in iter(lambda:src.read(1<<16),b''):raw.write(b)
pump_errors=[]
def pump(src,path,compress):
 try:pump_impl(src,path,compress)
 except BaseException as error:pump_errors.append(repr(error))
ts=[threading.Thread(target=pump,args=(p.stdout,D/'run.stdout',False)),threading.Thread(target=pump,args=(p.stderr,D/'run.stderr.gz',True))]
for t in ts:t.start()
rc=p.wait()
for t in ts:t.join()
if pump_errors:raise RuntimeError('output capture failed: '+repr(pump_errors))
for kind in ('trace.tsv','replay.json'):
 path=B/(D.name+'-'+kind)
 if path.exists():
  with path.open('rb') as f,(D/(kind+'.gz')).open('wb') as raw:
   with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as z:
    for b in iter(lambda:f.read(1<<16),b''):z.write(b)
(D/'run-result.json').write_text(json.dumps({'exit_code':rc},indent=2)+'\n');print(name,'exit',rc,flush=True)

if rc != 0: raise SystemExit(rc)
