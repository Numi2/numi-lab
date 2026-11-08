"""Recover the frozen evaluations and last half of a native learning trial."""
from pathlib import Path
import hashlib,importlib.util,json,math,os,stat,struct,subprocess,time

root=Path.cwd();base=root/'artifacts/crow-rl-object-learning-20261108'
tmp=Path('/private/tmp/crow-object-learning-git-recovery-20261108')
metadata=json.loads((base/'runtime-recovery.json').read_text())
spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def write(name,value):
    with (base/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
write('verification-plan.json',{
 'prediction':'Six restored frozen native evaluations reproduce rollout bytes and all accepted body/q/v endpoint fields. Coupled checkpoint8 resumes8 updates and reproduces checkpoint16 model/Adam tensors/metadata, canonical policies/rollout and native components, plus scientific learner metrics.',
 'budget':'Six32-lane160-tick evaluations180s each and one128-lane8-update160-tick training continuation600s. No retries or tuning. Temporary raw bound768MiB; no video.',
 'scope':'Same device/OS and external Python/MLX, Git-restored exact executables/library/shaders/sources/assets/curriculum/checkpoint. Reproduction, not new independent behavior evidence.',
 'measurement_scope':'Record timing and allocator metrics separately; compare causal bytes/tensors and scientific fields exactly.'})
env={k:v for k,v in os.environ.items() if not k.startswith('DYLD_')}
env.update(DYLD_PRINT_LIBRARIES='1',METALROBO_LIBRARY=str(tmp/'lib/libmetalrobo.dylib'),
 METALROBO_METALLIB=str(tmp/'shaders/MetalRobo.metallib'),PYTHONPATH=str(tmp/'python'))
def replace(c,flag,value):c[c.index(flag)+1]=str(value)
def run(name,c,timeout):
 write(name+'.command.json',c);start=time.monotonic()
 p=subprocess.run(c,cwd=tmp,env=env,capture_output=True,text=True,timeout=timeout)
 (base/(name+'.stdout')).write_text(p.stdout);(base/(name+'.stderr')).write_text(p.stderr)
 write(name+'.receipt.json',{'returncode':p.returncode,'wall_seconds':time.monotonic()-start,
  'stdout_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(p.stderr.encode()).hexdigest()})
 if p.returncode:raise RuntimeError('Git-restored execution failed: '+name)
 loaded=[s for s in p.stderr.splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
 if not loaded or any(not s.endswith(str(tmp/'lib/libmetalrobo.dylib')) for s in loaded):raise RuntimeError('Wrong recovery library')
 v=json.loads(p.stdout)
 if v['failed_environment_steps']:raise RuntimeError('Physics rejected recovery transition')
 return v
def parts(path):
 data=path.read_bytes();tag,abi,reserved,identity,size,digest=struct.unpack_from('<8sIIQQ32s',data)
 if tag!=b'NUMILF01' or abi!=1 or reserved or size!=len(data)-64 or hashlib.sha256(data[64:]).digest()!=digest:
  raise RuntimeError('Invalid native life envelope')
 offset=104;values=[]
 for _ in range(3):
  length,=struct.unpack_from('<Q',data,offset);offset+=8
  values.append(data[offset:offset+length]);offset+=length
 if offset!=len(data) or values[2]:raise RuntimeError('Wrong object-life contract')
 return values
try:
 restoration=json.loads((tmp/'runtime-restoration.json').read_text())
 for f in restoration['files']:
  if a.file_hash(tmp/f['path'])!=f['sha256']:raise RuntimeError('Changed restored Git input')
 evaluations=[]
 for name,record in metadata['evaluation_references'].items():
  c=list(record['command']);c[2]=str(tmp/'bin/metalrobo_task_rollout')
  for flag,relative in {
   '--policy-pack':record['policy_input'],
   '--metallib':'shaders/MetalRobo.metallib',
   '--visual-observation-config':'environments/american-crow/object-push-v1/retinal-observation.json',
   '--object-push-curriculum':'environments/american-crow/object-push-v1/curriculum.json',
   '--save-life-checkpoint':name+'.life','--rollout-pack':name+'.rolloutpack',
   '--final-state-json':name+'.state.json'}.items():replace(c,flag,tmp/relative)
  v=run('evaluation-'+name,c,180)
  if v['accepted_lifetime_control_ticks_by_environment']!=[160]*32:raise RuntimeError('Wrong recovered endpoint clock')
  if a.file_hash(tmp/(name+'.rolloutpack'))!=record['rollout_sha256']:raise RuntimeError('Recovered evaluation rollout differs')
  got=json.loads((tmp/(name+'.state.json')).read_text())
  want=json.loads((tmp/'reference/evaluation'/(name+'.state.json')).read_text())
  if got!=want:raise RuntimeError('Recovered accepted endpoint fields differ: '+name)
  evaluations.append(name)
 target=tmp/'training';target.mkdir()
 c=list(metadata['training_reference']['command']);c[2]=str(tmp/'bin/metalrobo_task_train')
 i=c.index('--policy-pack');del c[i:i+2]
 replace(c,'--updates',8)
 c+=['--resume-training-checkpoint',str(tmp/'input/source.trainingcheckpoint')]
 for flag in ('--updated-policy-pack','--deployment-policy-pack','--incumbent-policy-pack',
              '--rollout-pack','--output-learner-state','--checkpoint-directory'):
  replace(c,flag,target/Path(c[c.index(flag)+1]).name)
 for flag,relative in {
  '--python-root':'python','--native-library':'lib/libmetalrobo.dylib','--metallib':'shaders/MetalRobo.metallib',
  '--visual-observation-config':'environments/american-crow/object-push-v1/retinal-observation.json',
  '--object-push-curriculum':'environments/american-crow/object-push-v1/curriculum.json'}.items():replace(c,flag,tmp/relative)
 v=run('training-recovery',c,600)
 if v['training_samples']!=163840 or v['accepted_lifetime_control_ticks_by_environment']!=[2560]*128 or not v['life_state_restored']:
  raise RuntimeError('Recovered learner clock/sample/restore differs')
 ignored={'update_seconds','mlx_peak_memory_bytes','mlx_active_memory_bytes','mlx_cache_memory_bytes'}
 measurements=[]
 for got,want in zip(v['learning_update_history'],metadata['training_reference']['summary']['learning_update_history'][8:],strict=True):
  got=got.copy();want=want.copy();present=ignored & got.keys()
  if present!=ignored & want.keys() or not {'update_seconds','mlx_peak_memory_bytes'}<=present:raise RuntimeError('Process measurement schema differs')
  measurements.append({'original':{k:want[k] for k in present},'recovered':{k:got[k] for k in present}})
  for k in present:
   if not math.isfinite(got[k]) or got[k]<0:raise RuntimeError('Invalid process measurement')
   got.pop(k);want.pop(k)
  if got!=want:raise RuntimeError('Recovered scientific learner metrics differ')
 import mlx.core as mx
 import numpy as np
 tensors=0;canonical=[];native=0
 for name,record in metadata['training_reference']['files'].items():
  p=target/name;q=tmp/'reference/training'/name
  if p.name=='learner.safetensors':
   x,xm=mx.load(p,return_metadata=True);y,ym=mx.load(q,return_metadata=True)
   if xm!=ym or x.keys()!=y.keys():raise RuntimeError('Recovered model/Adam metadata differs')
   for k in x:
    if np.asarray(x[k]).tobytes()!=np.asarray(y[k]).tobytes():raise RuntimeError('Recovered model/Adam array differs: '+k)
    tensors+=1
  elif p.name=='life.native':
   if parts(p)!=parts(q):raise RuntimeError('Recovered native state differs')
   native+=1
  elif p.suffix in ('.policypack','.rolloutpack'):
   if a.file_hash(p)!=record['sha256']:raise RuntimeError('Recovered canonical bytes differ')
   canonical.append(name)
 results={'evaluation_rollouts_and_all_endpoint_fields_exact':evaluations,'model_Adam_arrays_exact':tensors,
  'model_Adam_metadata_exact':True,'canonical_training_files_exact':canonical,'native_boundaries_exact':native,
  'scientific_learner_metrics_exact':True,'process_measurements':measurements}
 files=[];directories=[]
 for directory,subdirs,names in os.walk(tmp,followlinks=False):
  for name in subdirs:
   p=Path(directory)/name
   if p.is_symlink():raise RuntimeError('Unexpected recovery directory symlink')
   directories.append(p)
  for name in names:
   p=Path(directory)/name;s=p.lstat()
   if not stat.S_ISREG(s.st_mode):raise RuntimeError('Unexpected recovery file type')
   files.append({'path':str(p),'bytes':s.st_size,'mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns,'sha256':a.file_hash(p)})
 total=sum(f['bytes'] for f in files)
 if total>768*1024**2:raise RuntimeError('Recovery storage bound exceeded')
 h=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(tmp)],capture_output=True,text=True,timeout=60)
 if h.returncode not in (0,1) or h.stdout or h.stderr:raise RuntimeError('Open recovery handles')
 for f in files:
  p=Path(f['path']);s=p.lstat()
  if s.st_size!=f['bytes'] or s.st_mtime_ns!=f['mtime_ns'] or stat.S_IMODE(s.st_mode)!=f['mode'] or a.file_hash(p)!=f['sha256']:
   raise RuntimeError('Recovery file changed before cleanup')
 report={'status':'passed','archive_commit':a.git('rev-parse','HEAD').decode().strip(),'results':results,'restoration':restoration,
  'temporary_cleanup':{'status':'validated','files':files,'open_handles':0,'removed_logical_bytes':total}}
 write('verification-before-cleanup.json',report)
 for f in files:Path(f['path']).unlink()
 for p in sorted(directories,key=lambda p:len(p.parts),reverse=True):p.rmdir()
 tmp.rmdir();report['temporary_cleanup']['status']='removed';write('verification.json',report)
 print(json.dumps({'status':'passed','evaluation_streams':len(evaluations),'model_Adam_arrays':tensors,'temporary_bytes_removed':total}))
except Exception as exc:
 write('verification-failure.json',{'error':str(exc),'type':type(exc).__name__})
 raise
