"""Reproduce all distance/contact controls and the exact native learning failure from Git."""
from pathlib import Path
import hashlib,importlib.util,json,math,os,stat,struct,subprocess,time

root=Path.cwd();base=root/'artifacts/crow-rl-ccd-sphere-box-20261108'
tmp=Path('/private/tmp/crow-ccd-sphere-box-git-recovery-20261108')
metadata=json.loads((base/'runtime-recovery.json').read_text())
spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def write(name,value):
    with (base/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
write('verification-plan.json',{
 'prediction':'Four query streams reproduce canonical result bytes. Coupled contact controls reproduce physical arrays, metrics and task streams, with owner-recomputed path-bound checkpoint identities. Coupled checkpoint10 resumes the original remaining10-update command and must reproduce exactly the original pair100/lane26/control5/substep14 failure plus all partial model/Adam tensors/metadata and canonical policies/rollout. Success or a different failure contradicts recovery.',
 'budget':'Four query streams120s each, one native contact suite timeout600s, one128-lane remaining10-update160-tick training command timeout300s (expected failure after3 completed updates). No retries or tuning. Temporary raw bound1GiB; no video.',
 'scope':'Same device/OS and external Python/MLX, Git-restored exact executables/library/shaders/sources/assets/curriculum/checkpoint. Reproduction, not new independent behavior evidence.',
 'measurement_scope':'Record timing and allocator metrics separately; compare causal bytes/tensors and scientific fields exactly.'})
env={k:v for k,v in os.environ.items() if not k.startswith('DYLD_')}
env.update(DYLD_PRINT_LIBRARIES='1',METALROBO_LIBRARY=str(tmp/'lib/libmetalrobo.dylib'),
 METALROBO_METALLIB=str(tmp/'shaders/MetalRobo.metallib'),PYTHONPATH=str(tmp/'python'),DYLD_LIBRARY_PATH=str(tmp/'lib'))
def replace(c,flag,value):c[c.index(flag)+1]=str(value)
def run(name,c,timeout,*,library=True,json_result=True):
 write(name+'.command.json',c);start=time.monotonic()
 p=subprocess.run(c,cwd=tmp,env=env,capture_output=True,text=True,timeout=timeout)
 (base/(name+'.stdout')).write_text(p.stdout);(base/(name+'.stderr')).write_text(p.stderr)
 write(name+'.receipt.json',{'returncode':p.returncode,'wall_seconds':time.monotonic()-start,
  'stdout_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(p.stderr.encode()).hexdigest()})
 if p.returncode:raise RuntimeError('Git-restored execution failed: '+name)
 loaded=[s for s in p.stderr.splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
 if library and (not loaded or any(not s.endswith(str(tmp/'lib/libmetalrobo.dylib')) for s in loaded)):raise RuntimeError('Wrong recovery library')
 v=json.loads(p.stdout) if json_result else {}
 if v.get('failed_environment_steps',0):raise RuntimeError('Physics rejected recovery transition')
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
 queries=[]
 for name,record in metadata['distance_references'].items():
  result=tmp/('query-'+name+'.bin')
  run('query-'+name,[str(tmp/record['binary_relative']),str(tmp/record['shader_input']),str(tmp/record['cases_input']),str(result)],120,library=False)
  if a.file_hash(result)!=record['results_sha256']:raise RuntimeError('Recovered query stream differs: '+name)
  queries.append(name)
 run('generic-contact',[str(tmp/'bin/metalrobo_metal_world_contact_probe')],180,json_result=False)
 if 'metal_world_contact=ok ' not in (base/'generic-contact.stdout').read_text():raise RuntimeError('Recovered generic contact suite failed')
 run('contact-controls',[str(tmp/'bin/metalrobo_crow_object_push_check'),str(tmp/'contact')],600,json_result=False)
 contact_summary=json.loads((tmp/'contact/summary.json').read_text())
 if contact_summary['status']!='passed_engineering_controls' or contact_summary['accepted_transitions']!=7296:
  raise RuntimeError('Recovered contact controls failed')
 contacts=[];path_identities={}
 for name,record in metadata['contact_references'].items():
  directory=tmp/'contact'/name;got=json.loads((directory/'summary.json').read_text());want=dict(record['summary'])
  path_identities[name]={}
  for key in ('world_sha256','retina_sha256'):
   path_identities[name][key]={'original':want.pop(key),'restored':got.pop(key)}
  if got!=want:raise RuntimeError('Recovered physical contact metrics/task streams differ: '+name)
  for filename,file in record['files'].items():
   if a.file_hash(directory/filename)!=file['sha256']:raise RuntimeError('Recovered contact physical array differs: '+name+'/'+filename)
  contacts.append(name)
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
 replace(c,'--updates',10)
 c+=['--resume-training-checkpoint',str(tmp/'input/source.trainingcheckpoint')]
 for flag in ('--updated-policy-pack','--deployment-policy-pack','--incumbent-policy-pack',
              '--rollout-pack','--output-learner-state','--checkpoint-directory'):
  replace(c,flag,target/Path(c[c.index(flag)+1]).name)
 for flag,relative in {
  '--python-root':'python','--native-library':'lib/libmetalrobo.dylib','--metallib':'shaders/MetalRobo.metallib',
  '--visual-observation-config':'environments/american-crow/object-push-v1/retinal-observation.json',
  '--object-push-curriculum':'environments/american-crow/object-push-v1/curriculum.json'}.items():replace(c,flag,tmp/relative)
 write('failure-reproduction.command.json',c);start=time.monotonic()
 p=subprocess.run(c,cwd=tmp,env=env,capture_output=True,text=True,timeout=300)
 (base/'failure-reproduction.stdout').write_text(p.stdout);(base/'failure-reproduction.stderr').write_text(p.stderr)
 write('failure-reproduction.receipt.json',{'returncode':p.returncode,'wall_seconds':time.monotonic()-start,
  'stdout_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(p.stderr.encode()).hexdigest()})
 if p.returncode!=1 or p.stdout or p.stderr.splitlines()[-1]!=metadata['training_reference']['error_line']:
  raise RuntimeError('Git-restored native failure differs')
 loaded=[s for s in p.stderr.splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
 if not loaded or any(not s.endswith(str(tmp/'lib/libmetalrobo.dylib')) for s in loaded):raise RuntimeError('Wrong failure reproduction library')
 if list((target/'checkpoints').glob('*')):raise RuntimeError('Recovered failure boundary changed')
 import mlx.core as mx
 import numpy as np
 tensors=0;canonical=[]
 for name,record in metadata['training_reference']['files'].items():
  path=target/name
  if name=='learner.safetensors':
   x,xm=mx.load(path,return_metadata=True);y,ym=mx.load(tmp/'reference/training'/name,return_metadata=True)
   if xm!=ym or x.keys()!=y.keys():raise RuntimeError('Partial model/Adam metadata differs')
   for key in x:
    if np.asarray(x[key]).tobytes()!=np.asarray(y[key]).tobytes():raise RuntimeError('Partial model/Adam array differs: '+key)
    tensors+=1
  else:
   if a.file_hash(path)!=record['sha256']:raise RuntimeError('Partial canonical artifact differs: '+name)
   canonical.append(name)
 results={'generic_contact_suite_passed':True,'query_streams_exact':queries,
  'contact_arms_physical_arrays_metrics_task_streams_exact':contacts,
  'contact_path_bound_native_checkpoint_identities':path_identities,
  'native_failure_line_exact':True,'error_line':metadata['training_reference']['error_line'],
  'canonical_partial_artifacts_exact':canonical,'partial_model_Adam_arrays_exact':tensors,
  'partial_model_Adam_metadata_exact':True,'new_coupled_checkpoints':0,'behavioral_success':False}
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
 if total>1024**3:raise RuntimeError('Recovery storage bound exceeded')
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
 print(json.dumps({'status':'passed','failure_reproduced_exact':True,'query_streams':len(queries),'contact_arms':len(contacts),'model_Adam_arrays':tensors,'temporary_bytes_removed':total}))
except Exception as exc:
 write('verification-failure.json',{'error':str(exc),'type':type(exc).__name__})
 raise
