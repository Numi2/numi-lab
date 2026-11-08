"""Reproduce the archived native failure, then remove only verified temporary copies."""
from pathlib import Path
import hashlib,importlib.util,json,os,stat,subprocess,time
root=Path.cwd();base=root/'artifacts/crow-rl-object-continuation-failure-20261108'
tmp=Path('/private/tmp/crow-object-continuation-failure-git-recovery-20261108')
metadata=json.loads((base/'runtime-recovery.json').read_text())
spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def write(name,value):
 with (base/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
write('verification-plan.json',{
 'prediction':'One exact Git-restored coupled checkpoint16 continuation reproduces the same declared error line (lane111, local control6, substep31, stable key106,111), final partial canonical policies/rollout and all model/Adam arrays and metadata.',
 'validity':'The exact scientific failure must recur; a successful or different failing run is not recovery success. Verify actual loaded native library and every restored input hash before execution.',
 'budget':'One original16-update command, expected to fail after3 completed updates; timeout240s. No retry/tuning. Temporary raw<=384MiB. No video.',
 'scope':'Same host/device/OS/Python/MLX. Failure reproduction, not a successful continuation or independent behavior result.'})
try:
 restoration=json.loads((tmp/'runtime-restoration.json').read_text())
 for f in restoration['files']:
  if a.file_hash(tmp/f['path'])!=f['sha256']:raise RuntimeError('Changed restored Git input')
 env={k:v for k,v in os.environ.items() if not k.startswith('DYLD_')}
 env.update(DYLD_PRINT_LIBRARIES='1',METALROBO_LIBRARY=str(tmp/'lib/libmetalrobo.dylib'),
            METALROBO_METALLIB=str(tmp/'shaders/MetalRobo.metallib'),PYTHONPATH=str(tmp/'python'))
 target=tmp/'training';target.mkdir()
 c=list(metadata['training_reference']['command']);c[2]=str(tmp/'bin/metalrobo_task_train')
 for flag in ('--updated-policy-pack','--deployment-policy-pack','--incumbent-policy-pack',
              '--rollout-pack','--output-learner-state','--checkpoint-directory'):
  c[c.index(flag)+1]=str(target/Path(c[c.index(flag)+1]).name)
 for flag,relative in {'--resume-training-checkpoint':'input/source.trainingcheckpoint','--python-root':'python',
  '--native-library':'lib/libmetalrobo.dylib','--metallib':'shaders/MetalRobo.metallib',
  '--visual-observation-config':'environments/american-crow/object-push-v1/retinal-observation.json',
  '--object-push-curriculum':'environments/american-crow/object-push-v1/curriculum.json'}.items():c[c.index(flag)+1]=str(tmp/relative)
 write('failure-reproduction.command.json',c);start=time.monotonic()
 p=subprocess.run(c,cwd=tmp,env=env,capture_output=True,text=True,timeout=240)
 (base/'failure-reproduction.stdout').write_text(p.stdout);(base/'failure-reproduction.stderr').write_text(p.stderr)
 write('failure-reproduction.receipt.json',{'returncode':p.returncode,'wall_seconds':time.monotonic()-start,
  'stdout_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(p.stderr.encode()).hexdigest()})
 if p.returncode!=1 or p.stdout or p.stderr.splitlines()[-1]!=metadata['training_reference']['error_line']:
  raise RuntimeError('Git-restored native failure differs')
 loaded=[s for s in p.stderr.splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
 if not loaded or any(not s.endswith(str(tmp/'lib/libmetalrobo.dylib')) for s in loaded):raise RuntimeError('Wrong recovery library')
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
 files=[];directories=[]
 for directory,subdirs,names in os.walk(tmp,followlinks=False):
  for name in subdirs:
   path=Path(directory)/name
   if path.is_symlink():raise RuntimeError('Unexpected recovery symlink')
   directories.append(path)
  for name in names:
   path=Path(directory)/name;s=path.lstat()
   if not stat.S_ISREG(s.st_mode):raise RuntimeError('Unexpected recovery type')
   files.append({'path':str(path),'bytes':s.st_size,'mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns,'sha256':a.file_hash(path)})
 total=sum(f['bytes'] for f in files)
 if total>384*1024**2:raise RuntimeError('Recovery temporary raw bound exceeded')
 h=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(tmp)],capture_output=True,text=True,timeout=60)
 if h.returncode not in (0,1) or h.stdout or h.stderr:raise RuntimeError('Open recovery handles')
 for f in files:
  path=Path(f['path']);s=path.lstat()
  if s.st_size!=f['bytes'] or s.st_mtime_ns!=f['mtime_ns'] or stat.S_IMODE(s.st_mode)!=f['mode'] or a.file_hash(path)!=f['sha256']:
   raise RuntimeError('Recovery copy changed before cleanup')
 report={'status':'passed','archive_commit':a.git('rev-parse','HEAD').decode().strip(),'restoration':restoration,
  'results':{'native_failure_line_exact':True,'error_line':metadata['training_reference']['error_line'],
   'canonical_partial_artifacts_exact':canonical,'partial_model_Adam_arrays_exact':tensors,'partial_model_Adam_metadata_exact':True,
   'new_coupled_checkpoints':0,'behavioral_success':False},
  'temporary_cleanup':{'status':'validated','files':files,'open_handles':0,'removed_logical_bytes':total}}
 write('verification-before-cleanup.json',report)
 for f in files:Path(f['path']).unlink()
 for path in sorted(directories,key=lambda x:len(x.parts),reverse=True):path.rmdir()
 tmp.rmdir();report['temporary_cleanup']['status']='removed';write('verification.json',report)
 print(json.dumps({'status':'passed','failure_reproduced_exact':True,'model_Adam_arrays':tensors,'temporary_bytes_removed':total}))
except Exception as exc:
 write('verification-failure.json',{'type':type(exc).__name__,'error':str(exc)})
 raise
