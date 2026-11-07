from pathlib import Path
import hashlib,importlib.util,json,os,stat,struct,subprocess
import mlx.core as mx
import numpy as np
root=Path.cwd();base=root/'artifacts/crow-rl-action-holds-20261107';tmp=Path('/private/tmp/crow-action-holds-git-recovery-20261107')
spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
m=json.loads((base/'runtime-recovery.json').read_text());plan={'budget':'One600second standard training update and one180second mid-hold rollout, no retries','prediction':'Git-restored native/Python/product inputs reproduce policy/candidate/rollout bytes,53 learner/Adam tensors and metadata, and all three native state sections; mid-hold tail rollout and all native sections exact. Adjacent Git-restored library must actually load.','scope':'Matching device/OS/dependencies; integrated training and life recovery only, no learned memory claim'}
if not (base/'verification-plan.json').exists():
 with (base/'verification-plan.json').open('x') as f:f.write(json.dumps(plan,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def native(p):
 data=p.read_bytes();offset=104;parts=[]
 if data[:8]!=b'NUMILF01':raise RuntimeError('Wrong native envelope')
 for _ in range(3):
  size,=struct.unpack_from('<Q',data,offset);offset+=8;parts.append(hashlib.sha256(data[offset:offset+size]).hexdigest());offset+=size
 if offset!=len(data):raise RuntimeError('Wrong native section lengths')
 return parts
env={k:v for k,v in os.environ.items() if not k.startswith('DYLD_')};env['DYLD_PRINT_LIBRARIES']='1'
def run(name,c,timeout):
 command_file=base/(name+'.command.json');reuse=command_file.exists()
 if reuse:
  if json.loads(command_file.read_text())!=c:raise RuntimeError('Existing command differs; preserve failure')
  stdout=(base/(name+'.stdout')).read_text();stderr=(base/(name+'.stderr')).read_text()
 else:
  command_file.write_text(json.dumps(c,indent=2)+'\n');p=subprocess.run(c,capture_output=True,text=True,env=env,timeout=timeout)
  stdout,stderr=p.stdout,p.stderr
  (base/(name+'.stdout')).write_text(stdout);(base/(name+'.stderr')).write_text(stderr)
  if p.returncode:raise RuntimeError(name+' failed')
 loaded=[line for line in stderr.splitlines() if line.startswith('dyld[') and 'libmetalrobo.dylib' in line]
 if not loaded or any(not line.endswith(str(tmp/'lib/libmetalrobo.dylib')) for line in loaded):raise RuntimeError(name+' wrong library loaded')
 summary=json.loads(stdout)
 if name=='causal-prefix-recovery':
  if summary.get('status')!='passed' or summary.get('accepted_ticks')!=64 or summary.get('lanes')!=8:raise RuntimeError('Native causal prefix rejected')
 elif summary.get('failed_environment_steps')!=0:raise RuntimeError('Physics rejected')
 return {'summary':summary,'loaded_library':loaded,'library_override':False,'reused_completed_process_output':reuse}
try:
 c=m['training_command'][:];c[0]=str(tmp/'bin/metalrobo_task_train');dest=tmp/'replay';dest.mkdir(exist_ok=True)
 paths={'--metallib':tmp/'shaders/MetalRobo.metallib','--native-library':tmp/'lib/libmetalrobo.dylib','--python-root':tmp/'python','--visual-observation-config':tmp/'project/environments/american-crow/action-holds-v1/retinal-observation.json','--resume-training-checkpoint':tmp/'input/checkpoint','--checkpoint-directory':dest/'checkpoints','--updated-policy-pack':dest/'updated.policypack','--deployment-policy-pack':dest/'deployment.policypack','--incumbent-policy-pack':dest/'incumbent.policypack','--rollout-pack':dest/'training.rolloutpack','--output-learner-state':dest/'learner.safetensors'}
 for flag,path in paths.items():c[c.index(flag)+1]=str(path)
 training=run('training-recovery',c,600);ref=dest/'checkpoints/revision-00000003.trainingcheckpoint'
 for name,record in m['training_reference'].items():
  if sha(ref/name)!=record['sha256']:raise RuntimeError('Recovered training artifact differs: '+name)
 tensors,meta=mx.load(ref/'learner.safetensors',return_metadata=True)
 if meta!=m['learner_reference']['metadata'] or set(tensors)!=set(m['learner_reference']['tensors']):raise RuntimeError('Learner metadata/topology differs')
 for key,value in tensors.items():
  x=np.asarray(value);rec=m['learner_reference']['tensors'][key]
  if str(x.dtype)!=rec['dtype'] or list(x.shape)!=rec['shape'] or hashlib.sha256(x.tobytes()).hexdigest()!=rec['sha256']:raise RuntimeError('Learner tensor differs: '+key)
 if native(ref/'life.native')!=m['training_native_sections']:raise RuntimeError('Native training future differs')
 c=m['early_command'][:];c[0]=str(tmp/'bin/metalrobo_task_rollout')
 for flag,path in {'--metallib':tmp/'shaders/MetalRobo.metallib','--visual-observation-config':tmp/'project/environments/american-crow/action-holds-v1/retinal-observation.json','--policy-pack':tmp/'input/initial.policypack','--resume-life-checkpoint':tmp/'input/early.lifecheckpoint','--rollout-pack':tmp/'early.rolloutpack','--save-life-checkpoint':tmp/'early.lifecheckpoint'}.items():c[c.index(flag)+1]=str(path)
 early=run('early-recovery',c,180)
 if sha(tmp/'early.rolloutpack')!=m['early_reference']['sha256'] or native(tmp/'early.lifecheckpoint')!=m['early_native_sections']:raise RuntimeError('Early-cue native future differs')
 files=[];dirs=[]
 for directory,subdirs,names in os.walk(tmp,followlinks=False):
  for name in subdirs:
   p=Path(directory)/name
   if p.is_symlink():raise RuntimeError('Unexpected recovery symlink')
   dirs.append(p)
  for name in names:
   p=Path(directory)/name;s=p.lstat()
   if not stat.S_ISREG(s.st_mode):raise RuntimeError('Unexpected recovery file type')
   files.append({'path':str(p),'bytes':s.st_size,'mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns,'sha256':sha(p)})
 handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(tmp)],capture_output=True,text=True,timeout=60)
 if handles.returncode not in (0,1) or handles.stdout or handles.stderr:raise RuntimeError('Open recovery handles')
 for r in files:
  p=Path(r['path']);s=p.lstat()
  if s.st_size!=r['bytes'] or stat.S_IMODE(s.st_mode)!=r['mode'] or s.st_mtime_ns!=r['mtime_ns'] or sha(p)!=r['sha256']:raise RuntimeError('Recovery changed during cleanup')
 report={'status':'passed','archive_commit':a.git('rev-parse','HEAD').decode().strip(),'plan':plan,'training':training,'early_cue':early,'learner_tensors_exact':len(tensors),'native_training_sections':m['training_native_sections'],'native_early_sections':m['early_native_sections'],'restoration':json.loads((tmp/'runtime-restoration.json').read_text()),'temporary_cleanup':{'status':'validated','files':files,'open_handles':0,'df_before':subprocess.check_output(['df','-k','.'],text=True)}}
 (base/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 for r in files:Path(r['path']).unlink()
 for p in sorted(dirs,key=lambda p:len(p.parts),reverse=True):p.rmdir()
 tmp.rmdir();report['temporary_cleanup'].update(status='removed',df_after=subprocess.check_output(['df','-k','.'],text=True));(base/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':'passed','learner_tensors_exact':len(tensors),'native_components_exact':6,'temporary_files_removed':len(files),'temporary_bytes_removed':sum(r['bytes'] for r in files)}))
except Exception as error:
 (base/'verification-failure.json').write_text(json.dumps({'error':str(error),'type':type(error).__name__,'plan':plan},indent=2)+'\n');raise
