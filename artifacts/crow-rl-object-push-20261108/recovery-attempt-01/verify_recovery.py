from pathlib import Path
import importlib.util,json,os,stat,subprocess
r=Path.cwd();base=r/'artifacts/crow-rl-object-push-20261108';tmp=Path('/private/tmp/crow-object-push-git-recovery-20261108')
spec=importlib.util.spec_from_file_location('archive',r/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
m=json.loads((base/'runtime-recovery.json').read_text());plan={'budget':'One complete7296-transition native replay and two ordinary160-tick128-lane fixed-rate learner updates; no tuning or retries.', 'prediction':'Native contact/sensor/reward/partition/continuation files reproduce from actual Git-restored runtime, with native-owner explanations for path-bound visual identities only. Fresh ordinary trainer reproduces policy/rollout bytes and all model/Adam arrays plus finite scientific metrics.', 'scope':'Estimated fixed object-push apparatus; no learned manipulation or production performance claim.'}
(base/'verification-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith('DYLD_')};env['DYLD_PRINT_LIBRARIES']='1';env['METALROBO_LIBRARY']=str(tmp/'lib/libmetalrobo.dylib');env['METALROBO_METALLIB']=str(tmp/'shaders/MetalRobo.metallib');env['PYTHONPATH']=str(tmp/'python')
def run(name,c,timeout):
 (base/(name+'.command.json')).write_text(json.dumps(c,indent=2)+'\n')
 p=subprocess.run(c,capture_output=True,text=True,env=env,timeout=timeout,cwd=tmp)
 (base/(name+'.stdout')).write_text(p.stdout);(base/(name+'.stderr')).write_text(p.stderr);(base/(name+'.receipt.json')).write_text(json.dumps({'returncode':p.returncode},indent=2)+'\n')
 if p.returncode:raise RuntimeError('Recovered native execution failed: '+name)
 loaded=[line for line in p.stderr.splitlines() if line.startswith('dyld[') and 'libmetalrobo.dylib' in line]
 if not loaded or any(not line.endswith(str(tmp/'lib/libmetalrobo.dylib')) for line in loaded):raise RuntimeError('Wrong recovery library: '+name)
 return {'stdout':p.stdout,'loaded_library':loaded}
try:
 compiler=['clang++','-std=c++20','-fobjc-arc','-I'+str(tmp/'source/include'),str(tmp/'source/identity_oracle.mm'),'-L'+str(tmp/'lib'),'-lmetalrobo','-framework','Foundation','-Wl,-rpath,@loader_path/../lib','-o',str(tmp/'bin/identity_oracle')]
 built=subprocess.run(compiler,capture_output=True,text=True,timeout=120)
 (base/'oracle-build.json').write_text(json.dumps({'command':compiler,'returncode':built.returncode,'stdout':built.stdout,'stderr':built.stderr},indent=2)+'\n')
 if built.returncode:raise RuntimeError('Native fingerprint oracle compilation failed')
 dest=tmp/'native-replay';ref=m['native_reference']
 execution=run('native-recovery',[str(tmp/'bin/metalrobo_crow_object_push_check'),str(dest)],600)
 oracle=run('native-identity',[str(tmp/'bin/identity_oracle'),ref['original_root'],str(dest),str(tmp/'oracle-assets')],120)
 identities=json.loads(oracle['stdout']);entries=a.tree();manifest=json.loads((base/'manifest.json').read_text());exact=0;relocations=[]
 with a.Objects() as objects:
  for name,record in ref['files'].items():
   if name.endswith('/visual-scene.json'):
    f=next(f for f in manifest['files'] if f['sha256']==record['sha256'] and f['bytes']==record['bytes'])
    original=json.loads(b''.join(a.content(f,entries,objects)));restored=json.loads((dest/name).read_text());pair=identities[name.split('/')[0]]
    for value,location in ((original,'original'),(restored,'restored')):
     if value['fingerprint']!=pair[location]['manifest'] or value['render_scene']['fingerprint']!=pair[location]['render_scene']:raise RuntimeError('Native owner failed to explain relocated visual identity: '+name)
    del original['fingerprint'];del original['render_scene']['fingerprint'];del restored['fingerprint'];del restored['render_scene']['fingerprint']
    if original!=restored:raise RuntimeError('Other visual fields differ: '+name)
    relocations.append({'path':name,'native_owner_identities':pair,'all_other_JSON_fields_exact':True})
   else:
    if a.file_hash(dest/name)!=record['sha256']:raise RuntimeError('Recovered native evidence differs: '+name)
    exact+=1
  if {str(p.relative_to(dest)) for p in dest.rglob('*')if p.is_file()}!=set(ref['files']):raise RuntimeError('Recovered native file set differs')
 target=tmp/'learning-replay';target.mkdir()
 c=list(m['learning_reference']['command']);c[0]=str(tmp/'bin/metalrobo_task_train')
 for option in ('--policy-pack','--updated-policy-pack','--deployment-policy-pack','--incumbent-policy-pack','--rollout-pack','--output-learner-state','--checkpoint-directory'):
  i=c.index(option)+1;c[i]=str(target/Path(c[i]).name)
 for option,path in (('--visual-observation-config',tmp/'project/environments/american-crow/object-push-v1/retinal-observation.json'),('--python-root',tmp/'python'),('--native-library',tmp/'lib/libmetalrobo.dylib'),('--metallib',tmp/'shaders/MetalRobo.metallib')):
  c[c.index(option)+1]=str(path)
 learning=run('learning-recovery',c,300);v=json.loads(learning['stdout']);expected=m['learning_reference']['summary']
 if v['failed_environment_steps']!=0 or v['training_samples']!=40960:raise RuntimeError('Recovered learning failed native sample/status checks')
 for got,want in zip(v['learning_update_history'],expected['learning_update_history'],strict=True):
  got=got.copy();want=want.copy()
  for key in ('update_seconds',):got.pop(key);want.pop(key)
  if got!=want:raise RuntimeError('Recovered scientific learner metrics differ')
 import mlx.core as mx
 import numpy as np
 tensors=0;binary_exact=[];metadata_exact=[]
 for name,record in m['learning_reference']['files'].items():
  if not name.endswith(('.policypack','.rolloutpack','.safetensors')):continue
  p=target/name;q=tmp/'reference-learning'/name
  if name.endswith('.safetensors'):
   g,gm=mx.load(str(p),return_metadata=True);w,wm=mx.load(str(q),return_metadata=True)
   if gm!=wm or set(g)!=set(w):raise RuntimeError('Recovered learner keys/metadata differ: '+name)
   for key in g:
    if not np.array_equal(np.array(g[key]),np.array(w[key])):raise RuntimeError('Recovered learner/Adam tensor differs: '+name+'/'+key)
    tensors+=1
   metadata_exact.append(name)
  else:
   if a.file_hash(p)!=record['sha256']:raise RuntimeError('Recovered policy/rollout bytes differ: '+name)
   binary_exact.append(name)
 results={'native':{'execution':execution,'identity_oracle':oracle,'byte_exact_reference_files':exact,'relocated_manifests_verified':relocations},'learning':{'execution':learning,'policy_rollout_bytes_exact':binary_exact,'learner_metadata_exact':metadata_exact,'model_Adam_tensors_exact':tensors,'scientific_metrics_exact':True}}
 files=[];dirs=[]
 for directory,subdirs,names in os.walk(tmp,followlinks=False):
  for name in subdirs:
   p=Path(directory)/name
   if p.is_symlink():raise RuntimeError('Unexpected recovery symlink')
   dirs.append(p)
  for name in names:
   p=Path(directory)/name;s=p.lstat()
   if not stat.S_ISREG(s.st_mode):raise RuntimeError('Unexpected recovery file type')
   files.append({'path':str(p),'bytes':s.st_size,'mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns,'sha256':a.file_hash(p)})
 handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(tmp)],capture_output=True,text=True,timeout=60)
 if handles.returncode not in (0,1) or handles.stdout or handles.stderr:raise RuntimeError('Open recovery handles')
 for rec in files:
  p=Path(rec['path']);s=p.lstat()
  if s.st_size!=rec['bytes'] or stat.S_IMODE(s.st_mode)!=rec['mode'] or s.st_mtime_ns!=rec['mtime_ns'] or a.file_hash(p)!=rec['sha256']:raise RuntimeError('Recovery file changed before cleanup')
 report={'status':'passed','archive_commit':a.git('rev-parse','HEAD').decode().strip(),'plan':plan,'results':results,'restoration':json.loads((tmp/'runtime-restoration.json').read_text()),'temporary_cleanup':{'status':'validated','files':files,'open_handles':0,'df_before':subprocess.check_output(['df','-k','.'],text=True)}}
 (base/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 for rec in files:Path(rec['path']).unlink()
 for p in sorted(dirs,key=lambda p:len(p.parts),reverse=True):p.rmdir()
 tmp.rmdir();report['temporary_cleanup'].update(status='removed',df_after=subprocess.check_output(['df','-k','.'],text=True));(base/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':'passed','native_files_exact':exact,'relocated_manifests_verified':len(relocations),'learner_Adam_tensors_exact':tensors,'temporary_bytes_removed':sum(rec['bytes'] for rec in files)}))
except Exception as error:
 (base/'verification-failure.json').write_text(json.dumps({'error':str(error),'type':type(error).__name__,'plan':plan},indent=2)+'\n');raise
