from pathlib import Path
import hashlib,importlib.util,json,os,stat,subprocess
root=Path.cwd();base=root/'artifacts/crow-rl-memory-retention-20261108';tmp=Path('/private/tmp/crow-memory-retention-git-recovery-20261108')
spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
m=json.loads((base/'runtime-recovery.json').read_text());plan={'prediction':'All Git restored checkpoint/input/runtime files match exact captured SHA256. Restored Python actually loads all53 model/Adam tensors, and restored CAPI library admits the final65 checkpoint with zero physical advances. Adjacent restored libraries must actually load.','budget':'One120second MLX restore process and one180second native loader process; no physical simulation or retry.','limit':'The captured native layout is faulty and future repeatability is NOT qualified. This is evidence/recovery preservation, not scientific success.'}
(base/'verification-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(name,c,timeout):
 (base/(name+'.command.json')).write_text(json.dumps(c,indent=2)+'\n');env={k:v for k,v in os.environ.items() if not k.startswith('DYLD_')};env['DYLD_PRINT_LIBRARIES']='1'
 p=subprocess.run(c,capture_output=True,text=True,timeout=timeout,env=env);(base/(name+'.stdout')).write_text(p.stdout);(base/(name+'.stderr')).write_text(p.stderr)
 if p.returncode:raise RuntimeError('Git-restored process failed: '+name)
 lines=[line for line in p.stderr.splitlines() if line.startswith('dyld[') and 'libmetalrobo.dylib' in line]
 if not lines or any(not line.endswith(str(tmp/'lib/libmetalrobo.dylib')) for line in lines):raise RuntimeError('Wrong native library loaded: '+name)
 value=json.loads(p.stdout)
 if value.get('status')!='passed' or value.get('physical_steps_advanced')!=0:raise RuntimeError('Unexpected native progress')
 return {'summary':value,'loaded_library':lines}
try:
 for name,rec in m['checkpoint_reference'].items():
  if sha(tmp/'input/checkpoint'/name)!=rec['sha256']:raise RuntimeError('Checkpoint Git bytes differ: '+name)
 learner=run('learner-recovery',[str(root/'python/.venv/bin/python'),str(base/'verify_learner.py'),str(tmp),str(base/'runtime-recovery.json')],120)
 last=m['last_lifetime'];native=run('native-restore-admission',[str(tmp/'bin/life_restore_check'),str(tmp/'project'),str(tmp/'input/checkpoint/policy.policypack'),str(tmp/'input/checkpoint/life.native'),str(last['physical_seed']),str(last['allocation_seed'])],180)
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
 for v in files:
  p=Path(v['path']);s=p.lstat()
  if s.st_size!=v['bytes'] or stat.S_IMODE(s.st_mode)!=v['mode'] or s.st_mtime_ns!=v['mtime_ns'] or sha(p)!=v['sha256']:raise RuntimeError('Recovery changed during cleanup')
 report={'status':'passed','archive_commit':a.git('rev-parse','HEAD').decode().strip(),'plan':plan,'learner':learner,'native_restore':native,'future_repeatability_qualified':False,'known_native_history_race_preserved':True,'restoration':json.loads((tmp/'runtime-restoration.json').read_text()),'temporary_cleanup':{'status':'validated','files':files,'open_handles':0}}
 (base/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 for v in files:Path(v['path']).unlink()
 for p in sorted(dirs,key=lambda p:len(p.parts),reverse=True):p.rmdir()
 tmp.rmdir();report['temporary_cleanup']['status']='removed';(base/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':'passed','actual_model_and_Adam_tensors_exact':learner['summary']['actual_model_and_Adam_tensors_exact'],'physical_steps_advanced':0,'future_repeatability_qualified':False,'temporary_files_removed':len(files),'temporary_bytes_removed':sum(v['bytes'] for v in files)}))
except Exception as error:
 (base/'verification-failure.json').write_text(json.dumps({'error':str(error),'type':type(error).__name__},indent=2)+'\n');raise
