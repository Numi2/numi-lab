"""Execute the minimal protected-sweep runtime restored from actual Git objects."""
from pathlib import Path
import importlib.util,json,os,stat,subprocess,time,hashlib
r=Path.cwd();base=r/'artifacts/crow-rl-ccd-stationary-20261108'
tmp=Path('/private/tmp/crow-ccd-stationary-recovery-20261108');tmp.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('archive',r/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def write(name,v):
 with (base/name).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
write('verification-plan.json',{'prediction':'Actual Git-restored native probe and both shaders reproduce the all 4240 repaired-sweep stream bytes and all 8194 stationary/moving-control records exactly twice from the Git-restored snapshot.','budget':'Four native executions timeout180s each including cold compilation; temporary<=64MiB; no trajectory or training.','scope':'Causal repaired-sweep and independent-controls reproduction from exact Git bytes. Full learner recovery is not independently executed for this diagnostic capture.'})
try:
 entries=a.tree();meta=json.loads((base/'runtime-recovery.json').read_text());records=[]
 with a.Objects() as objects:
  manifest=json.loads(objects.read(entries[str(base.relative_to(r)/'manifest.json')][1]));files={f['path']:f for f in manifest['files']}
  for target,source in meta['files'].items():
   f=files[source];p=tmp/a.safe_path(target);p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('xb') as out:
    for block in a.content(f,entries,objects):out.write(block)
   p.chmod(f['mode']);assert a.file_hash(p)==f['sha256'];records.append({'path':target,'sha256':f['sha256'],'bytes':f['bytes']})
 cmd=[str(tmp/'bin/probe'),str(tmp/'shaders/production.metallib'),str(tmp/'shaders/query.metallib'),str(tmp/'input/failure.ccdsweep')]
 results=[]
 for name in ('replay','repeat'):
  command=cmd+[str(tmp/(name+'.bin'))];write(name+'.command.json',command);t=time.monotonic()
  p=subprocess.run(command,cwd=tmp,capture_output=True,timeout=180)
  (base/(name+'.stdout')).write_bytes(p.stdout);(base/(name+'.stderr')).write_bytes(p.stderr)
  write(name+'.receipt.json',{'exit_code':p.returncode,'wall_seconds':time.monotonic()-t,'stdout_sha256':hashlib.sha256(p.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(p.stderr).hexdigest()})
  assert p.returncode==0;v=json.loads(p.stdout);assert v['replay_records'][0]['flags']==33 and v['replay_records'][0]['status']==0;results.append(v)
 assert results[0]==results[1]
 assert (tmp/'replay.bin').read_bytes()==(tmp/'repeat.bin').read_bytes()==(tmp/'input/reference.bin').read_bytes()
 for name in ('cases','cases-repeat'):
  command=[str(tmp/'bin/cases'),str(tmp/'shaders/production.metallib'),str(tmp/'input/cases.bin'),str(tmp/(name+'.bin'))];write(name+'.command.json',command);t=time.monotonic()
  p=subprocess.run(command,cwd=tmp,capture_output=True,timeout=180)
  (base/(name+'.stdout')).write_bytes(p.stdout);(base/(name+'.stderr')).write_bytes(p.stderr)
  write(name+'.receipt.json',{'exit_code':p.returncode,'wall_seconds':time.monotonic()-t,'stdout_sha256':hashlib.sha256(p.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(p.stderr).hexdigest()})
  assert p.returncode==0
 assert (tmp/'cases.bin').read_bytes()==(tmp/'cases-repeat.bin').read_bytes()==(tmp/'input/cases-reference.bin').read_bytes()
 for f in records:assert a.file_hash(tmp/f['path'])==f['sha256']
 files=[];directories=[]
 for directory,subdirs,names in os.walk(tmp,followlinks=False):
  for n in subdirs:
   p=Path(directory)/n;assert not p.is_symlink();directories.append(p)
  for n in names:
   p=Path(directory)/n;s=p.lstat();assert stat.S_ISREG(s.st_mode)
   files.append({'path':str(p),'bytes':s.st_size,'mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns,'sha256':a.file_hash(p)})
 total=sum(f['bytes'] for f in files);assert total<=64*1024**2
 handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(tmp)],capture_output=True,text=True,timeout=60)
 assert handles.returncode in (0,1) and not handles.stdout and not handles.stderr
 for f in files:
  p=Path(f['path']);s=p.lstat();assert s.st_size==f['bytes'] and s.st_mtime_ns==f['mtime_ns'] and stat.S_IMODE(s.st_mode)==f['mode'] and a.file_hash(p)==f['sha256']
 result={'status':'passed','archive_commit':a.git('rev-parse','HEAD').decode().strip(),'restored_files':records,'repaired_candidate_and_all_stream_bytes_exact':True,'all_8194_case_records_exact':True,'repeat_exact':True,'full_learner_recovery_executed':False,'temporary_cleanup':{'status':'validated','files':files,'bytes':total,'open_handles':0}}
 write('verification-before-cleanup.json',result)
 for f in files:Path(f['path']).unlink()
 for p in sorted(directories,key=lambda p:len(p.parts),reverse=True):p.rmdir()
 tmp.rmdir();result['temporary_cleanup']['status']='removed';write('verification.json',result)
 print(json.dumps({'status':'passed','replay_exact':True,'temporary_bytes_removed':total}))
except Exception as e:write('verification-failure.json',{'type':type(e).__name__,'error':str(e)});raise
