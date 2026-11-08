"""Execute the restored negative probe and verify its complete input recovery."""
from pathlib import Path
import hashlib,importlib.util,json,os,stat,subprocess,time
root=Path.cwd();base=root/'artifacts/crow-rl-ccd-postsubmission-capture-20261108'
tmp=Path('/private/tmp/crow-ccd-postsubmission-git-recovery-20261108')
spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
metadata=json.loads((base/'runtime-recovery.json').read_text())
def write(name,value):
 with (base/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
write('verification-plan.json',{'prediction':'Actual Git-restored standalone probe rejects the byte-restored post-submission snapshot with exactly the original exit1/error before GPU work and produces no result file. All restored files and the complete checkpoint COMMITTED/manifest/component hashes match Git.',
 'budget':'One native probe timeout60s; temporary<=384MiB; no training, simulation, learning-rate trial or retry.',
 'scope':'Negative measurement reproduction plus complete byte recovery; no causal failing-sweep replay or independently executed full learner recovery.'})
try:
 restoration=json.loads((tmp/'runtime-restoration.json').read_text())
 for f in restoration['files']:assert a.file_hash(tmp/f['path'])==f['sha256']
 cp=tmp/'input/source.trainingcheckpoint';manifest=json.loads((cp/'manifest.json').read_text())
 assert (cp/'COMMITTED').read_text()==a.file_hash(cp/'manifest.json')
 assert manifest['completedUpdates']==10 and manifest['acceptedControlTicks']==[1600]*128
 for name,f in manifest['files'].items():assert (cp/name).stat().st_size==f['bytes'] and a.file_hash(cp/name)==f['sha256']
 command=[str(tmp/'bin/metalrobo_ccd_sweep_probe'),str(tmp/'shaders/CCDSweepProbe.metallib'),str(tmp/'input/failure.ccdsweep'),str(tmp/'unexpected-result.bin')]
 write('negative-probe.command.json',command);start=time.monotonic()
 p=subprocess.run(command,cwd=tmp,capture_output=True,text=True,timeout=60)
 (base/'negative-probe.stdout').write_text(p.stdout);(base/'negative-probe.stderr').write_text(p.stderr)
 write('negative-probe.receipt.json',{'returncode':p.returncode,'wall_seconds':time.monotonic()-start,'stdout_sha256':hashlib.sha256(p.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(p.stderr.encode()).hexdigest()})
 assert p.returncode==metadata['negative_probe_reference']['returncode'] and not p.stdout and p.stderr==metadata['negative_probe_reference']['stderr']
 assert not (tmp/'unexpected-result.bin').exists()
 results={'negative_probe_rejection_exact':True,'native_GPU_queries_executed':0,'all_restored_file_hashes_exact':True,
  'complete_coupled_checkpoint_manifest_and_files_exact':True,'causal_failed_sweep_replay_qualified':False,'full_learner_recovery_executed':False}
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
 print(json.dumps({'status':'passed','negative_probe_reproduced_exact':True,'temporary_bytes_removed':total}))
except Exception as exc:
 write('verification-failure.json',{'error':str(exc),'type':type(exc).__name__});raise
