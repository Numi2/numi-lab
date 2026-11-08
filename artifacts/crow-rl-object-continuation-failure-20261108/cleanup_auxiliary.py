from pathlib import Path
import hashlib,importlib.util,json,stat,subprocess
root=Path.cwd();base=root/'artifacts/crow-rl-object-continuation-failure-20261108'
spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
head=a.git('rev-parse','HEAD').decode().strip();remote=a.git('ls-remote','origin','refs/heads/agent/american-crow-rl-20261107').decode().split()[0]
if head!=remote:raise RuntimeError('Remote recovery is not current')
entries=a.tree();mapping={root/'build/crow-object-continuation-build-01/verify_failure_recovery.py':base/'verify_recovery.py',
 Path('/private/tmp/crow-ccd-pair-identity-20261108'):base/'pair_identity'}
for stem,archived in [('archive','archive-capture'),('restore','restore-wrapper'),('verify','verify-wrapper'),('prune','prune-wrapper')]:
 for stream in ('stdout','stderr'):
  mapping[Path('/private/tmp')/('crow-object-continuation-failure-'+stem+'-20261108.'+stream)]=base/(archived+'.'+stream)
before=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
files=[]
with a.Objects() as objects:
 for source,target in mapping.items():
  relative=str(target.relative_to(root));mode,oid=entries[relative];saved=objects.read(oid);s=source.lstat()
  if not stat.S_ISREG(s.st_mode) or source.read_bytes()!=saved:raise RuntimeError('Auxiliary copy differs from Git: '+str(source))
  files.append({'path':str(source),'git_path':relative,'git_blob':oid,'bytes':s.st_size,'mode':stat.S_IMODE(s.st_mode),
                'mtime_ns':s.st_mtime_ns,'sha256':a.file_hash(source)})
h=subprocess.run(['/usr/sbin/lsof','-Fpn',*[f['path'] for f in files]],capture_output=True,text=True,timeout=60)
if h.returncode not in (0,1) or h.stdout or h.stderr:raise RuntimeError('Open auxiliary handles')
for f in files:
 p=Path(f['path']);s=p.lstat()
 if (s.st_size,s.st_mtime_ns,stat.S_IMODE(s.st_mode),a.file_hash(p))!=(f['bytes'],f['mtime_ns'],f['mode'],f['sha256']):
  raise RuntimeError('Auxiliary file changed')
report={'commit':head,'remote_commit':remote,'files':files,'open_handles':0,'removed_logical_bytes':sum(f['bytes'] for f in files),
        'df_before':before,'status':'validated'}
a.write_json(base/'auxiliary-cleanup-before.json',report)
for f in files:Path(f['path']).unlink()
(root/'build/crow-object-continuation-build-01').rmdir()
report['status']='removed';report['df_after']=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
a.write_json(base/'auxiliary-cleanup.json',report)
print(json.dumps({'status':'removed','files':len(files),'bytes':report['removed_logical_bytes']}))
