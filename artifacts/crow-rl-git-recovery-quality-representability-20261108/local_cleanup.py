from pathlib import Path
import hashlib,json,stat,subprocess
ROOT=Path('/Users/home/Documents/Codex/2026-11-07/c');W=ROOT/'work/quality-representability';REPO='/Users/n/Developer/numi-crow-rl-20261107'
REMOTE=r'''
from pathlib import Path
import hashlib,json,subprocess,sys
R=Path.cwd();A=R/'artifacts/crow-rl-quality-representability-20261108';G=R/'artifacts/crow-rl-git-recovery-quality-representability-20261108'
sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects,content
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/agent/american-crow-rl-20261107'],text=True).split()[0];assert head==remote
entries=tree()
with Objects() as objects:
 def read(p):return objects.read(entries[str(p.relative_to(R))][1])
 plan=json.loads(read(G/'plan.json'));manifest=json.loads(objects.read(entries[plan['manifest']][1]));sources={f['path']:f for f in manifest['files']}
 raw=read(A/'local-authoring-inventory.json');inventory=json.loads(raw)
 for f in inventory:
  archived=sources[f['archive_path']];assert archived['sha256']==f['sha256'] and archived['bytes']==f['bytes']
  data=b''.join(content(archived,entries,objects));assert len(data)==f['bytes'] and hashlib.sha256(data).hexdigest()==f['sha256']
 controls=json.loads(read(G/'local-control-inventory.json'))
 mapping={'crow-representability-local-inventory.json':A/'local-authoring-inventory.json','quality-representability-recovery.py':G/'execute_git_recovery.py','cleanup_representability_local.py':G/'local_cleanup.py'}
 for f in controls:
  p=mapping[Path(f['local_path']).name];data=read(p);assert hashlib.sha256(data).hexdigest()==f['sha256'] and len(data)==f['bytes']
 report=dict(archive_commit=head,remote_commit=remote,inventory=inventory,controls=controls,actual_git_files_verified=len(inventory)+len(controls))
 (G/'local-recovery-proof.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report))
'''
remote_command='export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH PYTHONDONTWRITEBYTECODE=1; cd '+REPO+' && python3 -'
p=subprocess.run(['ssh','macmini',remote_command],input=REMOTE,text=True,capture_output=True,timeout=120)
assert p.returncode==0,p.stderr;proof=json.loads(p.stdout);items=proof['inventory']+proof['controls']
actual={str(p) for p in W.rglob('*') if p.is_file()};assert actual=={f['local_path'] for f in proof['inventory']}
allowed={str(ROOT/'work'/name) for name in ['crow-representability-local-inventory.json','quality-representability-recovery.py','cleanup_representability_local.py']}
assert {f['local_path'] for f in proof['controls']}==allowed
handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(W)],capture_output=True,text=True,timeout=60);assert handles.returncode in (0,1) and not handles.stdout and not handles.stderr
for f in items:
 p=Path(f['local_path']);s=p.lstat();assert not p.is_symlink() and stat.S_ISREG(s.st_mode) and s.st_size==f['bytes'] and stat.S_IMODE(s.st_mode)==f['mode'] and s.st_mtime_ns==f['mtime_ns'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
before=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
for f in items:Path(f['local_path']).unlink()
for p in sorted((p for p in W.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.rmdir()
W.rmdir()
report=dict(archive_commit=proof['archive_commit'],remote_commit=proof['remote_commit'],removed_files=len(items),removed_logical_bytes=sum(f['bytes'] for f in items),actual_git_files_verified=proof['actual_git_files_verified'],open_handles=0,df_before=before,df_after=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True))
receipt=ROOT/'work/crow-representability-local-cleanup.json';receipt.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report))
