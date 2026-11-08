from pathlib import Path
import hashlib,json,os,shlex,stat,subprocess
R=Path.cwd();W=R/'work';O=W/'local-scratch-export';remote_root='/Users/n/Developer/numi-crow-rl-20261107';G='artifacts/crow-rl-git-recovery-bounded-drive-20261108'
command='export PATH=/opt/homebrew/bin:/usr/local/bin:$PATH; export PYTHONDONTWRITEBYTECODE=1; cd '+shlex.quote(remote_root)+' && python3 '+shlex.quote(G+'/local-scratch/verify_local_scratch.py')+' --stdout'
proof=json.loads(subprocess.check_output(['ssh','macmini',command],text=True));manifest_raw=(O/'manifest.json').read_bytes();assert hashlib.sha256(manifest_raw).hexdigest()==proof['manifest_sha256'];m=json.loads(manifest_raw)
expected={f['path'] for f in m['files']};actual={str(p.relative_to(R)) for p in W.rglob('*') if p.is_file() and not p.is_relative_to(O)};assert actual==expected
archive={f['path']:f for f in proof['archive_files']};assert {str(p.relative_to(O)) for p in O.rglob('*') if p.is_file()}==set(archive)
inventory=[]
for f in m['files']:
 p=R/f['path'];s=p.lstat();assert p.is_relative_to(W) and not p.is_relative_to(O) and stat.S_ISREG(s.st_mode) and not p.is_symlink();assert s.st_size==f['bytes'] and stat.S_IMODE(s.st_mode)==f['mode'] and s.st_mtime_ns==f['mtime_ns'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];inventory.append(dict(f,kind='source'))
for name,f in archive.items():
 p=O/name;s=p.lstat();assert p.is_relative_to(O) and stat.S_ISREG(s.st_mode) and not p.is_symlink() and s.st_size==f['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];inventory.append({'path':str(p.relative_to(R)),'kind':'export-copy','bytes':s.st_size,'sha256':f['sha256'],'mode':stat.S_IMODE(s.st_mode),'mtime_ns':s.st_mtime_ns})
handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(W)],capture_output=True,text=True,timeout=60);assert handles.returncode in (0,1) and not handles.stdout and not handles.stderr,(handles.returncode,handles.stdout,handles.stderr)
before=subprocess.check_output(['df','-k','.'],text=True)
# Validate the entire deletion inventory once more before its first mutation.
for f in inventory:
 p=R/f['path'];s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_size==f['bytes'] and s.st_mtime_ns==f['mtime_ns'] and stat.S_IMODE(s.st_mode)==f['mode'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
for f in inventory:(R/f['path']).unlink()
for p in sorted((p for p in W.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.rmdir()
W.rmdir()
report={'archive_commit':proof['archive_commit'],'remote_commit':proof['remote_commit'],'manifest_sha256':proof['manifest_sha256'],'deleted_files':len(inventory),'deleted_logical_bytes':sum(f['bytes'] for f in inventory),'source_files':len(m['files']),'source_bytes':m['summary']['source_bytes'],'open_handles':0,'inventory':inventory,'df_before':before,'df_after':subprocess.check_output(['df','-k','.'],text=True)}
writer="import pathlib,sys;p=pathlib.Path("+repr(G+'/local-scratch-cleanup.json')+");assert not p.exists();p.write_bytes(sys.stdin.buffer.read())"
subprocess.run(['ssh','macmini','cd '+shlex.quote(remote_root)+' && python3 -c '+shlex.quote(writer)],input=(json.dumps(report,indent=2,sort_keys=True)+'\n').encode(),check=True)
print(json.dumps({k:v for k,v in report.items() if k!='inventory'}))
