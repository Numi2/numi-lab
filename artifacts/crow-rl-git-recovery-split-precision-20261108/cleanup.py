from pathlib import Path
import gzip,hashlib,json,stat,subprocess,sys
R=Path.cwd();G=R/'artifacts/crow-rl-git-recovery-split-precision-20261108';X=R/'build/crow-split-precision-git-recovery-01'
sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects,content
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/agent/american-crow-rl-20261107'],text=True).split()[0];assert head==remote
entries=tree()
with Objects() as objects:
 def read(p):return objects.read(entries[str(p.relative_to(R))][1])
 plan=json.loads(read(G/'plan.json'));manifest=objects.read(entries[plan['manifest']][1]);assert hashlib.sha256(manifest).hexdigest()==plan['manifest_sha256'];sources={f['path']:f for f in json.loads(manifest)['files']};inventory=json.loads(read(G/'temporary-inventory.json'));verified=set()
 for f in inventory:
  key=f['sha256'],f['bytes']
  if key not in verified:
   raw=b''.join(content(sources[f['archive_source']],entries,objects)) if 'archive_source' in f else gzip.decompress(objects.read(entries[f['recovery_blob']][1]))
   assert len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256'];verified.add(key)
actual={str(p.relative_to(R)) for p in X.rglob('*') if p.is_file()};assert actual=={f['path'] for f in inventory}
handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(X)],capture_output=True,text=True,timeout=60);assert handles.returncode in (0,1) and not handles.stdout and not handles.stderr
for f in inventory:
 p=R/f['path'];s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink() and s.st_size==f['bytes'] and stat.S_IMODE(s.st_mode)==f['mode'] and s.st_mtime_ns==f['mtime_ns'] and hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
before=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
for f in inventory:(R/f['path']).unlink()
for p in sorted((p for p in X.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.rmdir()
X.rmdir()
report=dict(archive_commit=head,remote_commit=remote,actual_git_unique_contents_verified=len(verified),removed_files=len(inventory),removed_logical_bytes=sum(f['bytes'] for f in inventory),open_handles=0,df_before=before,df_after=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True))
(G/'temporary-cleanup.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report))
