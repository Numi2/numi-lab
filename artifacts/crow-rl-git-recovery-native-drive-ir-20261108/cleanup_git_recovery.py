from pathlib import Path
import gzip,hashlib,json,stat,subprocess,sys
R=Path.cwd();G=R/'artifacts/crow-rl-git-recovery-native-drive-ir-20261108';X=R/'build/crow-native-drive-ir-git-recovery-01'
sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects,content

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/agent/american-crow-rl-20261107'],text=True).split()[0];assert head==remote
entries=tree()
with Objects() as objects:
 def read_committed(p):
  raw=objects.read(entries[str(p.relative_to(R))][1]);assert raw==p.read_bytes();return json.loads(raw)
 plan=read_committed(G/'plan.json');inventory=read_committed(G/'temporary-inventory.json');manifest=json.loads(objects.read(entries[plan['manifest']][1]));sources={f['path']:f for f in manifest['files']};verified={}
 for f in inventory:
  key=f['sha256'],f['bytes']
  if key in verified:continue
  if 'archive_source' in f:data=b''.join(content(sources[f['archive_source']],entries,objects))
  else:data=gzip.decompress(objects.read(entries[f['recovery_blob']][1]))
  assert len(data)==f['bytes'] and hashlib.sha256(data).hexdigest()==f['sha256'];verified[key]=True
handles=subprocess.run(['/usr/sbin/lsof','-Fpn','+D',str(X)],capture_output=True,text=True,timeout=60)
write(G/'cleanup-open-handles.json',{'exit_code':handles.returncode,'stdout':handles.stdout,'stderr':handles.stderr})
assert handles.returncode in (0,1) and not handles.stdout and not handles.stderr
actual={str(p.relative_to(R)) for p in X.rglob('*') if p.is_file()};assert actual=={f['path'] for f in inventory}
for f in inventory:
 p=R/f['path'];assert p.is_relative_to(X) and not p.is_symlink();s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_size==f['bytes'] and s.st_mtime_ns==f['mtime_ns'] and stat.S_IMODE(s.st_mode)==f['mode'] and sha(p)==f['sha256']
before=subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)
for f in inventory:(R/f['path']).unlink()
for p in sorted((p for p in X.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.rmdir()
X.rmdir()
report={'archive_commit':head,'remote_commit':remote,'removed_files':len(inventory),'removed_logical_bytes':sum(f['bytes'] for f in inventory),'actual_git_unique_contents_verified':len(verified),'open_handles':0,'df_before':before,'df_after':subprocess.check_output(['df','-k','/System/Volumes/Data'],text=True)}
write(G/'temporary-cleanup.json',report);print(json.dumps(report),flush=True)
