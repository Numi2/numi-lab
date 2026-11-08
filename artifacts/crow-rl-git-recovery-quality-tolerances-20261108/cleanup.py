from pathlib import Path
import gzip,hashlib,json,os,stat,subprocess,sys
R=Path.cwd();G=R/'artifacts/crow-rl-git-recovery-quality-tolerances-20261108'
sys.path.insert(0,str(R/'tools'));import crow_evidence_archive as archive
def sha(x):return hashlib.sha256(x).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip();assert branch==archive.BRANCH
head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],text=True).split()[0]==head
entries=archive.tree();path=str((G/'temporary-inventory.json').relative_to(R))
with archive.Objects() as objects:
 raw=objects.read(entries[path][1]);assert raw==(G/'temporary-inventory.json').read_bytes();inventory=json.loads(raw)
 manifest_raw=objects.read(entries[inventory['source_manifest']][1]);assert sha(manifest_raw)==inventory['source_manifest_sha256']
 manifest=json.loads(manifest_raw);originals={f['path']:f for f in manifest['files']}
 X=R/inventory['root'];assert X==R/'build/crow-quality-tolerances-recovery-01'
 p=subprocess.run(['lsof','-nP','+D',str(X)],capture_output=True)
 assert p.returncode==1 and not p.stdout,(p.returncode,p.stdout.decode(),p.stderr.decode())
 expected={f['path'] for f in inventory['files']};assert {str(p.relative_to(X)) for p in X.rglob('*') if p.is_file()}==expected
 for f in inventory['files']:
  path=X/f['path'];s=path.lstat()
  assert stat.S_ISREG(s.st_mode) and (sha(path.read_bytes()),s.st_size,stat.S_IMODE(s.st_mode),s.st_mtime_ns)==(f['sha256'],f['bytes'],f['mode'],f['mtime_ns'])
  b=f['backing']
  recovered=b''.join(archive.content(originals[b['original']],entries,objects)) if 'original' in b else gzip.decompress(objects.read(entries[b['gzip']][1]))
  assert (sha(recovered),len(recovered))==(f['sha256'],f['bytes'])
for f in inventory['files']:(X/f['path']).unlink()
for p in sorted([p for p in X.rglob('*') if p.is_dir()],key=lambda p:len(p.parts),reverse=True):p.rmdir()
X.rmdir()
write(G/'cleanup.json',dict(schema='crow-quality-tolerances-recovery-cleanup-v1',verified_commit=head,files_removed=len(inventory['files']),bytes_removed=sum(f['bytes'] for f in inventory['files']),actual_git_recovery_verified=True,exact_hash_mode_mtime=True,open_handles=0,root_absent=not X.exists()))
print((G/'cleanup.json').read_text(),flush=True)
