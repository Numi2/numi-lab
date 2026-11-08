from pathlib import Path
import gzip,hashlib,json,os,stat,subprocess,sys,time
R=Path.cwd();ar='artifacts/crow-rl-split-cones-20261108';br='build/crow-split-cones-build-01'
A=R/ar;G=R/'artifacts/crow-rl-git-recovery-split-cones-20261108';X=R/'build/crow-split-cones-recovery-01'
sys.path.insert(0,str(R/'tools'));import crow_evidence_archive as archive
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
entries=archive.tree();mapped={};start=time.monotonic()
with archive.Objects() as objects:
 manifest_raw=objects.read(entries[ar+'/payload/manifest.json'][1]);manifest=json.loads(manifest_raw)
 assert manifest_raw==(A/'payload/manifest.json').read_bytes()
 for f in manifest['files']:mapped.setdefault((f['sha256'],f['bytes']),dict(original=f['path']))
 originals={p:objects.read(entries[ar+'/'+p][1]) for p in ['case-analysis.json','derivative-calibration.json','result.json']}
for name in ['case-analysis.json','derivative-calibration.json']:assert (X/ar/name).read_bytes()==originals[name],name
baseline=json.loads(originals['result.json']);replayed=json.loads((X/ar/'result.json').read_text())
for key in ['seconds','raw_bytes']:baseline.pop(key);replayed.pop(key)
assert baseline==replayed,(baseline,replayed)
comparisons=[]
for f in manifest['files']:
 if '/observations/' in f['path'] and f['path'].endswith(('scratch.f32x2','status.bin')):
  assert sha((X/f['path']).read_bytes())==f['sha256'];comparisons.append(f['path'])
assert len(comparisons)==72
for f in manifest['files']:
 p=X/'original'/f['path'];s=p.stat()
 assert (sha(p.read_bytes()),s.st_size,stat.S_IMODE(s.st_mode),s.st_mtime_ns)==(f['sha256'],f['bytes'],f['mode'],f['mtime_ns'])
blobs=G/'recovery-blobs';blobs.mkdir();inventory=[]
for p in sorted(X.rglob('*')):
 if not p.is_file():continue
 assert not p.is_symlink()
 s=p.stat();raw=p.read_bytes();key=(sha(raw),len(raw));backing=mapped.get(key)
 if backing is None:
  dest=blobs/(key[0]+'.gz')
  if not dest.exists():dest.write_bytes(gzip.compress(raw,mtime=0))
  backing=dict(gzip=str(dest.relative_to(R)));mapped[key]=backing
 inventory.append(dict(path=str(p.relative_to(X)),sha256=key[0],bytes=key[1],mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,backing=backing))
write(G/'temporary-inventory.json',dict(schema='crow-split-cones-temporary-v1',root=str(X.relative_to(R)),source_manifest=ar+'/payload/manifest.json',source_manifest_sha256=sha(manifest_raw),files=inventory))
result=dict(schema='crow-split-cones-git-recovery-v1',manifest_sha256=sha(manifest_raw),
 native_commands=36,native_binary_outputs_exact=len(comparisons),case_and_calibration_reports_exact=True,scientific_result_exact_excluding_operational_fields=True,excluded_fields=['seconds','raw_bytes'],
 original_files_checked=len(manifest['files']),temporary_files=len(inventory),temporary_bytes=sum(x['bytes'] for x in inventory),
 seconds=time.monotonic()-start,source_versions='Actual Git blobs; original bound README restored from plan base commit because result documentation was appended after observations.',
 all_passed=True,production_changed=False,goal_complete=False)
write(G/'result.json',result);print(json.dumps(result),flush=True)
