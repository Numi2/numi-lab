from pathlib import Path
import gzip,hashlib,json,os,stat,subprocess,sys,time
R=Path.cwd();ar='artifacts/crow-rl-quality-tolerances-20261108';br='build/crow-quality-tolerances-build-01';gr='artifacts/crow-rl-git-recovery-quality-tolerances-20261108'
A=R/ar;G=R/gr;X=R/'build/crow-quality-tolerances-recovery-01'
sys.path.insert(0,str(R/'tools'));import crow_evidence_archive as archive
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
entries=archive.tree();X.mkdir();mapped={}
with archive.Objects() as objects:
 def committed(path):return objects.read(entries[path][1])
 manifest_raw=committed(ar+'/payload/manifest.json');manifest=json.loads(manifest_raw);assert manifest_raw==(A/'payload/manifest.json').read_bytes()
 for f in manifest['files']:
  raw=b''.join(archive.content(f,entries,objects));assert len(raw)==f['bytes'] and sha(raw)==f['sha256']
  outputs=any('/'+part+'/' in f['path'] for part in ['observations','analysis','split-observations'])
  for p in [X/'original'/f['path']]+([] if outputs else [X/f['path']]):
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(f['mode']);os.utime(p,ns=(f['mtime_ns'],f['mtime_ns']))
  mapped.setdefault((f['sha256'],f['bytes']),dict(original=f['path']))
 for path in entries:
  if path.startswith(ar+'/') and len(Path(path).parts)==3:
   dest=X/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(committed(path))
 reports=['case-analysis.json','result.json','exploratory-analysis.json','split-result.json']
 originals={name:committed(ar+'/'+name) for name in reports}
 for plan_name in ['plan.json','split-plan.json']:
  plan=json.loads(committed(ar+'/'+plan_name))
  for name,f in plan['sources'].items():
   expected=f if isinstance(f,str) else f['sha256'];assert sha((X/ar/name).read_bytes())==expected
start=time.monotonic();steps=[]
for script,native in [('run.py',11),('explore.py',0),('split-run.py',6)]:
 cmd=[str(R/'python/.venv/bin/python'),str(X/ar/script)]
 with (G/(script+'.stdout')).open('xb') as stdout,(G/(script+'.stderr')).open('xb') as stderr:
  code=subprocess.run(cmd,cwd=X,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=stdout,stderr=stderr,timeout=300).returncode
 steps.append(dict(script=script,returncode=code,native_commands=native));write(G/'execution.json',steps);assert code==0
for name,raw in originals.items():assert (X/ar/name).read_bytes()==raw,name
compared=[]
for f in manifest['files']:
 name=f['path']
 physical=('/observations/' in name or '/analysis/' in name or '/split-observations/' in name) and not name.endswith(('.stdout','.stderr','runtime.json'))
 if physical:assert sha((X/name).read_bytes())==f['sha256'];compared.append(name)
for f in manifest['files']:
 p=X/'original'/f['path'];s=p.stat();assert (sha(p.read_bytes()),s.st_size,stat.S_IMODE(s.st_mode),s.st_mtime_ns)==(f['sha256'],f['bytes'],f['mode'],f['mtime_ns'])
blobs=G/'recovery-blobs';blobs.mkdir();inventory=[]
for p in sorted(X.rglob('*')):
 if not p.is_file():continue
 assert not p.is_symlink();s=p.stat();raw=p.read_bytes();key=(sha(raw),len(raw));backing=mapped.get(key)
 if backing is None:
  dest=blobs/(key[0]+'.gz')
  if not dest.exists():dest.write_bytes(gzip.compress(raw,mtime=0))
  backing=dict(gzip=str(dest.relative_to(R)));mapped[key]=backing
 inventory.append(dict(path=str(p.relative_to(X)),sha256=key[0],bytes=key[1],mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,backing=backing))
write(G/'temporary-inventory.json',dict(schema='crow-quality-tolerances-temporary-v1',root=str(X.relative_to(R)),source_manifest=ar+'/payload/manifest.json',source_manifest_sha256=sha(manifest_raw),files=inventory))
result=dict(schema='crow-quality-tolerances-git-recovery-v1',manifest_sha256=sha(manifest_raw),native_commands=17,
 physical_and_numerical_files_exact=len(compared),scientific_reports_exact=reports,original_files_checked=len(manifest['files']),
 temporary_files=len(inventory),temporary_bytes=sum(f['bytes'] for f in inventory),seconds=time.monotonic()-start,
 all_passed=True,production_changed=False,goal_complete=False)
write(G/'result.json',result);print(json.dumps(result),flush=True)
