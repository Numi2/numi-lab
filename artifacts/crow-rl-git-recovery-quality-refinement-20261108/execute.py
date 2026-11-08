from pathlib import Path
import gzip,hashlib,json,os,stat,subprocess,sys,time
R=Path.cwd();ar='artifacts/crow-rl-quality-refinement-20261108';br='build/crow-quality-refinement-build-01';gr='artifacts/crow-rl-git-recovery-quality-refinement-20261108'
A=R/ar;G=R/gr;X=R/'build/crow-quality-refinement-recovery-01'
sys.path.insert(0,str(R/'tools'));import crow_evidence_archive as archive
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
entries=archive.tree();X.mkdir();mapped={}
with archive.Objects() as objects:
 def committed(path):return objects.read(entries[path][1])
 manifest_raw=committed(ar+'/payload/manifest.json');manifest=json.loads(manifest_raw);assert manifest_raw==(A/'payload/manifest.json').read_bytes()
 for f in manifest['files']:
  raw=b''.join(archive.content(f,entries,objects));assert len(raw)==f['bytes'] and sha(raw)==f['sha256']
  outputs=any('/'+part+'/' in f['path'] for part in ['observations','integration-observations'])
  for p in [X/'original'/f['path']]+([] if outputs else [X/f['path']]):
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(f['mode']);os.utime(p,ns=(f['mtime_ns'],f['mtime_ns']))
  mapped.setdefault((f['sha256'],f['bytes']),dict(original=f['path']))
 for path in entries:
  if path.startswith(ar+'/') and len(Path(path).parts)==3:
   dest=X/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(committed(path))
 reports=['component-analysis.json','component-result.json','integration-analysis.json','integration-result.json']
 originals={name:committed(ar+'/'+name) for name in reports}
 for plan_name in ['plan.json']:
  plan=json.loads(committed(ar+'/'+plan_name))
  for name,f in plan['sources'].items():
   expected=f if isinstance(f,str) else f['sha256']
   dest=X/name
   if not dest.exists():
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(committed(name))
   assert sha(dest.read_bytes())==expected
 aliases=json.loads(committed(ar+'/runtime-recovery.json'))['aliases']
 for target,source in aliases.items():
  dest=X/target;dest.parent.mkdir(parents=True,exist_ok=True)
  import shutil
  shutil.copy2(X/source,dest)
start=time.monotonic();steps=[]
for script,native in [('run.py',25),('integration-run.py',10)]:
 cmd=[str(R/'python/.venv/bin/python'),str(X/ar/script)]
 with (G/(script+'.stdout')).open('xb') as stdout,(G/(script+'.stderr')).open('xb') as stderr:
  code=subprocess.run(cmd,cwd=X,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=stdout,stderr=stderr,timeout=300).returncode
 steps.append(dict(script=script,returncode=code,native_commands=native));write(G/'execution.json',steps);assert code==0
for name,raw in originals.items():assert (X/ar/name).read_bytes()==raw,name
compared=[]
for f in manifest['files']:
 name=f['path']
 physical=('/observations/' in name or '/integration-observations/' in name) and not name.endswith(('.stdout','.stderr','runtime.json','.tsv'))
 if physical:assert sha((X/name).read_bytes())==f['sha256'];compared.append(name)
import csv
motor_records=0
for name in ['motor.tsv','zero.tsv']:
 def records(p):
  return [{k:v for k,v in r.items() if k!='gpu_ms'} for r in csv.DictReader(p.open(),delimiter='\t')]
 path=br+'/integration-observations/'+name
 original=records(X/'original'/path);replayed=records(X/path)
 assert original==replayed;motor_records+=len(original)
assert motor_records==624
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
write(G/'temporary-inventory.json',dict(schema='crow-quality-refinement-temporary-v1',root=str(X.relative_to(R)),source_manifest=ar+'/payload/manifest.json',source_manifest_sha256=sha(manifest_raw),files=inventory))
result=dict(schema='crow-quality-refinement-git-recovery-v1',manifest_sha256=sha(manifest_raw),native_commands=35,motor_physical_records_exact=motor_records,
 physical_and_numerical_files_exact=len(compared),scientific_reports_exact=reports,original_files_checked=len(manifest['files']),
 temporary_files=len(inventory),temporary_bytes=sum(f['bytes'] for f in inventory),seconds=time.monotonic()-start,
 all_passed=True,production_changed=True,goal_complete=False)
write(G/'result.json',result);print(json.dumps(result),flush=True)
