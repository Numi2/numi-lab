from pathlib import Path
import gzip,hashlib,json,os,stat,subprocess,sys,time
R=Path.cwd();ar='artifacts/crow-rl-grasp-controls-20261108';br='build/crow-grasp-controls-build-01';gr='artifacts/crow-rl-git-recovery-grasp-controls-20261108'
A=R/ar;G=R/gr;X=R/'build/crow-grasp-controls-recovery-01'
sys.path.insert(0,str(R/'tools'));import crow_evidence_archive as archive
def sha(x):return hashlib.sha256(x).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
entries=archive.tree();X.mkdir();mapped={}
with archive.Objects() as objects:
 def committed(p):return objects.read(entries[p][1])
 raw_manifest=committed(ar+'/payload/manifest.json');m=json.loads(raw_manifest);assert raw_manifest==(A/'payload/manifest.json').read_bytes()
 for f in m['files']:
  raw=b''.join(archive.content(f,entries,objects));assert (sha(raw),len(raw))==(f['sha256'],f['bytes'])
  for p in [X/'original'/f['path']]+([] if '/observations/' in f['path'] else [X/f['path']]):
   p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(f['mode']);os.utime(p,ns=(f['mtime_ns'],f['mtime_ns']))
  mapped.setdefault((f['sha256'],f['bytes']),dict(original=f['path']))
 for p in entries:
  if p.startswith(ar+'/') and len(Path(p).parts)==3:
   dest=X/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(committed(p))
 plan=json.loads(committed(ar+'/plan.json'));runtime=json.loads(committed(ar+'/runtime.json'));old=archive.tree(runtime['source_commit'])
 for p,h in plan['sources'].items():
  raw=committed(p)
  if sha(raw)!=h:raw=objects.read(old[p][1])
  assert sha(raw)==h,p
  dest=X/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
 for p,f in runtime['files'].items():assert sha((X/p).read_bytes())==f['sha256'],p
 for p,h in plan['inputs'].items():assert sha((X/p).read_bytes())==h,p
 expected=json.loads(committed(ar+'/execution.json'));reports={n:committed(ar+'/'+n) for n in ['analysis.json','result.json']}
 assert all(e['returncode'] in [0,1] for e in expected),'A wall-time prefix requires a separately scoped replay'
O=X/br/'observations';O.mkdir();execution=[];start_all=time.monotonic()
env={**os.environ,'DYLD_LIBRARY_PATH':str(X/br/'runtime'),'DYLD_PRINT_LIBRARIES':'1','CROW_PROBE_METALLIB':str(X/br/'runtime/metalrobo/MetalRobo.metallib'),'PYTHONDONTWRITEBYTECODE':'1'}
for e in expected:
 name=e['name'];cmd=[str(X)+v[len(str(R)):] if v.startswith(str(R)+'/') else v for v in e['argv']];local=dict(env)
 if name not in ['admission','calibration']:local['METALROBO_QUALITY_FAILURE_CAPTURE']=str(O/(name+'-capture.bin'))
 start=time.monotonic()
 with (O/(name+'.stdout')).open('xb') as stdout,(O/(name+'.stderr')).open('xb') as stderr:
  got=subprocess.run(cmd,cwd=X,env=local,stdout=stdout,stderr=stderr,timeout=600 if name not in ['admission','calibration'] else 30).returncode
 assert got==e['returncode'],(name,got,e['returncode'])
 assert any(s.rstrip().endswith(str(X/br/'runtime/libmetalrobo.dylib')) for s in (O/(name+'.stderr')).read_text().splitlines()),name
 execution.append(dict(name=name,returncode=got,seconds=time.monotonic()-start));write(G/'execution.json',execution);print(json.dumps(execution[-1]),flush=True)
with (G/'analysis.stdout').open('xb') as stdout,(G/'analysis.stderr').open('xb') as stderr:
 code=subprocess.run([str(R/'python/.venv/bin/python'),str(X/ar/'analyze.py')],cwd=X,env=env,stdout=stdout,stderr=stderr,timeout=30).returncode
assert code==0
for n,raw in reports.items():assert (X/ar/n).read_bytes()==raw,n
compared=[]
for f in m['files']:
 p=f['path']
 if '/observations/' not in p:continue
 exact=not p.endswith(('.stdout','.stderr','runtime.json')) or p.endswith(('admission.stdout','calibration.stdout'))
 if exact:assert sha((X/p).read_bytes())==f['sha256'],p;compared.append(p)
for f in m['files']:
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
write(G/'temporary-inventory.json',dict(schema='crow-grasp-controls-temporary-v1',root=str(X.relative_to(R)),source_manifest=ar+'/payload/manifest.json',source_manifest_sha256=sha(raw_manifest),files=inventory))
write(G/'result.json',dict(schema='crow-grasp-controls-git-recovery-v1',native_commands=len(execution),all_native_commands_exact_returncodes=True,
 physical_and_numerical_files_exact=len(compared),scientific_reports_exact=list(reports),original_files_git_recovered_and_metadata_checked=len(m['files']),
 temporary_files=len(inventory),temporary_bytes=sum(f['bytes'] for f in inventory),manifest_sha256=sha(raw_manifest),seconds=time.monotonic()-start_all,
 all_passed=True,scope='Exact archived binaries/shader and native command prefix, independently regenerated analysis. No fresh source build.',production_changed=False,goal_complete=False))
print((G/'result.json').read_text(),flush=True)
