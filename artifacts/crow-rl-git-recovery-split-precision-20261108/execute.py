from pathlib import Path
import gzip,hashlib,json,os,stat,subprocess,sys,time
R=Path.cwd();A=R/'artifacts/crow-rl-split-precision-20261108';G=R/'artifacts/crow-rl-git-recovery-split-precision-20261108';X=R/'build/crow-split-precision-git-recovery-01'
sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects,content
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((G/'plan.json').read_text());entries=tree()
with Objects() as objects:
 def read(name):return objects.read(entries[name][1])
 raw=read(plan['manifest']);assert hashlib.sha256(raw).hexdigest()==plan['manifest_sha256'];m=json.loads(raw);assert m['summary']['source_bytes']<=plan['restore_max_bytes'];X.mkdir()
 for f in m['files']:
  p=X/'original'/f['path'];p.parent.mkdir(parents=True,exist_ok=True);data=b''.join(content(f,entries,objects));assert len(data)==f['bytes'] and hashlib.sha256(data).hexdigest()==f['sha256'];p.write_bytes(data);p.chmod(f['mode']);os.utime(p,ns=(f['mtime_ns'],f['mtime_ns']))
 for name,(mode,oid) in entries.items():
  if name.startswith(str(A.relative_to(R))+'/') and '/payload/' not in name:
   p=X/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(objects.read(oid));p.chmod(int(mode,8)&0o777)
 P=json.loads(read(str(A.relative_to(R)/'plan.json')));runtime=json.loads(read(str(A.relative_to(R)/'runtime.json')))
 replay_inputs=list(runtime['files'])+[c['path'] for c in P['cases']]
 for name in replay_inputs:
  src=X/'original'/name;p=X/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(src.read_bytes());p.chmod(stat.S_IMODE(src.stat().st_mode))
print(json.dumps(dict(restored_files=len(m['files']),raw_bytes=m['summary']['source_bytes'],working_inputs=len(replay_inputs))),flush=True)
env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';cmd=[str(R/'python/.venv/bin/python'),str(X/A.relative_to(R)/'run.py')];start=time.monotonic()
with (X/'replay.stdout').open('xb') as out,(X/'replay.stderr').open('xb') as err:
 try:code=subprocess.run(cmd,cwd=X,env=env,stdout=out,stderr=err,timeout=120).returncode
 except subprocess.TimeoutExpired:code=None
execution=dict(argv=cmd,cwd=str(X),returncode=code,seconds=time.monotonic()-start);write(G/'execution.json',execution);assert code==0,execution
comparisons=[]
with Objects() as objects:
 for name in ['analysis.json','result.json']:
  expected=json.loads(objects.read(entries[str(A.relative_to(R)/name)][1]));actual=json.loads((X/A.relative_to(R)/name).read_text())
  excluded=['raw_bytes'] if name=='result.json' else []
  if excluded:expected={k:v for k,v in expected.items() if k not in excluded};actual={k:v for k,v in actual.items() if k not in excluded}
  assert actual==expected,name;comparisons.append(dict(report=name,all_scientific_fields_exact=True,excluded_administrative_fields=excluded))
original=X/'original'/'build/crow-split-precision-build-01/observations';replayed=X/'build/crow-split-precision-build-01/observations';binary_count=0
for p in sorted(original.rglob('*')):
 if p.is_file() and p.name in ['status.bin','scratch.f32x2','arithmetic.f32x2']:
  dest=replayed/p.relative_to(original);assert dest.read_bytes()==p.read_bytes();binary_count+=1;comparisons.append(dict(binary=str(p.relative_to(X/'original')),byte_exact=True,sha256=sha(dest)))
assert binary_count==26
for f in m['files']:
 p=X/'original'/f['path'];assert sha(p)==f['sha256'] and p.stat().st_size==f['bytes']
write(G/'comparisons.json',comparisons)
known={(f['sha256'],f['bytes']):f['path'] for f in m['files']};inventory=[];(G/'recovery-blobs').mkdir()
for p in sorted(X.rglob('*')):
 if not p.is_file():continue
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink();h=sha(p);f=dict(path=str(p.relative_to(R)),bytes=s.st_size,sha256=h,mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns)
 if (h,s.st_size) in known:f['archive_source']=known[(h,s.st_size)]
 else:
  dest=G/'recovery-blobs'/(h+'.gz')
  if not dest.exists():dest.write_bytes(gzip.compress(p.read_bytes(),compresslevel=6,mtime=0))
  f['recovery_blob']=str(dest.relative_to(R))
 inventory.append(f)
assert sum(f['bytes'] for f in inventory)<=plan['temporary_budget_bytes'];write(G/'temporary-inventory.json',inventory)
report=dict(schema='crow-split-precision-executed-git-recovery-v1',restored_files=len(m['files']),all_original_raw_byte_exact_after_execution=True,native_submissions=14,scientific_reports_exact=2,binary_files_exact=binary_count,temporary_files=len(inventory),temporary_bytes=sum(f['bytes'] for f in inventory),closure_handle_check='Pending separate process after executor exits',limits='Exact scoped replay of two exposed algebraic problems and controls; not whole world or ecological qualification',goal_complete=False)
write(G/'verification.json',report);print(json.dumps(report),flush=True)
