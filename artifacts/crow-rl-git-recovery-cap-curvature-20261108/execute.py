from pathlib import Path
import csv,gzip,hashlib,json,os,stat,subprocess,sys,time
R=Path.cwd();A=R/'artifacts/crow-rl-cap-curvature-20261108';G=R/'artifacts/crow-rl-git-recovery-cap-curvature-20261108';X=R/'build/crow-cap-curvature-git-recovery-01'
sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects,content
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=json.loads((G/'plan.json').read_text());entries=tree()
with Objects() as objects:
 def read(name):return objects.read(entries[name][1])
 raw=read(plan['manifest']);assert hashlib.sha256(raw).hexdigest()==plan['manifest_sha256'];m=json.loads(raw);assert m['summary']['source_bytes']<=plan['restore_max_bytes'];X.mkdir()
 for f in m['files']:
  p=X/'original'/f['path'];p.parent.mkdir(parents=True,exist_ok=True);raw=b''.join(content(f,entries,objects));assert len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256'];p.write_bytes(raw);p.chmod(f['mode']);os.utime(p,ns=(f['mtime_ns'],f['mtime_ns']))
 for name,(mode,oid) in entries.items():
  if name.startswith(str(A.relative_to(R))+'/') and '/payload/' not in name:
   p=X/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(objects.read(oid));p.chmod(int(mode,8)&0o777)
 P=json.loads(read(str(A.relative_to(R)/'plan.json')));runtime=json.loads(read(str(A.relative_to(R)/'runtime.json')))
 paths=set(P['inputs'])|set(runtime['files'])
 originalB=X/'original'/'build/crow-cap-curvature-build-01'
 for sub in ['runtime','integrated','reference']:
  paths.update(str(p.relative_to(X/'original')) for p in (originalB/sub).rglob('*') if p.is_file())
 for name in paths:
  src=X/'original'/name;p=X/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(src.read_bytes());p.chmod(stat.S_IMODE(src.stat().st_mode))
print(json.dumps(dict(restored_files=len(m['files']),raw_bytes=m['summary']['source_bytes'],working_inputs=len(paths))),flush=True)
B=X/'build/crow-cap-curvature-build-01';Y=X/'world-replay';Y.mkdir();execution=[];comparisons=[]
def run(name,cmd,cwd=X,runtime=None,expected=0,capture=None):
 env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
 for k in ['METALROBO_QUALITY_FAILURE_CAPTURE','METALROBO_BOUNDED_FAILURE_CAPTURE','METALROBO_CCD_FAILURE_CAPTURE','METALROBO_METALLIB']:env.pop(k,None)
 if runtime:env.update(DYLD_LIBRARY_PATH=str(runtime),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(runtime/'metalrobo/MetalRobo.metallib'))
 if capture:env['METALROBO_QUALITY_FAILURE_CAPTURE']=str(capture)
 start=time.monotonic()
 with (Y/(name+'.stdout')).open('xb') as out,(Y/(name+'.stderr')).open('xb') as err:
  try:code=subprocess.run([str(x) for x in cmd],cwd=cwd,env=env,stdout=out,stderr=err,timeout=120).returncode
  except subprocess.TimeoutExpired:code=None
 rec=dict(name=name,argv=[str(x) for x in cmd],returncode=code,seconds=time.monotonic()-start)
 if runtime:
  loads=[s for s in (Y/(name+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s];rec['loader_lines']=loads;assert loads and all(s.endswith(str(runtime/'libmetalrobo.dylib')) for s in loads)
 execution.append(rec);write(G/'execution.json',execution);print(json.dumps(rec),flush=True);assert code==expected
run('paired-constitutive-replay',[R/'python/.venv/bin/python',X/A.relative_to(R)/'run.py'])
with Objects() as objects:
 for name in ['projection-analysis.json','closure-analysis.json','result.json']:
  expected=json.loads(objects.read(entries[str(A.relative_to(R)/name)][1]));actual=json.loads((X/A.relative_to(R)/name).read_text());assert expected==actual,name;comparisons.append(dict(report=name,all_scientific_fields_exact=True))
for name in P['order']:
 for suffix in ['.f32','.jsonl']:
  p=B/'observations'/(name+suffix);old=originalB/'observations'/(name+suffix);assert p.read_bytes()==old.read_bytes();comparisons.append(dict(constitutive_output=name+suffix,byte_exact=True,sha256=sha(p)))
O=originalB/'integration-observations'
for arm,sub in [('old','runtime'),('new','integrated')]:
 runtime=B/sub
 run(arm+'-motor',[runtime/'motor-probe',runtime/'metalrobo/MetalRobo.metallib',Y/(arm+'-motor.tsv'),'--native-drive'],runtime=runtime)
 run(arm+'-zero',[runtime/'motor-probe',runtime/'metalrobo/MetalRobo.metallib',Y/(arm+'-zero.tsv'),'--zero-gain'],runtime=runtime)
 run(arm+'-default',[B/'reference/fixture-default',Y/(arm+'-default'),'0','1','32'],runtime=runtime)
 run(arm+'-coupled',[B/'reference/crow-fixture',Y/(arm+'-coupled'),'0','1','32','2'],runtime=runtime,expected=1,capture=Y/(arm+'-coupled.bin'))
 for kind,count in [('motor',576),('zero',48)]:
  expected=list(csv.DictReader((O/(arm+'-'+kind+'.tsv')).open(),delimiter='\t'));actual=list(csv.DictReader((Y/(arm+'-'+kind+'.tsv')).open(),delimiter='\t'));assert len(expected)==len(actual)==count
  strip=lambda rows:[{k:v for k,v in r.items() if k!='gpu_ms'} for r in rows]
  assert strip(expected)==strip(actual);comparisons.append(dict(case=arm+'-'+kind,records=count,physical_fields_exact=True,excluded_fields=['gpu_ms']))
 for p in sorted((O/(arm+'-default')).iterdir()):
  if p.is_file() and p.name!='runtime.json':
   actual=Y/(arm+'-default')/p.name;assert actual.read_bytes()==p.read_bytes();comparisons.append(dict(case=arm+'-default',file=p.name,byte_exact=True,sha256=sha(actual)))
 assert (Y/(arm+'-coupled.bin')).read_bytes()==(O/(arm+'-coupled.bin')).read_bytes();comparisons.append(dict(case=arm+'-coupled',capture_exact=True,sha256=sha(Y/(arm+'-coupled.bin'))))
new=B/'integrated'
for name,exe,args in [('unified-owner','unified-probe',[]),('ir-owner','ir-probe',[]),('contact-owner','contact-probe',[new/'metalrobo/MetalRobo.metallib'])]:
 run(name,[new/exe]+args,runtime=new)
 actual=(Y/(name+'.stdout')).read_text();expected=(O/(name+'.stdout')).read_text()
 if name!='contact-owner':assert actual==expected;comparisons.append(dict(owner=name,scientific_stdout_exact=True))
 else:
  excluded=set(plan['contact_owner_excluded_fields'])
  parse=lambda t:{k:v for token in t.split() if '=' in token for k,v in [token.split('=',1)] if k not in excluded}
  assert parse(actual)==parse(expected);comparisons.append(dict(owner=name,physical_fields_exact=True,excluded_fields=sorted(excluded),fields=parse(actual)))
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
size=sum(f['bytes'] for f in inventory);assert size<=plan['temporary_budget_bytes'];write(G/'temporary-inventory.json',inventory)
report=dict(schema='crow-cap-curvature-executed-git-recovery-v1',original_files_restored=len(m['files']),all_original_raw_byte_exact_after_execution=True,native_commands=19,scientific_reports_exact=3,constitutive_outputs_byte_exact=8,default_physical_outputs_byte_exact=20,coupled_failure_captures_exact=2,motor_records_physical_exact=1248,owning_native_probes_passed=3,comparison_records=len(comparisons),temporary_files=len(inventory),temporary_bytes=size,closure_handle_check='Pending separate process after executor exits',limits='Exact scoped numerical/physics regression replay; no independent ecological qualification',goal_complete=False)
write(G/'verification.json',report);print(json.dumps(report),flush=True)
