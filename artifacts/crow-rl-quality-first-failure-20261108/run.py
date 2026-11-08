from pathlib import Path
import csv,hashlib,json,os,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-rl-quality-first-failure-20261108';B=R/'build/crow-quality-first-failure-build-01';T=R/'.numi/retained-runtime/quality-first-failure-20261108'
REF=B/'reference/build/crow-native-drive-ir-build-01';OLD=B/'reference/.numi/retained-runtime/native-drive-ir-zero-gain-20261108'
records=[];checks=[]
def write(p,v):p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def execute(name,args,expected,capture=None,shader=None,library=True):
 env=os.environ.copy()
 for k in ['METALROBO_QUALITY_FAILURE_CAPTURE','METALROBO_CCD_FAILURE_CAPTURE','METALROBO_BOUNDED_FAILURE_CAPTURE','METALROBO_METALLIB']:env.pop(k,None)
 env.update(DYLD_LIBRARY_PATH=str(T),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(shader or T/'metalrobo/MetalRobo.metallib'))
 if capture is not None:env['METALROBO_QUALITY_FAILURE_CAPTURE']=str(capture)
 start=time.monotonic()
 with (B/(name+'.stdout')).open('xb') as out,(B/(name+'.stderr')).open('xb') as err:
  try:code=subprocess.run([str(x) for x in args],env=env,stdout=out,stderr=err,timeout=120).returncode
  except subprocess.TimeoutExpired:code=None
 loads=[s for s in (B/(name+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
 rec=dict(name=name,command=[str(x) for x in args],capture=str(capture),expected_exit=expected,exit_code=code,seconds=time.monotonic()-start,loader_lines=loads)
 records.append(rec);write(A/'execution.json',records);print(json.dumps(rec),flush=True)
 assert code==expected,rec
 if library:assert loads and all(x.endswith(str(T/'libmetalrobo.dylib')) for x in loads),rec
def exact(p,q):
 rec=dict(actual=str(p.relative_to(R)),reference=str(q.relative_to(R)),byte_exact=p.read_bytes()==q.read_bytes(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 checks.append(rec);write(A/'comparisons.json',checks);assert rec['byte_exact'],rec
def directory(p,q):
 for f in sorted(q.iterdir()):
  if f.is_file() and f.name!='runtime.json':exact(p/f.name,f)
shader=T/'metalrobo/MetalRobo.metallib'
execute('ir-owner',[T/'ir-probe'],0)
execute('contact-owner',[T/'contact-probe',shader],0)
execute('native-576',[T/'probe',shader,B/'native.tsv','--native-drive'],0,B/'accepted-must-not-capture.bin')
assert not (B/'accepted-must-not-capture.bin').exists()
old=list(csv.DictReader((REF/'native-zero.tsv').open(),delimiter='\t'));new=list(csv.DictReader((B/'native.tsv').open(),delimiter='\t'))
assert len(old)==len(new)==576
assert [{k:v for k,v in r.items() if k!='gpu_ms'} for r in old]==[{k:v for k,v in r.items() if k!='gpu_ms'} for r in new]
write(A/'native-physical-comparison.json',dict(physical_records_exact=576,excluded=['gpu_ms'],timings_retained=True))
execute('zero-48',[T/'probe',shader,B/'zero.tsv','--zero-gain'],0,B/'zero-must-not-capture.bin')
assert not (B/'zero-must-not-capture.bin').exists();exact(B/'zero.tsv',REF/'zero-new.tsv')
execute('capture-off',[REF/'crow-fixture',B/'crow-off','0','1','32','2'],1)
directory(B/'crow-off',REF/'crow-mode2-zero')
execute('capture-one',[REF/'crow-fixture',B/'crow-one','0','1','32','2'],1,B/'capture-one.bin')
directory(B/'crow-one',REF/'crow-mode2-zero')
assert (B/'capture-one.bin').is_file()
execute('capture-eight',[REF/'crow-fixture',B/'crow-eight','0','8','32','2'],1,B/'capture-eight.bin')
exact(B/'capture-eight.bin',B/'capture-one.bin')
execute('old-shader-rejected',[REF/'crow-fixture',B/'crow-old-shader','0','1','32','2'],1,B/'old-must-not-capture.bin',OLD/'metalrobo/MetalRobo.metallib')
assert not (B/'old-must-not-capture.bin').exists()
assert 'mr_unified_quality_solve_queued_capture' in (B/'old-shader-rejected.stderr').read_text()
execute('default-crow',[REF/'fixture-default',B/'default-crow','0','1','32'],0,B/'default-must-not-capture.bin')
assert not (B/'default-must-not-capture.bin').exists();directory(B/'default-crow',REF/'default-crow-zero')
execute('isolated-replay',[B/'replay',B/'capture-one.bin',shader,B/'replay.bin'],0,library=False)
exact(B/'replay.bin',B/'capture-one.bin')
write(A/'instrument-analysis.json',dict(verdict='supported_scoped_instrument',native_commands=len(records),byte_exact_comparisons=len(checks),physical_records_compared_excluding_gpu_timing=576,captured_bytes=(B/'capture-one.bin').stat().st_size,protected_first_failure_survives_eight_controls=True,isolated_full_record_exact=True,policy_or_profile_promoted=False,goal_complete=False))
print((A/'instrument-analysis.json').read_text(),flush=True)
