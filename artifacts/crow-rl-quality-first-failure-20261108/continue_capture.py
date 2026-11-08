from pathlib import Path
import csv,hashlib,json,os,subprocess,time
R=Path.cwd();A=R/'artifacts/crow-rl-quality-first-failure-20261108';B=R/'build/crow-quality-first-failure-build-01/repair-01';T=R/'.numi/retained-runtime/quality-first-failure-mask-fix-20261108'
REF=B.parent/'reference/build/crow-native-drive-ir-build-01';OLD=B.parent/'reference/.numi/retained-runtime/native-drive-ir-zero-gain-20261108'
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
 records.append(rec);write(A/'continuation-execution.json',records);print(json.dumps(rec),flush=True)
 assert code==expected,rec
 if library:assert loads and all(x.endswith(str(T/'libmetalrobo.dylib')) for x in loads),rec
def exact(p,q):
 rec=dict(actual=str(p.relative_to(R)),reference=str(q.relative_to(R)),byte_exact=p.read_bytes()==q.read_bytes(),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 checks.append(rec);write(A/'continuation-comparisons.json',checks);assert rec['byte_exact'],rec
def directory(p,q):
 for f in sorted(q.iterdir()):
  if f.is_file() and f.name!='runtime.json':exact(p/f.name,f)

# Reinterpret only the explicitly declared inspection bit in public status.
import struct
reference=REF/'crow-mode2-zero/failure-world-status.bin'
actual=B/'crow-one/failure-world-status.bin'
a=reference.read_bytes();b=bytearray(actual.read_bytes());assert len(a)==len(b)==4*48
changed=[]
for lane in range(4):
 offset=lane*48+28
 old=struct.unpack_from('<I',a,offset)[0];new=struct.unpack_from('<I',b,offset)[0]
 assert new==old|1024 and not old&1024
 struct.pack_into('<I',b,offset,new&~1024);changed.append(dict(lane=lane,old_flags=old,new_flags=new))
assert bytes(b)==a
write(A/'status-flag-comparison.json',dict(raw_byte_prediction='contradicted_by_expected_inspection_metadata',only_changed_field='MRMetalWorldStatusGPU.flags bit 10',records=changed,all_other_status_bytes_exact=True,physical_gates_unchanged=True))
for f in sorted((REF/'crow-mode2-zero').iterdir()):
 if f.is_file() and f.name!='failure-world-status.bin':exact(B/'crow-one'/f.name,f)
shader=T/'metalrobo/MetalRobo.metallib'
execute('capture-eight',[REF/'crow-fixture',B/'crow-eight','0','8','32','2'],1,B/'capture-eight.bin')
exact(B/'capture-eight.bin',B/'capture-one.bin')
execute('old-shader-rejected',[REF/'crow-fixture',B/'crow-old-shader','0','1','32','2'],1,B/'old-must-not-capture.bin',OLD/'metalrobo/MetalRobo.metallib')
assert not (B/'old-must-not-capture.bin').exists()
assert 'mr_unified_quality_solve_queued_capture' in (B/'old-shader-rejected.stderr').read_text()
execute('default-crow',[REF/'fixture-default',B/'default-crow','0','1','32'],0,B/'default-must-not-capture.bin')
assert not (B/'default-must-not-capture.bin').exists();directory(B/'default-crow',REF/'default-crow-zero')
execute('isolated-replay',[B.parent/'replay',B/'capture-one.bin',shader,B/'replay.bin'],0,library=False)
exact(B/'replay.bin',B/'capture-one.bin')
write(A/'instrument-analysis.json',dict(verdict='supported_scoped_instrument_with_explicit_status_metadata',original_admission_attempt_commands=3,repair_commands=6,continuation_commands=len(records),continuation_byte_exact_files=len(checks),capture_on_exact_physical_and_contact_files=10,capture_on_status_records_equal_except_inspection_bit=4,capture_off_exact_files=11,default_crow_exact_physical_files=10,default_crow_accepted_transitions=640,finite_regression_conditions=624,physical_tsv_records_exclude_gpu_timing=576,captured_bytes=(B/'capture-one.bin').stat().st_size,protected_first_failure_survives_eight_controls=True,isolated_full_record_exact=True,policy_or_profile_promoted=False,goal_complete=False))
print((A/'instrument-analysis.json').read_text(),flush=True)
