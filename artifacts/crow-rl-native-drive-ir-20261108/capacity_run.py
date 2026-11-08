from pathlib import Path
import csv,hashlib,json,os,shutil,subprocess,sys,time
R=Path.cwd();A=R/'artifacts/crow-rl-native-drive-ir-20261108';B=R/'build/crow-native-drive-ir-build-01';T=R/'.numi/retained-runtime/native-drive-ir-capacity-20261108'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
if '--prepare' in sys.argv:
 T.mkdir(parents=True,exist_ok=False);(T/'metalrobo').mkdir()
 for src,dest in [('build/bin/metalrobo_motor_contact_probe','probe'),('build/bin/metalrobo_metal_world_contact_probe','contact-probe'),('build/bin/metalrobo_constraint_ir_probe','ir-probe'),('build/lib/libmetalrobo.dylib','libmetalrobo.dylib'),('build/shaders/MetalRobo.metallib','metalrobo/MetalRobo.metallib')]:shutil.copy2(R/src,T/dest)
 commands=[dict(name='capacity-host',args=[str(T/'probe'),'--native-drive-calibrate']),dict(name='capacity-ir-owner',args=[str(T/'ir-probe')]),dict(name='capacity-contact-owner',args=[str(T/'contact-probe'),str(T/'metalrobo/MetalRobo.metallib')]),dict(name='capacity-default-crow',args=[str(B/'fixture-default'),str(B/'default-crow-capacity'),'0','1','32']),dict(name='capacity-native-replay',args=[str(T/'probe'),str(T/'metalrobo/MetalRobo.metallib'),str(B/'native-capacity.tsv'),'--native-drive']),dict(name='capacity-crow-candidate',args=[str(B/'crow-fixture'),str(B/'crow-mode2-capacity'),'0','1','32','2'])]
 paths=[A/'capacity-plan.json',A/'capacity_run.py',A/'crow-fixture.mm',B/'fixture-default',B/'crow-fixture',R/'src/metal/MetalWorld.mm',R/'apps/motor_contact_probe.cpp']+[p for p in T.rglob('*') if p.is_file()]
 write(A/'capacity-execution-plan.json',dict(commands=commands,source_and_runtime_sha256={str(p.relative_to(R)):sha(p) for p in paths},timeout_seconds_each=120));raise SystemExit()
p=json.loads((A/'capacity-execution-plan.json').read_text())
for f,h in p['source_and_runtime_sha256'].items():assert sha(R/f)==h,f
env=os.environ.copy();env.update(DYLD_LIBRARY_PATH=str(T),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(T/'metalrobo/MetalRobo.metallib'))
for k in ['METALROBO_BOUNDED_FAILURE_CAPTURE','METALROBO_CCD_FAILURE_CAPTURE','METALROBO_METALLIB']:env.pop(k,None)
commands=[];comparisons=[]
for c in p['commands']:
 start=time.monotonic()
 with (B/(c['name']+'.stdout')).open('xb') as out,(B/(c['name']+'.stderr')).open('xb') as err:
  try:code=subprocess.run(c['args'],env=env,stdout=out,stderr=err,timeout=p['timeout_seconds_each']).returncode
  except subprocess.TimeoutExpired:code=None
 loads=[s for s in (B/(c['name']+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
 rec=dict(c,exit_code=code,seconds=time.monotonic()-start,loader_lines=loads);commands.append(rec);write(A/'capacity-execution.json',commands);print(json.dumps(rec),flush=True)
 assert loads and all(s.endswith(str(T/'libmetalrobo.dylib')) for s in loads)
 if c['name']=='capacity-crow-candidate':
  D=B/'crow-mode2-capacity';rows=list(csv.DictReader((D/'trace.tsv').open(),delimiter='\t')) if (D/'trace.tsv').exists() else []
  failure=json.loads((D/'failure.json').read_text()) if (D/'failure.json').exists() else None
  terminal=json.loads((D/'terminal.json').read_text()) if (D/'terminal.json').exists() else None
  maxpen=max([max(0,-float(r['upper_gap_m']),-float(r['lower_gap_m'])) for r in rows],default=None)
  initial={name:dict(present=(D/name).exists(),byte_exact=(D/name).read_bytes()==(B/'crow-mode0'/name).read_bytes() if (D/name).exists() else False) for name in ['initial-q.f32','initial-v.f32','initial-scene.bin','initial-geometry.json']}
  passed=code==0 and len(rows)==640 and maxpen is not None and maxpen<=.0008 and not failure and not terminal and all(f['byte_exact'] for f in initial.values())
  report=dict(verdict='supported_engineering_gate' if passed else 'contradicted',candidate_exit_code=code,candidate_trace_rows=len(rows),maximum_recorded_penetration_m=maxpen,native_failure=failure,terminal=terminal,initial_physical_files=initial,diagnostic=[s for s in (B/(c['name']+'.stderr')).read_text().splitlines() if not s.startswith('dyld[')][-8:],default_physical_files_exact=10,native_replay_conditions=576,host_template_checks=10,policy_or_profile_promoted=False,limits='Capacity-only engineering correction. Hybrid CCD remains rejected. No new scientific qualification or tolerance change.')
  write(A/'capacity-analysis.json',report);print(json.dumps(report),flush=True)
  assert code in (0,1),rec
 else:
  assert code==0,rec
  if c['name']=='capacity-host':assert json.loads((B/(c['name']+'.stdout')).read_text())['host_checks']==10
  if c['name']=='capacity-default-crow':
   for ref in sorted((B/'default-crow').iterdir()):
    if ref.is_file() and ref.name!='runtime.json':
     actual=B/'default-crow-capacity'/ref.name;equal=actual.read_bytes()==ref.read_bytes();comparisons.append(dict(name=ref.name,byte_exact=equal,sha256=sha(actual)));assert equal,ref.name
   assert len(comparisons)==10;write(A/'capacity-default-comparisons.json',comparisons)
  if c['name']=='capacity-native-replay':
   old=list(csv.DictReader((B/'native.tsv').open(),delimiter='\t'));new=list(csv.DictReader((B/'native-capacity.tsv').open(),delimiter='\t'));assert len(old)==len(new)==576
   for a,b in zip(old,new):assert {k:v for k,v in a.items() if k!='gpu_ms'}=={k:v for k,v in b.items() if k!='gpu_ms'},(a,b)
   write(A/'capacity-native-comparison.json',dict(physical_records_byte_exact=576,excluded_columns=['gpu_ms'],original_and_replayed_timings_retained=True))
