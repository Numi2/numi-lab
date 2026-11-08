from pathlib import Path
import csv,hashlib,json,os,shutil,subprocess,sys,time
R=Path.cwd();A=R/'artifacts/crow-rl-native-drive-ir-20261108';B=R/'build/crow-native-drive-ir-build-01';T=R/'.numi/retained-runtime/native-drive-ir-zero-gain-20261108'
old_shader=R/'.numi/retained-runtime/native-drive-ir-capacity-20261108/metalrobo/MetalRobo.metallib'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
if '--prepare' in sys.argv:
 T.mkdir(parents=True,exist_ok=False);(T/'metalrobo').mkdir()
 for src,dest in [('build/bin/metalrobo_motor_contact_probe','probe'),('build/bin/metalrobo_metal_world_contact_probe','contact-probe'),('build/bin/metalrobo_constraint_ir_probe','ir-probe'),('build/lib/libmetalrobo.dylib','libmetalrobo.dylib'),('build/shaders/MetalRobo.metallib','metalrobo/MetalRobo.metallib')]:shutil.copy2(R/src,T/dest)
 shader=str(T/'metalrobo/MetalRobo.metallib')
 commands=[dict(name='zero-host',args=[str(T/'probe'),'--native-drive-calibrate'],expected=0),dict(name='zero-ir-owner',args=[str(T/'ir-probe')],expected=0),dict(name='zero-contact-owner',args=[str(T/'contact-probe'),shader],expected=0),dict(name='zero-known-bad',args=[str(T/'probe'),str(old_shader),str(B/'zero-old.tsv'),'--zero-gain'],expected=1),dict(name='zero-repaired',args=[str(T/'probe'),shader,str(B/'zero-new.tsv'),'--zero-gain'],expected=0),dict(name='zero-native-replay',args=[str(T/'probe'),shader,str(B/'native-zero.tsv'),'--native-drive'],expected=0),dict(name='zero-default-crow',args=[str(B/'fixture-default'),str(B/'default-crow-zero'),'0','1','32'],expected=0),dict(name='zero-crow-replay',args=[str(B/'crow-fixture'),str(B/'crow-mode2-zero'),'0','1','32','2'],expected=1)]
 paths=[A/'zero-gain-plan.json',A/'zero_gain_run.py',B/'fixture-default',B/'crow-fixture',old_shader,R/'src/metal/BoundedDrive.metal',R/'apps/motor_contact_probe.cpp']+[p for p in T.rglob('*') if p.is_file()]
 write(A/'zero-gain-execution-plan.json',dict(commands=commands,source_and_runtime_sha256={str(p.relative_to(R)):sha(p) for p in paths},timeout_seconds_each=120));raise SystemExit()
p=json.loads((A/'zero-gain-execution-plan.json').read_text())
for f,h in p['source_and_runtime_sha256'].items():assert sha(R/f)==h,f
env=os.environ.copy();env.update(DYLD_LIBRARY_PATH=str(T),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(T/'metalrobo/MetalRobo.metallib'))
for k in ['METALROBO_BOUNDED_FAILURE_CAPTURE','METALROBO_CCD_FAILURE_CAPTURE','METALROBO_METALLIB']:env.pop(k,None)
commands=[]
for c in p['commands']:
 start=time.monotonic()
 with (B/(c['name']+'.stdout')).open('xb') as out,(B/(c['name']+'.stderr')).open('xb') as err:
  try:code=subprocess.run(c['args'],env=env,stdout=out,stderr=err,timeout=p['timeout_seconds_each']).returncode
  except subprocess.TimeoutExpired:code=None
 loads=[s for s in (B/(c['name']+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
 rec=dict(c,exit_code=code,seconds=time.monotonic()-start,loader_lines=loads);commands.append(rec);write(A/'zero-gain-execution.json',commands);print(json.dumps(rec),flush=True)
 assert code==c['expected'] and loads and all(s.endswith(str(T/'libmetalrobo.dylib')) for s in loads),rec
 if c['name']=='zero-known-bad':assert 'Native zero-gain physical gate failed' in (B/(c['name']+'.stderr')).read_text()
 if c['name']=='zero-native-replay':
  old=list(csv.DictReader((B/'native.tsv').open(),delimiter='\t'));new=list(csv.DictReader((B/'native-zero.tsv').open(),delimiter='\t'));assert len(old)==len(new)==576
  for a,b in zip(old,new):assert {k:v for k,v in a.items() if k!='gpu_ms'}=={k:v for k,v in b.items() if k!='gpu_ms'},(a,b)
 if c['name']=='zero-default-crow':
  refs=[f for f in (B/'default-crow').iterdir() if f.is_file() and f.name!='runtime.json'];assert len(refs)==10
  for f in refs:assert (B/'default-crow-zero'/f.name).read_bytes()==f.read_bytes(),f.name
 if c['name']=='zero-crow-replay':
  refs=[f for f in (B/'crow-mode2-capacity').iterdir() if f.is_file()];checks={f.name:(B/'crow-mode2-zero'/f.name).exists() and (B/'crow-mode2-zero'/f.name).read_bytes()==f.read_bytes() for f in refs};write(A/'zero-crow-comparisons.json',checks);assert all(checks.values()),checks
rows=list(csv.DictReader((B/'zero-new.tsv').open(),delimiter='\t'));bad=list(csv.DictReader((B/'zero-old.tsv').open(),delimiter='\t'));assert len(rows)==48 and len(bad)==9
report=dict(verdict='supported_finite_zero_gain_regression',accepted_zero_gain_conditions=48,old_shader_rejected_after_published_conditions=len(bad),maximum_force_error_N=max(float(r['force_error']) for r in rows),maximum_momentum_residual_Ns=max(abs(float(r['momentum_residual'])) for r in rows),maximum_q_hv_error_m=max(abs(float(r['q_residual'])) for r in rows),native_576_physical_records_exact=True,excluded_native_column=['gpu_ms'],default_crow_physical_files_exact=10,crow_candidate_first_step_failure_reproduced_exactly=True,policy_or_profile_promoted=False)
assert report['maximum_force_error_N']<=1e-7 and report['maximum_momentum_residual_Ns']<=2e-9 and report['maximum_q_hv_error_m']<=2e-9
write(A/'zero-gain-analysis.json',report);print(json.dumps(report),flush=True)
