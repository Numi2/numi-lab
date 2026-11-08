from pathlib import Path
import csv, hashlib, json, os, shutil, subprocess, sys, time
R=Path.cwd();A=R/'artifacts/crow-rl-native-drive-ir-20261108';B=R/'build/crow-native-drive-ir-build-01'
T=R/'.numi/retained-runtime/native-drive-ir-20261108'
sys.path.insert(0,str(R/'tools'))
from crow_evidence_archive import tree,Objects,content

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

if '--prepare' in sys.argv:
 T.mkdir(parents=True,exist_ok=False);(T/'metalrobo').mkdir()
 for source,dest in [('build/bin/metalrobo_motor_contact_probe','probe'),('build/bin/metalrobo_metal_world_contact_probe','contact-probe'),('build/lib/libmetalrobo.dylib','libmetalrobo.dylib'),('build/shaders/MetalRobo.metallib','metalrobo/MetalRobo.metallib')]:shutil.copy2(R/source,T/dest)
 previous='artifacts/crow-rl-motor-contact-20261108/payload/manifest.json';entries=tree()
 with Objects() as o:
  raw=o.read(entries[previous][1]);m=json.loads(raw);source=next(f for f in m['files'] if f['path']=='build/crow-motor-contact-build-01/fixture-default');data=b''.join(content(source,entries,o));assert hashlib.sha256(data).hexdigest()==source['sha256'];(B/'fixture-default').write_bytes(data);(B/'fixture-default').chmod(source['mode'])
 commands=[dict(name='host-template-calibration',args=[str(T/'probe'),'--native-drive-calibrate']),dict(name='default-contact-owner',args=[str(T/'contact-probe'),str(T/'metalrobo/MetalRobo.metallib')]),dict(name='default-crow-replay',args=[str(B/'fixture-default'),str(B/'default-crow'),'0','1','32']),dict(name='native-drive-conditions',args=[str(T/'probe'),str(T/'metalrobo/MetalRobo.metallib'),str(B/'native.tsv'),'--native-drive'])]
 paths=['include/metalrobo/constraint_ir_shared.h','include/metalrobo/ConstraintIR.hpp','include/metalrobo/MetalWorld.hpp','src/core/ConstraintIR.cpp','src/metal/MetalWorld.mm','src/metal/BoundedDrive.metal','apps/motor_contact_probe.cpp',str((A/'plan.json').relative_to(R)),str((A/'prepare_and_run.py').relative_to(R)),str((B/'fixture-default').relative_to(R))]+[str(p.relative_to(R)) for p in T.rglob('*') if p.is_file()]
 write(A/'execution-plan.json',dict(commands=commands,source_and_runtime_sha256={p:sha(R/p) for p in paths},dynamic_library=str(T/'libmetalrobo.dylib'),metallib=str(T/'metalrobo/MetalRobo.metallib'),default_reference_manifest=previous,default_reference_manifest_sha256=hashlib.sha256(raw).hexdigest(),timeout_seconds_each=120,stop='First unexpected native exit, loader binding, physical control comparison or validity gate. Preserve failures; no retry or threshold change.'))
 print(json.dumps({'prepared_runtime':str(T),'commands':len(commands),'raw_bytes':sum(p.stat().st_size for root in [B,T] for p in root.rglob('*') if p.is_file())}))
 raise SystemExit()

p=json.loads((A/'execution-plan.json').read_text())
for path,h in p['source_and_runtime_sha256'].items():assert sha(R/path)==h,path
env=os.environ.copy();env.update(DYLD_LIBRARY_PATH=str(T),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=p['metallib'])
for key in ['METALROBO_METALLIB','METALROBO_BOUNDED_FAILURE_CAPTURE','METALROBO_CCD_FAILURE_CAPTURE']:env.pop(key,None)
commands=[]
for c in p['commands']:
 start=time.monotonic()
 with (B/(c['name']+'.stdout')).open('xb') as out,(B/(c['name']+'.stderr')).open('xb') as err:
  try:code=subprocess.run(c['args'],env=env,stdout=out,stderr=err,timeout=p['timeout_seconds_each']).returncode
  except subprocess.TimeoutExpired:code=None
 loads=[s for s in (B/(c['name']+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
 rec=dict(c,exit_code=code,seconds=time.monotonic()-start,loader_lines=loads);commands.append(rec);write(A/'execution.json',commands);print(json.dumps(rec),flush=True)
 if code!=0:
  write(A/'analysis.json',dict(verdict='contradicted_native_acceptance_or_physical_validity',failed_command=c['name'],exit_code=code,diagnostic=(B/(c['name']+'.stderr')).read_text().splitlines()[-26:],policy_or_profile_promoted=False));raise SystemExit(1)
 assert loads and all(s.endswith(p['dynamic_library']) for s in loads)
 if c['name']=='default-crow-replay':
  entries=tree();checks={}
  with Objects() as o:
   raw=o.read(entries[p['default_reference_manifest']][1]);assert hashlib.sha256(raw).hexdigest()==p['default_reference_manifest_sha256'];m=json.loads(raw)
   for f in m['files']:
    if f['path'].startswith('build/crow-motor-contact-build-01/default-crow/') and f['path'].split('/')[-1]!='runtime.json':
     name=f['path'].split('/')[-1];actual=(B/'default-crow'/name).read_bytes();expected=b''.join(content(f,entries,o));checks[name]=dict(byte_exact=actual==expected,sha256=hashlib.sha256(actual).hexdigest());assert actual==expected,name
  assert len(checks)==10;write(A/'default-regression.json',checks)
rows=list(csv.DictReader((B/'native.tsv').open(),delimiter='\t'));assert len(rows)==576
report=dict(verdict='supported',accepted_native_conditions=len(rows),maximum_force_law_error_N=max(float(r['force_error']) for r in rows),maximum_momentum_residual_Ns=max(abs(float(r['momentum_residual'])) for r in rows),maximum_q_hv_error_m=max(abs(float(r['q_residual'])) for r in rows),default_crow_physical_files_byte_exact=10,host_template_checks=8,default_native_contact_owner_passed=True,policy_or_profile_promoted=False,limits=json.loads((A/'plan.json').read_text())['limits'])
assert report['maximum_force_law_error_N']<=1e-7 and report['maximum_momentum_residual_Ns']<=2e-9 and report['maximum_q_hv_error_m']<=2e-9
write(A/'analysis.json',report);print(json.dumps(report),flush=True)
