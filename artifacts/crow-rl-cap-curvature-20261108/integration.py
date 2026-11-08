from pathlib import Path
import csv,hashlib,json,os,shutil,subprocess,sys,time
R=Path.cwd();A=R/'artifacts/crow-rl-cap-curvature-20261108';B=R/'build/crow-cap-curvature-build-01'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def run(name,cmd,runtime=None,expected=0,capture=None,timeout=120):
 O=B/'integration-observations';O.mkdir(exist_ok=True);env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
 for k in ['METALROBO_QUALITY_FAILURE_CAPTURE','METALROBO_BOUNDED_FAILURE_CAPTURE','METALROBO_CCD_FAILURE_CAPTURE','METALROBO_METALLIB']:env.pop(k,None)
 if runtime:env.update(DYLD_LIBRARY_PATH=str(runtime),DYLD_PRINT_LIBRARIES='1',CROW_PROBE_METALLIB=str(runtime/'metalrobo/MetalRobo.metallib'))
 if capture:env['METALROBO_QUALITY_FAILURE_CAPTURE']=str(capture)
 t=time.monotonic()
 with (O/(name+'.stdout')).open('xb') as out,(O/(name+'.stderr')).open('xb') as err:
  try:code=subprocess.run([str(x) for x in cmd],cwd=R,env=env,stdout=out,stderr=err,timeout=timeout).returncode
  except subprocess.TimeoutExpired:code=None
 rec=dict(name=name,argv=[str(x) for x in cmd],returncode=code,seconds=time.monotonic()-t,runtime=str(runtime) if runtime else None)
 records=json.loads((A/'integration-execution.json').read_text()) if (A/'integration-execution.json').exists() else []
 if runtime:
  loads=[s for s in (O/(name+'.stderr')).read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s];rec['loader_lines']=loads;assert loads and all(s.endswith(str(runtime/'libmetalrobo.dylib')) for s in loads)
 records.append(rec);write(A/'integration-execution.json',records);print(json.dumps(rec),flush=True);assert code==expected,rec
if sys.argv[1]=='prepare':
 assert json.loads((A/'result.json').read_text())['all_predictions_supported']
 (B/'reference').mkdir();sys.path.insert(0,str(R/'tools'));from crow_evidence_archive import tree,Objects,content
 entries=tree()
 with Objects() as objects:
  m=json.loads(objects.read(entries['artifacts/crow-rl-quality-first-failure-20261108/payload/manifest.json'][1]))
  for name in ['crow-fixture','fixture-default']:
   match=[f for f in m['files'] if f['path'].endswith('/reference/build/crow-native-drive-ir-build-01/'+name)];assert len(match)==1
   f=match[0];raw=b''.join(content(f,entries,objects));assert len(raw)==f['bytes'] and hashlib.sha256(raw).hexdigest()==f['sha256'];p=B/'reference'/name;p.write_bytes(raw);p.chmod(f['mode'])
 old=B/'runtime';(old/'metalrobo').mkdir();shutil.copy2(R/'build/shaders/MetalRobo.metallib',old/'metalrobo/MetalRobo.metallib');shutil.copy2(R/'build/bin/metalrobo_motor_contact_probe',old/'motor-probe')
 p=R/'apps/metal_unified_quality_probe.cpp';source=p.read_text();(B/'original-owner-probe.cpp').write_text(source)
 needle='} // namespace\n\nint main()';assert source.count(needle)==1
 candidate=source.replace(needle,(A/'cap_regression.cpp.inc').read_text()+needle).replace('        constexpr std::size_t environments = 2u;','        const float cappedOptimality = verifyCappedNewtonClosure();\n        constexpr std::size_t environments = 2u;').replace('            << " allocated_bytes="','            << " capped_newton_optimality=" << cappedOptimality\n            << " allocated_bytes="')
 (B/'candidate-owner-probe.cpp').write_text(candidate)
 plan=dict(schema='crow-cap-curvature-production-integration-v1',baseline_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_changes=['src/metal/MetalUnifiedQualitySolver.metal','apps/metal_unified_quality_probe.cpp'],owner_candidate_sha256=sha(B/'fixed/MetalUnifiedQualitySolver.metal'),probe_original_sha256=sha(p),probe_candidate_sha256=sha(B/'candidate-owner-probe.cpp'),integration_script_sha256=sha(A/'integration.py'),helper_sha256=sha(A/'cap_regression.cpp.inc'),baseline_runtime={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in [old/'libmetalrobo.dylib',old/'metalrobo/MetalRobo.metallib',old/'motor-probe',B/'reference/crow-fixture',B/'reference/fixture-default']},predictions=dict(full_release_pass=True,permanent_capped_closure_direct_pcg_pass=True,motor_physical_records_exact=576,zero_gain_physical_records_exact=48,default_completed_transitions=640,default_physical_files_exact=10,coupled_rejection_capture_exact=True,protected_policy_unchanged=True,ir_and_contact_owners_pass=True),budget=dict(build_seconds=300,native_seconds_each=120,raw_bytes_max=50000000),limits='Production capped derivative correctness fix. No precision-mode or acceptance-tolerance change. Coupled uncapped crow is expected to retain its rejection; broad environment remains unfinished.')
 write(A/'integration-plan.json',plan);print(json.dumps(plan));sys.exit(0)
plan=json.loads((A/'integration-plan.json').read_text());assert sha(A/'integration.py')==plan['integration_script_sha256']
for name,f in plan['baseline_runtime'].items():assert sha(R/name)==f['sha256']
O=B/'integration-observations'
if sys.argv[1]=='baseline':
 old=B/'runtime'
 run('old-motor',[old/'motor-probe',old/'metalrobo/MetalRobo.metallib',O/'old-motor.tsv','--native-drive'],old)
 run('old-zero',[old/'motor-probe',old/'metalrobo/MetalRobo.metallib',O/'old-zero.tsv','--zero-gain'],old)
 run('old-default',[B/'reference/fixture-default',O/'old-default','0','1','32'],old)
 run('old-coupled',[B/'reference/crow-fixture',O/'old-coupled','0','1','32','2'],old,expected=1,capture=O/'old-coupled.bin')
 sys.exit(0)
assert sys.argv[1]=='integrate'
p=R/'src/metal/MetalUnifiedQualitySolver.metal';assert sha(p)==json.loads((A/'plan.json').read_text())['owner_sha256'];p.write_bytes((B/'fixed/MetalUnifiedQualitySolver.metal').read_bytes());assert sha(p)==plan['owner_candidate_sha256']
p=R/'apps/metal_unified_quality_probe.cpp';assert sha(p)==plan['probe_original_sha256'];p.write_bytes((B/'candidate-owner-probe.cpp').read_bytes());assert sha(p)==plan['probe_candidate_sha256']
run('full-release-build',['cmake','--build','build','--config','Release','-j','6'],timeout=300)
new=B/'integrated';new.mkdir();(new/'metalrobo').mkdir()
for src,dst in [('build/lib/libmetalrobo.dylib','libmetalrobo.dylib'),('build/shaders/MetalRobo.metallib','metalrobo/MetalRobo.metallib'),('build/bin/metalrobo_motor_contact_probe','motor-probe'),('build/bin/metalrobo_metal_unified_quality_probe','unified-probe'),('build/bin/metalrobo_constraint_ir_probe','ir-probe'),('build/bin/metalrobo_metal_world_contact_probe','contact-probe')]:shutil.copy2(R/src,new/dst)
bindings={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in new.rglob('*') if p.is_file()};write(A/'integrated-runtime.json',dict(files=bindings))
run('new-motor',[new/'motor-probe',new/'metalrobo/MetalRobo.metallib',O/'new-motor.tsv','--native-drive'],new)
run('new-zero',[new/'motor-probe',new/'metalrobo/MetalRobo.metallib',O/'new-zero.tsv','--zero-gain'],new)
run('new-default',[B/'reference/fixture-default',O/'new-default','0','1','32'],new)
run('new-coupled',[B/'reference/crow-fixture',O/'new-coupled','0','1','32','2'],new,expected=1,capture=O/'new-coupled.bin')
run('unified-owner',[new/'unified-probe'],new)
run('ir-owner',[new/'ir-probe'],new)
run('contact-owner',[new/'contact-probe',new/'metalrobo/MetalRobo.metallib'],new)
physical=[]
for kind,count in [('motor',576),('zero',48)]:
 old=list(csv.DictReader((O/('old-'+kind+'.tsv')).open(),delimiter='\t'));after=list(csv.DictReader((O/('new-'+kind+'.tsv')).open(),delimiter='\t'));assert len(old)==len(after)==count
 excluded=['gpu_ms'] if 'gpu_ms' in old[0] else []
 assert [{k:v for k,v in r.items() if k not in excluded} for r in old]==[{k:v for k,v in r.items() if k not in excluded} for r in after]
 physical.append(dict(case=kind,records=count,physical_fields_exact=True,excluded_fields=excluded))
default=[]
for p in sorted((O/'old-default').iterdir()):
 if p.is_file() and p.name!='runtime.json':
  assert p.read_bytes()==(O/'new-default'/p.name).read_bytes();default.append(dict(name=p.name,byte_exact=True,sha256=sha(p)))
assert len(default)==10 and len(list(csv.DictReader((O/'new-default/trace.tsv').open(),delimiter='\t')))==640
assert (O/'old-coupled.bin').read_bytes()==(O/'new-coupled.bin').read_bytes()
policy=sha(R/'environments/american-crow/action-holds-v1/policies/deployment97.policypack');assert policy=='6a009417eac3e04831c95f46c24e78f943e825892124b972a58b5ee9eeed0389'
size=sum(p.stat().st_size for p in B.rglob('*') if p.is_file());assert size<=plan['budget']['raw_bytes_max']
report=dict(schema='crow-cap-curvature-production-integration-result-v1',full_release_pass=True,permanent_capped_closure_direct_pcg_pass=True,physical=physical,default_physical_files=default,default_completed_transitions=640,coupled_rejection_capture_exact=True,protected_policy_sha256=policy,ir_and_contact_owners_pass=True,raw_bytes=size,current_runtime={k:sha(R/k) for k in ['build/lib/libmetalrobo.dylib','build/shaders/MetalRobo.metallib','build/bin/metalrobo_motor_contact_probe','build/bin/metalrobo_metal_unified_quality_probe']},production_source_sha256={name:sha(R/name) for name in plan['source_changes']},goal_complete=False,limits=plan['limits'])
write(A/'integration-result.json',report);print(json.dumps(report),flush=True)
