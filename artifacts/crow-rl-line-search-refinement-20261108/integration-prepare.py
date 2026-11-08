from pathlib import Path
import hashlib,json,shutil,sys
R=Path.cwd();A=R/'artifacts/crow-rl-line-search-refinement-20261108';B=R/'build/crow-line-search-refinement-build-01'
sys.path.insert(0,str(R/'tools'));import crow_evidence_archive as archive
refdir=B/'reference';refdir.mkdir();runtime=B/'runtime';(runtime/'metalrobo').mkdir(parents=True)
for src,dest in [('build/lib/libmetalrobo.dylib',runtime/'libmetalrobo.dylib'),('build/shaders/MetalRobo.metallib',runtime/'metalrobo/MetalRobo.metallib')]:shutil.copy2(R/src,dest)
entries=archive.tree();recovered=[]
with archive.Objects() as objects:
 for path in ['artifacts/crow-rl-cap-curvature-20261108/payload/manifest.json','artifacts/crow-rl-quality-tolerances-20261108/payload/manifest.json']:
  m=json.loads(objects.read(entries[path][1]))
  for f in m['files']:
   src=f['path'];dest=None
   if 'cap-curvature' in path:
    if src.endswith('/reference/fixture-default'):dest=refdir/'fixture-default'
    if '/integration-observations/new-default/' in src and not src.endswith('runtime.json'):dest=refdir/'default-gold'/Path(src).name
    if src.endswith('/integration-observations/new-motor.tsv'):dest=refdir/'motor-gold.tsv'
    if src.endswith('/integration-observations/new-zero.tsv'):dest=refdir/'zero-gold.tsv'
   else:
    if src=='build/crow-quality-tolerances-build-01/crow-fixture':dest=refdir/'crow-fixture'
    if src.endswith('/strict-0-capture.bin'):dest=refdir/'strict-gold.bin'
   if dest:
    raw=b''.join(archive.content(f,entries,objects));assert hashlib.sha256(raw).hexdigest()==f['sha256']
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw);dest.chmod(f['mode'])
    recovered.append(dict(path=str(dest.relative_to(R)),sha256=f['sha256']))
assert len(list((refdir/'default-gold').iterdir()))==10 and len(recovered)==15
files={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for root in [runtime,refdir] for p in root.rglob('*') if p.is_file()}
for n in ['metalrobo_metal_unified_quality_probe','metalrobo_constraint_ir_probe','metalrobo_metal_world_contact_probe','metalrobo_motor_contact_probe']:
 p=R/'build/bin'/n;files[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
plan=dict(schema='crow-quality-refinement-integration-plan-v1',files=files,
 predictions=dict(owning_regressions_pass=True,motor_physical_records_exact=624,default_physical_files_exact=10,strict_capture_diff_only_hessian_stride=True,
 api_default_complete_640_twice=True,api_default_penetration_max_m=.0008,repeated_api_physical_outputs_exact=True),
 limits=dict(native_commands=10,per_command_seconds=180,total_seconds=900,raw_bytes=40000000),
 scope='First physical qualification of the production candidate. Same actions/geometry/cadence and API defaults as the rejected trial. No retry, profile/policy promotion or tuning after observation.')
(A/'integration-plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
