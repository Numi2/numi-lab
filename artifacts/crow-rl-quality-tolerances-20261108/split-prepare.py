from pathlib import Path
import hashlib,json,sys
R=Path.cwd();A=R/'artifacts/crow-rl-quality-tolerances-20261108';B=R/'build/crow-quality-tolerances-build-01'
sys.path.insert(0,str(R/'tools'));import crow_evidence_archive as archive
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
entries=archive.tree();dest=B/'split-runtime';dest.mkdir()
with archive.Objects() as objects:
 manifest=json.loads(objects.read(entries['artifacts/crow-rl-split-cones-20261108/payload/manifest.json'][1]))
 for name in ['probe','solver.metallib']:
  source='build/crow-split-cones-build-01/'+name;matches=[f for f in manifest['files'] if f['path']==source];assert len(matches)==1
  raw=b''.join(archive.content(matches[0],entries,objects));assert hashlib.sha256(raw).hexdigest()==matches[0]['sha256']
  p=dest/name;p.write_bytes(raw);p.chmod(matches[0]['mode'])
files={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(dest.iterdir())}
inputs={arm:dict(path='build/crow-quality-tolerances-build-01/analysis/'+arm+'-split-input.bin',sha256=sha(B/'analysis'/(arm+'-split-input.bin'))) for arm in ['api-default','opt6','opt7']}
write(A/'split-plan.json',dict(schema='crow-quality-tolerances-retained-split-plan-v1',files=files,inputs=inputs,repeats=2,
 predictions=dict(native_and_independent_residual_max=1e-10,velocity_reference_abs_error_max=2e-8,rounded_satisfies_original_declared_optimality=True,repeated_binary_outputs_exact=True),
 limits=dict(native_commands=6,per_command_seconds=30,total_seconds=180),
 scope='Post-observation numerical diagnosis using the previously sealed native solver, not a new physical trajectory or profile promotion. Expanded state is independently checked; rounded state must meet the original captured tolerance.'))
