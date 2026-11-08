"""One unchanged-recipe fresh execution to capture the known CCD failure."""
from pathlib import Path
import importlib.util,json,struct,subprocess
from metalrobo.crow_environment import execute,write,sha
root=Path.cwd();out=root/'.numi/runs/american-crow/ccd-failure-sweep-01'
assert out.is_dir()
spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
base='artifacts/crow-rl-ccd-sphere-box-20261108';entries=a.tree()
with a.Objects() as objects:
 m=json.loads(objects.read(entries[base+'/runtime-recovery.json'][1]))
control=root/'.numi/runs/american-crow/ccd-capture-control-01/summary.json'
assert json.loads(control.read_text())['status']=='passed'
d=out/'training';c=list(m['training_reference']['command'])
for flag in ('--updated-policy-pack','--deployment-policy-pack','--incumbent-policy-pack',
             '--rollout-pack','--output-learner-state','--checkpoint-directory'):
 c[c.index(flag)+1]=str(d/Path(c[c.index(flag)+1]).name)
c[c.index('--policy-pack')+1]=str(out/'initial.policypack')
c.insert(2,'METALROBO_CCD_FAILURE_CAPTURE='+str(out/'failure.ccdsweep'))
write(out/'analysis-revision.json',{
 'native_execution_repeated':False,'original_plan_retained':True,
 'reason':'Original header check assumed global failing_substep14. The snapshot faithfully stores contact-status physicsSubstep13: rigid CCD resolution can fail before narrowphase stamps the current pass. The separate world transaction latch records current pass14. Verify both values and all retained artifacts without rerunning native execution.',
 'source_files':{p:sha(root/p) for p in ('src/metal/Collision.metal','src/metal/MetalWorldContact.metal','src/metal/CCDFailureCapture.inc')},
 'original_failure_sha256':sha(out/'failure.json'),'original_plan_sha256':sha(out/'plan.json'),'reader_sha256':sha(Path(__file__))})
try:
 receipt=json.loads((out/'native.receipt.json').read_text())
 assert receipt['exit_code']==1 and not receipt['timed_out'] and not receipt['interrupted']
 assert sha(out/'native.stderr')==receipt['stderr_sha256'] and sha(out/'native.json')==receipt['stdout_sha256']
 stderr=(out/'native.stderr').read_text()
 assert stderr.splitlines()[-1]==m['training_reference']['error_line']
 loaded=[s for s in stderr.splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
 assert loaded and all(s.endswith(str(root/'build/lib/libmetalrobo.dylib')) for s in loaded)
 data=(out/'failure.ccdsweep').read_bytes();assert len(data)<=1024**2
 assert data[:8]==b'CCDSNP01' and struct.unpack_from('<6I',data,8)==(1,26,5,13,100,9)
 import mlx.core as mx,numpy as np
 x,xm=mx.load(d/'learner.safetensors',return_metadata=True);y,ym=mx.load(out/'reference.safetensors',return_metadata=True)
 assert xm==ym and x.keys()==y.keys()
 for k in x:assert np.asarray(x[k]).tobytes()==np.asarray(y[k]).tobytes()
 canonical=[]
 for name,f in m['training_reference']['files'].items():
  if name=='learner.safetensors':continue
  assert sha(d/name)==f['sha256'];canonical.append(name)
 checkpoint=json.loads((d/'checkpoints/revision-00000011.trainingcheckpoint/manifest.json').read_text())
 assert checkpoint['completedUpdates']==10 and checkpoint['acceptedControlTicks']==[1600]*128
 raw=sum(p.stat().st_size for p in out.rglob('*') if p.is_file());assert raw<=512*1024**2
 write(out/'summary-v2.json',{'status':'expected_failure_captured','native_failure_exact':True,
  'canonical_training_files_exact':canonical,'model_Adam_arrays_exact':len(x),'model_Adam_metadata_exact':True,
  'contact_status_physics_substep':13,'world_transaction_failing_substep':14,'original_header_prediction':'contradicted','native_execution_repeated_for_reanalysis':False,'snapshot_bytes':len(data),'snapshot_sha256':sha(out/'failure.ccdsweep'),'raw_bytes_before_summary':raw,
  'isolated_sweep_replay_qualified':False,'scientific_run_retry_for_success':False,
  'scope':'Equivalent diagnostic execution of a known failure. No physics repair or learned behavior qualification.'})
 print((out/'summary-v2.json').read_text())
except Exception as exc:
 write(out/'reanalysis-failure.json',{'type':type(exc).__name__,'error':str(exc)});raise
