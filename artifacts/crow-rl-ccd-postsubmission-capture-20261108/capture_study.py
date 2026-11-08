"""One unchanged-recipe fresh execution to capture the known CCD failure."""
from pathlib import Path
import importlib.util,json,struct,subprocess
from metalrobo.crow_environment import execute,write,sha
root=Path.cwd();out=root/'.numi/runs/american-crow/ccd-failure-sweep-01'
out.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
base='artifacts/crow-rl-ccd-sphere-box-20261108';entries=a.tree()
with a.Objects() as objects:
 m=json.loads(objects.read(entries[base+'/runtime-recovery.json'][1]))
 manifest=json.loads(objects.read(entries[base+'/manifest.json'][1]))
 for name,source in [('initial.policypack',m['checkpoint']['input/initial.policypack']),
                     ('reference.safetensors',m['checkpoint']['reference/training/learner.safetensors'])]:
  f=next(f for f in manifest['files'] if f['path']==source)
  with (out/name).open('xb') as stream:
   for block in a.content(f,entries,objects):stream.write(block)
  (out/name).chmod(f['mode']);assert sha(out/name)==f['sha256']
control=root/'.numi/runs/american-crow/ccd-capture-control-01/summary.json'
assert json.loads(control.read_text())['status']=='passed'
d=out/'training';d.mkdir();c=list(m['training_reference']['command'])
for flag in ('--updated-policy-pack','--deployment-policy-pack','--incumbent-policy-pack',
             '--rollout-pack','--output-learner-state','--checkpoint-directory'):
 c[c.index(flag)+1]=str(d/Path(c[c.index(flag)+1]).name)
c[c.index('--policy-pack')+1]=str(out/'initial.policypack')
c.insert(2,'METALROBO_CCD_FAILURE_CAPTURE='+str(out/'failure.ccdsweep'))
write(out/'plan.json',{
 'question':'Which native projected sweep/event state causes the known head-box/sphere convergence failure?',
 'model':'Failure-only host inspection preserves all physics/learning computation. A fresh identical seeded process must reproduce the exact declared native failure and all partial canonical training artifacts and model/Adam arrays.',
 'prediction':'Same lane26/control5/substep14/pair100 error after13updates; snapshot header binds that boundary and is<=1MiB. Later isolated replay must match the stored candidate before the snapshot can explain the failure.',
 'refutation':'Any different trajectory, partial model or error invalidates capture equivalence. A captured buffer alone is not proof of the failing sweep.',
 'prior_evidence':base,'source_commit':a.git('rev-parse','HEAD').decode().strip(),
 'runtime':{p:sha(root/p) for p in ('build/bin/metalrobo_task_train','build/lib/libmetalrobo.dylib','build/shaders/MetalRobo.metallib','build/shaders/RetinalStream.metallib')},
 'capture_source_sha256':sha(root/'src/metal/CCDFailureCapture.inc'),'script_sha256':sha(Path(__file__)),
 'initial_policy_sha256':sha(out/'initial.policypack'),'success_path_control_sha256':sha(control),
 'fixed_recipe':'Original unbiased policy and development seeds,128lanes,160ticks,20plannedupdates,LR.0003,gamma.999,lambda1,two full-batch steps,32physics substeps. No repair, LR tuning, held-out evaluation or policy selection.',
 'budget':'One original20-update command timeout900s, expected to fail after13completedupdates. No retry. Study raw<=512MiB; snapshot<=1MiB; no video.',
 'command':c})
try:
 try:
  execute(out,'native',c,root,900)
 except RuntimeError:
  receipt=json.loads((out/'native.receipt.json').read_text())
  if receipt['exit_code']!=1 or receipt['timed_out'] or receipt['interrupted']:raise
 else:raise RuntimeError('Expected native failure did not recur')
 stderr=(out/'native.stderr').read_text()
 assert stderr.splitlines()[-1]==m['training_reference']['error_line']
 loaded=[s for s in stderr.splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s]
 assert loaded and all(s.endswith(str(root/'build/lib/libmetalrobo.dylib')) for s in loaded)
 data=(out/'failure.ccdsweep').read_bytes();assert len(data)<=1024**2
 assert data[:8]==b'CCDSNP01' and struct.unpack_from('<6I',data,8)==(1,26,5,14,100,9)
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
 write(out/'summary.json',{'status':'expected_failure_captured','native_failure_exact':True,
  'canonical_training_files_exact':canonical,'model_Adam_arrays_exact':len(x),'model_Adam_metadata_exact':True,
  'snapshot_bytes':len(data),'snapshot_sha256':sha(out/'failure.ccdsweep'),'raw_bytes_before_summary':raw,
  'isolated_sweep_replay_qualified':False,'scientific_run_retry_for_success':False,
  'scope':'Equivalent diagnostic execution of a known failure. No physics repair or learned behavior qualification.'})
 print((out/'summary.json').read_text())
except Exception as exc:
 write(out/'failure.json',{'type':type(exc).__name__,'error':str(exc)});raise
