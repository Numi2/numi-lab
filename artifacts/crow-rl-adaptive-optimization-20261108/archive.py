from pathlib import Path
import hashlib,importlib.util,json,shutil,subprocess,sys
import mlx.core as mx
import numpy as np
r=Path.cwd();out=r/'artifacts/crow-rl-adaptive-optimization-20261108';runs=r/'.numi/runs/american-crow';trial=runs/'adaptive-optimization-02'
spec=importlib.util.spec_from_file_location('archive',r/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def record(p):return {'sha256':a.file_hash(p),'bytes':p.stat().st_size,'mode':p.stat().st_mode&0o777}
retained=r/'.numi/retained-runtime/adaptive-optimization-01'
for relative in ('bin/metalrobo_task_rollout','bin/metalrobo_task_train','lib/libmetalrobo.dylib','shaders/MetalRobo.metallib','shaders/GRUMemory.metallib','shaders/RetinalStream.metallib'):
 p=r/'build'/relative;q=retained/relative;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
old=json.loads((r/'artifacts/crow-rl-retinal-normalization-20261108/runtime-recovery.json').read_text());m={'manifests':old['manifests']+['artifacts/crow-rl-adaptive-optimization-20261108/manifest.json'],'native':{},'python_sources':{},'git_inputs':{},'checkpoint':{}}
for p in retained.rglob('*'):
 if p.is_file():m['native'][str(p.relative_to(retained))]={**record(p),'archive_source':str(p.relative_to(r))}
for p in (r/'python/metalrobo').rglob('*.py'):
 name=str(p.relative_to(r));m['python_sources'][name]={**record(p),'git_blob':a.git('rev-parse','HEAD:'+name).decode().strip()}
for p in (r/'environments/american-crow').rglob('*'):
 if p.is_file():
  name=str(p.relative_to(r));m['git_inputs']['project/'+name]={**record(p),'git_blob':a.git('rev-parse','HEAD:'+name).decode().strip()}
bundle=trial/'training/checkpoints/revision-00000081.trainingcheckpoint'
for p in bundle.iterdir():
 if p.is_file():m['checkpoint']['input/checkpoint/'+p.name]=str(p.relative_to(r))
m['training_command']=json.loads((runs/'adaptive-optimization-continuation-02/tail8.command.json').read_text());ref=runs/'adaptive-optimization-continuation-02/tail8/checkpoints/revision-00000082.trainingcheckpoint'
m['training_reference']={name:record(ref/name) for name in ('policy.policypack','candidate.policypack','rollout.rolloutpack')}
tensors,meta=mx.load(ref/'learner.safetensors',return_metadata=True);m['learner_reference']={'metadata':meta,'tensors':{k:{'dtype':str(np.asarray(v).dtype),'shape':list(v.shape),'sha256':hashlib.sha256(np.asarray(v).tobytes()).hexdigest()} for k,v in tensors.items()}}
m['training_native_sections']=json.loads((runs/'adaptive-optimization-continuation-02/summary.json').read_text())['native_sections_sha256']
m['recovery_scope']='Actual Git-restored one-update64-lane adaptive final81 training continuation; matching OS/device and Python dependencies required.'
analysis={'status':'adaptive_optimization_active_native_memory_still_unqualified','source_commit':a.git('rev-parse','HEAD').decode().strip(),'campaign':json.loads((trial/'summary.json').read_text()),'offline_mechanism':json.loads((trial/'diagnostic/summary.json').read_text()),'final_training_continuation':{k:v for k,v in json.loads((runs/'adaptive-optimization-continuation-02/summary.json').read_text()).items() if k!='records'},'learned_memory_qualified':False}
subprocess.run([sys.executable,'tools/crow_evidence_archive.py','capture',str(out.relative_to(r))],check=True)
(out/'runtime-recovery.json').write_text(json.dumps(m,indent=2)+'\n');(out/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
(out/'README.md').write_text("""# KL-adaptive crow optimization

Sixteen new independent64-lane320-tick lifetimes continue the Git-restored
normalized policy65 and Adam state with explicit initial learning-rate override.
Two PPO epochs, unchanged physical task, sensory semantics, action holds and
intake-only reward.327680 accepted transitions; no reported physics rejection.
Intake37/256 first lifetimes and56/256 last. Throughput3853 looptransitions/s.
This is promising training evidence; the blocks are not held-out evaluation.
Per-update KL/clipping/gradient/rate metrics are preserved. Last averaged
KL0.033 with~22% clipping means the controller is active but bounded optimality
or convergence is unproven. No larger campaign is justified by intake alone.

On identical last-life pre-choice native inputs, source65 head-side mean gap
is0.5103; adaptive81 is1.1406. Exchanging early RGB flips both gaps; a memory
cut at40 removes them. These offline algebraic controls are NOT physical
causal memory qualification. Native held-out intact/reset/absent/swapped cue
behavior and broader lifelong ecology/manipulation/social learning remain open.

The trainer originally rejected initialization-only optimizer overrides whenever
life checkpointing was enabled. A scoped guard now permits the LR override only
for new independent cued populations; full life restoration still rejects it.
The initialization-only flag is canonicalized to the legacy false contract entry;
actual optimizer configuration/state remain bound and restored without reoverride.
A startup failure before native execution is preserved. The first continuation
harness retained conflicting explicit input arguments and correctly rejected
before execution; it is preserved, followed by corrected controls.

Final81 freshly restores at chunks8/1 to exact82:53 model/Adam tensors and metadata,
policy/candidate/rollout bytes and all three native state components. Git recovery
actually repeats that future using restored exact runtime/source/product inputs,
audits the adjacent loaded library and verifies bytes before raw/duplicate cleanup.
No video or repeated training snapshots are generated.
""")
