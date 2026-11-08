from pathlib import Path
import hashlib,importlib.util,json,shutil,subprocess,sys
import mlx.core as mx
import numpy as np
r=Path.cwd();out=r/'artifacts/crow-rl-precision-training-20261108';runs=r/'.numi/runs/american-crow';trial=runs/'precision-training-01'
spec=importlib.util.spec_from_file_location('archive',r/'tools/crow_evidence_archive.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
def record(p):return {'sha256':a.file_hash(p),'bytes':p.stat().st_size,'mode':p.stat().st_mode&0o777}
retained=r/'.numi/retained-runtime/precision-training-01'
for relative in ('bin/metalrobo_task_rollout','bin/metalrobo_task_train','lib/libmetalrobo.dylib','shaders/MetalRobo.metallib','shaders/GRUMemory.metallib','shaders/RetinalStream.metallib'):
 p=r/'build'/relative;q=retained/relative;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
old=json.loads((r/'artifacts/crow-rl-native-causal-memory-20261108/runtime-recovery.json').read_text());m={'manifests':old['manifests']+['artifacts/crow-rl-precision-training-20261108/manifest.json'],'native':{},'python_sources':{},'git_inputs':{},'checkpoint':{}}
for p in retained.rglob('*'):
 if p.is_file():m['native'][str(p.relative_to(retained))]={**record(p),'archive_source':str(p.relative_to(r))}
for p in (r/'python/metalrobo').rglob('*.py'):
 name=str(p.relative_to(r));m['python_sources'][name]={**record(p),'git_blob':a.git('rev-parse','HEAD:'+name).decode().strip()}
for p in (r/'environments/american-crow').rglob('*'):
 if p.is_file():
  name=str(p.relative_to(r));m['git_inputs']['project/'+name]={**record(p),'git_blob':a.git('rev-parse','HEAD:'+name).decode().strip()}
bundle=trial/'training/checkpoints/revision-00000097.trainingcheckpoint'
for p in bundle.iterdir():
 if p.is_file():m['checkpoint']['input/checkpoint/'+p.name]=str(p.relative_to(r))
m['training_command']=json.loads((runs/'precision-training-continuation-01/tail8.command.json').read_text());ref=runs/'precision-training-continuation-01/tail8/checkpoints/revision-00000098.trainingcheckpoint'
m['training_reference']={name:record(ref/name) for name in ('policy.policypack','candidate.policypack','rollout.rolloutpack')}
tensors,meta=mx.load(ref/'learner.safetensors',return_metadata=True);m['learner_reference']={'metadata':meta,'tensors':{k:{'dtype':str(np.asarray(v).dtype),'shape':list(v.shape),'sha256':hashlib.sha256(np.asarray(v).tobytes()).hexdigest()} for k,v in tensors.items()}}
m['training_native_sections']=json.loads((runs/'precision-training-continuation-01/summary.json').read_text())['native_sections_sha256']
m['recovery_scope']='Actual Git-restored one-update128-lane precision final97 training continuation; matching OS/device and Python dependencies required.'
analysis={'status':'128_lane_admitted_feeding_still_flat','source_commit':a.git('rev-parse','HEAD').decode().strip(),'campaign':json.loads((trial/'summary.json').read_text()),'offline_mechanism':json.loads((trial/'diagnostic/summary.json').read_text()),'final_training_continuation':{k:v for k,v in json.loads((runs/'precision-training-continuation-01/summary.json').read_text()).items() if k!='records'},'learned_memory_qualified':False}
subprocess.run([sys.executable,'tools/crow_evidence_archive.py','capture',str(out.relative_to(r))],check=True)
(out/'runtime-recovery.json').write_text(json.dumps(m,indent=2)+'\n');(out/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
(out/'README.md').write_text("""#128-lane recurrent crow learning admission

Git-restored policy81/Adam continues on16 new128-lane320-tick native populations:
655360 accepted transitions, two PPO epochs and5120-sample full-lifetime minibatches
(16Adam steps/update), existing adaptive LR with initial0.001 and ceiling0.002.
No physics, camera, resource, reward, normalization or action-hold change.

Intake101/512 first lifetimes versus109/512 last does not establish improvement.
Head std0.5598->0.4647; beak std0.6684->0.6723. Offline cue-dependent head gap
remains~1.12; swapped early RGB reverses it and memory cut40 removes it.
This does not newly qualify native memory or feeding for final97. Policy81 was
previously causally qualified only in the fixed head-reachable fixture. Later
native evaluation must assess both Gaussian and deployment-mean behavior.

Observed training throughput5203 transitions/s, MLX peak382185480bytes, zero swap
before/after and no reported physics rejections. Native retained104768418bytes,
transient private76181276bytes. These are separate counters, not whole-system
peak memory. Earlier64-lane recipe gave3853transitions/s and~1.5GB MLX peak;
recipes differ in population/minibatch/rate, so this is not a matched scaling claim.

Fresh final97 checkpoint continuations at chunks8/1 produce exact98 policy,
candidate, rollout, all53 learner/Adam tensors/metadata and all three native state
components. Actual Git-restored future repeats that128-lane update, audits the
loaded adjacent library and compares exact data before raw/duplicate cleanup.
Four public/native seed-bound rejection controls pass before output creation.
Full native causal evaluation now accepts explicit seed sets; prior held-out seeds
become validation after inspection. Full lifelong intelligence remains unfinished.
""")
