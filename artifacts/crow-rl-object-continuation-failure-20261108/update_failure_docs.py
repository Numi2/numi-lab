from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path.cwd();base=root/'artifacts/crow-rl-object-continuation-failure-20261108'
profile=root/'profiles/american-crow-rl-v1.json'
d=json.loads(profile.read_text());o=d['object_push_development']
o['first_unbiased_motor_learning_trial']['next']='Fixed-rate continuation attempted; native CCD failure below must be diagnosed before further learning.'
o['fixed_rate_continuation_attempt']={
 'evidence':'artifacts/crow-rl-object-continuation-failure-20261108','status':'inconclusive_native_physics_failure',
 'source_update':16,'planned_new_updates':16,'last_saved_policy_revision':20,'completed_new_learner_updates':3,
 'new_coupled_checkpoints':0,'held_out_evaluations_run':0,'fixed_learning_rate':.0003,'gamma':.999,'lambda':1,
 'native_failure_status':11,'native_failure_name':'MR_STEP_DID_NOT_CONVERGE','environment':111,
 'submission_local_control_step':6,'failing_substep':31,'compiled_pair_index':106,
 'pair_colliders':['crow_lower_beak/collision','crow_manipulandum/collision'],
 'pair_shape_types':['box','sphere'],'eligible_pairs':109,'pair_mapping_simulation_steps':0,
 'capacities_exceeded':False,'exact_Git_restored_failure_reproduced':True,'partial_model_Adam_arrays_exact':41,
 'canonical_partial_artifacts_exact':4,'temporary_recovery_bytes_removed':264003714,
 'adaptive_LR_selected':False,'learned_manipulation_qualified':False,
 'next':'Instrument the native box-sphere CCD query/advancement failure, preserve a regression, and validate physics/contact/replay before learning resumes.'}
profile.write_text(json.dumps(d,indent=2)+'\n')
doc=root/'docs/CROW_RL_ENVIRONMENT.md';text=doc.read_text()
marker='## Current native history contract (8 November 2026)'
section='''## Fixed-rate object continuation: native failure (8 November 2026)

The next prospectively bounded comparison restores the exact update16/revision17
coupled checkpoint from Git and attempts16 new updates on episodes16..31 at the
same fixedLR.0003,gamma.999,lambda1. No scripted head offset enters the learner.
The planned frozen-policy comparison on64 fresh placements requires both>=20mm
mean endpoint improvement and>=20 percentage-point success gain within20mm.

The result is **inconclusive**, because native physics rejects a control step.
The trainer exits1 after89.200s; the last saved model/candidate is revision20,
consistent with three new completed learner updates. No new coupled checkpoint
or held-out evaluation is produced. The intended evaluation seeds2651317903/904
remain unused. No final trainer JSON/accepted-lifetime ledger exists, so these
partial files do not establish complete sample accounting, throughput, physical
improvement or a resumable coupled life. They remain preserved as evidence.

Status11 (`MR_STEP_DID_NOT_CONVERGE`) occurs in lane111, submission-local tick6,
substep31, failing pair106. Capacity counts5/12/5/12 remain below limits
109/256/109/160, and quality-Newton counters are zero. A zero-simulation-step
native compiler inspection with the exact world fingerprint identifies pair106
as **lower-beak box / movable-object sphere**. The source failure path is
consistent with unresolved rigid CCD; the precise distance-query/advancement
cause is not yet established. This is a physics correction priority, not an
actor/critic or adaptive-learning-rate result.

The complete failed study, planned but unexecuted analysis helpers and exact
runtime are archived at `artifacts/crow-rl-object-continuation-failure-20261108`.
Actual Git-restored execution reproduces the exact failure line, all4 partial
canonical policy/rollout files, all41 model/Adam arrays and metadata. It publishes
no new coupled checkpoint. This is reproduction of failure, not a retry for a
passing behavioral result. The verified264003714B recovery copy is removed after
hash/mode/mtime/closed-handle checks. Pre-run and post-failure swap both report0.
The selected42 raw/runtime/helper files total261831236B and add87226190 new
compressed Git bytes; no video or image capture is produced.

The next work instruments the native box-sphere CCD owner, preserves this case
as a regression and qualifies any correction with physical contact/miss and
deterministic replay controls before learning resumes. AdaptiveLR remains
unselected, and the deployment97 feeding policy is unchanged. Full lifelong
ecology, solid-food handling, generalized manipulation and social learning
remain unfinished.

'''
if marker not in text or '## Fixed-rate object continuation:' in text:raise RuntimeError('Unexpected documentation insertion state')
doc.write_text(text.replace(marker,section+marker))
readme=base/'README.md';s=readme.read_text()
s=s.replace('The exact collider pair and distance-query failure\nmechanism still require instrumentation.',
'''A zero-simulation-step compiler inspection of the exact physical world identifies
pair106 as the lower-beak **box** and movable-object **sphere** (`pair-identity.json`).
The distance-query/advancement failure mechanism still requires instrumentation.''')
s+='''\n## Completed committed-object reproduction\n\n+`verification.json` records successful reproduction of the exact native failure,
all4 partial canonical policy/rollout files and all41 model/Adam arrays and metadata.
The restored run publishes no new coupled checkpoint. Its verified264003714B
temporary copy is removed after exact hash/mode/mtime/closed-handle checks.
No held-out behavioral evaluation or learning-rate trial is added.\n'''.replace('\n+','\n')
readme.write_text(s)
for stem in ('restore','verify'):
 for stream in ('stdout','stderr'):
  src=Path('/private/tmp')/('crow-object-continuation-failure-'+stem+'-20261108.'+stream)
  with (base/(stem+'-wrapper.'+stream)).open('xb') as f:f.write(src.read_bytes())
helper=Path('/private/tmp/crow-ccd-pair-identity-20261108');shutil.copy2(helper,base/'pair_identity')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
record={'scope':'Zero-step compiled-pair identity inspection; no dynamics or physics-accuracy test.',
 'source_sha256':sha(base/'pair_identity.cpp'),'executable_sha256':sha(helper),
 'native_library_sha256':sha(root/'build/lib/libmetalrobo.dylib'),
 'result_sha256':sha(base/'pair-identity.json'),
 'compile_command':['xcrun','clang++','-std=c++20','-Iinclude',str((base/'pair_identity.cpp').relative_to(root)),
                    '-Lbuild/lib','-lmetalrobo','-Wl,-rpath,'+str(root/'build/lib'),'-o',str(helper)],
 'execution_command':[str(helper)],'compiler':subprocess.check_output(['xcrun','clang++','--version'],text=True)}
with (base/'pair-identity.receipt.json').open('x') as f:json.dump(record,f,indent=2)
print('Updated scoped research record and recovery evidence')
