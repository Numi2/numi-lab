"""Archive the analytic distance qualification and contradicted runtime-repair trial."""
from pathlib import Path
import importlib.util,json,shutil
root=Path.cwd();spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
destination=Path('artifacts/crow-rl-ccd-sphere-box-20261108')
retained=Path('.numi/retained-runtime/ccd-sphere-box-20261108');retained.mkdir(parents=True,exist_ok=False)
native={}
for relative in ('bin/metalrobo_task_train','bin/metalrobo_task_rollout','bin/metalrobo_crow_object_push_check',
                 'bin/metalrobo_ccd_distance_probe','bin/metalrobo_metal_world_contact_probe','lib/libmetalrobo.dylib','shaders/MetalRobo.metallib',
                 'shaders/RetinalStream.metallib','shaders/GRUMemory.metallib','shaders/CCDDistanceProbe.metallib'):
 source=Path('build')/relative;target=retained/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 native[relative]={'archive_source':str(target),'sha256':a.file_hash(target)}
legacy=retained/'bin/metalrobo_ccd_distance_probe_legacy'
shutil.copy2(Path('build/crow-ccd-distance-build-01/metalrobo_ccd_distance_probe'),legacy)
native['bin/metalrobo_ccd_distance_probe_legacy']={'archive_source':str(legacy),'sha256':a.file_hash(legacy)}
entries=a.tree()
def record(name):return {'git_blob':entries[name][1],'sha256':a.file_hash(name),'mode':int(entries[name][0],8)&0o777}
trial=Path('.numi/runs/american-crow/object-ccd-learning-01')
failure=(trial/'training-run.stderr').read_text().splitlines()[-1]
if not failure.startswith('task_train failed: Retinal native rollout failed:'):raise RuntimeError('Unexpected failure identity')
import mlx.core as mx
_,learner_metadata=mx.load(trial/'training/learner.safetensors',return_metadata=True)
if learner_metadata['policy_revision']!='14':raise RuntimeError('Unexpected partial model boundary')
a.write_json(trial/'failure-analysis.json',{
 'status':'inconclusive_behavior; runtime_repair_prediction_contradicted',
 'reason':'The fresh fixed-rate native process rolled back in a different sphere-box pair before the declared20-update endpoint.',
 'failure':failure,'original_training_receipt_sha256':a.file_hash(trial/'training-run.receipt.json'),
 'last_completed_model_revision':14,'completed_learner_updates':13,'last_coupled_checkpoint_revision':11,
 'last_coupled_checkpoint_completed_updates':10,'last_coupled_checkpoint_accepted_ticks_per_lane':1600,
 'held_out_evaluations_run':0,
 'metric_limit':'No final trainer JSON or complete accepted-lifetime ledger was published. Partial model/Adam files are evidence, not a coupled resumable life checkpoint or throughput qualification.',
 'pair_identity':json.loads(Path('build/crow-ccd-distance-build-01/pair-identity.json').read_text()),
 'owner_interpretation':'Pair100 is crow_head/collision box against the movable sphere. The analytic distance query passes independently; the complete native CCD advancement/event path remains unqualified. Status11 and ample capacity support a convergence failure, not overflow. Exact failing projected sweep/CCD event state still needs instrumentation.',
 'decision':'Preserve and reproduce this failure from checkpoint10; instrument the native CCD owner before another learning trial. No adaptive learning rate or policy promotion.',
 'behavior_prediction':'Not evaluated; no learned-manipulation success claim.'})
checkpoint={'input/initial-mean.policypack':str(trial/'initial-mean.policypack'),
            'input/initial.policypack':str(trial/'initial.policypack')}
for path in (trial/'training/checkpoints/revision-00000011.trainingcheckpoint').iterdir():
 if path.is_file():checkpoint['input/source.trainingcheckpoint/'+path.name]=str(path)
training_files={}
for path in (trial/'training').iterdir():
 if not path.is_file():continue
 name=path.name;training_files[name]={'sha256':a.file_hash(path),'bytes':path.stat().st_size}
 if name=='learner.safetensors':checkpoint['reference/training/'+name]=str(path)
evaluation={}
queries={}
for phase in ('baseline','candidate','confirmation','build-admission'):
 p=Path('.numi/runs/american-crow')/('ccd-distance-'+phase+'-01')
 checkpoint['input/queries/'+phase+'/cases.f32']=str(p/'cases.f32')
 shader=p/'probe.metallib' if phase!='build-admission' else retained/'shaders/CCDDistanceProbe.metallib'
 checkpoint['input/queries/'+phase+'/probe.metallib']=str(shader)
 queries[phase]={'binary_relative':'bin/metalrobo_ccd_distance_probe' if phase=='build-admission' else 'bin/metalrobo_ccd_distance_probe_legacy','cases_input':'input/queries/'+phase+'/cases.f32','shader_input':'input/queries/'+phase+'/probe.metallib',
                 'results_sha256':a.file_hash(p/'results.bin'),
                 'analysis':json.loads((p/('analysis-v2.json' if phase in ('baseline','candidate') else 'analysis.json')).read_text())}
contact_root=Path('.numi/runs/american-crow/ccd-object-contact-01');contacts={}
for p in sorted(contact_root.glob('*/summary.json')):
 arm=p.parent.name;files={}
 for name in ('final-q.f32','final-v.f32','final-scene.bin'):
  files[name]={'sha256':a.file_hash(p.parent/name),'bytes':(p.parent/name).stat().st_size}
 contacts[arm]={'summary':json.loads(p.read_text()),'files':files}
extra=('tests/check_ccd_distance.py','apps/ccd_distance_probe.metal','apps/ccd_distance_probe.mm','src/metal/Collision.metal')
metadata={'schema':'numi.crow-ccd-sphere-box-git-recovery.v1','source_commit':a.git('rev-parse','HEAD').decode().strip(),
 'manifests':[str(destination/'manifest.json')],'native':native,
 'python_sources':{p:record(p) for p in entries if p.startswith('python/metalrobo/') and p.endswith('.py')},
 'git_inputs':{p:record(p) for p in entries if p.startswith('environments/american-crow/object-push-v1/') or p in extra},
 'checkpoint':checkpoint,'evaluation_references':evaluation,'distance_references':queries,'contact_references':contacts,
 'generic_contact_reference':{'command':json.loads(Path('.numi/runs/american-crow/ccd-generic-contact-01/native.receipt.json').read_text())['command']},
 'training_reference':{'command':json.loads((trial/'training-run.receipt.json').read_text())['command'],
   'files':training_files,'error_line':failure},
 'scope':'Actual Git-restored runtime/source/inputs on the same device and OS. Four query streams including old shader, native contact controls, and checkpoint10 continuation to the exact declared native failure and partial model state. Recovery reproduces evidence; it is not new scientific or biological validation.'}
a.capture(destination);a.write_json(destination/'runtime-recovery.json',metadata)
print(json.dumps({'status':'captured','native_files':len(native),'recovery_payloads':len(checkpoint),'query_streams':len(queries),'contact_arms':len(contacts)}))
