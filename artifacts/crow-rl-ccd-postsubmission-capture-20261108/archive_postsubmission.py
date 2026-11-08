"""Preserve the rejected post-submission sweep capture without a retry."""
from pathlib import Path
import importlib.util,json,shutil
root=Path.cwd();spec=importlib.util.spec_from_file_location('archive',root/'tools/crow_evidence_archive.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
destination=Path('artifacts/crow-rl-ccd-postsubmission-capture-20261108')
retained=Path('.numi/retained-runtime/ccd-postsubmission-capture-20261108');retained.mkdir(parents=True,exist_ok=False)
trial=Path('.numi/runs/american-crow/ccd-failure-sweep-01');probe=Path('.numi/runs/american-crow/ccd-failure-sweep-analysis-01')
native={}
for relative in ('bin/metalrobo_task_train','bin/metalrobo_task_rollout','lib/libmetalrobo.dylib',
                 'shaders/MetalRobo.metallib','shaders/RetinalStream.metallib','shaders/GRUMemory.metallib'):
 target=retained/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path('build')/relative,target)
 native[relative]={'archive_source':str(target),'sha256':a.file_hash(target)}
for relative,source in [('bin/metalrobo_ccd_sweep_probe','metalrobo_ccd_sweep_probe'),('shaders/CCDSweepProbe.metallib','probe.metallib')]:
 target=retained/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(Path('build/crow-ccd-failure-sweep-build-01')/source,target)
 native[relative]={'archive_source':str(target),'sha256':a.file_hash(target)}
entries=a.tree()
def record(name):return {'git_blob':entries[name][1],'sha256':a.file_hash(name),'mode':int(entries[name][0],8)&0o777}
checkpoint={'input/failure.ccdsweep':str(trial/'failure.ccdsweep')}
for path in (trial/'training/checkpoints/revision-00000011.trainingcheckpoint').iterdir():
 if path.is_file():checkpoint['input/source.trainingcheckpoint/'+path.name]=str(path)
for name in ('ccd_sweep_probe.mm','ccd_sweep_probe.metal'):
 target=Path('build/crow-ccd-failure-sweep-build-01')/name;shutil.copy2(Path('apps')/name,target)
 checkpoint['source/'+name]=str(target)
summary=json.loads((trial/'summary-v2.json').read_text())
receipt=json.loads((probe/'replay.receipt.json').read_text());stderr=(probe/'replay.stderr').read_text()
assert receipt['exit_code']==1 and stderr=='Reported failed candidate absent\n'
a.write_json(probe/'negative-analysis.json',{
 'status':'replay_admission_prediction_contradicted','native_execution_repeated':False,
 'finding':'Post-submission CCD candidate segment has two active records, both pair108. Reported failed pair100 is absent; event cursors have advanced to final substep. The captured scratch is not admitted as the failed sweep.',
 'original_plan_retained':True,'native_probe_rejected_before_GPU_encoding':True,
 'decision':'Capture the first unsafe unresolved candidate plus both endpoint projections and the event cursor inside the GPU resolver into protected inspection storage. Do not tune CCD budgets or learning rate from the rejected post-submission scratch.'})
extra=('src/metal/CCDFailureCapture.inc','src/metal/MetalWorld.mm','src/metal/Collision.metal','include/metalrobo/engine_types.h')
metadata={'schema':'numi.crow-ccd-postsubmission-capture-git-recovery.v1','source_commit':a.git('rev-parse','HEAD').decode().strip(),
 'manifests':[str(destination/'manifest.json')],'native':native,
 'python_sources':{p:record(p) for p in entries if p.startswith('python/metalrobo/') and p.endswith('.py')},
 'git_inputs':{p:record(p) for p in entries if p.startswith('environments/american-crow/object-push-v1/') or p in extra},
 'checkpoint':checkpoint,'capture_reference':summary,
 'negative_probe_reference':{'returncode':1,'stderr':stderr,'command':receipt['command']},
 'scope':'Complete Git byte restoration. Execute the restored standalone parser/probe against the rejected snapshot and require its same pre-GPU rejection. No claim of a failing-sweep replay, CCD repair, fresh learning success or independently executed full learner recovery for this diagnostic archive.'}
a.capture(destination);a.write_json(destination/'runtime-recovery.json',metadata)
print(json.dumps({'status':'captured','native_files':len(native),'recovery_inputs':len(checkpoint),'negative_capture_preserved':True}))
