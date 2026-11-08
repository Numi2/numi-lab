"""Bound actual terminal results and runtime before archival recovery."""
from pathlib import Path
import importlib.util,json,shutil,subprocess

r=Path.cwd();b=Path('artifacts/crow-rl-post-step-rate-20261108')
out=Path('.numi/runs/american-crow/post-step-rate-01')
assert (out/'trial-complete.json').is_file()and(out/'analysis.json').is_file()
spec=importlib.util.spec_from_file_location('archive',r/'tools/crow_evidence_archive.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
retained=Path('.numi/retained-runtime/post-step-rate-20261108');retained.mkdir(parents=True,exist_ok=True)
native={}
for relative in ('bin/metalrobo_task_train','bin/metalrobo_task_rollout','bin/metalrobo_crow_memory_causal_check',
                 'lib/libmetalrobo.dylib','shaders/MetalRobo.metallib','shaders/RetinalStream.metallib','shaders/GRUMemory.metallib'):
    p=retained/relative;p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert a.file_hash(p)==a.file_hash(Path('build')/relative)
    else:shutil.copy2(Path('build')/relative,p)
    native[relative]={'archive_source':str(p),'sha256':a.file_hash(p)}
entries=a.tree()
def record(path):
    return {'git_blob':entries[path][1],'sha256':a.file_hash(path),'mode':int(entries[path][0],8)&0o777}
python_sources={n:record(n)for n in entries if n.startswith('python/metalrobo/')and n.endswith('.py')}
git_inputs={n:record(n)for n in entries if n.startswith('environments/american-crow/')and n.endswith(('.json','.bin'))}
oracle_sources={'source/'+n:record(n)for n in entries if n.startswith('include/')and n.endswith(('.h','.hpp'))}
oracle_sources['source/apps/CrowCameraFixture.hpp']=record('apps/CrowCameraFixture.hpp')
prior=json.loads(Path('artifacts/crow-rl-optimizer-audit-20261108/runtime-recovery.json').read_text())
oracle_sources['source/identity_oracle.mm']=prior['git_inputs']['source/identity_oracle.mm']
files={'source/replay-last.py':'build/crow-post-step-rate-build-01/replay-last.py'}
for arm in ('fixed','adaptive'):
    for revision,names in ((96,('policy.policypack','learner.safetensors')),):
        for name in names:
            source=str(out/f'{arm}/checkpoints/revision-{revision:08d}.trainingcheckpoint/{name}')
            files[f'trial/{arm}/checkpoints/revision-{revision:08d}.trainingcheckpoint/{name}']=source
    for name in ('training.rolloutpack','learner.safetensors'):
        files[f'trial/{arm}/{name}']=str(out/arm/name)
files['input/adaptive.policypack']=str(out/'adaptive/updated.policypack')
files['input/replay-reference.json']=str(out/'last-update-replay.json')
evaluation=out/'heldout/adaptive-0/evaluation/native-evidence'
eval_files={str(p.relative_to(evaluation)):{'sha256':a.file_hash(p),'bytes':p.stat().st_size}
            for p in evaluation.rglob('*')if p.is_file()}
a.capture(b/'payload')
a.write_json(b/'runtime-recovery.json',{'schema':'numi.crow-post-step-rate-recovery.v1',
    'source_commit':a.git('rev-parse','HEAD').decode().strip(),'native':native,'python_sources':python_sources,
    'git_inputs':git_inputs,'oracle_sources':oracle_sources,'files':files,'evaluation_reference':{'seed_base':2651411901,
        'original_root':str((r/evaluation).resolve()),'files':eval_files},
    'original_runtime_native_continuation':str(out/'native-continuation/summary.json'),
    'recreated_final_native_checkpoints':{arm:str(out/f'native-continuation/{arm}/checkpoints/revision-00000097.trainingcheckpoint')for arm in ('fixed','adaptive')},
    'scope':'All evidence bytes retained and actual Git verified. Executed recovery replays both actual last learner updates and one complete adaptive12-arm causal evaluation (3 of9 heldout seeds), not complete32-update histories or broad lifelong qualification.'})
for name in ('analysis.json','last-update-replay.json','trial-complete.json'):
    shutil.copy2(out/name,b/name)
shutil.copy2('build/crow-post-step-rate-build-01/analyze.py',b/'analyze.py')
shutil.copy2('build/crow-post-step-rate-build-01/replay-last.py',b/'replay-last.py')
(b/'.gitattributes').write_text('*.stdout -text -diff\n*.stderr -text -diff\n')
print(json.dumps({'status':'captured','files':len(eval_files),'native_files':len(native)}))
