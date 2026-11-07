from pathlib import Path
import gzip,hashlib,json,re,shutil,subprocess
root=Path('/Users/n/Developer/numi-crow-rl-20261107')
runs=root/'.numi/runs/american-crow';archive=root/'artifacts/crow-rl-retinal-live-20261107'
assert set(p.name for p in archive.iterdir())=={'native-regression-10.log'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def copy(a,b):
    b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b);assert sha(a)==sha(b)
inventory=[]
def tree(source,target):
    for p in sorted(source.rglob('*')):
        if not p.is_file():continue
        out=target/p.relative_to(source);out.parent.mkdir(parents=True,exist_ok=True)
        compressed=p.suffix in {'.bin','.f32','.u32'}
        if compressed:
            out=out.with_name(out.name+'.gz')
            with p.open('rb') as raw,out.open('wb') as final:
                with gzip.GzipFile(filename='',fileobj=final,mode='wb',mtime=0) as encoded:shutil.copyfileobj(raw,encoded)
            h=hashlib.sha256()
            with gzip.open(out,'rb') as f:
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            assert h.hexdigest()==sha(p)
        else:copy(p,out)
        inventory.append({'source':str(p.relative_to(root)),'archive':str(out.relative_to(archive)),
            'original_bytes':p.stat().st_size,'original_sha256':sha(p),'encoding':'gzip' if compressed else 'identity',
            'bytes':out.stat().st_size,'sha256':sha(out)})
for i in range(1,13):
    trial=runs/f'20261107-retinal-live-admission-{i:02}'
    tree(trial,archive/'trials'/trial.name)
    for receipt in trial.rglob('*.receipt.json'):
        r=json.loads(receipt.read_text());stem=receipt.name.removesuffix('.receipt.json')
        for suffix,key in [('.json','stdout_sha256'),('.stderr','stderr_sha256')]:
            if key in r:assert sha(receipt.with_name(stem+suffix))==r[key]
    source=trial/'source-inputs';manifest=json.loads((source/'manifest.json').read_text())
    assert sha(source/'tracked.patch')==manifest['tracked_patch_sha256']
    for name,expected in manifest['retained_untracked'].items():assert sha(source/'untracked'/name)==expected
replay=runs/'20261107-retinal-live-feeding-replay-01';tree(replay,archive/'trials'/replay.name)
replay_summary=json.loads((replay/'summary.json').read_text())
assert [r['fingerprint'] for r in replay_summary['records']]==['2255877855914022940','3082487550463979285']
trial=runs/'20261107-retinal-live-admission-12';plan=json.loads((trial/'plan.json').read_text())
summary=json.loads((trial/'summary.json').read_text());assert summary['status']=='passed' and summary['accepted_native_transitions']==384
assert summary['chunk_bit_identical'] and summary['per_lane_reset_bit_identical']
metrics={p.parent.name:json.loads(p.read_text()) for p in (trial/'native-evidence').glob('*/evidence.json')}
assert len(metrics)==6 and max(m['maximum_delayed_rgb_error'] for m in metrics.values())<3e-6
negatives=[json.loads(r) for r in (trial/'native-evidence/visible-chunk1/observer-negative-controls.jsonl').read_text().splitlines()]
assert len(negatives)==8 and all(r['rejected'] for r in negatives)
regression=(archive/'native-regression-10.log').read_text();assert 'retinal_live' in regression and 'Test Failed' not in regression
latest=root/'build/Testing/Temporary/LastTest.log';copy(latest,archive/'native-live-final.log')
for log in [regression,latest.read_text()]:
    for path in re.findall(r'Evidence retained at ([^\s]+)',log):
        p=Path(path);assert p.is_dir() and p.parent==root/'build'
        destination=archive/'native-ctest-evidence'/p.name
        if not destination.exists():tree(p,destination)
runtime=root/'.numi/retained-runtime/crow-retinal-live-admission-12';runtime.mkdir(parents=True,exist_ok=False)
receipts=[]
for key,relative in [('retinal_live_check','bin/metalrobo_crow_retinal_live_check'),('library','lib/libmetalrobo.dylib'),('metallib','shaders/MetalRobo.metallib'),('retinal_metallib','shaders/RetinalStream.metallib')]:
    item=plan['preflight']['inputs'][key];source=Path(item['path']);assert sha(source)==item['sha256']
    target=runtime/relative;copy(source,target);target.chmod(source.stat().st_mode)
    receipts.append({'kind':key,'path':str(target),'bytes':target.stat().st_size,'sha256':sha(target)})
(archive/'runtime-receipt.json').write_text(json.dumps(receipts,indent=2)+'\n')
for name,command in [('python-checks',['python/.venv/bin/python','-m','unittest','discover','-s','tests','-p','crow_*test.py']),('doctor',['tools/numi','crow-env','doctor'])]:
    result=subprocess.run(command,cwd=root,text=True,capture_output=True)
    (archive/(name+'.stdout')).write_text(result.stdout);(archive/(name+'.stderr')).write_text(result.stderr)
    (archive/(name+'.receipt.json')).write_text(json.dumps({'command':command,'exit_code':result.returncode})+'\n');assert result.returncode==0
analysis={'schema':'numi.crow-retinal-live-archive.v1','status':'native-live-pixel-actor-controls-admitted',
    'current_admission':summary,'metrics':metrics,'observer_negatives':negatives,
    'native_regression_checks_passed':10,'latest_live_regression_repeated_after_capacity_refinement':True,'python_checks_passed':19,
    'legacy_feeding_fingerprints':replay_summary['records'],'attempts_retained':12,
    'manually_terminated_attempts':[6,7],'stops':'own native child SIGTERM after stack sample showed 64 dispatch workers blocked in Metal IO; receipts exit=-15, not successful runs',
    'final_capacity':'original crow pair/manifold/convex/CCD bounds; raw256/block160/rows480/endpoints320 from explicit substep requirements',
    'runtime_source_base':plan['preflight']['source_commit'],'runtime_source_delta':plan['preflight']['tracked_diff_sha256'],
    'scope':'authored deterministic pixel actor; two lanes, fixed physical collision-proxy fixture; no learned visual behavior',
    'render_every_control_tick':True,'GPU_sample_rate_hz':15,'latency_seconds':.04,
    'whole_world_restore':False,'v13_C_API_Swift_MLX_training_entry':False,'production_batch_performance_qualified':False,'ecological_environment_qualified':False}
(archive/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
(archive/'payload-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
copy(Path(__file__),archive/'archive-authoring.py')
(archive/'.gitattributes').write_text('*.gz binary\n*.mrvpack binary\n*.policypack binary\n*.patch -whitespace\n*.log whitespace=-blank-at-eof\n')
(archive/'README.md').write_text('''# Live native retinal actor evidence

The native observer now composes both physical head cameras, GPU retinal
cadence/delay and policy inference in the owning command buffer. No host
pixel readback, task-clock decision, commit or wait occurs in that callback.
The controlled actor reads only one declared RGB cell for its yaw rule.

Final trial12: all six bounded controls pass, 384 accepted transitions, zero
sensor errors/physical terminations, two cold-start blind ticks and lane1
reset at tick17 with uninterrupted lane0 unchanged. Observations, actions,
latents, transitions, full q/v trajectories and final q/v/scene states are
bit-identical across chunk1/8. Physical occlusion reverses the tick2 action
from +0.426903 to -0.356149. Disconnected pixel coefficients give zero actions.
The independent FP64 delayed RGB oracle has maximum error9.72599e-8 against
registered3e-6. Eight malformed observer inputs reject. Ten native regression
checks, nineteen Python checks and the final-capacity live recheck pass.
Legacy frozen feeding fingerprints remain2255877855914022940 and
3082487550463979285. These are authored mechanism controls, not visual learning.

The fixed development observer reconstitutes camera/world/visual content from
actual physics and bound source hashes. Its estimated instantaneous parallel
pinhole cameras are not measured American-crow eyes. It currently renders each
control tick and samples on GPU at15Hz with40ms delay. Production batch scaling,
v13 C API/Swift/MLX learning, whole world/RNG/sensory/memory restore, robust
locomotion, physical food handling, physiology, memory and social agents remain
open. The lifelong environment is not qualified.

All12 attempts, exact source deltas/untracked sources, prospective plans,
receipts, raw measurements and authored actors are preserved. Attempts01–03
failed setup/admission;04–05 passed initial controls;06–07 were explicitly
stopped after repeated Metal IO loading exhausted64 dispatch workers, with
samples and exit=-15 receipts. Per-run autorelease cleanup alone did not solve
it. Shared bounded per-device loading resolves the repeat-construction stall:
08 passes all six runs in4.10s including construction and inspection.09 adds
negative/isolation/trajectory checks.10 records capacity usage.11 fails an
undersized capacity refinement;12 uses measured substep requirements (230 raw,
140 blocks), rounding to256 raw/160 blocks and matching row/endpoint bounds,
while retaining original pair/manifold/convex/CCD bounds. No behavioral or
numerical threshold was relaxed. Final-substep usage alone was insufficient.

Large raw arrays are losslessly gzip-compressed. payload-inventory.json records
both original and archived hashes/bytes. Decompress each gzip to its source
basename for inspection. RGB is environment,y,x,RGBAfloat32; actor is
step,environment,1251float32; actions/latents are step,environment,18float32.
Actors: first97 existing channels, left then right12x16x3 RGB, valid, age sec.
Body and transition record sizes are in trial evidence; q/v trajectories use
actual model dimensions23/22. No snapshot is a full-world checkpoint.

The exact final executable/library/core and retinal shaders are hash-verified
in .numi/retained-runtime/crow-retinal-live-admission-12 (outside Git due size),
with runtime-receipt.json. To reproduce, preserve its bin/lib/shaders layout
and use DYLD_LIBRARY_PATH pointing at its lib directory when running the saved
binary with a fresh output directory. Older attempts preserve exact source
and input hashes, not every historical binary. The broad native regression
log predates the capacity-only app refinement; native-live-final.log verifies
the final code. The unchanged library supports both runs.
''')
print(json.dumps({'archive':str(archive),'inventoried_payloads':len(inventory),'archived_bytes':sum(p.stat().st_size for p in archive.rglob('*') if p.is_file()),'result':summary}))
