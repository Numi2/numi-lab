from pathlib import Path
import hashlib,json,re,shutil,subprocess
root=Path('/Users/n/Developer/numi-crow-rl-20261107')
runs=root/'.numi/runs/american-crow'
archive=root/'artifacts/crow-rl-retinal-timing-20261107'
archive.mkdir(exist_ok=False)
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
        out=target/p.relative_to(source);copy(p,out)
        inventory.append({'source':str(p.relative_to(root)),'archive':str(out.relative_to(archive)),
                          'bytes':p.stat().st_size,'sha256':sha(p)})
for name in ['retinal-timing-admission-01','retinal-timing-admission-02','retinal-timing-admission-03','retinal-feeding-replay-01','retinal-validation-01']:
    trial=runs/('20261107-'+name);tree(trial,archive/'trials'/trial.name)
    for receipt in trial.rglob('*.receipt.json'):
        r=json.loads(receipt.read_text());stem=receipt.name.removesuffix('.receipt.json')
        for suffix,key in [('.json','stdout_sha256'),('.stderr','stderr_sha256')]:
            if key in r:assert sha(receipt.with_name(stem+suffix))==r[key]
    source=trial/'source-inputs'
    if source.is_dir():
        manifest=json.loads((source/'manifest.json').read_text())
        assert sha(source/'tracked.patch')==manifest['tracked_patch_sha256']
        for name,expected in manifest['retained_untracked'].items():assert sha(source/'untracked'/name)==expected
last=root/'build/Testing/Temporary/LastTest.log'
text=last.read_text();assert 'retinal_timing' in text and 'Test Failed' not in text
copy(last,archive/'native-ctest-current.log')
for path in re.findall(r'Evidence retained at ([^\s]+)',text):
    p=Path(path);assert p.is_dir() and p.parent==root/'build'
    tree(p,archive/'native-ctest-evidence'/p.name)
trial=runs/'20261107-retinal-timing-admission-03'
summary=json.loads((trial/'summary.json').read_text());assert summary['status']=='passed'
assert summary['FP64_max_absolute_error']<=3e-6 and summary['exact_partition_invariance']
plan=json.loads((trial/'plan.json').read_text())
runtime=root/'.numi/retained-runtime/crow-retinal-timing-admission-03'
runtime.mkdir(parents=True,exist_ok=False)
receipts=[]
for key in ['retinal_check','library','retinal_metallib','metallib']:
    item=plan['preflight']['inputs'][key];source=Path(item['path']);assert sha(source)==item['sha256']
    target=runtime/source.name;copy(source,target)
    receipts.append({'kind':key,'path':str(target),'bytes':target.stat().st_size,'sha256':sha(target)})
(archive/'runtime-receipt.json').write_text(json.dumps(receipts,indent=2)+'\n')
replay=json.loads((runs/'20261107-retinal-feeding-replay-01/summary.json').read_text())
assert [r['fingerprint'] for r in replay['records']]==['2255877855914022940','3082487550463979285']
for name,command in [('python-checks',['python/.venv/bin/python','-m','unittest','discover','-s','tests','-p','crow_*test.py']),('doctor',['tools/numi','crow-env','doctor'])]:
    result=subprocess.run(command,cwd=root,text=True,capture_output=True)
    (archive/(name+'.stdout')).write_text(result.stdout);(archive/(name+'.stderr')).write_text(result.stderr)
    (archive/(name+'.receipt.json')).write_text(json.dumps({'command':command,'exit_code':result.returncode})+'\n');assert result.returncode==0
(archive/'analysis.json').write_text(json.dumps({'schema':'numi.retinal-timing-archive.v1',
    'status':'controlled-GPU-timing-instrument-admitted','current_admission':summary,
    'source_index_and_configuration_rejection_cases':8,'positive_submission_chunks':[1,8],
    'cadence_delay_boundary_conditions':[[1,1,0],[1,1,1],[1,1,16],[1,50,2]],
    'main_input_lanes':3,'main_calls_per_partition':51,'total_fixture_lane_observations_per_current_trial':984,
    'native_ctests_passed':8,'python_checks_passed':19,'feeding_replay':replay['records'],
    'renderer_actor_integration':False,'full_life_restore':False,'ecological_environment_qualified':False},indent=2)+'\n')
(archive/'payload-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
copy(Path(__file__),archive/'archive-authoring.py')
(archive/'.gitattributes').write_text('*.bin binary\n*.f32 binary\n*.u32 binary\n*.mrvpack binary\n*.patch -whitespace\n*.log whitespace=-blank-at-eof\n')
(archive/'README.md').write_text('''# Native retinal stream: timing admission

The lifelong American-crow project now has a native RGB-only timing layer,
not a complete visual learner. The existing v12 actor remains unchanged at
97 non-image inputs. The renderer and this stream have not yet been composed
inside accepted rollout/inference. Geometry and head-camera calibration are
in the separate crow-rl-head-camera-20261107 archive.

RetinalStreamContext owns private GPU pending/visible frames and timing state.
It borrows RGB, native task episode/initialization fields, actor output and the
caller's command buffer. It encodes area averaging, fixed c/(1+c) radiance
response, rational sampling, delayed delivery and publication. No commits,
waits, host reads or host sampling decisions occur in encode(). No depth,
instance identities, segmentation, poses or object coordinates enter it.
Setup and explicit publication may synchronize; neither is a per-step loop.

The development schedule is 3/10 samples per 50 Hz accepted tick (15 Hz),
two-tick delivery latency (40 ms), two eyes and explicit valid/age features.
Repeated ticks are idempotent despite changed RGB. Episode changes clear
pending and visible images for the affected lane. Clock gaps, backwards clocks,
uninitialized clocks and invalid captured radiance latch native per-lane errors;
publication becomes invalid/zero. The tested negatives are clock gaps,
nonfinite RGB, and finite-radiance reduction overflow. No reset clears errors.
One command buffer at a time may mutate a stream; failed GPU commands poison
the context. A debug snapshot is not a checkpoint and cannot restore images,
world, policy or memory. All rollout acceptance must inspect stream errors.

Trial 01 passed the original instrument. Trial 02 prospectively extended
parameter boundaries and finite-radiance overflow controls. Trial 03 added
source-image index admission and raw output retention before assertions.
All retained trials passed their preregistered predictions. Current controls
cover three GPU-fixture lanes, fifty ticks with a duplicated seventh tick,
a single-lane episode reset, exact chunks 1/8, and an independent FP64 analytic
area/timing oracle. Maximum feature error is 4.77e-8 against the registered
3e-6 limit. Ratios 1/1 and 1/50 and delay boundaries 0, 1 and 16 ticks pass.
Eight malformed configurations reject, including an overflowing source-index
configuration, as do short/null output, null clock and missing-eye calls.
Main, boundary and negative raw actor planes, guard features, state and error
records are retained. Output layout metadata is authoritative for trial 03;
earlier layout files predate per-file call counts. These are native synthetic
instrument controls, not embodied trajectories or independent animals.

All eight focused native CTests and 19 Python crow checks passed. Their retained
logs and copied CTest evidence remain bound to this source. The earlier full
suite before the source-index guard is retained separately too. The frozen
feeding actor still exactly matches its earlier ordinary/residual replay with
the initial deterministic base. The new sensory kernels are in a separate
RetinalStream.metallib; the established physics/policy metallib hash is unchanged.

Every outer trial retains an exact tracked patch and untracked source inputs
against Git commit 20327805; archive creation rechecked these hashes and native
stdout/stderr receipts. runtime-receipt.json identifies exact current binaries
and both shader libraries retained on macmini outside Git. Source and raw
controlled evidence are in Git. No learned visual behavior, biological eye
calibration, full-life checkpoint, ecological performance or qualification is
claimed. Next: bind a distinct actor sensor contract, compose both head cameras
and this stream on the native accepted timeline, verify causal observations and
occlusion, then measure full rollout and learning throughput before training.
''')
print(json.dumps({'archive':str(archive),'inventoried_files':len(inventory),'maximum_feature_error':summary['FP64_max_absolute_error']}))
