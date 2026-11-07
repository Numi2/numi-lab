from pathlib import Path
import hashlib,json,re,shutil,subprocess
root=Path('/Users/n/Developer/numi-crow-rl-20261107')
runs=root/'.numi/runs/american-crow'
archive=root/'artifacts/crow-rl-retinal-contract-20261107'
assert set(p.name for p in archive.iterdir())=={'inputs'}
(archive/'trials').mkdir(exist_ok=False)
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
        inventory.append({'source':str(p.relative_to(root)),'archive':str(out.relative_to(archive)),'bytes':p.stat().st_size,'sha256':sha(p)})
for name in ['retinal-contract-admission-01','retinal-contract-admission-02','retinal-contract-feeding-replay-01']:
    trial=runs/('20261107-'+name);tree(trial,archive/'trials'/trial.name)
    for receipt in trial.rglob('*.receipt.json'):
        r=json.loads(receipt.read_text());stem=receipt.name.removesuffix('.receipt.json')
        for suffix,key in [('.json','stdout_sha256'),('.stderr','stderr_sha256')]:
            if key in r:assert sha(receipt.with_name(stem+suffix))==r[key]
    source=trial/'source-inputs';manifest=json.loads((source/'manifest.json').read_text())
    assert sha(source/'tracked.patch')==manifest['tracked_patch_sha256']
    for name,expected in manifest['retained_untracked'].items():assert sha(source/'untracked'/name)==expected
trial=runs/'20261107-retinal-contract-admission-02'
summary=json.loads((trial/'summary.json').read_text());assert summary['status']=='passed'
assert all(summary[k] for k in ['missing_observer_rejected','mismatched_observer_identity_rejected','stale_shader_rejected','current_shader_reaches_declared_callback'])
plan=json.loads((trial/'plan.json').read_text())
legacy=archive/'inputs/legacy-MetalRobo.metallib'
receipt=json.loads((archive/'inputs/legacy-shader-receipt.json').read_text())
assert sha(legacy)==receipt['sha256']==plan['preflight']['inputs']['retinal_legacy_shader']['sha256']
last=root/'build/Testing/Temporary/LastTest.log';testlog=last.read_text()
assert 'retinal_contract' in testlog and 'Test Failed' not in testlog
copy(last,archive/'native-ctest.log')
for path in re.findall(r'Evidence retained at ([^\s]+)',testlog):
    p=Path(path);assert p.is_dir() and p.parent==root/'build';tree(p,archive/'native-ctest-evidence'/p.name)
runtime=root/'.numi/retained-runtime/crow-retinal-contract-admission-02';runtime.mkdir(parents=True,exist_ok=False)
receipts=[]
for key in ['retinal_contract_check','library','metallib']:
    item=plan['preflight']['inputs'][key];source=Path(item['path']);assert sha(source)==item['sha256']
    target=runtime/source.name;copy(source,target)
    receipts.append({'kind':key,'path':str(target),'bytes':target.stat().st_size,'sha256':sha(target)})
(archive/'runtime-receipt.json').write_text(json.dumps(receipts,indent=2)+'\n')
replay=json.loads((runs/'20261107-retinal-contract-feeding-replay-01/summary.json').read_text())
assert [r['fingerprint'] for r in replay['records']]==['2255877855914022940','3082487550463979285']
for name,command in [('python-checks',['python/.venv/bin/python','-m','unittest','discover','-s','tests','-p','crow_*test.py']),('doctor',['tools/numi','crow-env','doctor'])]:
    result=subprocess.run(command,cwd=root,text=True,capture_output=True)
    (archive/(name+'.stdout')).write_text(result.stdout);(archive/(name+'.stderr')).write_text(result.stderr)
    (archive/(name+'.receipt.json')).write_text(json.dumps({'command':command,'exit_code':result.returncode})+'\n');assert result.returncode==0
(archive/'analysis.json').write_text(json.dumps({'schema':'numi.crow-retinal-contract-archive.v1',
    'status':'typed-contract-and-native-rejection-controls-admitted','current_admission':summary,
    'native_ctests_passed':9,'python_checks_passed':19,'feeding_replay':replay['records'],
    'C_API_structs_changed':False,'Cxx_consumers_require_rebuild':True,
    'legacy_task_struct_and_executable_tag':47,'new_retinal_semantic_tag':48,
    'compiled_assets_are_metadata_only':True,'rendered_actor_images':False,
    'v13_rollout_and_learning_exposed':False,'ecological_environment_qualified':False},indent=2)+'\n')
(archive/'payload-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
copy(Path(__file__),archive/'archive-authoring.py')
(archive/'.gitattributes').write_text('*.bin binary\n*.f32 binary\n*.u32 binary\n*.mrvpack binary\n*.metallib binary\n*.patch -whitespace\n*.log whitespace=-blank-at-eof\n')
(archive/'README.md').write_text('''# Retinal actor contract and native admission

This checkpoint defines the native v13 retinal feeding contract and protects
its executable boundary. It does not yet deliver rendered images to the actor.
A v13 rollout/training CLI is not exposed. The next required implementation is
the real two-camera GPU observer, native callback composition, visibility and
delay controls, a pixel-dependent actor, and complete performance/learning tests.
The complete lifelong environment remains unqualified.

The distinct contract has 1,251 actor features: 97 existing channels, two
16 x 12 RGB planes, valid and age. The critic remains 97 and actions remain 18.
Mechanics, action meanings and finite-resource program identity are unchanged.
Camera positions/orientations, source/output dimensions, optics, sampling,
latency, content and response semantics bind both observation and task
fingerprints. The source is SensorPack, not an objective or hidden teacher.
New native observation metadata requires the exact observation fingerprint.

Shared task structs and legacy compilation retain ABI/tag 47. Retinal tasks
receive executable tag 48, supported explicitly by task kernels. The host
checks a capability in the same task-shader translation unit before submitting
48. The old ABI-47 shader is retained, with its exact receipt, in inputs/ for
portable negative controls. It is an actual prior native shader, not a mock.
C API structs are unchanged; C++ consumers need a rebuild for the new observer
metadata. No legacy policy is dimension-rebound or silently promoted to v13.

Trial 01 passed compiler, sensor-contract and native missing/stale checks.
Trial 02 added exact observer identity and a mismatch control, and removed two
instrument compiler warnings. Its 12 malformed manifests reject, and nine
camera/content/timing changes with the same 1,251 inputs reject the old policy.
Asset path relocation with identical content preserves identity. Native missing
and mismatched observers and the old shader are rejected before a live physics
submission. The current shader reaches the expected callback layout, whose
instrument deliberately returns false rather than pretending to supply images.
The fixture visual paths/hashes are compiler metadata only; no visual files are
loaded and no actor images are rendered by this probe. These are instrument
controls, not animal trajectories, learning or full environment qualification.

All nine focused native CTests and 19 Python crow checks passed. CTest output
and native evidence directories are copied here. The frozen v12 feeding actor
retains both exact prior replay fingerprints, ordinary and residual, under
chunks 1/8 and the original initial deterministic base. Thus legacy task,
observation, action and accepted rollout behavior remain unchanged after the
additive semantic tag and shader capability change.

Trials retain exact source deltas and untracked source against commit 461d7b73.
Archive creation rechecked their source and stdout/stderr hashes. The exact
current executable/library/shader trio remains on macmini under the paths in
runtime-receipt.json. Source, old shader input, controls, retained CTest images
and state and raw reports are in Git. payload-inventory.json binds copied data.
''')
print(json.dumps({'archive':str(archive),'inventoried_files':len(inventory),'legacy_shader_sha256':sha(legacy)}))
