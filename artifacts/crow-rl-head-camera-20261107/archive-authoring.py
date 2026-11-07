from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/Users/n/Developer/numi-crow-rl-20261107')
runs=root/'.numi/runs/american-crow'
archive=root/'artifacts/crow-rl-head-camera-20261107'
archive.mkdir(exist_ok=False)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def copy(a,b):
    b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b)
    assert sha(a)==sha(b)
names=['head-camera-admission-01','head-camera-admission-02','head-camera-admission-03','head-camera-admission-04','head-camera-feeding-replay-01','head-camera-feeding-replay-02']
entries=[]
for name in names:
    source=runs/('20261107-'+name)
    for p in sorted(source.rglob('*')):
        if not p.is_file():continue
        target=archive/'trials'/p.relative_to(runs)
        copy(p,target)
        entries.append({'path':str(p.relative_to(root)),'archive':str(target.relative_to(archive)),'bytes':p.stat().st_size,'sha256':sha(p)})
    for receipt in source.rglob('*.receipt.json'):
        r=json.loads(receipt.read_text())
        stem=receipt.name.removesuffix('.receipt.json')
        for suffix,key in [('.json','stdout_sha256'),('.stderr','stderr_sha256')]:
            if key in r:assert sha(receipt.with_name(stem+suffix))==r[key]
trial=runs/'20261107-head-camera-admission-04'
assert json.loads((trial/'summary.json').read_text())['status']=='passed'
rows=[json.loads(s) for s in (trial/'native-evidence/measurements.jsonl').read_text().splitlines()]
assert len(rows)==8 and all(r['identity_errors']==0 and r['maximum_depth_error_m']<=.0002 and r['maximum_camera_pose_error']<=2e-6 for r in rows)
for r in rows:
    if r['control']=='neutral':assert min(r['target_pixels'])>=20
    if r['control'] in ('occluded','yaw'):assert max(r['target_pixels'])==0
summary={'schema':'numi.crow-head-camera-archive.v1','status':'geometry-instrument-admitted',
         'stable_geometry_pixels':sum(r['compared_pixels'] for r in rows),
         'stable_empty_pixels':sum(r['empty_compared_pixels'] for r in rows),
         'maximum_depth_error_m':max(r['maximum_depth_error_m'] for r in rows),
         'maximum_camera_pose_component_error':max(r['maximum_camera_pose_error'] for r in rows),
         'malformed_authoring_cases_rejected':7,'actor_vision_integrated':False,
         'biological_eye_calibration':False,'ecological_environment_qualified':False}
# Preserve the exact passing binary trio outside Git; the portable source,
# build recipe, toolchain identity and hashed runtime inventory are in Git.
plan=json.loads((trial/'plan.json').read_text())
runtime=root/'.numi/retained-runtime/crow-head-camera-admission-04'
runtime.mkdir(parents=True,exist_ok=False)
runtime_receipt=[]
for key in ('eye_check','library','metallib'):
    item=plan['preflight']['inputs'][key];source=Path(item['path']);assert sha(source)==item['sha256']
    target=runtime/source.name;copy(source,target)
    runtime_receipt.append({'kind':key,'path':str(target),'bytes':target.stat().st_size,'sha256':sha(target)})
(archive/'runtime-receipt.json').write_text(json.dumps(runtime_receipt,indent=2)+'\n')
# Recover older authoring inputs only when they match the preregistered hashes.
current_proxy=(root/'src/core/CollisionVisualProxy.cpp').read_text()
proxy02=current_proxy.replace('primitive.geometry={firstIndex,static_cast<std::uint32_t>(pack.indices.size())-firstIndex,0,\n                            static_cast<std::uint32_t>(pack.instances.size())};','primitive.geometry={firstIndex,static_cast<std::uint32_t>(pack.indices.size())-firstIndex,0,0};')
proxy01=proxy02.replace('    normal.w=1.0f; // Right-handed tangent frame, required by VisualAssetPackV2.\n','')
app03=(root.parent/'numi-crow-rl-20261107-eye03-source.mm').read_text()
app02=app03.replace('numi.crow-head-camera-admission.v3','numi.crow-head-camera-admission.v2').replace('head-camera-admission-01 rejected missing vertex tangent sign; 02 rejected primitive-to-instance ownership before rendering; corrected per-shape instance index; geometric thresholds unchanged','head-camera-admission-01 rejected a missing vertex tangent sign before rendering; corrected to +1 with unchanged geometric thresholds and added raw exports')
recovered=[]
for attempt,relative,payload in [(1,'src/core/CollisionVisualProxy.cpp',proxy01),(2,'src/core/CollisionVisualProxy.cpp',proxy02),(2,'apps/crow_eye_check.mm',app02),(3,'src/core/CollisionVisualProxy.cpp',current_proxy),(3,'apps/crow_eye_check.mm',app03),(4,'apps/crow_eye_check.mm',(root/'apps/crow_eye_check.mm').read_text())]:
    expected=json.loads((runs/f'20261107-head-camera-admission-{attempt:02d}/plan.json').read_text())['preflight']['untracked_inputs'][relative]
    actual=hashlib.sha256(payload.encode()).hexdigest()
    record={'attempt':attempt,'source':relative,'expected_sha256':expected,'recovered_exact':actual==expected}
    if actual==expected:
        target=archive/f'verified-authoring-inputs/attempt-{attempt:02d}'/relative
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(payload)
    recovered.append(record)
(archive/'source-recovery-receipt.json').write_text(json.dumps(recovered,indent=2)+'\n')
# The new source-inputs snapshot in replay02 contains the complete tested
# runtime delta against its exact base Git commit. Validate every file now.
snapshot=runs/'20261107-head-camera-feeding-replay-02/source-inputs'
manifest=json.loads((snapshot/'manifest.json').read_text())
assert sha(snapshot/'tracked.patch')==manifest['tracked_patch_sha256']
for name,expected in manifest['retained_untracked'].items():assert sha(snapshot/'untracked'/name)==expected
copy(root/'build/Testing/Temporary/LastTest.log',archive/'native-ctest.log')
for name,command in [('python-checks',['python/.venv/bin/python','-m','unittest','discover','-s','tests','-p','crow_*test.py']),('doctor',['tools/numi','crow-env','doctor'])]:
    result=subprocess.run(command,cwd=root,text=True,capture_output=True)
    (archive/(name+'.stdout')).write_text(result.stdout);(archive/(name+'.stderr')).write_text(result.stderr)
    (archive/(name+'.receipt.json')).write_text(json.dumps({'command':command,'exit_code':result.returncode})+'\n');assert result.returncode==0
replay=json.loads((runs/'20261107-head-camera-feeding-replay-02/summary.json').read_text())
assert [r['fingerprint'] for r in replay['records']]==['2255877855914022940','3082487550463979285']
summary['unchanged_feeding_replay']=replay['records']
(archive/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
(archive/'payload-inventory.json').write_text(json.dumps(entries,indent=2)+'\n')
copy(Path(__file__),archive/'archive-authoring.py')
(archive/'.gitattributes').write_text('*.bin binary\n*.f32 binary\n*.u32 binary\n*.mrvpack binary\n*.log whitespace=-blank-at-eof\n')
(archive/'README.md').write_text('''# American crow: native head-camera geometry

Two estimated parallel head cameras now follow accepted active-head poses.
This is publication-time geometric calibration; the v12 learner still has
97 non-image inputs. It does not establish biological eyes, sensor cadence,
latency, learned vision, search, or an ecological environment.

Attempt 04 passed the prospectively extended instrument: 198,286 stable
geometry pixels and 92,374 stable empty pixels, zero identity/validity errors,
maximum optical-z depth error 55.83 micrometres (limit 200), and maximum camera
pose component error 8.51e-8 (limit 2e-6). Neutral views have 360 target pixels
per camera/lane; physical occlusion and averted yaw hide it. Physical pitch
agrees with the independent FP64 geometry oracle. Accepted yaw body states
are bit-identical under chunks 1/8. Every RGB/depth/identity/validity render
repeats exactly. Seven malformed-authoring controls reject invalid input.
Two lanes, four controls and one partition repeat are calibration units,
not independent animals or learned behavioral evidence.

Attempts 01 and 02 failed before rendering: missing tangent sign and wrong
primitive-to-instance ownership. Their failures remain unchanged. Attempt 03
passed the original geometric prediction. Attempt 04 added empty-space and
invalid-authoring controls with unchanged original tolerances. Raw images,
poses, depth, identities, validity, visual assets and PNGs are archived.
Per-camera layout JSON accompanies attempt 04 binary records; native headers
own structured body layouts. PNGs are sRGB display previews of linear RGB.
Collision meshes exactly retain boxes and bounded +Y planes (16 m half-extent)
and do not change the physical world. Camera positions, parallel directions,
45 degree FOV and optics are engineering estimates, not crow measurements.

All seven focused native CTests and 19 Python crow checks passed. The frozen
feeding actor has unchanged ordinary and residual replay fingerprints when
using the original initial deterministic base. Feeding replay01 used the
trained deterministic base instead: it passed partition invariance but is
not the prior residual-reference comparison. Replay02 uses the original base
and matches both prior fingerprints exactly. Both results remain retained.

Replay02 includes a complete source delta against commit 543e59ab, including
new native files, with hashes verified again during archival. Earlier trials
pre-date automatic source-delta retention. verified-authoring-inputs/ contains
only reconstructed older files whose hashes exactly match their original
plans; source-recovery-receipt.json states which were recovered. The complete
attempt01 app source is not reconstructed. Its failed schema validation is
retained evidence, not a reproducible successful scientific result. Current
source and the complete passing delta are retained. Runtime-receipt.json
identifies the exact passing binary trio retained on macmini outside Git.

The next instrument must deliver causal device-resident RGB without identity
or object-position privileges, explicit sampling/latency and complete reset/
restore semantics before connecting a distinct actor contract or learning.
''')
print(json.dumps({'archive':str(archive),'files':len(entries),'analysis':summary}))
