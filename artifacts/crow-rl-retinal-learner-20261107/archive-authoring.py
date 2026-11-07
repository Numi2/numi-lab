from pathlib import Path
import gzip,hashlib,json,re,shutil,subprocess
root=Path('/Users/n/Developer/numi-crow-rl-20261107')
runs=root/'.numi/runs/american-crow';archive=root/'artifacts/crow-rl-retinal-learner-20261107'
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
  out=target/p.relative_to(source);out.parent.mkdir(parents=True,exist_ok=True)
  compressed=p.suffix in {'.bin','.f32','.u32','.rolloutpack','.safetensors'}
  if compressed:
   out=out.with_name(out.name+'.gz')
   with p.open('rb') as raw,out.open('wb') as final:
    with gzip.GzipFile(filename='',fileobj=final,mode='wb',mtime=0) as encoded:shutil.copyfileobj(raw,encoded)
   h=hashlib.sha256()
   with gzip.open(out,'rb') as f:
    for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
   assert h.hexdigest()==sha(p)
  else:copy(p,out)
  inventory.append({'source':str(p.relative_to(root)),'archive':str(out.relative_to(archive)),'original_bytes':p.stat().st_size,'original_sha256':sha(p),'encoding':'gzip' if compressed else 'identity','bytes':out.stat().st_size,'sha256':sha(out)})
trials=[runs/f'20261107-retinal-api-admission-{i:02}' for i in range(1,6)]+[runs/f'20261107-retinal-learning-probe-{i:02}' for i in range(1,4)]+[runs/'20261107-retinal-api-feeding-replay-01']
for trial in trials:
 tree(trial,archive/'trials'/trial.name)
 for receipt in trial.rglob('*.receipt.json'):
  r=json.loads(receipt.read_text());stem=receipt.name.removesuffix('.receipt.json')
  for suffix,key in [('.json','stdout_sha256'),('.stderr','stderr_sha256')]:
   if key in r:assert sha(receipt.with_name(stem+suffix))==r[key]
 source=trial/'source-inputs';m=json.loads((source/'manifest.json').read_text())
 assert sha(source/'tracked.patch')==m['tracked_patch_sha256']
 for name,expected in m['retained_untracked'].items():assert sha(source/'untracked'/name)==expected
api=json.loads((trials[4]/'summary.json').read_text());assert api['status']=='passed' and api['world_reset_preserves_food'] and api['chunk1_8_bit_identical']
learning=[json.loads((p/'summary.json').read_text()) for p in trials[5:8]]
assert [x['environments'] for x in learning]==[16,64,128]
for value in learning:
 assert value['status']=='passed' and value['records'][1]['learner_state_restored']
 assert [(r['initial_revision'],r['final_revision']) for r in value['records']]==[(1,3),(3,4)]
 assert sum(r['samples'] for r in value['records'])==value['environments']*128*3
replay=json.loads((trials[-1]/'summary.json').read_text());assert [r['fingerprint'] for r in replay['records']]==['2255877855914022940','3082487550463979285']
for name in ['retinal-api-native-regression.log','retinal-api-final-build.log','retinal-api-python.stdout','retinal-api-python.stderr']:
 copy(root/'.numi'/name,archive/name)
regression=(archive/'retinal-api-native-regression.log').read_text();assert '100% tests passed out of 11' in regression
assert 'Ran 19 tests' in (archive/'retinal-api-python.stderr').read_text() and 'OK' in (archive/'retinal-api-python.stderr').read_text()
last=root/'build/Testing/Temporary/LastTest.log';copy(last,archive/'native-ctest-details.log')
for path in re.findall(r'Evidence retained at ([^\s]+)',last.read_text()):
 p=Path(path);assert p.is_dir() and p.parent==root/'build';tree(p,archive/'native-ctest-evidence'/p.name)
runtime=root/'.numi/retained-runtime/crow-retinal-learner-admission-05';runtime.mkdir(parents=True,exist_ok=False)
plan=json.loads((trials[4]/'plan.json').read_text());receipts=[]
for key,relative in [('retinal_api_check','bin/metalrobo_crow_retinal_api_check'),('train','bin/metalrobo_task_train'),('rollout','bin/metalrobo_task_rollout'),('library','lib/libmetalrobo.dylib'),('metallib','shaders/MetalRobo.metallib'),('retinal_metallib','shaders/RetinalStream.metallib')]:
 item=plan['preflight']['inputs'][key];source=Path(item['path']);assert sha(source)==item['sha256']
 for p in trials[5:8]:assert json.loads((p/'plan.json').read_text())['preflight']['inputs'][key]['sha256']==item['sha256']
 target=runtime/relative;copy(source,target);target.chmod(source.stat().st_mode);receipts.append({'kind':key,'path':str(target),'bytes':target.stat().st_size,'sha256':sha(target)})
(archive/'runtime-receipt.json').write_text(json.dumps(receipts,indent=2)+'\n')
analysis={'schema':'numi.crow-retinal-learner-archive.v1','status':'native-swift-mlx-engineering-integration-admitted','api':api,'learning':learning,'legacy_fingerprints':replay['records'],'native_checks_passed':11,'python_checks_passed':19,'api_attempts_retained':5,'failed_api_attempts':[1,2,3,4],'source_base':plan['preflight']['source_commit'],'source_delta':plan['preflight']['tracked_diff_sha256'],'scope':'supported fixed development collision proxies, native/Swift/MLX learner updates and learner-only restore','render_every_control_tick':True,'full_process_peak_memory_measured':False,'whole_world_restore':False,'visual_learning_qualified':False,'production_performance_qualified':False,'ecological_environment_qualified':False}
(archive/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
(archive/'payload-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
copy(Path(__file__),archive/'archive-authoring.py')
(archive/'.gitattributes').write_text('*.gz binary\n*.mrvpack binary\n*.policypack binary\n*.patch -whitespace\n*.log whitespace=-blank-at-eof\n')
(archive/'README.md').write_text('''# Retinal learner integration evidence

V13 now runs through the additive C entry, Swift scheduler and MLX learner.
The native API control checks six malformed configurations, complete chunk1/8
stream identity including final state, final/next retinal equality, cold-start
blind ticks and world-reset food conservation. Trial05 passes; all four failed
predecessors are preserved. Independent earlier live pixel/oracle controls also
pass in the eleven-check native regression. Nineteen Python checks pass.

Failures01/02 showed identical actions/physics but different RGB by partition.
The phase-counter hypothesis alone did not resolve this. The actual terminal
state selector guard skipped controlStep==count and left stale projection input.
Direct accepted-state projection fixes it. The fixed noiseless renderer phase
remains explicit, versioned sensor semantics. Trials03/04 then failed consumed
food before reset, first without settling and then with the established gesture.
V13 was missing the compiler's avian ground-curriculum flag; adding it produces
trial05 success with unchanged gesture and exact timing/partition predictions.

Three independent engineering probes use16/64/128 lanes, two initial128-step
updates and one resumed update (revisions1→3→4). All samples publish and learner
state restores. These are not independent trained animals, visual-learning
qualification or complete-world restoration. Each process starts a fresh world.
Exploratory intake alone is not evidence of learned image-dependent behavior.
At128 lanes the loop measures5,346/5,305 steps/s and whole-process5,032/4,933.
MLX peak ~555 MB excludes full process and renderer allocations. Production
scaling/peak-memory qualification and cadence-aware GPU rendering remain open.

Source deltas/untracked sources, plans, receipts, policies, checkpoints and
rollout packs are preserved. Binary raw arrays/rollouts/checkpoints use lossless
gzip; payload-inventory.json records original and archived hashes and sizes.
Recover with gzip.decompress and verify original_sha256. Explicit mrvpack scene
assets are development collision proxies, not biological appearance.
Runtime-receipt.json identifies the retained final runtime on macmini; executables
require the matching saved lib directory in DYLD_LIBRARY_PATH. Failed attempts'
source deltas and runtime hashes are retained, not all prior binary versions.
Legacy feeding ordinary/residual fingerprints remain2255877855914022940 and
3082487550463979285. No qualification seeds are used, no policy is promoted.

Persistent memory, full-world/RNG/sensory/memory restore, physical food handling,
physiology, social agents and ecological qualification remain open. This is
progress on the lifelong environment, not completion of the full objective.
''')
print(json.dumps({'archive':str(archive),'payloads':len(inventory),'bytes':sum(p.stat().st_size for p in archive.rglob('*') if p.is_file())}))
