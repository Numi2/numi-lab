from pathlib import Path
import gzip,hashlib,json,re,shutil,subprocess
root=Path('/Users/n/Developer/numi-crow-rl-20261107')
runs=root/'.numi/runs/american-crow';archive=root/'artifacts/crow-rl-life-checkpoint-20261107'
archive.mkdir(exist_ok=False)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def copy(a,b):
 b.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(a,b);assert sha(a)==sha(b)
inventory=[]
def tree(source,target):
 assert source.is_dir(),source
 for p in sorted(source.rglob('*')):
  if not p.is_file():continue
  assert not p.is_symlink(),p
  out=target/p.relative_to(source);out.parent.mkdir(parents=True,exist_ok=True)
  compressed=p.suffix in {'.bin','.f32','.u32','.rolloutpack','.safetensors','.lifecheckpoint'}
  if compressed:
   out=out.with_name(out.name+'.gz')
   with p.open('rb') as raw,out.open('xb') as final:
    with gzip.GzipFile(filename='',fileobj=final,mode='wb',mtime=0) as encoded:shutil.copyfileobj(raw,encoded)
   with gzip.open(out,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==sha(p)
  else:copy(p,out)
  inventory.append({'source':str(p.relative_to(root)),'archive':str(out.relative_to(archive)),'original_bytes':p.stat().st_size,'original_sha256':sha(p),'encoding':'gzip' if compressed else 'identity','bytes':out.stat().st_size,'sha256':sha(out)})
trials=[runs/f'20261107-life-admission-{i:02}' for i in range(1,7)]+[runs/f'20261107-life-swift-probe-{i:02}' for i in (1,2)]+[runs/'20261107-life-api-admission-01',runs/'20261107-life-learner-probe-01',runs/'20261107-life-feeding-replay-01']
for trial in trials:
 tree(trial,archive/'trials'/trial.name)
 for receipt in trial.rglob('*.receipt.json'):
  r=json.loads(receipt.read_text());stem=receipt.name.removesuffix('.receipt.json')
  for suffix,key in [('.json','stdout_sha256'),('.stderr','stderr_sha256')]:
   if key in r:assert sha(receipt.with_name(stem+suffix))==r[key],receipt
 source=trial/'source-inputs';m=json.loads((source/'manifest.json').read_text())
 assert sha(source/'tracked.patch')==m['tracked_patch_sha256']
 for name,expected in m['retained_untracked'].items():assert sha(source/'untracked'/name)==expected
final=json.loads((trials[5]/'summary.json').read_text());assert final['status']=='passed' and final['whole_v13_development_life_restore'] and not final['policy_publication_advances_physics']
assert final['negative_controls_rejected']==10 and final['all_hidden_native_retinal_resource_state_bit_identical']
assert json.loads((trials[4]/'summary.json').read_text())['status']=='passed'
swift=json.loads((trials[7]/'summary.json').read_text());assert swift['status']=='passed'
assert len({r['fingerprint'] for r in swift['records']})==1
for row in swift['records']:assert row['ledger']['accepted_control_ticks_by_environment']==[167,167]
api=json.loads((trials[8]/'summary.json').read_text());assert api['status']=='passed'
learning=json.loads((trials[9]/'summary.json').read_text());assert learning['status']=='passed' and learning['records'][1]['learner_state_restored']
assert [(r['initial_revision'],r['final_revision']) for r in learning['records']]==[(1,3),(3,4)]
replay=json.loads((trials[10]/'summary.json').read_text());assert [r['fingerprint'] for r in replay['records']]==['2255877855914022940','3082487550463979285']
verification=runs/'20261107-life-verification';tree(verification,archive/'verification')
assert '100% tests passed out of 12' in (verification/'native.log').read_text()
assert 'Ran 19 tests' in (verification/'python.stderr').read_text() and 'OK' in (verification/'python.stderr').read_text()
for version in ('05','06'):
 assert json.loads((verification/f'retained-runtime-{version}.stdout').read_text())['status']=='passed'
 tree(runs/f'20261107-life-retained-replay-{version}',archive/'retained-runtime-replays'/version)
last=verification/'LastTest.log'
for name in re.findall(r'Evidence retained at ([^\s]+)',last.read_text()):
 p=Path(name);assert p.is_dir() and p.parent==root/'build';tree(p,archive/'native-ctest-evidence'/p.name)
for p in sorted(runs.glob('world-checkpoint-build-*.log')):copy(p,archive/'build-logs'/p.name)
receipts={}
for version,path in [('05',root/'.numi/retained-runtime/life-admission-05'),('06',root/'.numi/retained-runtime/life-admission-06-final')]:
 rows=json.loads((path/'receipt.json').read_text())
 for row in rows:assert sha(Path(row['path']))==row['sha256']
 receipts[version]=rows
(archive/'runtime-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
# Preserve the final reviewable source delta as well as each historical trial.
source=archive/'source-final';source.mkdir()
patch=subprocess.check_output(['git','diff','HEAD','--binary'],cwd=root);(source/'tracked.patch').write_bytes(patch)
untracked={}
for name in ['apps/CrowRetinalFixture.hpp','apps/crow_life_checkpoint_check.mm','cmake/CheckCrowLifeCheckpoint.cmake','include/metalrobo/detail/Checkpoint.hpp']:
 copy(root/name,source/'untracked'/name);untracked[name]=sha(root/name)
(source/'manifest.json').write_text(json.dumps({'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'tracked_patch_sha256':hashlib.sha256(patch).hexdigest(),'retained_untracked':untracked},indent=2)+'\n')
analysis={'schema':'numi.crow-life-checkpoint-archive.v1','status':'native-v13-development-life-restoration-admitted','native_life':final,'swift':swift,'api_regression':api,'learner_only_regression':learning,'legacy_fingerprints':replay['records'],'native_checks_passed':12,'python_checks_passed':19,'native_attempts_retained':6,'failed_native_attempts':[1,2,3,4],'swift_attempts_retained':2,'failed_swift_attempts':[1],'retained_runtime_versions':['05','06'],'retained_runtime_replays_passed':True,'source_base':json.loads((trials[5]/'plan.json').read_text())['preflight']['source_commit'],'scope':'exact-runtime native-v13 supported fixed development collision proxies; complete present native state, no trainable persistent memory','combined_learner_life_resume':False,'trainable_memory_qualified':False,'ecological_environment_qualified':False}
(archive/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
(archive/'payload-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
copy(Path(__file__),archive/'archive-authoring.py')
(archive/'.gitattributes').write_text('*.gz binary\n*.mrvpack binary\n*.policypack binary\n*.patch -whitespace\n*.log whitespace=-blank-at-eof\n')
(archive/'README.md').write_text('''# Native v13 life checkpoint evidence

Native world/contact/reset/task/RNG history, pending/visible retinal state and
finite food now survive a process restart in the current supported development
crow fixture. Trial06 snapshots tick135 after consumed food, with a frame pending
for tick136, and next revision20 published without advancing physics. Independent
processes continue32 ticks at chunks1/8 with one per-lane reset. All future actor,
action, latent, transition, final q/v and food bytes match the uninterrupted path.
All three final hidden components also match. Ten invalid restores reject without
published mutation; existing-file save rejection also passes. Swift probe02
resumes135→167 without warmup and with identical partition results/food.
The image-dependent stochastic policy is authored, not learned memory.

Attempts01–04 retain their failures: fixture revision error; policy immutable
heap growth clearing a still-valid resident world's upload marker; and raw
flapping-program hashing incorporating eight nonsemantic alignment bytes.
Trial05 passes after those fixes. Trial06 additionally covers publishing next
policy weights between accepted transitions. Swift01 retains a skipped policy
warmup erroneously falling into host-action warmup;02 has the corrected branch.
No acceptance threshold was weakened. All12 native and19 Python checks pass.
Frozen legacy ordinary/residual replay fingerprints remain2255877855914022940
and3082487550463979285. A16-lane Swift/MLX learner regression publishes1→3→4,
but its resumed learner process starts a fresh world. Optimizer/life combined
resume remains pending. No qualification seeds were used or policy promoted.

The checkpoint binds exact library/shaders/device/OS, physical/task/sensor and
next-policy/resource identities. It is not a portable future-version format.
C struct sizes/ABI47 stay fixed; C++ clients must rebuild for the appended
opt-in context option. Save/restore require exclusive quiescent caller ownership.
File and nested payload length/SHA guards detect corruption, not malicious
rewriting. Invalid components are restored into candidates before publication.
Writes are exclusive and fsync file plus directory; partial files are retained.

Payload-inventory.json records original and archived lengths/SHA-256. Recover
*.gz to the name without .gz using gzip.decompress and verify original_sha256.
The directory tree, source deltas and every failed attempt remain separate.
Runtime-receipts.json identifies matching retained binaries for05 and06 on
macmini. Both retained copies passed independent native replay. Executables
require the matching saved lib directory in DYLD_LIBRARY_PATH and matching
core/retinal shaders; do not substitute a current library for an old checkpoint.
Failed earlier versions retain source/runtime hashes, not all old executables.
Build logs include the failed missing-include build. Runtime files themselves
remain on macmini; receipts and research evidence are versioned here.

This qualifies development state persistence, not lifelong embodied intelligence.
Trainable persistent memory, combined optimizer/life resume, robust locomotion,
solid-food handling, physiology, social agents, biological fidelity and untouched
held-out ecology qualification remain open. See docs/CROW_RL_ENVIRONMENT.md.
''')
print(json.dumps({'archive':str(archive),'payloads':len(inventory),'bytes':sum(p.stat().st_size for p in archive.rglob('*') if p.is_file()),'status':'all copies and lossless roundtrips verified'}))
