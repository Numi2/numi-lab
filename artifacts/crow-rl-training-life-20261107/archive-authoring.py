from pathlib import Path
import gzip,hashlib,json,re,shutil,subprocess
root=Path('/Users/n/Developer/numi-crow-rl-20261107');runs=root/'.numi/runs/american-crow'
archive=root/'artifacts/crow-rl-training-life-20261107';archive.mkdir(exist_ok=False)
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
  compressed=p.suffix in {'.bin','.f32','.u32','.rolloutpack','.safetensors','.lifecheckpoint','.native'}
  if compressed:
   out=out.with_name(out.name+'.gz')
   with p.open('rb') as raw,out.open('xb') as final:
    with gzip.GzipFile(filename='',fileobj=final,mode='wb',mtime=0) as encoded:shutil.copyfileobj(raw,encoded)
   with gzip.open(out,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==sha(p)
  else:copy(p,out)
  inventory.append({'source':str(p.relative_to(root)),'archive':str(out.relative_to(archive)),'original_bytes':p.stat().st_size,'original_sha256':sha(p),'encoding':'gzip' if compressed else 'identity','bytes':out.stat().st_size,'sha256':sha(out)})
trials=[runs/f'20261107-life-training-probe-{i:02}' for i in range(1,6)]+[runs/f'20261107-training-life-api-admission-{i:02}' for i in range(1,4)]+[runs/'20261107-training-life-legacy-learner-01',runs/'20261107-training-life-feeding-replay-01']+[runs/f'20261107-training-life-retained-replay-{i}' for i in ('04','05')]
for trial in trials:
 tree(trial,archive/'trials'/trial.name)
 for receipt in trial.rglob('*.receipt.json'):
  r=json.loads(receipt.read_text());stem=receipt.name.removesuffix('.receipt.json')
  for suffix,key in [('.json','stdout_sha256'),('.stderr','stderr_sha256')]:
   if key in r:assert sha(receipt.with_name(stem+suffix))==r[key],receipt
 source=trial/'source-inputs'
 if source.exists():
  m=json.loads((source/'manifest.json').read_text());assert sha(source/'tracked.patch')==m['tracked_patch_sha256']
  for name,expected in m['retained_untracked'].items():assert sha(source/'untracked'/name)==expected
# Validate every positive bundle's immutable payloads and commit marker in-place.
for index in (3,4):
 for manifest in trials[index].rglob('manifest.json'):
  value=json.loads(manifest.read_text())
  if value.get('schema')!='numi.training-life-checkpoint.v1' or any(p.startswith('negative-') for p in manifest.relative_to(trials[index]).parts):continue
  assert (manifest.parent/'COMMITTED').read_text()==sha(manifest)
  for name,row in value['files'].items():
   p=manifest.parent/name;assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
final=json.loads((trials[4]/'summary.json').read_text());assert final['status']=='passed' and len(final['rejected_controls'])==9
assert len(final['boundary_comparisons'])==6 and all(r['all_payloads_and_optimizer_tensors_bit_identical'] for r in final['boundary_comparisons'])
assert final['finite_resource_checks']['split']['consumed_micrograms_by_environment'][12]==20000
assert json.loads((trials[3]/'summary.json').read_text())['status']=='passed'
legacy=json.loads((trials[8]/'summary.json').read_text());assert legacy['status']=='passed'
# The expected historical strings are read from their owning previously committed receipt.
previous=json.loads((root/'artifacts/crow-rl-life-checkpoint-20261107/trials/20261107-life-learner-probe-01/summary.json').read_text())
assert [r['checkpoint_sha256'] for r in legacy['records']]==[r['checkpoint_sha256'] for r in previous['records']]
feeding=json.loads((trials[9]/'summary.json').read_text());assert [r['fingerprint'] for r in feeding['records']]==['2255877855914022940','3082487550463979285']
for trial in trials[10:]:assert json.loads((trial/'summary.json').read_text())['both_future_boundaries_bit_identical']
verification=runs/'20261107-training-life-verification';tree(verification,archive/'verification')
assert '100% tests passed out of 12' in (verification/'native.log').read_text()
assert 'Ran 26 tests' in (verification/'python.stderr').read_text() and 'OK' in (verification/'python.stderr').read_text()
for name in re.findall(r'Evidence retained at ([^\s]+)',(verification/'LastTest.log').read_text()):
 p=Path(name);assert p.is_dir() and p.parent==root/'build';tree(p,archive/'native-ctest-evidence'/p.name)
for pattern in ('training-life-build-*.log','20261107-life-training-probe-*.console.*'):
 for p in sorted(runs.glob(pattern)):copy(p,archive/'authoring-logs'/p.name)
receipts={}
for version in ('04','05'):
 runtime=root/f'.numi/retained-runtime/life-training-probe-{version}';rows=json.loads((runtime/'receipt.json').read_text())
 for row in rows:assert sha(Path(row['path']))==row['sha256']
 receipts[version]=rows
(archive/'runtime-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
source=archive/'source-final';source.mkdir();patch=subprocess.check_output(['git','diff','HEAD','--binary'],cwd=root);(source/'tracked.patch').write_bytes(patch)
name='apps/TrainingLifeCheckpoint.swift';copy(root/name,source/'untracked'/name)
(source/'manifest.json').write_text(json.dumps({'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'tracked_patch_sha256':hashlib.sha256(patch).hexdigest(),'retained_untracked':{name:sha(root/name)}},indent=2)+'\n')
analysis={'schema':'numi.crow-training-life-archive.v1','status':'native-v13-coupled-Swift-MLX-life-optimizer-resume-admitted','final_probe':final,'attempts_retained':5,'failed_attempts':[1,2,3],'first_attempt_scope':'planning failure before plan publication; exact source retained by API-admission01; failure receipt preserves provenance','native_api_attempts_retained':3,'native_checks_passed':12,'python_checks_passed':26,'ordinary_learner_checkpoint_hashes_unchanged':True,'ordinary_learner':legacy,'legacy_feeding':feeding,'matching_runtime_and_Python_versions_retained':['04','05'],'retained_runtime_replays_passed':True,'same_installed_interpreter_MLX_NumPy_required':True,'forced_crash_or_power_loss_qualified':False,'trainable_memory_qualified':False,'ecological_environment_qualified':False}
(archive/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
(archive/'payload-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
for name in ('retain-training-life-runtime.py','replay-retained-training-life.py'):copy(runs/name,archive/name)
copy(Path(__file__),archive/'archive-authoring.py')
(archive/'.gitattributes').write_text('*.gz binary\n*.mrvpack binary\n*.policypack binary\n*.patch -whitespace\n*.log whitespace=-blank-at-eof\n')
(archive/'README.md').write_text('''# Coupled life and optimizer training checkpoint evidence

V13 training now resumes the same native life with its exact next policy and MLX
model/Adam state. An exclusive directory bundles policy, optimizer, native life,
generating rollout, unselected deterministic candidate and a manifest. Payloads,
manifest and directory entries are fsynced before atomic SHA-bound COMMITTED
publication. Partial directories are retained and inadmissible. This exercises an
interrupted-directory model, not forced process-crash or power-loss qualification.

Trial05:16 lanes,37 ticks/update, two epochs, uneven minibatches97. Four reference
updates equal one two-update split plus two independent two-update continuations
at chunks1/8. At every shared optimizer boundary all rollout/policy/candidate
bytes, model/Adam tensors+metadata and three native components match bit for bit.
Accepted clocks37,74,111,148 and revisions1→3→5 have no restore warmup. Lane12
consumed20,000 micrograms before interruption and retains it with conserved food.
Nine declared-cause controls reject uncommitted/bad-marker/missing-or-corrupt
optimizer/changed-configuration/mixed-policy/wrong-worker/wrong-scheduler and
input-output-overwrite attempts. No qualification seeds or promotion are used.

Attempts01/02 retain planning and boolean-parser failures before context creation.
Attempt03 retains premature legacy telemetry rejection before resident execution.
Telemetry now publishes after accepted updates rather than adding a fake step.
04 passes original comparisons/seven negatives.05 adds exact clock/food checks,
two negatives and an imported learner-code closure identity. Both successful
binary/Python-source versions independently resume their saved original bundles
and reproduce both future optimizer boundaries using retained runtime copies.
The exact installed Python/MLX/NumPy distributions are still required and checked.
Earlier failed runtimes have source deltas/hashes, not every historical binary.

Checkpoint publication averages16.1–16.5 ms per stage in05 (small warm-cache
instrument, not sustained scaling). Twelve native and26 Python checks pass.
Ordinary learner-only training reproduces its prior optimizer checkpoint hashes;
frozen feeding ordinary/residual fingerprints remain2255877855914022940 and
3082487550463979285. No learning algorithm or native transition changed here.

All plans, failed stages, update-boundary bundles, generating rollouts, candidate
policies, source patches, raw tests, receipts and matching-runtime replays are
preserved. payload-inventory.json records original and archived sizes/SHA-256.
Decompress *.gz to the name without .gz and verify original_sha256 before use.
Native life and safetensors/rollout binary payloads are losslessly compressed.
Invalid negative-input directories deliberately remain invalid and separate.
Runtime-receipts.json locates native binaries/shaders and complete Python-source
copies on macmini. Use the matching saved lib in DYLD_LIBRARY_PATH and saved
python directory via --python-root. Do not mix binaries, worker code or installed
MLX/NumPy versions with old checkpoints. See retained replay plans for exact CLI.

This admits coupled persistence for the present supported development fixture.
The controller remains an MLP without trainable persistent memory. Biological
fidelity, solid-food manipulation, physiology, social learning, robust locomotion,
production performance and held-out ecological qualification remain open.
''')
print(json.dumps({'archive':str(archive),'payloads':len(inventory),'bytes':sum(p.stat().st_size for p in archive.rglob('*') if p.is_file()),'status':'all payload copies and lossless roundtrips verified'}))
