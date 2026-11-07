from pathlib import Path
import gzip,hashlib,json,re,shutil,subprocess
root=Path('/Users/n/Developer/numi-crow-rl-20261107');runs=root/'.numi/runs/american-crow'
archive=root/'artifacts/crow-rl-gru-memory-20261107';archive.mkdir(exist_ok=False)
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
  compressed=p.suffix in {'.bin','.f32','.f64','.u32','.rolloutpack','.safetensors','.memorycheckpoint'}
  if compressed:
   out=out.with_name(out.name+'.gz')
   with p.open('rb') as raw,out.open('xb') as final:
    with gzip.GzipFile(filename='',fileobj=final,mode='wb',mtime=0) as encoded:shutil.copyfileobj(raw,encoded)
   with gzip.open(out,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==sha(p)
  else:copy(p,out)
  inventory.append({'source':str(p.relative_to(root)),'archive':str(out.relative_to(archive)),'original_bytes':p.stat().st_size,'original_sha256':sha(p),'encoding':'gzip' if compressed else 'identity','bytes':out.stat().st_size,'sha256':sha(out)})
trials=[runs/'gru-memory-native-01',runs/'gru-memory-mlx-01']+[runs/f'gru-memory-admission-{i:02}' for i in range(3,7)]+[runs/'gru-memory-api-calibration-01',runs/'gru-memory-ordinary-learner-01',runs/'gru-memory-feeding-replay-01']
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
final=json.loads((runs/'gru-memory-admission-06/summary.json').read_text())
assert final['status']=='passed' and final['native']['negative_controls']==31
assert final['native']['maximum_FP64_error']<=3e-6 and final['retained_runtime_recovery_exact']
assert final['learner']['maximum_gradient_error']<=3e-6 and final['learner']['compiled_eager_max_abs']==0
recovery=json.loads((runs/'gru-memory-admission-06/retained-recovery.json').read_text())
assert recovery['independent_learner_summary_and_all_arrays_exact']
ordinary=json.loads((runs/'gru-memory-ordinary-learner-01/summary.json').read_text())
previous=json.loads((root/'artifacts/crow-rl-training-life-20261107/trials/20261107-training-life-legacy-learner-01/summary.json').read_text())
assert [r['checkpoint_sha256'] for r in ordinary['records']]==[r['checkpoint_sha256'] for r in previous['records']]
feeding=json.loads((runs/'gru-memory-feeding-replay-01/summary.json').read_text())
assert [r['fingerprint'] for r in feeding['records']]==['2255877855914022940','3082487550463979285']
for name in ('gru-memory-verification','gru-memory-verification-final'):
 verification=runs/name;tree(verification,archive/'verification'/name)
 assert '100% tests passed out of 13' in (verification/'native.log').read_text()
 assert 'Ran 26 tests' in (verification/'python.stderr').read_text()
 for text in re.findall(r'Evidence retained at ([^\s]+)',(verification/'LastTest.log').read_text()):
  p=Path(text);assert p.is_dir() and p.parent==root/'build';tree(p,archive/'native-ctest-evidence'/p.name)
assert 'Ran 26 tests' in (runs/'gru-memory-verification-final/python-final.stderr').read_text()
for p in sorted(runs.glob('gru-memory-*.log')):copy(p,archive/'authoring-logs'/p.name)
receipts={}
for name in ['gru-memory-native-01']+[f'gru-memory-admission-{i:02}' for i in range(3,7)]:
 directory=root/'.numi/retained-runtime'/name
 receipts[name]=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(directory.rglob('*')) if p.is_file()]
 assert receipts[name]
# Preserve the exact earlier native/MLX component sources, including the old
# duplicate-reset-trace semantics. Later successful source deltas are in trials.
tree(root/'.numi/retained-runtime/gru-memory-native-01/source',archive/'earlier-native-source')
(archive/'runtime-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
source=archive/'source-final';source.mkdir();patch=subprocess.check_output(['git','diff','HEAD','--binary'],cwd=root);(source/'tracked.patch').write_bytes(patch)
untracked={}
for name in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=root,text=True).splitlines():
 if Path(name).parts[0]=='artifacts':continue
 p=root/name
 if p.is_file():copy(p,source/'untracked'/name);untracked[name]=sha(p)
(source/'manifest.json').write_text(json.dumps({'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'tracked_patch_sha256':hashlib.sha256(patch).hexdigest(),'retained_untracked':untracked},indent=2)+'\n')
analysis={'schema':'numi.crow-gru-component-archive.v1','status':'native-and-differentiable-MLX-component-admitted','final_trial':final,
 'retained_recovery':recovery,'native_checks_passed':13,'python_checks_passed':26,'ordinary_learner_optimizer_hashes_unchanged':True,
 'ordinary_learner':ordinary,'feeding_fingerprints_unchanged':True,'feeding':feeding,
 'attempts':{'native01':'passed weaker fixture; duplicate reset trace written to scratch instead of comparing; source/runtime retained',
 'mlx01':'passed native-layout/FP64 derivatives instrument',
 'build02':'rejected fixture span-template deduction before executable publication',
 'launch01and02':'wrong dispatcher path rejected before any trial was created',
 'admission03':'passed corrected duplicate/reset trace and24 controls; retained binaries used flat layout, not recovery-qualified',
 'admission04':'passed retained-layout recovery,24 controls; outer plan predates explicit recovery stopping detail',
 'build05':'rejected malformed fixture digest xor type before executable publication',
 'admission05':'passed31 controls and retained-runtime/source recovery; learner recovery summary compared',
 'admission06':'passed31 controls; retained-runtime future and checkpoint exact; independent retained learner summary and all arrays exact'},
 'failed_build_evidence_scope':'compiler diagnostics retained; exact source deltas for successful versions, not every failed edit',
 'runtime_recovery_scope':'retained native binary/library/shader tree and two learner source modules; installed matching Python/MLX/NumPy still required',
 'component_optional_no_production_actor_execution_changed':True,'full_loop_performance_qualified':False,
 'PolicyPack_native_actor_and_rollout_integrated':False,'recurrent_PPO_integrated':False,'coupled_life_includes_GRU':False,
 'trainable_crow_memory_qualified':False,'ecological_environment_qualified':False}
(archive/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
(archive/'payload-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
copy(Path(__file__),archive/'archive-authoring.py')
(archive/'.gitattributes').write_text('*.gz binary\n*.npz binary\n*.patch -whitespace\n*.log -whitespace\n')
(archive/'README.md').write_text('''# Native working-memory component evidence

Final trial06 admits an optional native reset-after GRU and matching MLX
sequence/BPTT algebra. It does not yet run inside a crow actor, PolicyPack,
recurrent PPO, rollout format or coupled life checkpoint. The current trained
feeding actor remains feedforward. No ecological or production-performance
qualification is claimed.

Two synthetic neural shapes (3/5/7 and4/257/65) run40 authored ticks, reset
lanes at11/23 and publish next weights at18. Native/FP64 error is2.067e-7
against a prospective3e-6 threshold. Duplicate inference including all reset
traces, previews, chunks1/8 and fresh-process restore reproduce every future
trace and final memory checkpoint exactly.31 controls reject malformed weights,
checksums, seven re-signed semantic state corruptions, buffers/offsets,
ownership, budget and GPU input/clock invalidity without partial hidden updates.

MLX/native error is at most2.087e-7. All parameter/input sequence derivatives
agree with independent FP64 reverse-mode within8.253e-9; twelve central
differences verify that reference within2.508e-12. Compiled/eager gradients
match exactly. Reset/truncation gradient boundaries, a causal input change,
eight objective-reducing SGD steps and four learner contract controls pass.
This is mathematical admission, not a learned embodied-memory outcome.

Final retained native runtime restores future state/checkpoint byte for byte;
a separate retained-source learner reproduces its summary and all arrays
exactly. Installed matching Python/MLX/NumPy remain required. Native runtime
files remain under .numi/retained-runtime on macmini; SHA/size receipts and
source versions are archived here. These binaries are not claimed as portable
or independently recoverable on another device/runtime.

13 native and26 Python checks pass. Ordinary learner-only model/Adam checkpoint
hashes and frozen ordinary/residual feeding fingerprints remain unchanged.
The initial weaker checks and all compiler/launcher diagnostics are retained;
analysis.json distinguishes each version and its evidence limits.

Trials include prospective plans, raw neural fixtures, native traces, independent
FP64 output, MLX tensors/gradients, component checkpoints and process receipts.
Lossless gzip files have original/archive sizes and SHA-256 entries in
payload-inventory.json. Decompress to restore original payload names. Source
deltas bind to the recorded Git base; final production source is in the owning
repository paths. No qualification seeds were used or candidate promoted.
''')
print(json.dumps({'archive':str(archive),'files':len(inventory),'compressed_payload_bytes':sum(r['bytes'] for r in inventory),'status':'verified'}))
