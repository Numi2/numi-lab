from pathlib import Path
import gzip,hashlib,json,re,shutil,subprocess
root=Path('/Users/n/Developer/numi-crow-rl-20261107');runs=root/'.numi/runs/american-crow'
archive=root/'artifacts/crow-rl-recurrent-actor-20261107';assert sorted(p.name for p in archive.iterdir())==['inputs']
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def copy(a,b):
 b.parent.mkdir(parents=True,exist_ok=True);assert not b.exists();shutil.copyfile(a,b);assert sha(a)==sha(b)
inventory=[]
def tree(source,target):
 assert source.is_dir(),source
 for p in sorted(source.rglob('*')):
  if not p.is_file():continue
  assert not p.is_symlink(),p
  out=target/p.relative_to(source);out.parent.mkdir(parents=True,exist_ok=True)
  compressed=p.suffix in {'.bin','.f32','.f64','.u32','.rolloutpack','.safetensors','.lifecheckpoint'}
  if compressed:
   out=out.with_name(out.name+'.gz')
   with p.open('rb') as raw,out.open('xb') as final:
    with gzip.GzipFile(filename='',fileobj=final,mode='wb',mtime=0) as encoded:shutil.copyfileobj(raw,encoded)
   with gzip.open(out,'rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==sha(p)
  else:copy(p,out)
  inventory.append({'source':str(p.relative_to(root)),'archive':str(out.relative_to(archive)),'original_bytes':p.stat().st_size,'original_sha256':sha(p),'encoding':'gzip' if compressed else 'identity','bytes':out.stat().st_size,'sha256':sha(out)})
trials=[runs/f'recurrent-actor-native-{i:02}' for i in range(1,5)]+[runs/f'recurrent-actor-{name}-01' for name in ('admission','api-calibration','ordinary-learner','feeding-replay')]
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
final=json.loads((runs/'recurrent-actor-admission-01/summary.json').read_text());assert final['status']=='passed' and final['retained_runtime_future_exact']
assert final['native']['invalid_contract_controls']==7 and final['native']['publication_rejection_controls']==3 and final['native']['stale_policy_shader_rejected']
ordinary=json.loads((runs/'recurrent-actor-ordinary-learner-01/summary.json').read_text())
previous=json.loads((root/'artifacts/crow-rl-gru-memory-20261107/trials/gru-memory-ordinary-learner-01/summary.json').read_text())
assert [r['checkpoint_sha256'] for r in ordinary['records']]==[r['checkpoint_sha256'] for r in previous['records']]
feeding=json.loads((runs/'recurrent-actor-feeding-replay-01/summary.json').read_text());assert [r['fingerprint'] for r in feeding['records']]==['2255877855914022940','3082487550463979285']
verification=runs/'recurrent-actor-verification';tree(verification,archive/'verification');assert '100% tests passed out of 14' in (verification/'native.log').read_text();assert 'Ran 26 tests' in (verification/'python.stderr').read_text()
for text in re.findall(r'Evidence retained at ([^\s]+)',(verification/'LastTest.log').read_text()):
 p=Path(text);assert p.is_dir() and p.parent==root/'build';tree(p,archive/'native-ctest-evidence'/p.name)
for p in sorted(runs.glob('recurrent-actor-*.log')):copy(p,archive/'authoring-logs'/p.name)
receipts={}
for name in [f'recurrent-actor-native-{i:02}' for i in range(1,5)]+['recurrent-actor-admission-01']:
 directory=root/'.numi/retained-runtime'/name
 receipts[name]=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(directory.rglob('*')) if p.is_file()];assert receipts[name]
(archive/'runtime-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
old=archive/'inputs/PolicyABI3.metallib';(archive/'inputs/receipt.json').write_text(json.dumps({'path':str(old.relative_to(root)),'source':'.numi/retained-runtime/life-training-probe-05/shaders/MetalRobo.metallib','sha256':sha(old),'bytes':old.stat().st_size,'purpose':'retinal taskABI48 but policyABI3; reject recurrent execution for declared policyABI4 cause'},indent=2)+'\n')
source=archive/'source-final';source.mkdir();patch=subprocess.check_output(['git','diff','HEAD','--binary'],cwd=root);(source/'tracked.patch').write_bytes(patch);untracked={}
for name in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=root,text=True).splitlines():
 if Path(name).parts[0]=='artifacts':continue
 p=root/name
 if p.is_file():copy(p,source/'untracked'/name);untracked[name]=sha(p)
(source/'manifest.json').write_text(json.dumps({'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'tracked_patch_sha256':hashlib.sha256(patch).hexdigest(),'retained_untracked':untracked},indent=2)+'\n')
analysis={'schema':'numi.crow-recurrent-actor-archive.v1','status':'native-recurrent-actor-and-life-state-admitted','final_trial':final,
 'native_checks_passed':14,'python_checks_passed':26,'ordinary_learner_optimizer_hashes_unchanged':True,'ordinary_learner':ordinary,
 'feeding_fingerprints_unchanged':True,'feeding':feeding,
 'attempts':{'build01':'rejected Metal helper address-space mismatch before shader publication; compiler diagnostics retained',
 'native01':'all future streams and physical/food/GRU components exact; rejected retinal byte comparison because final endpoint camera processing differed; source/runtime retained',
 'native02':'aligned final camera capture in all arms without physical step/tolerance change; seven compiler negatives; passed all original comparisons',
 'native03':'added resident architecture-drop, clear and legacy-v7 publication controls, explicit component memory accounting; passed',
 'native04':'added required retained policyABI3 shader control; rejects specifically for ABI4 capability before execution; passed',
 'admission01':'passed Numi orchestration plus independently retained runtime future replay; source/runtime retained'},
 'retained_recovery':json.loads((runs/'recurrent-actor-admission-01/retained-recovery.json').read_text()),
 'failed_build_evidence_scope':'compiler diagnostics retained; exact source deltas for successful native versions, not every failed compiler edit',
 'runtime_scope':'retained native binary/library/core-retina-GRU shader trees on macmini; hashes/source archived; exact device/OS/runtime required',
 'recurrent_policy_and_native_actor_integrated':True,'coupled_life_includes_GRU':True,
 'recurrent_rollout_and_sequence_PPO_integrated':False,'embodied_memory_qualified':False,'full_loop_performance_qualified':False,'ecological_environment_qualified':False}
(archive/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n');(archive/'payload-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
copy(Path(__file__),archive/'archive-authoring.py');(archive/'.gitattributes').write_text('*.gz binary\n*.metallib binary\n*.patch -whitespace\n*.log -whitespace\n')
(archive/'README.md').write_text('''# Native recurrent actor and life-state evidence

The optional native crow actor now supports PolicyPack5 encoder→GRU→head,
policyABI4 shaders, captured pre-cell/reset traces and GRU state inside the
full v13 life checkpoint. Recurrent sequence PPO is not yet implemented;
legacy rollout7 publication explicitly rejects recurrent actors. This is an
authored supported engineering controller, not learned embodied memory.

Two real native crow lanes run40ticks, reset lane1 at9 and publish next GRU
weights at18 without a physical step. Actor/pre-cell FP64 error1.422e-8 is
below the prospective5e-6 bound. Chunks1/8 and extra intermediate previews
preserve real observations/actions/hidden/reset/transitions/q/v bit for bit.
A new process reproduces every future stream and all three final life-state
component blobs exactly. Seven invalid compiler contracts, three resident
publication controls and a stale policyABI3 shader reject. The stale shader
control checks the declared ABI4 cause, not merely any error.

Numi admission01 retains the matching native binary/library/shader tree and
independently restores the original checkpoint with it. All future stream
bytes match. That external retained child does not compare final component
blobs; the native admission's internal fresh-child test does. Matching device,
OS and native runtime are required. Receipts bind on-device runtimes; source
versions and compressed checkpoints are archived here.

14 native and26 Python checks pass. Ordinary feedforward learner model/Adam
hashes and ordinary/residual feeding replay fingerprints remain unchanged.
The first native attempt failed endpoint retinal byte comparison because one
arm deferred the final camera capture. Later arms align that endpoint without
an extra physical tick or changed numerical tolerance. All earlier streams,
sources/runtimes and failed compiler diagnostics remain preserved. analysis.json
distinguishes each attempt and scope.

Lossless gzip payloads retain original/archive SHA-256 and size in the inventory.
Decompress to original names before replay. Source patches apply to recorded
Git bases; final production sources remain in the owning repository paths.
No qualification seeds were used, no policy promoted, and no ecological or
production-throughput qualification is claimed. Contiguous sequence PPO,
coupled recurrent training resume and causal embodied memory tasks are next.
''')
print(json.dumps({'archive':str(archive),'files':len(inventory),'payload_bytes':sum(r['bytes'] for r in inventory),'status':'verified'}))
