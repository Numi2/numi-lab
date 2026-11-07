from pathlib import Path
import gzip,hashlib,json,re,shutil,subprocess
root=Path('/Users/n/Developer/numi-crow-rl-20261107');runs=root/'.numi/runs/american-crow'
archive=root/'artifacts/crow-rl-recurrent-policy-20261107';archive.mkdir(exist_ok=False)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def copy(a,b):
 b.parent.mkdir(parents=True,exist_ok=True);assert not b.exists();shutil.copyfile(a,b);assert sha(a)==sha(b)
inventory=[]
def tree(source,target):
 assert source.is_dir(),source
 for p in sorted(source.rglob('*')):
  if not p.is_file() or '__pycache__' in p.parts:continue
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
trials=[runs/'recurrent-policy-native-01']+[runs/f'recurrent-policy-admission-{i:02}' for i in (1,2)]+[runs/f'recurrent-policy-{name}-01' for name in ('api-calibration','ordinary-learner','feeding-replay')]
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
final=json.loads((runs/'recurrent-policy-admission-02/summary.json').read_text());assert final['status']=='passed' and final['retained_learner_summary_arrays_and_policy_bytes_exact']
assert final['policy_interop']['contract_rejections']==17 and final['policy_interop']['canonical_recurrent_policies_byte_exact']==2 and final['policy_interop']['canonical_dense_control_byte_exact']
assert final['policy_interop']['MLX_native_sequence_max_abs']<=5e-6
plan=json.loads((runs/'recurrent-policy-admission-02/plan.json').read_text());sources=json.loads((runs/'recurrent-policy-admission-02/retained-learner-source.json').read_text())
for name,record in sources.items():assert record['sha256']==plan['preflight']['inputs']['recurrent_source_'+name]['sha256']==sha(Path(record['path']))
ordinary=json.loads((runs/'recurrent-policy-ordinary-learner-01/summary.json').read_text());previous=json.loads((root/'artifacts/crow-rl-recurrent-actor-20261107/trials/recurrent-actor-ordinary-learner-01/summary.json').read_text())
assert [r['checkpoint_sha256'] for r in ordinary['records']]==[r['checkpoint_sha256'] for r in previous['records']]
feeding=json.loads((runs/'recurrent-policy-feeding-replay-01/summary.json').read_text());assert [r['fingerprint'] for r in feeding['records']]==['2255877855914022940','3082487550463979285']
verification=runs/'recurrent-policy-verification';tree(verification,archive/'verification');assert '100% tests passed out of 14' in (verification/'native.log').read_text();assert 'Ran 26 tests' in (verification/'python-final.stderr').read_text()
for text in re.findall(r'Evidence retained at ([^\s]+)',(verification/'LastTest.log').read_text()):
 p=Path(text);assert p.is_dir() and p.parent==root/'build';tree(p,archive/'native-ctest-evidence'/p.name)
for p in sorted(runs.glob('recurrent-policy-*.log')):copy(p,archive/'authoring-logs'/p.name)
receipts={}
for name in ['recurrent-policy-admission-01','recurrent-policy-admission-02']:
 directory=root/'.numi/retained-runtime'/name;receipts[name]=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(directory.rglob('*')) if p.is_file()];assert receipts[name]
 tree(directory/'python',archive/'retained-learner-source'/name)
(archive/'runtime-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
source=archive/'source-final';source.mkdir();patch=subprocess.check_output(['git','diff','HEAD','--binary'],cwd=root);(source/'tracked.patch').write_bytes(patch);untracked={}
for name in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=root,text=True).splitlines():
 if Path(name).parts[0]=='artifacts':continue
 p=root/name
 if p.is_file():copy(p,source/'untracked'/name);untracked[name]=sha(p)
(source/'manifest.json').write_text(json.dumps({'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'tracked_patch_sha256':hashlib.sha256(patch).hexdigest(),'retained_untracked':untracked},indent=2)+'\n')
analysis={'schema':'numi.crow-recurrent-policy-archive.v1','status':'native-Python-policy-interface-admitted','final_trial':final,
 'native_checks_passed':14,'python_checks_passed':26,'ordinary_learner_optimizer_hashes_unchanged':True,'ordinary_learner':ordinary,
 'feeding_fingerprints_unchanged':True,'feeding':feeding,
 'attempts':{'native01':'direct policy interface instrument passed using actor-admission01 native arrays from parent milestone; input SHA receipts retained; no separate pre-execution source snapshot for this exploratory invocation',
 'admission01':'passed full native life/retained runtime and canonical policy/MLX parity/17 controls; retained learner recovery exact',
 'admission02':'same checks passed; adds explicit preflight-to-retained four-module source-hash binding'},
 'direct_native01_input_origin':'artifacts/crow-rl-recurrent-actor-20261107/trials/recurrent-actor-admission-01/native-evidence at parent commit7e70166d',
 'retained_recovery_scope':'native binary/library/shader trees on macmini; archived source/size/hash receipts; final native internal child compares all three life components and retained external child compares future streams; separate retained Python source reproduces summary/all arrays/canonical policies',
 'recurrent_policy_C_ABI':1,'legacy_C_struct_and_dense_policy_v4_bytes_unchanged':True,
 'recurrent_policy_parameter_reader_writer_integrated':True,'recurrent_rollout_and_sequence_PPO_integrated':False,
 'embodied_memory_qualified':False,'full_loop_performance_qualified':False,'ecological_environment_qualified':False}
(archive/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n');(archive/'payload-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
copy(Path(__file__),archive/'archive-authoring.py');(archive/'.gitattributes').write_text('*.gz binary\n*.npz binary\n*.patch -whitespace\n*.log -whitespace\n')
(archive/'README.md').write_text('''# Native recurrent policy interface evidence

PolicyPack5 now crosses the canonical C/Python artifact boundary. The additive
ABI1 memory descriptor leaves the existing C policy structure/writer unchanged.
Python validates encoder/GRU/head topology, tensor shapes, finiteness and GRU
FP32 accumulator bounds. Flat actor evaluation and all three feedforward
learner/initialization/projection entry points explicitly reject recurrent packs.
This admits the parameter interface, not recurrent PPO or embodied learning.

Final Numi admission02 reruns native crow actor/life persistence and independently
retained native continuation. Two recurrent policies and one dense control roundtrip
through Python/native writers byte for byte. MLX actions and pre-cell state match
all80 native observation slots within1.491e-8 against the prospective5e-6 bound,
including reset and a same-life weight publication boundary.17 controls reject
four flat uses, three writer errors, three raw C descriptor errors and seven
correctly re-signed semantic PolicyPack corruptions. All controls/raw files remain.

Retained native/Python source independently reproduces its summary, every
sequence array and all three canonical policy files exactly. Four source modules
are bound to preflight hashes. Matching installed Python/MLX/NumPy and exact
native device/OS/runtime are still required. Receipts identify on-device native
runtimes; source modules are archived here. This is not portable recovery on
arbitrary hardware. Native internal child checks all three final life component
blobs; retained external child checks future streams only, as recorded.

14 native and26 Python regression checks pass. Ordinary feedforward model/Adam
checkpoint hashes and frozen ordinary/residual feeding fingerprints remain
unchanged. The direct initial instrument and the earlier successful admission
are preserved with their input/source evidence limits in analysis.json. No
qualification seeds were used and no candidate was promoted.

Lossless gzip payloads retain original/archive SHA-256 and sizes in the inventory.
Decompress to original names before replay. Source deltas apply to their recorded
Git base. Contiguous recurrent rollout records, sequence PPO/BPTT, coupled
recurrent optimizer resume and causal embodied memory tasks remain required.
Legacy rollout7 publication still rejects recurrent actors explicitly.
''')
print(json.dumps({'archive':str(archive),'files':len(inventory),'payload_bytes':sum(r['bytes'] for r in inventory),'status':'verified'}))
