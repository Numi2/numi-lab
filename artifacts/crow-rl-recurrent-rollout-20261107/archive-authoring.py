from pathlib import Path
import gzip,hashlib,json,re,shutil,subprocess
root=Path('/Users/n/Developer/numi-crow-rl-20261107');runs=root/'.numi/runs/american-crow'
archive=root/'artifacts/crow-rl-recurrent-rollout-20261107';archive.mkdir(exist_ok=False)
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
trials=[runs/f'recurrent-rollout-{name}-01' for name in ('native','swift','mlx')]+[runs/f'recurrent-rollout-admission-{i:02}' for i in range(1,5)]+[runs/f'recurrent-rollout-api-calibration-{i:02}' for i in range(1,4)]+[runs/f'recurrent-rollout-{name}-{i:02}' for name in ('ordinary-learner','feeding-replay') for i in (1,2)]
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
final=json.loads((runs/'recurrent-rollout-admission-04/summary.json').read_text())
assert final['status']=='passed' and final['retained_Swift_and_learner_summary_all_sequence_arrays_exact'] and final['retained_regular_CLI_sequence_summary_and_all_arrays_exact']
assert final['policy']['actor']['native']['publication_rejection_controls']==5 and final['policy']['actor']['native']['recurrent_rollout_rejection_controls']==10
assert final['swift']['rejection_controls']==10 and final['learner']['parser_and_flat_consumption_rejections']==11
assert final['regular_CLI_sequence']['MLX_full50tick_native_sequence_max_abs']<=5e-6 and final['regular_CLI_sequence']['continuation_zero_hidden_action_difference']>=1e-4
plan=json.loads((runs/'recurrent-rollout-admission-04/plan.json').read_text())
for receipt_name in ('retained-learner-source.json','retained-rollout-inputs.json'):
 for name,record in json.loads((runs/'recurrent-rollout-admission-04'/receipt_name).read_text()).items():
  key='recurrent_source_'+name if receipt_name=='retained-learner-source.json' else name
  assert record['sha256']==plan['preflight']['inputs'][key]['sha256']==sha(Path(record['path']))
ordinary=json.loads((runs/'recurrent-rollout-ordinary-learner-02/summary.json').read_text());previous=json.loads((root/'artifacts/crow-rl-recurrent-policy-20261107/trials/recurrent-policy-ordinary-learner-01/summary.json').read_text())
assert [r['checkpoint_sha256'] for r in ordinary['records']]==[r['checkpoint_sha256'] for r in previous['records']]
feeding=json.loads((runs/'recurrent-rollout-feeding-replay-02/summary.json').read_text());assert [r['fingerprint'] for r in feeding['records']]==['2255877855914022940','3082487550463979285']
verification=runs/'recurrent-rollout-verification';tree(verification,archive/'verification')
for log in ('native.log','native-final.log'):assert '100% tests passed out of 15' in (verification/log).read_text()
assert 'Ran 26 tests' in (verification/'python-current.stderr').read_text()
for log in ('LastTest.log','LastTest-final.log'):
 for text in re.findall(r'Evidence retained at ([^\s]+)',(verification/log).read_text()):
  p=Path(text);assert p.is_dir() and p.parent==root/'build';tree(p,archive/'native-ctest-evidence'/p.name)
for p in sorted(runs.glob('recurrent-rollout-*.log')):copy(p,archive/'authoring-logs'/p.name)
assert '[190/190]' in (runs/'recurrent-rollout-full-build-01.log').read_text() and '[109/109]' in (runs/'recurrent-rollout-full-build-02.log').read_text()
receipts={}
for name in ['recurrent-rollout-native-01']+[f'recurrent-rollout-admission-{i:02}' for i in range(1,5)]:
 directory=root/'.numi/retained-runtime'/name
 receipts[name]=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(directory.rglob('*')) if p.is_file() and '__pycache__' not in p.parts];assert receipts[name]
 if (directory/'python').is_dir():tree(directory/'python',archive/'retained-learner-source'/name)
(archive/'runtime-receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
source=archive/'source-final';source.mkdir();patch=subprocess.check_output(['git','diff','HEAD','--binary'],cwd=root);(source/'tracked.patch').write_bytes(patch);untracked={}
for name in subprocess.check_output(['git','ls-files','--others','--exclude-standard'],cwd=root,text=True).splitlines():
 if Path(name).parts[0]=='artifacts':continue
 p=root/name
 if p.is_file():copy(p,source/'untracked'/name);untracked[name]=sha(p)
(source/'manifest.json').write_text(json.dumps({'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'tracked_patch_sha256':hashlib.sha256(patch).hexdigest(),'retained_untracked':untracked},indent=2)+'\n')
analysis={'schema':'numi.crow-recurrent-rollout-archive.v1','status':'native-Swift-MLX-recurrent-record-handoff-admitted','final_trial':final,
 'native_checks_passed':15,'python_checks_passed':26,'full_incremental_builds_passed':2,
 'ordinary_learner_optimizer_hashes_unchanged':True,'ordinary_learner':ordinary,'feeding_fingerprints_unchanged':True,'feeding':feeding,
 'attempts':{'native01':'passed format8 export/native byte roundtrip, ten memory-descriptor controls and prior actor/life controls; both policy-segment endpoints aligned; exact compiled source/runtime retained after execution',
 'swift01':'passed concat chunk1/training append chunk8 segment bytes/final q/v; ten misuse controls; exact compiled source/runtime retained after execution',
 'mlx01':'passed all raw learner/hidden/reset arrays, MLX artifact sequence parity and eleven parser/flat-use controls; no separate pre-execution source snapshot for this direct invocation; matching module snapshot retained in admission01',
 'admission01':'passed native/policy/Swift/MLX handoff plus independent retained-runtime/source recovery on the40-tick fixture',
 'admission02':'adds ordinary CLI37+13ticks, chunks1/8 and retained CLI continuation; full50tick MLX parity and modeled zero-hidden counterfactual pass',
 'admission03':'adds independent retained Python source recovery of all regular-CLI sequence summary/arrays; training owner reserves update memory arena',
 'admission04':'adds resident changed-fingerprint revision uniqueness: reused/reversed revisions reject before publication; full build and all prior controls pass'},
 'wire_contract':'format8 memory ABI1; step-major real actor slots only; pre-cell float32[T,N,H], reset uint32[T,N]; previews excluded; one compiled generating policy fingerprint/revision per record',
 'legacy_contract':'unchanged C batch/dense writer; dense records remain byte-identical7; old writer and flat policy_batch explicitly reject recurrent use',
 'counterfactual_scope':'modeled MLX zero-initial-hidden continuation only; no native physical memory intervention or learned-memory claim',
 'retained_recovery_scope':'exact native binary/library/shader trees remain on macmini with archived source/size/hash receipts; retained Swift and five Python modules reproduce summaries/all sequence arrays/canonical bytes; exact device/OS and installed Python/MLX/NumPy required',
 'recurrent_rollout_format_integrated':True,'sequence_PPO_integrated':False,'coupled_recurrent_optimizer_life_resume_qualified':False,
 'embodied_memory_qualified':False,'full_loop_performance_qualified':False,'ecological_environment_qualified':False}
(archive/'analysis.json').write_text(json.dumps(analysis,indent=2)+'\n');(archive/'payload-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
copy(Path(__file__),archive/'archive-authoring.py');(archive/'.gitattributes').write_text('*.gz binary\n*.npz binary\n*.patch -whitespace\n*.log -whitespace\n')
(archive/'README.md').write_text('''# Recurrent rollout handoff and ordinary CLI continuation

Final admission04 connects native format8, Swift collection/publication and
Python/MLX decoding/forward sequence algebra. The optional section stores every
real pre-cell hidden state and episode reset mask; terminal previews are excluded.
Generating policy fingerprint/revision is bound per segment. Resident recurrent
changes require increasing revision, preventing mixed weights under one revision.
Dense writer/C batch remain unchanged and dense format7 bytes are preserved.
Legacy publication and flat policy_batch refuse recurrent data explicitly.

Native18/22-tick segments around weight publication, lane1 reset9 and extra
intermediate previews match native roundtrip and Swift concat chunk1/training
append chunk8 byte for byte, including final q/v. Python reproduces all raw
learner/hidden/reset arrays. MLX latents/pre-cell values match80 real slots within
1.491e-8 against the prospective5e-6 bound. Ten native malformed-memory exports,
ten Swift misuse and eleven parser/flat-consumption controls reject. Seven
compiler, five resident publication and a stale-shader control also pass.

Ordinary CLI uses unused development seed2651109501, runs37ticks and resumes13
in new processes at chunk1/8. Canonical bytes/fingerprints match. Lifetime clocks
are37 then50 with no warmup. A retained CLI independently restores the37-tick
checkpoint and reproduces the continuation. Complete50-tick MLX sequence parity
is1.491e-8 over100 real slots. Continuation hidden norm is0.2611; a modeled zero
hidden continuation changes the action mean by0.002862 against the prospective
minimum0.0001. This is neural algebra, not a native physical intervention or
learned-memory behavior. No qualification seeds are used or policy promoted.

Retained runtime/Swift/five Python modules independently reproduce fixture and
regular-CLI summaries, all sequence arrays and canonical policy/rollout bytes.
Sources bind to preflight hashes. Native runtimes remain on macmini with size/hash
receipts; exact device/OS/runtime and installed Python/MLX/NumPy are required.
Native internal child compares all three life-state components; external retained
native child compares future streams, as scoped in the nested receipts.

Two full incremental builds,15 native and26 Python checks pass. Ordinary
feedforward model/Adam hashes and frozen ordinary/residual feeding fingerprints
remain unchanged. Earlier direct/successful variants and their source evidence
limits are preserved in analysis.json. Raw records, corruption controls, process
receipts, checkpoints and source versions are retained here. Gzip payloads have
original/archive SHA-256 and sizes in the inventory; decompress to original names.

This admits collection/publication and forward sequence fidelity. Contiguous
sequence PPO/BPTT, recurrent optimizer/life checkpoint resume, causal embodied
memory tasks, physical manipulation/social learning and production performance
remain required. Existing training entry points still reject recurrent policies.
''')
print(json.dumps({'archive':str(archive),'files':len(inventory),'payload_bytes':sum(r['bytes'] for r in inventory),'status':'verified'}))
