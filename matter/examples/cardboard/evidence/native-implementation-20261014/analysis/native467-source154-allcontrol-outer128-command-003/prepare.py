from pathlib import Path
import hashlib,json,shutil
B=Path('/Users/home/cardboard-implementation-20261014');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
P=B/'analysis/native467-source154-allcontrol-outer128-command-002';O=B/'analysis/native467-source154-allcontrol-outer128-command-003'
parity=B/'analysis/source154-default64-native466-parity-001'
assert json.loads((parity/'report.json').read_text())['parity_pass']
receipt=B/'solver/cgls-newton128-default64-parity-466/receipt.json';r=json.loads(receipt.read_text());assert r['exit_code']==2 and not r['timed_out'] and r['source_inputs_unchanged']
assert not json.loads((B/'analysis/native466-launch-guard-001/receipt.json').read_text())['space_abort']
O.mkdir(exist_ok=False);(O/'admission-inputs').mkdir()
cp=P/'native-467-command.json';cmd=json.loads(cp.read_text());sep=cmd.index('--')
additional=[]
inputs=[parity/'report.json',parity/'manifest.json',receipt,B/'analysis/native466-launch-guard-001/receipt.json',B/'analysis/source154-build-topology-erratum-001/manifest.json']
rows=[]
for i,p in enumerate(inputs):
 target=O/'admission-inputs'/f'admission-{i:02d}-{p.parent.name}-{p.name}';shutil.copy2(p,target);assert sha(target)==sha(p)
 additional += ['--input-file',str(target)];rows.append({'original':str(p),'copied':str(target),'sha256':sha(target)})
cmd[sep:sep]=additional;new=O/'native-467-command.json';new.write_text(json.dumps(cmd,indent=2)+'\n')
xs=[cmd[i+1] for i,v in enumerate(cmd) if v=='--input-file'];assert len(set(Path(x).name for x in xs))==len(xs)
(O/'admission.json').write_text(json.dumps({'schema':'numi.cardboard.native467-admission.v1','prior_command':str(cp),'prior_command_sha256':sha(cp),'new_command_sha256':sha(new),'full_attempt_default64_parity_passed':True,'native466_unmodified_inputs':True,'build_topology_difference_explicitly_retained':True,'only_external_evidence_inputs_added':True,'physical_argv_unchanged':True,'environment_unchanged':True,'inputs':rows,'gpu_execution':'not launched'},indent=2)+'\n')
runner=B/'run_native467.py';s=runner.read_text();(O/'prior-runner.py').write_text(s)
old="command_path = B / 'analysis/native467-source154-allcontrol-outer128-command-002/native-467-command.json'";assert old in s
s=s.replace(old,"command_path = B / 'analysis/native467-source154-allcontrol-outer128-command-003/native-467-command.json'")
s=s.replace("# Read-only build/input admission", "assert json.loads((B/'analysis/source154-default64-native466-parity-001/report.json').read_text())['parity_pass']\n\n# Read-only build/input admission")
compile(s,str(runner),'exec');runner.write_text(s);(O/'runner.py').write_text(s)
(O/'prepare.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps({'files':{str(p.relative_to(O)):sha(p) for p in sorted(O.rglob('*')) if p.is_file()}},indent=2)+'\n')
print(json.dumps({'command_sha256':sha(new),'input_count':len(xs)}))
