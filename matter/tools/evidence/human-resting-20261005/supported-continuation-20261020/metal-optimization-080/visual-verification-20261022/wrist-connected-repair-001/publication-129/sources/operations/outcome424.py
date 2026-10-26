from pathlib import Path
import json,hashlib
D=Path('/Users/numi/Work/numi-human-connected-surface-069');P=D/'D/postfit-backtrack-424/run-001';j=json.loads((P/'launch.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();changes=[dict(path=k,launch_sha256=v,current_sha256=sha(k))for k,v in j['pins'].items()if sha(k)!=v];assert len(changes)==1 and Path(changes[0]['path']).name=='test_run_validated_local_backtracking_419.py'
runtime=[k for k in j['pins']if not Path(k).name.startswith('test_')];assert all(sha(k)==j['pins'][k]for k in runtime)
lines=(P/'execution.log').read_text().splitlines();last=json.loads(lines[-1]);assert last['status']=='RUNNER_EXITED'and last['returncode']==0
out=dict(status='CONSTRUCTION_TRIAL_REJECTED_NO_ZERO_SELF_CANDIDATE',wrapper_returncode=0,last_exact_self_pairs=10,exact_audit_count=8,selected_result_present=(P/'backtrack-373/selected-result.json').exists(),runtime_source_pins_unchanged=True,post_run_all_source_assertion_failed=True,changed_test_files=changes,reason='Test file changed during run; runtime pins unchanged, but all-source assertion correctly failed. Independent geometry still fails self-zero.',native_admission=False,accepted_physical_steps=0)
(P/'outcome.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

