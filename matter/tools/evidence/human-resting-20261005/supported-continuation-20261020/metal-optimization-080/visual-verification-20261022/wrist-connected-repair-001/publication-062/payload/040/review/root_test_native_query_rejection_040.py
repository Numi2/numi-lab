from pathlib import Path
import subprocess,json,hashlib
e=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");d=e/'bone-clearance-fit-040-shared-columns';p=d/'attempt-004'
script=d/'native-source-route-requery-001/run_candidate_query_040.py';out=d/'negative-gate-004-no-launch';assert not out.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cmd=["/Users/n/numi-human-prep-venv-20261005/bin/python",str(script),'--candidate-npz',str(p/'candidate.npz'),'--candidate-sha256',sha(p/'candidate.npz'),'--candidate-report',str(p/'fit-report.json'),'--candidate-report-sha256',sha(p/'fit-report.json'),'--output-dir',str(out)]
r=subprocess.run(cmd,capture_output=True,text=True)
assert r.returncode!=0
assert 'candidate basic geometry status is not passing' in r.stderr+r.stdout
assert not out.exists()
receipt={'status':'PASS_REJECTED_CANDIDATE_BLOCKED_BEFORE_NATIVE_LAUNCH','returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'output_directory_created':False,'native_launch':False,'script_sha256':sha(script),'candidate_sha256':sha(p/'candidate.npz'),'report_sha256':sha(p/'fit-report.json')}
(e/'root-negative-native-query-gate-040.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

