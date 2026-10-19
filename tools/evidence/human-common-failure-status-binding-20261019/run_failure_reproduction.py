import json,hashlib,os,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
base=Path('/Users/n/numi-human-separation-20261018/C/registered-organ-batch-243/native-run-joint-v3-tetgen-ratecap0-fg16-002/organ-batch/invocation.json')
d=json.loads(base.read_text()); run=r/'native-001';run.mkdir(exist_ok=False)
a=list(d['argv']);a[0]=str(r/'failure-reproduction/build/numi-human-native');a[4]=str(run/'output')
e=dict(d['environment']);e['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT']=str(run/'common-field-failure.json');e['DYLD_PRINT_LIBRARIES']='0'
cmd=d['launcher_argv'][:3]+a
record={'base_invocation':str(base),'base_invocation_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'argv':cmd,'environment':e,'binary_sha256':hashlib.sha256(Path(a[0]).read_bytes()).hexdigest(),'visual_verification':False}
(run/'invocation.json').write_text(json.dumps(record,indent=2)+chr(10))
t=time.monotonic()
with (run/'native.log').open('w') as log:
 p=subprocess.run(cmd,env={**os.environ,**e},stdout=log,stderr=subprocess.STDOUT,timeout=180)
x=json.loads((run/'common-field-failure.json').read_text());l=x['first_failure_latch']
checks={'expected_failure':p.returncode!=0,'matter_code':l['matter_code']==x['matter_code']==10,'matter_object':l['matter_status'][1]==x['matter_object_index'],'matter_index':l['matter_status'][2]==x['matter_failing_index'],'matter_diagnostics':l['matter_diagnostics']==x['matter_diagnostics']}
out={'checks':checks,'pass':all(checks.values()),'exit_code':p.returncode,'wall_seconds':time.monotonic()-t,'scope':'known rejected step; diagnostic status identity only, not new physical qualification'}
(run/'result.json').write_text(json.dumps(out,indent=2)+chr(10));print(json.dumps(out,indent=2))
assert out['pass']
