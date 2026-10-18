#!/usr/bin/env python3
import hashlib, json, os, subprocess, sys, time
from pathlib import Path

root=Path('/Users/n/numi-human-separation-20261018/C/blood-transfer-208/transaction-fixture')
build=json.loads((root/'build-control-comparison-005.json').read_text())
binary=Path(build['binary'])
matter_source=Path(build['matter_source'])
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
run_id='native-control-comparison-005'
out=root/run_id
out.mkdir(parents=True, exist_ok=False)
env=os.environ.copy()
env['NUMI_MATTER_TEST_SOURCE']=str(matter_source)
env['NUMI_MATTER_TEST_SOURCE_SHA256']=sha(matter_source)
wrapper='/Users/n/numi-human-final-integration-publish-1397/tools/run_resting_job_exclusive.py'
python='/Users/n/numi-human-prep-venv-20261005/bin/python3'
cmd=[python,wrapper,'--',str(binary)]
inv={'schema':'numi.blood_mpm_transaction_fixture_invocation.v1','binary':str(binary),'binary_sha256':sha(binary),'runtime_library':build['runtime_library'],'runtime_library_sha256':sha(build['runtime_library']),'matter_source':str(matter_source),'matter_source_sha256':env['NUMI_MATTER_TEST_SOURCE_SHA256'],'handoff_source':build['source_handoff_msl'],'handoff_source_sha256':build['source_handoff_msl_sha256'],'handoff_flattened_msl':build['flattened_handoff_msl'],'handoff_flattened_msl_sha256':build['flattened_handoff_msl_sha256'],'wrapper':wrapper,'python':python,'command':cmd,'environment':{'NUMI_MATTER_TEST_SOURCE':env['NUMI_MATTER_TEST_SOURCE'],'NUMI_MATTER_TEST_SOURCE_SHA256':env['NUMI_MATTER_TEST_SOURCE_SHA256']},'timeout_seconds':240}
(out/'invocation.json').write_text(json.dumps(inv,indent=2)+'\n')
start=time.time()
try:
    proc=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=240)
    output=proc.stdout
    code=proc.returncode
    timed_out=False
except subprocess.TimeoutExpired as exc:
    output=(exc.stdout or '')
    if isinstance(output,bytes): output=output.decode(errors='replace')
    code=124
    timed_out=True
(out/'native.log').write_text(output)
result={'schema':'numi.blood_mpm_transaction_fixture_result.v1','exit_code':code,'timed_out':timed_out,'wall_seconds':time.time()-start,'log_sha256':hashlib.sha256(output.encode()).hexdigest()}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
print(output)
sys.exit(code)
