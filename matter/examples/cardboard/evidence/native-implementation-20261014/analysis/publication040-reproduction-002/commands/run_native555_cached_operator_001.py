from pathlib import Path
import json,hashlib,subprocess,os,datetime,time
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');C=Q/'solver/control19-newton3-resume-diagnostic-555';S=Q/'snapshots/source-qualification-checkpoint-proof-221-003';P=Q/'analysis/native555-cached-operator-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not P.exists();P.mkdir()
tool=S/'matter/tools/cardboard_cached_operator_audit.py'
capture=C/'diagnostics/capture-operator-fd-capture/capture';assert sha(capture)=='c0a111c510ef4c5891ace3bb58f65af48740733d7b46987be370561d1c22ab53'
argv=[str(Q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(tool),'--capture',str(capture),'--prediction',str(C/'prediction.json'),'--receipt',str(C/'receipt.json'),'--stderr',str(C/'stderr.log'),'--source-snapshot',str(S),'--environment','0','--output',str(P/'operator')]
e=dict(os.environ);e.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']});e['PYTHONDONTWRITEBYTECODE']='1'
(P/'launch.json').write_text(json.dumps({'argv':argv,'tool_sha256':sha(tool),'capture_sha256':sha(capture),'native_transactions':0},indent=2)+'\n')
t=time.monotonic();timed=False;rc=None;err=None
try:
 with (P/'stdout.log').open('xb') as out,(P/'stderr.log').open('xb') as stderr:rc=subprocess.run(argv,stdout=out,stderr=stderr,env=e,timeout=600).returncode
except subprocess.TimeoutExpired:timed=True;err='Owned600sCPUtimeout'
r={'schema':'numi.cardboard.native555-operator-cpu-receipt.v1','exit_code':rc,'timed_out':timed,'error':err,'wall_seconds':time.monotonic()-t,'native_transactions':0,'qualification':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(P/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
