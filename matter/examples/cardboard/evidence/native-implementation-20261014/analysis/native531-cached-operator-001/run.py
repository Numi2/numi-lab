from pathlib import Path
import json,hashlib,subprocess,os,datetime,time
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');C=Q/'solver/control17-root17-newton34-replay-531';S=Q/'snapshots/source-contact-damped-memory-198-001';P=Q/'analysis/native531-cached-operator-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not P.exists();P.mkdir()
tool=S/'matter/tools/cardboard_cached_operator_audit.py';assert sha(tool)=='4a7e9c0cbf09eee0ad1f8f5b2b09314a2c0789a908aaab530b56cfff6a1c82ff'
pred=json.loads((C/'prediction.json').read_text());capture=Path(pred['environment']['NM_FEM_OPERATOR_FD_CAPTURE_PATH']);assert sha(capture)=='c9a538bd6357136130fc3d971c7d2f1e61fc52bdef2a0737ba00790bd34d8ec2'
argv=[str(Q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(tool),'--capture',str(capture),'--prediction',str(C/'prediction.json'),'--receipt',str(C/'receipt.json'),'--stderr',str(C/'stderr.log'),'--source-snapshot',str(S),'--environment','0','--output',str(P/'operator')]
e=dict(os.environ);e.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']});e['PYTHONDONTWRITEBYTECODE']='1'
(P/'launch.json').write_text(json.dumps({'argv':argv,'tool_sha256':sha(tool),'capture_sha256':sha(capture),'native_transactions':0},indent=2)+'\n')
t=time.monotonic();timed=False;rc=None;err=None
try:
 with (P/'stdout.log').open('xb') as out,(P/'stderr.log').open('xb') as stderr:
  p=subprocess.run(argv,stdout=out,stderr=stderr,env=e,timeout=600);rc=p.returncode
except subprocess.TimeoutExpired:timed=True;err='Owned600sCPUtimeout'
r={'schema':'numi.cardboard.native531-operator-cpu-receipt.v1','exit_code':rc,'timed_out':timed,'error':err,'wall_seconds':time.monotonic()-t,'native_transactions':0,'qualification':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(P/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
