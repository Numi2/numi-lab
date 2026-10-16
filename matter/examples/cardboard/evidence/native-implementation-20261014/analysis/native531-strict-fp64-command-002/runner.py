
from pathlib import Path
import argparse,datetime,hashlib,json,os,struct,subprocess,sys,time
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
P=Path(__file__).resolve().parent
CASE=Q/'solver/control17-root17-newton34-replay-531'
S=Q/'snapshots/source-contact-damped-memory-198-001'
PREP=Q/'analysis/source198-strict-layer-replay-preparation-001'
OUTPUT=Q/'analysis/native531-strict-fp64-replay-001'
PY=Q/'venvs/cardboard-reference-185/bin/python3'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');args=ap.parse_args()
 m=json.loads((P/'manifest.json').read_text())
 for n,h in m['files_sha256'].items():assert sha(P/n)==h,n
 assert sha(PREP/'manifest.json')=='7d0a496578e88064b68ec69f6b95c1f5b19fafc48110af67d1364eb9bde7f36a'
 prep=json.loads((PREP/'manifest.json').read_text())
 for n,h in prep['files_sha256'].items():assert sha(PREP/n)==h,n
 for n,h in prep['source_tools'].items():assert sha(n)==h,n
 assert sha(prep['base_tool']['path'])==prep['base_tool']['sha256']
 assert sha(S/'frozen-source.json')=='16474dca955f9a953f78cf211054c10967c8d31cd5daa8bf5d8476741ffe8cfc'
 frozen=json.loads((S/'frozen-source.json').read_text())
 for n,h in frozen['source_sha256'].items():assert sha(S/n)==h,n
 pred=json.loads((CASE/'prediction.json').read_text());receipt=json.loads((CASE/'receipt.json').read_text())
 assert receipt['execution_started'] and receipt['source_inputs_unchanged'] and not receipt['timed_out']
 assert receipt['exit_code']==2
 assert pred['source_snapshot']==str(S)
 env=pred['environment']
 assert env['NM_FEM_OPERATOR_FD_SELECTOR']=='17:34'
 assert env['NM_FEM_BASE_CAUCHY_FALLBACK_SELECTOR']=='all:3'
 assert env['NM_FEM_BASE_CGLS_DAMPING_KG']=='1e-3'
 capture=Path(env['NM_FEM_OPERATOR_FD_CAPTURE_PATH'])
 assert capture.is_file() and not capture.is_symlink() and capture.resolve().is_relative_to(CASE.resolve())
 with capture.open() as f:header=json.loads(next(f))
 assert header['record']=='header' and header['root']==17 and header['newton_iteration']==34
 assert header['environment_count']==2 and header['node_count_per_environment']==882 and header['tetrahedron_count_per_environment']==1920
 paths=[capture,CASE/'prediction.json',CASE/'receipt.json',CASE/'stderr.log',CASE/'output/manifest.json',PREP/'strict_layers.py',PREP/'manifest.json']
 inputs={str(p):sha(p) for p in paths}
 argv=[str(PY),'-B',str(PREP/'strict_layers.py'),'--case',str(CASE),'--output',str(OUTPUT/'layers'),'--environment','0']
 pre={'schema':'numi.cardboard.native531-strict-fp64-preflight.v1','argv':argv,'inputs_sha256':inputs,'reference_tools_sha256':prep['source_tools'],'source_files_verified':len(frozen['source_sha256']),'capture_selector':'17:34','execute_requested':args.execute,'native_transactions':0,'qualification':False}
 print(json.dumps(pre),flush=True)
 if not args.execute:return 0
 assert not OUTPUT.exists();OUTPUT.mkdir()
 (OUTPUT/'launch.json').write_text(json.dumps(pre,indent=2)+'\n')
 e=dict(os.environ);e.update({'PYTHONPATH':str(S/'matter/tools'),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','MKL_NUM_THREADS':'1'})
 start=time.monotonic();timed=False;error=None;rc=None
 try:
  with (OUTPUT/'stdout.log').open('xb') as out,(OUTPUT/'stderr.log').open('xb') as err:
   result=subprocess.run(argv,env=e,stdout=out,stderr=err,timeout=3600);rc=result.returncode
 except subprocess.TimeoutExpired:
  timed=True;error='Owned independent CPU reference reached3600s timeout; partial output retained.'
 except BaseException as exc:error=repr(exc)
 unchanged=all(sha(p)==h for p,h in inputs.items())
 result_path=OUTPUT/'layers/report.json';report=json.loads(result_path.read_text()) if result_path.is_file() else None
 completed=rc==0 and not timed and error is None and unchanged and report is not None
 if completed:
  try:
   scope=dict(report['scope']);dt=scope.pop('dt_s')
   assert scope=={'environment':0,'control':17,'newton':34,'tets':1920,'nodes':882}
   assert struct.pack('<f',dt)==struct.pack('<f',float(header['dt_seconds']))
   assert len(report['states'])==5 and all(v is False for v in report['six_qualification_axes'].values())
  except BaseException as exc:
   completed=False;error='Post-run scope validation failed: '+repr(exc)
 rec={'schema':'numi.cardboard.native531-strict-fp64-receipt.v1','exit_code':rc,'timed_out':timed,'error':error,'complete_diagnostic':completed,'inputs_unchanged':unchanged,'wall_seconds':time.monotonic()-start,'native_transactions':0,'qualification':False,'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 (OUTPUT/'receipt.json').write_text(json.dumps(rec,indent=2)+'\n')
 print(json.dumps(rec),flush=True)
 return 0 if completed else 2
if __name__=='__main__':raise SystemExit(main())
