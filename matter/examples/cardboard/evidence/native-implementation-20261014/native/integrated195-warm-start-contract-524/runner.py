from pathlib import Path
import json,hashlib,datetime,subprocess,os,time
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');D=Q/'build-source195-001';S=Q/'snapshots/source-contact-damped-cgls-memory-195-001';P=Q/'native/integrated195-warm-start-contract-524';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
b=json.loads((D/'source-build.json').read_text());f=json.loads((S/'frozen-source.json').read_text());assert b['success'] and b['source_sha256']==f['source_sha256']
for n,h in b['artifacts_sha256'].items():assert sha(n)==h,n
workers=[x for x in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(k in x for k in ('numi-matter-','geometry-replay','contact_barrier_trial_replay','damped_cgls_helper_replay')) and not x.split()[1].startswith('Z')];assert not workers,workers
P.mkdir();(P/'runner.py').write_bytes(Path(__file__).read_bytes())
cmd=[str(D/'numi-matter-cardboard-warm-start-check')]
r={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_build_sha256':sha(D/'source-build.json'),'argv':cmd,'physical_transactions':0,'scope':'Native private warm-start kernel contract only; no physical solve.'};(P/'prediction.json').write_text(json.dumps(r,indent=2)+'\n');start=time.monotonic()
with (P/'stdout.log').open('xb') as out,(P/'stderr.log').open('xb') as err:
 try:p=subprocess.run(cmd,cwd=P,env={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_MATTER_','NUMI_CARDBOARD_'))},stdout=out,stderr=err,timeout=120);code=p.returncode;timed=False
 except subprocess.TimeoutExpired:code=None;timed=True
r.update(exit_code=code,timed_out=timed,wall_seconds=time.monotonic()-start,stdout_sha256=sha(P/'stdout.log'),stderr_sha256=sha(P/'stderr.log'))
for n,h in b['artifacts_sha256'].items():assert sha(n)==h,n
(P/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
