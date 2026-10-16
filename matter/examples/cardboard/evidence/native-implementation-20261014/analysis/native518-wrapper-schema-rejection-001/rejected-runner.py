from pathlib import Path
import json,hashlib,datetime,os,subprocess,time
Q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
P=Q/'analysis/native-damped-cgls-helper-replay-190-002';O=Q/'native/damped-cgls-kernel-replay-518'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
mp=P/'package-manifest.json';assert sha(mp)=='f363916f0edb0c9fd8b9ac319e15427426ceeec9efcc05bb31a464dde9412b53'
m=json.loads(mp.read_text())
for n,h in m['files'].items():assert sha(P/n)==h,n
S=Path(m['source_snapshot']);b=Path(m['source_build_manifest'])
assert sha(b)==m['source_build_manifest_sha256']
sb=json.loads(b.read_text())
for n,h in sb['source_sha256'].items():assert sha(S/n)==h,n
for n,h in sb['artifacts_sha256'].items():assert sha(n)==h,n
new=Q/'snapshots/source-contact-damped-cgls-memory-195-001'
assert sha(new/'matter/src/metal/base_cgls.metalinc')==sha(S/'matter/src/metal/base_cgls.metalinc')
workers=[x for x in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(n in x for n in ('numi-matter-','geometry-replay','contact_barrier_trial_replay','damped_cgls_helper_replay')) and not x.split()[1].startswith('Z')]
assert not workers,workers
O.mkdir();(O/'runner.py').write_bytes(Path(__file__).read_bytes())
(O/'preflight.json').write_text(json.dumps({'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'package_manifest_sha256':sha(mp),'package_files':len(m['files']),'source_build_sha256':sha(b),'source_files_verified':len(sb['source_sha256']),'actual_production_cgls_source_equals_integrated195':True,'physical_transactions':0,'scope':'Only actual production damped CGLS recurrence through independent fixture A/AT; no physical solve or acceptance certificate.','gpu_workers':workers},indent=2)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_MATTER_','NUMI_CARDBOARD_'))}
for opt in ['O3','O0']:
 out=O/opt;out.mkdir()
 cmd=[str(P/'build'/opt/'damped_cgls_helper_replay'),str(P/'build'/opt/'damped_cgls_fixture.metallib')]
 record={'argv':cmd,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'timeout_s':120}
 (out/'prediction.json').write_text(json.dumps(record,indent=2)+'\n')
 start=time.monotonic()
 with (out/'stdout.json').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
  try:r=subprocess.run(cmd,cwd=out,env=env,stdout=stdout,stderr=stderr,timeout=120);exit_code=r.returncode;timed=False
  except subprocess.TimeoutExpired:exit_code=None;timed=True
 record.update(exit_code=exit_code,timed_out=timed,wall_seconds=time.monotonic()-start,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stdout_sha256=sha(out/'stdout.json'),stderr_sha256=sha(out/'stderr.log'),physical_transactions=0)
 (out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
for n,h in m['files'].items():assert sha(P/n)==h,n
(O/'finished.json').write_text(json.dumps({'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'package_files_unchanged':len(m['files']),'physical_transactions':0},indent=2)+'\n')
