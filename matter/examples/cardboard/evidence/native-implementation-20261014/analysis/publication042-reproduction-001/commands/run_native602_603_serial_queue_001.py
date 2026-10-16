from pathlib import Path
import json,hashlib,time,subprocess,datetime,os
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/native602-603-serialized-queue-001';d.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=q/'run_native603_composed254_regression_001.py';expected=sha(p)
spec={'schema':'numi.cardboard.native602-603-serialized-queue.v1','wait_for':str(q/'analysis/native602-command-001/receipt.json'),'next_script':str(p),'next_script_sha256':expected,'gpu_workers':1,'scope':'Independent canonical254 small complete-history regression after actual-state observer602 closes. Failure of either remains explicit and does not qualify full target.'}
(d/'queue-spec.json').write_text(json.dumps(spec,indent=2)+'\n');print(json.dumps({'queue_waiting_for602':True,'pid':os.getpid()}),flush=True)
deadline=time.monotonic()+7500
while not Path(spec['wait_for']).exists():
 if time.monotonic()>deadline:
  (d/'timeout.json').write_text(json.dumps({'reason':'No602 wrapper receipt within7500s;603 not launched'})+'\n');raise SystemExit(124)
 time.sleep(2)
assert sha(p)==expected
live=[l for l in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if len(l.split())>=3 and Path(l.split(maxsplit=2)[2]).name.startswith('numi-matter-') and not l.split()[1].startswith('Z')];assert not live,live
start=time.monotonic()
with (d/'native603.launch.log').open('xb') as out:res=subprocess.run(['/usr/bin/python3',str(p)],stdout=out,stderr=subprocess.STDOUT,timeout=1020)
r={'schema':'numi.cardboard.native602-603-queue-result.v1','native602_receipt_sha256':sha(spec['wait_for']),'native603_script_exit_code':res.returncode,'native603_wall_seconds':time.monotonic()-start,'gpu_workers':1,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
