from pathlib import Path
import hashlib,json,subprocess,time,datetime
B=Path('/Users/n/cardboard-implementation-20261014-host-macmini');P=B/'analysis/contact-scaled-nearest-point-overlay-187-002';S=B/'snapshots/source-portable-paper-reference-185';O=B/'native/contact-scaled-nearest-point-503'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not O.exists()
b=json.loads((P/'build-receipt.json').read_text())
expected={'host_sha256':'a33f96e086194a686e6a2381887565264fe045525345e176c39b8c8d6b8e18ee','kernel_sha256':'431b59711e8ce4860196d8f28e85250fca82e76d1fe55ddfa4a37544bf767a68','metallib_sha256':'ad1c9f749c1ab320c3234639ebc1ddecdbc5348626eb9ca16fa4c243716a97fc'}
for key,val in expected.items():assert b[key]==val
for n,key in [('geometry-replay','host_sha256'),('geometry_replay.metal','kernel_sha256'),('geometry-replay.metallib','metallib_sha256')]:assert sha(P/n)==expected[key]
f=json.loads((S/'frozen-source.json').read_text());assert sha(S/'frozen-source.json')==b['source185_frozen_sha256']
for rel,h in f['source_sha256'].items():assert sha(S/rel)==h
processes=subprocess.run(['/bin/ps','-axo','pid=,comm='],capture_output=True,check=True).stdout.decode();workers=[line for line in processes.splitlines() if any(x in line for x in ['/numi-matter-cardboard-probe','/numi-matter-cardboard-full-contact-armijo-check','/contact_barrier_trial_replay','/geometry-replay'])];assert not workers,workers
O.mkdir(parents=True)
cmd=[str(P/'geometry-replay'),str(P/'geometry-replay.metallib'),str(O/'output.bin')]
pre={'schema':'numi.cardboard.native503-readonly-launch.v1','argv':cmd,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runner_sha256':sha(__file__),'build_receipt_sha256':sha(P/'build-receipt.json'),'source_frozen_sha256':sha(S/'frozen-source.json'),**expected,'workers':workers,'physical_transactions':0,'prediction':'48 source-bound analytical nearest-feature cases, four physical scales, origin and translated coordinates. Candidate dimensionless barycentric/segment parameters and normalized nearest-point errors must be <=1e-3; preserve all original baseline failures.'}
(O/'prediction.json').write_text(json.dumps(pre,indent=2)+'\n');t=time.monotonic();r=subprocess.run(cmd,capture_output=True,timeout=60);(O/'stdout.log').write_bytes(r.stdout);(O/'stderr.log').write_bytes(r.stderr)
receipt={**pre,'returncode':r.returncode,'seconds':time.monotonic()-t,'outputs_sha256':{p.name:sha(p) for p in O.iterdir() if p.is_file()},'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'production_integration':False,'physical_qualification':False}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(r.returncode)
