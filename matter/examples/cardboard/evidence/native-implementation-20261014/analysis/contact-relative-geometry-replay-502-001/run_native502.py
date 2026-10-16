from pathlib import Path
import hashlib,json,subprocess,time,datetime
B=Path('/Users/n/cardboard-implementation-20261014-host-macmini');P=B/'analysis/contact-armijo-energy-replay-499-001';K=B/'analysis/contact-relative-geometry-replay-502-001';O=B/'native/contact-relative-geometry-replay-502'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not O.exists()
assert sha(P/'contact_barrier_trial_replay')=='97c0b4953a9e8b60058f62227a8febe0d9bc2cc843c1bf63a948b93377b5b4f4'
assert sha(K/'replay.metallib')=='c2a73db08c893bb775315fc6552eee64282cf6b02b87dad5cb34029678fd1c5f'
assert sha(K/'prediction.json')=='c108295275e202071579b8fa034801f0597ccb515607a88516f781d2a97807d1'
pred=json.loads((K/'prediction.json').read_text());assert sha(K/'contact_barrier_trial_replay.metal')==pred['kernel_sha256']
assert sha(P/'case-496-inputs-v3/manifest.json')==pred['case_manifest_sha256']
assert sha(P/'compiled-source-closure.json')==pred['source176_closure_sha256']
inputs=json.loads((P/'case-496-inputs-v3/manifest.json').read_text())
for rel,row in inputs['files'].items(): assert sha(P/'case-496-inputs-v3'/rel)==row['sha256']
closure=json.loads((P/'compiled-source-closure.json').read_text())
for row in closure['dependencies']: assert sha(Path(closure['source_root'])/row['relative_path'])==row['sha256']
processes=subprocess.run(['/bin/ps','-axo','pid=,comm='],capture_output=True,check=True).stdout.decode()
workers=[line for line in processes.splitlines() if any(x in line for x in ['/numi-matter-cardboard-probe','/numi-matter-cardboard-full-contact-armijo-check','/contact_barrier_trial_replay'])]
assert not workers,workers
O.mkdir(parents=True)
cmd=[str(P/'contact_barrier_trial_replay'),str(K/'replay.metallib'),str(P/'case-496-inputs-v3'),str(O/'output.bin')]
pre={'schema':'numi.cardboard.native502-readonly-launch.v1','started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':cmd,'workers':workers,'source_closure_sha256':sha(P/'compiled-source-closure.json'),'inputs_manifest_sha256':sha(P/'case-496-inputs-v3/manifest.json'),'host_sha256':sha(P/'contact_barrier_trial_replay'),'metallib_sha256':sha(K/'replay.metallib'),'build_receipt_sha256':sha(K/'build-receipt.json'),'runner_sha256':sha(Path(__file__)),'physical_transactions':0,'ccd_qualification':False}
(O/'prediction.json').write_text(json.dumps(pre,indent=2)+'\n')
t=time.monotonic();r=subprocess.run(cmd,capture_output=True,timeout=60);(O/'stdout.log').write_bytes(r.stdout);(O/'stderr.log').write_bytes(r.stderr)
receipt={**pre,'returncode':r.returncode,'wall_seconds':time.monotonic()-t,'outputs_sha256':{p.name:sha(p) for p in O.iterdir() if p.is_file()},'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'qualification':'Read-only query-relative contact energy/position diagnostic only; no production change, transaction, CCD or physical qualification.'}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(r.returncode)
