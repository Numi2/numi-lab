from pathlib import Path
import hashlib,json,subprocess,time,datetime
B=Path('/Users/n/cardboard-implementation-20261014-host-macmini');P=B/'analysis/contact-armijo-energy-replay-499-001';O=B/'native/contact-barrier-energy-replay-501'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not O.exists()
assert sha(P/'contact_barrier_trial_replay_ccd')=='43104b2b91ec185312f253b5b09084e6f6cb894a95e521f66268979c4d7e8266'
assert sha(P/'contact_barrier_trial_replay_ccd.metallib')=='59f3e742b7d4d13529d620d94c045f266db6494996674a070af836921d45d2c8'
assert sha(P/'contact_barrier_trial_replay_ccd.metal')=='7a7c5ec3cc3bb3cb77a462c6890d1cc9395afd06ab31edc4f782b79f5c5b2ec1'
assert sha(P/'case-496-inputs-v4/manifest.json')=='b6c2e1872afe5f6e6685a57525fe91e604b20bf6ab5aa0fcb1acd546e80dea0f'
assert sha(P/'compiled-source-closure-ccd.json')=='474e1bff85417d65ca282e5e825d4332c52fec3c80d5f7d4fc4b794d922edced'
inputs=json.loads((P/'case-496-inputs-v4/manifest.json').read_text())
for rel,row in inputs['files'].items(): assert sha(P/'case-496-inputs-v4'/rel)==row['sha256']
closure=json.loads((P/'compiled-source-closure-ccd.json').read_text())
for row in closure['dependencies']: assert sha(Path(closure['source_root'])/row['relative_path'])==row['sha256']
processes=subprocess.run(['/bin/ps','-axo','pid=,comm='],capture_output=True,check=True).stdout.decode()
workers=[line for line in processes.splitlines() if any(x in line for x in ['/numi-matter-cardboard-probe','/numi-matter-cardboard-full-contact-armijo-check','/contact_barrier_trial_replay'])]
assert not workers,workers
O.mkdir(parents=True)
cmd=[str(P/'contact_barrier_trial_replay_ccd'),str(P/'contact_barrier_trial_replay_ccd.metallib'),str(P/'case-496-inputs-v4'),str(O/'output.bin')]
pre={'schema':'numi.cardboard.native501-readonly-launch.v1','started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':cmd,'workers':workers,'source_closure_sha256':sha(P/'compiled-source-closure-ccd.json'),'inputs_manifest_sha256':sha(P/'case-496-inputs-v4/manifest.json'),'host_sha256':sha(P/'contact_barrier_trial_replay_ccd'),'metallib_sha256':sha(P/'contact_barrier_trial_replay_ccd.metallib'),'runner_sha256':sha(Path(__file__)),'physical_transactions':0,'ccd_qualification':False}
(O/'prediction.json').write_text(json.dumps(pre,indent=2)+'\n')
t=time.monotonic();r=subprocess.run(cmd,capture_output=True,timeout=60);(O/'stdout.log').write_bytes(r.stdout);(O/'stderr.log').write_bytes(r.stderr)
receipt={**pre,'returncode':r.returncode,'wall_seconds':time.monotonic()-t,'outputs_sha256':{p.name:sha(p) for p in O.iterdir() if p.is_file()},'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'qualification':'Read-only production contact barrier energy and straight sweep to rounded endpoints, 24rows. No full production CCD/trajectory or physical qualification.'}
(O/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));raise SystemExit(r.returncode)
