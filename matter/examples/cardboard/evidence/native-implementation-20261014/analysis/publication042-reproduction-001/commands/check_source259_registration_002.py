from pathlib import Path
import json,hashlib,subprocess,datetime
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-integrated-globalization-contact-memory-259-002';t=q/'analysis/source263-best-of-registration-repair-001/test_best_of_pipeline_registration.py';d=q/'analysis/source259-registration-regression-002';d.mkdir(exist_ok=False);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(s/'frozen-source.json')=='a9b8287bc59d44ceb71af3857762b95841a22572dbe02552c136d3a2918eb3d0'
assert sha(t)=='917ff5ba0f885b95af51c67d17a0257ef08094223ad34e314298892a27ea6ab9'
a=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(t),'--source',str(s)]
p=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE);(d/'stdout.log').write_bytes(p.stdout);(d/'stderr.log').write_bytes(p.stderr)
r={'schema':'numi.cardboard.source259-registration-regression.v1','argv':a,'exit_code':p.returncode,'source_frozen_sha256':sha(s/'frozen-source.json'),'test_sha256':sha(t),'scope':'Five literal runtime dispatches each have exactly one production pipeline registration and one Metal definition. Static regression only; enabled Runtime requires a native trajectory.','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n')
(d/'manifest.json').write_text(json.dumps({'inputs':{str(x.relative_to(q)):sha(x) for x in [Path(__file__),s/'frozen-source.json',s/'matter/src/runtime.mm',t]},'artifacts':{x.name:sha(x) for x in d.iterdir() if x.is_file()}},indent=2)+'\n')
print(json.dumps(r));print(p.stdout.decode());assert p.returncode==0
