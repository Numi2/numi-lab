from pathlib import Path
import json,hashlib,subprocess,os,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-qualification-checkpoint-proof-221-003';b=q/'build-source221-checkpoint-proof-003';d=q/'analysis/source221-checkpoint-proof-build-002'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((s/'frozen-source.json').read_text());assert sha(s/'frozen-source.json')==json.loads((q/'analysis/source221-qualification-checkpoint-proof-003/overlay.json').read_text())['frozen_sha256']
assert all(sha(s/n)==h for n,h in f['source_sha256'].items());d.mkdir(exist_ok=False)
a=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--jobs','4','--target','numi-matter-cardboard-probe','--target','numi-matter-cardboard-coupon-resume-check']
(d/'command.json').write_text(json.dumps({'argv':a,'source_frozen_sha256':sha(s/'frozen-source.json'),'gpu_execution':False},indent=2)+'\n')
with (d/'build.log').open('xb') as out:r=subprocess.run(a,stdout=out,stderr=subprocess.STDOUT,env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'},cwd=s)
receipt={'exit_code':r.returncode,'source_unchanged':all(sha(s/n)==h for n,h in f['source_sha256'].items()),'log_sha256':sha(d/'build.log'),'build_source_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_execution':False}
(d/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True);sys.exit(r.returncode)
