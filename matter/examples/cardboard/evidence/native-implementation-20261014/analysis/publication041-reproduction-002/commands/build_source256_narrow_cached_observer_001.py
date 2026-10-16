from pathlib import Path
import json,hashlib,subprocess,os,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-legacy-program-primary-cache-observer-256-002';d=q/'analysis/source256-build-001';b=q/'build-source256-legacy-program-cache-observer-001';d.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(s/'frozen-source.json')=='258f9b543d18b7316636d9df7cf73b3028f9cafbbf7946697cfd03103da847c0'
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
a=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--cmake-source-relative','.','--jobs','2','--target','numi-matter-cardboard-probe']
with (d/'build.log').open('xb') as out:p=subprocess.run(a,env=env,cwd=s,stdout=out,stderr=subprocess.STDOUT)
r={'schema':'numi.cardboard.narrow-observer-build.v1','source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'argv':a,'exit_code':p.returncode,'GPU_execution':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
if p.returncode==0:
 new=b/'matter/shaders/NumiMatter.metallib';old=q/'build-source240-refined-regional-graph-003/matter/shaders/NumiMatter.metallib'
 r.update(new_metallib_sha256=sha(new),old_metallib_sha256=sha(old),metallib_byte_identical=sha(new)==sha(old))
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(p.returncode)
