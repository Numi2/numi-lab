from pathlib import Path
import subprocess,json,os,hashlib,datetime
q=Path("/Users/n/cardboard-implementation-20261014-host-macmini");s=q/'snapshots/source-cohesive-quadrature-area-205';out=q/'analysis/source205-cohesive-quadrature-area-001'
cmd=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(q/'build-source205-area-001'),'--jobs','4','--target','numi-matter-cardboard-probe','--target','numi-matter-cardboard-cohesive-capacity-area-check']
(out/'build-command.json').write_text(json.dumps({'argv':cmd,'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':False},indent=2)+'\n')
with (out/'build.log').open('xb') as log:
 r=subprocess.run(cmd,env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'},stdout=log,stderr=subprocess.STDOUT)
report={'exit':r.returncode,'at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
p=q/'build-source205-area-001/source-build.json'
if p.exists():report.update(build_manifest_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),build_success=json.loads(p.read_bytes()).get('success'))
(out/'build-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report));raise SystemExit(r.returncode)
