from pathlib import Path
import json,hashlib,subprocess,os,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-refined-regional-graph-240-003';d=q/'analysis/source240-refined-native-component-build-004';b=q/'build-source240-refined-native-components-004'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(s/'frozen-source.json')=='bcb37b35391f2dc9ab04492cba44306825540028edb354b219493412ad638644'
f=json.loads((s/'frozen-source.json').read_text());cfg=json.loads((s/'configure-support-source.json').read_text());before={n:sha(s/n) for n in {**cfg['files'],**f['source_sha256']}}
d.mkdir();env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
targets=['numi-matter-fem-cgls-arena-8192-check','numi-matter-fgmres-true-residual-native-check','numi-matter-fem-regional-galerkin-native-check']
a=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--cmake-source-relative','.','--jobs','2']
for t in targets:a+=['--target',t]
(d/'command.json').write_text(json.dumps({'argv':a,'gpu_execution':False},indent=2)+'\n')
with (d/'build.log').open('xb') as out:p=subprocess.run(a,cwd=s,stdout=out,stderr=subprocess.STDOUT,env=env)
r={'schema':'numi.cardboard.source240-component-build.v1','argv':a,'exit_code':p.returncode,'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'all_inputs_unchanged':all(sha(s/n)==h for n,h in before.items()),'gpu_execution':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(p.returncode)
