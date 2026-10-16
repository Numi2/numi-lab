from pathlib import Path
import json,subprocess,os,hashlib,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-refined-regional-graph-240-003';b=q/'build-source240-refined-native-components-004';d=q/'analysis/native579-582-refined-components-command-001';d.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(s/'frozen-source.json')=='bcb37b35391f2dc9ab04492cba44306825540028edb354b219493412ad638644'
assert sha(b/'source-build.json')=='96fe976fa419ced284a73c067ce8468e076e6d0e7dba8d52b702f9f8eff5832c'
assert (q/'solver/strict-state-source232-full-coupon-578/receipt.json').exists()
cases=[
 ('refined-coupon-cgls-579','numi-matter-fem-cgls-arena-8192-check',['--refined-coupon']),
 ('maximum-coupon-cgls-580','numi-matter-fem-cgls-arena-8192-check',['--maximum-coupon-arena']),
 ('refined-coupon-true-residual-581','numi-matter-fgmres-true-residual-native-check',['--refined-coupon']),
 ('refined-regional-native-regression-582','numi-matter-fem-regional-galerkin-native-check',[str(s/'matter/materials/starch2007_finite_glue_elastic.nmatter')])]
(d/'command-spec.json').write_text(json.dumps({'source':str(s),'build':str(b),'cases':cases,'gpu_workers':1,'qualification':'Manufactured native component checks; no target coupon or shipping trajectory is qualified.'},indent=2)+'\n')
results=[];env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
for name,target,args in cases:
 ps=subprocess.check_output(['ps','-axo','pid,comm'],text=True)
 live=[l for l in ps.splitlines() if l.split() and Path(l.split(maxsplit=1)[-1]).name.startswith('numi-matter-')]
 assert not live,live
 case=q/'native'/name;assert not case.exists()
 a=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_run.py'),'--source-snapshot',str(s),'--runtime-dir',str(b),'--build-provenance',str(b/'source-build.json'),'--case',str(case),'--timeout','180','--question','Do the expanded pure coupon arenas execute their bounded native kernels without changing the original physical operator or acceptance?','--prediction','No physical qualification. Preserve all failures. Exact authored dimensions and component outputs will be audited separately.','--input-file',str(d/'command-spec.json'),'--',str(b/'matter'/target),*args]
 (d/(name+'.argv.json')).write_text(json.dumps(a,indent=2)+'\n')
 with (d/(name+'.launcher.log')).open('xb') as out:p=subprocess.run(a,cwd=s,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=240)
 results.append({'case':str(case),'exit_code':p.returncode,'receipt_sha256':sha(case/'receipt.json') if (case/'receipt.json').exists() else None})
 print(json.dumps(results[-1]),flush=True)
r={'schema':'numi.cardboard.refined-native-components.v1','results':results,'all_pass':all(x['exit_code']==0 for x in results),'gpu_workers':1,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');sys.exit(int(not r['all_pass']))
