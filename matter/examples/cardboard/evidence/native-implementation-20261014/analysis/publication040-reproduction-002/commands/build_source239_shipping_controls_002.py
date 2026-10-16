from pathlib import Path
import json,shutil,hashlib,subprocess,os,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');base=q/'snapshots/source-shipping-resolution-controls-239-001';s=q/'snapshots/source-shipping-resolution-controls-239-002';d=q/'analysis/source239-shipping-resolution-controls-002';b=q/'build-source239-shipping-resolution-controls-002'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((base/'frozen-source.json').read_text());cfg=json.loads((base/'configure-support-source.json').read_text());s.mkdir();d.mkdir()
for n,h in {**cfg['files'],**f['source_sha256']}.items():
 assert sha(base/n)==h,n;p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
n='matter/tools/test_cardboard_shipping_resolution_controls.py';p=s/n;t=p.read_text()
t=t.replace('"--shipping-preflight-only","--validate-only"]','"--shipping-preflight-only","--validate-only","--output","/tmp/numi-cardboard-validation-unused"]')
t=t.replace('"literature2009","--validate-only","--nx"','"literature2009","--validate-only","--output","/tmp/numi-cardboard-validation-unused","--nx"')
p.write_text(t);f['source_sha256'][n]=sha(p);f['source239_cli_test_correction']={'reason':'Required output argument was missing in the test invocation; validate-only does not write it. Prior failed report preserved.','prior_report':str(q/'analysis/source239-shipping-resolution-controls-001/report.json'),'production_driver_unchanged_from_001':True}
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n');(s/'configure-support-source.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
a=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--cmake-source-relative','.','--jobs','2','--target','numi-matter-cardboard-probe']
(d/'command.json').write_text(json.dumps({'argv':a,'gpu_execution':False},indent=2)+'\n')
with (d/'build.log').open('xb') as out:rc=subprocess.run(a,cwd=s,stdout=out,stderr=subprocess.STDOUT,env=env).returncode
tests=[]
if rc==0:
 cmds=[[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/n),'--probe',str(b/'matter/numi-matter-cardboard-probe')]]
 cmds += [[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools'/x)] for x in ['test_cardboard_solver_environment.py','test_cardboard_stage_resolution_study.py']]
 for i,a in enumerate(cmds):
  p=subprocess.run(a,cwd=s,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env);(d/f'test-{i}.stdout').write_bytes(p.stdout);(d/f'test-{i}.stderr').write_bytes(p.stderr);tests.append({'argv':a,'exit_code':p.returncode,'stdout_sha256':sha(d/f'test-{i}.stdout'),'stderr_sha256':sha(d/f'test-{i}.stderr')})
  if p.returncode:break
r={'schema':'numi.cardboard.source239-shipping-resolution-controls.v1','source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'build_exit_code':rc,'tests':tests,'all_pass':rc==0 and len(tests)==3 and all(x['exit_code']==0 for x in tests),'all_inputs_unchanged':all(sha(s/x)==h for x,h in {**cfg['files'],**f['source_sha256']}.items()),'gpu_execution':False,'actual_mesh_validation_pending':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(int(not r['all_pass']))
