from pathlib import Path
import json,hashlib,subprocess,os,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-strict-refined-contact-observers-241-004';d=q/'analysis/source241-build-001';b=q/'build-source241-integrated-observers-001';d.mkdir();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(s/'frozen-source.json')=='f64e6c5b12a0d69b944e002eb4c499533ac9b6cd6b852a96d869ce053c41c1db'
f=json.loads((s/'frozen-source.json').read_text());cfg=json.loads((s/'configure-support-source.json').read_text());env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'};py=str(q/'venvs/cardboard-reference-185/bin/python3');commands=[]
a=[py,'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--cmake-source-relative','.','--jobs','2']
for target in ['numi-matter-cardboard-probe','numi-matter-cardboard-full-contact-armijo-check','numi-matter-cardboard-refined-graph-layout-check','numi-matter-fgmres-true-residual-native-check']:a+=['--target',target]
with (d/'build.log').open('xb') as out:p=subprocess.run(a,cwd=s,env=env,stdout=out,stderr=subprocess.STDOUT)
commands.append({'argv':a,'exit_code':p.returncode})
if p.returncode==0:
 cmds=[[str(b/'matter/numi-matter-cardboard-refined-graph-layout-check')],[str(b/'matter/numi-matter-cardboard-full-contact-armijo-check'),str(s/'matter/materials/nagasawa2013_medium_orthotropic.nmatter'),'--cpu-only'],[py,'-B',str(s/'matter/tools/test_cardboard_shipping_resolution_controls.py'),'--probe',str(b/'matter/numi-matter-cardboard-probe')]]
 cmds+=[[py,'-B',str(s/'matter/tools'/n)] for n in ['test_true_residual_certification_contract.py','test_cardboard_solver_environment.py','test_cardboard_stage_resolution_study.py','test_deformable_limit_trace_serializer.py','test_cardboard_cached_operator_audit.py']]
 for i,a in enumerate(cmds):
  p=subprocess.run(a,cwd=s,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE);(d/f'test-{i}.stdout').write_bytes(p.stdout);(d/f'test-{i}.stderr').write_bytes(p.stderr);commands.append({'argv':a,'exit_code':p.returncode})
  if p.returncode:break
r={'schema':'numi.cardboard.source241-build-cpu.v1','source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'commands':commands,'all_pass':len(commands)==9 and all(x['exit_code']==0 for x in commands),'all_inputs_unchanged':all(sha(s/n)==h for n,h in {**cfg['files'],**f['source_sha256']}.items()),'gpu_execution':False,'native_composed_trajectory_qualified':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(int(not r['all_pass']))
