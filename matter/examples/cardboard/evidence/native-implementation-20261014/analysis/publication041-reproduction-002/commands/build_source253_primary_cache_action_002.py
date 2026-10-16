from pathlib import Path
import json,hashlib,subprocess,os,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-primary-cache-fd-action-253-003';d=q/'analysis/source253-build-002';b=q/'build-source253-primary-cache-action-002';d.mkdir();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(s/'frozen-source.json')=='0ea56b99f98a0dc371e50299db10e01ba41be5150c51c5e876d2bef044141289'
f=json.loads((s/'frozen-source.json').read_text());cfg=json.loads((s/'configure-support-source.json').read_text());env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'};py=str(q/'venvs/cardboard-reference-185/bin/python3');commands=[]
a=[py,'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--cmake-source-relative','.','--jobs','2']
for target in ['numi-matter-cardboard-probe','numi-matter-cardboard-full-contact-armijo-check','numi-matter-cardboard-refined-graph-layout-check','numi-matter-cardboard-contact-sample-compaction-check']:a+=['--target',target]
with (d/'build.log').open('xb') as out:p=subprocess.run(a,cwd=s,env=env,stdout=out,stderr=subprocess.STDOUT)
commands.append({'argv':a,'exit_code':p.returncode})
if p.returncode==0:
 cmds=[[str(b/'matter/numi-matter-cardboard-refined-graph-layout-check')],[str(b/'matter/numi-matter-cardboard-full-contact-armijo-check'),str(s/'matter/materials/nagasawa2013_medium_orthotropic.nmatter'),'--cpu-only'],[py,'-B',str(s/'matter/tools/test_cardboard_shipping_resolution_controls.py'),'--probe',str(b/'matter/numi-matter-cardboard-probe')]]
 cmds+=[[py,'-B',str(s/'matter/tools'/n)] for n in ['test_true_residual_certification_contract.py','test_cardboard_solver_environment.py','test_cardboard_stage_resolution_study.py','test_deformable_limit_trace_serializer.py','test_cardboard_cached_operator_audit.py','test_cardboard_primary_cache_action.py']]
 for i,a in enumerate(cmds):
  p=subprocess.run(a,cwd=s,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE);(d/f'test-{i}.stdout').write_bytes(p.stdout);(d/f'test-{i}.stderr').write_bytes(p.stderr);commands.append({'argv':a,'exit_code':p.returncode})
  if p.returncode:break
r={'schema':'numi.cardboard.source253-build-cpu.v1','source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'commands':commands,'all_pass':len(commands)==10 and all(x['exit_code']==0 for x in commands),'all_inputs_unchanged':all(sha(s/n)==h for n,h in {**cfg['files'],**f['source_sha256']}.items()),'gpu_execution':False,'native_composed_trajectory_qualified':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(int(not r['all_pass']))
