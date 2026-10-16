from pathlib import Path
import json,hashlib,subprocess,os,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');base=q/'snapshots/source-strict-local-projection-main-237-001';s=q/'snapshots/source-refined-coupon-arenas-238-001';d=q/'analysis/source238-refined-coupon-build-001';b=q/'build-source238-refined-coupon-001';prep=q/'analysis/source238-refined-coupon-arena-preparation-001';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
d.mkdir(exist_ok=False);f=json.loads((base/'frozen-source.json').read_text());cfg=json.loads((base/'configure-support-source.json').read_text());changed=json.loads((prep/'changed-files.json').read_text())
assert len(changed)==8 and len(set(changed))==8
for n in f['source_sha256']:
 if n in changed:f['source_sha256'][n]=sha(s/n)
 else:assert sha(s/n)==f['source_sha256'][n],n
 if n in cfg['files']:cfg['files'][n]=sha(s/n)
f['source238_refined_coupon_arenas']={'base':str(base),'base_frozen_sha256':sha(base/'frozen-source.json'),'changed':changed,'scope':'checked contact-free static FEM coupon arenas at N32768/T131072; extended true residual requires this pure layout; coupled/freeBOX caps unchanged','physical_operator_changed':False,'loading_protocol_changed':False,'newton_or_linear_tolerances_changed':False}
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n');(s/'configure-support-source.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
a=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--cmake-source-relative','.','--jobs','2']
for t in ['numi-matter-cardboard-probe','numi-matter-true-residual-arena-plan-check','numi-matter-pure-fem-armijo-cgls-arena-plan-check','numi-matter-fem-cgls-arena-8192-check','numi-matter-fgmres-true-residual-native-check']:a+=['--target',t]
(d/'command.json').write_text(json.dumps({'argv':a,'gpu_execution':False},indent=2)+'\n')
with (d/'build.log').open('xb') as out:rc=subprocess.run(a,cwd=s,stdout=out,stderr=subprocess.STDOUT,env=env).returncode
tests=[]
if rc==0:
 cmds=[[str(b/'matter'/n)] for n in ['numi-matter-true-residual-arena-plan-check','numi-matter-pure-fem-armijo-cgls-arena-plan-check']]
 cmds += [[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools'/n)] for n in ['test_true_residual_certification_contract.py','test_cardboard_solver_environment.py','test_cardboard_stage_resolution_study.py']]
 for i,a in enumerate(cmds):
  p=subprocess.run(a,cwd=s,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env);(d/f'test-{i}.stdout').write_bytes(p.stdout);(d/f'test-{i}.stderr').write_bytes(p.stderr);tests.append({'argv':a,'exit_code':p.returncode,'stdout_sha256':sha(d/f'test-{i}.stdout'),'stderr_sha256':sha(d/f'test-{i}.stderr')})
  if p.returncode:break
report={'schema':'numi.cardboard.source238-refined-coupon-build.v1','source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'build_exit_code':rc,'tests':tests,'all_pass':rc==0 and len(tests)==5 and all(t['exit_code']==0 for t in tests),'all_inputs_unchanged':all(sha(s/n)==h for n,h in {**cfg['files'],**f['source_sha256']}.items()),'gpu_execution':False,'native_component_checks_pending':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);sys.exit(int(not report['all_pass']))
