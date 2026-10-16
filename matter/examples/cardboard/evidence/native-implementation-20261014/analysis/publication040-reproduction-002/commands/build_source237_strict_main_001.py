from pathlib import Path
import json,hashlib,shutil,subprocess,os,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';base=q/'snapshots/source231-upstream-compatibility-234-001';donor=q/'snapshots/source-strict-local-projection-upstream-232';old=q/'snapshots/source230-upstream-compatibility-231-001';s=q/'snapshots/source-strict-local-projection-main-237-001';d=q/'analysis/source237-strict-main-composition-001';b=q/'build-source237-strict-main-001';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip();assert parent=='577ae2ff37d1a3834f8ffe851ec15d296226d41b'
assert not subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=r)
assert sha(base/'frozen-source.json')=='f1acaea6d4838d3036f8f29d7217f935d7fa560eaecb16ec9fcb157a082ce896'
assert sha(donor/'frozen-source.json')=='dce16c539dbc01a1dad5be25b4c4f6b31594f9f7b0704415d175cc73345981f9'
f=json.loads((base/'frozen-source.json').read_text());cfg=json.loads((base/'configure-support-source.json').read_text());of=json.loads((old/'frozen-source.json').read_text());df=json.loads((donor/'frozen-source.json').read_text())
changes=[n for n,h in df['source_sha256'].items() if of['source_sha256'].get(n)!=h];assert len(changes)==8
s.mkdir(exist_ok=False);d.mkdir(exist_ok=False)
for n,h in {**cfg['files'],**f['source_sha256']}.items():
 assert sha(base/n)==h,n;p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
for n in changes:
 assert sha(base/n)==of['source_sha256'][n],n
 assert sha(donor/n)==df['source_sha256'][n],n
 shutil.copy2(donor/n,s/n);f['source_sha256'][n]=sha(s/n)
 if n in cfg['files']:cfg['files'][n]=sha(s/n)
composition={'parent':parent,'base_snapshot':str(base),'base_frozen_sha256':sha(base/'frozen-source.json'),'strict_donor_snapshot':str(donor),'strict_donor_frozen_sha256':sha(donor/'frozen-source.json'),'changes':changes,'unmodified_files':len(f['source_sha256'])-len(changes),'strict_policy_default_off':True,'constitutive_equations_unchanged':True,'purpose':'Reconcile verified targeted local-projection policy with published039 and latest preserved Human input; full coupon remains source232 pending and no transfer of trajectory qualification.'}
f['source237_composition']=composition
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n');(s/'configure-support-source.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n');(d/'composition.json').write_text(json.dumps(composition,indent=2)+'\n')
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
cmd=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--cmake-source-relative','.','--jobs','2']
for t in ['numi-matter-cardboard-probe','numi-matter-cardboard-coupon-resume-check','numi-matter-cardboard-bilinear15-capture-diagnostic-check','numi-matter-cardboard-bilinear15-projection-policy-check']:cmd+=['--target',t]
(d/'command.json').write_text(json.dumps({'argv':cmd,'gpu_execution':False},indent=2)+'\n')
with (d/'build.log').open('xb') as out:rc=subprocess.run(cmd,cwd=s,stdout=out,stderr=subprocess.STDOUT,env=env).returncode
tests=[]
if rc==0:
 cmds=[[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools'/n)] for n in ['test_true_residual_certification_contract.py','test_cardboard_solver_environment.py','test_cardboard_stage_resolution_study.py']]
 cmds += [[str(b/'matter/numi-matter-cardboard-bilinear15-projection-policy-check')],[str(b/'matter/numi-matter-cardboard-bilinear15-capture-diagnostic-check'),'--source547-cpu-admission',str(q/'analysis/native547-critical23-sweep-input-002/input.json')],['/usr/bin/clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror',str(s/'matter/tools/cardboard_coupon_resume_check.cpp'),'-o',str(d/'assert-enabled-coupon-resume-check')],[str(d/'assert-enabled-coupon-resume-check')]]
 for i,a in enumerate(cmds):
  with (d/f'test-{i}.stdout').open('xb') as out,(d/f'test-{i}.stderr').open('xb') as err:code=subprocess.run(a,cwd=s,stdout=out,stderr=err,env=env).returncode
  tests.append({'argv':a,'exit_code':code,'stdout_sha256':sha(d/f'test-{i}.stdout'),'stderr_sha256':sha(d/f'test-{i}.stderr')})
  if code:break
report={'schema':'numi.cardboard.source237-strict-main.v1','parent':parent,'source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'build_exit_code':rc,'tests':tests,'all_pass':rc==0 and len(tests)==7 and all(t['exit_code']==0 for t in tests),'all_inputs_unchanged':all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(s/n)==h for n,h in cfg['files'].items()),'gpu_execution':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);sys.exit(int(not report['all_pass']))
