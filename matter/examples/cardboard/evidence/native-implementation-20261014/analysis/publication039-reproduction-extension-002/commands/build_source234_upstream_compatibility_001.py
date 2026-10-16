from pathlib import Path
import json,hashlib,shutil,subprocess,os,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';base=q/'snapshots/source230-upstream-compatibility-231-001';s=q/'snapshots/source231-upstream-compatibility-234-001';d=q/'analysis/source234-upstream-compatibility-001';b=q/'build-source234-upstream-compatibility-001';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
parent=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip();assert parent=='938e30343dd1cc313e222c8d2e8c11f5d486754d'
assert not subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=r)
assert sha(base/'frozen-source.json')=='dc6e332ceda2033e258147bc4b4df174f207cf1511c8ea0221c7e945839b57b8';f=json.loads((base/'frozen-source.json').read_text());cfg=json.loads((base/'configure-support-source.json').read_text());s.mkdir(exist_ok=False);d.mkdir(exist_ok=False)
for n,h in {**cfg['files'],**f['source_sha256']}.items():
 assert sha(base/n)==h,n;p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
paths=['apps/NumiHumanRestingIntercostalRibAttachment.hpp','tools/test_intercostal_rib_motion_attachment.mm'];upstream=[]
for n in paths:
 raw=subprocess.check_output(['git','show',parent+':'+n],cwd=r);assert (r/n).read_bytes()==raw;p=s/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);f['source_sha256'][n]=sha(p)
 if n in cfg['files']:cfg['files'][n]=sha(p)
 upstream.append({'path':n,'sha256':sha(p)})
f['source_sha256']=dict(sorted(f['source_sha256'].items()));f['source234_upstream_compatibility']={'parent_snapshot':str(base),'parent_frozen_sha256':sha(base/'frozen-source.json'),'upstream_commit':parent,'upstream_only_files':upstream,'all_source231_cardboard_bytes_unchanged':all(sha(s/n)==h for n,h in json.loads((base/'frozen-source.json').read_text())['source_sha256'].items()),'scope':'Reconcile two disjoint Human files already on main; keep all source231 cardboard inputs exact. Historical source231 measurements stay bound to231.'}
assert f['source234_upstream_compatibility']['all_source231_cardboard_bytes_unchanged']
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n');cfg['source234_updates']={'upstream_commit':parent,'files':upstream};(s/'configure-support-source.json').write_text(json.dumps(cfg,indent=2,sort_keys=True)+'\n')
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
a=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--cmake-source-relative','.','--jobs','2']
for t in ['numi-matter-cardboard-probe','numi-matter-cardboard-coupon-resume-check','numi-matter-cardboard-coupled-free-box-operator-check','numi-matter-cardboard-free-box-proxy-admission-check','numi-matter-cardboard-full-contact-armijo-check','numi-matter-fgmres-true-residual-native-check','numi-matter-fem-cgls-arena-8192-check']:a+=['--target',t]
(d/'command.json').write_text(json.dumps({'argv':a,'upstream':upstream,'gpu_execution':False},indent=2)+'\n')
with (d/'build.log').open('xb') as out:rc=subprocess.run(a,cwd=s,stdout=out,stderr=subprocess.STDOUT,env=env).returncode
tests=[]
if rc==0:
 cmds=[[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools'/n)] for n in ['test_true_residual_certification_contract.py','test_cardboard_solver_environment.py','test_cardboard_stage_resolution_study.py']]
 cmds += [['/usr/bin/clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror',str(s/'matter/tools/cardboard_coupon_resume_check.cpp'),'-o',str(d/'assert-enabled-coupon-resume-check')],[str(d/'assert-enabled-coupon-resume-check')],['/usr/bin/clang++','-std=c++20','-O2','-I',str(s/'apps'),str(s/'tools/test_resting_referenced_source_attachments.cpp'),'-o',str(d/'assert-enabled-upstream-attachments-check')],[str(d/'assert-enabled-upstream-attachments-check')]]
 cmds += [['/usr/bin/clang++','-x','objective-c++','-std=c++20','-O2','-fobjc-arc','-framework','Foundation','-I',str(s/'apps'),str(s/'tools/test_intercostal_rib_motion_attachment.mm'),'-o',str(d/'upstream-intercostal-attachments-check')],[str(d/'upstream-intercostal-attachments-check')]]
 for i,a in enumerate(cmds):
  with (d/f'test-{i}.stdout').open('xb') as out,(d/f'test-{i}.stderr').open('xb') as err:code=subprocess.run(a,cwd=s,stdout=out,stderr=err,env=env).returncode
  tests.append({'argv':a,'exit_code':code,'stdout_sha256':sha(d/f'test-{i}.stdout'),'stderr_sha256':sha(d/f'test-{i}.stderr')})
  if code:break
report={'schema':'numi.cardboard.source234-upstream-compatibility.v1','parent':parent,'source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'build_exit_code':rc,'tests':tests,'all_pass':rc==0 and len(tests)==9 and all(t['exit_code']==0 for t in tests),'all_source231_cardboard_bytes_unchanged':True,'all_inputs_unchanged':all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(s/n)==h for n,h in cfg['files'].items()),'gpu_execution':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);sys.exit(int(not report['all_pass']))
