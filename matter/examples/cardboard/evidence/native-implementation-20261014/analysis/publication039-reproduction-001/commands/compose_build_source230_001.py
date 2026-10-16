from pathlib import Path
import json,hashlib,shutil,datetime,subprocess,os,sys,difflib
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
base=q/'snapshots/source-composed-pure-proxy-228-004';ancestor=q/'snapshots/source-qualification-checkpoint-proof-221-003';donor=q/'snapshots/source-diagnostic-epsilon-replay-229-001'
s=q/'snapshots/source-composed-pure-proxy-replay-230-001';d=q/'analysis/source230-composition-build-001';b=q/'build-source230-integrated-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
expected=['7e749050f8092408bb7d341be1dbf99a1d76324a96121c1e9ecf79b4c3fcca08','fb885540a2f096ee70df5a03701a8d2710832c6df402e8e097ff95c736e7c1a5','6cf5be6795c1d77c8a3aa5a87665bd98d53ba21363abe893a4604e44273e2cd6']
maps=[]
for root,h in zip([base,ancestor,donor],expected):
 assert sha(root/'frozen-source.json')==h
 m=json.loads((root/'frozen-source.json').read_text());assert all(sha(root/n)==v for n,v in m['source_sha256'].items());maps.append(m)
f,af,df=maps;s.mkdir(exist_ok=False);d.mkdir(exist_ok=False)
for n in f['source_sha256']:
 p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
delta=[n for n,h in df['source_sha256'].items() if af['source_sha256'].get(n)!=h];assert len(delta)==3;records=[]
for n in delta:
 assert f['source_sha256'][n]==af['source_sha256'][n],n
 shutil.copy2(donor/n,s/n);records.append({'file':n,'mode':'direct_disjoint','parent_sha256':f['source_sha256'][n],'donor_sha256':sha(donor/n)})
cfg=json.loads((base/'configure-support-source.json').read_text())
for n,h in cfg['files'].items():
 assert sha(base/n)==h;p=s/n
 if p.exists():assert sha(p)==h
 else:p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(base/n,p)
shutil.copy2(base/'configure-support-source.json',s/'configure-support-source.json')
f['source_sha256']={n:sha(s/n) for n in sorted(f['source_sha256'])}
f['source230_composition']={'parent_snapshot':str(base),'parent_frozen_sha256':expected[0],'ancestor_snapshot':str(ancestor),'ancestor_frozen_sha256':expected[1],'donor_snapshot':str(donor),'donor_frozen_sha256':expected[2],'files':records,'scope':'Compose proof-only bounded epsilon observer replay admission with checked pure8192 arenas and explicit4096 freeBOX/prescribed tool scope. No material equations, physical operator or nonlinear acceptance changes; no shipping qualification.'}
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n')
(d/'source.patch').write_text(''.join(''.join(difflib.unified_diff((base/n).read_text().splitlines(True),(s/n).read_text().splitlines(True),fromfile='a/'+n,tofile='b/'+n)) for n in delta))
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
argv=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--cmake-source-relative','.','--jobs','2']
targets=['numi-matter-cardboard-probe','numi-matter-cardboard-coupon-resume-check','numi-matter-cardboard-coupled-free-box-operator-check','numi-matter-cardboard-free-box-proxy-admission-check','numi-matter-cardboard-full-contact-armijo-check','numi-matter-fgmres-true-residual-native-check','numi-matter-fem-cgls-arena-8192-check']
for t in targets:argv+=['--target',t]
(d/'command.json').write_text(json.dumps({'argv':argv,'source_frozen_sha256':sha(s/'frozen-source.json'),'gpu_execution':False},indent=2)+'\n')
with (d/'build.log').open('xb') as out:rc=subprocess.run(argv,stdout=out,stderr=subprocess.STDOUT,env=env,cwd=s).returncode
tests=[]
if rc==0:
 cmds=[[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools'/n)] for n in ['test_true_residual_certification_contract.py','test_cardboard_solver_environment.py','test_cardboard_stage_resolution_study.py']]
 cmds += [[str(b/'matter/numi-matter-cardboard-free-box-proxy-admission-check')],['/usr/bin/clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror',str(s/'matter/tools/cardboard_coupon_resume_check.cpp'),'-o',str(d/'assert-enabled-coupon-resume-check')],[str(d/'assert-enabled-coupon-resume-check')],[str(b/'matter/numi-matter-cardboard-probe'),'--verify-coupon-source-runner',str(q/'solver/fresh-source210-strict-inner-full-coupon-545')]]
 for i,a in enumerate(cmds):
  with (d/f'test-{i}.stdout').open('xb') as out,(d/f'test-{i}.stderr').open('xb') as err:r=subprocess.run(a,stdout=out,stderr=err,env=env,cwd=s).returncode
  tests.append({'argv':a,'exit_code':r,'stdout_sha256':sha(d/f'test-{i}.stdout'),'stderr_sha256':sha(d/f'test-{i}.stderr')})
  if r:break
r={'schema':'numi.cardboard.source230-composition-build.v1','build_exit_code':rc,'tests':tests,'all_pass':rc==0 and len(tests)==7 and all(t['exit_code']==0 for t in tests),'source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json') if (b/'source-build.json').exists() else None,'patch_sha256':sha(d/'source.patch'),'all_inputs_unchanged':all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(s/n)==h for n,h in cfg['files'].items()),'gpu_execution':False,'physical_transactions':0,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(int(not r['all_pass']))
