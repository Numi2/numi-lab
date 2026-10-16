from pathlib import Path
import json,hashlib,shutil,subprocess,os,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');r=q/'checkout-publication032';base=q/'snapshots/source-capacity-gates-checkpoint-proof-222-001';s=q/'snapshots/source222-upstream-compatibility-223-001';d=q/'analysis/source223-upstream-compatibility-001';b=q/'build-source223-upstream-compatibility-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((base/'frozen-source.json').read_text());s.mkdir(exist_ok=False);d.mkdir(exist_ok=False)
for n,h in f['source_sha256'].items():
 assert sha(r/n)==h,n
 p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r/n,p)
extra=['apps/NumiHumanRestingSourceAllocation.hpp','apps/NumiHumanRestingVisual.hpp','tools/test_resting_referenced_source_attachments.cpp']
for n in extra:
 p=s/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(r/n,p);f['source_sha256'][n]=sha(p)
f['source223_upstream_compatibility']={'base_snapshot':str(base),'base_frozen_sha256':sha(base/'frozen-source.json'),'upstream_commit':'68e19b6509731d6d6700dd508b249fb68d7cd7c2','extra_upstream_paths':extra,'cardboard819_source_bytes_unchanged':True,'source222_code_delta_already_present':True}
(s/'frozen-source.json').write_text(json.dumps(f,indent=2,sort_keys=True)+'\n')
argv=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--jobs','4','--target','numi-matter-cardboard-probe','--target','numi-matter-cardboard-coupon-resume-check']
(d/'build-command.json').write_text(json.dumps({'argv':argv,'frozen_sha256':sha(s/'frozen-source.json')},indent=2)+'\n')
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
with (d/'build.log').open('xb') as out:rc=subprocess.run(argv,stdout=out,stderr=subprocess.STDOUT,env=env,cwd=s).returncode
tests=[
 [str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/test_cardboard_stage_resolution_study.py')],
 ['/usr/bin/clang++','-std=c++20','-O2','-I',str(s/'apps'),str(s/'tools/test_resting_referenced_source_attachments.cpp'),'-o',str(d/'upstream-referenced-attachments-check')],
 [str(d/'upstream-referenced-attachments-check')],
 [str(b/'numi-matter-cardboard-coupon-resume-check')]
]
results=[]
for i,a in enumerate(tests):
 with (d/('test-'+str(i)+'.stdout')).open('xb') as out,(d/('test-'+str(i)+'.stderr')).open('xb') as err:
  rr=subprocess.run(a,stdout=out,stderr=err,env=env,cwd=s)
 results.append({'argv':a,'exit_code':rr.returncode,'stdout_sha256':sha(d/('test-'+str(i)+'.stdout')),'stderr_sha256':sha(d/('test-'+str(i)+'.stderr'))})
 if rr.returncode:break
report={'schema':'numi.cardboard.source223-upstream-compatibility.v1','upstream_commit':'68e19b6509731d6d6700dd508b249fb68d7cd7c2','source222_cardboard_bytes_unchanged':True,'source_snapshot':str(s),'frozen_sha256':sha(s/'frozen-source.json'),'build_exit_code':rc,'source_build_sha256':sha(b/'source-build.json'),'tests':results,'all_pass':rc==0 and len(results)==4 and all(v['exit_code']==0 for v in results),'gpu_execution':False,'physical_transactions':0,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'frozen-source.json').write_bytes((s/'frozen-source.json').read_bytes());(d/'source-build.json').write_bytes((b/'source-build.json').read_bytes());(d/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);sys.exit(int(not report['all_pass']))
