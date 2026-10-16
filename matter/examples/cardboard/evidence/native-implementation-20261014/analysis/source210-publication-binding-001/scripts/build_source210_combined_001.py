from pathlib import Path
import subprocess,json,os,hashlib,datetime,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-composed-strict-periodic-freebox-210-buildfix-003';d=q/'analysis/source210-combined-build-001';b=q/'build-source210-combined-001';d.mkdir(exist_ok=False)
env={**os.environ,'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','PYTHONDONTWRITEBYTECODE':'1'}
cmd=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_build.py'),'--source-snapshot',str(s),'--build-dir',str(b),'--jobs','4']
for t in ['numi-matter-cardboard-probe','numi-matter-cardboard-coupled-free-box-operator-check','numi-matter-fgmres-true-residual-native-check','numi-matter-cardboard-coupon-resume-check','numi-matter-cardboard-shipping-snapshot-check']:cmd+=['--target',t]
commands=[('build',cmd,str(s)),('cpu-contracts',[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B','-m','unittest','test_cardboard_cached_operator_audit','test_cardboard_solver_environment','test_true_residual_certification_contract'],str(s/'matter/tools')),('cpu-checkpoint-tests',['/opt/homebrew/bin/ctest','--test-dir',str(b),'-R','cardboard-(coupon-resume|shipping-snapshot)-cpu','--output-on-failure'],str(s))]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
reports=[]
for name,cmd,cwd in commands:
 (d/(name+'-command.json')).write_text(json.dumps({'argv':cmd,'cwd':cwd,'native_gpu':False},indent=2)+'\n')
 with (d/(name+'.log')).open('xb') as out:r=subprocess.run(cmd,cwd=cwd,stdout=out,stderr=subprocess.STDOUT,env=env)
 row={'stage':name,'exit_code':r.returncode,'log_sha256':sha(d/(name+'.log'))};reports.append(row);print(json.dumps(row),flush=True)
 if r.returncode:break
bm=b/'source-build.json'
receipt={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(bm) if bm.exists() else None,'commands':reports,'native_gpu_executed':False}
(d/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True);sys.exit(int(any(r['exit_code'] for r in reports)))
