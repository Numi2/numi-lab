from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
R=Path.cwd();B=R/'build/crow-ccd-articulated-sweep-build-01';A=R/'artifacts/crow-rl-ccd-128-integration-20261108'
T=R/'.numi/retained-runtime/ccd-event-capture-20261108/integrated128'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
report=json.loads((A/'results.json').read_text())
prior=json.loads((B/'integration-execution.json').read_text())[-1]
assert prior['name']=='integrated-object-push' and prior['exit_code']==1
error=[l for l in (B/'integrated-object-push.stderr').read_text().splitlines() if not l.startswith('dyld[')]
assert error==['Checkpoint identity file unavailable'],error
assert not (T/'metalrobo/RetinalStream.metallib').exists()
report.update(native_object_push_regression_passed=False,initial_object_push_failure={'receipt':prior,'error':error,'cause':'RetinalStream defaultLibrary searches the loaded dylib directory/metalrobo then its parent/shaders. The retained runtime omitted RetinalStream.metallib. Failure precedes native physical stepping; both authored fixture directories exist but no arm trace exists.'})
assert not any((B/'integrated-object-push/positive-chunk1').iterdir())
write(A/'results.json',report)
plan={'purpose':'Repair incomplete retained runtime packaging, then execute unchanged object-push engineering controls once. Preserve the original failure and all physics/convergence thresholds.','preflight_revision':'v1 incorrectly asserted no arm directory; execute creates it before constructing retina. It is empty. v1 stopped before plan sealing, file copy or native invocation; source retained under raw root. v2 asserts the actual empty arm directory.','source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'added_file':'metalrobo/RetinalStream.metallib','source':'build/shaders/RetinalStream.metallib','sha256':sha(R/'build/shaders/RetinalStream.metallib'),'evidence_for_cause':['src/metal/RetinalStream.mm defaultLibrary and constructor fileDigest before loading/stepping','positive-chunk1 is empty; no arm trace exists'],'prediction':'Missing retinal file is resolved; all unchanged owner controls pass. If they fail, preserve first failure and do not retry.','budget':{'native_executions':1,'timeout_seconds':180},'scope':'Packaging repair and native engineering regression, not full crow environmental or biological qualification.'}
write(A/'packaging-repair-plan.json',plan)
subprocess.run(['git','add',str(A/'packaging-repair-plan.json'),str(A/'results.json'),str(A/'package_owner.py')],check=True)
subprocess.run(['git','commit','-m','Seal retained retinal runtime packaging repair'],check=True)
subprocess.run(['git','push','origin','HEAD'],check=True)
shutil.copy2(R/'build/shaders/RetinalStream.metallib',T/'metalrobo/RetinalStream.metallib')
assert sha(T/'metalrobo/RetinalStream.metallib')==plan['sha256']
env=os.environ.copy();env.pop('METALROBO_METALLIB',None);env.update(DYLD_LIBRARY_PATH=str(T),DYLD_PRINT_LIBRARIES='1')
cmd=[str(T/'object-push-check'),str(B/'packaged-object-push')]
start=time.monotonic()
with (B/'packaged-object-push.stdout').open('xb') as out,(B/'packaged-object-push.stderr').open('xb') as err:
    code=subprocess.run(cmd,env=env,stdout=out,stderr=err,timeout=180).returncode
receipt={'command':cmd,'exit_code':code,'seconds':time.monotonic()-start,'runtime':{str(p.relative_to(R)):sha(p) for p in T.rglob('*') if p.is_file()}}
write(B/'packaging-repair-execution.json',receipt)
report.update(native_object_push_regression_passed=code==0,packaging_repair=receipt)
if code==0:report['object_push_summary']=json.loads((B/'packaged-object-push/summary.json').read_text())
else:report['packaged_object_push_failure']=(B/'packaged-object-push/failure.txt').read_text() if (B/'packaged-object-push/failure.txt').exists() else 'Inspect typed stderr.'
write(A/'results.json',report)
print(json.dumps({k:v for k,v in report.items() if k in ('native_object_push_regression_passed','object_push_summary','packaged_object_push_failure')},indent=2),flush=True)
