from pathlib import Path
import datetime,hashlib,json,os,subprocess,time,sys
B=Path('/Users/n/cardboard-implementation-20261014-host-macmini');S=B/'snapshots/source-scaled-contact-geometry-188-buildfix002';D=B/'build-source188-002';P=B/'analysis/source188-native-validation-001'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
f=json.loads((S/'frozen-source.json').read_text());sp=json.loads((S/'configure-support-source.json').read_text());b=json.loads((D/'source-build.json').read_text())
for n,h in f['source_sha256'].items():assert sha(S/n)==h,n
for n,h in sp['files'].items():assert sha(S/n)==h,n
assert b['success'] and b['source_sha256']==f['source_sha256']
for n,h in b['artifacts_sha256'].items():assert sha(Path(n))==h,n
assert b['started_at_utc']>json.loads((B/'analysis/source188-contact-geometry-buildfix-002/complete-transfer-verification.json').read_text())['verified_at_utc']
P.mkdir();(P/'source-build-binding.json').write_text(json.dumps({'source_build_sha256':sha(D/'source-build.json'),'frozen_sha256':sha(S/'frozen-source.json'),'artifacts':b['artifacts_sha256'],'source_files':814,'support_files':1034},indent=2))
env={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_CARDBOARD_','NUMI_MATTER_'))};env['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin';env['PYTHONDONTWRITEBYTECODE']='1'
def run(name,cmd,folder,timeout,gpu=False):
 if gpu:
  workers=[x for x in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(n in x for n in ('numi-matter-','geometry-replay','contact_barrier_trial_replay')) and not x.split()[1].startswith('Z')];assert not workers,workers
 folder.mkdir();r={'name':name,'argv':cmd,'source_build_sha256':sha(D/'source-build.json'),'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'timeout_s':timeout,'gpu':gpu};(folder/'prediction.json').write_text(json.dumps(r,indent=2))
 start=time.monotonic()
 with (folder/'stdout.log').open('xb') as out,(folder/'stderr.log').open('xb') as err:
  try:p=subprocess.run(cmd,cwd=folder,env=env,stdout=out,stderr=err,timeout=timeout);code=p.returncode;timed=False
  except subprocess.TimeoutExpired:code=None;timed=True
 r.update(exit_code=code,timed_out=timed,wall_seconds=time.monotonic()-start,finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stdout_sha256=sha(folder/'stdout.log'),stderr_sha256=sha(folder/'stderr.log'),physical_qualification=False)
 (folder/'receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
 return r
r=run('cpu-validation',['/opt/homebrew/bin/ctest','--test-dir',str(D),'-R',r'^matter.compiler.cardboard_(shipping_memory|full_contact_scratch_budget|coupon_resume)$|^matter.cli.cardboard_argument_admission$|^matter.analysis.cardboard_compensated_fp64_reference$','--output-on-failure','-j','1'],P/'cpu-checks',180)
assert r['exit_code']==0
r=run('native504-geometry',[str(D/'numi-matter-cardboard-contact-geometry-check'),'--output',str(B/'native/contact-scaled-geometry-504/output.bin')],B/'native/contact-scaled-geometry-504',60,True)
assert r['exit_code']==0
r=run('native505-contact-fixture',[str(D/'numi-matter-cardboard-full-contact-armijo-check'),str(S/'matter/materials/nagasawa2013_medium_orthotropic.nmatter'),'--compensated-material-arithmetic'],B/'native/contact-scaled-geometry-fixture-505',300,True)
for n,h in f['source_sha256'].items():assert sha(S/n)==h,n
for n,h in b['artifacts_sha256'].items():assert sha(Path(n))==h,n
(P/'finished.json').write_text(json.dumps({'finished_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fixture_exit_code':r['exit_code'],'source_and_artifacts_unchanged':True,'all_six_qualification_axes_complete':False},indent=2))
