from pathlib import Path
import json,hashlib,subprocess,os,time,datetime,sys
q=Path("/Users/n/cardboard-implementation-20261014-host-macmini");s=q/'snapshots/source-free-box-capsule-true-residual-212-012';b=q/'build-source212-freebox-proxy-true-residual-012';c=q/'native/true-residual-full-contact-546'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
f=json.loads((s/'frozen-source.json').read_text());bm=json.loads((b/'source-build.json').read_text())
assert sha(s/'frozen-source.json')=='499468e952b33f98419490c9c9987f742f79791d8822285919189205eae85555'
assert bm['success'] and bm['source_sha256']==f['source_sha256']
def verify():return all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(n)==h for n,h in bm['artifacts_sha256'].items())
assert verify()
ps=subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True)
workers=[l for l in ps.splitlines() if any(x in l for x in ['numi-matter-','generic-batch','geometry-replay']) and not l.split()[1].startswith('Z')]
assert not workers,workers
assert not c.exists();c.mkdir()
material=s/'matter/materials/nagasawa2013_medium_orthotropic.nmatter'
specs=[('operator',[str(b/'numi-matter-cardboard-coupled-free-box-operator-check')],60),
('contact',[str(b/'numi-matter-cardboard-full-contact-armijo-check'),str(material),'--sparse-free-body-parity','--diagnostic-state-output',str(c/'contact'/'accepted-state-output')],900)]
reports=[]
env={'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','HOME':'/Users/n','NUMI_MATTER_METALLIB':str(b/'shaders/NumiMatter.metallib'),'NM_FEM_TRUE_RESIDUAL_CERTIFY':'1','NM_FEM_TRUE_RESIDUAL_ROOT':'all'}
for name,argv,timeout in specs:
 d=c/name;d.mkdir()
 command={'schema':'numi.cardboard.native546-command.v1','argv':argv,'environment':env,'timeout_s':timeout,'source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_json':str(b/'source-build.json'),'source_build_sha256':sha(b/'source-build.json'),'material_sha256':sha(material),'all_source_and_artifacts_verified':verify(),'question':'Can the bounded free-BOX/prescribed-capsule full-contact trial and final certificate coexist with same-operator true-residual certification?','prediction':'No trajectory or gate pass is assumed. Keep dense/sparse parity, actual tested alpha, rejected-state rollback and true residual failures visible.','qualification_limit':'Native small fixture; no coupon resolution, source-grade calibration, shipping assembly or crush/impact qualification.'}
 (d/'command.json').write_text(json.dumps(command,indent=2)+'\n')
 start=time.monotonic();timed=False;code=None;err=None
 try:
  with (d/'stdout.log').open('xb') as out,(d/'stderr.log').open('xb') as er:
   p=subprocess.Popen(argv,stdout=out,stderr=er,env=env,start_new_session=True)
   print(json.dumps({'case':name,'pid':p.pid,'started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True)
   try:code=p.wait(timeout=timeout)
   except subprocess.TimeoutExpired:
    import signal
    timed=True;os.killpg(p.pid,signal.SIGTERM)
    try:code=p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=10)
 except BaseException as ex:err=repr(ex)
 r={'schema':'numi.cardboard.native546-receipt.v1','case':name,'exit_code':code,'timed_out':timed,'error':err,'wall_seconds':time.monotonic()-start,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_and_artifacts_unchanged':verify(),'stdout_sha256':sha(d/'stdout.log') if (d/'stdout.log').exists() else None,'stderr_sha256':sha(d/'stderr.log') if (d/'stderr.log').exists() else None,'physical_transactions':'derive from complete retained native stdout; never assume on exit alone','full_solver_qualified':False}
 (d/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');reports.append(r);print(json.dumps(r),flush=True)
 if timed or err or code!=0 or not r['source_and_artifacts_unchanged']:break
(c/'summary.json').write_text(json.dumps(reports,indent=2)+'\n')
sys.exit(int(len(reports)!=len(specs) or any(r['exit_code']!=0 or r['timed_out'] or r['error'] or not r['source_and_artifacts_unchanged'] for r in reports)))
