from pathlib import Path
import json,hashlib,subprocess,os,time,datetime,signal,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/source-production-cohesive-pages-227-001';s=q/'snapshots/source-production-cohesive-pages-227-001';b=q/'build-source-production-cohesive-pages-227-001';c=q/'native/production-cohesive-page-kernels-574';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(d/'manifest.json')=='39a9680c89b589e72fa0a151d3834a16cd662fff3f01afe8bf4dc72fbf5e118e' and sha(d/'source-build.json')=='9a5552c9d60ac26ca11f2a40d3f9b3e4ba97c83198460b45e9fa174a1537e216' and sha(s/'frozen-source.json')=='f880325285a44d854253eb5b71be40f57f08800576734324b6233efbfde94260'
j=json.loads((d/'source-build.json').read_text());f=json.loads((s/'frozen-source.json').read_text())
def verify():return all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(b/n)==v['sha256'] for n,v in j['artifacts'].items())
assert verify()
workers=[l for l in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(x in l for x in ['numi-matter-','generic-batch','geometry-replay']) and not l.split()[1].startswith('Z')];assert not workers,workers;c.mkdir(exist_ok=False)
a=[str(b/'numi-matter-cardboard-cohesive-page-runtime-kernel-check'),str(b/'shaders/NumiMatter.metallib')];env={'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','HOME':'/Users/n'}
(c/'command.json').write_text(json.dumps({'schema':'numi.cardboard.native574-command.v1','argv':a,'environment':env,'source_frozen_sha256':sha(s/'frozen-source.json'),'build_record_sha256':sha(d/'source-build.json'),'artifacts':j['artifacts'],'physical_transactions':0,'prediction':'No pass assumed. Test actual production cohesive page accumulation/apply kernels only, not complete nonlinear residual/force/operator trajectory or shipping admission.'},indent=2)+'\n')
t=time.monotonic();timed=False
with (c/'stdout.log').open('xb') as out,(c/'stderr.log').open('xb') as err:
 p=subprocess.Popen(a,stdout=out,stderr=err,env=env,start_new_session=True)
 try:code=p.wait(timeout=120)
 except subprocess.TimeoutExpired:
  timed=True;os.killpg(p.pid,signal.SIGTERM)
  try:code=p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=10)
r={'schema':'numi.cardboard.native574-receipt.v1','exit_code':code,'timed_out':timed,'wall_seconds':time.monotonic()-t,'all_inputs_unchanged':verify(),'stdout_sha256':sha(c/'stdout.log'),'stderr_sha256':sha(c/'stderr.log'),'physical_transactions':0,'shipping_or_full_solver_qualification':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(c/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(code or int(timed or not r['all_inputs_unchanged']))
