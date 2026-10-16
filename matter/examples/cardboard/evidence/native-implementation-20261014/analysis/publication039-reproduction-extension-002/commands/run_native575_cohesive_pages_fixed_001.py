from pathlib import Path
import json,hashlib,subprocess,os,time,datetime,signal,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');d=q/'analysis/source-production-cohesive-pages-227-002';s=q/'snapshots/source-production-cohesive-pages-227-002';b=q/'build-source-production-cohesive-pages-227-002';c=q/'native/production-cohesive-page-kernels-fixed-575';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(d/'manifest.json')=='8edd89487c3f31698afd6791ef8f3bb96144fa16289c7e87d98c616178499569' and sha(d/'source-build.json')=='49ab08292ba1238a47ba5862d4faa7d9684ff80dbf3d6f2bba0dc19232bba510' and sha(s/'frozen-source.json')=='1c4fbc29aa5db5ef413caffda0a666f55d2c0fdcec5b803d7546c9bcfd8037a1'
j=json.loads((d/'source-build.json').read_text());f=json.loads((s/'frozen-source.json').read_text())
def verify():return all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(b/n)==v['sha256'] for n,v in j['artifacts'].items())
assert verify()
workers=[l for l in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(x in l for x in ['numi-matter-','generic-batch','geometry-replay']) and not l.split()[1].startswith('Z')];assert not workers,workers;c.mkdir(exist_ok=False)
a=[str(b/'numi-matter-cardboard-cohesive-page-runtime-kernel-check'),str(b/'shaders/NumiMatter.metallib')];env={'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','HOME':'/Users/n'}
(c/'command.json').write_text(json.dumps({'schema':'numi.cardboard.native575-command.v1','argv':a,'environment':env,'source_frozen_sha256':sha(s/'frozen-source.json'),'build_record_sha256':sha(d/'source-build.json'),'artifacts':j['artifacts'],'physical_transactions':0,'prediction':'No pass assumed. Test actual production cohesive page accumulation/apply kernels only, not complete nonlinear residual/force/operator trajectory or shipping admission.'},indent=2)+'\n')
t=time.monotonic();timed=False
with (c/'stdout.log').open('xb') as out,(c/'stderr.log').open('xb') as err:
 p=subprocess.Popen(a,stdout=out,stderr=err,env=env,start_new_session=True)
 try:code=p.wait(timeout=120)
 except subprocess.TimeoutExpired:
  timed=True;os.killpg(p.pid,signal.SIGTERM)
  try:code=p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=10)
r={'schema':'numi.cardboard.native575-receipt.v1','exit_code':code,'timed_out':timed,'wall_seconds':time.monotonic()-t,'all_inputs_unchanged':verify(),'stdout_sha256':sha(c/'stdout.log'),'stderr_sha256':sha(c/'stderr.log'),'physical_transactions':0,'shipping_or_full_solver_qualification':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(c/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(code or int(timed or not r['all_inputs_unchanged']))
