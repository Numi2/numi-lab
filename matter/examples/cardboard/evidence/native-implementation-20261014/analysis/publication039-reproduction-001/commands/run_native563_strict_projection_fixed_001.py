from pathlib import Path
import json,hashlib,subprocess,os,time,datetime,signal,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');c=q/'native/strict-local-projection-critical23-fixed-563';d=q/'analysis/source-local-projection-state-certification-220-003'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(d/'manifest.json')=='9433667e5bc44cda47e5dd07140525c14c2e27a431896fea6e8d596423ee9e51'
manifest=json.loads((d/'manifest.json').read_text());assert all(sha(d/n)==h for n,h in manifest['files'].items())
r=json.loads((d/'report.json').read_text());s=Path(r['source']['path']);b=Path(r['build']['path']);f=json.loads((s/'frozen-source.json').read_text())
assert sha(s/'frozen-source.json')=='ce4db34124a09c8d1356af11c22a42cad9779629cf1800965ed6e5878d5f4639'
art={a['path']:a['sha256'] for a in r['build']['artifacts'].values()};inp=q/'analysis/native547-critical23-sweep-input-002/input.json'
def verify():return all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(n)==h for n,h in art.items()) and sha(inp)=='a71da761e88cbaca694e7c0cd6333511f46a0a36a12b76b5081b0d26309bb74e'
assert verify()
workers=[l for l in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(x in l for x in ['numi-matter-','generic-batch','geometry-replay']) and not l.split()[1].startswith('Z')];assert not workers,workers
c.mkdir(exist_ok=False)
a=[str(b/'numi-matter-cardboard-bilinear15-capture-diagnostic-check'),'--native-source547-strict-projection',str(inp),sha(inp),str(s/'matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter'),str(s/'matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter'),str(c/'output')]
assert all(Path(n).exists() for n in [a[0],a[2],a[4],a[5]])
env={'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','HOME':'/Users/n','NUMI_MATTER_METALLIB':str(b/'shaders/NumiMatter.metallib')}
(c/'command.json').write_text(json.dumps({'schema':'numi.cardboard.native563-command.v1','argv':a,'environment':env,'source_snapshot':str(s),'source_frozen_sha256':sha(s/'frozen-source.json'),'build_artifacts':art,'package_manifest_sha256':sha(d/'manifest.json'),'input_sha256':sha(inp),'physical_transactions':0,'question':'Retest strict local projected-state certification after repairing only the endpoint identity checker that rejected native558.','prediction':'Native rows require independent FP64 adjudication; no global coupon qualification. Native558 checker failure retained.'},indent=2)+'\n')
t=time.monotonic();timed=False
with (c/'stdout.log').open('xb') as out,(c/'stderr.log').open('xb') as err:
 p=subprocess.Popen(a,stdout=out,stderr=err,env=env,start_new_session=True)
 try:code=p.wait(timeout=120)
 except subprocess.TimeoutExpired:
  timed=True;os.killpg(p.pid,signal.SIGTERM)
  try:code=p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=10)
rr={'schema':'numi.cardboard.native563-receipt.v1','exit_code':code,'timed_out':timed,'wall_seconds':time.monotonic()-t,'all_inputs_unchanged':verify(),'stdout_sha256':sha(c/'stdout.log'),'stderr_sha256':sha(c/'stderr.log'),'physical_transactions':0,'full_solver_qualification':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(c/'receipt.json').write_text(json.dumps(rr,indent=2)+'\n');print(json.dumps(rr),flush=True);sys.exit(code or int(timed or not rr['all_inputs_unchanged']))
