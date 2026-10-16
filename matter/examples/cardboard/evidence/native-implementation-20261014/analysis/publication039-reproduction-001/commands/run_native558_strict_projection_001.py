from pathlib import Path
import json,hashlib,subprocess,os,time,datetime,signal,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');c=q/'native/strict-local-projection-critical23-558'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=q/'analysis/source-local-projection-state-certification-220-001';j=json.loads((p/'report.json').read_text());v=json.loads((p/'build-validation.json').read_text())
assert sha(p/'manifest.json')=='5a658d3288ec7c44e972f9a3746b6903b6bb61c9c259977ee6c4d94aac1d9a06'
s=Path(v['source_snapshot']);f=json.loads((s/'frozen-source.json').read_text());assert sha(s/'frozen-source.json')==j['snapshot_frozen_source_json_sha256']
art={r['path']:r['sha256'] for r in v['artifacts']};inp=q/'analysis/native547-critical23-sweep-input-002/input.json'
def verify():return all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(n)==h for n,h in art.items()) and sha(inp)=='a71da761e88cbaca694e7c0cd6333511f46a0a36a12b76b5081b0d26309bb74e'
assert verify()
workers=[l for l in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(x in l for x in ['numi-matter-','generic-batch','geometry-replay']) and not l.split()[1].startswith('Z')]
assert not workers,workers
c.mkdir(exist_ok=False);a=j['native_command'][:-1]+[str(c/'output')]
env={'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','HOME':'/Users/n','NUMI_MATTER_METALLIB':str(Path(v['build_directory'])/'shaders/NumiMatter.metallib')}
cmd={'schema':'numi.cardboard.native558-command.v1','argv':a,'environment':env,'source_frozen_sha256':sha(s/'frozen-source.json'),'build_validation_sha256':sha(p/'build-validation.json'),'artifact_bindings':art,'input_sha256':sha(inp),'all_inputs_verified':True,'physical_transactions':0,'question':'Does the tighter local projection state certificate preserve validity and improve the captured critical material tangent endpoints?','prediction':'No result assumed; native rows require independent FP64 adjudication; no global coupon qualification.'}
(c/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
t=time.monotonic();timed=False;code=None
with (c/'stdout.log').open('xb') as out,(c/'stderr.log').open('xb') as err:
 child=subprocess.Popen(a,stdout=out,stderr=err,env=env,start_new_session=True)
 try:code=child.wait(timeout=120)
 except subprocess.TimeoutExpired:
  timed=True;os.killpg(child.pid,signal.SIGTERM)
  try:code=child.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
r={'schema':'numi.cardboard.native558-receipt.v1','exit_code':code,'timed_out':timed,'wall_seconds':time.monotonic()-t,'all_inputs_unchanged':verify(),'stdout_sha256':sha(c/'stdout.log'),'stderr_sha256':sha(c/'stderr.log'),'physical_transactions':0,'full_solver_qualification':False,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(c/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(code or int(timed or not r['all_inputs_unchanged']))
