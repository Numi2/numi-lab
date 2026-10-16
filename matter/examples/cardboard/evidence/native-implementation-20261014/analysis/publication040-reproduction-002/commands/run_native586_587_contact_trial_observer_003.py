from pathlib import Path
import json,hashlib,subprocess,os,time,datetime,signal
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');o=q/'analysis/deformable-contact-trial-alpha-trace-236-002/source236-overlay.json';overlay=json.loads(o.read_text());s=Path(overlay['candidate_snapshot']);b=o.parent/'build-source236-001';d=q/'analysis/native586-587-contact-trial-observer-command-002';d.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();assert sha(o)=='da72754f6363cd6144a45a59b1afa6ef582d4b02aa8dbc092c71a708fa39cffe';assert sha(b/'source-build.json')=='38cff4debf3ee25a66bdf534a2b66e25dfb47dfaf91edfe29dfd405ef1f0c380'
bm=json.loads((b/'source-build.json').read_text());assert bm['success'] and len(bm['source_sha256'])==823
def verify():return all(sha(s/n)==h for n,h in bm['source_sha256'].items()) and all(sha(n)==h for n,h in bm['artifacts_sha256'].items())
assert verify();results=[]
for number,on in [(586,False),(587,True)]:
 workers=[l for l in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if 'numi-matter-' in l and not l.split()[1].startswith('Z')]
 assert not workers,workers
 c=q/'native'/('contact-trial-observer-'+('on' if on else 'off')+'-'+str(number));c.mkdir()
 a=[str(b/'matter/numi-matter-cardboard-full-contact-armijo-check'),str(s/'matter/materials/nagasawa2013_medium_orthotropic.nmatter'),'--sparse-free-body-parity','--newton-iterations','64','--diagnostic-state-output',str(c/'accepted-state-output'),'--first-failure-dispatch-root','1199']
 env={'PATH':'/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin','HOME':'/Users/n','NUMI_MATTER_METALLIB':str(b/'matter/shaders/NumiMatter.metallib')}
 if on:env.update({'NM_FEM_DEFORMABLE_LIMIT_TRACE_ROOT':'1199','NM_FEM_DEFORMABLE_LIMIT_TRACE_SLOT':'32'})
 (c/'command.json').write_text(json.dumps({'schema':'numi.cardboard.contact-trial-observer-command.v1','argv':a,'environment':env,'overlay_sha256':sha(o),'source_build_sha256':sha(b/'source-build.json'),'candidate_source_bindings_sha256':hashlib.sha256(json.dumps(bm['source_sha256'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),'question':'Does per-alpha observation retain exact production histories and expose the original contact rejection?','prediction':'Failure retained; no pass assumed and no thresholds changed. Diagnostic-off/diagnostic-on histories must be byte-identical.','timeout_s':180},indent=2)+'\n')
 started=time.monotonic();timed=False
 with (c/'stdout.log').open('xb') as out,(c/'stderr.log').open('xb') as err:
  p=subprocess.Popen(a,env=env,stdout=out,stderr=err,start_new_session=True)
  try:rc=p.wait(timeout=180)
  except subprocess.TimeoutExpired:
   timed=True;os.killpg(p.pid,signal.SIGTERM)
   try:rc=p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);rc=p.wait(timeout=10)
 r={'schema':'numi.cardboard.contact-trial-observer-receipt.v1','case':str(c),'exit_code':rc,'timed_out':timed,'wall_seconds':time.monotonic()-started,'all_inputs_unchanged':verify(),'stdout_sha256':sha(c/'stdout.log'),'stderr_sha256':sha(c/'stderr.log'),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'shipping_qualified':False}
 (c/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');results.append(r);print(json.dumps(r),flush=True)
(d/'report.json').write_text(json.dumps({'results':results,'gpu_workers':1,'requires_parity_adjudication':True},indent=2)+'\n')
