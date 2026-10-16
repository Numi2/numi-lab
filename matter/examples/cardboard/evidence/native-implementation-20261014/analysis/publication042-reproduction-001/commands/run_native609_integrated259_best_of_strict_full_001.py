from pathlib import Path
import json,hashlib,subprocess,os,time,datetime,signal,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-integrated-globalization-contact-memory-259-003';b=q/'build-source259-integrated-003';c=q/'solver/integrated259-best-of-strict-full-coupon-609';d=q/'analysis/native609-command-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(s/'frozen-source.json')=='663a2bf205f2439200077d997804c429c33635d8606462f7a3f700a2d29a2743' and sha(b/'source-build.json')=='f524a5bc44c8bbe2e30dd8662bc5e3e7dcfaa475bc0f84b2b1e1057abe47aa54'
f=json.loads((s/'frozen-source.json').read_text());bm=json.loads((b/'source-build.json').read_text());cfg=json.loads((s/'configure-support-source.json').read_text())
assert bm['success']
def verify():return all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(s/n)==h for n,h in cfg['files'].items()) and all(sha(n)==h for n,h in bm['artifacts_sha256'].items())
assert verify();d.mkdir(exist_ok=False);assert not c.exists()
old=json.loads((q/'solver/strict-state-source240-newton128-full-coupon-585/prediction.json').read_text());argv=list(old['argv']);argv[0]=str(b/'matter/numi-matter-cardboard-probe');argv[argv.index('--metallib')+1]=str(b/'matter/shaders/NumiMatter.metallib');argv[argv.index('--output')+1]=str(c/'output');env=dict(old['environment']);env['NM_FEM_BASE_CGLS_BEST_OF']='1'
spec={'schema':'numi.cardboard.native609-command.v1','argv':argv,'environment':env,'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json'),'question':'Can native best-of globalization pass the original strict control19 stall and complete the unchanged60-control .375degree coupon schedule?','prediction':'Fresh initial state, no cross-program checkpoint resume. Keep585 recipe, tools,8/2/1 mesh,dt5e-5,R5e-7,128Newton,FGMRES1024/restart16,local32/strict projection,mu1e-3/64CGLS and60-control loading/hold/unload/relax/release/post-release duration exactly. The sole operational change is BEST_OF=1. A healthy accepted Newton direction is compared with a separately tested CGLS direction from the same candidate/history, using the strictly lower recomputed residual and exactly its tested alpha. Full production certificate and rollback remain authoritative. Failure/timeouts remain failures; accepted prefixes do not qualify full history. This is still a bounded .375degree solver fixture, not90degree scoring, mesh/time convergence or assembly.','full_target_coupon_qualified':False}
sp=d/'command.json';sp.write_text(json.dumps(spec,indent=2)+'\n')
cmd=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_run.py'),'--source-snapshot',str(s),'--runtime-dir',str(b),'--build-provenance',str(b/'source-build.json'),'--case',str(c),'--timeout','7200','--question',spec['question'],'--prediction',spec['prediction']]
for k,v in sorted(env.items()):cmd+=['--env',k+'='+v]
cmd+=['--input-file',str(sp),'--input-file',str(q/'analysis/source259-build-003/report.json'),'--']+argv
workers=[l for l in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(x in l for x in ['numi-matter-','generic-batch','geometry-replay']) and not l.split()[1].startswith('Z')];assert not workers,workers
(d/'launch.json').write_text(json.dumps({'argv':cmd,'gpu_workers_before':workers,'script_sha256':sha(__file__)},indent=2)+'\n')
clean={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_MATTER_','NUMI_CARDBOARD_'))};clean['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin';clean['PYTHONDONTWRITEBYTECODE']='1'
t=time.monotonic();timed=False
with (d/'wrapper.log').open('xb') as out:
 p=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,env=clean,start_new_session=True)
 print(json.dumps({'pid':p.pid,'case':str(c)}),flush=True)
 try:code=p.wait(timeout=7260)
 except subprocess.TimeoutExpired:
  timed=True;os.killpg(p.pid,signal.SIGTERM)
  try:code=p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=10)
r={'exit_code':code,'timed_out':timed,'wall_seconds':time.monotonic()-t,'all_inputs_unchanged':verify(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(code or int(timed or not r['all_inputs_unchanged']))
