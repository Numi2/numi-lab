from pathlib import Path
import json,hashlib,subprocess,os,time,datetime,signal,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-primary-cache-fd-action-253-003';b=q/'build-source253-primary-cache-action-002';c=q/'solver/primary-cache-action-diagnostic-regression-599';d=q/'analysis/native599-command-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(s/'frozen-source.json')=='0ea56b99f98a0dc371e50299db10e01ba41be5150c51c5e876d2bef044141289' and sha(b/'source-build.json')=='1b838291f7362318f3b4e3ae4dbd2b78351e1d3a338fc6773d95e19547a44c5b'
f=json.loads((s/'frozen-source.json').read_text());bm=json.loads((b/'source-build.json').read_text());cfg=json.loads((s/'configure-support-source.json').read_text())
assert bm['success']
def verify():return all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(s/n)==h for n,h in cfg['files'].items()) and all(sha(n)==h for n,h in bm['artifacts_sha256'].items())
assert verify();d.mkdir(exist_ok=False);assert not c.exists()
old=json.loads((q/'solver/source210-periodic-observer-off-544/prediction.json').read_text());argv=list(old['argv']);argv[0]=str(b/'matter/numi-matter-cardboard-probe');argv[argv.index('--metallib')+1]=str(b/'matter/shaders/NumiMatter.metallib');argv[argv.index('--output')+1]=str(c/'output');env=dict(old['environment']);assert '--cache-fem-algorithmic-tangent' in argv;env['NM_FEM_OPERATOR_FD_SELECTOR']='0:0'
spec={'schema':'numi.cardboard.native599-command.v1','argv':argv,'environment':env,'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json'),'question':'Does corrected253 capture the actual cached primary action at control0 Newton0 while preserving588 full seven-control production histories?','prediction':'Require complete seven-control production outputs identical588. Capture must explicitly bind cache_mode1 to actualcache/validitybuffers; compare with independentCPUassembledcachedC operator. This .01degree boundedregression doesnotqualifytargetcoupon/convergence, and no noise-floorpass assumed.','full_target_coupon_qualified':False}
sp=d/'command.json';sp.write_text(json.dumps(spec,indent=2)+'\n')
cmd=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_run.py'),'--source-snapshot',str(s),'--runtime-dir',str(b),'--build-provenance',str(b/'source-build.json'),'--case',str(c),'--timeout','900','--diagnostic-run','--diagnostic-capture','NM_FEM_OPERATOR_FD_CAPTURE_PATH','--question',spec['question'],'--prediction',spec['prediction']]
for k,v in sorted(env.items()):cmd+=['--env',k+'='+v]
cmd+=['--input-file',str(sp),'--input-file',str(q/'analysis/source253-build-002/report.json'),'--']+argv
workers=[l for l in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(x in l for x in ['numi-matter-','generic-batch','geometry-replay']) and not l.split()[1].startswith('Z')];assert not workers,workers
(d/'launch.json').write_text(json.dumps({'argv':cmd,'gpu_workers_before':workers,'script_sha256':sha(__file__)},indent=2)+'\n')
clean={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_MATTER_','NUMI_CARDBOARD_'))};clean['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin';clean['PYTHONDONTWRITEBYTECODE']='1'
t=time.monotonic();timed=False
with (d/'wrapper.log').open('xb') as out:
 p=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,env=clean,start_new_session=True)
 print(json.dumps({'pid':p.pid,'case':str(c)}),flush=True)
 try:code=p.wait(timeout=960)
 except subprocess.TimeoutExpired:
  timed=True;os.killpg(p.pid,signal.SIGTERM)
  try:code=p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=10)
r={'exit_code':code,'timed_out':timed,'wall_seconds':time.monotonic()-t,'all_inputs_unchanged':verify(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(code or int(timed or not r['all_inputs_unchanged']))
