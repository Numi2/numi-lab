from pathlib import Path
import json,hashlib,datetime,subprocess,os,time,importlib.util,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini')
s=q/'snapshots/source-qualification-checkpoint-proof-221-003';b=q/'build-source221-checkpoint-proof-003'
c=q/'solver/control19-newton3-small-epsilon-resume-diagnostic-564';d=q/'analysis/native564-command-001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((q/'analysis/fresh-source210-strict-inner-full-coupon-545-command-001/command-spec.json').read_text())
f=json.loads((s/'frozen-source.json').read_text());bm=json.loads((b/'source-build.json').read_text())
assert sha(s/'frozen-source.json')=='fb885540a2f096ee70df5a03701a8d2710832c6df402e8e097ff95c736e7c1a5' and sha(b/'source-build.json')=='66652297bd3af8707e9999724dec12fc8994ad88ee17904bd10f0e79f4a1d677' and bm['success']
assert all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(n)==h for n,h in bm['artifacts_sha256'].items())
checkpoint=q/'solver/fresh-source210-strict-inner-full-coupon-545/output/checkpoints/coupon-control-000015.nmcb'
metadata=checkpoint.with_suffix('.resume')
assert sha(checkpoint)=='99b25ef939234f42b0265c7131c40d3e5d81fad439534800c0f0e93984f40e82'
assert not c.exists();d.mkdir(exist_ok=False)
argv=list(old['native_argv']);argv[argv.index('--output')+1]=str(c/'output')
argv=[v.replace(str(q/'build-source210-combined-001'),str(b)) for v in argv]
i=argv.index('--coupon-checkpoint-every');del argv[i:i+2]
argv+=['--coupon-resume-checkpoint',str(checkpoint),'--coupon-resume-metadata',str(metadata),'--coupon-resume-diagnostic-replay']
env=dict(old['environment']);env['NM_FEM_OPERATOR_FD_SELECTOR']='19:3';env['NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT']='19';env['NM_FEM_OPERATOR_FD_EPSILON_SCALE']='0.00390625'
prediction={'schema':'numi.cardboard.native564-command.v1','native_argv':argv,'environment':env,'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json'),'checkpoint_sha256':sha(checkpoint),'metadata_sha256':sha(metadata),'question':'At the exact control19 Newton3 state, reduce diagnostic epsilon by256 to test smooth tangent/FD consistency while retaining the complete original production suffix.','prediction':'No derivative pass assumed. Epsilons1/8192 and1/16384 may resolve curvature or remain limited by FP32 noise. Compare against original545/555 suffix and exact independent FP64 material/assembly reference; no solver or physical qualification.','source_runner_proof':'Source221 validates output-free original qualification runner records and corrected53-key registry; no historical metadata altered. Twelve real and tampered proof cases passed CPU tests.','outer_residual_tolerance_unchanged':5e-7,'timeout_s':1800}
sp=d/'command-spec.json';sp.write_text(json.dumps(prediction,indent=2)+'\n')
cmd=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_run.py'),'--source-snapshot',str(s),'--runtime-dir',str(b),'--build-provenance',str(b/'source-build.json'),'--case',str(c),'--timeout','1800','--diagnostic-run','--diagnostic-capture','NM_FEM_OPERATOR_FD_CAPTURE_PATH','--question',prediction['question'],'--prediction',prediction['prediction']]
for k,v in sorted(env.items()):cmd+=['--env',k+'='+v]
for p in [sp,checkpoint,metadata,q/'analysis/source221-runner-proof-cpu-tests-001/report.json']:cmd+=['--input-file',str(p)]
cmd+=['--']+argv
workers=[l for l in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(x in l for x in ['numi-matter-','generic-batch','geometry-replay']) and not l.split()[1].startswith('Z')]
assert not workers,workers
(d/'launch.json').write_text(json.dumps({'argv':cmd,'workers':workers,'script_sha256':sha(__file__),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
clean={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_MATTER_','NUMI_CARDBOARD_'))};clean['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin';clean['PYTHONDONTWRITEBYTECODE']='1'
t=time.monotonic();timed=False;code=None
with (d/'wrapper.log').open('xb') as out:
 p=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,env=clean,start_new_session=True)
 print(json.dumps({'started':str(c),'pid':p.pid}),flush=True)
 try:code=p.wait(timeout=1920)
 except subprocess.TimeoutExpired:
  import signal
  timed=True;os.killpg(p.pid,signal.SIGTERM)
  try:code=p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=10)
r={'exit_code':code,'timed_out':timed,'wall_seconds':time.monotonic()-t,'source_and_artifacts_unchanged':all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(n)==h for n,h in bm['artifacts_sha256'].items()),'checkpoint_unchanged':sha(checkpoint)==prediction['checkpoint_sha256'] and sha(metadata)==prediction['metadata_sha256'],'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(d/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(code or int(timed or not r['source_and_artifacts_unchanged'] or not r['checkpoint_unchanged']))
