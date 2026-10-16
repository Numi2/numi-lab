from pathlib import Path
import json,hashlib,subprocess,os,time,datetime,signal,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-strict-local-projection-upstream-232';b=q/'build-source232-executed-bound-003';c=q/'solver/strict-state-source232-full-coupon-578';d=q/'analysis/native578-command-002';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(s/'frozen-source.json')=='dce16c539dbc01a1dad5be25b4c4f6b31594f9f7b0704415d175cc73345981f9' and sha(b/'source-build.json')=='c4da6d95ae469b5c63c363d1e57866b18e35513906ecba86e975e1a76f449ffe'
f=json.loads((s/'frozen-source.json').read_text());bm=json.loads((b/'source-build.json').read_text());cfg=json.loads((s/'configure-support-source.json').read_text());assert bm['success']
def verify():return all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(s/n)==h for n,h in cfg['files'].items()) and all(sha(n)==h for n,h in bm['artifacts_sha256'].items())
assert verify()
old=json.loads((q/'analysis/fresh-source210-strict-inner-full-coupon-545-command-001/command-spec.json').read_text());a=list(old['native_argv']);a[0]=str(b/'matter/numi-matter-cardboard-probe');a[a.index('--metallib')+1]=str(b/'matter/shaders/NumiMatter.metallib');a[a.index('--output')+1]=str(c/'output');assert '--strict-source-paper-state-certification' not in a;a+=['--strict-source-paper-state-certification'];env=dict(old['environment'])
assert a[a.index('--residual-tolerance')+1]=='5e-7' and a[a.index('--newton-iterations')+1]=='64' and a[a.index('--fgmres-restart')+1]=='16' and a[a.index('--fgmres-iterations')+1]=='1024'
assert not any('RESUME' in k or 'FD_SELECTOR' in k for k in env)
workers=[l for l in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(x in l for x in ['numi-matter-','generic-batch','geometry-replay']) and not l.split()[1].startswith('Z')];assert not workers,workers;d.mkdir(exist_ok=False);assert not c.exists()
spec={'schema':'numi.cardboard.native578-command.v1','native_argv':a,'environment':env,'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json'),'question':'Does the locally verified strict projected-state correction complete the fresh60-control coupon under the original545 mesh, time, material law, tool path, strict inner forcing and outer tolerance?','prediction':'No pass assumed. Start from initial history; do not resume previous-looser-projection accepted states. Candidate strict local state correction tolerance2e-10 fixes10 targeted material tangent rows, but global19 derivative and full convergence remain unqualified. Preserve every accepted/rejected/timeout history and periodic checkpoint.','only_authored_policy_change_vs545':'--strict-source-paper-state-certification; same32local iteration budget and unchanged residual/outer criteria','outer_residual_tolerance':5e-7,'restart':16,'native_timeout_s':7200,'all_six_full_sequence_axes_open':True}
(d/'strict-candidate-manual-build.json').write_bytes((q/'analysis/source232-strict-local-projection-integration-001/source-build.json').read_bytes())
sp=d/'command-spec.json';sp.write_text(json.dumps(spec,indent=2)+'\n')
cmd=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_run.py'),'--source-snapshot',str(s),'--runtime-dir',str(b),'--build-provenance',str(b/'source-build.json'),'--case',str(c),'--timeout','7200','--question',spec['question'],'--prediction',spec['prediction']]
for k,v in sorted(env.items()):cmd+=['--env',k+'='+v]
for p in [sp,d/'strict-candidate-manual-build.json',q/'analysis/native563-strict-local-projection-adjudication-001/report.json']:cmd+=['--input-file',str(p)]
cmd+=['--']+a
(d/'launch.json').write_text(json.dumps({'argv':cmd,'script_sha256':sha(__file__),'gpu_workers_before':workers,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
clean={k:v for k,v in os.environ.items() if not k.startswith(('NM_','NUMI_MATTER_','NUMI_CARDBOARD_'))};clean['PATH']='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin';clean['PYTHONDONTWRITEBYTECODE']='1';t=time.monotonic();timed=False
with (d/'wrapper.log').open('xb') as out:
 p=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,env=clean,start_new_session=True);print(json.dumps({'pid':p.pid,'case':str(c)}),flush=True)
 try:code=p.wait(timeout=7260)
 except subprocess.TimeoutExpired:
  timed=True;os.killpg(p.pid,signal.SIGTERM)
  try:code=p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=10)
r={'exit_code':code,'timed_out':timed,'wall_seconds':time.monotonic()-t,'all_inputs_unchanged':verify(),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(d/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True);sys.exit(code or int(timed or not r['all_inputs_unchanged']))
