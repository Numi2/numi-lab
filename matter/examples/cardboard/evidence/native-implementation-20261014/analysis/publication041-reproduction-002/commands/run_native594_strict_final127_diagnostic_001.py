from pathlib import Path
import json,hashlib,subprocess,os,time,datetime,signal,sys
q=Path('/Users/n/cardboard-implementation-20261014-host-macmini');s=q/'snapshots/source-refined-regional-graph-240-003';b=q/'build-source240-refined-regional-graph-003';c=q/'solver/strict-state-source240-control19-final-newton-diagnostic-594';d=q/'analysis/native594-command-001';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(s/'frozen-source.json')=='bcb37b35391f2dc9ab04492cba44306825540028edb354b219493412ad638644' and sha(b/'source-build.json')=='04087ebb2983861913b5dfb1c985f314f62a515281a8bcd00e4ca1f6fab1f92d'
f=json.loads((s/'frozen-source.json').read_text());bm=json.loads((b/'source-build.json').read_text());cfg=json.loads((s/'configure-support-source.json').read_text());assert bm['success']
def verify():return all(sha(s/n)==h for n,h in f['source_sha256'].items()) and all(sha(s/n)==h for n,h in cfg['files'].items()) and all(sha(n)==h for n,h in bm['artifacts_sha256'].items())
assert verify()
old=json.loads((q/'analysis/fresh-source210-strict-inner-full-coupon-545-command-001/command-spec.json').read_text());a=list(old['native_argv']);a[0]=str(b/'matter/numi-matter-cardboard-probe');a[a.index('--metallib')+1]=str(b/'matter/shaders/NumiMatter.metallib');a[a.index('--output')+1]=str(c/'output');assert '--strict-source-paper-state-certification' not in a;a+=['--strict-source-paper-state-certification'];a[a.index('--newton-iterations')+1]='128';env=dict(old['environment'])
assert a[a.index('--residual-tolerance')+1]=='5e-7' and a[a.index('--newton-iterations')+1]=='128' and a[a.index('--fgmres-restart')+1]=='16' and a[a.index('--fgmres-iterations')+1]=='1024'
assert not any('RESUME' in k or 'FD_SELECTOR' in k for k in env)
cp=q/'solver/strict-state-source240-newton128-full-coupon-585/output/checkpoints/coupon-control-000015.nmcb'
meta=cp.with_suffix('.resume')
assert sha(cp)=='18e699074c60839f5129d9155f9dd95802145a5610b5b83e4fd20c373a0837ae'
a+=['--coupon-resume-checkpoint',str(cp),'--coupon-resume-metadata',str(meta),'--coupon-resume-diagnostic-replay']
k=a.index('--coupon-checkpoint-every');del a[k:k+2]
env.update({'NM_FEM_NEWTON_TRACE_ROOT':'19','NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT':'19','NM_FEM_OPERATOR_FD_SELECTOR':'19:127'})

workers=[l for l in subprocess.check_output(['ps','-axo','pid=,stat=,comm='],text=True).splitlines() if any(x in l for x in ['numi-matter-','generic-batch','geometry-replay']) and not l.split()[1].startswith('Z')];assert not workers,workers;d.mkdir(exist_ok=False);assert not c.exists()
spec={'schema':'numi.cardboard.native594-command.v1','native_argv':a,'environment':env,'source_frozen_sha256':sha(s/'frozen-source.json'),'source_build_sha256':sha(b/'source-build.json'),'question':'Capture strict-projection control19 finalNewton127 exact material/operator state and descent trace from the same accepted checkpoint15; does observer replay preserve585 production suffix?','prediction':'585 failed control19 at R5.44235e-7 after128Newton; final inner1024 iterations stagnated7.99063e-9. No tolerance change or completion assumed. Replay585 acceptedcheckpoint15 with same operator,128Newton,localstrictpolicy and all physical parameters. Add only diagnostic trace/FD capture at19:127 without periodic checkpoint capture (mutually exclusive with resume). Compare actual suffix histories ; no new checkpoint expected from resume. No acceptedprefix convergence claim.','authored_policy_changes_vs545':['--strict-source-paper-state-certification, local correction2e-10 with same32local iterations','same128Newton as585; observers and checkpoint cadence only'],'vs585_physical_and_numerical_controls_unchanged':True,'checkpoint_sha256':sha(cp),'checkpoint_metadata_sha256':sha(meta),'observer_changes':['Newton19 and Armijo19 traces','FD19:127','periodic capture removed as required by resume contract'],'outer_residual_tolerance':5e-7,'restart':16,'native_timeout_s':7200,'all_six_full_sequence_axes_open':True}
sp=d/'command-spec.json';sp.write_text(json.dumps(spec,indent=2)+'\n')
cmd=[str(q/'venvs/cardboard-reference-185/bin/python3'),'-B',str(s/'matter/tools/cardboard_source_bound_run.py'),'--source-snapshot',str(s),'--runtime-dir',str(b),'--build-provenance',str(b/'source-build.json'),'--case',str(c),'--timeout','7200','--question',spec['question'],'--prediction',spec['prediction']]
for k,v in sorted(env.items()):cmd+=['--env',k+'='+v]
for p in [sp,cp,meta]:cmd+=['--input-file',str(p)]
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
