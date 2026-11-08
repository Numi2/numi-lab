from pathlib import Path
import csv,hashlib,json,os,shutil,subprocess,time
R=Path.cwd();B=R/'build/crow-implicit-drive-audit-build-01';A=R/'artifacts/crow-rl-implicit-drive-audit-20261108';T=R/'.numi/retained-runtime/implicit-drive-audit-20261108'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
assert json.loads((A/'results.json').read_text())['verdict']=='predicted_saturation_equation_disagreement_confirmed'
inputs=[B/'bounded.cpp',B/'probe.cpp',B/'bounded-probe',T/'libmetalrobo.dylib',T/'metalrobo/MetalRobo.metallib']
plan={'purpose':'Test an alternative bounded backward-Euler motor constitutive row through the existing native canonical generalized ConstraintIR solver. No production runtime integration or crow motor trial.','parent':'Native96-row diagnosis confirms unconditional implicit inertia after pre-clamping under-realizes saturated torque.','model':'M physical+armature is unchanged. Use ordinary native effort mode with zero effort, add generalized motor row J=1, targetVelocity=-K*(q-target)/(D+hK), dissipation=1/(D+hK), compliance0, impulse bounds+/-h*T. Native evaluation yields regularization1/[h(D+hK)]. Native solver and integration own all state. No host trajectory update.','units':'Nm,kg*m2,radians,seconds; generalized angular impulse N*m*s.','conditions':{'development_replications':'Eight exposed initial states from the diagnosis, four durations, CCD disabled/hybrid =64 rows. These are engineering replications, not unused confirmation.','new_conditions':'Eight declared unused mirrored initial states, including unsaturated and saturated moving cases, four durations, both CCD modes =64 rows. Two new velocity upper-stop coupled conditions at v=0, target+/-0.4, four durations, both modes =16 rows.','source_revision':'Compiled8-control prototype was never executed; expanded to16 before this plan/native invocation.','independence':'Manufactured deterministic numerical conditions; no biological or training replicate claims.'},'prediction':{'all144_native_final_velocity_absolute_error_vs_independent_scalar_reference_max':2e-6,'all144_q_integration_absolute_error_max':2e-7,'all144_motor_impulse_bounds_excess_max_Nms':1e-10,'all144_realized_motor_torque_constitutive_residual_max_Nm':1e-7,'all144_momentum_balance_residual_max_Nm':1e-7,'positive_stop_velocity_absolute_max_rad_s':2e-6,'negative_stop_impulse_absolute_max_Nms':1e-10},'validity':'All144 accepted; canonical IR/world compilation succeeds, native publication includes solved motor/stop impulses, loaded dylib matches sealed runtime. Missing row/native rejection is inconclusive.','reference':'Existing independently calibrated closed-form scalar saturated implicit solution; with unilateral upper-velocity stop v<=0, positive-target rest solution v=0 and motor/stop impulses both+h*T. Negative target free solution remains admissible, stop impulse0.','inputs':{str(p.relative_to(R)):sha(p) for p in inputs},'budget':{'native_invocations':1,'native_one_microstep_environments':144,'timeout_seconds':90,'new_output_max_bytes':1000000,'stop':'One invocation, preserve all outcomes, no retry/tuning or promotion.'}}
write(A/'bounded-plan.json',plan)
for src in (B/'bounded.cpp',B/'run_bounded.py'):shutil.copy2(src,A/src.name)
subprocess.run(['git','add',str(A)],check=True);subprocess.run(['git','commit','-m','Seal native bounded motor constitutive alternative'],check=True);subprocess.run(['git','push','origin','HEAD'],check=True)
for p,h in plan['inputs'].items():assert sha(R/p)==h
env=os.environ.copy();env.update(DYLD_LIBRARY_PATH=str(T),DYLD_PRINT_LIBRARIES='1')
cmd=[B/'bounded-probe',T/'metalrobo/MetalRobo.metallib',B/'bounded.tsv'];start=time.monotonic()
with (B/'bounded.stdout').open('xb') as out,(B/'bounded.stderr').open('xb') as err:
    code=subprocess.run([str(x) for x in cmd],env=env,stdout=out,stderr=err,timeout=90).returncode
receipt={'command':[str(x) for x in cmd],'exit_code':code,'seconds':time.monotonic()-start};write(B/'bounded-execution.json',receipt)
report={'execution':receipt,'production_integrated':False,'crow_release_qualified':False,'profile_or_policy_promoted':False}
if code!=0:
    report.update(verdict='inconclusive_native_instrument_failure',failure='Inspect bounded.stderr; no retry.');write(A/'bounded-results.json',report);print(json.dumps(report));raise SystemExit(1)
rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((B/'bounded.tsv').open(),delimiter='\t')];assert len(rows)==144
def clip(v,t):return max(-t,min(t,v))
analysis=[]
for r in rows:
    M,K,D,Tau,h,q,v,t=[r[k] for k in ('mass','k','d','limit','h','q','v','target')]
    unbounded=(M*v+h*K*(t-q))/(M+h*D+h*h*K);ref=max(v-h*Tau/M,min(v+h*Tau/M,unbounded));stop=r['path']>=3
    if stop:ref=min(0.,ref)
    motor=r['motor_impulse'];reaction=r['stop_impulse'];want=clip(K*(t-r['final_q'])-D*r['final_v'],Tau)
    analysis.append(r|{'reference_v':ref,'reference_velocity_error':abs(r['final_v']-ref),'q_integration_error':abs(r['final_q']-(q+h*r['final_v'])),'motor_impulse_bounds_excess':max(0.,abs(motor)-h*Tau),'motor_constitutive_residual_Nm':abs(motor/h-want),'momentum_balance_residual_Nm':abs(M*(r['final_v']-v)/h-(motor-reaction)/h)})
positive=[r for r in analysis if r['path']>=3 and r['lane']==3];negative=[r for r in analysis if r['path']>=3 and r['lane']==4]
metrics={k:max(r[k] for r in analysis) for k in ('reference_velocity_error','q_integration_error','motor_impulse_bounds_excess','motor_constitutive_residual_Nm','momentum_balance_residual_Nm')}
metrics.update(positive_stop_maximum_absolute_velocity=max(abs(r['final_v']) for r in positive),negative_stop_maximum_absolute_impulse=max(abs(r['stop_impulse']) for r in negative))
limits=dict(zip(metrics,(2e-6,2e-7,1e-10,1e-7,1e-7,2e-6,1e-10)))
checks={k:v<=limits[k] for k,v in metrics.items()}
loaded=[s for s in (B/'bounded.stderr').read_text().splitlines() if s.startswith('dyld[') and 'libmetalrobo.dylib' in s];assert loaded and all(s.endswith(str(T/'libmetalrobo.dylib')) for s in loaded)
report.update(verdict='bounded_constitutive_model_passed_native_conditions' if all(checks.values()) else 'bounded_model_prediction_contradicted',rows=len(rows),metrics=metrics,checks=checks,loader_lines=loaded,limits='Constant-coefficient one-microstep authored rows at fixed q inputs. Full runtime needs device-side per-segment refresh, real motor envelopes/delays, task/energy attribution, coupled contact qualification and performance testing. No full crow integration claimed.')
write(A/'bounded-results.json',report);write(B/'bounded-analysis.json',analysis);print(json.dumps(report,indent=2),flush=True)
