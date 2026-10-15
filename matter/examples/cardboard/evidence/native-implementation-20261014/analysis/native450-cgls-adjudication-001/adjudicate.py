from pathlib import Path
import datetime,hashlib,json
B=Path('/Users/home/cardboard-implementation-20261014')
O=B/'analysis/native450-cgls-adjudication-001'
O.mkdir(exist_ok=True)
assert not (O/'report.json').exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths={
 'candidate_audit':B/'analysis/native450-cgls-candidate-audit-001/report.json',
 'candidate_seal':B/'analysis/native450-cgls-candidate-audit-seal-001/seal.json',
 'observer_parity':B/'analysis/native450-capture-off-parity-001/report.json',
 'tangent_energy':B/'analysis/source125-per-tet-tangent-energy-001/report.json',
 'native450_receipt':B/'solver/bounded-cgls64-exact-state-450/receipt.json',
 'native452_receipt':B/'solver/bounded-cgls64-capture-off-parity-452/receipt.json',
 'source142_preflight':B/'analysis/source142-five-file-preflight-001/report.json',
}
a=json.loads(paths['candidate_audit'].read_text());p=json.loads(paths['observer_parity'].read_text())
assert a['run_binding']['prediction_inputs']['all_prediction_inputs_verified']
assert a['armijo_trial']['accepted_non_noop_CGLS_trial']
assert p['parity_pass'] and p['history_comparison']['physical_output_files_byte_exact']==35
assert p['run_on']['receipt_exit_code']==p['run_off']['receipt_exit_code']==2
trial=a['armijo_trial']['records'][1]['selected_environment']
assert trial['selected_alpha']==trial['environment_line_search_alpha']==trial['trial_alpha']==1
stderr=B/'solver/bounded-cgls64-exact-state-450/stderr.log'
if not stderr.exists(): stderr=B/'solver/bounded-cgls64-exact-state-450/stderr.txt'
# Discover the receipt-bound native stderr filename without assuming a suffix.
assert stderr.exists(),list((B/'solver/bounded-cgls64-exact-state-450').iterdir())
records=[]
for line in stderr.read_text().splitlines():
 if line.startswith('fem_residual_armijo_iteration='):
  r=json.loads(line.split('=',1)[1])
  if r.get('root')==1: records.append(r)
post=[]
for r in records:
 n=r.get('newton_iteration',r.get('iteration'))
 if n is not None and 49<=n<=62:
  e=r['environments'][0]
  post.append({'newton_iteration':n,'direction_kind':r.get('direction_kind'),'accepted':e['accepted'],'selected_alpha':e['selected_alpha'],'trial_l2':e['trial_l2']})
assert len(post)==14 and all(r['accepted'] and r['selected_alpha']>0 for r in post)
report={
 'schema':'numi.cardboard.native450_root_adjudication.v1',
 'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'bindings':{k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()},
 'native450_complete_attempt_failed':True,
 'native450_accepted_control_steps':1,
 'selected_directional_native_action_gate':'PASS at the captured nonzero directions; relative A*x error 1.448038139e-7 and A-transpose*r error 1.437122784e-7; not a full-matrix derivative certificate',
 'selected_native_nonlinear_descent':{'base_l2':trial['base_l2'],'trial_l2':trial['trial_l2'],'exact_tested_and_selected_alpha':1,'relative_reduction':1-trial['trial_l2']/trial['base_l2'],'old_newton_certificate_transferred':False},
 'observer_history_parity':{'pass':True,'full_attempt':True,'physical_outputs':35,'armijo_records':64,'true_residual_records':4096},
 'later_observed_primary_steps':post,
 'terminal_trace_limit':'The 64-record cap contains N0-47, both N48 passes, and N49-62. N63 Armijo outcome is unavailable. True-residual traces reach N63 and the terminal bounded control fails/rolls back. Do not infer the unrecorded final Armijo branch.',
 'repeated_control1_trial_admitted':False,
 'repeated_control1_trial_decision':'Defer the proposed same-budget1:all451 experiment. Its GPU mask makes fallback inert on observed accepted N49-62; it could alter unobserved N63 and is not proven redundant. It does not directly address the observed tiny accepted Newton steps.',
 'next_implementation':'Default-off explicit residual-minimizing CGLS-primary direction policy from selected Newton48, with all ordinary nonlinear tests, final certificate and rollback unchanged. This is a new solver experiment, not a constitutive parameter change.',
 'physical_operator_interpretation':'The independent per-tet contraction attributes the captured negative quadratic form mainly to compressive initial-stress/geometric stiffness, offset by a positive aggregate algorithmic material contribution. No sign defect is demonstrated. Source transfer and compression/shear hardening assumptions remain open.',
 'limits':['Native FP32 finite-difference material/element gates remain noise-inconclusive.','No complete coupon cycle, spatial/time/slower-loading convergence or nonlinear-contact trajectory is qualified.','No calibrated corrugated damage, independent starch fixture validation, physical shipping assembly, Fpeak, gravity crush or drop result.','448 is retained as an incomplete ENOSPC operational incident without a terminal native receipt or inferred CGLS outcome.'],
 'six_axis_verdicts':{'solver_correctness':'bounded CGLS helper/action/descent and observer-history parity qualified; full trajectory and derivative/contact qualification open','numerical_convergence':'unqualified','component_calibration':'partial source-bound constituent fits; corrugated damage/debonding open','analogue_validation':'whole-fixture held-out validation open','physical_assembly_in_simulation':'unexecuted','crush_impact_prediction':'unexecuted'},
}
(O/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(O/'adjudicate.py').write_bytes(Path(__file__).read_bytes())
(O/'manifest.json').write_text(json.dumps({'schema':'numi.cardboard.root_adjudication_seal.v1','files':{p.name:sha(p) for p in O.iterdir() if p.is_file()},'physical_validation':False},indent=2)+'\n')
print(O,sha(O/'report.json'))
