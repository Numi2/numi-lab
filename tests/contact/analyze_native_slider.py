#!/usr/bin/env python3
"""Evaluate a complete native slider trace against the preregistered gates."""
import bisect,hashlib,json,math,pathlib,sys
from slider_work_validation import validate_fem_work_rows

if len(sys.argv)!=2: raise SystemExit('usage: analyze_native_slider.py NATIVE_TRACE.json')
path=pathlib.Path(sys.argv[1]);d=json.loads(path.read_text())
evidence=pathlib.Path(__file__).resolve().parents[2]/'docs/evidence/midpoint-first-failure-20261109'
if d.get('schema')!='numi.slider-coupled-diagnostic.v1':raise SystemExit('unknown native trace')
gap=d['initial_gap']
if abs(gap-.000095)<1e-12:
 refpath=evidence/'slider-near-rk4-h025.json';qualification=evidence/'slider-near-reference-analysis.json'
 q=json.loads(qualification.read_text()); reference_ok=q['temporal_reference']['passes']
elif abs(gap-.0002)<1e-12:
 refpath=evidence/'slider-v2-rk4-h025.json';qualification=evidence/'slider-reference-analysis.json'
 q=json.loads(qualification.read_text());reference_ok=q['passes']
else:raise SystemExit('unregistered initial gap')
r=json.loads(refpath.read_text());steps=d['steps'];h=d['h'];duration=.01
if abs(gap-.000095)<1e-12:
 state_qualification=json.loads((evidence/'slider-near-state-reference-001.json').read_text())
 if set(state_qualification['inputs_sha256'])!={'slider-near-rk4-h050.json','slider-near-rk4-h025.json'}:
  raise SystemExit('near-contact state reference inputs missing')
 for name,expected in state_qualification['inputs_sha256'].items():
  if hashlib.sha256((evidence/name).read_bytes()).hexdigest()!=expected:
   raise SystemExit('near-contact state reference changed')
 if not state_qualification['passes']:
  raise SystemExit('near-contact state reference is not time-qualified')
work_label='near' if abs(gap-.000095)<1e-12 else 'primary'
work_path=evidence/('slider-work-history-002-'+work_label+'-h025.json')
work_qualification=evidence/'slider-work-history-reference-002-analysis.json'
work_receipt=evidence/'slider-work-history-reference-002-execution.json'
work_audit=json.loads(work_qualification.read_text());work_execution=json.loads(work_receipt.read_text())
if not (work_audit['passes'] and work_audit['fixtures'][work_label]['passes'] and
        work_execution['result']=='four references completed with exact historical-field parity'):
 raise SystemExit('work reference is not qualified')
for label,index in (('050',2 if work_label=='near' else 0),('025',3 if work_label=='near' else 1)):
 candidate=evidence/('slider-work-history-002-'+work_label+'-h'+label+'.json')
 run=work_execution['runs'][index]
 if not (run['historic_fields_exact_match'] and run['output_sha256']==hashlib.sha256(candidate.read_bytes()).hexdigest()):
  raise SystemExit('work reference is not hash-bound')
work_reference=json.loads(work_path.read_text())
torque_qualification=json.loads((evidence/'slider-torque-reference-001.json').read_text())
torque_ref=torque_qualification['fixtures'][work_label]
expected_torque_files={('slider-near-rk4-h' if work_label=='near' else 'slider-v2-rk4-h')+label+'.json' for label in ('050','025')}
if not torque_qualification['passes'] or not torque_ref['passes'] or set(torque_ref['inputs_sha256'])!=expected_torque_files:
 raise SystemExit('torque reference is not qualified')
for name,expected in torque_ref['inputs_sha256'].items():
 if hashlib.sha256((evidence/name).read_bytes()).hexdigest()!=expected:
  raise SystemExit('torque reference changed')

area_reference=json.loads((evidence/'compact-reference-parity-n2-ordered-001.json').read_text())['contact_area_weights_m2']
if not reference_ok:raise SystemExit('independent force reference did not qualify')
if not steps:raise SystemExit('native rollout has no recorded steps')
for index,step in enumerate(steps):
 if not step.get('accepted',False):
  raise SystemExit('native step {} rejected: world_status={} matter_status={} host_status={} failure_index={}'.format(
   step.get('step',index),step.get('world_status','unavailable'),
   step.get('matter_status','unavailable'),step.get('host_status','unavailable'),
   step.get('failure_index','unavailable')))
if not any(math.isclose(h,x,rel_tol=1e-6) for x in (15.625e-6,7.8125e-6)):
 raise SystemExit('timestep outside the preregistered native comparison')
if [s['step'] for s in steps]!=list(range(len(steps))) or abs(len(steps)*h-duration)>1e-8:
 raise SystemExit('trace is not a complete ordered 10 ms rollout')
if not (math.isclose(d['slider_mass_kg'],.01,rel_tol=1e-6) and math.isclose(d['initial_slider_velocity_mps'],.02,rel_tol=1e-6)
 and math.isclose(d['reference_area_m2'],.0004,rel_tol=2e-6) and math.isclose(sum(d['reference_area_weights_m2']),.0004,rel_tol=2e-6)
 and len(d['reference_area_weights_m2'])==len(area_reference)==27
 and all(abs(a-b)<=1e-11 for a,b in zip(d['reference_area_weights_m2'],area_reference))
 and math.isclose(d['activation_width_m'],.0001,rel_tol=1e-6)
 and math.isclose(d['contact_coefficient'],1e16,rel_tol=1e-6)):raise SystemExit('fixture inputs differ from reference')
if not all(math.isfinite(v) for s in steps for v in s.values() if isinstance(v,(float,int))):raise SystemExit('nonfinite trace')
imp=[s['fem_impulse'] for s in steps];times=[0.];cumulative=[0.]
for x in imp:times.append(times[-1]+h);cumulative.append(cumulative[-1]+x)
def sample(t):
 i=bisect.bisect_left(times,t)
 if i==0:return 0.
 if i>=len(times):return cumulative[-1]
 return cumulative[i-1]+(cumulative[i]-cumulative[i-1])*(t-times[i-1])/h
bins=[(sample((i+1)*.001)-sample(i*.001))/.001 for i in range(10)]
rel=lambda a,b:abs(a-b)/max(abs(b),1e-30)
refpeak=r['peak_force_N'];refimp=r['impulse_Ns']
force_error=max(abs(a-b) for a,b in zip(bins,r['force_bins_N']))/max(r['force_bins_N'])
peak_error=rel(max(x/h for x in imp),refpeak)
imp_error=rel(cumulative[-1],refimp)
cum_error=max(abs(sample(t[0])-t[2]) for t in r['trace'])/abs(refimp)
impulse_ratio=max(abs(s['fem_impulse']+s['owner_momentum_impulse'])/max(1e-12,1e-5*abs(s['fem_impulse'])) for s in steps)
wrench_impulse_ratio=max(abs(s['fem_impulse']+s['owner_wrench_impulse'])/max(1e-12,1e-5*abs(s['fem_impulse'])) for s in steps)
# The existing impulse gate induces a work-reconstruction bound: impulse
# tolerance times the actual midpoint speed, plus a double roundoff floor.
previous_v=d['initial_slider_velocity_mps'];work_ratio=0.
fem_work_ratio=max(validate_fem_work_rows(s,d) for s in steps)
for s in steps:
 midpoint_v=.5*(previous_v+s['v'])
 bound=max(1e-12,1e-5*abs(s['fem_impulse']))*abs(midpoint_v)+1e-24
 work_ratio=max(work_ratio,abs(s['owner_contact_work']-s['owner_wrench_work'])/bound)
 previous_v=s['v']
# Compare cumulative work as independently reconstructed native step integrals.
# Signed pair work can approach zero, so normalize each history to its own
# maximum reference magnitude with the predeclared absolute floor.
native_work=[[0.],[0.],[0.]]
for step in steps:
 fem,slider=step['fem_contact_work'],step['owner_contact_work']
 for values,increment in zip(native_work,(fem,slider,fem+slider)):
  values.append(values[-1]+increment)
def sample_work(t,values):
 i=bisect.bisect_left(times,t)
 if i==0:return values[0]
 if i>=len(times):return values[-1]
 return values[i-1]+(values[i]-values[i-1])*(t-times[i-1])/h
work_results={}
for col,name in enumerate(('fem','slider','relative'),1):
 values=native_work[col-1]; reference=work_reference['work_trace']
 maximum=max(abs(sample_work(row[0],values)-row[col]) for row in reference)
 scale=max(max(abs(row[col]) for row in reference),1e-12)
 work_results[name]={'maximum_absolute_error_J':maximum,'reference_history_scale_J':scale,
                     'maximum_normalized_error':maximum/scale,
                     'final_absolute_error_J':abs(values[-1]-reference[-1][col])}
torque_ratio=max(max(abs(s['fem_torque_impulse_'+a]+s['owner_moment_impulse_'+a]),
 abs(s['owner_moment_impulse_'+a]+s['support_moment_impulse_'+a]))/
 max(1e-14,1e-5*abs(s['fem_torque_impulse_'+a])) for s in steps for a in ('x','y'))
torque_reference_results={}
for axis in ('x','y'):
 value=sum(step['fem_torque_impulse_'+axis] for step in steps)
 reference=r['torque_impulse_'+axis+'_Nms']
 error=abs(value-reference)
 torque_reference_results[axis]={'native_total_Nms':value,'reference_total_Nms':reference,
                                'absolute_error_Nms':error,
                                'normalized_error':error/max(abs(reference),1e-12)}
results={'reference_qualification_passed':reference_ok,'work_history_comparison':work_results,'torque_reference_comparison':torque_reference_results,'force_bin_relative_error':force_error,
 'sampled_peak_force_relative_error':peak_error,'total_impulse_relative_error':imp_error,
 'maximum_cumulative_impulse_relative_error':cum_error,'minimum_gap_m':min(s['gap'] for s in steps),
 'minimum_midpoint_gap_m':min(s['midpoint_gap'] for s in steps),
 'contact_work_plus_potential_change_J':sum(s['owner_contact_work']+s['fem_contact_work'] for s in steps)+steps[-1]['contact_potential_after_J']-steps[0]['contact_potential_before_J'],
 'maximum_step_work_plus_potential_change_J':max(abs(s['owner_contact_work']+s['fem_contact_work']+s['contact_potential_after_J']-s['contact_potential_before_J']) for s in steps),
 'maximum_owner_work_reconstruction_error_J':max(abs(s['owner_contact_work']-s['owner_wrench_work']) for s in steps),
 'maximum_impulse_balance_gate_ratio':impulse_ratio,'maximum_wrench_impulse_balance_gate_ratio':wrench_impulse_ratio,
 'maximum_torque_balance_gate_ratio':torque_ratio,
 'maximum_owner_work_reconstruction_gate_ratio':work_ratio,
 'maximum_fem_work_reconstruction_gate_ratio':fem_work_ratio,
 'max_independent_joint_residual_Ns':max(s['independent_joint_residual_Ns'] for s in steps),
 'physical_residual_limit_Ns':d['physical_residual_limit_Ns'],
 'slider_position_relative_error':abs(steps[-1]['q']-r['slider_final_z_m'])/max(abs(r['slider_final_z_m']),1e-8),
 'slider_velocity_relative_error':abs(steps[-1]['v']-r['slider_final_v_mps'])/max(abs(r['slider_final_v_mps']),1e-6),
 'fem_work_absolute_error_J':abs(sum(s['fem_contact_work'] for s in steps)-r['fem_contact_work_J']),
 'slider_work_absolute_error_J':abs(sum(s['owner_contact_work'] for s in steps)-r['slider_contact_work_J']),
 'pair_work_absolute_error_J':abs(sum(s['owner_contact_work']+s['fem_contact_work'] for s in steps)-r['relative_contact_work_J']),
 'fem_contact_work_J':sum(s['fem_contact_work'] for s in steps),
 'slider_contact_work_J':sum(s['owner_contact_work'] for s in steps),
 'max_slider_kinematic_defect_m':max(abs(s['slider_kinematic_defect_m']) for s in steps),
 'max_fem_kinematic_defect_m':max(s['fem_kinematic_defect_m'] for s in steps)}
checks={'force_bin':force_error<=.0025,'sampled_peak':peak_error<=.01,'total_impulse':imp_error<=.0025,
 'cumulative_impulse':cum_error<=.01,'clearance':results['minimum_gap_m']>1e-7 and results['minimum_midpoint_gap_m']>1e-7,
 'equal_opposite_impulse':impulse_ratio<=1,'equal_opposite_wrench_impulse':wrench_impulse_ratio<=1,
 'contact_and_support_torque':torque_ratio<=1,
 'torque_x_reference':torque_reference_results['x']['normalized_error']<=.01,
 'torque_y_reference':torque_reference_results['y']['normalized_error']<=.01,
 'owner_work_reconstruction':work_ratio<=1,
 'fem_work_reconstruction':fem_work_ratio<=1,
 'fem_work_history':work_results['fem']['maximum_normalized_error']<=.01,
 'slider_work_history':work_results['slider']['maximum_normalized_error']<=.01,
 'relative_work_history':work_results['relative']['maximum_normalized_error']<=.01,
 'independent_residual':results['max_independent_joint_residual_Ns']<=results['physical_residual_limit_Ns']}
checks['slider_position']=results['slider_position_relative_error']<=.0025
checks['slider_velocity']=results['slider_velocity_relative_error']<=.0025
out={'schema':'numi.native-slider-gates.v1','input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
 'reference':refpath.name,'reference_sha256':hashlib.sha256(refpath.read_bytes()).hexdigest(),
 'qualification':qualification.name,'work_reference':work_path.name,
 'work_reference_sha256':hashlib.sha256(work_path.read_bytes()).hexdigest(),'checks':checks,'results':results,'passes':all(checks.values()),
 'limitations':'Does not establish rollback, spatial convergence, or a performance benefit; both work reconstructions use the impulse gate, with FEM checked per node, the three work histories use the predeclared 1% normalized gate, while potential-energy differences are diagnostic.'}
print(json.dumps(out,indent=2,sort_keys=True));sys.exit(0 if out['passes'] else 1)
