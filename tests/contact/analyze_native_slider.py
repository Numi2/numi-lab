#!/usr/bin/env python3
"""Evaluate a complete native slider trace against the preregistered gates."""
import bisect,hashlib,json,math,pathlib,sys

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
area_reference=json.loads((evidence/'compact-reference-parity-n2-ordered-001.json').read_text())['contact_area_weights_m2']
if not reference_ok or not steps or not all(s['accepted'] for s in steps):raise SystemExit('reference or native acceptance incomplete')
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
torque_ratio=max(max(abs(s['fem_torque_impulse_'+a]+s['owner_moment_impulse_'+a]),
 abs(s['owner_moment_impulse_'+a]+s['support_moment_impulse_'+a]))/
 max(1e-14,1e-5*abs(s['fem_torque_impulse_'+a])) for s in steps for a in ('x','y'))
results={'reference_qualification_passed':reference_ok,'force_bin_relative_error':force_error,
 'sampled_peak_force_relative_error':peak_error,'total_impulse_relative_error':imp_error,
 'maximum_cumulative_impulse_relative_error':cum_error,'minimum_gap_m':min(s['gap'] for s in steps),
 'minimum_midpoint_gap_m':min(s['midpoint_gap'] for s in steps),
 'contact_work_plus_potential_change_J':sum(s['owner_contact_work']+s['fem_contact_work'] for s in steps)+steps[-1]['contact_potential_after_J']-steps[0]['contact_potential_before_J'],
 'maximum_step_work_plus_potential_change_J':max(abs(s['owner_contact_work']+s['fem_contact_work']+s['contact_potential_after_J']-s['contact_potential_before_J']) for s in steps),
 'maximum_owner_work_reconstruction_error_J':max(abs(s['owner_contact_work']-s['owner_wrench_work']) for s in steps),
 'maximum_impulse_balance_gate_ratio':impulse_ratio,'maximum_torque_balance_gate_ratio':torque_ratio,
 'max_independent_joint_residual_Ns':max(s['independent_joint_residual_Ns'] for s in steps),
 'physical_residual_limit_Ns':d['physical_residual_limit_Ns'],
 'slider_position_relative_error':rel(steps[-1]['q'],r['slider_final_z_m']),
 'slider_velocity_relative_error':rel(steps[-1]['v'],r['slider_final_v_mps']),
 'fem_work_absolute_error_J':abs(sum(s['fem_contact_work'] for s in steps)-r['fem_contact_work_J']),
 'slider_work_absolute_error_J':abs(sum(s['owner_contact_work'] for s in steps)-r['slider_contact_work_J']),
 'pair_work_absolute_error_J':abs(sum(s['owner_contact_work']+s['fem_contact_work'] for s in steps)-r['relative_contact_work_J']),
 'fem_contact_work_J':sum(s['fem_contact_work'] for s in steps),
 'slider_contact_work_J':sum(s['owner_contact_work'] for s in steps),
 'max_slider_kinematic_defect_m':max(abs(s['slider_kinematic_defect_m']) for s in steps),
 'max_fem_kinematic_defect_m':max(s['fem_kinematic_defect_m'] for s in steps)}
checks={'force_bin':force_error<=.0025,'sampled_peak':peak_error<=.01,'total_impulse':imp_error<=.0025,
 'cumulative_impulse':cum_error<=.01,'clearance':results['minimum_gap_m']>1e-7 and results['minimum_midpoint_gap_m']>1e-7,
 'equal_opposite_impulse':impulse_ratio<=1,'contact_and_support_torque':torque_ratio<=1,
 'independent_residual':results['max_independent_joint_residual_Ns']<=results['physical_residual_limit_Ns']}
if abs(gap-.0002)<1e-12:
 checks['slider_position']=results['slider_position_relative_error']<=.0025
 checks['slider_velocity']=results['slider_velocity_relative_error']<=.0025
out={'schema':'numi.native-slider-gates.v1','input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
 'reference':refpath.name,'reference_sha256':hashlib.sha256(refpath.read_bytes()).hexdigest(),
 'qualification':qualification.name,'checks':checks,'results':results,'passes':all(checks.values()),
 'limitations':'Does not establish rollback, spatial convergence, or a performance benefit; work is reported without a new numerical gate.'}
print(json.dumps(out,indent=2,sort_keys=True));sys.exit(0 if out['passes'] else 1)
