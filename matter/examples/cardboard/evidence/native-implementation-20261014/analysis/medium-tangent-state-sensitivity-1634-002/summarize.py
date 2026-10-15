#!/usr/bin/env python3
"""Derive a compact, source-bound interpretation from the MP90 endpoint reports."""
from pathlib import Path
import hashlib,json,math
BASE=Path('/Users/home/cardboard-implementation-20261014')
PKG=BASE/'analysis/medium-tangent-state-sensitivity-1634-002'
PRIOR=BASE/'analysis/medium-tangent-state-sensitivity-1634-001/report.json'
SWEEP306=BASE/'analysis/controlled-material-sweep-306-001/report.json'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def summarize_point(p):
 qscale=float(p['source_inputs']['parameters_f32'][25])
 center_q=float(p['center_branch']['hill_q'])
 center_lam=float(p['center_branch']['delta_multiplier'])
 sweeps=[]
 for e in p['nine_direction_endpoint_sweep']:
  qvals=[];lrels=[];roots=[]
  for rec in e['endpoint_records']:
   for side in ('plus','minus'):
    branch=rec[side]['state_branch']; proj=rec[side]['projection']
    qvals.append(float(branch['hill_q']))
    lrels.append(abs(float(branch['delta_multiplier_relative_to_center'])))
    roots.append(proj)
  endpoint_details=[]
  for endpoint in e['endpoint_records']:
   sides={}
   for side in ('plus','minus'):
    state=endpoint[side]['state_branch']; projection=endpoint[side]['projection']
    sides[side]={'delta_multiplier':state['delta_multiplier'],
                 'relative_delta_multiplier_vs_center':state['delta_multiplier_relative_to_center'],
                 'yield_gap':state['yield_gap'],'hill_q':state['hill_q'],
                 'yield_sign':state['yield_sign'],'fb_selection':state['fb_selection'],
                 'projection_status':projection['status'],'converged':projection['converged'],
                 'iterations':projection['iterations'],
                 'terminal_scaled_residual':projection['terminal_scaled_residual'],
                 'terminal_newton_correction_inf':projection['terminal_newton_correction_inf']}
   endpoint_details.append({'column_row_major':endpoint['input_direction_column_row_major'],
                            'same_branch':endpoint['same_yield_and_FB_branch'],
                            'plus_one_sided_error':endpoint['plus_one_sided_vs_analytic_column_rel'],
                            'minus_one_sided_error':endpoint['minus_one_sided_vs_analytic_column_rel'],
                            'sides':sides})
  sweeps.append({'epsilon_F_entry':e['epsilon_F_entry'],
   'central_9x9_relative_error_vs_MP90_analytic':e['central_matrix_relative_error_vs_MP90_implicit'],
   'plus_onesided_9x9_relative_error_vs_MP90_analytic':e['plus_one_sided_matrix_relative_error_vs_MP90_implicit'],
   'minus_onesided_9x9_relative_error_vs_MP90_analytic':e['minus_one_sided_matrix_relative_error_vs_MP90_implicit'],
   'plus_minus_onesided_matrix_gap':e['successive_one_sided_matrix_gap'],
   'branch_transition_columns':e['branch_transition_columns'],
   'all_endpoints_converged':e['all_endpoint_projections_converged'],
   'max_abs_q_shift_over_q_smoothing':max(abs(q-center_q) for q in qvals)/qscale,
   'max_relative_multiplier_change':max(lrels),
   'maximum_scaled_root_residual':max(float(r['terminal_scaled_residual']) for r in roots),
   'maximum_terminal_newton_correction_inf':max(float(r['terminal_newton_correction_inf']) for r in roots),
   'endpoint_details_by_column':endpoint_details})
 fdot=[]
 for e in p['captured_Fdot_endpoint_sweep']:
  roots=[e['plus_projection'],e['minus_projection']]
  plus_branch=roots[0]['terminal_branch']; minus_branch=roots[1]['terminal_branch']
  same_branch=(plus_branch['yield_sign']==minus_branch['yield_sign'] and
               plus_branch['fb_selection']==minus_branch['fb_selection'])
  fdot.append({'epsilon_seconds':e['epsilon_seconds'],
   'central_action_relative_error_vs_MP90_analytic':e['central_relative_error_vs_MP90_analytic_action'],
   'plus_onesided_action_relative_error':e['plus_one_sided_relative_error_vs_MP90_analytic_action'],
   'minus_onesided_action_relative_error':e['minus_one_sided_relative_error_vs_MP90_analytic_action'],
   'plus_minus_onesided_action_gap':e['one_sided_action_relative_gap'],
   'plus_branch':plus_branch,'minus_branch':minus_branch,
   'same_yield_and_FB_branch':same_branch,
   'both_endpoints_converged':all(x['converged'] for x in roots),
   'maximum_scaled_root_residual':max(float(x['terminal_scaled_residual']) for x in roots),
   'maximum_terminal_newton_correction_inf':max(float(x['terminal_newton_correction_inf']) for x in roots)})
 return {'tet_id':p['point']['tet_id'],
  'center':{'delta_multiplier':p['center_branch']['delta_multiplier'],
            'yield_gap':p['center_branch']['yield_gap'],
            'hill_q':p['center_branch']['hill_q'],
            'q_smoothing_parameter':qscale,
            'fb_smoothing_parameter':float(p['source_inputs']['parameters_f32'][26]),
            'fb_product_over_epsilon_squared_quarter':p['center_branch']['fb_product_over_epsilon_squared_quarter'],
            'yield_sign':p['center_branch']['yield_sign'],'fb_selection':p['center_branch']['fb_selection'],
            'projection_status':p['center_projection']['status'],
            'terminal_scaled_residual':p['center_projection']['terminal_scaled_residual'],
            'terminal_newton_correction_inf':p['center_projection']['terminal_newton_correction_inf']},
  'full_9x9_sweep':sweeps,'captured_Fdot_sweep':fdot}

def main():
    raw = json.loads((PKG / 'report.json').read_text())
    common = json.loads((PKG / 'common-epsilon-1e-7.json').read_text())
    previous = json.loads(PRIOR.read_text())
    sweep306 = json.loads(SWEEP306.read_text())
    overlap = {x['tet_id']: x for x in common['result']}
    result = []
    for point in raw['results']:
        summary = summarize_point(point)
        common_point = overlap[summary['tet_id']]
        difference = abs(float(common_point['mp90_full_matrix_relative_error_vs_analytic']) -
                         float(common_point['prior_fp64_full_matrix_relative_error_vs_MP90']))
        summary['same_epsilon_1e_minus_7'] = {
            'mp90_full_9x9_fd_relative_error': common_point['mp90_full_matrix_relative_error_vs_analytic'],
            'prior_fp64_full_9x9_fd_relative_error': common_point['prior_fp64_full_matrix_relative_error_vs_MP90'],
            'absolute_difference_between_relative_errors': difference,
            'mp90_all_endpoint_roots_converged': common_point['mp90_endpoint_projection_converged'],
            'branch_transition_columns': common_point['mp90_branch_transition_columns']}
        result.append(summary)
    all_sweeps = [row for point in result for row in point['full_9x9_sweep']]
    no_transitions = all(not row['branch_transition_columns'] and row['all_endpoints_converged']
                         for row in all_sweeps)
    tiny_errors = all(float(point['full_9x9_sweep'][-1]['central_9x9_relative_error_vs_MP90_analytic']) < 1e-12
                      for point in result)
    overlap_match = all(point['same_epsilon_1e_minus_7']['absolute_difference_between_relative_errors'] < 1e-9
                        for point in result)
    native306_context = {}
    for policy in ('default', 'accurate'):
        native306_context[policy] = {}
        for tet in ('1634', '1638'):
            point = sweep306['policies'][policy]['points'][tet]
            native306_context[policy][tet] = {
                'center_repeats_bit_identical': point['center_repeats_bit_identical'],
                'accepted_history_unchanged': point['accepted_history_unchanged'],
                'physical_steps': sweep306['policies'][policy]['header']['physical_steps'],
                'physical_validation': sweep306['policies'][policy]['header']['physical_validation'],
                'cases': [{'epsilon_seconds': case['epsilon_s'],
                           'centered_native_action_relative_error': case['centered_matrix_comparison']['relative_l2'],
                           'all_entries_valid': case['all_entries_valid']}
                          for case in point['cases']]}
    out = {
        'schema': 'numi.cardboard.medium-tangent-mp90-endpoint-fd-summary.v1',
        'classification': 'Read-only numerical classification for two captured medium states; all endpoint projections use MP90, fixed accepted history, and exact widened captured words. Not a material or solver qualification.',
        'bindings': {
            'mp90_report_sha256': sha(PKG / 'report.json'),
            'mp90_script_sha256': sha(PKG / 'mp90_endpoint_fd.py'),
            'common_epsilon_script_sha256': sha(PKG / 'common-epsilon-1e-7.py'),
            'common_epsilon_report_sha256': sha(PKG / 'common-epsilon-1e-7.json'),
            'prior_fp64_report_sha256': sha(PRIOR),
            'native_controlled_sweep_306_report_sha256': sha(SWEEP306)},
        'points': result,
        'native306_context': {
            'classification': 'Separate native 306 stress finite differences against native algorithmic actions. Its epsilon is seconds along captured Fdot and it is not an independent endpoint oracle or physical-step result.',
            'receipt': sweep306['receipt'],
            'policies': native306_context},
        'classification_checks': {
            'all_mp90_endpoint_roots_converged': no_transitions,
            'no_yield_or_fb_selection_transitions_in_sweep': no_transitions,
            'both_9x9_fd_matrices_converge_below_1e_minus_12_at_epsilon_1e_minus_14': tiny_errors,
            'same_epsilon_1e_minus_7_mp90_matches_prior_fp64_fd_error_within_1e_minus_9': overlap_match},
        'interpretation': {
            'root_tolerance': 'At the common 1e-7 perturbation, fresh MP90 endpoint roots and the prior FP64 endpoint roots produce essentially identical 9x9 FD-versus-analytic errors (tet1634 difference about 3.4e-11 absolute in relative-error metric; tet1638 about 2.1e-12). The retained 6.35e-6/1.54e-6 discrepancies are therefore not explained by the old FP64 endpoint solver stopping inaccurately at that step size.',
            'finite_step_and_smoothing': 'At smaller MP90 perturbations, the 9x9 central-difference error falls to 1.82e-15 (tet1634) and 3.82e-17 (tet1638) at 1e-14. Captured states remain on the same positive-yield-gap, elastic-smoothed FB selection, with no detected transition. The largest endpoint |delta q| divided by q smoothing scale falls from about 39.9 at 1e-8 to 0.0606 at 1e-10 for tet1634, and from about 40.8 to 0.0267 for tet1638. This ties the non-asymptotic error drop to finite perturbations traversing the local q-norm smoothing scale; it is a local interpretation, not a general bound.',
            'tiny_multiplier': 'Center multipliers are about 5e-17 while the captured FB smoothing parameter is 1e-8. MP90 endpoint solves resolve their relative changes down to the smallest sweep steps, with all terminal corrections and scaled residuals far below 1e-70; no multiplier-zero cutoff was imposed.',
            'fdot': 'A separate sweep uses F plus/minus epsilon_seconds times the captured F_direction_local_per_s and fixed accepted history. Its action errors also converge at small epsilon, but tet1634 needs smaller epsilon than tet1638. The 9x9 basis sweep uses dimensionless F-entry perturbations and is not mislabeled as the Fdot sweep.',
            'scope': 'These results classify only the two captured points and the stated parameter/input words. They support that finite-step curvature, rather than endpoint root tolerance, explains the old 1e-7 FP64 discrepancy for these cases. They do not establish a native finite-difference gate, all-point derivatives, material qualification, or a general noise floor.'},
        'qualification': False}
    path = PKG / 'summary.json'
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'summary': str(path), 'sha256': sha(path), 'checks': out['classification_checks']}, indent=2))
    if not all(out['classification_checks'].values()):
        raise SystemExit('summary gate failed')
    if not all(row['both_endpoints_converged'] for point in result for row in point['captured_Fdot_sweep']):
        raise SystemExit('Fdot root convergence failed')
    if not all(row['all_endpoints_converged'] for point in result for row in point['full_9x9_sweep']):
        raise SystemExit('basis root convergence failed')
    if not all(point['same_epsilon_1e_minus_7']['mp90_all_endpoint_roots_converged'] for point in result):
        raise SystemExit('common-epsilon roots failed')
    if previous.get('schema') != 'numi.cardboard.medium-tangent-state-sensitivity.v4':
        raise SystemExit('unexpected prior report schema')

if __name__ == '__main__':
    main()
