#!/usr/bin/env python3
"""Strict offline audit of source-115 full-contact runs 383/385/386."""
from __future__ import annotations
import hashlib, json, math, pathlib, struct, sys

BASE = pathlib.Path('/Users/home/cardboard-implementation-20261014')
NATIVE = BASE / 'native'
RUNS = {
    '383': NATIVE / 'deterministic-private-ccd-full-contact-383',
    '385': NATIVE / 'full-contact-newton9-center-capture-385',
    '386': NATIVE / 'full-contact-capture-off-parity-386',
}
SOURCE = BASE / 'snapshots/source-private-ccd-fracture-115'
OUT = BASE / 'analysis/full-contact-newton9-residual-audit-385-001'

class AuditError(RuntimeError): pass

def fail_constant(value: str):
    raise AuditError(f'nonstandard JSON constant {value}')

def load_json(path: pathlib.Path):
    return json.loads(path.read_text(), parse_constant=fail_constant)

def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha_file(path: pathlib.Path) -> str:
    return sha_bytes(path.read_bytes())

def f32(x: float) -> float:
    return struct.unpack('<f', struct.pack('<f', x))[0]

def f32_ulp(x: float) -> float:
    """Spacing to the next representable FP32 value toward +infinity."""
    x = f32(x)
    if x == 0.0:
        return 2.0 ** -149
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    next_bits = bits - 1 if bits & 0x80000000 else bits + 1
    next_value = struct.unpack('<f', struct.pack('<I', next_bits))[0]
    return abs(float(next_value) - float(x))

def read_capture(run: pathlib.Path):
    manifest_path = run / 'center-capture.coupled.json'
    meta = load_json(manifest_path)
    payload_path = manifest_path.parent / pathlib.Path(meta['payload']['file']).name
    payload = payload_path.read_bytes()
    if len(payload) != meta['payload']['byte_count']:
        raise AuditError(f'{run.name}: payload byte count mismatch')
    if sha_bytes(payload) != meta['payload']['sha256']:
        raise AuditError(f'{run.name}: payload SHA mismatch')
    arrays = meta['arrays']
    if len(arrays) != meta['array_count'] or len(arrays) > meta['max_arrays']:
        raise AuditError(f'{run.name}: array count mismatch/cap exceeded')
    names = [a['name'] for a in arrays]
    if len(set(names)) != len(names):
        raise AuditError(f'{run.name}: duplicate array names')
    ordered = sorted(arrays, key=lambda a: a['offset'])
    cursor = 0
    by_name = {}
    for a in ordered:
        size = a['count'] * a['element_bytes']
        if size != a['byte_count'] or a['offset'] != cursor:
            raise AuditError(f"{run.name}: invalid extent/coverage for {a['name']}")
        block = payload[cursor:cursor + size]
        if sha_bytes(block) != a['sha256']:
            raise AuditError(f"{run.name}: array hash mismatch for {a['name']}")
        by_name[a['name']] = (a, block)
        cursor += size
    if cursor != len(payload) or cursor > meta['max_payload_bytes']:
        raise AuditError(f'{run.name}: payload coverage/cap mismatch')
    return meta, payload, by_name, manifest_path, payload_path

def unpack_vecs(by_name, name, fmt, width):
    a, blob = by_name[name]
    if a['dtype'] != fmt or a['element_bytes'] != width * 4:
        raise AuditError(f'{name}: unexpected dtype or stride')
    lane_type = 'I' if fmt.startswith('uint') else 'f'
    return [struct.unpack_from('<' + lane_type * width, blob, i * width * 4)
            for i in range(a['count'])]

def armijo_rows(run: pathlib.Path):
    out = []
    for line in (run / 'stderr.log').read_text().splitlines():
        if line.startswith('fem_residual_armijo_iteration='):
            out.append(json.loads(line.split('=', 1)[1], parse_constant=fail_constant))
    return out

def state_files(run: pathlib.Path):
    root = run / 'diagnostic-state'
    result = {}
    for p in sorted(root.rglob('*')):
        if p.is_file():
            result[str(p.relative_to(root))] = p
    return result

def filtered_capture_log(path: pathlib.Path):
    prefixes = ('fem_coupled_capture_enabled=', 'fem_coupled_capture_complete=',
                'fem_coupled_capture_skipped=')
    return [line for line in path.read_text().splitlines()
            if not line.startswith(prefixes)]

def compare_runs(a_key: str, b_key: str, capture_diagnostics: dict):
    a, b = RUNS[a_key], RUNS[b_key]
    af, bf = state_files(a), state_files(b)
    if af.keys() != bf.keys():
        raise AuditError(f'{a_key}/{b_key}: diagnostic-state path set differs')
    exact = []
    normalized = []
    for rel in sorted(af):
        if rel == 'dense/fixture-free-body-control-1201.log':
            ok = filtered_capture_log(af[rel]) == filtered_capture_log(bf[rel])
            if not ok: raise AuditError(f'{a_key}/{b_key}: non-capture control-1201 log differs')
            normalized.append(rel)
        else:
            if af[rel].read_bytes() != bf[rel].read_bytes():
                raise AuditError(f'{a_key}/{b_key}: state file differs: {rel}')
            exact.append(rel)
    rows_a, rows_b = armijo_rows(a), armijo_rows(b)
    if rows_a != rows_b or len(rows_a) != 32:
        raise AuditError(f'{a_key}/{b_key}: Armijo trace mismatch/count')
    if (a / 'stdout.log').read_bytes() != (b / 'stdout.log').read_bytes():
        raise AuditError(f'{a_key}/{b_key}: stdout differs')
    # Stderr may differ only in capture lifecycle notices; keep every other line exact.
    sa = [x for x in (a / 'stderr.log').read_text().splitlines()
          if not x.startswith(('fem_coupled_capture_enabled=',
                               'fem_coupled_capture_complete=',
                               'fem_coupled_capture_skipped='))]
    sb = [x for x in (b / 'stderr.log').read_text().splitlines()
          if not x.startswith(('fem_coupled_capture_enabled=',
                               'fem_coupled_capture_complete=',
                               'fem_coupled_capture_skipped='))]
    if sa != sb:
        raise AuditError(f'{a_key}/{b_key}: non-capture stderr differs')
    return {
        'runs': [a_key, b_key],
        'result': 'physical_diagnostic_history_equal',
        'transaction_state_files_byte_identical': exact,
        'capture_lifecycle_log_only_normalized': normalized,
        'armijo_records': len(rows_a),
        'armijo_records_exact': True,
        'stdout_exact': True,
        'stderr_exact_after_capture_lifecycle_lines': True,
    }

def main():
    run_meta = {}
    for k, p in RUNS.items():
        receipt = load_json(p / 'receipt.json')
        pred = load_json(p / 'prediction.json')
        if receipt['exit_code'] != 1 or receipt['timed_out'] or not receipt['source_inputs_unchanged'] or receipt['changed_inputs']:
            raise AuditError(f'{k}: unexpected receipt')
        run_meta[k] = (receipt, pred)
    # Exact source/build/input identity and only the requested capture selector/path variation.
    base_pred = run_meta['383'][1]
    input_sets = {k: pred['inputs_sha256'] for k, (_, pred) in run_meta.items()}
    if not (input_sets['383'] == input_sets['385'] == input_sets['386']):
        raise AuditError('source/build/external input SHA maps differ')
    argv_norm = {}
    env_norm = {}
    for k, (_, pred) in run_meta.items():
        argv = list(pred['argv'])
        i = argv.index('--diagnostic-state-output')
        argv[i + 1] = '<run-specific-diagnostic-state-output>'
        argv_norm[k] = argv
        env = dict(pred.get('environment', {}))
        env.pop('NM_FEM_COUPLED_CAPTURE_SELECTOR', None)
        env.pop('NM_FEM_COUPLED_CAPTURE_PATH', None)
        env_norm[k] = env
    if not (argv_norm['383'] == argv_norm['385'] == argv_norm['386']):
        raise AuditError('physical argv differ beyond output directory')
    if not (env_norm['383'] == env_norm['385'] == env_norm['386']):
        raise AuditError('environment differs beyond capture selector/path')
    if not (run_meta['383'][1]['source_snapshot'] == run_meta['385'][1]['source_snapshot'] == run_meta['386'][1]['source_snapshot']):
        raise AuditError('source snapshots differ')
    if not (run_meta['383'][1]['build_provenance'] == run_meta['385'][1]['build_provenance'] == run_meta['386'][1]['build_provenance']):
        raise AuditError('build provenance differs')

    # Capture is selected from the N9 center only; N7 is kept separately for prior evidence.
    cap, payload, arr, manifest_path, payload_path = read_capture(RUNS['385'])
    if cap['root'] != 1201 or cap['iteration'] != 9 or cap['capture_point'] != 'after_fgmres_measure_and_human_direction_map_before_armijo_trial':
        raise AuditError('385 capture root/iteration/phase mismatch')
    if not cap['rhs_preserved_before_fgmres'] or cap['center_operator_action']['cache_used']:
        raise AuditError('385 capture RHS or operator-generation marker mismatch')
    if cap['environment_count'] != 1 or cap['node_count'] != 36 or cap['tetrahedron_count'] != 48 or cap['rigid_generalized_capacity'] != 6 or cap['fgmres_layout']['unknown_count'] != 78:
        raise AuditError('385 capture dispatch scope mismatch')
    if cap['deformable_contact_capture']['truncated'] or not cap['deformable_contact_capture']['complete']:
        raise AuditError('385 deformable contact arena capture incomplete')

    rowmap = unpack_vecs(arr, 'fgmres_vector_row_map', 'uint32x4', 4)
    rhs = unpack_vecs(arr, 'full_rhs_before_fgmres', 'float32x4', 4)
    action = unpack_vecs(arr, 'center_production_jdelta_action', 'float32x4', 4)
    correction = unpack_vecs(arr, 'fem_solution_correction', 'float32x4', 4)
    after_workspace = unpack_vecs(arr, 'fem_rhs_after_fgmres', 'float32x4', 4)
    if not (len(rowmap) == len(rhs) == len(action) == len(correction) == 78):
        raise AuditError('linear vector sizes do not match the exact row map')
    expected_blocks = {1: [], 2: [], 3: []}
    for slot, row in enumerate(rowmap):
        env, block, local, src = row
        if env != 0 or block not in expected_blocks:
            raise AuditError(f'invalid row map entry at slot {slot}: {row}')
        expected_blocks[block].append((slot, local, src))
    if len(expected_blocks[1]) != 36 or len(expected_blocks[2]) != 36 or len(expected_blocks[3]) != 6:
        raise AuditError('row map block counts differ from FEM/field/rigid layout')
    if [x[1] for x in expected_blocks[1]] != list(range(36)) or [x[1] for x in expected_blocks[2]] != list(range(36)) or [x[1] for x in expected_blocks[3]] != list(range(6)):
        raise AuditError('row map local slots are not canonical')
    if any(x[2] != 0xFFFFFFFF for b in (1, 2) for x in expected_blocks[b]) or any(x[2] != 1 for x in expected_blocks[3]):
        raise AuditError('row map source proxy ids are inconsistent')

    # Promote exact captured FP32 operands and compute the same-generation linear equation residual.
    rows = []
    for i, (b, a) in enumerate(zip(rhs, action)):
        for lane in range(4):
            rows.append({'slot': i, 'lane': lane, 'block': rowmap[i][1],
                         'local_index': rowmap[i][2], 'rhs': float(b[lane]),
                         'jdelta': float(a[lane]), 'linear_residual': float(b[lane]) - float(a[lane])})
    valid = [x for x in rows if x['block'] != 3 or x['lane'] == 0]
    def l2(items, field): return math.sqrt(math.fsum(x[field] * x[field] for x in items))
    def block_metrics(block):
        vals = [x for x in valid if x['block'] == block]
        return {'stored_float4_slots': len(expected_blocks[block]),
                'scalar_lanes_in_norm': len(vals),
                'rhs_l2': l2(vals, 'rhs'), 'jdelta_l2': l2(vals, 'jdelta'),
                'b_minus_jdelta_l2': l2(vals, 'linear_residual'),
                'max_abs_b_minus_jdelta': max((abs(x['linear_residual']) for x in vals), default=0.0)}
    total = {'rhs_l2': l2(valid, 'rhs'), 'jdelta_l2': l2(valid, 'jdelta'),
             'b_minus_jdelta_l2': l2(valid, 'linear_residual'),
             'relative_linear_residual': l2(valid, 'linear_residual') / l2(valid, 'rhs'),
             'max_abs_b_minus_jdelta': max(abs(x['linear_residual']) for x in valid)}
    # Compare to the recorded Arnoldi residual estimate; the operator action is native production output.
    _, fg_blob = arr['fgmres_state']
    fg = struct.unpack('<8f', fg_blob)
    total['fgmres_state'] = {'diagnostics_current_residual_estimate': fg[0],
                             'diagnostics_initial_residual_norm': fg[1],
                             'converged_latch': fg[2], 'arnoldi_column_field': fg[3],
                             'nonlinear_first_residual': fg[4], 'nonlinear_current_residual': fg[5],
                             'last_correction_ratio': fg[6], 'nonlinear_converged_latch': fg[7]}
    total['captured_vs_recomputed_norm_delta'] = {
        'linear_residual_absolute': abs(total['b_minus_jdelta_l2'] - fg[0]),
        'linear_residual_relative_to_recomputed': abs(total['b_minus_jdelta_l2'] - fg[0]) / total['b_minus_jdelta_l2'],
        'initial_rhs_absolute': abs(total['rhs_l2'] - fg[1]),
    }
    total['block_metrics'] = {str(b): block_metrics(b) for b in [1, 2, 3]}
    total['after_fgmres_workspace_max_abs'] = max(abs(x) for v in after_workspace for x in v)
    total['after_workspace_is_not_used_as_rhs'] = True
    total['padding_max_abs'] = max(abs(v[lane]) for i, v in enumerate(rhs) for lane in range(4)
                                  if rowmap[i][1] == 3 and lane > 0)
    total['status_before_armijo_code'] = struct.unpack_from('<I', arr['solver_statuses'][1], 0)[0]
    total['status_before_armijo_fgmres_iterations'] = struct.unpack_from('<I', arr['solver_statuses'][1], 20)[0]
    total['center_operator_diagnostic_code'] = struct.unpack_from('<I', arr['center_operator_diagnostic_status'][1], 0)[0]
    total['same_generation_operator_cache_used'] = cap['center_operator_action']['cache_used']

    # Reconcile the Arnoldi forcing target with the distinct absolute
    # nonlinear-target early exit in the production cycle finalizer.
    relative_tolerance = 1.0e-7  # authored by this fixture in source at line 808
    diagonal_floor = 1.0e-8       # MixedSolverSource default in matter.hpp
    armijo_coefficient = 1.0e-4  # MixedSolverSource default in matter.hpp
    newton_index = cap['iteration']
    scheduled_forcing = 0.25 * (2.0 ** -min(newton_index, 4))
    forcing_floor = math.sqrt(relative_tolerance)
    effective_forcing = min(max(max(forcing_floor, scheduled_forcing),
                                relative_tolerance), 0.25)
    first_newton_norm = fg[4]
    nonlinear_threshold = max(
        diagonal_floor,
        relative_tolerance * max(first_newton_norm, 1.0))
    linear_forcing_target = effective_forcing * total['rhs_l2']
    last_n9 = next(x for x in armijo_rows(RUNS['385'])
                   if x['newton_iteration'] == newton_index)['environments'][0]
    last_alpha = float(last_n9['trial_alpha'])
    base_norm = float(last_n9['base_l2'])
    last_trial_norm = float(last_n9['trial_l2'])
    # Model the actual float operations in the final Armijo bound. The source
    # additionally requires strict trial < base, even if the bound rounds to
    # base. This is evidence about the last trial only, not earlier trials.
    armijo_factor_f32 = f32(1.0 - f32(armijo_coefficient * last_alpha))
    armijo_bound_f32 = f32(armijo_factor_f32 * base_norm)
    total['linear_and_nonlinear_gate_analysis'] = {
        'authored_relative_residual_tolerance': relative_tolerance,
        'authored_diagonal_floor': diagonal_floor,
        'authored_armijo_coefficient': armijo_coefficient,
        'newton_iteration_zero_based': newton_index,
        'scheduled_forcing': scheduled_forcing,
        'sqrt_tolerance_forcing_floor': forcing_floor,
        'effective_fgmres_relative_forcing': effective_forcing,
        'initial_linear_rhs_norm': total['rhs_l2'],
        'relative_forcing_absolute_target': linear_forcing_target,
        'captured_fp64_b_minus_jdelta_norm': total['b_minus_jdelta_l2'],
        'captured_native_arnoldi_estimate': fg[0],
        'arnoldi_estimate_under_outer_absolute_target': fg[0] <= nonlinear_threshold,
        'arnoldi_estimate_under_relative_forcing_target': fg[0] <= linear_forcing_target,
        'outer_absolute_nonlinear_target': nonlinear_threshold,
        'outer_target_formula': 'max(diagonalFloor, relativeResidual * max(firstNewtonNorm, 1))',
        'outer_target_first_newton_norm': first_newton_norm,
        'cycle_finalizer_reason': 'reachedNonlinearTarget accepts this FGMRES cycle because estimate <= 1e-7 absolute target, although it is above the 0.015625 * ||b|| forcing target; nonlinear_converged_latch remains false, so outer Armijo still runs.',
        'last_n9_armijo': {
            'base_norm': base_norm,
            'last_tested_alpha': last_alpha,
            'trial_norm': last_trial_norm,
            'raw_fp64_armijo_decrement': base_norm * armijo_coefficient * last_alpha,
            'fp32_armijo_bound': armijo_bound_f32,
            'fp32_ulp_at_base_norm': f32_ulp(base_norm),
            'trial_minus_base_norm': last_trial_norm - base_norm,
            'strict_descent_required': True,
            'note': 'Only the final tested trial is in this record; the earlier 23 trial values are not captured, so this does not explain every rejection.'
        },
        'metric_units_and_scope': 'The residual norm combines 36 FEM float4 rows and six rigid generalized scalar rows; field rows are zero. The explicit mechanical entries are impulse residuals, while rigid rotational entries are angular impulses. No mass/stiffness/contact term decomposition was captured, so this norm cannot attribute the remainder to a particular block.'
    }

    # Check the physical-state kinematics used by the source FEM apply path.
    def node_records(name):
        a, blob = arr[name]
        if a['dtype'] != 'record:NMFEMNodeStateGPU' or a['element_bytes'] != 80 or a['count'] != 36:
            raise AuditError(f'{name}: bad node ABI')
        return [struct.unpack_from('<20f', blob, i * 80) for i in range(36)]
    accepted, candidate = node_records('fem_accepted_nodes'), node_records('fem_candidate_nodes')
    dt = f32(cap['dt_seconds'])
    fixed_nodes = [i for i, n in enumerate(candidate) if n[11] > 0.5]
    modes = sorted(set(n[19] for n in candidate))
    position_exact = delta_velocity_exact = 0
    u_max_error = 0.0
    position_comps = delta_comps = 0
    for old, cur in zip(accepted, candidate):
        for j in range(3):
            dv = f32(cur[4+j] - old[4+j])
            delta_velocity_exact += dv == cur[12+j]
            delta_comps += 1
            if cur[19] == 1.0:
                pos = f32(cur[8+j] + cur[16+j])
                u_ref = f32(old[16+j] + f32(dt * cur[4+j]))
                u_max_error = max(u_max_error, abs(u_ref - cur[16+j]))
            else:
                pos = f32(old[j] + f32(dt * cur[4+j]))
            position_exact += pos == cur[j]
            position_comps += 1
    alpha_last = next(x for x in armijo_rows(RUNS['385']) if x['newton_iteration'] == 9)['environments'][0]['trial_alpha']
    max_delta_v = max(abs(correction[i][lane]) for i in range(36) for lane in range(3))
    total['kinematic_relation'] = {
        'all_candidate_reference_displacement_modes': modes,
        'fixed_fem_nodes': fixed_nodes,
        'fem_candidate_position_relation_exact_components': position_exact,
        'fem_candidate_position_components': position_comps,
        'fem_delta_velocity_relation_exact_components': delta_velocity_exact,
        'fem_delta_velocity_components': delta_comps,
        'reference_displacement_update_max_abs_error_under_separate_fp32_ops': u_max_error,
        'dt_seconds_f32': dt,
        'max_abs_newton_velocity_correction_m_per_s': max_delta_v,
        'ninth_armijo_last_tested_alpha': alpha_last,
        'max_abs_alpha_dt_delta_v_m': abs(dt * alpha_last) * max_delta_v,
        'alpha_position_effect_is_derived_not_captured_trial_geometry': True,
        'source_apply_semantics': 'FEM apply adds alpha*delta_v; mode 1 rebuilds u from accepted u + dt*v then position=rest+u; this is a scale estimate, not an observed private-trial candidate.'
    }
    # A separate-FP32 algebra replay estimates geometry resolution at the last
    # alpha. FMA/contraction and private trial execution are not captured here.
    changed_positions = 0
    free_position_components = 0
    maximum_position_change = 0.0
    for node_index, (old, cur) in enumerate(zip(accepted, candidate)):
        if cur[11] > 0.5:
            continue
        for axis in range(3):
            free_position_components += 1
            next_velocity = f32(cur[4 + axis] + f32(alpha_last * correction[node_index][axis]))
            next_u = f32(old[16 + axis] + f32(dt * next_velocity))
            next_x = f32(cur[8 + axis] + next_u)
            delta_x = float(next_x) - float(cur[axis])
            changed_positions += delta_x != 0.0
            maximum_position_change = max(maximum_position_change, abs(delta_x))
    total['kinematic_relation']['last_alpha_separate_fp32_replay'] = {
        'free_position_components': free_position_components,
        'components_changed_vs_center': changed_positions,
        'components_unchanged_vs_center': free_position_components - changed_positions,
        'max_abs_position_change_m': maximum_position_change,
        'status': 'modeled separate-FP32 update only; not a captured trial state and not evidence that geometry quantization caused the failure.'
    }

    # Preserve the observed N9 outcome and distinguish later fatal-scratch rows.
    ar = armijo_rows(RUNS['385'])
    def armijo_summary(n):
        e = ar[n]['environments'][0]
        return {k:e.get(k) for k in ('base_l2','base_health','base_residual_interpretation','trial_l2','trial_alpha','accepted','trial_outcome','solver_status_code')}
    total['armijo_newton8'] = armijo_summary(8)
    total['armijo_newton9'] = armijo_summary(9)
    total['armijo_newton10'] = armijo_summary(10)
    total['nonlinear_result'] = 'Newton 9 rejects every line-search candidate; Newton 10+ rows are blocked after fatal status and their raw base_l2 is not an assembled residual.'

    # Compare contact history lineages without conflating accepted timestep state and working history.
    history_names = ['contact_histories_accepted','contact_histories_candidate','contact_history_used_by_newton_center',
                     'deformable_contact_histories_accepted','deformable_contact_histories_candidate','deformable_history_used_by_newton_center',
                     'material_history_accepted','material_history_candidate','cohesive_faces_accepted','cohesive_faces_candidate']
    total['history_arrays'] = {n:{'sha256':arr[n][0]['sha256'],'count':arr[n][0]['count'],
                                   'dtype':arr[n][0]['dtype']} for n in history_names}
    total['working_contact_history_matches_prior_candidate'] = (
        arr['contact_history_used_by_newton_center'][0]['sha256'] == arr['contact_histories_candidate'][0]['sha256'] and
        arr['deformable_history_used_by_newton_center'][0]['sha256'] == arr['deformable_contact_histories_candidate'][0]['sha256'])
    total['working_contact_history_differs_from_timestep_accepted'] = (
        arr['contact_history_used_by_newton_center'][0]['sha256'] != arr['contact_histories_accepted'][0]['sha256'] and
        arr['deformable_history_used_by_newton_center'][0]['sha256'] != arr['deformable_contact_histories_accepted'][0]['sha256'])

    pairwise = [compare_runs('383','385',{}), compare_runs('385','386',{}), compare_runs('383','386',{})]
    manifests = {}
    for k, (rec, pred) in run_meta.items():
        p = RUNS[k]
        manifests[k] = {
            'receipt_sha256': sha_file(p/'receipt.json'), 'prediction_sha256': sha_file(p/'prediction.json'),
            'source_snapshot': pred['source_snapshot'],
            'source_build_sha256': pred['inputs_sha256']['external/source-build.json'],
            'executable_sha256': pred['inputs_sha256']['runtime/matter/numi-matter-cardboard-full-contact-armijo-check'],
            'input_file_count': len(pred['inputs_sha256']),
            'canonical_input_map_sha256': sha_bytes(json.dumps(pred['inputs_sha256'], sort_keys=True, separators=(',',':')).encode()),
            'exit_code': rec['exit_code'], 'timed_out': rec['timed_out'], 'wall_seconds': rec['wall_seconds'],
            'armijo_records': 32,
        }
    output = {
        'schema':'numi.cardboard.full_contact_newton9_residual_audit.v1',
        'verdict':'OFF_ON_PHYSICAL_HISTORY_PARITY_PASS; SAME_GENERATION_LINEAR_RESIDUAL_MEASURED; FGMRES_ACCEPTED_BY_OUTER_ABSOLUTE_TARGET; FULL_CONTACT_N9_ARMIJO_FAILURE_RETAINED',
        'source_binding': {'source_snapshot': str(SOURCE),
                           'source_file_sha256': {rel:sha_file(SOURCE/rel) for rel in [
                               'matter/src/runtime.mm','matter/src/metal/fem.metalinc','matter/src/metal/fgmres.metalinc','matter/include/numi/matter/shared.h',
                               'matter/include/numi/matter/matter.hpp','matter/tools/cardboard_full_contact_armijo_check.mm']},
                           'source_lines': {'row_map':'runtime.mm:15337-15353','capture arrays':'runtime.mm:15541-15549',
                                            'vector sign semantics':'runtime.mm:16531-16536','FEM apply':'fem.metalinc:3282-3335',
                                            'rigid apply':'fgmres.metalinc:4826-4840','node ABI':'shared.h:1111-1118',
                                            'status/FGMRES ABI':'shared.h:1171-1178;1390-1403',
                                            'forcing construction':'runtime.mm:17974-17983',
                                            'FGMRES cycle gates':'fgmres.metalinc:5823-5856',
                                            'Armijo acceptance gate':'fem.metalinc:4472-4555',
                                            'fixture tolerance':'cardboard_full_contact_armijo_check.mm:808',
                                            'solver defaults':'matter.hpp:415-428'}},
        'run_bindings': manifests,
        'physical_inputs_identical_after_output_path_normalization': True,
        'capture_env_only_variation': True,
        'capture_383': {'manifest_sha256':sha_file(RUNS['383']/'center-capture.coupled.json'),
                        'payload_sha256':sha_file(RUNS['383']/'center-capture.coupled.bin'),
                        'root':1201,'iteration':7,'array_count':145,'payload_bytes':11670948},
        'capture_385': {'manifest_sha256':sha_file(manifest_path),'payload_sha256':sha_file(payload_path),
                        'source_physics_fingerprint':cap['source_physics_fingerprint'],
                        'device_program_fingerprint':cap['device_program_fingerprint'],
                        'root':cap['root'],'iteration':cap['iteration'],'array_count':cap['array_count'],
                        'payload_bytes':len(payload),'all_array_hashes_verified':True,
                        'payload_fully_covered_by_contiguous_arrays':True,
                        'rhs_preserved_before_fgmres':cap['rhs_preserved_before_fgmres'],
                        'center_operator_diagnostic_code':total['center_operator_diagnostic_code']},
        'capture_386':'not requested; physical no-capture control',
        'physical_history_parity':pairwise,
        'same_generation_linear_residual':total,
        'limits':[
            'The linear residual is recomputed in FP64 from exact captured FP32 RHS and same-generation production J*delta operands; it is not an independent reassembly or nonlinear certificate.',
            'Capture is before the N9 Armijo trials. The final N9 trial alpha comes from the Armijo log; trial geometry is not captured, so displacement representability is an estimate only.',
            'All three complete native runs exit with expected fixture failure (exit 1). The repeated history does not qualify a completed or converged full-contact step.',
            'N10+ raw base_l2 is explicitly marked invalid after fatal solver status and is not interpreted as a residual.'
        ]
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'report.json').write_text(json.dumps(output,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'report':str(OUT/'report.json'),'verdict':output['verdict'],
                      'linear':{k:total[k] for k in ('rhs_l2','jdelta_l2','b_minus_jdelta_l2','relative_linear_residual')},
                      'pairwise':pairwise},indent=2))

if __name__ == '__main__':
    main()
