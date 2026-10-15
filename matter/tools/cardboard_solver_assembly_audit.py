#!/usr/bin/env python3
"""Audit captured bulk FEM transformations and gather using a source-bound FP64 replay.

This is a diagnostic of the selected operator capture, not the complete contact
operator or a numerical-convergence certificate. No solver step is executed.
"""
from pathlib import Path
import argparse
import json
import subprocess
import numpy as np
import cardboard_solver_pair_analyzer as pair
from cardboard_solver_capture_analysis import normalize_native, rotation, force

SCHEMA = 'numi.cardboard.solver-assembly-audit.v1'


def metric(actual, expected):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape:
        raise ValueError('comparison shape differs')
    return pair.metric(a.ravel().tolist(), b.ravel().tolist())


def matrix(values):
    a = np.asarray(values, dtype=float)
    if a.size != 9 or not np.isfinite(a).all():
        raise ValueError('invalid matrix input')
    return a.reshape(3, 3)


def starch_reference(tet, header):
    descriptor = next((m for m in header['materials']
                       if int(m['index']) == int(tet['material_index'])), None)
    if descriptor is None or descriptor['name'] != 'starch2007_finite_glue_elastic':
        raise ValueError('unsupported non-paper material in assembled capture')
    if descriptor['parameter_names'] != ['density', 'mu', 'lambda']:
        raise ValueError('starch parameter names/order differ')
    offset = int(descriptor['parameter_offset'])
    values = header['parameters_by_environment'][tet['environment']][offset:offset+3]
    if len(values) != 3 or min(values) <= 0:
        raise ValueError('invalid captured starch parameters')
    _, mu, lam = values
    f = matrix(tet['center']['F_local'])
    h = matrix(tet['center']['local_F_direction_per_unit_velocity'])
    determinant = np.linalg.det(f)
    if determinant <= 0:
        raise ValueError('nonpositive captured starch determinant')
    inv = np.linalg.inv(f)
    a = lam*np.log(determinant)-mu
    stress = mu*f + a*inv.T
    tangent = mu*h + lam*np.trace(inv@h)*inv.T-a*inv.T@h.T@inv.T
    return stress, tangent


def _capture_matrix(state, high_name, low_name):
    """Rebuild a native high/low FP32 pair after widening both words."""
    high_name_actual = high_name + '_high'
    high = state.get(high_name_actual, state.get(high_name))
    low = state.get(low_name, [0.0] * 9)
    return (np.asarray(high, dtype=np.float64).reshape(3, 3) +
            np.asarray(low, dtype=np.float64).reshape(3, 3))


def _gradient_from_nodal_vectors(values, inverse_rest):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (4, 3):
        raise ValueError('realized tetrahedron ray must have four 3-vectors')
    return (values[1:] - values[0]).T @ inverse_rest


def reconstruct_realized_pair(header, node_by_id, tet, environment, epsilon_index):
    """Recover actual FP32 side rays, separate from the requested Newton ray.

    The captured two-word F is authoritative for the constitutive input. The
    node velocity differences are independently reconstructed through the
    tetrahedron's reference gradient. A disagreement between them is reported
    as kinematic/representation error; it is never silently folded into a
    constitutive tangent result.
    """
    epsilons = header.get('epsilon_values', [])
    if not 0 <= epsilon_index < len(epsilons):
        raise ValueError('realized ray epsilon index is out of range')
    epsilon = float(epsilons[epsilon_index])
    dt = float(header['dt_seconds_by_environment'][environment])
    if not np.isfinite(epsilon) or epsilon <= 0 or not np.isfinite(dt) or dt <= 0:
        raise ValueError('invalid realized ray epsilon or timestep')
    ids = tet.get('node_ids', tet.get('nodes'))
    if not isinstance(ids, list) or len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError('invalid tetrahedron identity in realized ray')
    try:
        local_nodes = [node_by_id[(environment, int(i))] for i in ids]
    except KeyError as error:
        raise ValueError('realized ray references a missing node') from error
    inverse_rest = np.asarray(tet['inverse_rest_rows'], dtype=np.float64).reshape(3, 3)
    q = rotation(tet.get('frame_xyzw', tet.get('material_frame_quaternion_xyzw')))
    center = tet['center']
    f_center = _capture_matrix(center, 'F_local', 'F_local_low')
    intended_h = np.asarray(center.get('local_F_direction_per_unit_velocity',
                                        center.get('F_direction_local_per_s')),
                            dtype=np.float64).reshape(3, 3)
    side_records = tet.get('perturbations', tet.get('sides'))
    first_side = 2 * epsilon_index
    if not isinstance(side_records, list) or len(side_records) < first_side + 2:
        raise ValueError('realized ray requires captured plus and minus side states')
    side_data = []
    for side_offset, sign in enumerate((1.0, -1.0)):
        side_index = first_side + side_offset
        side = side_records[side_index]
        f_side = _capture_matrix(side, 'F_local', 'F_local_low')
        h_f = (f_side - f_center) / (sign * epsilon * dt)
        delta_v = np.asarray([
            node['epsilon_pair_realized_velocity_delta_m_s'][side_index]
            for node in local_nodes], dtype=np.float64)
        h_v_world = _gradient_from_nodal_vectors(delta_v, inverse_rest) / (sign * epsilon)
        # Matter rotates only the reference/material coordinate on the right:
        # F_local = F_world * Q, hence H_local = H_world * Q.
        h_v_local = h_v_world @ q
        side_node_u = np.asarray([
            node['epsilon_pair_realized_u_delta_m'][side_index]
            for node in local_nodes], dtype=np.float64)
        expected_u = dt * delta_v
        side_data.append(dict(side_index=side_index, sign=int(sign), F_local=f_side,
            H_from_F_local=h_f, H_from_velocity_local=h_v_local,
            delta_v=delta_v, delta_u=side_node_u, expected_delta_u=expected_u,
            u_delta_error=side_node_u - expected_u))
    plus, minus = side_data
    centered_delta_v = (plus['delta_v'] - minus['delta_v']) / (2.0 * epsilon)
    h_velocity_center = _gradient_from_nodal_vectors(centered_delta_v, inverse_rest) @ q
    h_f_center = (plus['F_local'] - minus['F_local']) / (2.0 * epsilon * dt)
    center_u = np.asarray([node['center']['reference_u_m'] for node in local_nodes], dtype=np.float64)
    ulp = np.abs(np.spacing(center_u.astype(np.float32)).astype(np.float64))
    actual_du = np.concatenate([plus['delta_u'].ravel(), minus['delta_u'].ravel()])
    requested_du = np.concatenate([plus['expected_delta_u'].ravel(), minus['expected_delta_u'].ravel()])
    actual_ulp = np.concatenate([
        np.divide(np.abs(plus['delta_u']).ravel(), ulp.ravel(),
                  out=np.zeros(12), where=ulp.ravel() > 0),
        np.divide(np.abs(minus['delta_u']).ravel(), ulp.ravel(),
                  out=np.zeros(12), where=ulp.ravel() > 0)])
    requested_zeroed = int(np.count_nonzero((actual_du == 0.0) & (requested_du != 0.0)))
    ray_nonzero = bool(np.linalg.norm(centered_delta_v) > 0.0)
    velocity_metric = metric(h_velocity_center, intended_h)
    deformation_metric = metric(h_f_center, intended_h)
    deformation_velocity_metric = metric(h_f_center, h_velocity_center)
    ray_comparison_status = ('measured' if ray_nonzero else
                              'inconclusive_zero_realized_velocity_ray')
    # Keep the measured numeric discrepancy when the ray vanishes, but mark
    # every tangent comparison as unusable evidence: no directional derivative
    # was sampled in that case.
    for comparison in (velocity_metric, deformation_metric,
                       deformation_velocity_metric):
        comparison['comparison_status'] = ray_comparison_status
    return dict(environment=environment, tet_id=int(tet['tet_id']), epsilon=epsilon, dt=dt,
        intended_H_local=intended_h, plus=plus, minus=minus,
        centered_delta_v=centered_delta_v, H_velocity_center_local=h_velocity_center,
        H_F_center_local=h_f_center,
        centered_ray_status=('measured' if ray_nonzero else 'inconclusive_zero_realized_velocity_ray'),
        kinematic_metrics=dict(
            realized_velocity_vs_intended_H=velocity_metric,
            realized_F_vs_intended_H=deformation_metric,
            realized_F_vs_velocity_H=deformation_velocity_metric,
            max_abs_side_u_minus_dt_delta_v_m=float(np.max(np.abs(actual_du-requested_du))),
            requested_nonzero_u_components_lost=int(requested_zeroed),
            max_abs_reference_u_fp32_ulp_m=float(np.max(ulp)),
            max_abs_u_delta_in_fp32_ulps=float(np.max(actual_ulp)) if actual_ulp.size else 0.0,
            zero_ray_is_inconclusive=True))


def _realized_replay_rows(header, nodes, tets):
    descriptors = pair.material_descriptors(header)
    node_by_id = {(int(n['environment']), int(n['node_id'])): n
                  for n in nodes}
    rows, sources = [], {}
    rays = []
    for tet in tets:
        environment = int(tet['environment'])
        material_index = int(tet['material_index'])
        descriptor = descriptors.get(material_index)
        material_name = descriptor.get('name') if descriptor else None
        kind = pair.MATERIAL_KIND_BY_NAME.get(material_name)
        is_starch = material_name == 'starch2007_finite_glue_elastic'
        if kind is None and not is_starch:
            raise ValueError(f'unsupported material in realized-ray capture: {material_name}')
        parameters = pair.paper_parameter_row(header, descriptor, environment) if kind else None
        accepted = (pair.f32_array(tet['center']['accepted_state'], 13, 'accepted state')
                    if kind else None)
        for epsilon_index in range(len(header['epsilon_values'])):
            ray = reconstruct_realized_pair(header, node_by_id, tet, environment, epsilon_index)
            rays.append(ray)
            if kind is None:
                # Starch is audited through the independent analytic oracle in
                # audit_realized_environment; only its actual F/H ray is needed.
                continue
            center = tet['center']
            center_high = pair.f32_array(center.get('F_local_high', center['F_local']),
                                         9, 'captured center F high')
            center_low = pair.f32_array(center.get('F_local_low', [0.0]*9),
                                        9, 'captured center F low')
            center_state_high = pair.f32_array(center['projected_state'], 13,
                                               'captured center projected state')
            center_state_low = pair.f32_array(center.get('projected_state_low', [0.0]*13), 13,
                                              'captured center projected-state low word')
            center_hv = pair.f32_array(ray['H_velocity_center_local'].ravel().tolist(), 9,
                                       'realized center velocity-gradient ray')
            center_key = f"e{environment}_t{tet['tet_id']}_ray{epsilon_index}_center_velocity"
            center_packed = [*center_high, *center_low, *center_hv, *accepted,
                             *center_state_high, *center_state_low, *parameters]
            rows.append(' '.join([center_key, kind] +
                                 [format(float(v), '.17g') for v in center_packed]))
            sources[center_key] = (tet, ray)
            ray['center_velocity_replay_key'] = center_key
            for side_name in ('plus', 'minus'):
                side_ray = ray[side_name]
                if not np.isfinite(side_ray['H_from_F_local']).all():
                    raise ValueError('nonfinite realized deformation ray')
                if np.linalg.norm(side_ray['H_from_F_local']) == 0.0:
                    side_ray['replay_key'] = None
                    continue
                native_side = tet['perturbations'][side_ray['side_index']]
                f_high = pair.f32_array(native_side.get('F_local_high', native_side['F_local']),
                                        9, 'realized F high')
                f_low = pair.f32_array(native_side.get('F_local_low', [0.0]*9),
                                       9, 'realized F low')
                candidate_high = pair.f32_array(native_side['projected_state'], 13,
                                                'captured projected state')
                candidate_low = pair.f32_array(native_side.get('projected_state_low', [0.0]*13), 13,
                                               'captured projected-state low word')
                h = pair.f32_array(side_ray['H_from_F_local'].ravel().tolist(), 9,
                                   'realized local F ray')
                key = f"e{environment}_t{tet['tet_id']}_ray{epsilon_index}_{side_name}"
                packed = [*f_high, *f_low, *h, *accepted, *candidate_high,
                          *candidate_low, *parameters]
                rows.append(' '.join([key, kind] + [format(float(v), '.17g') for v in packed]))
                sources[key] = (tet, side_ray)
                side_ray['replay_key'] = key
                # Repeat the source oracle at the same side F/state along the
                # centered ray reconstructed from nodal velocity deltas. This
                # distinguishes F-storage quantization from the actual Newton
                # unknown direction without constructing a dense 9x9 matrix.
                hv = pair.f32_array(ray['H_velocity_center_local'].ravel().tolist(), 9,
                                    'realized velocity-gradient ray')
                velocity_key = key + '_velocity'
                packed_velocity = [*f_high, *f_low, *hv, *accepted,
                                   *candidate_high, *candidate_low, *parameters]
                rows.append(' '.join([velocity_key, kind] +
                                     [format(float(v), '.17g') for v in packed_velocity]))
                sources[velocity_key] = (tet, side_ray)
                side_ray['velocity_replay_key'] = velocity_key
    return rows, sources, rays


def _run_realized_source_replay(pair_summary, rows, output):
    binding = pair_summary.get('binding', {})
    command = binding.get('replay_command')
    if not isinstance(command, list) or len(command) < 3:
        # Older sealed pair summaries recorded the native invocation and
        # replay/source hashes but omitted replay_command. Reconstruct only the
        # sibling replay executable and source materials from those bound
        # paths, then require every artifact hash to match before execution.
        native = binding.get('native_command')
        snapshot = binding.get('source_snapshot')
        if not isinstance(native, list) or not native or not snapshot:
            raise ValueError('pair analysis has no exact source replay command or reconstructable bindings')
        probe = Path(native[0])
        source_root = Path(snapshot)
        command = [str(probe.with_name('numi-matter-cardboard-solver-pair-replay')),
                   str(source_root/'matter/materials/hajali2009_liner_hill_ideal.nmatter'),
                   str(source_root/'matter/materials/hajali2009_medium_hill_ideal.nmatter')]
    replay_exe, liner, medium = map(Path, command[:3])
    if not replay_exe.is_file() or not liner.is_file() or not medium.is_file():
        raise ValueError('captured source replay executable or material input is unavailable')
    if pair.sha256(replay_exe) != binding.get('replay_executable_sha256'):
        raise ValueError('realized-ray replay executable differs from pair-analysis binding')
    material_hashes = binding.get('replay_source_material_sha256', {})
    if (pair.sha256(liner) != material_hashes.get('liner') or
            pair.sha256(medium) != material_hashes.get('medium')):
        raise ValueError('realized-ray replay material differs from pair-analysis binding')
    input_text = '\n'.join(rows) + '\n'
    command_buffer = list(command)
    # The derivative sweep is activated only by the pair tool for the historic
    # _s0 key suffix. These realized-ray keys avoid that supplemental sweep.
    process = subprocess.run(command_buffer, input=input_text, text=True,
                             capture_output=True, check=False)
    parsed = [json.loads(line) for line in process.stdout.splitlines() if line.strip()]
    expected = [row.split()[0] for row in rows]
    actual = [row.get('key') for row in parsed]
    valid = (process.returncode == 0 and len(parsed) == len(rows) and
             all(row.get('valid') is True for row in parsed) and actual == expected)
    output.mkdir(parents=True, exist_ok=True)
    (output/'realized-ray-replay-input.tsv').write_text(input_text)
    (output/'realized-ray-replay-output.jsonl').write_text(process.stdout)
    (output/'realized-ray-replay-stderr.txt').write_text(process.stderr)
    if not valid:
        raise ValueError('independent realized-ray replay failed or reordered rows')
    return {row['key']: row for row in parsed}, dict(
        command=command_buffer, returncode=process.returncode, row_count=len(rows),
        executable_sha256=pair.sha256(replay_exe), liner_sha256=pair.sha256(liner),
        medium_sha256=pair.sha256(medium),
        input_sha256=pair.sha256(output/'realized-ray-replay-input.tsv'),
        output_sha256=pair.sha256(output/'realized-ray-replay-output.jsonl'),
        stderr_sha256=pair.sha256(output/'realized-ray-replay-stderr.txt'),
        status='complete_valid_source_replay')


def audit_realized_environment(header, nodes, tets, replay, realized_replay,
                              rays, environment):
    env_nodes = sorted((r for r in nodes if r['environment'] == environment),
                       key=lambda r: r['node_id'])
    count = int(header['node_count_per_environment'])
    if [r['node_id'] for r in env_nodes] != list(range(count)):
        raise ValueError('realized assembly audit node identities are incomplete')
    dt = float(header['dt_seconds_by_environment'][environment])
    masses = np.asarray([r['accepted']['mass_kg'] for r in env_nodes], dtype=np.float64)
    free = np.asarray([
        r['center']['constraint_tag'] not in (1,3) and masses[i] > 0
        for i, r in enumerate(env_nodes)], dtype=bool)
    env_tets = sorted((t for t in tets if t['environment'] == environment),
                      key=lambda t: t['tet_id'])
    env_rays = {(ray['tet_id'], header['epsilon_values'].index(ray['epsilon'])): ray
        for ray in rays if ray['environment'] == environment}
    reports = []
    for epsilon_index, epsilon in enumerate(header['epsilon_values']):
        if epsilon <= 0:
            raise ValueError('invalid epsilon in realized assembly audit')
        side_force_native = [np.zeros((count,3)) for _ in range(4)]
        side_force_source = [np.zeros((count,3)) for _ in range(4)]
        side_tangent_source = [np.zeros((count,3)) for _ in range(4)]
        side_tangent_velocity = [np.zeros((count,3)) for _ in range(4)]
        intended_tangent_source = np.zeros((count,3))
        center_velocity_tangent_source = np.zeros((count,3))
        per_tet = []
        for tet in env_tets:
            ids = tet['node_ids']
            inv = np.asarray(tet['inverse_rest_rows'], dtype=np.float64).reshape(3,3)
            volume = float(tet['reference_volume_m3'])
            q = rotation(tet['frame_xyzw'])
            if tet['material_index'] not in pair.material_descriptors(header):
                raise ValueError('realized source action has no material descriptor')
            descriptor = pair.material_descriptors(header)[tet['material_index']]
            material_name = descriptor['name']
            if material_name not in pair.MATERIAL_KIND_BY_NAME:
                # This assembly audit currently has a separately audited
                # finite-starch oracle; any other material is not silently
                # excluded from the action balance.
                if material_name != 'starch2007_finite_glue_elastic':
                    raise ValueError('unsupported material in realized assembly action')
            tet_ray = env_rays.get((int(tet['tet_id']), epsilon_index))
            if tet_ray is None:
                raise ValueError('missing reconstructed realized ray for tetrahedron')
            center_key = f"e{environment}_t{tet['tet_id']}_s0"
            center_tangent = None
            if material_name == 'starch2007_finite_glue_elastic':
                _, center_tangent = starch_reference(tet, header)
                center_velocity_tet = dict(tet, center=dict(tet['center'],
                    local_F_direction_per_unit_velocity=tet_ray['H_velocity_center_local']))
                _, center_tangent_velocity = starch_reference(center_velocity_tet, header)
            else:
                if center_key not in replay or replay[center_key].get('valid') is not True:
                    raise ValueError('missing center source tangent for realized audit')
                center_tangent = np.asarray(
                    replay[center_key]['oracle_tangent_at_reprojected_state_Pa_per_s'],
                    dtype=np.float64).reshape(3,3)
                center_velocity_key = tet_ray.get('center_velocity_replay_key')
                if center_velocity_key not in realized_replay:
                    raise ValueError('missing source tangent along realized center velocity ray')
                center_velocity_tangent = realized_replay[center_velocity_key]
                if center_velocity_tangent.get('valid') is not True:
                    raise ValueError('invalid source tangent along realized center velocity ray')
                center_tangent_velocity = np.asarray(
                    center_velocity_tangent['oracle_tangent_at_reprojected_state_Pa_per_s'],
                    dtype=np.float64).reshape(3,3)
            np.add.at(intended_tangent_source, ids,
                      force(center_tangent @ q.T, inv, volume))
            np.add.at(center_velocity_tangent_source, ids,
                      force(center_tangent_velocity @ q.T, inv, volume))
            for side_name in ('plus','minus'):
                side_ray = tet_ray[side_name]
                side_index = side_ray['side_index']
                side = tet['perturbations'][side_index]
                native_p = np.asarray(side['P_world_Pa'], dtype=np.float64).reshape(3,3)
                np.add.at(side_force_native[side_index], ids,
                          force(native_p, inv, volume))
                local_side_tet = dict(tet)
                local_side_tet['center'] = dict(tet['center'],
                    F_local=side_ray['F_local'],
                    local_F_direction_per_unit_velocity=side_ray['H_from_F_local'])
                if material_name == 'starch2007_finite_glue_elastic':
                    source_p, source_tangent = starch_reference(local_side_tet, header)
                    local_velocity_tet = dict(local_side_tet, center=dict(
                        local_side_tet['center'],
                        local_F_direction_per_unit_velocity=tet_ray['H_velocity_center_local']))
                    _, source_tangent_velocity = starch_reference(local_velocity_tet, header)
                else:
                    key = side_ray['replay_key']
                    if key is None or key not in realized_replay:
                        side_ray['source_action_status'] = 'inconclusive_zero_realized_F_ray'
                        continue
                    source = realized_replay[key]
                    source_p = np.asarray(source['oracle_reprojected_P_Pa'],
                                          dtype=np.float64).reshape(3,3)
                    source_tangent = np.asarray(
                        source['oracle_tangent_at_reprojected_state_Pa_per_s'],
                        dtype=np.float64).reshape(3,3)
                    side_ray['source_action_status'] = 'source_replay_valid'
                    velocity_key = side_ray.get('velocity_replay_key')
                    if velocity_key is None or velocity_key not in realized_replay:
                        side_ray['velocity_tangent_status'] = 'inconclusive_missing_velocity_ray_replay'
                        source_tangent_velocity = np.zeros((3,3))
                    else:
                        source_tangent_velocity = np.asarray(
                            realized_replay[velocity_key]['oracle_tangent_at_reprojected_state_Pa_per_s'],
                            dtype=np.float64).reshape(3,3)
                        side_ray['velocity_tangent_status'] = 'source_replay_valid'
                side_ray['source_P_local'] = source_p
                side_ray['source_tangent_local'] = source_tangent
                np.add.at(side_force_source[side_index], ids,
                          force(source_p @ q.T, inv, volume))
                np.add.at(side_tangent_source[side_index], ids,
                          force(source_tangent @ q.T, inv, volume))
                np.add.at(side_tangent_velocity[side_index], ids,
                          force(source_tangent_velocity @ q.T, inv, volume))
            per_tet.append(dict(tet_id=int(tet['tet_id']),
                plus_source_action=tet_ray['plus'].get('source_action_status','not_available'),
                minus_source_action=tet_ray['minus'].get('source_action_status','not_available'),
                realized_velocity_vs_intended_H=tet_ray['kinematic_metrics']['realized_velocity_vs_intended_H'],
                realized_F_vs_velocity_H=tet_ray['kinematic_metrics']['realized_F_vs_velocity_H'],
                representation=tet_ray['kinematic_metrics']))
        plus_index, minus_index = 2*epsilon_index, 2*epsilon_index+1
        dv_plus = np.asarray([r['epsilon_pair_realized_velocity_delta_m_s'][plus_index]
                              for r in env_nodes], dtype=np.float64)
        dv_minus = np.asarray([r['epsilon_pair_realized_velocity_delta_m_s'][minus_index]
                               for r in env_nodes], dtype=np.float64)
        dv_actual = (dv_plus-dv_minus)/(2.0*epsilon)
        mass_actual = masses[:,None]*dv_actual
        intended_dv = np.asarray([r['correction_delta_v_m_s'] for r in env_nodes],
                                 dtype=np.float64)
        mass_intended = masses[:,None]*intended_dv
        native_residual_plus = np.asarray([
            r['epsilon_pair_residual_impulses_Ns'][plus_index][:3] for r in env_nodes],
            dtype=np.float64)
        native_residual_minus = np.asarray([
            r['epsilon_pair_residual_impulses_Ns'][minus_index][:3] for r in env_nodes],
            dtype=np.float64)
        native_fd = -(native_residual_plus-native_residual_minus)/(2.0*epsilon)
        native_stress_fd = mass_actual-dt*(
            side_force_native[plus_index]-side_force_native[minus_index])/(2.0*epsilon)
        source_stress_fd = mass_actual-dt*(
            side_force_source[plus_index]-side_force_source[minus_index])/(2.0*epsilon)
        source_tangent_avg = 0.5*(
            side_tangent_source[plus_index]+side_tangent_source[minus_index])
        source_tangent_velocity_avg = 0.5*(
            side_tangent_velocity[plus_index]+side_tangent_velocity[minus_index])
        source_tangent_actual = mass_actual-dt*dt*source_tangent_avg
        source_tangent_velocity = mass_actual-dt*dt*source_tangent_velocity_avg
        source_tangent_intended = mass_intended-dt*dt*intended_tangent_source
        source_center_velocity_action = mass_actual-dt*dt*center_velocity_tangent_source
        native_operator = np.asarray([
            r['production_A_delta_v_impulse_Ns'][:3] for r in env_nodes],
            dtype=np.float64)
        # Constraint rows are not part of the free physical action comparison.
        def on_free(value):
            return value[free]
        zero_tangent_sides = int(sum(
            ray['centered_ray_status'] != 'measured'
            for ray in rays if ray['environment'] == environment and
            abs(ray['epsilon']-epsilon) <= 0.0))
        reports.append(dict(epsilon=float(epsilon),
            actual_ray_status=('measured' if np.linalg.norm(on_free(dv_actual)) > 0
                               else 'inconclusive_zero_realized_velocity_ray'),
            mass_action_actual_l2_Ns=float(np.linalg.norm(on_free(mass_actual))),
            mass_action_intended_l2_Ns=float(np.linalg.norm(on_free(mass_intended))),
            native_stiffness_action_intended_l2_Ns=float(np.linalg.norm(
                on_free(mass_intended-native_operator))),
            source_stiffness_action_actual_l2_Ns=float(np.linalg.norm(
                on_free(dt*dt*source_tangent_avg))),
            source_stiffness_action_realized_velocity_l2_Ns=float(np.linalg.norm(
                on_free(dt*dt*source_tangent_velocity_avg))),
            source_stiffness_action_intended_l2_Ns=float(np.linalg.norm(
                on_free(dt*dt*intended_tangent_source))),
            source_stiffness_action_center_velocity_l2_Ns=float(np.linalg.norm(
                on_free(dt*dt*center_velocity_tangent_source))),
            native_operator_vs_source_intended_tangent=metric(
                on_free(native_operator), on_free(source_tangent_intended)),
            source_center_velocity_vs_intended_stiffness=metric(
                on_free(center_velocity_tangent_source), on_free(intended_tangent_source)),
            native_operator_vs_source_center_velocity_action=metric(
                on_free(native_operator), on_free(source_center_velocity_action)),
            native_residual_FD_vs_native_stress_FD=metric(
                on_free(native_fd), on_free(native_stress_fd)),
            native_stress_FD_vs_source_stress_FD=metric(
                on_free(native_stress_fd), on_free(source_stress_fd)),
            source_stress_FD_vs_source_actual_tangent=metric(
                on_free(source_stress_fd), on_free(source_tangent_actual)),
            source_actual_vs_intended_tangent_action=metric(
                on_free(source_tangent_actual), on_free(source_tangent_intended)),
            source_realized_velocity_vs_intended_tangent_action=metric(
                on_free(source_tangent_velocity), on_free(source_tangent_intended)),
            source_F_ray_vs_velocity_ray_tangent_action=metric(
                on_free(source_tangent_actual), on_free(source_tangent_velocity)),
            native_operator_vs_source_actual_tangent=metric(
                on_free(native_operator), on_free(source_tangent_actual)),
            kinematic_velocity_ray_vs_intended=metric(
                on_free(dv_actual), on_free(intended_dv)),
            actual_ray_tangent_rows_missing_due_zero_F=int(
                sum(1 for tet in env_tets
                    if (env_rays[(int(tet['tet_id']),epsilon_index)]['plus'].get('replay_key') is None or
                        env_rays[(int(tet['tet_id']),epsilon_index)]['minus'].get('replay_key') is None)
                    and pair.MATERIAL_KIND_BY_NAME.get(
                        pair.material_descriptors(header)[int(tet['material_index'])]['name']) is not None)),
            zero_direction_inconclusive_count=zero_tangent_sides,
            decomposition_scope={
                'residual_vs_native_stress': 'residual assembly/scatter and impulse arithmetic',
                'native_vs_source_stress': 'native constitutive evaluation versus independent FP64 source stress at captured F',
                'source_secant_vs_source_tangent': 'finite epsilon and source branch/projection curvature',
                'actual_vs_intended_tangent': 'realized F/velocity ray representation and input perturbation difference',
                'sign_contract': 'A_FD=-(R_plus-R_minus)/(2 epsilon); A=M*dv-dt^2*K',
                'qualification': 'diagnostic_only_noise_floor_unmeasured'}))
        # This detailed element ledger is useful for tracing a single bad tet;
        # it is separate from the aggregate action balance above.
        reports[-1]['element_count'] = len(per_tet)
    return reports


def audit_environment(header, nodes, tets, replay, environment):
    nodes = sorted((r for r in nodes if r['environment'] == environment), key=lambda r:r['node_id'])
    count = int(header['node_count_per_environment'])
    if [r['node_id'] for r in nodes] != list(range(count)):
        raise ValueError('missing, duplicated, or noncontiguous node identity')
    dt = float(header['dt_seconds_by_environment'][environment])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('invalid object timestep')
    masses = np.array([r['accepted']['mass_kg'] for r in nodes])
    if any(r['center']['mass_kg'] != r['accepted']['mass_kg'] for r in nodes):
        raise ValueError('candidate masses differ from accepted masses')
    if not np.isfinite(masses).all() or (masses < 0).any():
        raise ValueError('invalid exact nodal masses')
    free = np.array([r['center']['constraint_tag'] not in (1,3) and masses[i] > 0
                     for i,r in enumerate(nodes)])
    dv = np.array([r['correction_delta_v_m_s'] for r in nodes])
    mass_action = masses[:,None]*dv
    force_sum = np.zeros((count,3))
    tangent_sum = np.zeros((count,3))
    oracle_candidate = np.zeros((count,3))
    oracle_reprojected = np.zeros((count,3))
    side_force = [np.zeros((count,3)) for _ in range(4)]
    audits = []
    paper_branch_classes = [set() for _ in range(2)]
    branch_crossing_tets = [0,0]
    near_yield_tets = [0,0]
    one_sided_crossings = [[0,0],[0,0]]
    one_sided_near_yield = [[0,0],[0,0]]
    env_tets = sorted((t for t in tets if t['environment'] == environment), key=lambda t:t['tet_id'])
    if [t['tet_id'] for t in env_tets] != list(range(int(header['tetrahedron_count_per_environment']))):
        raise ValueError('missing, duplicated, or noncontiguous tetrahedron identity')
    for tet in env_tets:
        ids = tet['node_ids']
        if len(ids) != 4 or len(set(ids)) != 4 or min(ids)<0 or max(ids)>=count:
            raise ValueError('invalid tetrahedron node identities')
        c = tet['center']
        tet_dt = c.get('dt_seconds_bulk_modulus_kind_frame_active', [dt])[0]
        if tet_dt != dt:
            raise ValueError('mixed object timesteps require a separate assembly audit')
        q = rotation(tet['frame_xyzw'])
        inv = matrix(tet['inverse_rest_rows'])
        volume = float(tet['reference_volume_m3'])
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError('invalid reference volume')
        p = matrix(c['P_local_Pa'])
        dp = matrix(c['dP_local_Pa_per_mps'])
        pworld = matrix(c['P_world_Pa'])
        dpworld = matrix(c['dP_world_Pa_per_mps'])
        elem = np.array(c['element_force_node_world_N'])
        elem_dp = np.array(c['operator_element_force_direction_nodes_N_per_s'])
        if elem.shape != (4,3) or elem_dp.shape != (4,3):
            raise ValueError('invalid captured four-node force shape')
        row = dict(environment=environment,tet_id=tet['tet_id'],material_index=tet['material_index'],
                   stress_frame=metric(pworld,p@q.T), tangent_frame=metric(dpworld,dp@q.T),
                   element_force=metric(elem,force(pworld,inv,volume)),
                   element_tangent=metric(elem_dp,force(dpworld,inv,volume)))
        key = f"e{environment}_t{tet['tet_id']}_s0"
        if key in replay:
            reference = replay[key]
            if reference.get('valid') is not True:
                raise ValueError('invalid independent paper reference row')
            oc = matrix(reference['oracle_tangent_at_captured_candidate_Pa_per_s'])
            op = matrix(reference['oracle_tangent_at_reprojected_state_Pa_per_s'])
            for ep in range(2):
                keys = [f"e{environment}_t{tet['tet_id']}_s{s}" for s in (2*ep+2,0,2*ep+1)]
                if any(k not in replay or replay[k].get('valid') is not True for k in keys):
                    raise ValueError('missing independent side reference')
                classes = [pair.classify_trial_q(replay[k]['elastic_trial_q'],1e-6) for k in keys]
                paper_branch_classes[ep].update(classes)
                near_yield_tets[ep] += 'near_yield_unresolved' in classes
                branch_crossing_tets[ep] += len(set(classes))>1 and 'near_yield_unresolved' not in classes
                # classes are [minus, center, plus]. Each one-sided action
                # retains its own branch eligibility; the other side cannot
                # invalidate a branch-preserving comparison.
                for side, outer in enumerate((classes[2], classes[0])):
                    local = (classes[1], outer)
                    one_sided_near_yield[ep][side] += 'near_yield_unresolved' in local
                    one_sided_crossings[ep][side] += len(set(local))>1 and 'near_yield_unresolved' not in local
        else:
            _, oc = starch_reference(tet,header)
            op = oc
        np.add.at(force_sum,ids,elem)
        np.add.at(tangent_sum,ids,elem_dp)
        np.add.at(oracle_candidate,ids,force(oc@q.T,inv,volume))
        np.add.at(oracle_reprojected,ids,force(op@q.T,inv,volume))
        for si,s in enumerate(tet['perturbations']):
            np.add.at(side_force[si],ids,force(matrix(s['P_world_Pa']),inv,volume))
        audits.append(row)
    native_action = np.array([r['production_A_delta_v_impulse_Ns'][:3] for r in nodes])
    assembled = mass_action-dt*dt*tangent_sum
    assembled[~free] = 0
    native_residual = np.array([r['center_residual_impulse_Ns'][:3] for r in nodes])
    velocity = np.array([r['center']['v_m_s'] for r in nodes])
    old_velocity = np.array([r['accepted']['v_m_s'] for r in nodes])
    external = native_residual - dt*force_sum + masses[:,None]*(velocity-old_velocity)
    # This external term is inferred, not an independently measured source force.
    report = dict(environment=environment,free_nodes=int(free.sum()),
        production_gather=metric(native_action[free],assembled[free]),
        captured_candidate_FP64_assembled_action=metric(native_action[free],(mass_action-dt*dt*oracle_candidate)[free]),
        reprojected_FP64_assembled_action=metric(native_action[free],(mass_action-dt*dt*oracle_reprojected)[free]),
        mass_action_l2_Ns=float(np.linalg.norm(mass_action[free])),
        captured_stiffness_action_l2_Ns=float(np.linalg.norm(dt*dt*tangent_sum[free])),
        production_action_l2_Ns=float(np.linalg.norm(native_action[free])),
        inferred_frozen_external_impulse_l2_Ns=float(np.linalg.norm(external[free])),
        external_source_independently_validated=False,
        strict_gate='inconclusive_no_measured_noise_floor',perturbations=[],centered_actions=[],one_sided_actions=[])
    for ep,epsilon in enumerate(header['epsilon_values']):
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('invalid representable perturbation')
        sides = []
        for side,sign in enumerate((1,-1)):
            si=2*ep+side
            key=f"epsilon_pair_{ep}_{'plus' if side==0 else 'minus'}"
            v=np.array([r[key]['v_m_s'] for r in nodes])
            native=np.array([r['epsilon_pair_residual_impulses_Ns'][si][:3] for r in nodes])
            expected=dt*side_force[si]-masses[:,None]*(v-old_velocity)+external
            actual_du=np.array([r['epsilon_pair_realized_u_delta_m'][si] for r in nodes])
            actual_dv=np.array([r['epsilon_pair_realized_velocity_delta_m_s'][si] for r in nodes])
            report['perturbations'].append(dict(epsilon=epsilon,side='plus' if sign==1 else 'minus',
                residual_gather=metric(native[free],expected[free]),
                realized_delta_v=metric(actual_dv[free],(sign*epsilon*dv)[free]),
                realized_u_velocity_relation=metric(actual_du[free],(dt*actual_dv)[free])))
            sides.append(native)
        fd=-(sides[0]-sides[1])/(2*epsilon)
        report['centered_actions'].append(dict(epsilon=epsilon,comparison=metric(native_action[free],fd[free]),
            branch_crossing_tetrahedra=branch_crossing_tets[ep],near_yield_tetrahedra=near_yield_tets[ep],
            smooth_branch_eligible=bool(np.linalg.norm(native_action[free]) > 0 and
                not branch_crossing_tets[ep] and not near_yield_tets[ep]),
            strict_gate='inconclusive_no_measured_noise_floor'))
        for side, sign in enumerate((1,-1)):
            # R has the opposite derivative sign to the velocity operator A.
            fd_one=-(sides[side]-native_residual)/(sign*epsilon)
            crossings=one_sided_crossings[ep][side]
            near=one_sided_near_yield[ep][side]
            report['one_sided_actions'].append(dict(epsilon=epsilon,
                side='plus' if sign==1 else 'minus',
                comparison=metric(native_action[free],fd_one[free]),
                branch_crossing_tetrahedra=crossings,near_yield_tetrahedra=near,
                smooth_branch_eligible=bool(np.linalg.norm(native_action[free])>0 and not crossings and not near),
                strict_gate='inconclusive_no_measured_noise_floor'))
    return report,audits


def run(capture,pair_analysis,output):
    if output.exists(): raise ValueError('output exists; preserve earlier analyses')
    summary=json.loads((pair_analysis/'summary.json').read_text())
    replay_path=pair_analysis/'replay-output.jsonl'
    replay_status = summary.get('replay',{})
    if (summary.get('schema')!=pair.SCHEMA or replay_status.get('invalid_rows')!=0 or
            replay_status.get('returncode')!=0 or replay_status.get('output_errors')):
        raise ValueError('complete valid pair-aware source replay is required')
    binding=summary.get('binding',{})
    if binding.get('capture_sha256')!=pair.sha256(capture) or binding.get('replay_output_sha256')!=pair.sha256(replay_path):
        raise ValueError('capture/replay hash differs from admitted pair analysis')
    records=normalize_native([json.loads(line) for line in capture.read_text().splitlines() if line.strip()])
    header=records[0]
    if len([r for r in records if r.get('record_type')=='integrity' and r.get('inputs_unchanged')])!=1:
        raise ValueError('capture terminal integrity is absent')
    replay={r['key']:r for r in (json.loads(line) for line in replay_path.read_text().splitlines() if line.strip())}
    nodes=[r for r in records if r['record_type']=='node']
    tets=[r for r in records if r['record_type']=='tetrahedron']
    realized_rows, _, realized_rays = _realized_replay_rows(
        header, nodes, tets)
    realized_replay, realized_replay_binding = _run_realized_source_replay(
        summary, realized_rows, output)
    environments,element_audits=[],[]
    for env in range(header['environment_count']):
        report,audits=audit_environment(header,nodes,tets,replay,env)
        report['realized_ray_decomposition'] = audit_realized_environment(
            header,nodes,tets,replay,realized_replay,realized_rays,env)
        environments.append(report); element_audits.extend(audits)
    layer_maxima={}
    for key in ('stress_frame','tangent_frame','element_force','element_tangent'):
        values=[(a[key]['relative_error'],a['environment'],a['tet_id']) for a in element_audits if a[key]['relative_error'] is not None]
        layer_maxima[key]={'max_relative_error':max(values)[0],'environment':max(values)[1],'tet_id':max(values)[2]} if values else None
    report=dict(schema=SCHEMA,binding=dict(capture_sha256=pair.sha256(capture),pair_summary_sha256=pair.sha256(pair_analysis/'summary.json'),
        replay_output_sha256=pair.sha256(replay_path),analyzer_sha256=pair.sha256(Path(__file__)),
        normalization_source_sha256=pair.sha256(Path(__file__).with_name('cardboard_solver_capture_analysis.py')),
        pair_analyzer_source_sha256=pair.sha256(Path(pair.__file__)),
        realized_ray_replay_binding=realized_replay_binding),
        scope='captured bulk-plus-mass operator and element-frame transformations; full nonlinear contact operator not qualified',
        run_completion=summary['run_completion'],layer_maxima=layer_maxima,environments=environments,
        strict_gate='inconclusive_no_measured_noise_floor',physical_validation=False,terminal_failure_explained=False)
    # The source replay creates this directory first to preserve its raw
    # input/output even if later audit stages fail.
    output.mkdir(parents=True, exist_ok=True)
    (output/'element-audit.jsonl').write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in element_audits))
    realized_element_rows=[]
    for ray in realized_rays:
        realized_element_rows.append(dict(
            environment=ray['environment'],tet_id=ray['tet_id'],epsilon=ray['epsilon'],
            dt_seconds=ray['dt'],centered_ray_status=ray['centered_ray_status'],
            intended_H_local=ray['intended_H_local'].tolist(),
            H_velocity_center_local=ray['H_velocity_center_local'].tolist(),
            H_F_center_local=ray['H_F_center_local'].tolist(),
            center_velocity_replay_key=ray.get('center_velocity_replay_key'),
            kinematic_metrics=ray['kinematic_metrics'],
            plus=dict(side_index=ray['plus']['side_index'],
                F_local=ray['plus']['F_local'].tolist(),
                H_from_F_local=ray['plus']['H_from_F_local'].tolist(),
                H_from_velocity_local=ray['plus']['H_from_velocity_local'].tolist(),
                replay_key=ray['plus'].get('replay_key'),
                velocity_replay_key=ray['plus'].get('velocity_replay_key')),
            minus=dict(side_index=ray['minus']['side_index'],
                F_local=ray['minus']['F_local'].tolist(),
                H_from_F_local=ray['minus']['H_from_F_local'].tolist(),
                H_from_velocity_local=ray['minus']['H_from_velocity_local'].tolist(),
                replay_key=ray['minus'].get('replay_key'),
                velocity_replay_key=ray['minus'].get('velocity_replay_key'))))
    (output/'realized-ray-element-audit.jsonl').write_text(
        ''.join(json.dumps(x,allow_nan=False)+'\n' for x in realized_element_rows))
    (output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture',type=Path);parser.add_argument('pair_analysis',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    try:
        result=run(args.capture,args.pair_analysis,args.output)
        print(json.dumps(dict(schema=SCHEMA,environments=result['environments'],layer_maxima=result['layer_maxima'],strict_gate=result['strict_gate']),allow_nan=False))
        return 3
    except (ValueError,KeyError,TypeError,OSError) as error:
        print(json.dumps(dict(error=str(error),admission=False)))
        return 2


if __name__=='__main__': raise SystemExit(main())
