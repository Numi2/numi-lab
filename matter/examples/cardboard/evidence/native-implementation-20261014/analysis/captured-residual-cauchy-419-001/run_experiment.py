#!/usr/bin/env python3
"""Source-bound FP64 residual-gradient experiment for capture 419.

This is an offline linear-algebra experiment, not a Matter solve or acceptance
path. It assembles the captured FEM-only cached-tangent operator, verifies its
forward/transpose actions with independent element gathers, and exports
Cauchy directions for a later production residual replay.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform

import numpy as np
from scipy import sparse

BASE = Path('/Users/home/cardboard-implementation-20261014')
CAPTURE = BASE / 'solver/two-stage-first-failure-rich-fd-419/operator-capture'
RUN = CAPTURE.parent
SNAPSHOT = BASE / 'snapshots/source-status5-vtccd-diagnostics-125'
MODULE = SNAPSHOT / 'matter/tools/cardboard_cached_operator_audit.py'
REPLAY = BASE / 'analysis/native-operator-term-replay-result-422'
SWEEP = BASE / 'analysis/native-residual-fd-sweep-result-424'
SWEEP_AUDIT = BASE / 'analysis/native424-residual-sweep-audit-001/report.json'
OUT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_module():
    spec = importlib.util.spec_from_file_location('captured_operator_audit', MODULE)
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot load source125 cached-operator audit module')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_capture(path: Path):
    # Preserve JSON integer negative zero should it appear in numeric payloads.
    # Remaining captured float words are explicitly rounded through FP32 by
    # the source-bound assembly module before use.
    def parse_int(token: str):
        return -0.0 if token == '-0' else int(token)
    header = integrity = None
    nodes, tets = [], []
    with path.open('r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_int=parse_int,
                                 parse_constant=lambda v: (_ for _ in ()).throw(
                                     ValueError(f'nonfinite JSON token {v}')))
            except Exception as e:
                raise ValueError(f'capture line {line_number} is not strict finite JSON: {e}') from e
            kind = row.get('record')
            if kind == 'header':
                if header is not None: raise ValueError('duplicate header')
                header = row
            elif kind == 'node': nodes.append(row)
            elif kind == 'tetrahedron': tets.append(row)
            elif kind == 'integrity':
                if integrity is not None: raise ValueError('duplicate integrity')
                integrity = row
            else: raise ValueError(f'unknown capture record at line {line_number}')
    if header is None or integrity is None:
        raise ValueError('capture missing header or terminal integrity record')
    return header, nodes, tets, integrity


def write_array(path: Path, array: np.ndarray, dtype: str):
    arr = np.asarray(array, dtype=np.dtype(dtype).newbyteorder('<'))
    arr.tofile(path)
    return {'path': path.name, 'dtype': arr.dtype.str,
            'shape': list(arr.shape), 'bytes': path.stat().st_size,
            'sha256': sha256(path)}


def norm(v):
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64)))


def require_clean_command_receipt(receipt, label):
    """Accept v1 native-run receipts with a nested command table or v2 flat form."""
    status = receipt
    if 'exit_code' not in status and isinstance(receipt.get('commands'), dict):
        status = receipt['commands'].get('native', {})
    if status.get('timed_out') is not False or status.get('exit_code') != 0:
        raise ValueError(f'{label} native replay is not a clean diagnostic completion')


def dot64(a, b):
    """Accumulate a real dot in Python binary64 without BLAS status leakage."""
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.shape != bb.shape:
        raise ValueError('dot operands have different shapes')
    return math.fsum(float(x) * float(y) for x, y in zip(aa, bb))


def residual_stats(r0, A, p):
    r1 = r0 - A @ p
    phi0 = 0.5 * dot64(r0, r0)
    phi1 = 0.5 * dot64(r1, r1)
    return {
        'step_l2_m_per_s': norm(p),
        'step_max_abs_m_per_s': float(np.max(np.abs(p))) if p.size else 0.0,
        'predicted_linear_residual_l2_Ns': norm(r1),
        'predicted_residual_ratio': norm(r1) / norm(r0),
        'predicted_merit_phi_before_N2s2': phi0,
        'predicted_merit_phi_after_N2s2': phi1,
        'predicted_merit_fractional_drop': (phi0 - phi1) / phi0,
        'directional_merit_derivative_at_zero': None,
    }


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUT,
                        help='new output directory; existing report/manifest is never overwritten')
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if any((OUT / n).exists() for n in ('report.json', 'manifest.json')):
        raise SystemExit('output already exists; refusing to overwrite a sealed package')
    coa = load_module()
    prediction = RUN / 'prediction.json'
    receipt = RUN / 'receipt.json'
    stderr = RUN / 'stderr.log'
    if not stderr.is_file():
        raise SystemExit(f'missing source run stderr {stderr}')
    binding = coa.bind_inputs(CAPTURE, prediction, receipt, stderr, 0,
                              source_snapshot_override=SNAPSHOT)
    header, nodes, tets, integrity = parse_capture(CAPTURE)
    if int(header.get('root', -1)) != 1 or int(header.get('newton_iteration', -1)) != 48:
        raise ValueError('capture selection is not root 1 / Newton 48')
    mass, scaled_k, A, delta, meta = coa.validate_and_assemble(
        header, nodes, tets, integrity, 0)
    nnode = int(meta['node_count'])
    free_nodes = np.asarray(meta['free_nodes'], dtype=np.int64)
    dt = float(meta['dt_seconds'])
    ndof = A.shape[0]

    # Load and cross-check the same-source native action/residual replay.
    replay_report_path = REPLAY / 'result/replay-report.json'
    replay_manifest_path = REPLAY / 'manifest.json'
    replay_report = json.loads(replay_report_path.read_text())
    replay_receipt = json.loads((REPLAY / 'receipt.json').read_text())
    require_clean_command_receipt(replay_receipt, '422')
    if replay_report.get('capture_selection') != {'control_step': 1, 'newton_iteration': 48}:
        raise ValueError('422 replay selection does not match capture 419')
    if replay_report.get('physical_transactions') != 0:
        raise ValueError('422 replay unexpectedly reports physical transactions')
    cached_rows = np.fromfile(REPLAY / 'result/cached-action.float4.bin', dtype='<f4')
    env_count = int(header['environment_count'])
    if cached_rows.size != env_count * nnode * 4:
        raise ValueError('422 cached action extent mismatch')
    cached_action = cached_rows.reshape(env_count, nnode, 4)[0, :, :3].astype(np.float64)[free_nodes].reshape(-1)
    captured_action = np.asarray(meta['action_component_arrays']['native_Ns'], dtype=np.float64)
    b = np.asarray(meta['native_rhs_free_impulse_Ns'], dtype=np.float64)
    assembled_action = np.asarray(A @ delta).reshape(-1)
    r = b - assembled_action
    r_captured_action = b - captured_action
    r_422_cached_action = b - cached_action

    # Independent elementwise matrix-free forward and transpose action.
    tet_rows = {int(row['tet_id']): row for row in tets if int(row['environment']) == 0}
    dof_map = np.full((nnode, 3), -1, dtype=np.int64)
    dof_map[free_nodes] = 3 * np.arange(free_nodes.size)[:, None] + np.arange(3)[None, :]
    dt2 = dt * dt
    local_cache = []
    for tid, tet in tet_rows.items():
        cache = tet['cached_algorithmic_tangent_local']
        tangent = coa.captured_f32_array(cache['matrix_row_major_local_dP_dF_Pa_per_dimensionless_F'],
                                          (81,), f'tet {tid} tangent').reshape(9, 9)
        meta_local = coa.captured_f32_array(tet['center']['dt_seconds_bulk_modulus_kind_frame_active'],
                                             (4,), f'tet {tid} metadata')
        inv = coa.captured_f32_array(tet['inverse_rest_rows_per_m'], (3, 3), f'tet {tid} inverse-rest')
        volume = coa.captured_f32_scalar(tet['rest_volume_m3'], f'tet {tid} volume')
        quat = coa.captured_f32_array(tet['material_frame_quaternion_xyzw'], (4,), f'tet {tid} frame')
        frame_active = bool(meta_local[3] == 1.0)
        Ke = coa.local_tet_stiffness(inv, volume, tangent, quat, frame_active) * dt2
        conn = np.asarray(tet['nodes'], dtype=np.int64)
        local_ids = np.array([dof_map[int(v), axis] for v in conn for axis in range(3)], dtype=np.int64)
        keep = np.flatnonzero(local_ids >= 0)
        local_cache.append((local_ids[keep], Ke[np.ix_(keep, keep)]))

    mass_diag = mass.diagonal()
    def matrix_free_action(x, transpose=False):
        x = np.asarray(x, dtype=np.float64)
        if x.shape != (ndof,): raise ValueError('matrix-free vector extent mismatch')
        y = mass_diag * x
        for ids, local in local_cache:
            local_x = x[ids]
            local_y = (local.T @ local_x) if transpose else (local @ local_x)
            np.add.at(y, ids, local_y)
        return y

    rng = np.random.default_rng(0x419A)
    vx = rng.standard_normal(ndof)
    vy = rng.standard_normal(ndof)
    mf_ax = matrix_free_action(vx)
    mf_aty = matrix_free_action(vy, transpose=True)
    csr_ax = np.asarray(A @ vx).reshape(-1)
    csr_aty = np.asarray(A.T @ vy).reshape(-1)
    dot_l = dot64(vy, mf_ax)
    dot_r = dot64(vx, mf_aty)
    adjoint_abs = abs(dot_l - dot_r)
    adjoint_rel = adjoint_abs / max(abs(dot_l), abs(dot_r), 1.0e-300)
    forward_rel = norm(mf_ax - csr_ax) / max(norm(csr_ax), 1.0e-300)
    transpose_rel = norm(mf_aty - csr_aty) / max(norm(csr_aty), 1.0e-300)
    if max(forward_rel, transpose_rel, adjoint_rel) > 2.0e-12:
        raise ValueError('independent matrix-free operator/adjoint check exceeded 2e-12')

    # Residual-merit Cauchy directions. For phi(delta)=1/2||b-A delta||^2,
    # the correction-increment descent direction is +A^T r.
    g = np.asarray(A.T @ r).reshape(-1)
    Ag = np.asarray(A @ g).reshape(-1)
    raw_denom = dot64(Ag, Ag)
    if not np.isfinite(raw_denom) or raw_denom <= 0.0:
        raise ValueError('raw Cauchy denominator is invalid')
    raw_alpha = float(dot64(g, g) / raw_denom)
    raw_p = raw_alpha * g

    col_norm = np.sqrt(np.asarray(A.power(2).sum(axis=0)).reshape(-1))
    if not np.isfinite(col_norm).all() or np.any(col_norm <= 0.0):
        raise ValueError('column equilibration contains a zero/nonfinite column')
    scale = 1.0 / col_norm
    scaled_gradient = scale * g
    scaled_A_gradient = np.asarray(A @ (scale * scaled_gradient)).reshape(-1)
    scaled_denom = dot64(scaled_A_gradient, scaled_A_gradient)
    if not np.isfinite(scaled_denom) or scaled_denom <= 0.0:
        raise ValueError('column-scaled Cauchy denominator is invalid')
    scaled_alpha = float(dot64(scaled_gradient, scaled_gradient) / scaled_denom)
    scaled_p = scale * (scaled_alpha * scaled_gradient)

    residual_views = {
        'captured_production_action': r_captured_action,
        '422_cached_action': r_422_cached_action,
    }
    residual_view_report = {}
    for label, rv in residual_views.items():
        gv = np.asarray(A.T @ rv).reshape(-1)
        cosine = dot64(g, gv) / (norm(g) * norm(gv))
        trial_r = rv - A @ scaled_p
        phi_before = 0.5 * dot64(rv, rv)
        phi_after = 0.5 * dot64(trial_r, trial_r)
        residual_view_report[label] = {
            'residual_l2_Ns': norm(rv),
            'difference_from_assembled_residual_l2_Ns': norm(rv - r),
            'gradient_l2': norm(gv),
            'gradient_cosine_to_assembled_gradient': cosine,
            'column_scaled_cauchy_directional_merit_derivative': -dot64(gv, scaled_p),
            'column_scaled_cauchy_predicted_residual_l2_Ns': norm(trial_r),
            'column_scaled_cauchy_predicted_residual_ratio': norm(trial_r) / norm(rv),
            'column_scaled_cauchy_predicted_merit_fractional_drop': (phi_before - phi_after) / phi_before,
        }

    # A transparent relative trust cap in velocity-correction coordinates.
    # 1e-6 of ||delta|| is deliberately reported, not asserted as a law.
    delta_norm = norm(delta)
    trust_radius = 1.0e-6 * delta_norm
    trust_scale = min(1.0, trust_radius / max(norm(scaled_p), 1.0e-300))
    trust_p = trust_scale * scaled_p

    dirs = {
        'raw_cauchy': raw_p,
        'column_equilibrated_cauchy': scaled_p,
        'column_equilibrated_cauchy_trust_1e-6_delta': trust_p,
        'column_equilibrated_cauchy_025': 0.25 * scaled_p,
        'column_equilibrated_cauchy_050': 0.50 * scaled_p,
    }
    position = np.asarray(meta['node_positions_m'], dtype=np.float64)
    position_f32 = position.astype(np.float32)
    trials = {}
    for name, p in dirs.items():
        stats = residual_stats(r, A, p)
        stats['directional_merit_derivative_at_zero'] = -dot64(g, p)
        full = np.zeros((nnode, 4), dtype=np.float64)
        full[free_nodes, :3] = p.reshape(-1, 3)
        displacement = dt * full[:, :3]
        moved = (position_f32.astype(np.float64) + displacement).astype(np.float32)
        changed = np.any(moved.view(np.uint32) != position_f32.view(np.uint32), axis=1)
        changed_free = np.any(changed[free_nodes])
        stats.update({
            'position_increment_linf_m': float(np.max(np.abs(displacement))) if displacement.size else 0.0,
            'position_words_changed_count_estimate': int(np.count_nonzero(changed)),
            'free_position_words_changed_count_estimate': int(np.count_nonzero(changed[free_nodes])),
            'position_word_change_note': 'Estimate only: center position cast to FP32, add dt*direction in FP64, round once to FP32; production kinematics may use a different accepted/current composition order.',
        })
        trials[name] = stats
        write_array(OUT / f'{name}.free-delta-v.float64le.bin', p, '<f8')
        write_array(OUT / f'{name}.full-node-delta-v.float32x4.bin', full, '<f4')

    # Also export selected authoritative data and matrix for an exact offline/native replay.
    artifacts = []
    artifacts.append(write_array(OUT / 'free-rhs.float64le.bin', b, '<f8'))
    artifacts.append(write_array(OUT / 'free-correction.float64le.bin', delta, '<f8'))
    artifacts.append(write_array(OUT / 'free-linear-residual.float64le.bin', r, '<f8'))
    artifacts.append(write_array(OUT / 'free-residual-gradient-AtransposeR.float64le.bin', g, '<f8'))
    artifacts.append(write_array(OUT / 'free-column-norms.float64le.bin', col_norm, '<f8'))
    artifacts.append(write_array(OUT / 'free-mass-diagonal.float64le.bin', mass_diag, '<f8'))
    artifacts.append(write_array(OUT / 'full-node-positions.float32x3.bin', position_f32, '<f4'))
    matrix_path = OUT / 'A_free_csr.npz'
    sparse.save_npz(matrix_path, A)
    artifacts.append({'path': matrix_path.name, 'format': 'scipy-csr-npz',
                      'shape': list(A.shape), 'nnz': int(A.nnz),
                      'bytes': matrix_path.stat().st_size, 'sha256': sha256(matrix_path)})
    # Add trial direction products to artifact table after all files exist.
    for name in dirs:
        for suffix in ('free-delta-v.float64le.bin', 'full-node-delta-v.float32x4.bin'):
            path = OUT / f'{name}.{suffix}'
            artifacts.append({'path': path.name, 'bytes': path.stat().st_size, 'sha256': sha256(path)})

    replay_cached = replay_report['environment_0']
    sweep_report = json.loads(SWEEP_AUDIT.read_text())
    sweep_receipt = json.loads((SWEEP / 'receipt.json').read_text())
    require_clean_command_receipt(sweep_receipt, '424')
    if sweep_report.get('capture_sha256') != sha256(CAPTURE):
        raise ValueError('424 residual sweep is not bound to capture 419')
    receipt_row = json.loads(receipt.read_text())
    report = {
        'schema': 'numi.cardboard.captured-residual-cauchy-experiment.v1',
        'verdict': 'CPU_OPERATOR_EXPERIMENT_COMPLETE_NONLINEAR_REPLAY_REQUIRED',
        'scope': 'Source-bound offline FP64 residual merit experiment for the exact rejected source125 root1/Newton48 FEM operator capture. This does not alter production and is not a physical acceptance or convergence result.',
        'selection': {'root': 1, 'newton_iteration': 48, 'environment': 0,
                      'world_fingerprint_hex': header.get('world_fingerprint_hex'),
                      'execution_fingerprint_hex': header.get('execution_fingerprint_hex'),
                      'physical_transactions': replay_report.get('physical_transactions')},
        'input_bindings': {
            'capture_path': str(CAPTURE), 'capture_sha256': sha256(CAPTURE),
            'prediction_sha256': sha256(prediction), 'receipt_sha256': sha256(receipt),
            'stderr_sha256': sha256(stderr), 'source_snapshot': str(SNAPSHOT),
            'assembly_module_path': str(MODULE), 'assembly_module_sha256': sha256(MODULE),
            'native_replay_manifest_sha256': sha256(replay_manifest_path),
            'native_replay_report_sha256': sha256(replay_report_path),
            'native_replay_receipt_sha256': sha256(REPLAY / 'receipt.json'),
            'native_cached_action_sha256': sha256(REPLAY / 'result/cached-action.float4.bin'),
            'native_residual_sweep_manifest_sha256': sha256(SWEEP / 'manifest.json'),
            'native_residual_sweep_report_sha256': sha256(SWEEP_AUDIT),
            'native_residual_sweep_receipt_sha256': sha256(SWEEP / 'receipt.json'),
            'native_residual_sweep_capture_sha256': sweep_report.get('capture_sha256'),
            'source_input_binding_result': binding,
            'run_receipt': {'exit_code': receipt_row.get('exit_code'),
                            'timed_out': receipt_row.get('timed_out'),
                            'source_inputs_unchanged': receipt_row.get('source_inputs_unchanged')},
        },
        'operator': {
            'assembly': meta['assembly'],
            'matrix_layout': meta['matrix_layout'],
            'shape': list(A.shape), 'nnz': int(A.nnz),
            'free_nodes': meta['free_node_count'], 'fixed_nodes': meta['fixed_node_count'],
            'constraint_tag_counts': meta['constraint_tag_counts'],
            'dt_seconds': dt, 'relative_skew_norm': meta['relative_skew_norm'],
            'fixed_correction_max_abs_m_per_s': meta['action_parity']['fixed_correction_max_abs_m_per_s'],
            'fixed_native_action_max_abs_Ns': meta['action_parity']['fixed_rows_max_abs_native_action_Ns'],
            'matrix_free_check': {
                'seed': 0x419A, 'forward_relative_error': forward_rel,
                'transpose_relative_error': transpose_rel,
                'adjoint_dot_left': dot_l, 'adjoint_dot_right': dot_r,
                'adjoint_absolute_difference': adjoint_abs,
                'adjoint_relative_difference': adjoint_rel,
                'gate': 'all three relative errors <= 2e-12', 'passed': True,
            },
        },
        'residual': {
            'definition': 'r = b - A*delta_v; phi = 0.5*||r||_2^2. Since A*delta_v approximates -dR/d(delta_v), the correction increment descent direction is +A^T*r.',
            'rhs_l2_Ns': norm(b), 'assembled_A_delta_l2_Ns': norm(assembled_action),
            'assembled_residual_l2_Ns': norm(r), 'assembled_relative_residual': norm(r) / norm(b),
            'captured_production_action_l2_Ns': norm(captured_action),
            '422_cached_native_action_l2_Ns': norm(cached_action),
            'captured_production_residual_l2_Ns': norm(r_captured_action),
            '422_cached_action_residual_l2_Ns': norm(r_422_cached_action),
            'assembled_vs_captured_action_l2_Ns': norm(assembled_action - captured_action),
            'assembled_vs_422_cached_action_l2_Ns': norm(assembled_action - cached_action),
            '422_reported_cached_residual_l2_Ns': replay_cached.get('cached_rhs_minus_Ax_l2_Ns'),
            '424_all_retained_endpoints_bitwise_equal': sweep_report.get('all_retained_endpoints_bitwise_equal'),
            '424_derivative_qualified': sweep_report.get('derivative_qualified'),
            'interpretation': 'The assembled residual is the FP64 residual of the captured tangent operator and captured correction. It is close to, but not identical with, native FP32 action replays; differences are reported separately, not treated as a noise floor.',
            'alternate_residual_view_sensitivity': residual_view_report,
        },
        'gradient': {
            'vector': 'g=A^T*r using the unsymmetrized captured operator',
            'l2': norm(g), 'max_abs': float(np.max(np.abs(g))),
            'unsymmetrized_operator_required': True,
        },
        'directions': trials,
        'trust_region': {
            'metric': 'Euclidean velocity-correction norm ||p||_2',
            'radius': trust_radius,
            'radius_fraction_of_captured_correction_norm': 1.0e-6,
            'column_equilibrated_cauchy_unclipped_norm': norm(scaled_p),
            'applied_scale': trust_scale,
            'note': 'This is an illustrative scale-aware cap, not a calibrated physical trust radius. The saved column-equilibrated Cauchy direction is the exact unconstrained FP64 minimizer along its scaled-gradient ray.'
        },
        'limitations': [
            'The input is a rejected Newton candidate; no accepted physical state or successful transaction is claimed.',
            'This is the FEM block only. The compiled fixture has zero rigid proxies, contact pairs, fields, and cohesive faces; no missing cross-block contribution is inferred.',
            'The nonlinear production residual was not evaluated at these new directions. Root must test the exported candidate directions through the production residual replay before attributing nonlinear descent.',
            'The 424 native endpoint sweep matched retained endpoints bitwise but did not qualify a finite-difference derivative; no numerical noise floor is inferred.',
            'FP32 position-word counts are single-add representability estimates only and do not reproduce the full production candidate preparation order.',
            'No operator, tolerance, force law, or acceptance criterion was changed.'
        ],
        'artifacts': artifacts,
        'python': platform.python_version(), 'numpy': np.__version__,
    }
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({
        'verdict': report['verdict'],
        'shape': report['operator']['shape'],
        'residual_l2_Ns': report['residual']['assembled_residual_l2_Ns'],
        'raw_cauchy': trials['raw_cauchy'],
        'column_equilibrated_cauchy': trials['column_equilibrated_cauchy'],
        'matrix_free_check': report['operator']['matrix_free_check'],
        'report': str(OUT / 'report.json')
    }, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
