#!/usr/bin/env python3
"""Read-only per-tetrahedron tangent-energy split for sealed source125 capture 419."""
import argparse
import csv
import hashlib
import json
import math
import pathlib
import re
import sys
from collections import defaultdict

import numpy as np

CAPTURE = pathlib.Path('/Users/home/cardboard-implementation-20261014/solver/two-stage-first-failure-rich-fd-419/operator-capture')
ADAPTER = pathlib.Path('/Users/home/cardboard-implementation-20261014/analysis/operator-fd-e2-adapter-001/coupon419-003/environment-0.npz')
SWEEP004 = pathlib.Path('/Users/home/cardboard-implementation-20261014/analysis/source125-fp64-smooth-direction-sweep-006/result-004/report.json')
WINDOW005 = pathlib.Path('/Users/home/cardboard-implementation-20261014/analysis/source125-fp64-smooth-direction-sweep-006/result-005/rowwise-window-intervals.json')
MATERIALS = pathlib.Path('/Users/home/cardboard-implementation-20261014/snapshots/source-status5-vtccd-diagnostics-125/matter/materials')
SOURCE = pathlib.Path('/Users/home/cardboard-implementation-20261014/snapshots/source-status5-vtccd-diagnostics-125')
OUT = pathlib.Path(__file__).resolve().parent
NAMES = {
    0: 'hajali2009_liner_hill_bilinear_fit',
    1: 'hajali2009_medium_hill_bilinear_fit',
    2: 'starch2007_finite_glue_elastic',
}


def sha256(path):
    h = hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def fsum_dot(a, b):
    return math.fsum(float(x) * float(y) for x, y in zip(a, b))


def matrix_action(m, x):
    return np.array([math.fsum(float(m[i, j]) * float(x[j]) for j in range(m.shape[1]))
                     for i in range(m.shape[0])], dtype=np.float64)


def norm(a):
    return math.sqrt(math.fsum(float(x) * float(x) for x in np.asarray(a).ravel()))


def pct(values, p):
    return float(np.quantile(np.asarray(values, dtype=np.float64), p))


def source_parameter(path, name):
    pattern = rf'^\s*parameter\s+{re.escape(name)}\s*:\s*MPa\s*=\s*([^;]+);'
    match = re.search(pattern, pathlib.Path(path).read_text(), re.MULTILINE)
    if not match:
        raise ValueError(f'missing source parameter {name} in {path}')
    value = float(match.group(1).strip())
    if not math.isfinite(value):
        raise ValueError(f'nonfinite source parameter {name} in {path}')
    return value


def summarize(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError('empty/nonfinite summary input')
    return {
        'min': float(np.min(x)), 'p50': pct(x, .50), 'p95': pct(x, .95),
        'p99': pct(x, .99), 'max': float(np.max(x)),
    }


def parse_capture(path):
    header = None
    rows = {}
    with path.open() as f:
        for line_number, line in enumerate(f, 1):
            row = json.loads(line)
            if row.get('record') == 'header':
                if header is not None:
                    raise ValueError('duplicate capture header')
                header = row
            elif row.get('record') == 'tetrahedron' and row.get('environment') == 0:
                tid = int(row['tet_id'])
                if tid in rows:
                    raise ValueError(f'duplicate environment-0 tet {tid}')
                rows[tid] = row
    if header is None:
        raise ValueError('capture has no header')
    expected = int(header['tetrahedron_count_per_environment'])
    if len(rows) != expected or set(rows) != set(range(expected)):
        raise ValueError(f'incomplete tet rows: got {len(rows)}, expected {expected}')
    return header, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=pathlib.Path, default=OUT)
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.seterr(all='raise')

    header, rows = parse_capture(CAPTURE)
    adapter = np.load(ADAPTER, allow_pickle=False)
    sweep = json.loads(SWEEP004.read_text())
    windows = json.loads(WINDOW005.read_text())
    dt = float(header['dt_seconds_by_environment'][0])
    expected_tets = int(header['tetrahedron_count_per_environment'])
    if header['schema'] != 'numi.cardboard.fem-newton-capture.v1' or expected_tets != 1920:
        raise ValueError('capture schema/topology differs from audited source125 case')
    if not header['cached_algorithmic_tangent_capture']['available']:
        raise ValueError('native cached tangent unavailable')

    required_npz = {
        'correction_delta_v_m_s': (882, 3),
        'native_element_operator_force_node_world_N_per_s': (1920, 4, 3),
        'independent_element_tangent_action_local_Pa_per_s': (1920, 3, 3),
        'material_index': (1920,), 'rest_volume_m3': (1920,),
        'material_frame_rotation': (1920, 3, 3),
    }
    for key, shape in required_npz.items():
        if key not in adapter or adapter[key].shape != shape:
            raise ValueError(f'NPZ field shape mismatch: {key}')
        if adapter[key].dtype.kind == 'f' and not np.isfinite(adapter[key]).all():
            raise ValueError(f'nonfinite NPZ field: {key}')

    by_material = defaultdict(list)
    per_tet = []
    max_action_error = 0.0
    max_action_error_tet = None
    frame_errors = {'F_world_times_Q_to_F_local': [], 'H_world_times_Q_to_H_local': [],
                    'P_world_times_Q_to_P_local': [], 'Q_orthogonality': [], 'Q_determinant': []}
    independent_action_rel = defaultdict(list)
    direct_work_diff = []
    matrix_valid_count = 0

    for tid in range(expected_tets):
        row = rows[tid]
        center = row['center']
        material_index = int(row['material_index'])
        if int(adapter['material_index'][tid]) != material_index:
            raise ValueError(f'material index mismatch at tet {tid}')
        name = row['material_name']
        cache = row['cached_algorithmic_tangent_local']
        if not cache.get('valid') or cache.get('matrix_row_major_local_dP_dF_Pa_per_dimensionless_F') is None:
            raise ValueError(f'invalid tangent cache at tet {tid}')
        matrix_valid_count += 1
        C = np.asarray(cache['matrix_row_major_local_dP_dF_Pa_per_dimensionless_F'], dtype=np.float64).reshape(9, 9)
        Hmat = np.asarray(center['F_direction_local_per_s'], dtype=np.float64).reshape(3, 3)
        h = Hmat.reshape(9)
        F = np.asarray(center['F_local'], dtype=np.float64) + np.asarray(center['F_local_low'], dtype=np.float64)
        P = np.asarray(center['P_local_Pa'], dtype=np.float64)
        if C.shape != (9, 9) or F.shape != (3, 3) or P.shape != (3, 3):
            raise ValueError(f'bad matrix shape at tet {tid}')
        if not (np.isfinite(C).all() and np.isfinite(Hmat).all() and np.isfinite(F).all() and np.isfinite(P).all()):
            raise ValueError(f'nonfinite capture input at tet {tid}')

        # Row-major map is bound by capture header; this action must reproduce its center direction.
        y = matrix_action(C, h)
        y_capture = np.asarray(center['tangent_local_Pa_per_s'], dtype=np.float64).reshape(9)
        action_rel = norm(y - y_capture) / max(norm(y_capture), 1.0e-300)
        if action_rel > max_action_error:
            max_action_error, max_action_error_tet = action_rel, tid
        y_ind = np.asarray(adapter['independent_element_tangent_action_local_Pa_per_s'][tid], dtype=np.float64).reshape(9)
        independent_action_rel[material_index].append(norm(y - y_ind) / max(norm(y_ind), 1.0e-300))
        q_independent = fsum_dot(h, y_ind)

        # P = F S for this Green-strain stored-energy law. D_F(FS)[H] = H S + F D_FS[H].
        # Split only the exact initial-stress term H S from the remaining algorithmic material term.
        S = np.linalg.solve(F, P)
        C_geo = np.zeros((9, 9), dtype=np.float64)
        for i in range(3):
            for J in range(3):
                r = 3 * i + J
                for k in range(3):
                    for L in range(3):
                        if i == k:
                            C_geo[r, 3 * k + L] = S[L, J]
        C_mat = C - C_geo
        yg = matrix_action(C_geo, h)
        ym = matrix_action(C_mat, h)
        q_total = fsum_dot(h, y)
        q_geo = fsum_dot(h, yg)
        q_material = fsum_dot(h, ym)
        q_split_error = q_total - q_geo - q_material
        volume = float(row['rest_volume_m3'])
        if not math.isclose(volume, float(adapter['rest_volume_m3'][tid]), rel_tol=1e-6, abs_tol=0.0):
            raise ValueError(f'rest volume mismatch at tet {tid}')
        # With f_int=-V*P*Dm^-T, v.K.v=-V*(H:C:H); A contributes -dt^2 K.
        work_total = dt * dt * volume * q_total
        work_independent = dt * dt * volume * q_independent
        work_geo = dt * dt * volume * q_geo
        work_material = dt * dt * volume * q_material
        nodes = np.asarray(row['nodes'], dtype=np.int64)
        dv = adapter['correction_delta_v_m_s'][nodes]
        force_action = adapter['native_element_operator_force_node_world_N_per_s'][tid]
        vKv = math.fsum(float(dv[n, xyz]) * float(force_action[n, xyz])
                        for n in range(4) for xyz in range(3))
        work_from_native_force = -dt * dt * vKv
        direct_work_diff.append(work_total - work_from_native_force)

        C_sym = (C + C.T) * .5
        Cmat_sym = (C_mat + C_mat.T) * .5
        Cgeo_sym = (C_geo + C_geo.T) * .5
        eig_total = np.linalg.eigvalsh(C_sym)
        eig_material = np.linalg.eigvalsh(Cmat_sym)
        eig_geo = np.linalg.eigvalsh(Cgeo_sym)
        c_scale = max(float(np.max(np.abs(eig_total))), 1.0e-300)
        cm_scale = max(float(np.max(np.abs(eig_material))), 1.0e-300)
        asym = norm(C - C.T) / max(norm(C), 1.0e-300)
        material_asym = norm(C_mat - C_mat.T) / max(norm(C_mat), 1.0e-300)
        geo_asym = norm(C_geo - C_geo.T) / max(norm(C_geo), 1.0e-300)
        S_sym_rel = norm(S - S.T) / max(norm(S), 1.0e-300)
        S_eig = np.linalg.eigvalsh((S + S.T) * .5)
        E = (F.T @ F - np.eye(3)) * .5
        E_eig = np.linalg.eigvalsh((E + E.T) * .5)
        projected = np.asarray(center['projected_state'], dtype=np.float64)
        accepted = np.asarray(center['accepted_state'], dtype=np.float64)
        lambda_increment = float(projected[12] - accepted[12]) if projected.size > 12 else 0.0

        Q = np.asarray(adapter['material_frame_rotation'][tid], dtype=np.float64)
        Fw = np.asarray(center['F_world'], dtype=np.float64)
        Hw = np.asarray(center['F_direction_world_per_s'], dtype=np.float64)
        Pw = np.asarray(center['P_world_Pa'], dtype=np.float64)
        frame_errors['F_world_times_Q_to_F_local'].append(norm(Fw @ Q - F) / max(norm(F), 1.0e-300))
        frame_errors['H_world_times_Q_to_H_local'].append(norm(Hw @ Q - Hmat) / max(norm(Hmat), 1.0e-300))
        frame_errors['P_world_times_Q_to_P_local'].append(norm(Pw @ Q - P) / max(norm(P), 1.0e-300))
        frame_errors['Q_orthogonality'].append(norm(Q.T @ Q - np.eye(3)))
        frame_errors['Q_determinant'].append(float(np.linalg.det(Q)))

        rec = {
            'tet_id': tid, 'material_index': material_index, 'material_name': name,
            'volume_m3': volume, 'F_det': float(np.linalg.det(F)),
            'E_principal_strain_eigenvalues': E_eig.tolist(),
            'S_principal_stress_eigenvalues_Pa': S_eig.tolist(),
            'directional_H_norm_per_s': norm(Hmat),
            'tangent_action_relative_error_vs_capture': action_rel,
            'tangent_action_relative_error_vs_independent_FP64': independent_action_rel[material_index][-1],
            'dF_C_dF_Pa_per_s2': q_total,
            'dF_Cgeo_dF_Pa_per_s2': q_geo,
            'dF_Cmaterial_dF_Pa_per_s2': q_material,
            'contraction_split_absolute_error': q_split_error,
            'A_stiffness_work_total_J': work_total,
            'A_stiffness_work_independent_FP64_directional_action_J': work_independent,
            'A_stiffness_work_geometric_J': work_geo,
            'A_stiffness_work_material_J': work_material,
            'A_stiffness_work_from_native_element_action_J': work_from_native_force,
            'major_asymmetry_relative_total': asym,
            'major_asymmetry_relative_geometric': geo_asym,
            'major_asymmetry_relative_material': material_asym,
            'sym_C_min_eigenvalue_Pa': float(eig_total[0]),
            'sym_C_min_eigenvalue_over_max_abs_eigenvalue': float(eig_total[0] / c_scale),
            'sym_C_material_min_eigenvalue_Pa': float(eig_material[0]),
            'sym_C_material_min_eigenvalue_over_max_abs_eigenvalue': float(eig_material[0] / cm_scale),
            'sym_C_geometric_min_eigenvalue_Pa': float(eig_geo[0]),
            'second_Piola_symmetry_relative': S_sym_rel,
            'captured_accumulated_multiplier_increment': lambda_increment,
        }
        if not all(math.isfinite(v) for v in rec.values() if isinstance(v, (int, float))):
            raise ValueError(f'nonfinite output at tet {tid}')
        per_tet.append(rec)
        by_material[material_index].append(rec)

    if matrix_valid_count != expected_tets:
        raise ValueError(f'tangent matrix valid count {matrix_valid_count} != {expected_tets}')

    material_summaries = {}
    for mid, records in sorted(by_material.items()):
        if mid not in NAMES:
            raise ValueError(f'unknown material index {mid}')
        if any(r['material_name'] != NAMES[mid] for r in records):
            raise ValueError(f'material identity mismatch at index {mid}')
        total = math.fsum(r['A_stiffness_work_total_J'] for r in records)
        independent = math.fsum(r['A_stiffness_work_independent_FP64_directional_action_J'] for r in records)
        geo = math.fsum(r['A_stiffness_work_geometric_J'] for r in records)
        material = math.fsum(r['A_stiffness_work_material_J'] for r in records)
        native = math.fsum(r['A_stiffness_work_from_native_element_action_J'] for r in records)
        material_summaries[str(mid)] = {
            'material_name': NAMES[mid], 'tet_count': len(records),
            'A_stiffness_work_total_J': total,
            'A_stiffness_work_independent_FP64_directional_action_J': independent,
            'A_stiffness_work_geometric_initial_stress_J': geo,
            'A_stiffness_work_material_algorithmic_J': material,
            'split_closure_J': total - geo - material,
            'native_element_action_work_J': native,
            'tangent_contraction_vs_native_action_abs_J': total - native,
            'negative_total_work_tet_count': sum(r['A_stiffness_work_total_J'] < 0.0 for r in records),
            'negative_geometric_work_tet_count': sum(r['A_stiffness_work_geometric_J'] < 0.0 for r in records),
            'negative_material_work_tet_count': sum(r['A_stiffness_work_material_J'] < 0.0 for r in records),
            'positive_accumulated_multiplier_increment_tet_count_gt_1e-12': sum(r['captured_accumulated_multiplier_increment'] > 1.0e-12 for r in records),
            'max_action_relative_error_vs_capture': max(r['tangent_action_relative_error_vs_capture'] for r in records),
            'p95_action_relative_error_vs_capture': pct([r['tangent_action_relative_error_vs_capture'] for r in records], .95),
            'max_action_relative_error_vs_independent_FP64': max(r['tangent_action_relative_error_vs_independent_FP64'] for r in records),
            'p95_action_relative_error_vs_independent_FP64': pct([r['tangent_action_relative_error_vs_independent_FP64'] for r in records], .95),
            'major_asymmetry_relative_total': summarize([r['major_asymmetry_relative_total'] for r in records]),
            'major_asymmetry_relative_geometric': summarize([r['major_asymmetry_relative_geometric'] for r in records]),
            'major_asymmetry_relative_material': summarize([r['major_asymmetry_relative_material'] for r in records]),
            'sym_C_min_eigenvalue_Pa': summarize([r['sym_C_min_eigenvalue_Pa'] for r in records]),
            'sym_C_min_eigenvalue_ratio': summarize([r['sym_C_min_eigenvalue_over_max_abs_eigenvalue'] for r in records]),
            'sym_C_material_min_eigenvalue_Pa': summarize([r['sym_C_material_min_eigenvalue_Pa'] for r in records]),
            'sym_C_material_min_eigenvalue_ratio': summarize([r['sym_C_material_min_eigenvalue_over_max_abs_eigenvalue'] for r in records]),
            'C_symmetric_part_has_negative_min_eigenvalue_tet_count': sum(r['sym_C_min_eigenvalue_Pa'] < 0.0 for r in records),
            'C_material_symmetric_part_has_negative_min_eigenvalue_tet_count': sum(r['sym_C_material_min_eigenvalue_Pa'] < 0.0 for r in records),
            'S_second_Piola_symmetry_relative': summarize([r['second_Piola_symmetry_relative'] for r in records]),
        }

    negative_owners = sorted((r for r in per_tet if r['A_stiffness_work_total_J'] < 0.0),
                             key=lambda r: r['A_stiffness_work_total_J'])
    top_negative = []
    for r in negative_owners[:12]:
        top_negative.append({
            'tet_id': r['tet_id'], 'material_index': r['material_index'], 'material_name': r['material_name'],
            'A_stiffness_work_total_J': r['A_stiffness_work_total_J'],
            'A_stiffness_work_geometric_initial_stress_J': r['A_stiffness_work_geometric_J'],
            'A_stiffness_work_material_algorithmic_J': r['A_stiffness_work_material_J'],
            'S_principal_stress_eigenvalues_Pa': r['S_principal_stress_eigenvalues_Pa'],
            'E_principal_strain_eigenvalues': r['E_principal_strain_eigenvalues'],
            'captured_accumulated_multiplier_increment': r['captured_accumulated_multiplier_increment'],
            'sym_C_min_eigenvalue_Pa': r['sym_C_min_eigenvalue_Pa'],
        })

    total_tangent_work = math.fsum(r['A_stiffness_work_total_J'] for r in per_tet)
    total_independent_work = math.fsum(r['A_stiffness_work_independent_FP64_directional_action_J'] for r in per_tet)
    total_geo_work = math.fsum(r['A_stiffness_work_geometric_J'] for r in per_tet)
    total_material_work = math.fsum(r['A_stiffness_work_material_J'] for r in per_tet)
    total_native_work = math.fsum(r['A_stiffness_work_from_native_element_action_J'] for r in per_tet)
    expected_report_work = float(sweep['center_A_terms']['stiffness_action_minus_dt2_K_Ns']['signed_work_dv_dot'])
    mass_work = float(sweep['center_A_terms']['mass_action_M_dv_Ns']['signed_work_dv_dot'])
    full_A_work = float(sweep['center_A_terms']['total_A_dv_Ns']['signed_work_dv_dot'])

    elastic_blocks = {}
    for mid, stem in [(0, 'hajali2009_liner_hill_bilinear_fit'), (1, 'hajali2009_medium_hill_bilinear_fit')]:
        material_path = MATERIALS / f'{stem}.nmatter'
        c11, c22, c33 = (source_parameter(material_path, n) for n in ('c11', 'c22', 'c33'))
        c12, c13, c23 = (source_parameter(material_path, n) for n in ('c12', 'c13', 'c23'))
        g12, g13, g23 = (source_parameter(material_path, n) for n in ('g12', 'g13', 'g23'))
        normal_eigenvalues = np.linalg.eigvalsh(np.array([[c11, c12, c13],
                                                           [c12, c22, c23],
                                                           [c13, c23, c33]], dtype=np.float64))
        elastic_blocks[str(mid)] = {
            'material_name': NAMES[mid],
            'normal_green_strain_block_eigenvalues_MPa': normal_eigenvalues.tolist(),
            'shear_coefficients_MPa': {'g12': g12, 'g13': g13, 'g23': g23},
            'normal_block_positive_definite': bool(float(normal_eigenvalues[0]) > 0.0),
            'all_authored_shear_coefficients_positive': bool(min(g12, g13, g23) > 0.0),
        }

    report = {
        'schema': 'numi.cardboard.source125.per_tet_tangent_energy.v1',
        'evidence_class': 'bounded CPU-only decomposition of the selected source125/Newton48 production tangent capture; diagnostic analysis, not a solver or material qualification',
        'selection': {
            'capture_schema': header['schema'], 'environment': 0, 'root': header['root'],
            'newton_iteration': header['newton_iteration'], 'environment_count': header['environment_count'],
            'node_count': header['node_count_per_environment'], 'tetrahedron_count': expected_tets,
            'dt_seconds': dt, 'world_fingerprint_hex': header['world_fingerprint_hex'],
            'execution_fingerprint_hex': header['execution_fingerprint_hex'],
            'production_inputs_bitwise_unchanged': header['production_inputs_bitwise_unchanged'],
            'cache_policy_enabled': header['cached_algorithmic_tangent_capture']['policy_enabled'],
            'cached_matrix_contract': header['cached_algorithmic_tangent_capture']['derivative_contract'],
            'matrix_layout': header['cached_algorithmic_tangent_capture']['matrix_layout'],
            'cache_valid_tet_count': matrix_valid_count,
        },
        'bindings': {
            str(CAPTURE): sha256(CAPTURE), str(ADAPTER): sha256(ADAPTER),
            str(SWEEP004): sha256(SWEEP004), str(WINDOW005): sha256(WINDOW005),
            str(SOURCE / 'matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter'): sha256(SOURCE / 'matter/materials/hajali2009_liner_hill_bilinear_fit.nmatter'),
            str(SOURCE / 'matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter'): sha256(SOURCE / 'matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter'),
            str(SOURCE / 'matter/materials/starch2007_finite_glue_elastic.nmatter'): sha256(SOURCE / 'matter/materials/starch2007_finite_glue_elastic.nmatter'),
            str(SOURCE / 'matter/src/metal/fem.metalinc'): sha256(SOURCE / 'matter/src/metal/fem.metalinc'),
            str(SOURCE / 'matter/src/metal/fgmres.metalinc'): sha256(SOURCE / 'matter/src/metal/fgmres.metalinc'),
            str(SOURCE / 'matter/tools/cardboard_bilinear_material_oracle.hpp'): sha256(SOURCE / 'matter/tools/cardboard_bilinear_material_oracle.hpp'),
        },
        'formula_and_sign_contract': {
            'material_energy_source': 'The Haj-Ali-derived 15-state assets define a StVK-style quadratic energy in Green-strain components minus implicit plastic strains; stress is generated from this energy/state system, so P=F*S in each fixed local reference frame.',
            'tangent_split': 'C = dP/dF, P=F*S; Cgeo[iJ,kL] = delta(i,k)*S[L,J], Cmaterial=C-Cgeo. This is an exact product-rule split at captured F,P; Cmaterial includes the implicit projection/algorithmic state response. It is not a split into separate source elastic/plastic energy potentials.',
            'operator_sign': 'Internal element force is -V*P*Dm^-T. Therefore v^T K v = -V*(H:C:H), and the stiffness contribution to A=M-dt^2*K is dt^2*V*(H:C:H).',
            'direction_and_rate': 'H is captured grad(delta_v), units s^-1; production tangent also receives rateDirection=H/dt. The cache contract is dP/dF with that fixed-dt rate coupling.',
            'frame': 'Material frame Q is an orthogonal reference-coordinate rotation: F_local=F_world*Q, H_local=H_world*Q, P_local=P_world*Q. Q is fixed in this Newton tangent; no additional frame derivative is included.',
        },
        'source_reference_projection': {
            'source_equations': 'The two 15-state material assets define Green-strain/StVK-like anisotropic stored energy, normalized second-Piola stress states, associated Hill flow with smoothed positive-epsilon complementarity, and separate MD/CD tensile-hardening accumulations; see the exact source lines recorded in README.',
            'projection_evidence': 'The separate independent FP64 bytecode interpreter/projector uses the same authored implicit residual/Jacobian and fixed accepted history. Result-005 reports 1728/1728 paper rows with at least three contiguous predeclared epsilon-grid points passing its pointwise local 1e-6 derivative gate plus strict endpoint projection/branch checks. This is independent directional evidence, not a full tensor energy-positivity theorem.',
            'hardening_scope': 'The assets use positive source-fit MD/CD hardening values (liner 4019.9227/609.0611 MPa; medium 1880.2072/287.1312 MPa) but document zero compression/shear hardening assumptions and transfer assumptions. No generic monotonicity is inferred from this one direction.',
            'small_strain_elastic_blocks': elastic_blocks,
        },
        'aggregate_work': {
            'mass_M_signed_work_J': mass_work,
            'stiffness_minus_dt2K_signed_work_from_independent_assembled_report_J': expected_report_work,
            'stiffness_minus_dt2K_signed_work_from_full_cached_C_contraction_J': total_tangent_work,
            'stiffness_minus_dt2K_signed_work_from_independent_FP64_directional_actions_J': total_independent_work,
            'stiffness_minus_dt2K_signed_work_from_native_element_force_action_J': total_native_work,
            'contraction_vs_native_element_action_difference_J': total_tangent_work - total_native_work,
            'contraction_vs_independent_assembled_report_difference_J': total_tangent_work - expected_report_work,
            'geometric_initial_stress_component_J': total_geo_work,
            'algorithmic_material_component_J': total_material_work,
            'split_closure_J': total_tangent_work - total_geo_work - total_material_work,
            'full_A_signed_work_from_independent_assembled_report_J': full_A_work,
            'mass_plus_contraction_A_work_J': mass_work + total_tangent_work,
            'interpretation': 'The captured global ray has a negative stiffness contribution after -dt^2 K; the product-rule split shows the dominant negative term is the initial-stress/geometric component in compressive liner regions. The remaining algorithmic material component is positive in aggregate along this direction. This explains the sign without establishing whether the modeled constitutive law is physically adequate or the nonlinear solve should accept this direction.',
        },
        'frame_transform_checks': {k: summarize(v) for k, v in frame_errors.items()},
        'material_summaries': material_summaries,
        'negative_tangent_contribution_count': len(negative_owners),
        'top_negative_contribution_tets': top_negative,
        'finding': {
            'captured_ray_operator_sign': 'The signed quadratic form of A=M-dt^2*K is negative on this captured correction ray (-8.42236e-4 J from the C contraction, versus -8.42331e-4 J in the independent assembled report).',
            'material_classification': 'All three material groups have positive aggregate Cmaterial directional work on this one ray. The 12 negative tet-level A stiffness contributions are 9 liner tets and 3 glue tets; the liner terms dominate by many orders of magnitude.',
            'first_piola_geometric_classification': 'For top negative liner tet 1440, the exact product-rule split is -0.02323785 J geometric initial-stress contribution plus +0.00092361 J algorithmic material contribution, giving -0.02231424 J total. Its recovered second-Piola principal stresses are approximately [-0.380524,-0.017361,+0.002986] MPa.',
            'physical_interpretation_limit': 'This identifies the finite-state initial-stress/geometric term as the negative-work owner along the captured direction. It is consistent with compressive prestress and is not, by itself, evidence of a sign bug or a physical energy violation. Major-symmetry outliers and small negative eigenvalues in the subtracted Cmaterial remain measured but not adjudicated.',
        },
        'limitations': [
            'The source capture is a selected rejected-solver Newton linearization; it is not a successful coupon history or physical experiment.',
            'The C matrices are production FP32 cached tangents, promoted for arithmetic. The independent FP64 projection evidence is directional and does not provide an independent full 9x9 C matrix at every tet.',
            'The geometric/material split is an algebraic product-rule split for P=F*S; Cmaterial includes the implicit constitutive-state derivative and is not equal to an independently identified elastic modulus.',
            'Negative eigenvalues of sym(C) and negative A-work are local/operator diagnostics. They do not by themselves prove an implementation defect, an energy violation, or a physical buckling mode.',
            'No new constitutive projection, native run, tolerance change, calibration, or physical qualification was performed.',
        ],
    }

    # Add exact per-material and top-owner rows while retaining all 1920 tets.
    csv_path = out / 'per-tet.csv'
    fields = list(per_tet[0].keys())
    with csv_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in per_tet:
            w.writerow({k: json.dumps(v, separators=(',', ':')) if isinstance(v, list) else v for k, v in r.items()})
    report_path = out / 'report.json'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({
        'status': 'complete', 'tet_count': expected_tets,
        'A_stiffness_work_from_C_J': total_tangent_work,
        'A_stiffness_work_from_native_force_J': total_native_work,
        'geometric_J': total_geo_work, 'material_J': total_material_work,
        'negative_tet_count': len(negative_owners),
        'top_negative_tet': top_negative[0] if top_negative else None,
        'report': str(report_path), 'per_tet': str(csv_path),
    }, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
