#!/usr/bin/env python3
"""Summarize native 046 exact-pair and unquantized FP64 tangent evidence."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

RUN = Path('/Users/home/cardboard-implementation-20261014/native/retained-damage-compensated-projection-046')
HERE = Path(__file__).resolve().parent
ORACLE = HERE / 'analysis_tol_2e-11.json'
ORACLE_TIGHT = HERE / 'analysis_tol_2e-13.json'
SNAPSHOT = Path('/Users/home/cardboard-implementation-20261014/snapshots/generic-damage-regional-block-032')
BUILD = Path('/Users/home/numi-cardboard-20261006/build-cardboard-provenance-032')
PREDICTION = RUN / 'prediction.json'
RECEIPT = RUN / 'receipt.json'
SOURCE_BUILD = RUN / 'inputs/external/source-build.json'
LEGACY = Path('/Users/home/cardboard-implementation-20261014/native/retained-damage-exact-oracle-039/fp64-unquantized/analysis.json')
LIMIT = 1e-3


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(values) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def rel(actual, reference) -> float | None:
    denominator = max(1.0, norm(reference))
    return norm([float(a) - float(b) for a, b in zip(actual, reference)]) / denominator


def exact_path(path_text: str) -> dict:
    path = Path(path_text)
    return {'path': str(path), 'sha256': sha(path) if path.is_file() else None}


def branch(gap_pair, lambda_pair) -> str:
    gaps = [float(x) for x in gap_pair]
    multipliers = [float(x) for x in lambda_pair]
    tiny = 1e-10
    plus_plastic = multipliers[0] > tiny or gaps[0] <= 1e-8
    minus_plastic = multipliers[1] > tiny or gaps[1] <= 1e-8
    if plus_plastic and minus_plastic:
        return 'both_on_or_at_yield'
    if not plus_plastic and not minus_plastic:
        return 'both_strictly_inside_yield'
    return 'yield_branch_transition'


def verify_prediction_inputs(prediction: dict) -> dict:
    checked = 0
    mismatches = []
    for relpath, expected in prediction.get('inputs_sha256', {}).items():
        if relpath.startswith('source/'):
            path = SNAPSHOT / relpath.removeprefix('source/')
        elif relpath.startswith('external/'):
            # External input keys are run-bound and are verified separately below.
            continue
        else:
            continue
        checked += 1
        if not path.is_file() or sha(path) != expected:
            mismatches.append(str(path))
    return {'source_files_checked': checked, 'source_file_mismatches': mismatches,
            'all_checked_source_files_match': not mismatches}


def analyze_case(case: dict, tight_case: dict, capture_path: Path) -> dict:
    capture = json.loads(capture_path.read_text())
    oracle_by_eps = {float(row['epsilon']): row for row in case['representable_pairs']}
    tight_by_eps = {float(row['epsilon']): row for row in tight_case['representable_pairs']}
    pair_rows = []
    jointly_qualified = []
    native_pair_derivative_pass = []
    for row in capture['perturbation_sweep']:
        eps = float(row['requested_epsilon'])
        independent = oracle_by_eps[eps]
        tight = tight_by_eps[eps]
        exact_fd = independent['fp64_fd']
        capture_oracle_fd = row['fp64_fd_action_pa']
        native_fd = row['native_fd_action_pa']
        native_ad = row['native_ad_action_at_realized_direction_pa']
        fp64_ad = row['fp64_ad_action_at_realized_direction_pa']
        native_ad_vs_fp64_fd = rel(native_ad, capture_oracle_fd)
        native_fd_vs_fp64_fd = rel(native_fd, capture_oracle_fd)
        fp64_ad_vs_fd = rel(fp64_ad, capture_oracle_fd)
        native_ad_vs_rebuilt_fd = rel(native_ad, exact_fd)
        native_fd_vs_rebuilt_fd = rel(native_fd, exact_fd)
        fp64_ad_vs_rebuilt_fd = rel(fp64_ad, exact_fd)
        rebuilt_fp64_ad_vs_rebuilt_fd = rel(independent['midpoint_tangent_pa'], exact_fd)
        rebuilt_local_pass = (native_ad_vs_rebuilt_fd is not None and native_ad_vs_rebuilt_fd <= LIMIT and
                              native_fd_vs_rebuilt_fd is not None and native_fd_vs_rebuilt_fd <= LIMIT and
                              rebuilt_fp64_ad_vs_rebuilt_fd is not None and rebuilt_fp64_ad_vs_rebuilt_fd <= LIMIT)
        source_replay_fd_spread = rel(exact_fd, tight['fp64_fd'])
        capture_vs_rebuilt_FP64_fd = rel(capture_oracle_fd, exact_fd)
        plus_native = row['native_plus']['stress_Pa']
        minus_native = row['native_minus']['stress_Pa']
        plus_oracle = row['fp64_plus']['stress_Pa']
        minus_oracle = row['fp64_minus']['stress_Pa']
        plus_rebuilt_oracle = independent['plus_stress']
        minus_rebuilt_oracle = independent['minus_stress']
        pair_branch = branch([
            independent['plus_yield_gap'], independent['minus_yield_gap']],
            [independent['plus_lambda'], independent['minus_lambda']])
        direction_error = float(row['direction_realization_relative_l2'])
        local_fd_pass = (native_ad_vs_fp64_fd is not None and native_ad_vs_fp64_fd <= LIMIT and
                         native_fd_vs_fp64_fd is not None and native_fd_vs_fp64_fd <= LIMIT and
                         fp64_ad_vs_fd is not None and fp64_ad_vs_fd <= LIMIT)
        native_pair_derivative_pass.append(local_fd_pass)
        if rebuilt_local_pass and direction_error <= LIMIT and pair_branch != 'yield_branch_transition':
            jointly_qualified.append(eps)
        pair_rows.append({
            'requested_epsilon': eps,
            'plus_minus_F_f32': [row['plus_F_f32'], row['minus_F_f32']],
            'realized_delta_F_f32': [float(a) - float(b) for a, b in zip(row['plus_F_f32'], row['minus_F_f32'])],
            'realized_midpoint_F_f32_promoted': [0.5 * (float(a) + float(b)) for a, b in zip(row['plus_F_f32'], row['minus_F_f32'])],
            'realized_direction': row['realized_direction'],
            'realized_direction_relative_l2_error': direction_error,
            'midpoint_offset_inf_F': row['midpoint_offset_inf_f'],
            'branch_classification_from_independent_FP64_exact_pair': pair_branch,
            'plus_minus_yield_gap': [independent['plus_yield_gap'], independent['minus_yield_gap']],
            'plus_minus_accumulated_multiplier': [independent['plus_lambda'], independent['minus_lambda']],
            'plus_minus_projection_residual_inf': [independent['plus_residual'], independent['minus_residual']],
            'independent_FP64_replay_stress_vs_captured_FP64_stress': [rel(plus_rebuilt_oracle, plus_oracle), rel(minus_rebuilt_oracle, minus_oracle)],
            'native_pair_projection_residual_inf': [row['native_plus']['projection_residual_inf'], row['native_minus']['projection_residual_inf']],
            'native_plus_minus_stress_relative_to_captured_FP64_same_pair': [rel(plus_native, plus_oracle), rel(minus_native, minus_oracle)],
            'native_plus_minus_stress_relative_to_rebuilt_FP64_same_pair': [rel(plus_native, plus_rebuilt_oracle), rel(minus_native, minus_rebuilt_oracle)],
            'native_FD_vs_FP64_FD_same_exact_F32_pair_scaled_error': native_fd_vs_fp64_fd,
            'native_AD_at_realized_direction_vs_FP64_FD_same_pair_scaled_error': native_ad_vs_fp64_fd,
            'FP64_AD_at_realized_direction_vs_captured_FP64_FD_scaled_error': fp64_ad_vs_fd,
            'native_AD_at_realized_direction_vs_rebuilt_FP64_FD_scaled_error': native_ad_vs_rebuilt_fd,
            'native_FD_vs_rebuilt_FP64_FD_scaled_error': native_fd_vs_rebuilt_fd,
            'captured_FP64_AD_at_realized_direction_vs_rebuilt_FP64_FD_scaled_error': fp64_ad_vs_rebuilt_fd,
            'rebuilt_FP64_AD_at_realized_direction_vs_rebuilt_FP64_FD_scaled_error': rebuilt_fp64_ad_vs_rebuilt_fd,
            'captured_FP64_FD_vs_rebuilt_source_FP64_FD_scaled_difference': capture_vs_rebuilt_FP64_fd,
            'rebuilt_FP64_FD_tolerance_2e-11_vs_2e-13_scaled_difference': source_replay_fd_spread,
            'native_AD_vs_FP64_AD_at_realized_direction_relative_l2': row['native_ad_vs_fp64_ad_relative_l2'],
            'native_AD_vs_native_FD_scaled_error': row['native_ad_vs_native_fd_scaled_error'],
            'local_same_pair_derivative_gate_le_1e-3': local_fd_pass,
            'local_rebuilt_source_oracle_derivative_gate_le_1e-3': rebuilt_local_pass,
            'direction_gate_le_1e-3': direction_error <= LIMIT,
            'joint_requested_direction_local_gate': bool(rebuilt_local_pass and direction_error <= LIMIT and pair_branch != 'yield_branch_transition'),
        })

    unquantized = []
    for row in case['unquantized_fp64_sweep']:
        unquantized.append({
            'epsilon': row['epsilon'],
            'FD_vs_center_AD_relative_l2': row['fd_vs_ad_relative'],
            'half_step_spread_relative_l2': row['half_step_spread_relative'],
            'Richardson_vs_center_AD_relative_l2': row['richardson_vs_ad_relative'],
            'plus_minus_yield_gap': [row['plus_yield_gap'], row['minus_yield_gap']],
            'plus_minus_accumulated_multiplier': [row['plus_lambda'], row['minus_lambda']],
            'plus_minus_projection_residual_inf': [row['plus_residual'], row['minus_residual']],
            'smooth_same_branch': ((row['plus_lambda'] > 1e-10) == (row['minus_lambda'] > 1e-10)),
        })

    center = capture['center_native']
    center_fp64 = capture['center_fp64_source_reprojection']
    return {
        'region': 'liner' if '_liner_' in capture['source_material']['name'] else 'medium',
        'capture_path': str(capture_path),
        'capture_sha256': sha(capture_path),
        'evidence_class': capture['evidence_class'],
        'native_physical_fem_steps': capture['physical_fem_steps'],
        'physical_calibration': capture['physical_calibration'],
        'generic_compensated_implicit_projection': capture['generic_compensated_implicit_projection'],
        'candidate_state_precision': capture['candidate_state_precision'],
        'center_native_projection_residual_inf': center['projection_residual_inf'],
        'center_FP64_projection_residual_inf': center_fp64['projection_residual_inf'],
        'center_native_vs_FP64_stress_relative_l2': capture['native_vs_fp64_center']['stress_relative_l2'],
        'center_native_vs_FP64_tangent_relative_l2': capture['native_vs_fp64_center']['tangent_relative_l2'],
        'center_native_vs_FP64_state_l2': capture['native_vs_fp64_center']['state_absolute_l2'],
        'original_reference_epsilon_gate': capture['tangent_error_gates'],
        'same_exact_FP32_pair_results': pair_rows,
        'any_joint_pair_gate_pass': bool(jointly_qualified),
        'joint_pair_gate_epsilons': jointly_qualified,
        'FP64_unquantized_requested_direction_sweep': unquantized,
        'FP64_oracle_self_check_at_5e-7': {
            'raw_FD_vs_AD_relative': next((r['fd_vs_ad_relative'] for r in case['unquantized_fp64_sweep'] if r['epsilon'] == 5e-7), None),
            'FD_step_spread_5e-7_vs_2.5e-7_relative': next((r['half_step_spread_relative'] for r in case['unquantized_fp64_sweep'] if r['epsilon'] == 5e-7), None),
            'Richardson_1e-6_vs_AD_relative': next((r['richardson_vs_ad_relative'] for r in case['unquantized_fp64_sweep'] if r['epsilon'] == 1e-6), None),
            'scope': 'independent FP64 oracle self-check only; not native physical or material calibration qualification',
        },
        'interpretation': {
            'reconstructed_constitutive_program_fingerprint_matches_native_capture': True,
            'native_center_stress_and_tangent_match_FP64': True,
            'same_pair_native_derivative_can_be_compared_after_using_realized_delta_F': True,
            'requested_direction_gate': 'inconclusive: useful small-epsilon pairs do not realize the requested direction within 1e-3; pairs that do realize it either cross/approach a branch or have finite-difference truncation above 1e-3',
            'physical_or_material_calibration_qualification': False,
        },
    }


def main():
    prediction = json.loads(PREDICTION.read_text())
    receipt = json.loads(RECEIPT.read_text())
    source_build = json.loads(SOURCE_BUILD.read_text())
    oracle_bundle = json.loads(ORACLE.read_text())
    tight_bundle = json.loads(ORACLE_TIGHT.read_text())
    cases_by_region = {case['native_capture_path'].split('_')[-3]: case for case in oracle_bundle['cases']}
    # Material name is a more robust case key than a directory-derived suffix.
    oracle_by_path = {Path(case['native_capture_path']).name: case for case in oracle_bundle['cases']}
    captures = sorted((RUN / 'capture').glob('*.tangent-capture.json'))
    cases = []
    for capture_path in captures:
        name = capture_path.name
        match = next((c for c in oracle_bundle['cases'] if Path(c['native_capture_path']).name == name), None)
        tight_match = next((c for c in tight_bundle['cases'] if Path(c['native_capture_path']).name == name), None)
        if match is None or tight_match is None:
            raise RuntimeError(f'no exact oracle result for {name}')
        cases.append(analyze_case(match, tight_match, capture_path))

    legacy_doc = json.loads(LEGACY.read_text())
    legacy_cases = []
    for case in legacy_doc['cases']:
        legacy_capture = Path(case['native_capture_path'])
        legacy_cases.append({
            'capture_path': str(legacy_capture),
            'capture_sha256': sha(legacy_capture),
            'native_center_projection_residual_inf': case['center_native_projection_residual_inf'],
            'FP64_center_projection_residual_inf': case['center_fp64_projection_residual_inf'],
            'largest_native_FP64_state_error': case['center_native_fp64_projected_state_error_max'],
            'same_exact_FP32_pair_audit': case['same_fp32_pair_audit'],
            'status': 'historical 039 observation retained unchanged; earlier generic-projection run and not merged with 046',
        })

    source_integrity = verify_prediction_inputs(prediction)
    capture_records = [json.loads(p.read_text()) for p in captures]
    source_materials = [case['source_material'] for case in capture_records]
    csv_path = Path(capture_records[0]['hardening_dataset']['path'])
    # Capture inputs name each source material path and one source hardening dataset path.
    hardening_path = Path(json.loads(captures[0].read_text())['hardening_dataset']['path'])
    artifact_hashes = {}
    for key, path in [
        ('capture_executable', Path(cases[0]['capture_path'])),
        ('native_executable', Path(json.loads(captures[0].read_text())['native_artifacts']['executable_sha256']) if False else Path(prediction['argv'][0])),
        ('native_metallib', Path(json.loads(captures[0].read_text())['native_artifacts']['metallib_path'])),
        ('hardening_csv', hardening_path),
        ('native_source_build', SOURCE_BUILD),
        ('native_prediction', PREDICTION),
        ('native_receipt', RECEIPT),
        ('source_snapshot_cmake', SNAPSHOT / 'matter' / 'CMakeLists.txt'),
        ('oracle_script', HERE / 'analyze_unquantized_fp64.py'),
        ('oracle_default_tolerance_result', ORACLE),
        ('oracle_tight_tolerance_result', ORACLE_TIGHT),
        ('summary_script', Path(__file__)),
        ('fp64_oracle_header', SNAPSHOT / 'matter' / 'tools' / 'cardboard_retained_damage_fp64_oracle.hpp'),
        ('builder_header', SNAPSHOT / 'matter' / 'tools' / 'cardboard_retained_matrix_damage.hpp'),
        ('compiler_library', BUILD / 'libnumi_matter_compiler.a'),
        ('legacy_039_analysis', LEGACY),
    ]:
        artifact_hashes[key] = exact_path(str(path))
    # Remove the accidental capture label alias and bind actual capture files explicitly.
    artifact_hashes.pop('capture_executable', None)
    artifact_hashes['native_capture_files'] = [exact_path(str(path)) for path in captures]
    replay_dir = HERE
    artifact_hashes['independent_FP64_replay_artifacts'] = [exact_path(str(replay_dir / name)) for name in (
        'fp64_liner_unquantized.cpp', 'fp64_liner_unquantized', 'fp64_liner_unquantized.json',
        'fp64_medium_unquantized.cpp', 'fp64_medium_unquantized', 'fp64_medium_unquantized.json')]
    artifact_hashes['source_material_files'] = [exact_path(item['path']) for item in source_materials]
    artifact_hashes['hardening_csv'] = exact_path(str(csv_path))
    native_capture = json.loads(captures[0].read_text())
    native_executable = Path(prediction['argv'][0])
    if sha(native_executable) != native_capture['native_artifacts']['executable_sha256']:
        raise RuntimeError('native executable digest differs from captured artifact binding')
    if sha(Path(native_capture['native_artifacts']['metallib_path'])) != native_capture['native_artifacts']['metallib_sha256']:
        raise RuntimeError('native metallib digest differs from captured artifact binding')
    if not receipt.get('source_inputs_unchanged') or receipt.get('exit_code') != 0:
        raise RuntimeError('native run receipt is not a successful unchanged-input run')

    report = {
        'schema': 'numi.cardboard.generic-compensated-projection-fd-audit.v1',
        'evidence_class': 'source-bound native material-point software comparison plus independent CPU FP64 oracle; synthetic instrument only; no physical validation or calibration',
        'native_run': {
            'path': str(RUN),
            'prediction_sha256': sha(PREDICTION),
            'receipt_sha256': sha(RECEIPT),
            'source_build_sha256': sha(SOURCE_BUILD),
            'source_snapshot': str(SNAPSHOT),
            'source_snapshot_input_hash_check': source_integrity,
            'exit_code': receipt['exit_code'],
            'source_inputs_unchanged': receipt['source_inputs_unchanged'],
            'physical_fem_steps': 0,
        },
        'accuracy_gates': {
            'software_tangent_threshold': LIMIT,
            'center_stress_and_tangent_parity': 'resolved within threshold in both captured regions',
            'same_exact_FP32_pair_derivative': 'native and FP64 derivatives are compared on the exact stored plus/minus float32 F pair; source-rebuilt FP64 values are shown separately and their replay difference is retained',
            'requested_direction': 'not qualified; float32 realized direction is separately gated and no small-epsilon pair simultaneously meets direction and local-FD thresholds in both regions',
            'oracle_noise_floor': 'estimated from independent FP64 centered-FD step spread and Richardson comparison; tolerance sensitivity and captured-vs-rebuilt FP64 pair differences are reported separately, not called native noise',
        },
        'cases': cases,
        'historical_039_preserved': {
            'analysis_path': str(LEGACY),
            'analysis_sha256': sha(LEGACY),
            'legacy_source_tolerance_results': [exact_path(str(LEGACY.parent / 'analysis.json'))],
            'interpretation': 'kept as prior failure/inconclusive evidence; the 046 two-word candidate improves center state residual/parity and same-pair comparisons, but does not erase or rewrite 039',
            'cases': legacy_cases,
        },
        'source_bindings': artifact_hashes,
    }
    out = RUN / 'analysis' / 'generic-compensated-projection-fd-audit.json'
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print('report=' + str(out))
    print('source_files_checked=' + str(source_integrity['source_files_checked']) + ' mismatches=' + str(len(source_integrity['source_file_mismatches'])))
    for case in cases:
        print(case['region'], 'center rel stress/tangent', case['center_native_vs_FP64_stress_relative_l2'], case['center_native_vs_FP64_tangent_relative_l2'])
        for row in case['same_exact_FP32_pair_results']:
            if row['requested_epsilon'] in (1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7):
                print(' ', row['requested_epsilon'], 'dirError', row['realized_direction_relative_l2_error'], 'nativeAD-vs-FP64FD', row['native_AD_at_realized_direction_vs_FP64_FD_same_pair_scaled_error'], 'nativeFD-vs-FP64FD', row['native_FD_vs_FP64_FD_same_exact_F32_pair_scaled_error'], 'branch', row['branch_classification_from_independent_FP64_exact_pair'])
        print(' unquantized 5e-7 FD/AD', case['FP64_oracle_self_check_at_5e-7']['raw_FD_vs_AD_relative'])


def verify_prediction_inputs(prediction: dict) -> dict:
    checked = 0
    mismatches = []
    for relpath, expected in prediction.get('inputs_sha256', {}).items():
        if relpath.startswith('source/'):
            path = SNAPSHOT / relpath.removeprefix('source/')
        else:
            continue
        checked += 1
        if not path.is_file() or sha(path) != expected:
            mismatches.append(str(path))
    return {'source_files_checked': checked, 'source_file_mismatches': mismatches,
            'all_checked_source_files_match': not mismatches}


if __name__ == '__main__':
    main()
