#!/usr/bin/env python3
"""Read-only independent construction verification for selected combined-backtrack-487 pass 006."""
from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.util
import json
import sys

import numpy as np

R = Path('/Users/numi/Work/numi-human-connected-surface-069')
D = R / 'D'
RUN = D / 'combined-backtrack-487/run-001'
BACK = RUN / 'backtrack-373'
PASS = BACK / 'pass-006'
BASE = D / 'outward-plane-backtrack-461/run-001/backtrack-373/pass-011/candidate.npz'
BASE_RUN = BASE.parent
FIT = R / 'barycentric-connected-clearance-170/combined-fit-486/run-001'
TRIAL = FIT / 'candidate.npz'
FIT_REPORT = FIT / 'report.json'
CANDIDATE = PASS / 'candidate.npz'
STATE = BACK / 'selected-result.json'
RESULT = PASS / 'result.json'
CONFIG = RUN / 'backtrack-config.json'
LAUNCH = RUN / 'launch.json'
INPUTS = BACK / 'source-inputs.json'
SCALES = PASS / 'node-step-scales.npy'
PREPROJECTION = RUN / 'preprojection.json'
SEED_REPORT = RUN / 'seed-scale-report.json'
SEED_CACHE = RUN / 'seed-rows.npz'
PREP = D / 'connected-continuation-prep476/run-001'
PREP_REPORT = PREP / 'preparation-report.json'
SEED_PREFLIGHT = PREP / 'self-seeds/report.json'
SEED_PAIRS = PREP / 'self-seeds/merged-self-pair-seeds.npy'
GUARD_DIR = D / 'fixed-bone-guard-refresh476/run-001'
GUARD_NPZ = GUARD_DIR / 'introduced-skin-bone-barriers.npz'
GUARD_REPORT = GUARD_DIR / 'report.json'
DEPTH_DIR = D / 'point-depth-refresh476/run-001'
DEPTH_NPZ = DEPTH_DIR / 'point-depth-rows.npz'
DEPTH_REPORT = DEPTH_DIR / '365-compatible-report.json'
SOURCE = R / 'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz'
OWNER401 = D / 'retained-gain-continuation-401/propose_joint_clearance_witnesses_401.py'
OWNER313 = R / 'publication-113/sources/connected-fixed-boundary-313/fixed_bone_triangle_rows_313.py'
OUT = BACK / 'root-independent-construction-verification.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def point_set(report):
    rows = report.get('inside_witnesses')
    require(isinstance(rows, list), '130 receipt lacks inside_witnesses')
    out = {(int(r['query_row']), int(r['bank_body']), int(r['bank_stable_id']))
           for r in rows}
    require(len(out) == len(rows), '130 receipt has duplicate typed inside identities')
    return out


def surface_set(report):
    require(report.get('status') == 'COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT'
            and report.get('complete') is True, '159 receipt is incomplete')
    banks = report.get('banks')
    require(isinstance(banks, list) and len(banks) == 185
            and report.get('bank_count') == 185
            and report.get('expected_bank_count') == 185,
            '159 receipt does not cover all 185 typed banks')
    pairs = set()
    count = 0
    for bank in banks:
        rows = (bank.get('result') or {}).get('triangle_pairs')
        if rows is None:
            continue
        require(int(bank['result'].get('count', -1)) == len(rows),
                '159 bank pair count disagrees with records')
        count += len(rows)
        for face, triangle in rows:
            pair = (int(face), int(bank['body']), int(bank['stable_bone_id']), int(triangle))
            require(pair not in pairs, '159 receipt has duplicate typed surface identities')
            pairs.add(pair)
    require(len(pairs) == count, '159 total pair count disagrees with typed identities')
    return pairs, count


def load_owner():
    sys.path.insert(0, str(OWNER401.parent))
    spec = importlib.util.spec_from_file_location('verify487_owner401', OWNER401)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    state, result, config, launch, inputs = map(read, (STATE, RESULT, CONFIG, LAUNCH, INPUTS))
    fit_report = read(FIT_REPORT)
    candidate_sha, base_sha, trial_sha = sha(CANDIDATE), sha(BASE), sha(TRIAL)
    require(candidate_sha == '660ce283bdd98483594be831d1be68dcf350da0b1b70a45837e1f9fa34c66de0'
            and state.get('candidate') == str(CANDIDATE)
            and state.get('candidate_sha256') == candidate_sha,
            'selected 487 pass-006 candidate identity mismatch')
    require(base_sha == '742315227c1d8225aac9307af828251a73a98a9c40b4f59fe791aeef5da1ba0f',
            '461 baseline candidate identity mismatch')
    require(trial_sha == '18523ecca52f0691aa1f9ba7495ced051a2e2a88cd0f2f8a626e7ca8df8eeff5',
            'fit-486 candidate identity mismatch')
    require(inputs.get('start') == str(BASE) and inputs.get('start_sha256') == base_sha
            and inputs.get('trial') == str(TRIAL) and inputs.get('trial_sha256') == trial_sha,
            '487 backtrack did not bind exact base461 and fit486 inputs')
    require(config.get('state', {}).get('candidate') == str(BASE)
            and config.get('trial') == str(TRIAL)
            and config.get('parent_fit') == str(FIT_REPORT)
            and config.get('fit_candidate_sha256') == trial_sha
            and config.get('fit_report_sha256') == sha(FIT_REPORT),
            '487 config does not pin base461 and fit486')
    require(launch.get('base_sha256') == base_sha and launch.get('fit_sha256') == trial_sha
            and launch.get('native_admission') is False
            and launch.get('accepted_physical_steps') == 0,
            '487 launch pins or admission scope mismatch')
    require(fit_report.get('status') == 'CONSTRUCTION_PROPOSAL_WRITTEN_UNADMITTED'
            and fit_report.get('native_admission') is False
            and fit_report.get('accepted_physical_steps') == 0,
            'fit486 status/admission mismatch')

    base, trial, cand = (np.load(p, allow_pickle=False) for p in (BASE, TRIAL, CANDIDATE))
    x, z, y = (np.asarray(a['points_world_m']) for a in (base, trial, cand))
    require(x.dtype == y.dtype == z.dtype == np.float32
            and x.shape == y.shape == z.shape == (20889, 3),
            'base/fit/output are not the pinned Float32 mesh')
    require(set(base.files) == set(cand.files), 'candidate NPZ key set differs from 461 base')
    changed_keys = []
    for key in base.files:
        if key == 'points_world_m':
            continue
        require(base[key].dtype == cand[key].dtype and base[key].shape == cand[key].shape
                and np.ascontiguousarray(base[key]).tobytes() == np.ascontiguousarray(cand[key]).tobytes(),
                'noncoordinate candidate array changed: ' + key)
        changed_keys.append(key)
    require(len(changed_keys) == 169, 'expected exactly 169 unchanged noncoordinate arrays')

    fit_support = np.asarray(config.get('fit_support_rows'), dtype=np.int64)
    trial_support = np.asarray(trial['local_witness_cluster_222_support_rows'], dtype=np.int64)
    require(np.array_equal(fit_support, trial_support)
            and len(fit_support) == 7304 and len(np.unique(fit_support)) == 7304,
            '487 support differs from pinned fit486 support')
    require(int(fit_report['support']['support_nodes']) == 7304
            and int(fit_report['support']['fixed_anchor_rows_total']) == 237,
            'fit486 report support/anchor counts mismatch')
    anchors = np.asarray(base['typed_coarse_anchor_output_rows_018'], dtype=np.int64)
    require(len(anchors) == 237 and np.array_equal(x[anchors], y[anchors]),
            'one or more of 237 typed anchors changed')
    changed = np.flatnonzero(np.any(x != y, axis=1))
    outside = np.setdiff1d(np.arange(len(x), dtype=np.int64), fit_support)
    require(np.isin(changed, fit_support).all() and np.array_equal(x[outside], y[outside]),
            'selected coordinates changed outside fit486 support')
    scales = np.load(SCALES, allow_pickle=False)
    reconstructed = (x.astype(np.float64) + scales[:, None] *
                     (z.astype(np.float64) - x.astype(np.float64))).astype(np.float32)
    require(scales.shape == (len(x),) and np.array_equal(reconstructed, y),
            'selected coordinates do not reconstruct from pinned fit486 delta and saved scales')
    require(array_sha(y) == result.get('emitted_points_sha256'),
            'selected emitted point bytes do not match 487 result receipt')

    require(fit_report['inputs'].get('base_candidate_sha256') == base_sha
            and fit_report['inputs'].get('area_reference_sha256') == base_sha,
            'fit486 does not bind base461 as base and fixed 037 area source')
    require(fit_report['inputs'].get('fixed_boundary_npz_sha256') == sha(GUARD_NPZ)
            and fit_report['inputs'].get('fixed_boundary_report_sha256') == sha(GUARD_REPORT)
            and fit_report['inputs'].get('point_depth_rows_npz_sha256') == sha(DEPTH_NPZ)
            and fit_report['inputs'].get('point_depth_rows_report_sha256') == sha(DEPTH_REPORT)
            and fit_report['inputs'].get('self_pairs_sha256') == sha(SEED_PAIRS),
            'fit486 did not consume matched refresh476 guard, depth and fullseed inputs')
    require(fit_report['inputs'].get('face_target_npz_sha256') == sha(PREP/'surface-targets-335/targets.npz')
            and fit_report['inputs'].get('face_target_report_sha256') == sha(PREP/'surface-targets-335/report.json')
            and fit_report['inputs'].get('point_target_npz_sha256') == sha(PREP/'point-targets-390/nearest-union-exits.npz')
            and fit_report['inputs'].get('point_target_report_sha256') == sha(PREP/'point-targets-390/nearest-union-exits.json'),
            'fit486 target inputs differ from matched continuation prep476')
    require(config.get('self_pair_seed_count') == 25984
            and config.get('self_pair_seed_sha256') == sha(SEED_PAIRS)
            and config.get('self_pair_seed_report_sha256') == sha(SEED_PREFLIGHT),
            '487 config self-seed pins differ from prep476')

    owner = load_owner()
    source_ref, source_receipt = owner.compose_source_reference(base, SOURCE)
    step_mm = np.linalg.norm(y.astype(np.float64) - x.astype(np.float64), axis=1) * 1000.0
    source_mm = np.linalg.norm(y.astype(np.float64) - source_ref, axis=1) * 1000.0
    area = owner.full_f32_area_gate(
        y, x.astype(np.float64) * 1000.0,
        np.asarray(base['area_reference_037_points_world_m_f64'], dtype=np.float64) * 1000.0,
        base['source_O_faces'], 0.1)
    require(area.get('pass') is True, 'selected Float32 output failed fixed 037 area gate')
    require(float(source_mm.max(initial=0.0)) <= 12.0 + 1e-6,
            'selected output exceeds the 12 mm source displacement cap')

    base_point_path, base_surface_path = BASE_RUN/'audit-130/summary.json', BASE_RUN/'audit-159/report.json'
    current_point_path, current_surface_path = PASS/'audit-130/summary.json', PASS/'audit-159/report.json'
    base_point, base_surface = read(base_point_path), read(base_surface_path)
    current_point, current_surface = read(current_point_path), read(current_surface_path)
    require(sha(current_point_path) == state.get('point_summary_sha256')
            and current_point.get('candidate_sha256') == candidate_sha
            and current_point.get('complete_surface_audit') is True
            and int(current_point.get('exact_PLC_surface_pairs', -1)) == 0,
            'selected 130 exact self receipt mismatch or incomplete')
    require(base_point.get('candidate_sha256') == base_sha
            and base_point.get('complete_surface_audit') is True
            and int(base_point.get('exact_PLC_surface_pairs', -1)) == 0
            and base_surface.get('candidate_sha256') == base_sha,
            '461 baseline exact audit receipts do not bind base')
    require(current_surface.get('candidate_sha256') == candidate_sha,
            'selected 159 report does not bind candidate')
    base_points, current_points = point_set(base_point), point_set(current_point)
    base_surfaces, base_surface_count = surface_set(base_surface)
    current_surfaces, current_surface_count = surface_set(current_surface)
    point_added, point_removed = current_points-base_points, base_points-current_points
    surface_added, surface_removed = current_surfaces-base_surfaces, base_surfaces-current_surfaces
    pm, jm, dm = result['point_merit'], result['joint_merit'], result['depth_merit']
    require((len(base_points), len(current_points)) ==
            (pm['before_inside_pairs'], pm['after_inside_pairs']) == (316, 298)
            and not point_added and len(point_removed) == 18,
            '487 point pair identity delta differs from exact 130 receipts')
    require((base_surface_count, current_surface_count) ==
            (jm['before_surface_pairs'], jm['after_surface_pairs']) == (3117, 3104)
            and len(surface_added) == 86 and len(surface_removed) == 99
            and dm.get('introduced_surface_pair_count') == len(surface_added),
            '487 surface identity delta differs from exact 159 receipts')
    require((len(base_points)+base_surface_count, len(current_points)+current_surface_count) ==
            (jm['before_total_pairs'], jm['after_total_pairs']) == (3433, 3402)
            and jm.get('typed_bank_count') == 185,
            '487 total pair counts differ from exact receipts')

    prep_report = read(PREP_REPORT)
    seed_preflight = read(SEED_PREFLIGHT)
    seed_scale = read(SEED_REPORT)
    seed_pairs = np.load(SEED_PAIRS, allow_pickle=False)
    seed_cache = np.load(SEED_CACHE, allow_pickle=False)
    projection = read(PREPROJECTION)
    seed_lp = projection.get('seed_cut_lp', {})
    fixed = seed_lp.get('fixed_guard_certificate', {})
    require(prep_report.get('status') == 'PASS_CONSTRUCTION_INPUTS_ONLY'
            and prep_report.get('candidate_sha256') == base_sha
            and seed_preflight.get('status') == 'PASS_REBUILT_CONSTRUCTION_SEEDS_ONLY'
            and sha(SEED_PAIRS) == seed_preflight.get('output_pairs_sha256')
            and len(seed_pairs) == 25984,
            'matched prep476 fullseed pins/count mismatch')
    require(seed_scale.get('status') == 'GRADED_426_WITH_FULL_SELF_SEEDS_AND_FIXED_BONE_GUARDS_447'
            and seed_scale.get('seed_pair_count') == len(seed_pairs) == 25984
            and seed_scale.get('seed_linear_row_count') == 146240
            and sha(SEED_CACHE) == seed_scale.get('seed_row_cache_sha256')
            and sha(SEED_PAIRS) == seed_scale.get('seed_pair_sha256'),
            '487 seed-scale fullseed pins or counts mismatch')
    require(len(seed_cache['lower']) == 146240 and len(seed_cache['indptr']) == 146241
            and int(seed_lp.get('seed_rows_total', -1)) == 146240
            and int(seed_lp.get('seed_rows_checked', -1)) == 146240
            and int(seed_lp.get('seed_violated_rows', -1)) == 0
            and float(seed_lp.get('seed_max_violation_mm', float('inf'))) <= 1e-12,
            '487 full self-seed certificate failed')
    require(set(map(int, seed_lp.get('anchor_rows_forced_zero', []))) == set(map(int, anchors)),
            '487 seed LP forced-zero anchor rows differ from the 237 typed base anchors')
    guard_receipt = fixed.get('input_receipt', {})
    guard_arrays = np.load(GUARD_NPZ, allow_pickle=False)
    guard_report = read(GUARD_REPORT)
    require(fixed.get('passed') is True and int(fixed.get('row_count', -1)) == 13950
            and int(fixed.get('checked_rows', -1)) == 13950
            and int(fixed.get('violated_rows', -1)) == 0
            and float(fixed.get('max_violation_mm', float('inf'))) <= float(fixed['tolerance_mm'])
            and float(fixed.get('anchor_max_abs_scale', float('inf'))) == 0.0,
            '487 fixed-bone guard alpha certificate failed')
    require(guard_receipt.get('base_candidate_sha256') == base_sha
            and guard_receipt.get('npz_sha256') == sha(GUARD_NPZ)
            and guard_receipt.get('report_sha256') == sha(GUARD_REPORT)
            and guard_report.get('base_candidate_sha256') == base_sha
            and guard_report.get('status') == 'COMPLETE_TYPED_INTRODUCED_SKIN_BONE_BARRIER_EXPORT'
            and int(guard_report['construction_scope']['retained_guard_pairs']) == 2325
            and len(guard_arrays['skin_face_rows']) * 6 == 13950
            and sha(OWNER313) == guard_receipt.get('313_owner_sha256'),
            '487 fixed-bone guard input pins/count mismatch')

    pin_paths = [STATE, RESULT, CONFIG, LAUNCH, INPUTS, PREPROJECTION, SEED_REPORT,
                 SEED_CACHE, SEED_PAIRS, SEED_PREFLIGHT, PREP_REPORT, GUARD_NPZ,
                 GUARD_REPORT, DEPTH_NPZ, DEPTH_REPORT, BASE, TRIAL, FIT_REPORT,
                 CANDIDATE, SCALES, current_point_path, current_surface_path,
                 base_point_path, base_surface_path, SOURCE, OWNER401, OWNER313, Path(__file__)]
    pins = {str(p): sha(p) for p in pin_paths}
    receipt = {
        'status': 'ROOT_INDEPENDENT_CONSTRUCTION_PASS',
        'classification': 'construction-only independent verification; no physical/native admission',
        'candidate_path': str(CANDIDATE), 'candidate_sha256': candidate_sha,
        'base461_path': str(BASE), 'base461_sha256': base_sha,
        'fit486_candidate_sha256': trial_sha, 'fit486_report_sha256': sha(FIT_REPORT),
        'fit486_support_rows': int(len(fit_support)),
        'fit486_support_sha256': array_sha(np.asarray(fit_support, dtype='<i8')),
        'fit_backtrack_scales_reconstruct_candidate': True,
        'outside_fit486_support_unchanged': True,
        'unchanged_noncoordinate_array_count': len(changed_keys),
        'anchor_count': int(len(anchors)), 'anchors_bitwise_unchanged': True,
        'changed_node_count': int(len(changed)),
        'max_step_mm': float(step_mm.max(initial=0.0)),
        'max_source_displacement_mm': float(source_mm.max(initial=0.0)),
        'source_cap_mm': 12.0, 'source_reference_receipt': source_receipt,
        'fixed_037_float32_area_gate': area,
        'exact_130': {'base_report_sha256': sha(base_point_path),
                      'candidate_report_sha256': sha(current_point_path),
                      'base_inside_pairs': len(base_points), 'candidate_inside_pairs': len(current_points),
                      'resolved_pairs': len(point_removed), 'introduced_pairs': len(point_added),
                      'candidate_exact_surface_pairs': int(current_point['exact_PLC_surface_pairs'])},
        'exact_159': {'base_report_sha256': sha(base_surface_path),
                      'candidate_report_sha256': sha(current_surface_path), 'typed_banks': 185,
                      'base_pairs': base_surface_count, 'candidate_pairs': current_surface_count,
                      'resolved_pairs': len(surface_removed), 'introduced_pairs': len(surface_added)},
        'joint_pair_counts': {'base': len(base_points)+base_surface_count,
                              'candidate': len(current_points)+current_surface_count,
                              'candidate_reduced': True},
        'full_seed_certificate': {'seed_pair_count': int(seed_scale['seed_pair_count']),
                                  'linear_rows': int(seed_scale['seed_linear_row_count']),
                                  'rows_checked': int(seed_lp['seed_rows_checked']),
                                  'violated_rows': int(seed_lp['seed_violated_rows']),
                                  'max_violation_mm': float(seed_lp['seed_max_violation_mm']),
                                  'seed_cache_sha256': sha(SEED_CACHE),
                                  'seed_pairs_sha256': sha(SEED_PAIRS),
                                  'preflight_report_sha256': sha(SEED_PREFLIGHT)},
        'fixed_guard_certificate': {'passed': bool(fixed['passed']),
                                    'linear_rows': int(fixed['row_count']),
                                    'checked_rows': int(fixed['checked_rows']),
                                    'violated_rows': int(fixed['violated_rows']),
                                    'max_violation_mm': float(fixed['max_violation_mm']),
                                    'tolerance_mm': float(fixed['tolerance_mm']),
                                    'anchor_max_abs_scale': float(fixed['anchor_max_abs_scale']),
                                    'typed_fixed_pairs': int(guard_receipt['typed_pair_count']),
                                    'guard_npz_sha256': sha(GUARD_NPZ),
                                    'guard_report_sha256': sha(GUARD_REPORT),
                                    '313_owner_sha256': guard_receipt['313_owner_sha256']},
        'reported_depth_metrics': {'before_max_depth_m': pm['before_maximum_depth_m'],
                                   'after_max_depth_m': pm['after_maximum_depth_m'],
                                   'before_squared_depth_m2': pm['before_squared_depth_m2'],
                                   'after_squared_depth_m2': pm['after_squared_depth_m2'],
                                   'depth_merit_classification': dm.get('classification')},
        'zero_clearance_achieved': False, 'geometry_admitted': False,
        'native_admission': False, 'accepted_physical_steps': 0,
        'solver_or_geometry_audit_rerun': False,
        'verification_script': str(Path(__file__)), 'verification_script_sha256': sha(Path(__file__)),
        'input_pins': pins,
    }
    OUT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({'status':receipt['status'],'candidate_sha256':candidate_sha,
                      'support_rows':len(fit_support),'unchanged_noncoordinate_arrays':len(changed_keys),
                      'anchors':len(anchors),'outside_support_unchanged':True,
                      'area_gate_pass':area.get('pass'),'source_max_mm':float(source_mm.max(initial=0.0)),
                      'point_pairs':[len(base_points),len(current_points)],
                      'surface_pairs':[base_surface_count,current_surface_count],
                      'seed_rows_checked':int(seed_lp['seed_rows_checked']),
                      'fixed_guard_rows_checked':int(fixed['checked_rows']),
                      'receipt_path':str(OUT),'receipt_sha256':sha(OUT)},sort_keys=True))


if __name__ == '__main__':
    main()
