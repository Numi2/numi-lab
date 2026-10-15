#!/usr/bin/env python3
"""CPU-only proof of raw JSON decimal versus native binary32 input replay.

Replays the exact compiled FP64 residual bytecode in center-replay.jsonl,
holding candidate/accepted state and compiled parameters constant. It compares
the raw JSON decimal F/F-low/rate values against their exact binary32-promoted
counterparts and checks the latter against captured native accurate residuals.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

BASE = Path('/Users/home/cardboard-implementation-20261014')
RUN = BASE / 'native/validated-same-equation-captured-material-replay-184'
PREDICTION = RUN / 'prediction.json'
RECEIPT = RUN / 'receipt.json'
REPLAY = RUN / 'center-replay.jsonl'
RAW_CAPTURE = BASE / 'solver/current-failure-22-state-capture-153/operator-capture'
EXTERNAL_MANIFEST = RUN / 'inputs/external/manifest.json'
BUILD_MANIFEST = RUN / 'inputs/external/source-build.json'
SNAPSHOT = BASE / 'snapshots/source-selected-newton-rap-and-validated-replay-integrated-060'
SHARED = SNAPSHOT / 'matter/include/numi/matter/shared.h'
CHECKER = SNAPSHOT / 'matter/tools/cardboard_paper_gpu_check.mm'
OUT = BASE / 'analysis/validated-replay184-f32-input-diagnostic-002'
TARGET_ENV = 0
TARGET_TET = 1291
TARGET_ROW = 10

OPCODES = {
    0: 'constant', 1: 'parameter', 2: 'F', 3: 'DF', 4: 'state', 5: 'add',
    6: 'subtract', 7: 'multiply', 8: 'divide', 9: 'negate', 10: 'log',
    11: 'exp', 12: 'sqrt', 13: 'abs', 14: 'min', 15: 'max',
    16: 'pow_integer', 17: 'clamp', 18: 'dt', 19: 'temperature', 20: 'rate',
    21: 'next_state', 22: 'expm1_minus_x', 23: 'fiber_exp_linear',
    24: 'select_nonnegative', 25: 'cache_store', 26: 'cache_load'}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def f32(value: float) -> float:
    return float(struct.unpack('<f', struct.pack('<f', float(value)))[0])


def f32_bits_to_double(hex_bits: str) -> float:
    return float(struct.unpack('<f', struct.pack('<I', int(hex_bits, 16)))[0])


def immediate_to_double(hex_bits: str) -> float:
    return f32_bits_to_double(hex_bits)


def eval_program(program: dict, parameters: list[float], deformation: list[float],
                 deformation_low: list[float], deformation_rate: list[float],
                 accepted: list[float], candidate_high: list[float],
                 candidate_low: list[float], dt: float) -> float:
    flags = int(program['flags'])
    use_low = (flags & 2) != 0
    stack: list[float] = []
    cache: list[float] = [0.0] * 16
    cache_written = [False] * 16
    domain_floor = f32(1.0e-12)

    def pop() -> float:
        if not stack:
            raise ValueError('bytecode stack underflow')
        return stack.pop()

    for row in program['instructions']:
        opcode, index, integer = int(row[0]), int(row[1]), int(row[2])
        if opcode == 0:
            value = immediate_to_double(row[4])
        elif opcode == 1:
            value = parameters[index]
        elif opcode == 2:
            value = deformation[index] + (deformation_low[index] if use_low else 0.0)
        elif opcode == 3:
            value = 0.0
        elif opcode == 4:
            value = accepted[index]
        elif opcode == 5:
            b, a = pop(), pop(); value = a + b
        elif opcode == 6:
            b, a = pop(), pop(); value = a - b
        elif opcode == 7:
            b, a = pop(), pop(); value = a * b
        elif opcode == 8:
            b, a = pop(), pop()
            if not abs(b) > domain_floor:
                raise ValueError('FP64 bytecode divide domain failure')
            value = a / b
        elif opcode == 9:
            value = -pop()
        elif opcode == 10:
            a = pop()
            if not a > 0.0: raise ValueError('FP64 bytecode log domain failure')
            value = math.log(a)
        elif opcode == 11:
            value = math.exp(pop())
        elif opcode == 12:
            a = pop()
            if a < 0.0: raise ValueError('FP64 bytecode sqrt domain failure')
            value = math.sqrt(a)
        elif opcode == 13:
            value = abs(pop())
        elif opcode == 14:
            b, a = pop(), pop(); value = min(a, b)
        elif opcode == 15:
            b, a = pop(), pop(); value = max(a, b)
        elif opcode == 16:
            a = pop()
            if integer < 0 and not abs(a) > domain_floor:
                raise ValueError('FP64 bytecode negative-power domain failure')
            power = abs(integer)
            factor, result = a, 1.0
            while power:
                if power & 1: result *= factor
                factor *= factor
                power >>= 1
            value = 1.0 / result if integer < 0 else result
        elif opcode == 17:
            upper, lower, x = pop(), pop(), pop()
            if upper < lower: raise ValueError('FP64 bytecode clamp bounds reversed')
            value = min(max(x, lower), upper)
        elif opcode == 18:
            value = dt
        elif opcode == 19:
            value = 293.15
        elif opcode == 20:
            value = deformation_rate[index]
        elif opcode == 21:
            value = candidate_high[index] + (candidate_low[index] if use_low else 0.0)
        elif opcode == 22:
            a = pop(); value = math.exp(a) - 1.0 - a
        elif opcode == 23:
            raise ValueError('fiber_exp_linear is unsupported by this audit; refuse partial replay')
        elif opcode == 24:
            negative, nonnegative, condition = pop(), pop(), pop()
            value = nonnegative if condition >= 0.0 else negative
        elif opcode == 25:
            if index >= len(cache) or not stack: raise ValueError('invalid cache store')
            cache[index] = stack[-1]; cache_written[index] = True
            continue
        elif opcode == 26:
            if index >= len(cache) or not cache_written[index]: raise ValueError('invalid cache load')
            value = cache[index]
        else:
            raise ValueError(f'unsupported opcode {opcode}')
        if not math.isfinite(value):
            raise ValueError(f'nonfinite bytecode result at opcode {opcode}')
        stack.append(value)
    if len(stack) != 1:
        raise ValueError(f'bytecode final stack depth is {len(stack)}')
    return stack[0]


def flat9(values) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape != (9,) or not np.isfinite(arr).all():
        raise ValueError('expected finite matrix with 9 values')
    return arr.tolist()


def rows13(rows) -> list[float]:
    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape != (13,) or not np.isfinite(arr).all():
        raise ValueError('expected 13 finite residual rows')
    return arr.tolist()


def main() -> dict:
    header = None
    summary = None
    centers = []
    with REPLAY.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('record') == 'header':
                if header is not None: raise ValueError('duplicate replay header')
                header = row
            elif row.get('record') == 'summary':
                if summary is not None: raise ValueError('duplicate replay summary')
                summary = row
            elif row.get('record') == 'captured_center_replay':
                centers.append(row)
    if not isinstance(header, dict) or header.get('schema') != 'numi.cardboard.paper-captured-center-replay.v2':
        raise ValueError('unsupported or missing replay header')
    if not isinstance(summary, dict): raise ValueError('missing replay summary')
    receipt = json.loads(RECEIPT.read_text())
    prediction = json.loads(PREDICTION.read_text())
    external = json.loads(EXTERNAL_MANIFEST.read_text())
    build = json.loads(BUILD_MANIFEST.read_text())
    capture_hash = sha256(RAW_CAPTURE)
    if capture_hash != header['capture_sha256'] or capture_hash != summary['capture_sha256']:
        raise ValueError('raw capture SHA does not match replay header/summary')
    if not receipt.get('execution_started') or not receipt.get('source_inputs_unchanged'):
        raise ValueError('native receipt lacks source-integrity/execution evidence')
    if receipt['exit_code'] != 2 or receipt['timed_out']:
        raise ValueError('expected retained non-timeout diagnostic exit2; refusing different run')
    if not build.get('success') or build.get('changed_sources') != []:
        raise ValueError('source build manifest is incomplete or has changed sources')
    if header['physical_validation'] is not False or header['physical_steps'] != 0:
        raise ValueError('unexpected physical validation claim in replay input')

    worlds = {item['label']: item for item in header['compiled_worlds']}
    # Exact native source-material name -> accurate compiled residual programs.
    world_for_material = {
        'hajali2009_liner_hill_ideal': worlds['liner_accurate_candidate'],
        'hajali2009_medium_hill_ideal': worlds['medium_accurate_candidate'],
    }
    for name, world in world_for_material.items():
        if world['material_flags'] != 53:
            raise ValueError(f'{name} accurate compiled material flags differ from captured policy')
        if len(world['source_parameters']) != 20 or len(world['residual_programs']) != 13:
            raise ValueError(f'{name} compiled parameter/residual inventory malformed')
    opcode_counts = collections.Counter()
    for world in world_for_material.values():
        for program in world['residual_programs']:
            for instruction in program['instructions']:
                opcode_counts[int(instruction[0])] += 1
    if opcode_counts[23]:
        raise ValueError('fiber opcode appears; interpreter would need exact production helper')

    # Verify every F/F-low/rate field in replay output is exactly the same
    # parsed decimal as its parent captured tetrahedron record.
    raw_capture_records = {}
    with RAW_CAPTURE.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('record') == 'tetrahedron':
                raw_capture_records[(int(row['environment']), int(row['tet_id']))] = row
    if len(raw_capture_records) != 3072:
        raise ValueError(f'raw capture tetrahedron coverage is {len(raw_capture_records)}, expected 3072')
    if len(centers) != int(summary['paper_centers_replayed']) or len(centers) != 2880:
        raise ValueError('paper center replay count does not equal the expected 2880')

    row10_raw_errors = []
    row10_float_errors = []
    row10_rate_only_errors = []
    row10_raw_vs_native = []
    row10_float_vs_native = []
    row10_stored_bytecode_errors = []
    allrow_stored_bytecode_errors = []
    input_delta = {name: {'count': 0, 'max_abs': 0.0, 'max_rel': 0.0, 'max_ulp': 0}
                   for name in ('F_high', 'F_low', 'Fdot_rate')}
    target = None
    evaluated = 0

    for center in centers:
        key = (int(center['environment']), int(center['tet_id']))
        capture = raw_capture_records.get(key)
        if capture is None: raise ValueError(f'replay center {key} absent from raw capture')
        source = capture['center']
        high_raw = flat9(center['F_local_high'])
        low_raw = flat9(center['F_local_low'])
        rate_raw = flat9(center['Fdot_local_per_s'])
        if high_raw != flat9(source['F_local']) or low_raw != flat9(source['F_local_low']) or \
           rate_raw != flat9(source['Fdot_local_per_s']):
            raise ValueError(f'replay F/F-low/rate decimal differs from raw capture at {key}')
        high_f32 = [f32(x) for x in high_raw]
        low_f32 = [f32(x) for x in low_raw]
        rate_f32 = [f32(x) for x in rate_raw]
        for label, raw, rounded in (('F_high', high_raw, high_f32),
                                    ('F_low', low_raw, low_f32),
                                    ('Fdot_rate', rate_raw, rate_f32)):
            stats = input_delta[label]
            for a, b in zip(raw, rounded):
                if a != b: stats['count'] += 1
                stats['max_abs'] = max(stats['max_abs'], abs(a - b))
                stats['max_rel'] = max(stats['max_rel'], abs(a - b) / max(abs(a), 1e-300))
                ua = struct.unpack('<I', struct.pack('<f', f32(a)))[0]
                ub = struct.unpack('<I', struct.pack('<f', f32(b)))[0]
                stats['max_ulp'] = max(stats['max_ulp'], abs(int(ua) - int(ub)))

        world = world_for_material[center['material_name']]
        if int(center['material_index']) != (0 if 'liner' in center['material_name'] else 1):
            raise ValueError(f'material index/name mismatch at {key}')
        params = [f32_bits_to_double(x['compiled_f32_bits']) for x in world['source_parameters']]
        accepted = [f32(x) for x in center['accepted_state']]
        candidate = center['candidate_state_snapshots']['accurate_candidate']
        if not candidate['available']:
            continue
        candidate_high = [f32(x) for x in candidate['high']]
        candidate_low = [f32(x) for x in candidate['low']]
        if len(accepted) != 13 or len(candidate_high) != 13 or len(candidate_low) != 13:
            raise ValueError(f'candidate state shape mismatch at {key}')
        dt = f32(center['captured_timestep_seconds'])
        bytecode_expected = next(x for x in center['compiled_residual_rows']['compiled_bytecode_fp64']['accurate_policy']
                                 if x['candidate'] == 'accurate_candidate')
        native = next(x for x in center['compiled_residual_rows']['native_accurate_policy']['accurate_candidate']
                      if x['candidate'] == 'accurate_candidate')
        if not bytecode_expected['valid'] or not native['valid']:
            continue
        expected_rows = rows13(bytecode_expected['residual_fp64'])
        native_rows = [float(a) + float(b) for a, b in zip(native['residual_high'], native['residual_low'])]
        raw_values, float_values, rate_only_values = [], [], []
        for program in world['residual_programs']:
            raw_values.append(eval_program(program, params, high_raw, low_raw, rate_raw,
                accepted, candidate_high, candidate_low, dt))
            float_values.append(eval_program(program, params, high_f32, low_f32, rate_f32,
                accepted, candidate_high, candidate_low, dt))
            rate_only_values.append(eval_program(program, params, high_raw, low_raw, rate_f32,
                accepted, candidate_high, candidate_low, dt))
        raw_values = rows13(raw_values)
        float_values = rows13(float_values)
        rate_only_values = rows13(rate_only_values)
        evaluated += 1
        for row in range(13):
            allrow_stored_bytecode_errors.append(abs(raw_values[row] - expected_rows[row]))
        row10_raw_errors.append(abs(raw_values[TARGET_ROW] - expected_rows[TARGET_ROW]))
        row10_float_errors.append(abs(float_values[TARGET_ROW] - expected_rows[TARGET_ROW]))
        row10_rate_only_errors.append(abs(rate_only_values[TARGET_ROW] - raw_values[TARGET_ROW]))
        row10_raw_vs_native.append(abs(raw_values[TARGET_ROW] - native_rows[TARGET_ROW]))
        row10_float_vs_native.append(abs(float_values[TARGET_ROW] - native_rows[TARGET_ROW]))
        if key == (TARGET_ENV, TARGET_TET):
            target = {'environment': key[0], 'tet_id': key[1],
                'material_name': center['material_name'], 'material_index': center['material_index'],
                'input_policy': {'program_flags': world['residual_programs'][TARGET_ROW]['flags'],
                    'low_words_used': (int(world['residual_programs'][TARGET_ROW]['flags']) & 2) != 0,
                    'rate_opcode_count_in_all_accurate_residual_programs': sum(
                        1 for p in world['residual_programs'] for ins in p['instructions'] if int(ins[0]) == 20),
                    'F_raw_json_double': high_raw, 'F_binary32_promoted': high_f32,
                    'F_low_raw_json_double': low_raw, 'F_low_binary32_promoted': low_f32,
                    'Fdot_rate_raw_json_double': rate_raw, 'Fdot_rate_binary32_promoted': rate_f32,
                    'accepted_state_binary32_promoted': accepted,
                    'candidate_state_high_binary32_promoted': candidate_high,
                    'candidate_state_low_binary32_promoted': candidate_low},
                'row10': {'raw_decimal_fp64_bytecode': raw_values[TARGET_ROW],
                    'binary32_promoted_fp64_bytecode': float_values[TARGET_ROW],
                    'rate_only_promoted_fp64_bytecode': rate_only_values[TARGET_ROW],
                    'stored_raw_decimal_bytecode_fp64': expected_rows[TARGET_ROW],
                    'native_accurate_residual_high_plus_low': native_rows[TARGET_ROW],
                    'raw_minus_native': raw_values[TARGET_ROW] - native_rows[TARGET_ROW],
                    'binary32_promoted_minus_native': float_values[TARGET_ROW] - native_rows[TARGET_ROW],
                    'raw_vs_stored_bytecode_abs_error': abs(raw_values[TARGET_ROW] - expected_rows[TARGET_ROW]),
                    'float_vs_stored_raw_bytecode_abs_error': abs(float_values[TARGET_ROW] - expected_rows[TARGET_ROW])}}

    if target is None: raise ValueError('target tet 1291/env0 was not replayed')
    if evaluated != 2880: raise ValueError(f'only {evaluated} accurate candidate centers were replayed')
    if max(allrow_stored_bytecode_errors) > 5e-14:
        raise ValueError('independent raw-input bytecode interpreter does not reproduce stored FP64 bytecode')
    if target['input_policy']['rate_opcode_count_in_all_accurate_residual_programs'] != 0:
        raise ValueError('RATE opcodes are present; rate quantization needs further row-wise attribution')

    report = {'schema': 'numi.cardboard.replay184-f32-input-roundtrip-audit.v1',
        'classification': 'Independent CPU arithmetic/input-roundtrip diagnosis; replay exit2 is preserved; no solver/material/physical qualification',
        'binding': {'run_directory': str(RUN), 'analysis_script_sha256': sha256(Path(__file__).resolve()),
            'center_replay_sha256': sha256(REPLAY),
            'raw_operator_capture_sha256': capture_hash, 'prediction_sha256': sha256(PREDICTION),
            'receipt_sha256': sha256(RECEIPT), 'external_manifest_sha256': sha256(EXTERNAL_MANIFEST),
            'source_build_manifest_sha256': sha256(BUILD_MANIFEST),
            'shared_opcode_map_sha256': sha256(SHARED), 'checker_source_sha256': sha256(CHECKER),
            'prediction_source_snapshot': prediction['source_snapshot'],
            'source_snapshot_shared_sha256': sha256(SHARED),
            'receipt_exit_code': receipt['exit_code'], 'receipt_timed_out': receipt['timed_out'],
            'executed_build_success': build['success'], 'source_inputs_unchanged': receipt['source_inputs_unchanged']},
        'capture_integrity': {'header_capture_hash_matches_raw_capture': True,
            'replay_F_Flow_rate_matches_parent_capture_json': True,
            'raw_capture_tetrahedron_records': len(raw_capture_records),
            'paper_centers_replayed': len(centers), 'physical_steps': header['physical_steps'],
            'physical_validation': header['physical_validation'], 'native_receipt_exit2_preserved': True},
        'input_policy': {'raw_decimal': 'JSON numeric values parsed as binary64, matching the replay checker JSON-to-double path',
            'native_promoted': 'each captured high/low/rate value rounded to IEEE-754 binary32 then widened exactly to binary64',
            'candidate_and_accepted_state': 'high/low/accepted words rounded to binary32 then widened; per-program flag bit2 controls consumption of candidate/F low words',
            'parameters_and_immediates': 'decoded from captured compiled FP32 bit patterns then widened to binary64',
            'timestep': 'rounded to captured binary32 timestep then widened',
            'rate_usage': 'accurate residual programs contain no RATE opcode; rate promotion is included as a null-effect control'},
        'compiled_policy': {'material_flags': 53,
            'accurate_residual_program_flags': sorted(set(int(p['flags']) for w in world_for_material.values() for p in w['residual_programs'])),
            'opcode_counts_accurate_programs': {OPCODES[k]: v for k, v in sorted(opcode_counts.items())},
            'interpreter': 'Independent Python FP64 stack evaluator following compiled bytecode order, FP32 immediate/parameter bits, exact domain floor and low-word program flag',
            'raw_decimal_interpreter_max_abs_difference_vs_stored_allrows': max(allrow_stored_bytecode_errors)},
        'global_input_rounding': input_delta,
        'all_center_accurate_candidate_row10': {'count': evaluated,
            'raw_decimal_vs_native_abs_error': {'median': float(np.median(row10_raw_vs_native)),
                'maximum': float(max(row10_raw_vs_native)), 'p95': float(np.percentile(row10_raw_vs_native, 95))},
            'binary32_promoted_vs_native_abs_error': {'median': float(np.median(row10_float_vs_native)),
                'maximum': float(max(row10_float_vs_native)), 'p95': float(np.percentile(row10_float_vs_native, 95))},
            'raw_bytecode_vs_stored_bytecode_abs_error_max': max(row10_raw_errors),
            'rate_only_promotion_effect_max_abs': max(row10_rate_only_errors)},
        'target_tet': target,
        'interpretation': ['The raw decimal FP64 bytecode replay independently reproduces the serialized compiled_bytecode_fp64 output.',
            'For tet1291 row10, compare the raw-decimal and binary32-promoted evaluations against the captured native accurate residual; the report retains all three values.',
            'The accurate residual program carries flag bit2 and includes F/candidate low words, so both high and low words are replayed under that flag.',
            'Any improved match after binary32 promotion identifies an input representation discrepancy in the FP64 oracle path; it does not claim an independent native arithmetic proof or solver/physical validation.',
            'The native replay exited 2 and reports zero physical steps; no pass is inferred.']}
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    result = main()
    print(json.dumps({'report': str(OUT / 'report.json'),
        'target_row10': result['target_tet']['row10'],
        'across_centers': result['all_center_accurate_candidate_row10']}, sort_keys=True))
