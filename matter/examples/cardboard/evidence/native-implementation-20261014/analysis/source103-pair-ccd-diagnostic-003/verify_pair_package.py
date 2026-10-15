#!/usr/bin/env python3
"""Fail-closed CPU validator for the source-103 CCD diagnostic package."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import re
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/Users/home/cardboard-implementation-20261014')
SOURCE = ROOT / 'snapshots/source-whole-mortar-proof-103'

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def strict_json(path: Path):
    def reject(value):
        raise ValueError(f'non-standard JSON token {value} in {path}')
    return json.loads(path.read_text(), parse_constant=reject)

def require(condition: bool, message: str):
    if not condition:
        raise ValueError(message)

def validate_prepared(prepared_path: Path):
    d = strict_json(prepared_path)
    require(d['schema'] == 'numi.cardboard.source103_pair_ccd_inputs.v1', 'wrong input schema')
    require(d['source_snapshot'] == str(SOURCE), 'wrong source snapshot')
    for key, relative in (
        ('source_contact_sha256', 'matter/src/metal/contact.metalinc'),
        ('source_common_sha256', 'matter/src/metal/common.metalinc'),
        ('source_main_shader_sha256', 'matter/src/matter_kernels.metal'),
    ):
        require(sha(SOURCE / relative) == d[key], f'frozen source hash mismatch: {relative}')
    receipt = Path(d['source_build_receipt'])
    require(sha(receipt) == d['source_build_receipt_sha256'], 'source build receipt hash mismatch')
    receipt_data = strict_json(receipt)
    require(receipt_data.get('success') is True, 'capture source-build receipt is not successful')
    require(receipt_data.get('source_snapshot') == str(SOURCE), 'source-build receipt snapshot mismatch')
    # The standalone Metal translation unit includes matter_kernels.metal,
    # whose transitive shader inputs are broader than contact.metalinc.
    # Verify the entire source inventory recorded by the capture build.
    inventory = receipt_data.get('source_sha256')
    require(isinstance(inventory, dict) and len(inventory) > 0, 'source-build receipt has no source inventory')
    checked = 0
    for relative, expected_hash in inventory.items():
        p = SOURCE / relative
        require(p.is_file() and sha(p) == expected_hash, f'source inventory mismatch: {relative}')
        checked += 1
    require(d['case_contract']['total'] == 30, 'unexpected total case count')
    rows = d['cases']
    require(len(rows) == 30 and len({r['name'] for r in rows}) == 30, 'case list missing/duplicated')
    expected_names = []
    for capture in ('center359', 'trial358'):
        expected_names.extend(f'{capture}_vt_direction{direction}_vertex{vertex}'
                              for direction in range(2) for vertex in range(3))
        expected_names.extend(f'{capture}_ee_edge{a}_edge{b}'
                              for a in range(3) for b in range(3))
    require([r['name'] for r in rows] == expected_names, 'case ordering/coverage mismatch')
    binary_path = prepared_path.parent / d['case_contract']['input_binary']
    raw = binary_path.read_bytes()
    require(len(raw) == d['case_contract']['total_input_bytes'] == 30 * 9 * 16,
            'input binary byte count mismatch')
    require(sha(binary_path) == d['case_contract']['input_sha256'], 'input binary hash mismatch')
    expected_binary = bytearray()
    ratios = {'vertex_triangle': [], 'edge_edge': []}
    certificate_metadata = []
    for i, row in enumerate(rows):
        start = np.asarray(row['points_start_m'], dtype='<f4')
        finish = np.asarray(row['points_finish_m'], dtype='<f4')
        require(start.shape == (12,) and finish.shape == (12,), f'bad endpoints in {row["name"]}')
        require(all(math.isfinite(float(x)) for x in start) and all(math.isfinite(float(x)) for x in finish),
                f'nonfinite endpoints in {row["name"]}')
        require([f'{x:08x}' for x in start.view('<u4')] == row['points_start_f32_bits'],
                f'start bits mismatch in {row["name"]}')
        require([f'{x:08x}' for x in finish.view('<u4')] == row['points_finish_f32_bits'],
                f'finish bits mismatch in {row["name"]}')
        records = np.zeros((9,4), dtype='<f4')
        records[0] = np.asarray([row['kind_id'], row['thickness_m'], row['feature_a'], row['feature_b']], dtype='<f4')
        records[1:5,:3] = start.reshape(4,3)
        records[5:9,:3] = finish.reshape(4,3)
        require([f'{x:08x}' for x in records.view('<u4').reshape(-1)] == row['record_f32_bits'],
                f'record bits mismatch in {row["name"]}')
        expected_binary.extend(records.tobytes(order='C'))
        reference = row['cpu_fp64_reference']
        bounds = reference['speed_bound_comparison']
        displacement = finish.astype(np.float64).reshape(4,3) - start.astype(np.float64).reshape(4,3)
        norm = lambda v: math.sqrt(math.fsum(float(q)*float(q) for q in v))
        if row['kind_id'] == 0:
            absolute = norm(displacement[0]) + max(norm(displacement[j]) for j in (1,2,3))
            relative = max(norm(displacement[0]-displacement[j]) for j in (1,2,3))
        else:
            absolute = max(norm(displacement[j]) for j in (0,1)) + max(norm(displacement[j]) for j in (2,3))
            relative = max(norm(displacement[j]-displacement[k]) for j in (0,1) for k in (2,3))
        require(math.isclose(absolute, bounds['production_absolute_endpoint_sum_m'], rel_tol=1e-13, abs_tol=1e-25),
                f'absolute displacement bound mismatch in {row["name"]}')
        require(math.isclose(relative, bounds['galilean_relative_lipschitz_bound_m'], rel_tol=1e-13, abs_tol=1e-25),
                f'relative bound mismatch in {row["name"]}')
        ratio = (absolute/relative) if relative > 0 else None
        require((ratio is None and bounds['absolute_to_relative_ratio'] is None) or
                (ratio is not None and math.isclose(ratio, bounds['absolute_to_relative_ratio'], rel_tol=1e-13)),
                f'speed-bound ratio mismatch in {row["name"]}')
        if ratio is not None:
            ratios[row['kind']].append(ratio)
        certificate_metadata.append({
            'case': row['name'], 'production_bound_m': absolute,
            'galilean_relative_bound_m': relative, 'ratio': ratio,
            'cpu_fp64_trace_terminal': reference['fp64_conservative_advance_comparison']['terminal'],
            'cpu_fp64_trace_certified': reference['fp64_conservative_advance_comparison']['certified'],
        })
    require(bytes(expected_binary) == raw, 'binary records do not reproduce from input metadata')
    summary = {
        'prepared_manifest_sha256': sha(prepared_path),
        'cases_bin_sha256': sha(binary_path),
        'case_count': len(rows),
        'capture_build_source_files_verified': checked,
        'case_counts': {'vertex_triangle': sum(r['kind_id']==0 for r in rows),
                        'edge_edge': sum(r['kind_id']==1 for r in rows)},
        'speed_bound_ratio_ranges': {k:{'min':min(v),'max':max(v)} for k,v in ratios.items()},
        'per_case': certificate_metadata,
        'native_result_admitted': False,
    }
    return d, summary

def parse_float4_bits(row, label: str):
    require(isinstance(row, list) and len(row) == 4, f'{label}: expected four float32 bit words')
    packed = bytearray()
    for word in row:
        require(isinstance(word, str) and re.fullmatch(r'[0-9a-f]{8}', word) is not None,
                f'{label}: malformed 32-bit float hex word {word!r}')
        packed.extend(struct.pack('<I', int(word, 16)))
    return bytes(packed), [struct.unpack('<f', packed[i:i+4])[0] for i in range(0, 16, 4)]

def native_scalar_matches_json(raw_value: float, json_value, label: str):
    if not math.isfinite(raw_value):
        require(json_value is None, f'{label}: nonfinite raw float must serialize as null')
        return
    require(isinstance(json_value, (int, float)) and not isinstance(json_value, bool),
            f'{label}: finite raw float must serialize as a JSON number')
    # NSNumber(float) serialization may choose a shortest decimal. Round the
    # decimal back to FP32 and compare bits rather than comparing widened
    # Python FP64 decimal values.
    require(struct.pack('<f', float(json_value)) == struct.pack('<f', raw_value),
            f'{label}: JSON scalar disagrees with its float32 bit word')

def validate_native(native_path: Path, prepared_path: Path, prepared: dict, summary: dict):
    result = strict_json(native_path)
    require(result['schema'] == 'numi.cardboard.source103_pair_ccd_native_results.v1', 'wrong native result schema')
    require(result['prepared_manifest_sha256'] == sha(prepared_path), 'native result binds different prepared JSON')
    require(result['cases_bin_sha256'] == prepared['case_contract']['input_sha256'], 'native result binds different cases.bin')
    for key in ('source_contact_sha256','source_common_sha256','source_main_shader_sha256'):
        require(result[key] == prepared[key], f'native result source mismatch: {key}')
    require(result['physical_transactions_executed'] == 0 and result['diagnostic_execution_only'] is True,
            'native run must be pure diagnostic with zero transactions')
    seal = strict_json(HERE / 'manifest.json')
    expected_metallib = (HERE / 'compile/diagnostic_pair.metallib').resolve()
    expected_metallib_sha = seal['compiled_artifacts']['compile/diagnostic_pair.metallib']
    require(sha(expected_metallib) == expected_metallib_sha,
            'local metallib differs from sealed package hash')
    reported_metallib = Path(result['metallib_path']).resolve()
    require(reported_metallib == expected_metallib,
            f'native result used unexpected metallib path: {reported_metallib}')
    require(result.get('metallib_sha256') == expected_metallib_sha,
            'native result metallib hash mismatch')
    cases = result['cases']
    require(len(cases) == len(prepared['cases']), 'native result case count mismatch')
    mismatches=[]
    for expected, actual in zip(prepared['cases'], cases):
        require(actual['name'] == expected['name'], f'native case order mismatch at {expected["name"]}')
        for key in ('kind_id','kind','feature_a','feature_b','capture','primitive_pair','candidate_slot','candidate_identity'):
            require(actual.get(key) == expected.get(key), f'native identity mismatch for {expected["name"]}:{key}')
        production = actual['production_result']
        clone = actual['trace_clone_result']
        production_rows = production.get('raw_float4_bits')
        clone_rows = clone.get('raw_float4_bits')
        require(isinstance(production_rows, list) and len(production_rows) == 4,
                f'{expected["name"]}: production raw record count must be 4')
        require(isinstance(clone_rows, list) and len(clone_rows) == 5,
                f'{expected["name"]}: clone raw record count must be 5 (includes speed bound)')
        prod_bytes=[]; prod_values=[]
        clone_bytes=[]; clone_values=[]
        for j,row in enumerate(production_rows):
            packed, vals = parse_float4_bits(row, f'{expected["name"]}:production[{j}]')
            prod_bytes.append(packed); prod_values.append(vals)
        for j,row in enumerate(clone_rows):
            packed, vals = parse_float4_bits(row, f'{expected["name"]}:clone[{j}]')
            clone_bytes.append(packed); clone_values.append(vals)
        # This exactly reproduces the source host contract: records 0..2 are
        # compared in full; record 3 compares only x (bary.z / param2). The
        # clone's record 4 is its extra speed-bound diagnostic, not a match
        # operand. Never trust the host's boolean field as the comparison.
        match = (prod_bytes[0] == clone_bytes[0] and
                 prod_bytes[1] == clone_bytes[1] and
                 prod_bytes[2] == clone_bytes[2] and
                 prod_bytes[3][:4] == clone_bytes[3][:4])
        require(actual.get('production_and_trace_clone_bitwise_equal') is match,
                f'{expected["name"]}: host equality flag disagrees with independent raw-bit comparison')
        require(match, f'{expected["name"]}: direct production and trace-clone result bits differ')
        for label, fields, values in (
                ('production', production, prod_values), ('trace clone', clone, clone_values)):
            scalar_fields=(('hit',0,0),('certified',0,1),('toi',0,2),('separation_m',0,3),
                    ('feature_parameter0',1,3),('feature_parameter1',2,3),('feature_parameter2',3,0))
            for key,row,col in scalar_fields:
                raw_value=values[row][col]
                if key in ('hit','certified'):
                    require(raw_value in (0.0,1.0) and fields[key] is (raw_value == 1.0),
                            f'{expected["name"]}:{label}.{key} disagrees with raw bits')
                else:
                    native_scalar_matches_json(raw_value,fields[key],f'{expected["name"]}:{label}.{key}')
            for key,row,cols in (('normal',1,3),('witness_point',2,3)):
                vals=values[row][:cols]
                actual_vals=fields[key]
                require(isinstance(actual_vals,list) and len(actual_vals)==len(vals),
                        f'{expected["name"]}:{label}.{key} shape mismatch')
                for j,(raw_value,json_value) in enumerate(zip(vals,actual_vals)):
                    native_scalar_matches_json(raw_value,json_value,
                        f'{expected["name"]}:{label}.{key}[{j}]')
        # Revalidate the reported iteration metadata against the raw summary.
        clone_iteration, clone_terminal, clone_effective = clone_values[3][1:4]
        clone_speed = clone_values[4][0]
        require(math.isfinite(clone_iteration) and clone_iteration.is_integer(),
                f'{expected["name"]}: iteration count is not an exact integer float')
        limit = 16 if expected['kind_id'] == 0 else 17
        iteration_count = int(clone_iteration)
        require(1 <= iteration_count <= limit, f'{expected["name"]}: iteration count out of range')
        require(clone['iteration_count'] == iteration_count, f'{expected["name"]}: iteration count disagrees with raw bits')
        require(math.isfinite(clone_terminal) and clone_terminal.is_integer() and int(clone_terminal) in range(1,7),
                f'{expected["name"]}: invalid terminal code')
        terminal=int(clone_terminal)
        require(clone['terminal_code'] == terminal, f'{expected["name"]}: terminal code disagrees with raw bits')
        native_scalar_matches_json(clone_effective, clone['effective_thickness_m'],
                                   f'{expected["name"]}:effective_thickness_m')
        native_scalar_matches_json(clone_speed, clone['speed_bound_m'], f'{expected["name"]}:speed_bound_m')
        require(math.isfinite(clone_effective) and clone_effective >= expected['thickness_m'],
                f'{expected["name"]}: effective thickness invalid')
        require(math.isfinite(clone_speed) and clone_speed >= 0.0,
                f'{expected["name"]}: speed bound invalid')
        production_hit=production['hit']; production_cert=production['certified']
        if terminal == 1:
            require(production_cert, f'{expected["name"]}: contact terminal is not certified')
            normal_finite=all(math.isfinite(v) for v in prod_values[1][:3])
            require(production_hit == normal_finite,
                    f'{expected["name"]}: hit return disagrees with finite-normal test')
        elif terminal in (2,3,4,5):
            require((not production_hit) and production_cert, f'{expected["name"]}: miss terminal disagrees with production status')
        else:
            require((not production_hit) and (not production_cert), f'{expected["name"]}: exhausted terminal disagrees with production status')
        trace_rows=actual['native_conservative_advance_trace']
        require(isinstance(trace_rows,list) and len(trace_rows)==iteration_count,
                f'{expected["name"]}: trace count differs from raw iteration count')
        require(len(trace_rows) <= limit, f'{expected["name"]}: trace over kind-specific iteration limit')
        for j,trace in enumerate(trace_rows):
            require(trace.get('iteration') == j, f'{expected["name"]}: trace indices are not contiguous')
            for key in ('time','distance_m','step'):
                x=trace[key]
                require(x is None or math.isfinite(x), f'nonfinite native trace {expected["name"]}:{key}')
            tc=trace.get('terminal_code')
            require(isinstance(tc,int) and tc in range(0,7), f'{expected["name"]}: invalid trace terminal word')
            if j < len(trace_rows)-1:
                require(tc == 0, f'{expected["name"]}: terminal marker appears before final trace row')
        final_trace_code=trace_rows[-1]['terminal_code']
        require(final_trace_code == (terminal if terminal < 6 else 0),
                f'{expected["name"]}: final trace marker disagrees with terminal code')
        native_scalar_matches_json(prod_values[0][2], trace_rows[-1]['time'],
                                   f'{expected["name"]}: final trace time')
        native_scalar_matches_json(prod_values[0][3], trace_rows[-1]['distance_m'],
                                   f'{expected["name"]}: final trace distance')
    all_match=(len(mismatches)==0)
    require(result.get('production_vs_trace_clone_all_bitwise_equal') is all_match,
            'overall host clone-equality flag disagrees with independently recomputed cases')
    summary['native_result_admitted'] = True
    summary['native_result_sha256'] = sha(native_path)
    summary['native_clone_bitwise_match'] = all_match
    summary['native_clone_bitwise_mismatch_count'] = len(set(mismatches))
    summary['native_clone_bitwise_mismatches'] = sorted(set(mismatches))
    summary['native_case_certification'] = [
        {'case':r['name'],'hit':r['production_result']['hit'],'certified':r['production_result']['certified'],
         'toi':r['production_result']['toi'],'separation_m':r['production_result']['separation_m'],
         'iteration_count':r['trace_clone_result']['iteration_count'],
         'terminal_code':r['trace_clone_result']['terminal_code'],
         'speed_bound_m':r['trace_clone_result']['speed_bound_m'],
         'effective_thickness_m':r['trace_clone_result']['effective_thickness_m']}
        for r in cases]
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prepared',type=Path,default=HERE/'prepared/prepared.json')
    ap.add_argument('--native-result',type=Path)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    prepared,summary=validate_prepared(args.prepared)
    if args.native_result:
        validate_native(args.native_result,args.prepared,prepared,summary)
    report={
      'schema':'numi.cardboard.source103_pair_ccd_package_audit.v1',
      'status':'PASS',
      'scope':'CPU package validation; native result is optional and is only parsed if supplied',
      'summary':summary,
      'limits':['No GPU is dispatched by this validator.','The FP64 CCD reference is not a formal certificate.',
                'The Galilean-relative bound is a mathematical hypothesis without outward FP32 error accounting.',
                'No production CCD, thickness, timestep, or acceptance logic is changed.'],
    }
    text=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text)
    print(text,end='')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        print(f'source103_pair_ccd_package_audit=FAIL {exc}',file=sys.stderr)
        sys.exit(1)
