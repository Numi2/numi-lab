#!/usr/bin/env python3
"""Independent, read-only audit of source380 tet 674 input/state provenance."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import struct
import sys

import numpy as np

BASE = pathlib.Path('/Users/home/cardboard-implementation-20261014')
RUNS = {
    '370': BASE / 'solver/cached-true-residual-certification-coupon-370',
    '380': BASE / 'solver/explicit32-certified-coupon-380',
}
OLD = BASE / 'analysis/source380-first-failure-tet674-001'
OUT = pathlib.Path(__file__).resolve().parent
SOURCE = BASE / 'snapshots/source-explicit-material-budget-109'
TET = 674
ENV = 0
NODE_IDS_EXPECTED = [228, 524, 522, 525]
NMMATERIAL_BYTES = 160
MATERIAL_BULK_OFFSET = 96
MATERIAL_VALIDITY_OFFSET = 144
MATERIAL_LOCAL_ITERATIONS_OFFSET = 80
NMFEM_NODE_BYTES = 80


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: pathlib.Path) -> str:
    return sha_bytes(path.read_bytes())


def load_json(path: pathlib.Path):
    return json.loads(path.read_text())


def package_sections(package: bytes):
    header = struct.unpack_from('<16sIIIIQQQ', package, 0)
    section_count = header[4]
    cursor = 56
    sections = {}
    for _ in range(section_count):
        sid, stride, count, nbytes, _checksum = struct.unpack_from('<IIQQQ', package, cursor)
        cursor += 32
        data = package[cursor:cursor + nbytes]
        if len(data) != nbytes or sid in sections:
            raise ValueError('malformed or duplicate compiled-package section')
        sections[sid] = (stride, count, data)
        cursor += nbytes
    if cursor != len(package):
        raise ValueError('compiled package section table does not end at file size')
    return header, sections


def load_replay_helpers():
    path = OLD / 'replay_tet674.py'
    spec = importlib.util.spec_from_file_location('source380_replay_helpers', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('could not load source-bound deformation transcription')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def f32_bits(values):
    a = np.asarray(values, dtype='<f4')
    return a.view('<u4').reshape(a.shape).tolist()


def main() -> int:
    run_data = {}
    parsed = {}
    for label, run in RUNS.items():
        output = run / 'output'
        cond = output / 'rejected-conditioning-000002'
        candidate = (cond / 'candidate_nodes.bin').read_bytes()
        accepted_bytes = (output / 'material_state_000002_bent.json').read_bytes()
        package = (output / 'compiled.nmatterpack').read_bytes()
        header, sections = package_sections(package)
        manifest = load_json(cond / 'manifest.json')
        accepted = json.loads(accepted_bytes)
        run_data[label] = {
            'run': str(run),
            'receipt': load_json(run / 'receipt.json'),
            'source_physics_fingerprint': manifest['source_physics_fingerprint'],
            'device_program_fingerprint': manifest['device_program_fingerprint'],
            'candidate_nodes': candidate,
            'accepted_bytes': accepted_bytes,
            'package': package,
            'package_header': header,
            'sections': sections,
            'manifest': manifest,
            'accepted': accepted,
            'input_hashes': {
                'receipt': sha_file(run / 'receipt.json'),
                'prediction': sha_file(run / 'prediction.json'),
                'compiled_nmatterpack': sha_bytes(package),
                'result': sha_file(output / 'result.json'),
                'output_manifest': sha_file(output / 'manifest.json'),
                'candidate_nodes': sha_bytes(candidate),
                'conditioning_manifest': sha_file(cond / 'manifest.json'),
                'accepted_material_state': sha_bytes(accepted_bytes),
            },
        }
        if manifest.get('accepted_physical_state') is not False:
            raise ValueError(f'run {label} candidate is not marked rejected')
        if len(candidate) != manifest['environment_count'] * manifest['fem_node_count'] * NMFEM_NODE_BYTES:
            raise ValueError(f'run {label} candidate buffer dimensions do not match manifest')
        if accepted['status_code'] != 5 or accepted['solver_certificate_raw_accepted_flag'] != 0:
            raise ValueError(f'run {label} is not the expected rejected/rolled-back export')

    input_path = OLD / 'native_point_input_674.json'
    point_doc = load_json(input_path)
    if point_doc.get('schema') != 'numi.cardboard.bilinear15-native-captured-point-inputs.v1' or len(point_doc.get('cases', [])) != 1:
        raise ValueError('selected native point input schema/cardinality changed')
    point = point_doc['cases'][0]
    if point['tet_id'] != TET:
        raise ValueError('selected native point input targets the wrong tet')

    # Validate the source package record and float4 parameter table by byte offsets.
    h380, sections380 = run_data['380']['package_header'], run_data['380']['sections']
    mat_stride, mat_count, mat_bytes = sections380[3]
    param_stride, param_count, param_bytes = sections380[6]
    tet_stride, tet_count, tet_bytes = sections380[20]
    if (mat_stride, mat_count, param_stride, tet_stride, tet_count) != (160, 3, 16, 96, 1920):
        raise ValueError('unexpected NMMaterialGPU/parameter/tet section dimensions')
    material_record = mat_bytes[1 * mat_stride:2 * mat_stride]
    words = struct.unpack_from('<18I', material_record, 0)
    parameter_offset, parameter_count, state_count = words[1:4]
    effective_iterations = struct.unpack_from('<I', material_record, MATERIAL_LOCAL_ITERATIONS_OFFSET)[0]
    bulk = struct.unpack_from('<4f', material_record, MATERIAL_BULK_OFFSET)
    validity = struct.unpack_from('<4f', material_record, MATERIAL_VALIDITY_OFFSET)
    if (parameter_count, state_count, effective_iterations) != (28, 15, 32):
        raise ValueError('run380 compiled material1 counts/budget do not match intended selected point')
    if bulk != (679.0, 293.1499938964844, 0.0, 0.0):
        raise ValueError(f'packed material1 bulk differs from expected exact float4: {bulk!r}')
    if not validity[0] <= 1.0003025531768799 <= validity[1]:
        raise ValueError('captured native J lies outside material validity bounds')
    if struct.unpack_from('<f', material_record, MATERIAL_BULK_OFFSET + 4)[0] != point['temperature_K']:
        raise ValueError('point input temperature is not the exact packed bulk.y value')
    if struct.unpack_from('<f', material_record, MATERIAL_BULK_OFFSET)[0] != point['parameters'][0]:
        raise ValueError('point input first constitutive parameter does not match packed density')
    parameter_records = np.frombuffer(param_bytes, dtype='<f4').reshape(-1, 4)
    compiled_parameters = parameter_records[parameter_offset:parameter_offset + parameter_count, 0]
    if not np.array_equal(compiled_parameters.view('<u4'), np.asarray(point['parameters'], dtype='<f4').view('<u4')):
        raise ValueError('point parameter array does not bit-match compiled float4 x-lanes')

    # Compare all recorded nodes and the whole accepted-state export across 370/380.
    same_candidate = run_data['370']['candidate_nodes'] == run_data['380']['candidate_nodes']
    same_accepted_export = run_data['370']['accepted_bytes'] == run_data['380']['accepted_bytes']
    if not same_candidate or not same_accepted_export:
        raise ValueError('source370/source380 candidate or accepted export differs')
    candidate = run_data['380']['candidate_nodes']
    acc380 = run_data['380']['accepted']
    tet380 = sections380[20][2][TET * tet_stride:(TET + 1) * tet_stride]
    node_ids = list(struct.unpack_from('<4I', tet380, 0))
    if node_ids != NODE_IDS_EXPECTED:
        raise ValueError(f'compiled tet node ids changed: {node_ids!r}')
    u = np.asarray([
        struct.unpack_from('<3f', candidate, (ENV * 882 + node) * NMFEM_NODE_BYTES + 64)
        for node in node_ids
    ], dtype=np.float32)
    v = np.asarray([
        struct.unpack_from('<3f', candidate, (ENV * 882 + node) * NMFEM_NODE_BYTES + 16)
        for node in node_ids
    ], dtype=np.float32)
    inv = np.asarray([
        struct.unpack_from('<3f', tet380, 16 + 16 * row) for row in range(3)
    ], dtype=np.float32)
    quaternion = np.asarray(struct.unpack_from('<4f', tet380, 80), dtype=np.float32)
    helper = load_replay_helpers()
    frame = helper.frame_from_quaternion(quaternion)
    f_high, f_low = helper.source_deformation_words(u, inv, frame)
    input_high = np.asarray(point['F_high'], dtype=np.float32).reshape(3, 3)
    input_low = np.asarray(point['F_low'], dtype=np.float32).reshape(3, 3)
    if not (np.array_equal(f_high.view('<u4'), input_high.view('<u4')) and np.array_equal(f_low.view('<u4'), input_low.view('<u4'))):
        raise ValueError('selected input F high/low words do not reproduce from captured geometry')
    f_widened = f_high.astype(np.float64) + f_low.astype(np.float64)
    source_f_det = float(np.linalg.det(f_widened))
    native_j = 1.0003025531768799
    source_geometry_j = float(np.linalg.det(
        np.eye(3) + np.column_stack((u[1] - u[0], u[2] - u[0], u[3] - u[0])).astype(np.float64) @ inv.astype(np.float64)
    ))
    # The accepted state comes from post-rollback export and must be the exact
    # constitutive input stored by the point file after f32 round trip.
    accepted_state = np.asarray(acc380['tetrahedra'][TET]['state'], dtype=np.float32)
    point_accepted = np.asarray(point['accepted_state'], dtype=np.float32)
    if not np.array_equal(accepted_state.view('<u4'), point_accepted.view('<u4')):
        raise ValueError('point accepted history does not match post-rollback tet 674 state')

    # Compare package physics except for the intentional material-local budget bump.
    s370, s380 = run_data['370']['sections'], run_data['380']['sections']
    section_equal = {str(sid): s370[sid] == s380[sid] for sid in sorted(s370)}
    if not all(section_equal[str(sid)] for sid in section_equal if sid != '3'):
        raise ValueError('compiled package differs outside material descriptor section 3')
    descriptors = []
    for index in range(3):
        a = s370[3][2][index * NMMATERIAL_BYTES:(index + 1) * NMMATERIAL_BYTES]
        b = s380[3][2][index * NMMATERIAL_BYTES:(index + 1) * NMMATERIAL_BYTES]
        diffs = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        changed_words = []
        for byte_offset in sorted({(i // 4) * 4 for i in diffs}):
            changed_words.append({
                'byte_offset': byte_offset,
                'run370_u32': struct.unpack_from('<I', a, byte_offset)[0],
                'run380_u32': struct.unpack_from('<I', b, byte_offset)[0],
            })
        descriptors.append({'material_index': index, 'changed_u32_words': changed_words})
    if descriptors[0]['changed_u32_words'] != [{'byte_offset': 80, 'run370_u32': 16, 'run380_u32': 32}] or descriptors[1]['changed_u32_words'] != [{'byte_offset': 80, 'run370_u32': 16, 'run380_u32': 32}] or descriptors[2]['changed_u32_words']:
        raise ValueError('run370/run380 material descriptor differences are not only intended local iteration budget words')

    paths = [
        OLD / 'native_point_input_674.json', OLD / 'native_point_input_674_summary.json',
        OLD / 'replay.json', OLD / 'replay_tet674.py', OLD / 'make_point_input.py',
        SOURCE / 'matter/include/numi/matter/shared.h', SOURCE / 'matter/src/metal/fem.metalinc',
        SOURCE / 'matter/src/metal/common.metalinc', SOURCE / 'matter/materials/hajali2009_medium_hill_bilinear_fit.nmatter',
    ]
    artifact_hashes = {str(p): sha_file(p) for p in paths}
    for label, run in RUNS.items():
        artifact_hashes.update({f'{label}:{key}': value for key, value in run_data[label]['input_hashes'].items()})

    # This old human/machine-readable field contradicts the numeric temperature.
    stale_semantics = point_doc.get('case_count_semantics', '')
    report = {
        'schema': 'numi.cardboard.source380_tet674_corrected_offset_independent_audit.v1',
        'classification': 'Independent CPU/read-only source and artifact audit; no GPU or native point run; not a diagnosis of the remaining native failure.',
        'source_bindings': artifact_hashes,
        'run_comparison': {
            'candidate_nodes_byte_identical_370_380': same_candidate,
            'candidate_nodes_bytes': len(candidate),
            'candidate_nodes_sha256': sha_bytes(candidate),
            'entire_post_rollback_material_state_export_byte_identical_370_380': same_accepted_export,
            'accepted_state_export_bytes': len(run_data['380']['accepted_bytes']),
            'accepted_state_export_sha256': sha_bytes(run_data['380']['accepted_bytes']),
            'tet674_accepted_state_bit_identical_after_fp32_roundtrip': True,
            'compiled_geometry_parameter_sections_equal': {str(sid): section_equal[str(sid)] for sid in [2, 5, 6, 20, 21]},
            'material_descriptors_differ_only_in_local_newton_budget': descriptors,
            'run370_effective_material_iteration_budget': 16,
            'run380_effective_material_iteration_budget': 32,
            'run370_source_physics_fingerprint': run_data['370']['source_physics_fingerprint'],
            'run380_source_physics_fingerprint': run_data['380']['source_physics_fingerprint'],
            'both_runs_failed_before_physical_acceptance': True,
        },
        'packed_material_offset_check': {
            'source_layout': 'NMMaterialGPU: stateTransferMask aligns at byte 72, localNewtonIterations byte 80, bulk float4 aligned at byte 96, validity float4 byte 144; size 160 bytes.',
            'material_index': 1,
            'record_size_bytes': mat_stride,
            'parameter_offset_float4': parameter_offset,
            'parameter_count': parameter_count,
            'state_count': state_count,
            'local_newton_iterations_offset_bytes': MATERIAL_LOCAL_ITERATIONS_OFFSET,
            'local_newton_iterations': effective_iterations,
            'bulk_offset_bytes': MATERIAL_BULK_OFFSET,
            'bulk_float4': list(bulk),
            'temperature_offset_bytes': MATERIAL_BULK_OFFSET + 4,
            'point_input_temperature_K': point['temperature_K'],
            'temperature_f32_bits': f'0x{struct.unpack_from("<I", material_record, MATERIAL_BULK_OFFSET + 4)[0]:08x}',
            'validity_offset_bytes': MATERIAL_VALIDITY_OFFSET,
            'validity_float4': list(validity),
            'point_parameters_match_compiled_float4_x_lanes_bitwise': True,
            'input_numeric_temperature_is_correct_packed_bulk_y': True,
            'stale_case_count_semantics_text': stale_semantics,
            'stale_text_mentions_zero_kelvin_despite_numeric_value': '0 K' in stale_semantics or '0 Kelvin' in stale_semantics,
        },
        'tet674_state_and_deformation_check': {
            'tetrahedron': TET,
            'environment': ENV,
            'material_index': 1,
            'node_ids': node_ids,
            'candidate_displacements_f32': u.tolist(),
            'candidate_velocities_f32_m_per_s': v.tolist(),
            'candidate_velocities_all_zero': bool(np.all(v == 0.0)),
            'accepted_history_f32_bits': [f'0x{x:08x}' for x in accepted_state.view('<u4').tolist()],
            'source_transcribed_F_high_f32_bits': [[f'0x{x:08x}' for x in row] for row in f32_bits(f_high)],
            'source_transcribed_F_low_f32_bits': [[f'0x{x:08x}' for x in row] for row in f32_bits(f_low)],
            'source_F_pair_reproduces_selected_input_bitwise': True,
            'source_F_pair_is_native_bit_confirmed': False,
            'source_F_high_low_widened_determinant': source_f_det,
            'source_geometry_determinant_from_candidate_nodes': source_geometry_j,
            'native_status_diagnostic_J': native_j,
            'source_F_determinant_minus_native_J': source_f_det - native_j,
            'source_geometry_J_minus_native_J': source_geometry_j - native_j,
            'native_J_inside_material_validity': validity[0] <= native_j <= validity[1],
            'native_failure_status': 'NM_STATUS_INVALID_DEFORMATION (5), Newton 0, FGMRES 0, no accepted candidate; the exported diagnostic gives J/max stress but not point-local terminal reason.',
        },
        'corrections_to_prior_record': {
            'prior_replay_diagnostic_det_difference': load_json(OLD / 'replay.json')['failed_record']['diagnostic_det_difference'],
            'prior_record_difference_is_wrong': True,
            'prior_difference_recomputed_from_candidate_geometry_and_native_J': source_geometry_j - native_j,
            'cause': 'The old replay field subtracts 0.99774569272994995 instead of the native status J 1.0003025531768799. Its 0.0025567206 value is not the candidate-to-native-J difference; the direct source-geometry difference is about -1.3981e-7.',
            'prior_point_input_zero_temperature_text_is_stale': '0 K' in stale_semantics,
            'numeric_temperature_is_nonzero_and_verified': True,
        },
        'limitations': [
            'Source-transcribed F high/low words are bitwise reproducible from captured nodes, packed inverse rest, and material quaternion using the CPU transcription, but have not been confirmed by native instrumentation.',
            'Native diagnostic J is within validity bounds and close to the source transcription; the exported status does not identify whether material projection, stress validity, or another evaluator gate caused status 5.',
            'The source380 CPU FP64 projected material state converges in four updates with budget 32, but that is not a native projection replay and does not establish the failure cause.',
            'Source370 and source380 have different physics fingerprints and iteration budgets; their exact rejected node buffers and accepted-state exports match, so the observed first-failure state is the same. This is not a controlled causal attribution to the budget change.',
        ],
    }
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({
        'output': str(OUT / 'report.json'),
        'candidate_nodes_equal': same_candidate,
        'accepted_state_export_equal': same_accepted_export,
        'temperature_K': bulk[1],
        'compiled_budget': effective_iterations,
        'source_F_det': source_f_det,
        'native_J': native_j,
        'source_F_det_minus_native_J': source_f_det - native_j,
        'source_geometry_J_minus_native_J': source_geometry_j - native_j,
        'source_geometry_delta_old_recorded': load_json(OLD / 'replay.json')['failed_record']['diagnostic_det_difference'],
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
