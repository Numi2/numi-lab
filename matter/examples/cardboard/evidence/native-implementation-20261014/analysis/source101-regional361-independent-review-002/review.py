#!/usr/bin/env python3
"""Independent CPU review of source101 selector-5 capture 361.

Reads only retained run/package data. Reconstructs the compiled-world graph
layout from the packed FEM node mask and tetrahedron connectivity, then checks
all captured maps, AP/RAP, and graph LU factors. This is not a solver run.
"""
from __future__ import annotations
import hashlib
import json
import math
import struct
from pathlib import Path
import numpy as np
from scipy import sparse

BASE = Path('/Users/home/cardboard-implementation-20261014')
OUT = BASE / 'analysis/source101-regional361-independent-review-002'
CAPTURE_RUN = BASE / 'solver/reset-augmentation-first-failure-rap-only-361'
CONTROL_RUN = BASE / 'solver/reset-augmentation-first-failure-capture-360'
PACK_PATH = CAPTURE_RUN / 'output/compiled.nmatterpack'
OPERATOR_DIR = BASE / 'analysis/source101-control2-newton23-audit-004/result-360'
SOURCE = BASE / 'snapshots/source-boundary-reset-order-101'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha(path.read_bytes())


def fnv64(data: bytes, seed: int = 1469598103934665603) -> int:
    value = seed
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xffffffffffffffff
    return value or 1


def fnv32_words(groups: list[np.ndarray]) -> int:
    value = 2166136261
    for group in groups:
        for word in np.asarray(group, dtype='<u4').reshape(-1):
            value = ((value ^ int(word)) * 16777619) & 0xffffffff
    return value


def parse_pack(path: Path) -> tuple[dict, dict[int, dict]]:
    data = path.read_bytes()
    if len(data) < 56:
        raise ValueError('compiled package is shorter than its v19 header')
    header = struct.unpack_from('<16sIIIIQQQ', data, 0)
    magic, version, endian, abi, section_count, physics_fp, world_fp, header_hash = header
    if magic != b'NUMIMATTERPKG\0\0\0' or version != 19 or endian != 0x01020304:
        raise ValueError('compiled package magic/version/endian mismatch')
    header_zeroed = bytearray(data[:56])
    header_zeroed[48:56] = b'\0' * 8
    if fnv64(header_zeroed) != header_hash:
        raise ValueError('compiled package header checksum mismatch')
    sections: dict[int, dict] = {}
    cursor = 56
    for _ in range(section_count):
        if cursor + 32 > len(data):
            raise ValueError('truncated package section header')
        section_id, element_size, element_count, byte_count, content_hash = struct.unpack_from(
            '<IIQQQ', data, cursor)
        cursor += 32
        if section_id in sections or byte_count != element_size * element_count:
            raise ValueError(f'duplicate or malformed package section {section_id}')
        raw = data[cursor:cursor + byte_count]
        if len(raw) != byte_count or fnv64(raw) != content_hash:
            raise ValueError(f'package section {section_id} extent/hash mismatch')
        sections[section_id] = {
            'element_size': element_size,
            'element_count': element_count,
            'byte_count': byte_count,
            'content_hash': content_hash,
            'raw': raw,
        }
        cursor += byte_count
    if cursor != len(data):
        raise ValueError('compiled package has trailing or unindexed bytes')
    if len(sections) != section_count or 19 not in sections or 20 not in sections:
        raise ValueError('compiled package lacks FEM node/tetrahedron sections')
    return {
        'sha256': sha(data),
        'version': version,
        'matter_abi_version': abi,
        'section_count': section_count,
        'physics_fingerprint_u64': physics_fp,
        'world_fingerprint_u64': world_fp,
        'header_hash_valid': True,
        'all_section_hashes_valid': True,
        'bytes': len(data),
    }, sections


def read_capture_arrays(manifest_path: Path, payload_path: Path) -> tuple[dict, dict[str, np.ndarray], dict[str, bytes]]:
    manifest = json.loads(manifest_path.read_text())
    payload = payload_path.read_bytes()
    if manifest.get('payload', {}).get('byte_count') != len(payload):
        raise ValueError('capture payload byte count mismatch')
    if manifest.get('payload', {}).get('sha256') != sha(payload):
        raise ValueError('capture payload SHA-256 mismatch')
    arrays: dict[str, np.ndarray] = {}
    raw_arrays: dict[str, bytes] = {}
    cursor = 0
    dtype_map = {'<f4': '<f4', '<u4': '<u4', '<f4x4': '<f4', '<u4x4': '<u4'}
    for spec in manifest.get('arrays', []):
        name = spec['name']
        offset = int(spec['offset'])
        byte_count = int(spec['byte_count'])
        element_bytes = int(spec['element_bytes'])
        shape = tuple(int(x) for x in spec['shape'])
        count = math.prod(shape)
        if offset != cursor or byte_count != count * element_bytes:
            raise ValueError(f'noncontiguous or malformed array extent: {name}')
        raw = payload[offset:offset + byte_count]
        if len(raw) != byte_count or sha(raw) != spec['sha256']:
            raise ValueError(f'array payload hash mismatch: {name}')
        raw_arrays[name] = raw
        if spec['dtype'] in dtype_map:
            dtype = dtype_map[spec['dtype']]
            shape_lanes = shape + ((4,) if spec['dtype'].endswith('x4') else ())
            arrays[name] = np.frombuffer(raw, dtype=dtype).reshape(shape_lanes)
        cursor += byte_count
    if cursor != len(payload):
        raise ValueError('array table does not consume the exact payload')
    return manifest, arrays, raw_arrays


def rebuild_layout(tetrahedra: np.ndarray, free_mask: np.ndarray) -> dict[str, np.ndarray]:
    """Independent deterministic implementation of source101's two partitions."""
    node_count = len(free_mask)
    adjacency = [set() for _ in range(node_count)]
    for tet in tetrahedra.tolist():
        if len(set(tet)) != 4 or any(node < 0 or node >= node_count for node in tet):
            raise ValueError('compiled tetrahedron has invalid node ids')
        for i, node in enumerate(tet):
            for j, other in enumerate(tet):
                if i != j and free_mask[node] == free_mask[other]:
                    adjacency[node].add(other)
    patch_nodes: list[list[int]] = []
    covered = [False] * node_count
    for partition in (
        [i for i, is_free in enumerate(free_mask) if is_free],
        [i for i, is_free in enumerate(free_mask) if not is_free],
    ):
        if not partition:
            continue
        anchors = partition[::128]
        if anchors[-1] != partition[-1]:
            anchors.append(partition[-1])
        for anchor in anchors:
            selected = [anchor]
            seen = {anchor}
            frontier = [anchor]
            head = 0
            while head < len(frontier) and len(selected) < 256:
                current = frontier[head]
                head += 1
                for neighbor in sorted(adjacency[current]):
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    selected.append(neighbor)
                    frontier.append(neighbor)
                    if len(selected) == 256:
                        break
            selected.sort()
            patch_nodes.append(selected)
            for node in selected:
                covered[node] = True
        for node in partition:
            if not covered[node]:
                patch_nodes.append([node])
    patches: list[list[int]] = []
    rows: list[list[int]] = []
    flat_nodes: list[int] = []
    coverage = np.zeros(node_count, dtype='<u4')
    matrix_offset = pivot_offset = 0
    for patch_index, nodes in enumerate(patch_nodes):
        patches.append([len(flat_nodes), len(nodes), matrix_offset, pivot_offset])
        for local_node, node in enumerate(nodes):
            flat_nodes.append(node)
            coverage[node] += 1
            rows.extend([[patch_index, local_node, axis, node] for axis in range(3)])
        dofs = 3 * len(nodes)
        matrix_offset += dofs * dofs
        pivot_offset += dofs
    offsets = np.zeros(node_count + 1, dtype='<u4')
    for node in flat_nodes:
        offsets[node + 1] += 1
    np.cumsum(offsets, out=offsets)
    values = np.zeros(len(flat_nodes), dtype='<u4')
    cursor = offsets[:-1].copy()
    for value_index, node in enumerate(flat_nodes):
        values[cursor[node]] = value_index
        cursor[node] += 1
    return {
        'patches': np.asarray(patches, dtype='<u4'),
        'patch_nodes': np.asarray(flat_nodes, dtype='<u4'),
        'rows': np.asarray(rows, dtype='<u4'),
        'node_patch_offsets': offsets,
        'node_patch_values': values,
        'node_coverage': coverage,
    }


def lu_reconstruct(lu: np.ndarray, pivots: np.ndarray) -> tuple[np.ndarray, bool]:
    n = len(pivots)
    if not np.array_equal(np.sort(pivots), np.arange(n)):
        raise ValueError('captured pivot vector is not a permutation')
    lower = np.tril(lu, -1)
    lower[np.diag_indices(n)] = 1.0
    permuted = np.einsum('ik,kj->ij', lower, np.triu(lu), optimize=False)
    reconstructed = np.empty_like(permuted)
    # Source swaps pivots alongside rows, so pivots[factor_row] is its original row.
    reconstructed[pivots] = permuted
    return reconstructed, True


def rel_metrics(actual: np.ndarray, reference: np.ndarray) -> dict:
    delta = np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    ref_norm = float(np.linalg.norm(reference))
    return {
        'relative_frobenius_error': float(np.linalg.norm(delta) / max(ref_norm, 1.0e-300)),
        'absolute_frobenius_error': float(np.linalg.norm(delta)),
        'maximum_absolute_error': float(np.max(np.abs(delta))),
        'reference_frobenius_norm_denominator': ref_norm,
        'actual_frobenius_norm': float(np.linalg.norm(actual)),
        'relative_denominator_floor': 1.0e-300,
    }


def normalized_argv(prediction: dict) -> list[str]:
    argv = list(prediction['argv'])
    index = argv.index('--output')
    argv[index + 1] = '<output>'
    return argv


def run_review() -> dict:
    cap_path = CAPTURE_RUN / 'operator-capture.regional.json'
    payload_path = CAPTURE_RUN / 'operator-capture.regional.bin'
    manifest, arrays, raw_arrays = read_capture_arrays(cap_path, payload_path)
    if (manifest['root'], manifest['iteration'], manifest['environment_count'],
        manifest['node_count'], manifest['tetrahedron_count'], manifest['mode_count']) != (2, 23, 2, 882, 1920, 26):
        raise ValueError('unexpected selector/root/iteration or capture dimensions')
    if manifest['capture_point'] != 'after_regional_rap_and_graph_factor_barrier_before_fgmres_correction':
        raise ValueError('capture point is not the requested same-Newton RAP/factor point')
    if not manifest['tangent_cache_used'] or not np.all(arrays['algorithmic_tangent_cache_valid'] == 1):
        raise ValueError('capture does not have a complete valid tangent cache')
    if not np.all(arrays['scheduler_exponents'] == 0):
        raise ValueError('unexpected scheduler exponents')

    # Verify exact package structure and source geometry/mask from the actual compiled world.
    pack_info, sections = parse_pack(PACK_PATH)
    node_section = sections[19]
    tet_section = sections[20]
    if (node_section['element_size'], node_section['element_count'],
        tet_section['element_size'], tet_section['element_count']) != (80, 882, 96, 1920):
        raise ValueError('compiled FEM section ABI/layout mismatch')
    node_raw = node_section['raw']
    source_fixed = np.fromiter(
        (struct.unpack_from('<f', node_raw, i * 80 + 44)[0] for i in range(882)),
        dtype=np.float32,
    )
    if not np.all(np.isin(source_fixed, (0.0, 1.0))):
        raise ValueError('compiled package FEM fixed tags are not binary')
    source_free_mask = (source_fixed == 0.0).astype(np.uint8)
    tet_raw = tet_section['raw']
    tetrahedra = np.empty((1920, 4), dtype='<u4')
    for i in range(1920):
        tetrahedra[i] = struct.unpack_from('<4I', tet_raw, i * 96)
    source_mask_sha = sha(source_free_mask.tobytes())
    topology_bytes = struct.pack('<II', 882, 1920) + tetrahedra.tobytes()
    source_topology_sha = sha(topology_bytes)
    if source_mask_sha != manifest['graph_schwarz']['source_free_mask_sha256']:
        raise ValueError('compiled package mask digest does not match native graph manifest')
    if source_topology_sha != manifest['graph_schwarz']['source_topology_sha256']:
        raise ValueError('compiled package topology digest does not match native graph manifest')
    rebuilt = rebuild_layout(tetrahedra, source_free_mask.astype(bool))
    captured_layout = {
        'patches': arrays['graph_patches'],
        'patch_nodes': arrays['graph_patch_nodes'],
        'rows': arrays['graph_rows'],
        'node_patch_offsets': arrays['graph_node_patch_offsets'],
        'node_patch_values': arrays['graph_node_patch_values'],
        'node_coverage': arrays['graph_node_coverage'],
    }
    layout_equal = {key: bool(np.array_equal(rebuilt[key], captured_layout[key])) for key in rebuilt}
    if not all(layout_equal.values()):
        raise ValueError(f'independent compiled-world layout does not match: {layout_equal}')
    checksum = fnv32_words([
        rebuilt['patches'], rebuilt['rows'], rebuilt['patch_nodes'],
        rebuilt['node_patch_offsets'], rebuilt['node_patch_values'],
    ])
    if checksum != manifest['graph_schwarz']['layout_checksum_fnv1a32']:
        raise ValueError('independent layout checksum does not match captured checksum')
    potential_mask = (rebuilt['node_coverage'] > 0).astype(np.uint8)
    potential_sha = sha(potential_mask.tobytes())
    if potential_sha != manifest['graph_schwarz']['potential_node_mask_sha256']:
        raise ValueError('independent potential-node mask does not match capture')
    if not np.all(rebuilt['node_coverage'] > 0):
        raise ValueError('compiled graph leaves a node outside potential coverage')

    # Independently recompute the sidecar's same-generation digest for environment 0.
    digest_names = ['candidate_nodes', 'material_history', 'scheduler_exponents',
                    'deformation_high', 'deformation_low',
                    'algorithmic_tangent_cache', 'algorithmic_tangent_cache_valid']
    # Environment slices are contiguous within each environment-major array.
    specs = {spec['name']: spec for spec in manifest['arrays']}
    env0_chunks = []
    for name in digest_names:
        spec = specs[name]
        per_env = int(spec['byte_count']) // manifest['environment_count']
        env0_chunks.append(raw_arrays[name][:per_env])
    recomputed_linearization_sha = sha(b''.join(env0_chunks))
    expected_linearization_sha = next(
        item['sha256'] for item in manifest['candidate_linearization_sha256']
        if item['environment'] == 0)
    candidate_node_sha = sha(env0_chunks[0])
    candidate_metadata = next(
        item for item in manifest['candidate_linearization_sha256']
        if item['environment'] == 0)
    if recomputed_linearization_sha != expected_linearization_sha or \
        candidate_node_sha != candidate_metadata['candidate_nodes_sha256']:
        raise ValueError('candidate linearization digest does not recompute from payload bytes')

    # Compare run 360 and 361 source/executable inputs, output bytes and physical traces.
    pred360 = json.loads((CONTROL_RUN / 'prediction.json').read_text())
    pred361 = json.loads((CAPTURE_RUN / 'prediction.json').read_text())
    if normalized_argv(pred360) != normalized_argv(pred361):
        raise ValueError('360/361 argv differs beyond output directory')
    inputs_equal = pred360['inputs_sha256'] == pred361['inputs_sha256']
    if not inputs_equal:
        raise ValueError('360/361 sealed input hash maps differ')
    env_differences = {
        key: [pred360['environment'].get(key), pred361['environment'].get(key)]
        for key in sorted(set(pred360['environment']) | set(pred361['environment']))
        if pred360['environment'].get(key) != pred361['environment'].get(key)
    }
    allowed_diagnostic_env = {
        'NM_FEM_OPERATOR_FD_CAPTURE_PATH', 'NM_FEM_REGIONAL_RAP_CAPTURE_ONLY',
        'NM_FEM_NEWTON_TRACE_ROOT', 'NM_FEM_TRUE_RESIDUAL_ROOT',
        'NM_FEM_TRUE_RESIDUAL_RESTART', 'NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR',
    }
    if not set(env_differences).issubset(allowed_diagnostic_env):
        raise ValueError(f'360/361 environment has undeclared differences: {env_differences}')
    staged_inputs_verified = all(
        sha_file(CAPTURE_RUN / 'inputs' / relative) == expected
        for relative, expected in pred361['inputs_sha256'].items()
    )
    if not staged_inputs_verified:
        raise ValueError('one or more retained 361 input bytes do not match prediction SHA map')
    build_path_361 = Path(pred361['build_provenance'])
    build_path_360 = Path(pred360['build_provenance'])
    if build_path_360 != build_path_361 or sha_file(build_path_361) != sha_file(build_path_360):
        raise ValueError('360/361 build provenance path or bytes differ')
    if pred360['source_snapshot'] != pred361['source_snapshot'] or pred360['runtime_dir'] != pred361['runtime_dir']:
        raise ValueError('360/361 source snapshot or runtime directory differs')
    if not pred361.get('executed_build_verified', False):
        raise ValueError('361 prediction does not record an executed build verification')
    output_files_360 = sorted(p.relative_to(CONTROL_RUN / 'output') for p in
        (CONTROL_RUN / 'output').rglob('*') if p.is_file())
    output_files_361 = sorted(p.relative_to(CAPTURE_RUN / 'output') for p in
        (CAPTURE_RUN / 'output').rglob('*') if p.is_file())
    if output_files_360 != output_files_361:
        raise ValueError('360/361 ordinary output file sets differ')
    output_equal = all(
        sha_file(CONTROL_RUN / 'output' / name) == sha_file(CAPTURE_RUN / 'output' / name)
        for name in output_files_360)
    if not output_equal:
        raise ValueError('360/361 ordinary output files differ')
    output_hashes = {str(name): sha_file(CAPTURE_RUN / 'output' / name) for name in output_files_361}
    output_manifest = json.loads((CAPTURE_RUN / 'output/manifest.json').read_text())
    result360 = json.loads((CONTROL_RUN / 'output/result.json').read_text())
    result361 = json.loads((CAPTURE_RUN / 'output/result.json').read_text())
    if (output_manifest.get('compiled_world_fingerprint') != pack_info['world_fingerprint_u64'] or
        result360.get('compiled_world_fingerprint') != pack_info['world_fingerprint_u64'] or
        result361.get('compiled_world_fingerprint') != pack_info['world_fingerprint_u64']):
        raise ValueError('compiled package fingerprint differs from run manifests/results')
    if (result360.get('status'), result361.get('status'), result360.get('accepted_steps'),
        result361.get('accepted_steps')) != ('failed', 'failed', 2, 2):
        raise ValueError('retained physical attempt verdict differs from the expected two-step failure')
    prefix = 'fem_residual_armijo_iteration='
    traces = []
    for run in (CONTROL_RUN, CAPTURE_RUN):
        traces.append([
            json.loads(line[len(prefix):])
            for line in (run / 'stderr.log').read_text().splitlines()
            if line.startswith(prefix)
        ])
    if traces[0] != traces[1] or len(traces[0]) != 64:
        raise ValueError('360/361 Armijo trace rows differ or are incomplete')
    receipts = [json.loads((run / 'receipt.json').read_text()) for run in (CONTROL_RUN, CAPTURE_RUN)]
    for receipt in receipts:
        if receipt.get('timed_out') or receipt.get('source_inputs_unchanged') is not True:
            raise ValueError('native receipt is timeout or source-mutating')
    if [r.get('exit_code') for r in receipts] != [2, 2]:
        raise ValueError('expected both physical attempts to terminate as preserved exit-2 failures')

    # Reconstruct center state records from run 360 to verify exact byte identity with 361.
    nodes = np.zeros((2, 882, 80), dtype=np.uint8)
    histories = np.zeros((2, 1920, 15), dtype='<f4')
    deformation_hi = np.zeros((2, 1920, 3, 4), dtype='<f4')
    deformation_lo = np.zeros_like(deformation_hi)
    tangent_cache = np.zeros((2, 1920, 81), dtype='<f4')
    seen_nodes: set[tuple[int, int]] = set()
    seen_tets: set[tuple[int, int]] = set()
    for line in (CONTROL_RUN / 'operator-capture').open():
        record = json.loads(line)
        if record.get('record') == 'header':
            if (record['world_fingerprint_hex'] != manifest['source_physics_fingerprint'] or
                record['execution_fingerprint_hex'] != manifest['device_program_fingerprint']):
                raise ValueError('360 center capture fingerprint does not match 361 regional manifest')
        elif record.get('record') == 'node':
            env, node = record['environment'], record['node_id']
            if (env, node) in seen_nodes:
                raise ValueError('duplicate 360 center node record')
            seen_nodes.add((env, node))
            state = np.asarray(record['center']['raw_state_float4s'], dtype='<f4').reshape(5, 4)
            nodes[env, node] = state.view(np.uint8).reshape(80)
        elif record.get('record') == 'tetrahedron':
            env, tet = record['environment'], record['tet_id']
            if (env, tet) in seen_tets:
                raise ValueError('duplicate 360 center tetrahedron record')
            seen_tets.add((env, tet))
            center = record['center']
            state = np.asarray(center['accepted_state'], dtype='<f4')
            if len(state) != record['material_state_count'] or len(state) > 15:
                raise ValueError('unexpected accepted material-history length')
            histories[env, tet, :len(state)] = state
            deformation_hi[env, tet, :, :3] = np.asarray(center['F_local'], dtype='<f4')
            deformation_lo[env, tet, :, :3] = np.asarray(center['F_local_low'], dtype='<f4')
            tangent_cache[env, tet] = np.asarray(
                record['cached_algorithmic_tangent_local'][
                    'matrix_row_major_local_dP_dF_Pa_per_dimensionless_F'], dtype='<f4')
    if len(seen_nodes) != 1764 or len(seen_tets) != 3840:
        raise ValueError('360 center capture coverage is incomplete')
    state_comparisons = {
        'candidate_nodes': nodes.tobytes() == raw_arrays['candidate_nodes'],
        'material_history': histories.tobytes() == raw_arrays['material_history'],
        'deformation_high': deformation_hi.tobytes() == raw_arrays['deformation_high'],
        'deformation_low': deformation_lo.tobytes() == raw_arrays['deformation_low'],
        'algorithmic_tangent_cache': tangent_cache.tobytes() == raw_arrays['algorithmic_tangent_cache'],
    }
    if not all(state_comparisons.values()):
        raise ValueError(f'360 center state differs from 361 selected generation: {state_comparisons}')
    packed_tags = np.fromiter(
        (struct.unpack_from('<f', raw_arrays['candidate_nodes'], env * 882 * 80 + node * 80 + 44)[0]
         for env in range(2) for node in range(882)), dtype=np.float32).reshape(2, 882)
    if not np.array_equal(packed_tags[0], source_fixed) or not np.array_equal(packed_tags[1], source_fixed):
        raise ValueError('selected candidate fixed/free masks differ from compiled source mask')

    # Independent sparse A is from the already sealed source101 cached-tangent audit (360).
    operator_path = OPERATOR_DIR / 'environment-0-operator-audit.operator.npz'
    vectors_path = OPERATOR_DIR / 'environment-0-operator-audit.vectors.npz'
    A = sparse.load_npz(operator_path).tocsr().astype(np.float64)
    vectors = np.load(vectors_path)
    free_nodes = vectors['free_node_ids'].astype(int)
    expected_free = np.flatnonzero(packed_tags[0] == 0.0)
    if not np.array_equal(free_nodes, expected_free):
        raise ValueError('operator audit free-node ordering differs from selected packed node mask')
    if A.shape != (3 * len(free_nodes), 3 * len(free_nodes)):
        raise ValueError('cached operator dimensions do not match packed free-node map')
    effective_p = arrays['effective_P'][0][:, free_nodes, :3].transpose(1, 2, 0).reshape(A.shape[0], 26).astype(np.float64)
    native_ap = arrays['AP'][0][:, free_nodes, :3].transpose(1, 2, 0).reshape(A.shape[0], 26).astype(np.float64)
    expected_ap = A @ effective_p
    ap_metrics = rel_metrics(native_ap, expected_ap)
    coarse_expected = np.einsum('ni,nj->ij', effective_p, native_ap, optimize=False)
    native_coarse, coarse_pivot_valid = lu_reconstruct(
        arrays['pivoted_RAP'][0].astype(np.float64), arrays['pivots'][0].astype(np.int64))
    coarse_metrics = rel_metrics(native_coarse, coarse_expected)
    factor_valid = arrays['factor_valid'].astype(int).tolist()
    if not np.all(np.asarray(factor_valid) == 1):
        raise ValueError('captured coarse factor is not fully valid')

    node_to_free = np.full(882, -1, dtype=np.int64)
    node_to_free[free_nodes] = np.arange(len(free_nodes))
    graph_patches = arrays['graph_patches'].astype(np.int64)
    graph_nodes = arrays['graph_patch_nodes'].astype(np.int64)
    graph_lu = arrays['graph_patch_matrices_lu']
    graph_pivots = arrays['graph_pivots']
    graph_valid = arrays['graph_factor_valid'].astype(int)
    if not np.all(graph_valid == 1):
        raise ValueError('captured selector-5 graph patch factor is invalid')
    patch_results = []
    total_fixed_patch_dofs = 0
    max_fixed_identity_error = 0.0
    max_fixed_free_cross_error = 0.0
    for patch_index, (node_offset, node_count, matrix_offset, pivot_offset) in enumerate(graph_patches):
        n = int(node_count) * 3
        patch_nodes = graph_nodes[int(node_offset):int(node_offset) + int(node_count)]
        global_dofs = np.asarray([
            3 * node_to_free[int(node)] + axis
            for node in patch_nodes for axis in range(3)
        ], dtype=np.int64)
        fixed_local = np.repeat(node_to_free[patch_nodes] < 0, 3)
        free_local = np.flatnonzero(~fixed_local)
        fixed_rows = np.flatnonzero(fixed_local)
        expected_patch = np.zeros((n, n), dtype=np.float64)
        free_global_dofs = global_dofs[free_local]
        expected_patch[np.ix_(free_local, free_local)] = A[
            free_global_dofs, :][:, free_global_dofs].toarray()
        expected_patch[fixed_rows, fixed_rows] = 1.0
        packed_lu = graph_lu[int(matrix_offset):int(matrix_offset) + n * n].reshape(n, n).astype(np.float64)
        packed_pivots = graph_pivots[int(pivot_offset):int(pivot_offset) + n].astype(np.int64)
        reconstructed_patch, _ = lu_reconstruct(packed_lu, packed_pivots)
        patch_metric = rel_metrics(reconstructed_patch, expected_patch)
        if len(fixed_rows):
            identity_error = float(np.max(np.abs(
                reconstructed_patch[np.ix_(fixed_rows, fixed_rows)] - np.eye(len(fixed_rows)))))
            cross_error = float(np.max(np.abs(reconstructed_patch[np.ix_(fixed_rows, free_local)]))) if len(free_local) else 0.0
        else:
            identity_error = 0.0
            cross_error = 0.0
        total_fixed_patch_dofs += len(fixed_rows)
        max_fixed_identity_error = max(max_fixed_identity_error, identity_error)
        max_fixed_free_cross_error = max(max_fixed_free_cross_error, cross_error)
        patch_results.append({
            'index': int(patch_index),
            'nodes': int(node_count),
            'dofs': int(n),
            'fixed_dofs': int(len(fixed_rows)),
            **patch_metric,
        })

    fail_paths = []
    for i in range(1, 6):
        folder = BASE / f'analysis/source101-regional361-actual-newton23-{i:03d}'
        failure = json.loads((folder / 'failure.json').read_text())
        fail_paths.append({
            'revision': i,
            'failure': failure,
            'failure_json_sha256': sha_file(folder / 'failure.json'),
            'audit_py_sha256': sha_file(folder / 'audit.py'),
            'manifest_sha256': sha_file(folder / 'manifest.json'),
        })

    summary = {
        'schema': 'numi.cardboard.source101-regional361-independent-review.v1',
        'verdict': 'same-generation captured operator/factors and compiled-world graph layout reproduced; numerical checks are descriptive and the physical transaction remains rejected',
        'native_status': {
            'capture_run_360_exit': receipts[0].get('exit_code'),
            'capture_run_361_exit': receipts[1].get('exit_code'),
            'timed_out': False,
            'physical_step_admitted': False,
            'ordinary_output_files_identical': len(output_files_360),
            'armijo_rows_identical': len(traces[0]),
            'same_compiled_input_hash_map': inputs_equal,
            'retained_staged_inputs_match_prediction_hashes': staged_inputs_verified,
            'input_hash_entry_count': len(pred361['inputs_sha256']),
            'build_provenance_sha256': sha_file(build_path_361),
            'source_snapshot': pred361['source_snapshot'],
            'runtime_directory': pred361['runtime_dir'],
            'executed_build_verified': pred361.get('executed_build_verified'),
            'compiled_world_fingerprint_u64': pack_info['world_fingerprint_u64'],
            'both_physical_results': {'status_360': result360['status'], 'status_361': result361['status'], 'accepted_steps_360': result360['accepted_steps'], 'accepted_steps_361': result361['accepted_steps']},
            'ordinary_output_hashes_sha256': output_hashes,
            'only_argv_difference_after_normalization': 'output directory',
            'comparison_scope': '360 and 361 use the same compiled source101 world and selector 5; this is diagnostic-capture parity, not selector-0 versus selector-5 ablation',
            'preconditioner_selector': manifest['preconditioner_selector'],
            'environment_differences': env_differences,
        },
        'failed_audit_revisions_001_to_005': fail_paths,
        'compiled_world_package': {
            **pack_info,
            'section_19_fem_nodes': {'count': 882, 'element_bytes': 80, 'fixed_tag_byte_offset': 44},
            'section_20_fem_tetrahedra': {'count': 1920, 'element_bytes': 96, 'connectivity_byte_offsets': [0, 4, 8, 12]},
            'source_free_nodes': int(np.sum(source_free_mask)),
            'source_fixed_nodes': int(np.sum(source_free_mask == 0)),
            'source_free_mask_sha256': source_mask_sha,
            'source_topology_sha256': source_topology_sha,
            'graph_layout_serialized_in_pack': False,
            'graph_layout_rebuilt_independently_from_pack_fem_sections': True,
        },
        'same_generation360_state_byte_comparisons': state_comparisons,
        'candidate_digest_recomputed_from_payload': {
            'environment': 0,
            'computed_sha256': recomputed_linearization_sha,
            'manifest_sha256': expected_linearization_sha,
            'candidate_nodes_sha256': candidate_node_sha,
            'candidate_nodes_manifest_sha256': candidate_metadata['candidate_nodes_sha256'],
        },
        'independent_current_graph_layout_equality': {
            'all_arrays_exact': all(layout_equal.values()),
            'arrays': layout_equal,
            'patch_count': int(len(rebuilt['patches'])),
            'patch_node_count': int(len(rebuilt['patch_nodes'])),
            'row_count': int(len(rebuilt['rows'])),
            'matrix_float_count': int(sum((3 * int(row[1])) ** 2 for row in rebuilt['patches'])),
            'pivot_count': int(sum(3 * int(row[1]) for row in rebuilt['patches'])),
            'coverage_min': int(rebuilt['node_coverage'].min()),
            'coverage_max': int(rebuilt['node_coverage'].max()),
            'potential_mask_sha256': potential_sha,
            'layout_checksum_fnv1a32': checksum,
            'previous_reader_all_false_reason': 'it reconstructed the older free-only layout and omitted the source101 initially-fixed supplementary partition; its false comparison does not indicate a production mismatch',
        },
        'production_operator_and_lu_reconstruction': {
            'capture_point': manifest['capture_point'],
            'environment': 0,
            'root': manifest['root'],
            'newton_iteration': manifest['iteration'],
            'linearization_generation': manifest['linearization_generation'],
            'linearization_sha256': expected_linearization_sha,
            'operator_reference': 'cached-tangent A reconstructed by source101-control2-newton23-audit-004 from this same capture generation; not an independent material-law evaluation',
            'A_shape': list(A.shape),
            'A_nnz': int(A.nnz),
            'P_shape': list(effective_p.shape),
            'AP_shape': list(native_ap.shape),
            'AP': ap_metrics,
            'RAP': coarse_metrics,
            'RAP_pivots_are_permutation': coarse_pivot_valid,
            'RAP_factor_valid_by_environment': factor_valid,
            'pivot_convention': 'factor row i corresponds to source row pivots[i]; source row swaps pivot entries with LU rows, so reconstruct original matrix by assigning (L@U)[i] into result[pivots[i]]',
            'graph_patch_count': int(len(patch_results)),
            'graph_factor_valid_count': int(np.sum(graph_valid == 1)),
            'maximum_patch_relative_frobenius_error': max(x['relative_frobenius_error'] for x in patch_results),
            'maximum_patch_absolute_error': max(x['maximum_absolute_error'] for x in patch_results),
            'max_fixed_identity_error': max_fixed_identity_error,
            'max_fixed_to_free_cross_error': max_fixed_free_cross_error,
            'fixed_patch_dof_copies_checked': total_fixed_patch_dofs,
            'patch_metrics': patch_results,
            'relative_error_denominators': {
                'formula': 'Frobenius norm(actual-reference) / max(Frobenius norm(reference), 1e-300)',
                'AP_reference_norm': ap_metrics['reference_frobenius_norm_denominator'],
                'RAP_reference_norm': coarse_metrics['reference_frobenius_norm_denominator'],
                'patch_reference_norms': [x['reference_frobenius_norm_denominator'] for x in patch_results],
            },
            'numeric_threshold_gate': 'none in source101 006 report or this independent review; values are descriptive reconstruction metrics, not a predeclared tolerance pass',
        },
        'source_rule_anchors': {
            'package_sections_and_compiled_FEM_topology': 'matter/src/package_impl.cpp:28-96,278-318',
            'runtime_source_mask_and_graph_builder': 'matter/src/runtime.mm:1274-1325',
            'two_partition_graph_ordering_and_patch_offsets': 'matter/src/regional_graph_schwarz.hpp:91-225,287-320',
            'production_patch_matrix_and_fixed_identity_rows_columns': 'matter/src/metal/fgmres.metalinc:2904-3102',
            'production_RAP_and_pivoted_factor': 'matter/src/metal/fgmres.metalinc:2482-2586',
            'production_graph_LU_and_pivot_convention': 'matter/src/metal/fgmres.metalinc:3128-3418',
        },
        'scope_and_limits': [
            'No native or physical transaction was run by this review.',
            'The exact graph-layout equality check is independent Python reconstruction from the actual compiled package FEM node mask and connectivity, against native layout arrays; it is not a separately compiled C++ builder test.',
            'The compiled package serializes FEM topology and fixed mask but does not serialize the runtime-built graph maps.',
            'AP/RAP and graph-LU comparisons use a cached-tangent reconstructed A from the same native linearization, so they validate operator/preconditioner assembly consistency, not independent constitutive accuracy.',
            'RAP arrays are present for both environments; graph patch LU arrays are captured only for environment 0.',
            'The two native attempts have exit code 2 and the physical step is rejected; no convergence, mesh/time refinement, or full-box qualification follows.',
            'No universal FP32 noise floor was measured; numerical residuals are reported with denominators, not promoted to a broad accuracy gate.',
        ],
    }
    return summary


def main() -> None:
    if (OUT / 'report.json').exists() or (OUT / 'manifest.json').exists():
        raise SystemExit('refusing to overwrite an existing review package')
    report = run_review()
    (OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    inputs = [
        CAPTURE_RUN / 'operator-capture.regional.json', CAPTURE_RUN / 'operator-capture.regional.bin',
        CAPTURE_RUN / 'prediction.json', CAPTURE_RUN / 'receipt.json', CAPTURE_RUN / 'stdout.log', CAPTURE_RUN / 'stderr.log', CAPTURE_RUN / 'output/compiled.nmatterpack',
        CONTROL_RUN / 'operator-capture', CONTROL_RUN / 'prediction.json', CONTROL_RUN / 'receipt.json', CONTROL_RUN / 'stdout.log', CONTROL_RUN / 'stderr.log',
        OPERATOR_DIR / 'environment-0-operator-audit.operator.npz',
        OPERATOR_DIR / 'environment-0-operator-audit.vectors.npz',
        BASE / 'analysis/source101-regional361-actual-newton23-006/audit.py',
        BASE / 'analysis/source101-regional361-actual-newton23-006/potential_mask_reader.py',
        SOURCE / 'matter/src/package_impl.cpp', SOURCE / 'matter/src/regional_graph_schwarz.hpp',
        SOURCE / 'matter/src/runtime.mm', SOURCE / 'matter/src/metal/fgmres.metalinc',
        SOURCE / 'matter/include/numi/matter/shared.h',
    ]
    input_hashes = {str(path): sha_file(path) for path in inputs}
    for i in range(1, 6):
        folder = BASE / f'analysis/source101-regional361-actual-newton23-{i:03d}'
        for path in (folder / 'failure.json', folder / 'audit.py', folder / 'manifest.json'):
            input_hashes[str(path)] = sha_file(path)
    for run in (CAPTURE_RUN, CONTROL_RUN):
        for path in sorted((run / 'output').rglob('*')):
            if path.is_file():
                input_hashes[str(path)] = sha_file(path)
    output_hashes = {
        'README.md': sha_file(OUT / 'README.md'),
        'review.py': sha_file(OUT / 'review.py'),
        'report.json': sha_file(OUT / 'report.json'),
    }
    (OUT / 'manifest.json').write_text(json.dumps({
        'schema': 'numi.cardboard.source101-regional361-independent-review-package.v1',
        'inputs_sha256': input_hashes,
        'package_files_sha256': output_hashes,
        'review_command': f'python3 {OUT / "review.py"}',
    }, indent=2) + '\n')
    print(json.dumps({
        'compiled_world_layout_equal': report['independent_current_graph_layout_equality']['all_arrays_exact'],
        'state_byte_matches': report['same_generation360_state_byte_comparisons'],
        'AP': report['production_operator_and_lu_reconstruction']['AP'],
        'RAP': report['production_operator_and_lu_reconstruction']['RAP'],
        'patch_count': report['production_operator_and_lu_reconstruction']['graph_patch_count'],
        'outputs_identical': report['native_status']['ordinary_output_files_identical'],
        'armijo_identical': report['native_status']['armijo_rows_identical'],
        'report': str(OUT / 'report.json'),
    }, indent=2))


if __name__ == '__main__':
    main()
