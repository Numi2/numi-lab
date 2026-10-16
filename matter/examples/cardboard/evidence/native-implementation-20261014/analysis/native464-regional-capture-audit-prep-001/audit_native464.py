#!/usr/bin/env python3
"""Strict CPU audit for source-151 native regional CGLS capture 464.

This consumer joins the regional/state sidecars to the immutable 419 operator
capture, then compares selector-5's captured *base* graph/AP/RAP arrays with
that same-generation global CSR. It deliberately does not claim that the
restart-augmented map was consumed by FGMRES.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
from typing import Any

BASE = Path('/Users/home/cardboard-implementation-20261014')
REGIONAL_READER_DIR = BASE / 'analysis/native464-regional-capture-command-001/validator'
GRAPH_REFERENCE_DIR = BASE / 'analysis/regional-rap-selector5-reference-151-002'
STATE_READER_PATH = BASE / 'analysis/base-cgls-state-reader-abi-repair-148-001/tools/read_state_capture.py'
CSR_PACKAGE = BASE / 'analysis/captured-base-state-cauchy-419-001'
CSR_SHA256 = '6b1d7f76b0a747cd28fa4ee53ef8b5316dd25ae390860a97242f1d91f63990ed'
OPERATOR_CAPTURE_DEFAULT = BASE / 'solver/two-stage-first-failure-rich-fd-419/operator-capture'
GRAPH_INPUT_DEFAULT = BASE / 'analysis/native464-regional-capture-command-001/preflight/actual-graph-input.bin'
BINDINGS_DEFAULT = BASE / 'analysis/native464-regional-capture-command-001/source151-regional-capture-bindings.json'
LINEAGE_DEFAULTS = {
    '462': BASE / 'solver/cgls-block-right-disabled-parity-462/base-cgls.bin.state.json',
    '463': BASE / 'solver/cgls-block-right-primary48-463/base-cgls.bin.state.json',
    '455': BASE / 'solver/cgls-state-sidecar-observer-455/base-cgls.bin.state.json',
}
STABLE_ARRAYS = (
    'accepted_nodes', 'candidate_nodes', 'accepted_tetrahedra',
    'candidate_tetrahedra', 'accepted_material_history',
    'environment_parameters', 'original_newton_correction',
    'base_residual_rhs', 'algorithmic_tangent_cache',
    'algorithmic_tangent_cache_valid',
)


class AuditError(RuntimeError):
    pass


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def json_no_dupes(path: Path) -> dict[str, Any]:
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise AuditError(f'duplicate JSON key {k!r} in {path}')
            out[k] = v
        return out
    def parse_int(token: str):
        # Preserve a JSON -0 float spelling used by the native writer without
        # changing integer IDs, which are printed as ordinary 0.
        return -0.0 if token == '-0' else int(token)
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_int=parse_int,
                       parse_constant=lambda x: (_ for _ in ()).throw(AuditError(f'nonstandard JSON token {x}')))
    if not isinstance(value, dict):
        raise AuditError(f'JSON root is not an object: {path}')
    return value


def f32(values) -> bytes:
    vals = list(values)
    return struct.pack('<' + 'f' * len(vals), *[float(x) for x in vals])


def u32(values) -> bytes:
    vals = list(values)
    return struct.pack('<' + 'I' * len(vals), *[int(x) for x in vals])


def load_state_reader():
    spec = importlib.util.spec_from_file_location('state_reader_148', STATE_READER_PATH)
    if spec is None or spec.loader is None:
        raise AuditError('could not import strict state reader')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_capture_readers():
    sys.path.insert(0, str(REGIONAL_READER_DIR))
    sys.path.insert(0, str(GRAPH_REFERENCE_DIR))
    import read_regional_capture_v2 as regional_v2
    import selector5_graph_reference as graph_ref
    return regional_v2, graph_ref


def read_state(path: Path, state_reader) -> dict[str, Any]:
    validation = state_reader.validate_capture(path)
    doc = state_reader._read_json_strict(path)
    payload_path = path.parent / doc['payload']['file']
    payload = payload_path.read_bytes()
    entries = {row['name']: row for row in doc['arrays']}
    views = {}
    for name, entry in entries.items():
        start, end = entry['offset'], entry['offset'] + entry['byte_count']
        views[name] = payload[start:end]
    return {'path': path, 'doc': doc, 'payload_path': payload_path,
            'payload_sha256': sha_bytes(payload), 'payload': payload,
            'entries': entries, 'arrays': views, 'validation': validation}


def read_operator_capture(path: Path) -> tuple[dict[str, Any], dict[tuple[int, int], dict[str, Any]], dict[tuple[int, int], dict[str, Any]]]:
    with path.open('r', encoding='utf-8') as stream:
        first = stream.readline()
        if not first:
            raise AuditError('operator capture is empty')
        header = json.loads(first, parse_int=lambda tok: -0.0 if tok == '-0' else int(tok))
        if header.get('record') != 'header':
            raise AuditError('operator capture lacks a header')
        nodes: dict[tuple[int, int], dict[str, Any]] = {}
        tets: dict[tuple[int, int], dict[str, Any]] = {}
        for line_no, line in enumerate(stream, 2):
            if not line.strip():
                continue
            row = json.loads(line, parse_int=lambda tok: -0.0 if tok == '-0' else int(tok))
            kind = row.get('record')
            if kind == 'node':
                key = (int(row['environment']), int(row['node_id']))
                if key in nodes:
                    raise AuditError(f'duplicate operator node record {key} at line {line_no}')
                nodes[key] = row
            elif kind == 'tetrahedron':
                key = (int(row['environment']), int(row['tet_id']))
                if key in tets:
                    raise AuditError(f'duplicate operator tet record {key} at line {line_no}')
                tets[key] = row
    return header, nodes, tets


def expected_tet(row: dict[str, Any]) -> bytes:
    node_ids = [int(x) for x in row['nodes']]
    inv = row['inverse_rest_rows_per_m']
    float_words = [*inv[0], row['rest_volume_m3'], *inv[1], 0.0, *inv[2], 0.0]
    ids = [row['material_index'], row['object_index'], row['topology_generation'], row['tet_flags']]
    frame = row['material_frame_quaternion_xyzw']
    return u32(node_ids) + f32(float_words) + u32(ids) + f32(frame)


def canonical_tet(data: bytes) -> tuple[int, ...]:
    if len(data) != 96:
        raise AuditError('tet record is not 96 bytes')
    words = list(struct.unpack('<24I', data))
    for i in (*range(4, 16), *range(20, 24)):
        if words[i] & 0x7fffffff == 0:
            words[i] = 0
    return tuple(words)


def state_expected_from_operator(state: dict[str, Any], operator_path: Path,
                                 header: dict[str, Any], nodes: dict, tets: dict) -> dict[str, Any]:
    doc, entries, raw = state['doc'], state['entries'], state['arrays']
    env_count, n_count, t_count = doc['environment_count'], doc['node_count'], doc['tetrahedron_count']
    stride = doc['material_state_stride']
    expected = {name: bytearray(entries[name]['byte_count']) for name in STABLE_ARRAYS}
    counts = Counter()
    for env in range(env_count):
        for node in range(n_count):
            row = nodes.get((env, node))
            if row is None:
                raise AuditError(f'419 operator capture missing node {(env, node)}')
            i = env * n_count + node
            accepted = f32(v for vec in row['accepted']['raw_state_float4s'] for v in vec)
            center = f32(v for vec in row['center']['raw_state_float4s'] for v in vec)
            correction = f32(list(row['correction_delta_v_m_s']) + [0.0])
            rhs = f32(row['center_residual_impulse_Ns'])
            expected['accepted_nodes'][i*80:(i+1)*80] = accepted
            expected['candidate_nodes'][i*80:(i+1)*80] = center
            expected['original_newton_correction'][i*16:(i+1)*16] = correction
            expected['base_residual_rhs'][i*16:(i+1)*16] = rhs
        for tet in range(t_count):
            row = tets.get((env, tet))
            if row is None:
                raise AuditError(f'419 operator capture missing tetrahedron {(env, tet)}')
            i = env * t_count + tet
            tb = expected_tet(row)
            expected['accepted_tetrahedra'][i*96:(i+1)*96] = tb
            expected['candidate_tetrahedra'][i*96:(i+1)*96] = tb
            hist = list(row['center']['accepted_state'])
            if len(hist) > stride:
                raise AuditError(f'419 history exceeds source capture stride at {(env, tet)}')
            expected['accepted_material_history'][i*stride*4:(i+1)*stride*4] = f32(hist + [0.0]*(stride-len(hist)))
            cache = row['cached_algorithmic_tangent_local']
            expected['algorithmic_tangent_cache_valid'][i*4:(i+1)*4] = u32([1 if cache.get('valid') is True else 0])
            if cache.get('valid') is True:
                vals = cache['matrix_row_major_local_dP_dF_Pa_per_dimensionless_F']
                expected['algorithmic_tangent_cache'][i*81*4:(i+1)*81*4] = f32(vals)
    params = header['parameters_by_environment']
    expected['environment_parameters'][:] = b''.join(f32(row) for row in params)
    report = {}
    for name, exp in expected.items():
        got = raw[name]
        if len(got) != len(exp):
            raise AuditError(f'array extent changed during join: {name}')
        if name.endswith('tetrahedra'):
            words = len(got) // 96
            exact = canonical = signed_zero_only = 0
            for i in range(words):
                a, b = got[i*96:(i+1)*96], exp[i*96:(i+1)*96]
                exact += a == b
                same = canonical_tet(a) == canonical_tet(b)
                canonical += same
                signed_zero_only += same and a != b
            report[name] = {'records': words, 'exact_records': exact,
                            'equal_after_signedzero_only_canonicalization': canonical,
                            'signedzero_only_records': signed_zero_only,
                            'other_mismatch_records': words-canonical}
        else:
            report[name] = {'bytes': len(got), 'exact_bytes': got == bytes(exp),
                            'mismatch_bytes': sum(a != b for a, b in zip(got, exp))}
    # The candidate material-state arena is a later raw scratch snapshot. Never
    # interpret it as the projected history used to construct this tangent.
    report['candidate_material_state_buffer'] = {
        'sha256': sha_bytes(raw['candidate_material_state_buffer']),
        'semantic': 'raw arena only; current-Newton projected history unavailable',
    }
    return report


def compare_lineage(states: dict[str, dict[str, Any]], target: str = '464') -> dict[str, Any]:
    reference = states[target]
    result = {}
    for label, state in states.items():
        if label == target:
            continue
        rows = {}
        for name in STABLE_ARRAYS:
            same = reference['arrays'][name] == state['arrays'][name]
            rows[name] = {'exact_bytes': same,
                          'left_sha256': sha_bytes(reference['arrays'][name]),
                          'right_sha256': sha_bytes(state['arrays'][name])}
        result[label] = {
            'root_newton_generation_matches': all(
                reference['doc'].get(k) == state['doc'].get(k)
                for k in ('root','physics_substep','microtick','microtick_count','micro_generation','newton_iteration','dt_seconds')),
            'array_comparisons': rows,
        }
    return result


def parse_graph_input(path: Path, bindings_path: Path) -> tuple[bytes, bytes, bytes, dict[str, Any]]:
    bindings = json_no_dupes(bindings_path)
    relative = 'preflight/actual-graph-input.bin'
    expected_sha = bindings['preflight_files'][relative]
    raw = path.read_bytes()
    actual_sha = sha_bytes(raw)
    if actual_sha != expected_sha:
        raise AuditError(f'compiled graph-input hash mismatch: {actual_sha} != {expected_sha}')
    if len(raw) < 12:
        raise AuditError('compiled graph-input header is truncated')
    env, nodes, tets = struct.unpack_from('<III', raw, 0)
    expect = 12 + 16*tets + nodes
    if len(raw) != expect:
        raise AuditError(f'compiled graph-input extent mismatch: got {len(raw)}, expected {expect}')
    connectivity = raw[12:12+16*tets]
    free = raw[12+16*tets:]
    if any(x not in (0,1) for x in free):
        raise AuditError('compiled graph free mask is not binary')
    topology = struct.pack('<II', nodes, tets) + connectivity
    potential = bytes([1])*nodes
    return topology, free, potential, {'sha256': actual_sha, 'environment_count': env,
                                      'node_count': nodes, 'tetrahedron_count': tets,
                                      'free_nodes': sum(free), 'fixed_nodes': nodes-sum(free)}


def load_sparse_A(csr_path: Path, csr_manifest_path: Path, free_mask: bytes):
    try:
        import numpy as np
        from scipy import sparse
    except Exception as exc:
        raise AuditError(f'NumPy/SciPy are required for the CSR audit: {exc}') from exc
    meta = json_no_dupes(csr_manifest_path)
    expected = next((x for x in meta['files'] if x['path'] == csr_path.name), None)
    if expected is None or expected.get('sha256') != CSR_SHA256 or sha_file(csr_path) != CSR_SHA256:
        raise AuditError('419 CSR package file hash is not the bound 419 matrix')
    A = sparse.load_npz(csr_path).tocsr().astype(np.float64)
    A.sum_duplicates(); A.sort_indices()
    free_nodes = [i for i, flag in enumerate(free_mask) if flag]
    dofs = [3*n+a for n in free_nodes for a in range(3)]
    if A.shape != (len(dofs), len(dofs)):
        raise AuditError(f'419 CSR shape {A.shape} disagrees with source free-mask dofs {len(dofs)}')
    dof_map = {full: compact for compact, full in enumerate(dofs)}
    return A, np, free_nodes, dofs, dof_map


class _SparseRow:
    def __init__(self, row_maps, free_mask: bytes, dof_map: dict[int, int], row: int):
        self.row_maps, self.free_mask, self.dof_map, self.row = row_maps, free_mask, dof_map, row
    def __getitem__(self, col: int) -> float:
        rn, cn = self.row // 3, col // 3
        if self.free_mask[rn] and self.free_mask[cn]:
            return self.row_maps[self.dof_map[self.row]].get(self.dof_map[col], 0.0)
        if not self.free_mask[rn] and self.row == col:
            return 1.0
        return 0.0
    def __len__(self) -> int:
        return 3 * len(self.free_mask)


class _SparseFullA:
    def __init__(self, A, free_mask: bytes, dof_map: dict[int, int]):
        self.free_mask, self.dof_map = free_mask, dof_map
        self.row_maps = []
        for row in range(A.shape[0]):
            lo, hi = int(A.indptr[row]), int(A.indptr[row+1])
            self.row_maps.append({int(A.indices[i]): float(A.data[i]) for i in range(lo, hi)})
    def __getitem__(self, row: int) -> _SparseRow:
        return _SparseRow(self.row_maps, self.free_mask, self.dof_map, row)
    def __len__(self) -> int:
        return 3 * len(self.free_mask)


def matrix_metrics(expected, got) -> dict[str, Any]:
    import numpy as np
    a, b = np.asarray(expected, dtype=np.float64), np.asarray(got, dtype=np.float64)
    if a.shape != b.shape:
        raise AuditError(f'matrix shape mismatch {a.shape} vs {b.shape}')
    d = b-a
    refnorm = float(np.linalg.norm(a.ravel()))
    gotnorm = float(np.linalg.norm(b.ravel()))
    dnorm = float(np.linalg.norm(d.ravel()))
    return {'shape': list(a.shape), 'max_abs': float(np.max(np.abs(d))) if d.size else 0.0,
            'l2_abs': dnorm, 'reference_l2': refnorm, 'observed_l2': gotnorm,
            'relative_l2': dnorm/refnorm if refnorm else dnorm,
            'max_abs_reference': float(np.max(np.abs(a))) if a.size else 0.0}


def check_receipt(case_dir: Path) -> dict[str, Any]:
    path = case_dir / 'receipt.json'
    if not path.is_file():
        return {'available': False, 'path': str(path)}
    doc = json_no_dupes(path)
    return {'available': True, 'path': str(path), 'sha256': sha_file(path),
            'schema': doc.get('schema'), 'execution_started': doc.get('execution_started'),
            'source_inputs_unchanged': doc.get('source_inputs_unchanged'),
            'changed_inputs': doc.get('changed_inputs'), 'exit_code': doc.get('exit_code'),
            'timed_out': doc.get('timed_out'), 'wall_seconds': doc.get('wall_seconds')}


def run(args) -> dict[str, Any]:
    state_reader = load_state_reader()
    regional_v2, graph_ref = load_capture_readers()
    case_dir = args.case.resolve()
    regional_path = args.regional_manifest or (case_dir / 'base-cgls.bin.regional.json')
    if not regional_path.exists():
        raise AuditError(f'regional capture is absent (audit must run after terminal): {regional_path}')
    regional_path = regional_path.resolve()
    regional_summary = regional_v2.read_capture(regional_path)
    regional_doc = regional_v2.v1._strict_json(regional_path)
    state_path = regional_path.parent / regional_doc['candidate_state_binding']['manifest_file']
    state464 = read_state(state_path, state_reader)

    states = {'464': state464}
    for label, path in [('462', args.state462), ('463', args.state463), ('455', args.state455)]:
        states[label] = read_state(path, state_reader)
    lineage = compare_lineage(states)

    header, node_records, tet_records = read_operator_capture(args.operator_capture)
    state_doc = state464['doc']
    dims_match = (header.get('root') == state_doc['root'] and
                  header.get('newton_iteration') == state_doc['newton_iteration'] and
                  header.get('environment_count') == state_doc['environment_count'] and
                  header.get('node_count_per_environment') == state_doc['node_count'] and
                  header.get('tetrahedron_count_per_environment') == state_doc['tetrahedron_count'] and
                  str(header.get('world_fingerprint_hex')).lower() == str(state_doc['world_fingerprint_u64_hex']).lower() and
                  float(header.get('dt_seconds')) == float(state_doc['dt_seconds']))
    input_match = state_expected_from_operator(state464, args.operator_capture,
                                               header, node_records, tet_records)
    array_join_exact = all(
        (v.get('exact_bytes') is True if 'exact_bytes' in v else
         v.get('other_mismatch_records') == 0)
        for k, v in input_match.items() if k != 'candidate_material_state_buffer')

    topology, free_mask, potential_mask, graph_input_meta = parse_graph_input(args.graph_input, args.bindings)
    if graph_input_meta['environment_count'] != state_doc['environment_count'] or graph_input_meta['node_count'] != state_doc['node_count'] or graph_input_meta['tetrahedron_count'] != state_doc['tetrahedron_count']:
        raise AuditError('compiled graph input dimensions do not match captured state')
    env_payload = state464['arrays']['accepted_tetrahedra']
    topo_matches_state = True
    for env in range(state_doc['environment_count']):
        for tet in range(state_doc['tetrahedron_count']):
            off = (env*state_doc['tetrahedron_count'] + tet)*96
            if env_payload[off:off+16] != topology[8+tet*16:8+(tet+1)*16]:
                topo_matches_state = False
                break
    tag_bytes = bytearray(state_doc['node_count'])
    candidate_nodes = state464['arrays']['candidate_nodes']
    selection_raw = None
    # Preserve the primary-policy selection words as metadata. This observer is
    # before CGLS direction construction; a zero selection vector does not make
    # either environment disappear from the copied regional arrays.
    sidecar_payload = (regional_path.parent / regional_doc['payload']['file']).read_bytes()
    regional_arrays = {x['name']: x for x in regional_doc['arrays']}
    sel_desc = regional_arrays.get('primary_policy_selected_environment')
    selected_envs = []
    if regional_doc.get('primary_policy_selected_newton') is True and sel_desc:
        start = sel_desc['offset']; count = sel_desc['byte_count']
        selection_raw = sidecar_payload[start:start+count]
        words = struct.unpack('<' + 'I'*(len(selection_raw)//4), selection_raw)
        selected_envs = [i for i, word in enumerate(words) if word != 0]
    if args.environment is None:
        raise AuditError('--environment is required; the primary selection vector is metadata, not a graph-array selector')
    env = args.environment
    if not (0 <= env < state_doc['environment_count']):
        raise AuditError(f'--environment {env} is outside the captured environment range')
    for node in range(state_doc['node_count']):
        fixed = struct.unpack_from('<f', candidate_nodes, (env*state_doc['node_count']+node)*80+44)[0]
        if fixed not in (0.0, 1.0, 3.0):
            raise AuditError(f'unsupported fixed-node tag {fixed} at environment {env}, node {node}')
        tag_bytes[node] = 1 if fixed == 0 else 0
    free_mask_join = bytes(tag_bytes) == free_mask

    exact_operator_join = dims_match and array_join_exact and topo_matches_state and free_mask_join
    if exact_operator_join:
        A, np, free_nodes, dofs, dof_map = load_sparse_A(args.csr, args.csr_manifest, free_mask)
        csr_full = _SparseFullA(A, free_mask, dof_map)
        graph_capture, view_meta = graph_ref.load_base_view(
            regional_path, environment=env, topology_bytes=topology,
            free_mask_bytes=free_mask, potential_mask_bytes=potential_mask,
            request_actual_consumed_view=False)
        # Vectorized equivalent of selector5_graph_reference's scalar patch
        # audit. The original implementation performs millions of Python-level
        # sparse lookups for this shipping-sized 27-patch capture.
        import numpy as np
        dof_axis = np.arange(3, dtype=np.int64)
        worst_free = worst_fixed = 0.0
        compared_free = compared_fixed = 0
        for patch in graph_capture.patches:
            patch_nodes = np.asarray(patch.nodes, dtype=np.int64)
            global_dofs = (3*patch_nodes[:, None] + dof_axis[None, :]).reshape(-1)
            node_free = np.asarray([free_mask[int(n)] != 0 for n in patch_nodes], dtype=bool)
            dof_free = np.repeat(node_free, 3)
            expected = np.zeros((global_dofs.size, global_dofs.size), dtype=np.float64)
            free_rows = np.flatnonzero(dof_free)
            free_cols = free_rows
            if free_rows.size:
                compact_rows = np.asarray([dof_map[int(global_dofs[i])] for i in free_rows], dtype=np.int64)
                compact_cols = compact_rows
                expected[np.ix_(free_rows, free_cols)] = A[compact_rows, :][:, compact_cols].toarray()
            fixed_rows = np.flatnonzero(~dof_free)
            if fixed_rows.size:
                expected[fixed_rows, fixed_rows] = 1.0
            observed = np.asarray(patch.original, dtype=np.float64)
            error = np.abs(observed - expected)
            free_block = np.outer(dof_free, dof_free)
            fixed_block = ~free_block
            if np.any(free_block):
                worst_free = max(worst_free, float(np.max(error[free_block])))
                compared_free += int(np.count_nonzero(free_block))
            if np.any(fixed_block):
                worst_fixed = max(worst_fixed, float(np.max(error[fixed_block])))
                compared_fixed += int(np.count_nonzero(fixed_block))
        patch_comparison = {
            'same_generation_identity_bound': True,
            'free_free_max_abs': worst_free,
            'fixed_identity_and_cross_max_abs': worst_fixed,
            'free_free_entries': compared_free,
            'fixed_identity_cross_entries': compared_fixed,
            'within_atol': worst_free <= args.atol and worst_fixed <= args.atol,
            'atol': args.atol,
            'implementation': 'vectorized exact equivalent of source151 reference scalar comparison',
        }
        # Independently apply the exact 419 free CSR to the captured P columns.
        mode_count = graph_capture.mode_count
        p_full = np.asarray(graph_capture.P, dtype=np.float64)
        ap_full = np.asarray(graph_capture.AP, dtype=np.float64)
        p_free = p_full[dofs, :]
        ap_direct = A @ p_free
        ap_expected_full = np.zeros_like(ap_full)
        ap_expected_full[dofs, :] = ap_direct
        ap_stats = matrix_metrics(ap_expected_full, ap_full)
        ptap_from_capture_ap = p_free.T @ ap_direct if False else p_free.T @ ap_full[dofs, :]
        ptap_from_csr = p_free.T @ (A @ p_free)
        base_factor_matrix = np.asarray(graph_ref.reconstruct_matrix_from_lu(graph_capture.base_factor), dtype=np.float64) if graph_capture.base_factor.valid else None
        rap_stats = {
            'base_factor_valid': graph_capture.base_factor.valid,
            'base_factor_pivots': list(graph_capture.base_factor.pivots),
            'captured_PtAP_vs_CSR_PtAP': matrix_metrics(ptap_from_csr, ptap_from_capture_ap),
            'base_LU_reconstruction_vs_CSR_PtAP': matrix_metrics(ptap_from_csr, base_factor_matrix) if base_factor_matrix is not None else None,
            'base_LU_reconstruction_vs_capture_AP_PtAP': matrix_metrics(ptap_from_capture_ap, base_factor_matrix) if base_factor_matrix is not None else None,
        }
        try:
            graph_ref.require_base_view(graph_capture, request_actual_consumed_view=True)
            actual_view_refusal = {'refused': False, 'error': None}
        except graph_ref.GraphReferenceError as exc:
            actual_view_refusal = {'refused': True, 'error': str(exc)}
        # Check field rows, raw CGLS state flags, and raw restart metadata but do
        # not infer consumed-cycle currentness from producer generation alone.
        unknowns = state_doc['unknown_count']
        field_off = graph_ref.reader._strict_json(state_path)['arrays']
        rhs = state464['arrays']['base_residual_rhs']
        correction = state464['arrays']['original_newton_correction']
        field_bytes = state_doc['environment_count']*state_doc['node_count']*16
        field_zero = (all(b == 0 for b in rhs[field_bytes:]) and
                      all(b == 0 for b in correction[field_bytes:]))
        state_arrays_meta = regional_doc['arrays']
        def u32array(name):
            e = regional_arrays[name]
            raw = sidecar_payload[e['offset']:e['offset']+e['byte_count']]
            return list(struct.unpack('<'+'I'*(len(raw)//4), raw))
        gen = u32array('restart_generation')
        mode_valid = u32array('restart_mode_valid')
        aug_valid = u32array('augmented_RAP_valid')
        base_view_status = {
            'label': view_meta['mode_view'],
            'augmentation_enabled': view_meta['augmentation_enabled'],
            'actual_selected_view_admissible': False,
            'actual_view_request_refused': actual_view_refusal,
            'restart_generation_by_environment': [gen[i*4:(i+1)*4] for i in range(state_doc['environment_count'])],
            'restart_mode_valid': mode_valid,
            'augmented_RAP_valid': aug_valid,
            'primary_policy_selection_words': list(struct.unpack('<'+'I'*(len(selection_raw)//4), selection_raw)) if selection_raw else None,
            'primary_policy_selection_semantics': regional_doc.get('primary_selection_semantics'),
            'field_backing_rhs_and_correction_exact_zero': field_zero,
        }
        operator_comparison = {'available': True, 'selected_environment': env,
                               'global_A_source': '419 exact FP64 free CSR, reconstructed into source fixed-row identity view',
                               'patch_matrices': patch_comparison,
                               'captured_AP_vs_419_A_times_P': ap_stats,
                               'base_RAP': rap_stats,
                               'view_admission': base_view_status}
    else:
        operator_comparison = {
            'available': False,
            'reason': 'exact state/topology/free-mask join to 419 was not established',
            'patch_matrices': None, 'captured_AP_vs_419_A_times_P': None,
            'base_RAP': None,
            'join_gates': {'dimensions_and_world_and_dt': dims_match,
                           'exact_419_state_arrays': array_join_exact,
                           'compiled_topology_matches_state': topo_matches_state,
                           'selected_free_mask_matches_compiled_input': free_mask_join},
        }
    receipts = {label: check_receipt(args.case if label == '464' else args.lineage_case_dirs[label])
                for label in ('464', '462', '463', '455')}
    return {
        'schema': 'numi.cardboard.native464_regional_capture_audit.v1',
        'audit_only': True,
        'case': str(case_dir),
        'capture_manifest': str(regional_path),
        'capture_payload_sha256': regional_doc['payload'].get('sha256'),
        'strict_regional_reader': regional_summary,
        'run_receipts': receipts,
        'operator_capture': {'path': str(args.operator_capture), 'sha256': sha_file(args.operator_capture),
                             'header': {k: header.get(k) for k in ('root','newton_iteration','environment_count','node_count_per_environment','tetrahedron_count_per_environment','world_fingerprint_hex','execution_fingerprint_hex','dt_seconds')},
                             'record_counts': {'nodes': len(node_records), 'tetrahedra': len(tet_records)}},
        'compiled_graph_input': graph_input_meta,
        'lineage_state_comparison_464_vs_462_463_455': lineage,
        'state_to_419_exact_input_join': {'dimensions_world_dt_match': dims_match,
                                         'array_comparisons': input_match,
                                         'all_required_operator_arrays_exact_or_tet_signedzero_only': array_join_exact,
                                         'compiled_topology_matches_accepted_state': topo_matches_state,
                                         'selected_free_mask_matches_compiled_input': free_mask_join,
                                         'exact_operator_join_admissible': exact_operator_join,
                                         'candidate_material_state_buffer_semantics': 'raw scratch; not proven current Newton projected history'},
        'operator_comparison': operator_comparison,
        'limits': [
            'The 419 CSR comparison is FP64 arithmetic over the exact captured CSR, not a Metal bitwise action replay.',
            'No augmented restart map is claimed to have been consumed; producer generation does not prove consumer-cycle currentness.',
            'Captured selector-5 graph/AP/RAP comparisons are bounded to the selected environment and one exact Newton linearization.',
            'This audit is not a solver-convergence, physical, mesh-scaling, or shipping qualification.',
        ],
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', type=Path, required=True, help='native 464 case directory; must contain terminal receipt')
    p.add_argument('--regional-manifest', type=Path)
    p.add_argument('--state462', type=Path, default=LINEAGE_DEFAULTS['462'])
    p.add_argument('--state463', type=Path, default=LINEAGE_DEFAULTS['463'])
    p.add_argument('--state455', type=Path, default=LINEAGE_DEFAULTS['455'])
    p.add_argument('--case462', type=Path, default=LINEAGE_DEFAULTS['462'].parent)
    p.add_argument('--case463', type=Path, default=LINEAGE_DEFAULTS['463'].parent)
    p.add_argument('--case455', type=Path, default=LINEAGE_DEFAULTS['455'].parent)
    p.add_argument('--operator-capture', type=Path, default=OPERATOR_CAPTURE_DEFAULT)
    p.add_argument('--graph-input', type=Path, default=GRAPH_INPUT_DEFAULT)
    p.add_argument('--bindings', type=Path, default=BINDINGS_DEFAULT)
    p.add_argument('--csr', type=Path, default=CSR_PACKAGE/'A_free_csr.npz')
    p.add_argument('--csr-manifest', type=Path, default=CSR_PACKAGE/'manifest.json')
    p.add_argument('--environment', type=int, required=True,
                   help='explicit environment index for the bounded base-view comparison; policy-selection words are reported separately')
    p.add_argument('--atol', type=float, default=1e-9)
    p.add_argument('--output', type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.lineage_case_dirs = {'462': args.case462, '463': args.case463, '455': args.case455}
    try:
        report = run(args)
    except (AuditError, OSError, ValueError, KeyError, IndexError, struct.error) as exc:
        report = {'schema': 'numi.cardboard.native464_regional_capture_audit.v1',
                  'audit_only': True, 'status': 'incomplete_or_rejected',
                  'error_type': type(exc).__name__, 'error': str(exc),
                  'case': str(args.case)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
        print(json.dumps(report, sort_keys=True))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'output': str(args.output),
                      'exact_operator_join_admissible': report['state_to_419_exact_input_join']['exact_operator_join_admissible'],
                      'operator_comparison_available': report['operator_comparison'].get('available')}, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
