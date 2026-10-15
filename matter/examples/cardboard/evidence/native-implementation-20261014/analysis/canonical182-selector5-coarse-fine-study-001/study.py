#!/usr/bin/env python3
"""Bounded CPU emulation of source-182 RAP plus selector-5 graph correction.

Source 182 is a selector-3 native capture. This study reuses its selected-
generation P/AP/RAP and audited A,b, but builds the selector-5 production
256-node/128-anchor global-ID layout from source 065. It is not a selector-5
native capture or qualification.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
import signal
import time
from pathlib import Path

import numpy as np
from scipy import linalg, sparse

BASE = Path('/Users/home/cardboard-implementation-20261014')
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'run-006'
REPORT = BASE / 'analysis/selected-newton-regional-rap-audit-182/operator-audit.json'
RAP_DIR = BASE / 'solver/selected-newton-regional-rap-and-operator-capture-182'
TOPOLOGY = BASE / 'analysis/selected-newton-regional-rap-audit-182/operator-audit.topology.npz'
LAYOUT_PATH = BASE / 'analysis/selector5-native-kernel-check-001/canonical182-input/layout.json'
LAYOUT_MANIFEST = BASE / 'analysis/selector5-native-kernel-check-001/canonical182-input/bundle.json'
READER_MANIFEST = BASE / 'analysis/selector5-regional-capture-reader-v2-001/manifest.json'
GRAPH_HELPER = BASE / 'snapshots/source-angular-and-graph-schwarz-integrated-065/matter/src/regional_graph_schwarz.hpp'
READER_PATH = BASE / 'analysis/selector5-regional-capture-reader-v2-001/read_selector5_capture.py'
OLD_STUDY_PATH = BASE / 'analysis/canonical182-coarse-schwarz-study-001/study.py'
MAX_SECONDS = 120.0
MAX_BYTES = 1 << 30
RESTART = 16
MAX_ITERATIONS = 256
TOL = 1e-8


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


OLD = load_module('source182_rap_graph_study', OLD_STUDY_PATH)
READER = load_module('selector5_reader_v2', READER_PATH)
STUDY = OLD.STUDY


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def append_event(path: Path, row: dict) -> None:
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
        stream.flush()


def production_layout(topology: dict, state: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, dict]:
    layout_doc = json.loads(LAYOUT_PATH.read_text())
    layout_manifest = json.loads(LAYOUT_MANIFEST.read_text())
    reader_manifest = json.loads(READER_MANIFEST.read_text())
    tetrahedra = np.asarray(topology['nodes'])
    node_count = int(np.asarray(topology['node_count']).reshape(-1)[0])
    if state.shape != (node_count, 20):
        raise ValueError('source182 candidate state does not match topology node count')
    fixed = np.asarray(state[:, 11], dtype=np.float32)
    if not np.isin(fixed, np.asarray([0.0, 1.0], dtype=np.float32)).all():
        raise ValueError('source182 candidate has non-binary fixed tags')
    free_mask = [bool(tag == 0.0) for tag in fixed]
    free_nodes = np.flatnonzero(fixed == 0.0).astype(np.int64)

    rebuilt = READER.rebuild_graph_layout(tetrahedra, free_mask,
                                         anchor_stride=128, maximum_nodes=256)
    for key, document_key in (
        ('patches', 'patches'), ('patch_nodes', 'patch_node_global_ids'),
        ('rows', 'rows'), ('node_patch_offsets', 'node_patch_offsets'),
        ('node_patch_values', 'node_patch_values'), ('node_coverage', 'node_coverage')):
        if rebuilt[key] != layout_doc[document_key]:
            raise ValueError(f'production selector-5 map mismatch for {key}')
    checksum = READER.fnv1a_layout(rebuilt)
    if checksum != layout_doc['fnv1a32_runtime_layout_checksum']:
        raise ValueError('production selector-5 layout checksum mismatch')

    # Independently bind the exact LE byte recipe used by the RAP-only overlay.
    topo_bytes = (int(node_count).to_bytes(4, 'little') +
                  int(len(tetrahedra)).to_bytes(4, 'little') +
                  np.asarray(tetrahedra, dtype='<u4', order='C').tobytes(order='C'))
    topo_digest = hashlib.sha256(topo_bytes).hexdigest()
    mask_digest = hashlib.sha256(bytes(1 if value else 0 for value in free_mask)).hexdigest()
    crosscheck = reader_manifest['source182_encoding_crosscheck']
    if topo_digest != crosscheck['canonical_source_topology_sha256']:
        raise ValueError('source182 canonical topology digest differs from v2 reader encoding crosscheck')
    if hashlib.sha256(TOPOLOGY.read_bytes()).hexdigest() != crosscheck['topology_npz_sha256']:
        raise ValueError('source182 topology NPZ hash differs from v2 reader encoding crosscheck')
    if mask_digest != crosscheck['source_free_mask_sha256']:
        raise ValueError('source182 free-mask digest differs from v2 reader encoding crosscheck')
    if mask_digest != layout_doc['free_mask_sha256']:
        raise ValueError('source free mask digest mismatch')

    patch_nodes = np.asarray(layout_doc['patch_node_global_ids'], dtype=np.int64)
    patches = []
    for offset, count, _matrix_offset, _pivot_offset in layout_doc['patches']:
        patch = patch_nodes[offset:offset + count].copy()
        if len(patch) != count or len(np.unique(patch)) != count:
            raise ValueError('invalid source selector-5 patch node range')
        patches.append(patch)
    dims = layout_doc['output_header']
    expected_dims = {
        'node_count': node_count, 'tet_count': len(tetrahedra),
        'patch_count': len(patches), 'patch_node_count': len(patch_nodes),
        'dof_count': int(np.sum([3 * len(p) for p in patches])),
    }
    for key, expected in expected_dims.items():
        if int(dims[key]) != expected:
            raise ValueError(f'source layout dimension mismatch for {key}')
    if layout_doc['source_capture_selector'] != 3 or layout_doc['input_capture_is_selector5'] is not False:
        raise ValueError('source layout lineage changed from selector-3 reconstruction')
    return patches, free_nodes, {
        'layout_document_sha256': sha256(LAYOUT_PATH),
        'layout_bundle_manifest_sha256': sha256(LAYOUT_MANIFEST),
        'reader_v2_manifest_sha256': sha256(READER_MANIFEST),
        'graph_helper_sha256': sha256(GRAPH_HELPER),
        'topology_npz_sha256': sha256(TOPOLOGY),
        'source_topology_sha256_recomputed': topo_digest,
        'source_free_mask_sha256_recomputed': mask_digest,
        'layout_checksum_fnv1a32': checksum,
        'dimensions': expected_dims,
        'patch_nodes_limit': 256,
        'anchor_stride': 128,
        'native_selector5_capture_present': False,
    }


def build_fp32_patch_solver(matrix: sparse.csr_matrix, free_nodes: np.ndarray,
                            patches: list[np.ndarray], coverage: np.ndarray,
                            base_memory: int):
    free_index = {int(node): index for index, node in enumerate(free_nodes)}
    factors = []
    estimated_workspace = 0
    minimum_relative_pivot = math.inf
    maximum_local_residual = 0.0
    for patch_index, nodes in enumerate(patches):
        dofs = np.concatenate([3 * free_index[int(node)] + np.arange(3, dtype=np.int64)
                               for node in nodes])
        local = matrix[dofs][:, dofs].toarray().astype(np.float32)
        if not np.isfinite(local).all():
            raise ValueError(f'local principal block {patch_index} is nonfinite')
        estimated_workspace += 3 * local.nbytes + 4 * local.shape[0] * 4
        if base_memory + estimated_workspace > MAX_BYTES:
            raise MemoryError('retained inputs, patch factors, and GMRES work exceed 1 GiB')
        scale = float(np.max(np.abs(local)))
        pivot_floor = max(scale * 1.0e-10, 1.0e-30)
        lu, piv = linalg.lu_factor(local, overwrite_a=False, check_finite=True)
        diagonal = np.abs(np.diag(lu))
        if (not np.isfinite(lu).all() or not np.isfinite(diagonal).all() or
                np.any(diagonal <= pivot_floor) or piv.shape != (local.shape[0],)):
            raise ValueError(f'local factor {patch_index} violates production pivot admission')
        minimum_relative_pivot = min(minimum_relative_pivot,
            float(np.min(diagonal) / max(np.max(diagonal), np.finfo(np.float32).tiny)))
        probe = (np.cos(np.arange(local.shape[0]) * 0.371) +
                 np.sin(np.arange(local.shape[0]) * 0.119)).astype(np.float32)
        probe_rhs = np.asarray([
            math.fsum(float(local[row, column]) * float(probe[column])
                      for column in range(local.shape[1]))
            for row in range(local.shape[0])
        ], dtype=np.float32)
        if not np.isfinite(probe_rhs).all():
            raise ValueError(f'local probe RHS {patch_index} is nonfinite')
        probe_x = linalg.lu_solve((lu, piv), probe_rhs, check_finite=False)
        residual = np.asarray([
            math.fsum(float(local[row, column]) * float(probe_x[column])
                      for column in range(local.shape[1])) - float(probe_rhs[row])
            for row in range(local.shape[0])
        ], dtype=np.float64)
        rel = float(np.linalg.norm(residual) / max(float(np.linalg.norm(probe_rhs)), 1e-300))
        if not math.isfinite(rel):
            raise ValueError(f'local solve residual {patch_index} is nonfinite')
        maximum_local_residual = max(maximum_local_residual, rel)
        factors.append((nodes.copy(), dofs, lu, piv))

    def apply_fine(rhs: np.ndarray) -> np.ndarray:
        rhs32 = np.asarray(rhs, dtype=np.float32)
        accumulated = np.zeros((int(np.max(free_nodes)) + 1, 3), dtype=np.float32)
        # Production apply first sums patch values in nodePatchValues order,
        # then divides once by the exact node coverage.
        for nodes, dofs, lu, piv in factors:
            local_rhs = rhs32[dofs]
            correction = linalg.lu_solve((lu, piv), local_rhs, check_finite=False).astype(np.float32)
            per_node = correction.reshape((len(nodes), 3))
            for local_node, node in enumerate(nodes):
                accumulated[int(node)] = np.asarray(
                    accumulated[int(node)] + per_node[local_node], dtype=np.float32)
        out = np.zeros_like(accumulated)
        for node in free_nodes:
            count = int(coverage[int(node)])
            if count <= 0:
                raise ValueError('free node has no selector-5 patch coverage')
            out[int(node)] = np.asarray(accumulated[int(node)] / np.float32(count), dtype=np.float32)
        return out[free_nodes].reshape(-1).astype(np.float64)

    return apply_fine, {
        'patch_count': len(factors), 'patch_node_entries': int(sum(map(len, patches))),
        'minimum_patch_nodes': int(min(map(len, patches))),
        'maximum_patch_nodes': int(max(map(len, patches))),
        'local_matrix_factor_precision': 'FP32 dense principal blocks and SciPy LU; native kernel emulation only',
        'pivot_floor': 'max(max_abs(local_block)*1e-10, 1e-30)',
        'minimum_relative_U_pivot': minimum_relative_pivot,
        'maximum_local_probe_relative_residual': maximum_local_residual,
        'estimated_patch_factor_workspace_bytes': estimated_workspace,
    }


def _fp32_simd_sum(products: np.ndarray) -> np.float32:
    # Approximate a 32-lane simd_sum reduction deterministically. It matches
    # the kernel's 32 strided partials and tree reduction but not hardware lane
    # scheduling/reassociation bit-for-bit.
    lanes = np.zeros(32, dtype=np.float32)
    for lane in range(32):
        acc = np.float32(0.0)
        for index in range(lane, len(products), 32):
            acc = np.float32(acc + products[index])
        lanes[lane] = acc
    width = 16
    while width:
        for lane in range(width):
            lanes[lane] = np.float32(lanes[lane] + lanes[lane + width])
        width //= 2
    return lanes[0]


def selector5_coarse_graph_action(rap: dict, free_nodes: np.ndarray,
                                  node_count: int, fine_apply):
    P = np.asarray(rap['P_free'], dtype=np.float32)
    AP = np.asarray(rap['AP_free'], dtype=np.float32)
    lu = np.asarray(rap['pivoted_lu'], dtype=np.float32)
    piv = np.asarray(rap['pivots'], dtype=np.int64)
    mode_count = P.shape[1]
    if AP.shape != P.shape or lu.shape != (mode_count, mode_count) or sorted(piv.tolist()) != list(range(mode_count)):
        raise ValueError('captured source182 RAP action dimensions/pivots are invalid')

    def solve_rap(rhs: np.ndarray) -> np.ndarray:
        values = np.asarray(rhs, dtype=np.float32)[piv].copy()
        for row in range(mode_count):
            value = np.float32(values[row])
            for col in range(row):
                value = np.float32(value - np.float32(lu[row, col] * values[col]))
            values[row] = value
        x = np.zeros(mode_count, dtype=np.float32)
        for row in range(mode_count - 1, -1, -1):
            value = np.float32(values[row])
            for col in range(row + 1, mode_count):
                value = np.float32(value - np.float32(lu[row, col] * x[col]))
            if lu[row, row] == 0 or not np.isfinite(lu[row, row]):
                raise ValueError('captured RAP triangular solve has invalid diagonal')
            x[row] = np.float32(value / lu[row, row])
        return x

    def apply(rhs: np.ndarray) -> np.ndarray:
        r32 = np.asarray(rhs, dtype=np.float32)
        coarse_rhs_values = []
        for mode in range(mode_count):
            node_dots = np.empty(len(free_nodes), dtype=np.float32)
            for index in range(len(free_nodes)):
                base = 3 * index
                x0 = np.float32(P[base, mode] * r32[base])
                x1 = np.float32(P[base + 1, mode] * r32[base + 1])
                x2 = np.float32(P[base + 2, mode] * r32[base + 2])
                node_dots[index] = np.float32(np.float32(x0 + x1) + x2)
            coarse_rhs_values.append(_fp32_simd_sum(node_dots))
        coarse_rhs = np.asarray(coarse_rhs_values, dtype=np.float32)
        coefficients = solve_rap(coarse_rhs)
        coarse = np.zeros((node_count, 3), dtype=np.float32)
        fine_rhs = r32.copy()
        for mode in range(mode_count):
            coefficient = coefficients[mode]
            for index, node in enumerate(free_nodes):
                for axis in range(3):
                    dof = 3 * index + axis
                    coarse[int(node), axis] = np.float32(
                        coarse[int(node), axis] + np.float32(coefficient * P[dof, mode]))
                    fine_rhs[3 * index + axis] = np.float32(
                        fine_rhs[3 * index + axis] - np.float32(coefficient * AP[dof, mode]))
        fine = fine_apply(fine_rhs)
        fine_nodes = np.asarray(fine, dtype=np.float32).reshape((len(free_nodes), 3))
        result = np.zeros((node_count, 3), dtype=np.float32)
        for index, node in enumerate(free_nodes):
            result[int(node)] = np.asarray(coarse[int(node)] + fine_nodes[index], dtype=np.float32)
        return result[free_nodes].reshape(-1).astype(np.float64)
    return apply


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=False)
    event_path = OUT / 'events.jsonl'
    start = time.monotonic()
    timed_out = {'value': False}
    def timeout_handler(_signal, _frame):
        timed_out['value'] = True
        raise TimeoutError('120-second total CPU cap reached')
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    try:
        report, matrix, vectors, topology, side_hashes = STUDY.load_hashed_operator(REPORT)
        rap_manifest_path = RAP_DIR / 'operator-capture.regional.json'
        rap_payload_path = RAP_DIR / 'operator-capture.regional.bin'
        rap, rap_manifest = OLD.load_source182_rap_arrays(
            rap_manifest_path, rap_payload_path, report, matrix, vectors)
        if report.get('capture_selection') != {'environment': 0, 'root': 3, 'newton_iteration': 23}:
            raise ValueError('source182 operator selection changed')
        if rap_manifest.get('capture_point') != 'after_regional_rap_factor_barrier_before_fgmres_correction':
            raise ValueError('source182 RAP capture point changed')
        rhs = np.asarray(vectors['native_rhs_free_impulse_Ns'], dtype=np.float64)
        state = np.asarray(vectors['captured_node_state_f32'], dtype=np.float32)
        patches, free_nodes, layout_info = production_layout(topology, state)
        if matrix.shape != (3 * len(free_nodes), 3 * len(free_nodes)) or rhs.shape != (matrix.shape[0],):
            raise ValueError('source182 A,b/free-node dimensions are inconsistent')
        coverage = np.zeros(int(np.max(free_nodes)) + 1, dtype=np.uint32)
        for patch in patches:
            coverage[patch] += 1
        if np.linalg.norm(rhs) == 0.0:
            raise ValueError('source182 selected right-hand side is zero')

        retained = (matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes +
                    sum(np.asarray(value).nbytes for value in vectors.values()) +
                    sum(np.asarray(value).nbytes for value in topology.values()))
        gmres_work = (3 * (RESTART + 1) + 12) * matrix.shape[0] * 8 + (RESTART + 1) ** 2 * 8
        if retained + gmres_work > MAX_BYTES:
            raise MemoryError('captured source arrays and GMRES work exceed 1 GiB before patch factors')
        fine_apply, local_info = build_fp32_patch_solver(matrix, free_nodes, patches, coverage,
                                                         retained + gmres_work)
        action = selector5_coarse_graph_action(rap, free_nodes, state.shape[0], fine_apply)
        membership_obj = {
            'schema': 'numi.cardboard.selector5-cpu-production-layout.v1',
            'source_selector': 3, 'patch_nodes_limit': 256, 'anchor_stride': 128,
            'layout_sha256': layout_info['layout_document_sha256'],
            'patches': [patch.astype(int).tolist() for patch in patches],
            'coverage': coverage.astype(int).tolist(),
        }
        member_path = OUT / 'patch-membership.json'
        member_path.write_text(json.dumps(membership_obj, sort_keys=True, separators=(',', ':')) + '\n')
        append_event(event_path, {
            'event': 'case_start', 'elapsed_s': time.monotonic() - start,
            'patch_count': len(patches), 'patch_node_entries': int(sum(map(len, patches))),
            'membership_sha256': sha256(member_path),
            'action': 'FP32 RAP coarse solve; residual r-AP*c; FP32 principal-block solves; sum corrections then divide once by exact node coverage; return Pc+fine',
        })
        result = STUDY.restarted_right_gmres(matrix, rhs, action, restart=RESTART,
            max_iterations=MAX_ITERATIONS, relative_tolerance=TOL)
        report_out = {
            'schema': 'numi.cardboard.source182-selector5-coarse-fine-cpu-emulation.v1',
            'classification': 'CPU algorithmic emulation only; source182 native capture is selector3, and no native selector5 FEM patch assembly or native selector5 factor capture was used',
            'bindings': {
                'study_script_sha256': sha256(Path(__file__).resolve()),
                'source182_operator_report': str(REPORT),
                'source182_operator_report_sha256': sha256(REPORT),
                'source182_prediction_sha256': report['binding']['prediction_sha256'],
                'source182_receipt_sha256': report['binding']['receipt_sha256'],
                'source182_snapshot': report['binding']['source_snapshot'],
                'source182_selection': report['capture_selection'],
                'source182_sidecar_hashes': side_hashes,
                'RAP_manifest_sha256': sha256(rap_manifest_path),
                'RAP_payload_sha256': sha256(rap_payload_path),
                'candidate_linearization_sha256_environment0': next(
                    row['sha256'] for row in rap_manifest['candidate_linearization_sha256']
                    if row.get('environment') == 0),
                'layout_inputs': layout_info,
                'operator_npz_sha256': sha256(REPORT.parent / 'operator-audit.operator.npz'),
                'vector_npz_sha256': sha256(REPORT.parent / 'operator-audit.vectors.npz'),
                'topology_npz_sha256': sha256(TOPOLOGY),
            },
            'operator': {
                'dimension': int(matrix.shape[0]), 'nnz': int(matrix.nnz),
                'rhs_l2_Ns': float(np.linalg.norm(rhs)),
                'native_A_action_parity_relative': report['operator']['action_parity']['relative_l2_error'],
                'unsymmetric': True,
            },
            'method': {
                'layout': 'source-065 production buildRegionalGraphSchwarzLayout defaults: maxPatchNodes=256, anchorStride=128, exact actual source182 global IDs; independently reconstructed and compared every ordered layout array',
                'coarse': 'selected-generation source182 selector-3 captured P/AP/pivoted RAP; FP32 CPU arithmetic approximating the production Metal sequence, not hardware bitwise reproduction',
                'fine': 'principal blocks extracted from audited source182 CSR A; FP32 LU/triangular solves; production overlap gather order and sum-then-divide policy',
                'outer_solver': {'implementation': 'existing source-bound CPU restarted right-GMRES utility', 'restart': RESTART, 'iteration_cap': MAX_ITERATIONS, 'true_relative_tolerance': TOL},
                'limits': {'wall_clock_seconds': MAX_SECONDS, 'host_memory_bytes': MAX_BYTES},
                'local_factor_diagnostics': local_info,
            },
            'result': result,
            'elapsed_seconds': time.monotonic() - start,
            'limitations': [
                'No selector-5 source182 capture exists; source182 RAP was selected under selector 3.',
                'No native selector-5 graph sidecar exists for source182; local matrices/factors were formed in CPU from captured CSR A.',
                'CPU arithmetic and SciPy factorization are not a Metal kernel or FGMRES performance/convergence qualification.',
                'The selected source182 Newton linearization belongs to a native run that later failed at Newton63; this does not qualify a completed physical step.',
                'No cardboard material calibration or physical result is inferred.',
            ],
        }
        report_path = OUT / 'report.json'
        report_path.write_text(json.dumps(report_out, sort_keys=True, indent=2, allow_nan=False) + '\n')
        append_event(event_path, {'event': 'case_complete', 'elapsed_s': time.monotonic() - start,
                                  'gmres_status': result.get('status'),
                                  'iterations': result.get('iterations'),
                                  'true_relative_residual': result.get('final_true_relative_residual')})
        (OUT / 'stdout.log').write_text(json.dumps({'status': 'complete', 'gmres_status': result.get('status'),
            'iterations': result.get('iterations'), 'true_relative_residual': result.get('final_true_relative_residual'),
            'elapsed_s': time.monotonic() - start}, sort_keys=True) + '\n')
        (OUT / 'stderr.log').write_text('')
        return 0
    except Exception as exc:
        append_event(event_path, {'event': 'study_failure', 'type': type(exc).__name__,
                                  'message': str(exc), 'elapsed_s': time.monotonic() - start})
        (OUT / 'stderr.log').write_text(f'{type(exc).__name__}: {exc}\n')
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    raise SystemExit(main())
