#!/usr/bin/env python3
"""Compare local Schwarz factor/apply precision on the exact source-175 A,b.

The outer restarted GMRES remains FP64/math.fsum for every case. This tests
three local-factor policies only; it is not a native implementation test.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import signal
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, onenormest

BASE = Path('/Users/home/cardboard-implementation-20261014/analysis')
REPORT = BASE / 'offline-operator-tool-development-058/run-175/operator-audit.json'
SCRIPT = Path(__file__).resolve()
OUT = SCRIPT.parent
SIZE_REPORT = BASE / 'source175-cached-operator-audit-001/cpu-schwarz-1024-057-fsum/report.json'
SIZE_MEMBERSHIP = BASE / 'source175-cached-operator-audit-001/cpu-schwarz-1024-057-fsum/patch-membership-256.json'
STUDY_PATH = Path('/Users/home/numi-cardboard-20261006/matter/tools/cardboard_offline_conditioning_study.py')
MAX_SECONDS = 120.0
MAX_HOST_BYTES = 1 << 30
PATCH_NODES = 256
ANCHOR_STRIDE = 128
GMRES_RESTART = 16
GMRES_MAX_ITERATIONS = 1024
GMRES_TOLERANCE = 1.0e-8

spec = importlib.util.spec_from_file_location('cardboard_offline_conditioning_study', STUDY_PATH)
STUDY = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(STUDY)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def row_fsum_product(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    source = np.asarray(vector, dtype=np.float64)
    return np.fromiter((math.fsum(float(matrix[i, j]) * float(source[j])
                                 for j in range(matrix.shape[1]))
                        for i in range(matrix.shape[0])),
                       dtype=np.float64, count=matrix.shape[0])


def append_jsonl(path: Path, row: dict) -> None:
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
        stream.flush()


def build_graph_patches_extended(topology, free_nodes, max_patch_nodes, anchor_stride):
    """Exact prior BFS/anchor traversal extended from patch cap 32 to 256."""
    if not 1 <= max_patch_nodes <= 256 or not 1 <= anchor_stride <= max_patch_nodes:
        raise ValueError('patch size must be [1,256] and anchor stride [1,patch size]')
    conn = np.asarray(topology['nodes'], dtype=np.int64)
    node_count = int(np.asarray(topology['node_count']).reshape(-1)[0])
    if conn.ndim != 2 or conn.shape[1] != 4 or np.any(conn < 0) or np.any(conn >= node_count):
        raise ValueError('captured tetrahedron connectivity is malformed or out of bounds')
    free = np.asarray(free_nodes, dtype=np.int64)
    if free.ndim != 1 or np.any(free < 0) or np.any(free >= node_count) or len(set(free.tolist())) != len(free):
        raise ValueError('captured free-node ordering is malformed')
    free_set = set(int(node) for node in free)
    adjacency = [set() for _ in range(node_count)]
    for tet in conn:
        ids = [int(node) for node in tet if int(node) in free_set]
        for node in ids:
            adjacency[node].update(other for other in ids if other != node)
    anchors = free[::anchor_stride].tolist()
    if not anchors or anchors[-1] != int(free[-1]):
        anchors.append(int(free[-1]))
    patches = []
    for anchor in anchors:
        selected = [int(anchor)]
        visited = {int(anchor)}
        frontier = [int(anchor)]
        while frontier and len(selected) < max_patch_nodes:
            current = frontier.pop(0)
            for neighbor in sorted(adjacency[current]):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                selected.append(neighbor)
                frontier.append(neighbor)
                if len(selected) == max_patch_nodes:
                    break
        patches.append(np.asarray(sorted(selected), dtype=np.int64))
    covered = set(int(node) for patch in patches for node in patch)
    for node in free:
        if int(node) not in covered:
            patches.append(np.asarray([int(node)], dtype=np.int64))
    membership = np.zeros(node_count, dtype=np.int32)
    for patch in patches:
        membership[patch] += 1
    if np.any(membership[free] == 0) or np.any(membership[np.setdiff1d(np.arange(node_count), free)] != 0):
        raise ValueError('patch membership does not cover exactly free nodes')
    return patches, membership


def build_preconditioner(matrix, free_nodes, patches, membership, mode, retained_bytes):
    dofs_by_node = {int(node): np.arange(3 * i, 3 * i + 3, dtype=np.int64)
                    for i, node in enumerate(free_nodes)}
    factors = []
    factor_workspace = 0
    rows = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        for patch_index, nodes in enumerate(patches):
            dofs = np.concatenate([dofs_by_node[int(node)] for node in nodes])
            dense64 = matrix[dofs][:, dofs].toarray()
            STUDY.require_finite(f'{mode} local matrix {patch_index}', dense64)
            dim = int(dense64.shape[0])
            factor_workspace += 3 * dense64.nbytes + 4 * dim * 8
            if retained_bytes + factor_workspace > MAX_HOST_BYTES:
                raise MemoryError('retained inputs plus estimated factor workspace exceed 1 GiB')
            if mode == 'fp64_factor_fp64_solve':
                with warnings.catch_warnings(record=True) as local_warnings:
                    warnings.simplefilter('always')
                    lu64, pivots = linalg.lu_factor(dense64, check_finite=False)
                lu = lu64
                factor_dtype = 'float64'
            elif mode == 'fp32_refactor_fp32_solve':
                dense32 = dense64.astype(np.float32)
                if not np.isfinite(dense32).all():
                    raise ValueError(f'patch {patch_index} overflows when casting local matrix to float32')
                with warnings.catch_warnings(record=True) as local_warnings:
                    warnings.simplefilter('always')
                    lu32, pivots = linalg.lu_factor(dense32, check_finite=False)
                lu = lu32
                factor_dtype = 'float32'
            elif mode == 'fp64_factor_rounded_fp32_fp32_solve':
                with warnings.catch_warnings(record=True) as local_warnings:
                    warnings.simplefilter('always')
                    lu64, pivots = linalg.lu_factor(dense64, check_finite=False)
                lu = lu64.astype(np.float32)
                if not np.isfinite(lu).all():
                    raise ValueError(f'patch {patch_index} FP64 LU overflows when rounded to float32')
                factor_dtype = 'float32-rounded-from-float64'
            else:
                raise ValueError(f'unsupported local precision mode {mode!r}')
            for warning in local_warnings:
                # Preserve each factor warning. A successful finite result does
                # not erase a library warning or turn it into an assumed pass.
                pass
            if not np.isfinite(lu).all():
                raise ValueError(f'{mode} patch {patch_index} factors are nonfinite')
            pivots = np.asarray(pivots)
            if pivots.shape != (dim,) or np.any(pivots < 0) or np.any(pivots >= dim):
                raise ValueError(f'{mode} patch {patch_index} pivot vector is malformed')
            diagonal = np.abs(np.diag(lu))
            pivot_scale = float(np.max(diagonal)) if diagonal.size else 0.0
            if not math.isfinite(pivot_scale) or pivot_scale <= 0.0 or \
               np.any(diagonal <= pivot_scale * 1.0e-12):
                raise ValueError(f'{mode} patch {patch_index} contains a zero/nonfinite/unresolved pivot')
            pivot_ratio = float(np.min(diagonal) / pivot_scale)
            probe = np.cos(np.arange(dim, dtype=np.float64) * 0.731) + \
                    np.sin(np.arange(dim, dtype=np.float64) * 0.193)
            rhs64 = row_fsum_product(dense64, probe)
            if factor_dtype == 'float64':
                local_rhs = rhs64
                local_solution = linalg.lu_solve((lu, pivots), local_rhs, check_finite=False)
            else:
                local_rhs = rhs64.astype(np.float32)
                if not np.isfinite(local_rhs).all():
                    raise ValueError(f'{mode} patch {patch_index} RHS becomes nonfinite in FP32')
                local_solution = linalg.lu_solve((lu, pivots), local_rhs, check_finite=False)
            solution64 = np.asarray(local_solution, dtype=np.float64)
            local_residual = float(np.linalg.norm(row_fsum_product(dense64, solution64) - rhs64) /
                                   max(np.linalg.norm(rhs64), np.finfo(np.float64).tiny))
            if not np.isfinite(solution64).all() or not math.isfinite(local_residual):
                raise ValueError(f'{mode} patch {patch_index} local solve became nonfinite')
            rows.append({'patch_index': patch_index, 'node_count': len(nodes), 'dof_count': dim,
                'factor_dtype': factor_dtype, 'all_factor_entries_finite': True,
                'pivot_count': int(pivots.size), 'pivot_vector_valid': True,
                'minimum_abs_u_pivot': float(np.min(diagonal)),
                'maximum_abs_u_pivot': float(np.max(diagonal)),
                'minimum_over_maximum_u_pivot': pivot_ratio,
                'local_solve_relative_residual_against_source_fp64_patch': local_residual,
                'factorization_warnings': [{'category': w.category.__name__, 'message': str(w.message)}
                    for w in local_warnings]})
            factors.append((nodes.copy(), dofs, lu, pivots, factor_dtype))

    def apply(rhs):
        source = np.asarray(rhs, dtype=np.float64)
        result = np.zeros_like(source)
        for nodes, dofs, lu, pivots, factor_dtype in factors:
            if factor_dtype == 'float64':
                local = linalg.lu_solve((lu, pivots), source[dofs], check_finite=False)
            else:
                local32 = np.asarray(source[dofs], dtype=np.float32)
                if not np.isfinite(local32).all():
                    raise ValueError(f'{mode} RHS cast to FP32 became nonfinite')
                local = linalg.lu_solve((lu, pivots), local32, check_finite=False)
            local64 = np.asarray(local, dtype=np.float64)
            if not np.isfinite(local64).all():
                raise ValueError(f'{mode} preconditioner application returned nonfinite values')
            weights = np.repeat(1.0 / membership[nodes].astype(np.float64), 3)
            np.add.at(result, dofs, local64 * weights)
        if not np.isfinite(result).all():
            raise ValueError(f'{mode} preconditioner output is nonfinite')
        return result

    # Estimate condition in FP64 for local matrices belonging to the three
    # smallest pivot-ratio patches. The condition estimates are not bounds.
    weakest = sorted(range(len(factors)), key=lambda i: rows[i]['minimum_over_maximum_u_pivot'])[:3]
    condition_rows = []
    for patch_index in weakest:
        nodes, dofs, lu, pivots, factor_dtype = factors[patch_index]
        dim = len(dofs)
        dense64 = matrix[dofs][:, dofs].toarray()
        if factor_dtype == 'float64':
            solve = lambda x, lu=lu, pivots=pivots: linalg.lu_solve((lu, pivots), x, check_finite=False)
            solve_t = lambda x, lu=lu, pivots=pivots: linalg.lu_solve((lu, pivots), x, trans=1, check_finite=False)
        else:
            solve = lambda x, lu=lu, pivots=pivots: np.asarray(linalg.lu_solve((lu, pivots), np.asarray(x, dtype=np.float32), check_finite=False), dtype=np.float64)
            solve_t = lambda x, lu=lu, pivots=pivots: np.asarray(linalg.lu_solve((lu, pivots), np.asarray(x, dtype=np.float32), trans=1, check_finite=False), dtype=np.float64)
        inv = LinearOperator((dim, dim), matvec=solve, rmatvec=solve_t, dtype=np.float64)
        inverse_norm = float(onenormest(inv, t=2, itmax=5))
        condition_rows.append({'patch_index': patch_index, 'node_count': int(len(nodes)),
            'factor_dtype': factor_dtype,
            'minimum_over_maximum_u_pivot': rows[patch_index]['minimum_over_maximum_u_pivot'],
            'source_matrix_one_norm': float(np.linalg.norm(dense64, ord=1)),
            'factor_inverse_one_norm_estimate': inverse_norm,
            'source_matrix_times_factor_inverse_norm_estimate': float(np.linalg.norm(dense64, ord=1) * inverse_norm),
            'method': 'onenormest(t=2,itmax=5) of stored-factor solve; estimate only'})
    return apply, {'factor_workspace_estimate_bytes': factor_workspace,
        'factorization_flops_estimate': int(sum((2.0 / 3.0) * (3 * len(patch)) ** 3 for patch in patches)),
        'all_factors_finite': True, 'all_pivot_vectors_valid': True,
        'factor_count': len(factors), 'pivot_count_total': int(sum(row['pivot_count'] for row in rows)),
        'minimum_relative_pivot_ratio': min(row['minimum_over_maximum_u_pivot'] for row in rows),
        'maximum_local_solve_relative_residual': max(row['local_solve_relative_residual_against_source_fp64_patch'] for row in rows),
        'local_factor_diagnostics': rows,
        'condition_estimates_weakest_pivot_patches': condition_rows}


def run():
    events_path = OUT / 'events.jsonl'
    if events_path.exists():
        raise FileExistsError(f'refusing to overwrite {events_path}')
    started = time.monotonic()
    timeout_state = {'expired': False}
    def timeout(_signum, _frame):
        timeout_state['expired'] = True
        raise TimeoutError('120-second time cap reached')
    old_handler = signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    completed = []
    try:
        report, matrix, vectors, topology, sidecar_hashes = STUDY.load_hashed_operator(REPORT)
        binding = report.get('binding', {})
        if report.get('schema') != 'numi.cardboard.cached-operator-audit.v1' or \
           binding.get('capture_run_fingerprints_match') is not True or \
           not binding.get('source_hashes_verified_against_prediction') or \
           report.get('capture_integrity', {}).get('production_inputs_bitwise_unchanged') is not True:
            raise ValueError('source-bound operator report admission failed')
        rhs = np.asarray(vectors['native_rhs_free_impulse_Ns'], dtype=np.float64)
        free_nodes = np.asarray(vectors['free_node_ids'], dtype=np.int64)
        if rhs.shape != (matrix.shape[0],) or matrix.shape[0] != 3 * len(free_nodes) or \
           not np.isfinite(rhs).all() or np.linalg.norm(rhs) == 0.0:
            raise ValueError('captured A/RHS dimensions or values are invalid')
        topo = {'nodes': np.asarray(topology['nodes']),
                'node_count': int(np.asarray(topology['node_count']).reshape(-1)[0])}
        patches, membership = build_graph_patches_extended(topo, free_nodes, PATCH_NODES, ANCHOR_STRIDE)
        expected_membership = json.loads(SIZE_MEMBERSHIP.read_text())
        actual_membership = {'patch_nodes': PATCH_NODES, 'anchor_stride': ANCHOR_STRIDE,
            'patches': [patch.astype(int).tolist() for patch in patches],
            'per_node_patch_coverage': membership.astype(int).tolist()}
        if actual_membership != expected_membership:
            raise ValueError('256-node membership differs from the prior source-bound size sweep')
        retained_bytes = int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes +
            sum(np.asarray(value).nbytes for value in vectors.values()) +
            sum(np.asarray(value).nbytes for value in topology.values()))
        gmres_bytes = int((3 * (GMRES_RESTART + 1) + 12) * matrix.shape[0] * 8 +
                          (GMRES_RESTART + 1) ** 2 * 8)
        cases = []
        for mode in ('fp64_factor_fp64_solve', 'fp32_refactor_fp32_solve',
                     'fp64_factor_rounded_fp32_fp32_solve'):
            if time.monotonic() - started >= MAX_SECONDS:
                raise TimeoutError(f'time cap before {mode}')
            # Per-patch construction checks the exact cumulative workspace as
            # it allocates; this upper bound includes that workspace plus the
            # retained source arrays and GMRES vectors.
            byte_bound = retained_bytes + gmres_bytes + sum(
                3 * (3 * len(patch)) ** 2 * 8 + 4 * (3 * len(patch)) * 8 for patch in patches)
            if byte_bound > MAX_HOST_BYTES:
                raise MemoryError(f'{mode} preflight estimate exceeds 1 GiB')
            append_jsonl(events_path, {'event': 'case_start', 'mode': mode,
                'max_patch_nodes': PATCH_NODES, 'anchor_stride': ANCHOR_STRIDE,
                'patch_count': len(patches), 'membership_sha256': sha256(SIZE_MEMBERSHIP),
                'preflight_host_bytes_estimate': byte_bound})
            apply, factor = build_preconditioner(matrix, free_nodes, patches, membership,
                mode, retained_bytes + gmres_bytes)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', RuntimeWarning)
                gmres = STUDY.restarted_right_gmres(matrix, rhs, apply,
                    restart=GMRES_RESTART, max_iterations=GMRES_MAX_ITERATIONS,
                    relative_tolerance=GMRES_TOLERANCE)
            gmres_warning_rows = [{'category': w.category.__name__, 'message': str(w.message),
                                   'filename': w.filename, 'line': w.lineno} for w in caught]
            result = {'event': 'case_complete', 'mode': mode,
                'max_patch_nodes': PATCH_NODES, 'anchor_stride': ANCHOR_STRIDE,
                'patch_count': len(patches), 'membership_sha256': sha256(SIZE_MEMBERSHIP),
                'gmres_restart': gmres['restart'], 'gmres_max_iterations': gmres['max_iterations'],
                'gmres_relative_tolerance': gmres['relative_tolerance'],
                'gmres_status': gmres['status'],
                'gmres_true_relative_residual': gmres['final_true_relative_residual'],
                'gmres_residual_history': gmres['true_residual_history'],
                'gmres_operator_matvecs': gmres['true_operator_matvec_count'],
                'gmres_preconditioner_applications': gmres['preconditioner_application_count'],
                'gmres_warnings': gmres_warning_rows, 'factor': factor,
                'estimated_host_peak_bytes': byte_bound,
                'elapsed_seconds': time.monotonic() - started}
            append_jsonl(events_path, result)
            cases.append(result)
            del apply, factor
            if time.monotonic() - started >= MAX_SECONDS:
                raise TimeoutError(f'time cap after preserving {mode}')
        import contextlib
        import io
        config = io.StringIO()
        with contextlib.redirect_stdout(config):
            np.show_config()
        previous_report = OUT.parent / 'cpu-schwarz-1024-057-fsum/report.json'
        summary = {'schema': 'numi.cardboard.source175-cpu-schwarz-precision-study.v1',
            'classification': 'CPU-only local-factor precision sensitivity on one captured rejected Newton linearization; no native RAP/Matter/physical qualification',
            'binding': {'operator_report': str(REPORT), 'operator_report_sha256': sha256(REPORT),
                'prediction_sha256': binding['prediction_sha256'], 'receipt_sha256': binding['receipt_sha256'],
                'stderr_sha256': binding['stderr_sha256'], 'capture_sha256': binding['capture_sha256'],
                'source_snapshot': binding['source_snapshot'], 'selection': report['capture_selection'],
                'sidecar_sha256': sidecar_hashes,
                'source_matrix_sha256': sidecar_hashes[REPORT.stem + '.operator.npz'],
                '256_node_size_sweep_report_sha256': sha256(previous_report),
                '256_node_membership_sha256': sha256(SIZE_MEMBERSHIP),
                'runner_script_sha256': sha256(SCRIPT), 'matrix_tool_sha256': sha256(STUDY_PATH)},
            'runtime': {'python': '3.10.12', 'numpy': np.__version__,
                'scipy': __import__('scipy').__version__, 'blas_configuration': config.getvalue()},
            'fixed_inputs': {'matrix': 'exact source-bound CSR A, no symmetrization',
                'rhs': 'exact captured native center residual impulse; unchanged between cases',
                'patches': {'max_nodes': PATCH_NODES, 'anchor_stride': ANCHOR_STRIDE,
                            'membership_sha256': sha256(SIZE_MEMBERSHIP)},
                'outer_gmres': {'precision': 'FP64, unchanged for every case', 'dot_update': 'rowwise math.fsum',
                    'restart': GMRES_RESTART, 'maximum_iterations': GMRES_MAX_ITERATIONS,
                    'relative_tolerance': GMRES_TOLERANCE}},
            'precision_modes': {'fp64_factor_fp64_solve': 'factor local A in FP64, apply/triangular solve in FP64',
                'fp32_refactor_fp32_solve': 'round local A to FP32, refactor in FP32, cast incoming vector to FP32, solve/apply in FP32, widen output for unchanged FP64 outer GMRES',
                'fp64_factor_rounded_fp32_fp32_solve': 'factor exact FP64 local A, round packed LU values to FP32, apply/triangular solve in FP32, widen output for unchanged FP64 outer GMRES'},
            'limits': ['The source175 selected root3/Newton23 operator came from a rejected diagnostic transaction.',
                'The FP32 modes emulate only local factor/apply arithmetic in CPU SciPy; they are not Metal-native measurements.',
                'No native RAP was loaded, compared, or admitted.',
                'Condition estimates are iterative estimates, not rigorous bounds.'],
            'host_and_time_budget': {'seconds_cap': MAX_SECONDS, 'elapsed_seconds': time.monotonic() - started,
                'bytes_cap': MAX_HOST_BYTES, 'retained_source_bytes_estimate': retained_bytes,
                'gmres_workspace_bytes_estimate': gmres_bytes,
                'maximum_case_estimated_peak_bytes': max(row['estimated_host_peak_bytes'] for row in cases),
                'process_peak_rss_bytes': int(__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss),
                'process_peak_rss_units': 'bytes on macOS'},
            'events_file': str(events_path), 'events_sha256': sha256(events_path),
            'cases': cases}
        out_path = OUT / 'report.json'
        out_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
        append_jsonl(events_path, {'event': 'study_complete', 'report_sha256': sha256(out_path),
                                   'elapsed_seconds': time.monotonic() - started})
        return summary
    except Exception as error:
        append_jsonl(events_path, {'event': 'study_failure', 'type': type(error).__name__,
            'message': str(error), 'completed_modes': [row['mode'] for row in completed],
            'timeout': timeout_state['expired'], 'elapsed_seconds': time.monotonic() - started})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    try:
        result = run()
    except Exception as error:
        print(f'PRECISION STUDY INCOMPLETE: {type(error).__name__}: {error}')
        raise
    print(json.dumps({'report': str(OUT / 'report.json'),
        'elapsed_seconds': result['host_and_time_budget']['elapsed_seconds'],
        'cases': [{'mode': row['mode'], 'status': row['gmres_status'],
                   'true_relative_residual': row['gmres_true_relative_residual']}
                  for row in result['cases']]}, sort_keys=True))
