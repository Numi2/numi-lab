#!/usr/bin/env python3
"""Bounded 64/128/256-node CPU Schwarz sensitivity on source-175 A,b.

Exploratory only: consumes the exact source-bound offline A and RHS, never a
native RAP. A 120-second alarm, 1-GiB checked working-set estimate, per-case
JSONL records, and exact membership files preserve partial outcomes.
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
PRELIMINARY = OUT.parent / 'cpu-schwarz-1024-057/report.json'
MAX_SECONDS = 120.0
MAX_HOST_BYTES = 1 << 30
TOLERANCE = 1.0e-8
RESTART = 16
MAX_ITERATIONS = 1024

TOOL_PATH = Path('/Users/home/numi-cardboard-20261006/matter/tools/cardboard_offline_conditioning_study.py')
spec = importlib.util.spec_from_file_location('offline_conditioning', TOOL_PATH)
STUDY = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(STUDY)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, value: dict) -> None:
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n')
        stream.flush()


def row_fsum_product(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Independent scalar row products, avoiding the audited Accelerate matmul warning path."""
    values = np.asarray(vector, dtype=np.float64)
    return np.fromiter((math.fsum(float(matrix[i, j]) * float(values[j])
                                 for j in range(matrix.shape[1]))
                        for i in range(matrix.shape[0])),
                       dtype=np.float64, count=matrix.shape[0])


def build_graph_patches(topology: dict, free_nodes: np.ndarray,
                        max_patch_nodes: int, anchor_stride: int):
    if not 1 <= max_patch_nodes <= 256:
        raise ValueError('experimental patch size must be in [1,256]')
    if not 1 <= anchor_stride <= max_patch_nodes:
        raise ValueError('anchor stride must be in [1,patch size]')
    connectivity = np.asarray(topology['nodes'], dtype=np.int64)
    node_count = int(np.asarray(topology['node_count']).reshape(-1)[0])
    if connectivity.ndim != 2 or connectivity.shape[1] != 4 or \
       np.any(connectivity < 0) or np.any(connectivity >= node_count):
        raise ValueError('captured tetrahedron connectivity is malformed or out of bounds')
    free = np.asarray(free_nodes, dtype=np.int64)
    if free.ndim != 1 or np.any(free < 0) or np.any(free >= node_count) or \
       len(set(free.tolist())) != len(free):
        raise ValueError('captured free-node ordering is malformed')
    free_set = set(int(node) for node in free)
    adjacency = [set() for _ in range(node_count)]
    for tet in connectivity:
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
    other = np.setdiff1d(np.arange(node_count), free)
    if np.any(membership[free] == 0) or np.any(membership[other] != 0):
        raise ValueError('patch coverage is not exactly the captured free-node set')
    return patches, membership


def build_diagnostic_preconditioner(matrix, free_nodes, patches, membership,
                                    retained_bytes: int):
    dofs_by_node = {int(node): np.arange(3 * i, 3 * i + 3, dtype=np.int64)
                    for i, node in enumerate(free_nodes)}
    factors = []
    factor_workspace_estimate = 0
    patch_rows = []
    largest_dim = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', RuntimeWarning)
        for patch_index, nodes in enumerate(patches):
            dofs = np.concatenate([dofs_by_node[int(node)] for node in nodes])
            dense = matrix[dofs][:, dofs].toarray()
            STUDY.require_finite(f'Schwarz patch {patch_index} matrix', dense)
            dimension = int(dense.shape[0])
            largest_dim = max(largest_dim, dimension)
            # Match the reviewed builder's conservative dense/LU/work estimate.
            factor_workspace_estimate += 3 * dense.nbytes + 4 * dimension * 8
            if retained_bytes + factor_workspace_estimate > MAX_HOST_BYTES:
                raise MemoryError('source arrays plus cumulative factor workspace exceed 1 GiB')
            norm1 = float(np.linalg.norm(dense, ord=1))
            lu, pivots = linalg.lu_factor(dense, check_finite=False)
            if not np.isfinite(lu).all():
                raise ValueError(f'patch {patch_index} LU factors are nonfinite')
            if pivots.shape != (dimension,) or np.any(pivots < 0) or np.any(pivots >= dimension):
                raise ValueError(f'patch {patch_index} pivot vector is invalid')
            diagonal = np.abs(np.diag(lu))
            scale = float(np.max(diagonal)) if diagonal.size else 0.0
            if not math.isfinite(scale) or scale <= 0.0 or np.any(diagonal <= scale * 1.0e-12):
                raise ValueError(f'patch {patch_index} has a pivot below the existing 1e-12 relative floor')
            ratio = float(np.min(diagonal) / scale)
            # One deterministic local solve residual checks that the stored LU
            # and pivot vector actually solve this patch, beyond finite pivots.
            probe = np.cos(np.arange(dimension, dtype=np.float64) * 0.731) + \
                    np.sin(np.arange(dimension, dtype=np.float64) * 0.193)
            local_rhs = row_fsum_product(dense, probe)
            local_solution = linalg.lu_solve((lu, pivots), local_rhs, check_finite=False)
            solve_residual = float(np.linalg.norm(row_fsum_product(dense, local_solution) - local_rhs) /
                                   max(np.linalg.norm(local_rhs), np.finfo(np.float64).tiny))
            if not np.isfinite(local_solution).all() or not math.isfinite(solve_residual):
                raise ValueError(f'patch {patch_index} local factor check became nonfinite')
            patch_rows.append({'patch_index': patch_index,
                'node_count': int(len(nodes)), 'dof_count': dimension,
                'matrix_one_norm': norm1,
                'min_abs_u_pivot': float(np.min(diagonal)),
                'max_abs_u_pivot': float(np.max(diagonal)),
                'min_over_max_u_pivot': ratio,
                'pivot_count': int(len(pivots)), 'pivot_vector_valid': True,
                'all_lu_entries_finite': True,
                'deterministic_local_solve_relative_residual': solve_residual})
            factors.append((nodes.copy(), dofs, lu, pivots))
    warning_rows = [{'category': row.category.__name__, 'message': str(row.message),
                     'filename': row.filename, 'line': row.lineno}
                    for row in caught]

    def apply(rhs):
        source = np.asarray(rhs, dtype=np.float64)
        result = np.zeros_like(source)
        for nodes, dofs, lu, pivots in factors:
            local = linalg.lu_solve((lu, pivots), source[dofs], check_finite=False)
            weights = np.repeat(1.0 / membership[nodes].astype(np.float64), 3)
            np.add.at(result, dofs, local * weights)
        if not np.isfinite(result).all():
            raise ValueError('Schwarz application returned nonfinite values')
        return result

    # Estimate condition only for the three weakest relative-pivot patches to
    # bound cost. These are estimates, not rigorous condition-number bounds.
    weakest = sorted(range(len(factors)), key=lambda i: patch_rows[i]['min_over_max_u_pivot'])[:3]
    condition_rows = []
    for patch_index in weakest:
        nodes, dofs, lu, pivots = factors[patch_index]
        dimension = len(dofs)
        inverse = LinearOperator((dimension, dimension),
            matvec=lambda x, lu=lu, pivots=pivots: linalg.lu_solve((lu, pivots), x, check_finite=False),
            rmatvec=lambda x, lu=lu, pivots=pivots: linalg.lu_solve((lu, pivots), x, trans=1, check_finite=False),
            dtype=np.float64)
        inverse_norm1 = float(onenormest(inverse, t=2, itmax=5))
        condition_rows.append({'patch_index': patch_index,
            'node_count': int(len(nodes)), 'dof_count': dimension,
            'min_over_max_u_pivot': patch_rows[patch_index]['min_over_max_u_pivot'],
            'matrix_one_norm': patch_rows[patch_index]['matrix_one_norm'],
            'inverse_one_norm_estimate': inverse_norm1,
            'condition_one_norm_estimate': patch_rows[patch_index]['matrix_one_norm'] * inverse_norm1,
            'method': 'SciPy onenormest(t=2,itmax=5) on local LU inverse; estimate only'})
    if not np.isfinite([row['condition_one_norm_estimate'] for row in condition_rows]).all():
        raise ValueError('patch condition estimator returned nonfinite values')
    return apply, {
        'factor_workspace_estimate_bytes': factor_workspace_estimate,
        'patch_factorization_flops_estimate': int(sum((2.0 / 3.0) * (3 * len(p)) ** 3 for p in patches)),
        'largest_patch_dof_dimension': largest_dim,
        'factor_count': len(factors),
        'pivot_count_total': int(sum(row['pivot_count'] for row in patch_rows)),
        'all_factors_finite': True,
        'all_pivot_vectors_valid': True,
        'minimum_relative_pivot_ratio': min(row['min_over_max_u_pivot'] for row in patch_rows),
        'maximum_local_solve_relative_residual': max(row['deterministic_local_solve_relative_residual'] for row in patch_rows),
        'patch_factor_diagnostics': patch_rows,
        'condition_estimates_for_three_weakest_pivot_patches': condition_rows,
        'factorization_runtime_warnings': warning_rows,
    }


def run():
    if not REPORT.is_file():
        raise FileNotFoundError(REPORT)
    event_path = OUT / 'events.jsonl'
    if event_path.exists():
        raise FileExistsError(f'refusing to overwrite prior event ledger: {event_path}')
    started = time.monotonic()
    timeout_record = {'expired': False}

    def on_timeout(_signum, _frame):
        timeout_record['expired'] = True
        raise TimeoutError('strict 120-second study cap reached')

    previous_handler = signal.signal(signal.SIGALRM, on_timeout)
    signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
    completed = []
    try:
        report, matrix, vectors, topology, sidecar_hashes = STUDY.load_hashed_operator(REPORT)
        binding = report.get('binding', {})
        if report.get('schema') != 'numi.cardboard.cached-operator-audit.v1' or \
           binding.get('capture_run_fingerprints_match') is not True or \
           not binding.get('source_hashes_verified_against_prediction') or \
           report.get('capture_integrity', {}).get('production_inputs_bitwise_unchanged') is not True:
            raise ValueError('source175 operator report admission failed')
        rhs = np.asarray(vectors['native_rhs_free_impulse_Ns'], dtype=np.float64)
        free = np.asarray(vectors['free_node_ids'], dtype=np.int64)
        if rhs.shape != (matrix.shape[0],) or matrix.shape[0] != 3 * len(free) or \
           not np.isfinite(rhs).all() or not np.linalg.norm(rhs) > 0.0:
            raise ValueError('captured source175 A/b/free-node dimensions are invalid')
        topo = {'nodes': np.asarray(topology['nodes']),
                'node_count': int(np.asarray(topology['node_count']).reshape(-1)[0])}
        previous_study_path = OUT.parent / 'cpu-schwarz-64-128/study.py'
        previous_study_hash = sha256(previous_study_path)
        low_checks = []
        for patch_size, stride in ((16, 8), (32, 16)):
            old_patches, old_membership = STUDY.build_graph_patches(topo, free, patch_size, stride)
            new_patches, new_membership = build_graph_patches(topo, free, patch_size, stride)
            if len(old_patches) != len(new_patches) or \
               any(not np.array_equal(a, b) for a, b in zip(old_patches, new_patches)) or \
               not np.array_equal(old_membership, new_membership):
                raise ValueError(f'new deterministic graph traversal changed existing {patch_size}-node membership')
            low_checks.append({'patch_nodes': patch_size, 'anchor_stride': stride,
                               'membership_identical_to_reviewed_builder': True,
                               'patch_count': len(old_patches)})
        if time.monotonic() - started >= MAX_SECONDS:
            raise TimeoutError('time cap before first case')
        retained_bytes = int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes +
            sum(np.asarray(value).nbytes for value in vectors.values()) +
            sum(np.asarray(value).nbytes for value in topology.values()))
        gmres_workspace_estimate = int((3 * (RESTART + 1) + 12) * matrix.shape[0] * 8 +
                                       (RESTART + 1) * (RESTART + 1) * 8)
        for patch_size, stride in ((64, 32), (128, 64), (256, 128)):
            if time.monotonic() - started >= MAX_SECONDS:
                raise TimeoutError(f'time cap before {patch_size}-node case')
            patches, membership = build_graph_patches(topo, free, patch_size, stride)
            estimated_factor_bytes = sum(3 * (3 * len(patch)) ** 2 * 8 +
                                         4 * (3 * len(patch)) * 8 for patch in patches)
            estimated_peak = retained_bytes + gmres_workspace_estimate + estimated_factor_bytes
            if estimated_peak > MAX_HOST_BYTES:
                raise MemoryError(f'{patch_size}-node estimated host peak {estimated_peak} exceeds 1 GiB')
            membership_file = OUT / f'patch-membership-{patch_size}.json'
            member_object = {'patch_nodes': patch_size, 'anchor_stride': stride,
                'patches': [patch.astype(int).tolist() for patch in patches],
                'per_node_patch_coverage': membership.astype(int).tolist()}
            membership_file.write_text(json.dumps(member_object, sort_keys=True, separators=(',', ':')) + '\n')
            start_event = {'event': 'case_start', 'patch_nodes': patch_size,
                'anchor_stride': stride, 'patch_count': len(patches),
                'membership_file': membership_file.name,
                'membership_sha256': sha256(membership_file),
                'preflight_estimated_peak_bytes': estimated_peak,
                'elapsed_seconds': time.monotonic() - started}
            append_jsonl(event_path, start_event)
            precondition, factors = build_diagnostic_preconditioner(matrix, free, patches,
                membership, retained_bytes + gmres_workspace_estimate)
            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter('always', RuntimeWarning)
                gmres = STUDY.restarted_right_gmres(matrix, rhs, precondition,
                    restart=RESTART, max_iterations=MAX_ITERATIONS,
                    relative_tolerance=TOLERANCE)
            gmres_warnings = [{'category': item.category.__name__, 'message': str(item.message),
                               'filename': item.filename, 'line': item.lineno}
                              for item in captured_warnings]
            case = {'event': 'case_complete', 'patch_nodes': patch_size,
                'anchor_stride': stride, 'patch_count': len(patches),
                'membership_file': membership_file.name,
                'membership_sha256': sha256(membership_file),
                'minimum_patch_nodes': min(len(patch) for patch in patches),
                'maximum_patch_nodes': max(len(patch) for patch in patches),
                'gmres_restart': gmres['restart'], 'gmres_iteration_cap': gmres['max_iterations'],
                'gmres_relative_tolerance': gmres['relative_tolerance'],
                'gmres_status': gmres['status'],
                'gmres_true_relative_residual': gmres['final_true_relative_residual'],
                'gmres_residual_history': gmres['true_residual_history'],
                'gmres_operator_matvecs': gmres['true_operator_matvec_count'],
                'gmres_preconditioner_applications': gmres['preconditioner_application_count'],
                'factor_diagnostics': factors, 'gmres_runtime_warnings': gmres_warnings,
                'estimated_host_peak_bytes': estimated_peak,
                'elapsed_seconds': time.monotonic() - started}
            append_jsonl(event_path, case)
            completed.append(case)
            del precondition, factors, patches, membership
            if time.monotonic() - started >= MAX_SECONDS:
                raise TimeoutError('strict 120-second study cap reached after preserving completed case')
        import contextlib
        import io
        config = io.StringIO()
        with contextlib.redirect_stdout(config):
            np.show_config()
        summary = {'schema': 'numi.cardboard.source175-cpu-schwarz-1024-study.v1',
            'classification': 'bounded CPU exploratory preconditioner-size sensitivity only; source175 is a rejected Newton linearization; no native RAP or physical qualification',
            'binding': {'operator_report': str(REPORT), 'operator_report_sha256': sha256(REPORT),
                'prediction_sha256': binding['prediction_sha256'], 'receipt_sha256': binding['receipt_sha256'],
                'stderr_sha256': binding['stderr_sha256'], 'capture_sha256': binding['capture_sha256'],
                'source_snapshot': binding['source_snapshot'], 'selection': report['capture_selection'],
                'operator_sidecar_sha256': sidecar_hashes,
                'runner_script_sha256': sha256(SCRIPT),
                'matrix_study_sha256': sha256(TOOL_PATH),
                'prior_16_32_runner_sha256': previous_study_hash,
                'warning_prone_preliminary_report_sha256': sha256(PRELIMINARY),
                'matrix_sha256': sidecar_hashes[REPORT.stem + '.operator.npz']},
            'runtime': {'python': '3.10.12', 'numpy': np.__version__,
                'scipy': __import__('scipy').__version__, 'blas_configuration': config.getvalue()},
            'operator': {'shape': list(matrix.shape), 'nnz': int(matrix.nnz),
                'native_action_parity_relative': report['operator']['action_parity']['relative_l2_error'],
                'captured_rhs_l2_Ns': float(np.linalg.norm(rhs)), 'matrix_assembly': report['operator']['assembly']},
            'method': {'rhs': 'exact source175 captured center residual impulse',
                'operator': 'same hash-bound nonsymmetric A; no symmetrization',
                'preconditioner': 'offline additive graph Schwarz with node partition-of-unity; exact prior traversal extended to larger patch sizes',
                'gmres': 'CPU right-preconditioned GMRES; restart16, max1024, true residual b-Ax each iteration, relative tolerance 1e-8; Z@y uses reviewed scalar math.fsum',
                'native_rap': 'not read or compared; this task is not native RAP qualification',
                'condition_estimates': 'all patches report LU pivot ratios and deterministic local solve residuals; onenormest condition estimates are computed for the three weakest relative-pivot patches only and are not rigorous bounds',
                'factor_local_matvec': 'rowwise scalar math.fsum, replacing the preliminary run diagnostic matmul calls that emitted Accelerate FP-status RuntimeWarnings despite finite results'},
            'limit_checks': {'wall_time_seconds': MAX_SECONDS, 'elapsed_seconds': time.monotonic() - started,
                'host_bytes': MAX_HOST_BYTES, 'retained_operator_and_sidecar_bytes_estimate': retained_bytes,
                'gmres_workspace_bytes_estimate': gmres_workspace_estimate,
                'maximum_case_estimated_host_peak_bytes': max(row['estimated_host_peak_bytes'] for row in completed),
                'observed_process_peak_rss_bytes': int(__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss),
                'process_peak_rss_units': 'bytes on macOS'},
            'existing_builder_membership_checks': low_checks,
            'case_event_ledger': str(event_path), 'case_event_ledger_sha256': sha256(event_path),
            'cases': completed,
            'limits': ['A is from source175 root3/Newton23, a rejected/diagnostic linearization rather than an accepted physical step.',
                'CPU Schwarz is not production Matter FGMRES and cannot validate native preconditioner execution.',
                'Native RAP is inadmissible and was not loaded or compared.',
                'Condition estimates are bounded iterative estimates, not rigorous bounds or qualification gates.']}
        summary_path = OUT / 'report.json'
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
        append_jsonl(event_path, {'event': 'study_complete', 'report_file': summary_path.name,
                                  'report_sha256': sha256(summary_path),
                                  'elapsed_seconds': time.monotonic() - started})
        return summary
    except Exception as error:
        append_jsonl(event_path, {'event': 'study_failure', 'type': type(error).__name__,
            'message': str(error), 'completed_patch_sizes': [row['patch_nodes'] for row in completed],
            'timeout': timeout_record['expired'], 'elapsed_seconds': time.monotonic() - started})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


if __name__ == '__main__':
    try:
        result = run()
    except Exception as error:
        print(f'EXPLORATORY STUDY INCOMPLETE: {type(error).__name__}: {error}')
        raise
    print(json.dumps({'report': str(OUT / 'report.json'), 'elapsed_seconds': result['limit_checks']['elapsed_seconds'],
        'cases': [{'patch_nodes': row['patch_nodes'], 'status': row['gmres_status'],
                   'true_relative_residual': row['gmres_true_relative_residual']}
                  for row in result['cases']]}, sort_keys=True))
