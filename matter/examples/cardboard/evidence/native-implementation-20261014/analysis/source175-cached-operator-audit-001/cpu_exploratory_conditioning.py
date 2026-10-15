#!/usr/bin/env python3
"""CPU-only direct solve and bounded Schwarz exploration for source-175 A.

This is deliberately separate from the native RAP comparison tool. It requires
only the source-bound cached-operator report and its hashed matrix/vector/
topology sidecars. It cannot qualify a Matter solve or admit the retrospective
RAP export; it records exact native input bindings and true residuals.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys, time, warnings
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, onenormest, splu

REPO_TOOLS = Path('/Users/home/numi-cardboard-20261006/matter/tools')
STUDY_PATH = REPO_TOOLS / 'cardboard_offline_conditioning_study.py'
SPEC = importlib.util.spec_from_file_location('cardboard_offline_conditioning_study', STUDY_PATH)
STUDY = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(STUDY)
MAX_SECONDS = 120.0
MAX_HOST_BYTES = 1 << 30


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def run(report_path: Path, out_dir: Path) -> dict:
    started = time.monotonic()
    report, A, vectors, topology, operator_hashes = STUDY.load_hashed_operator(report_path)
    if report.get('schema') != 'numi.cardboard.cached-operator-audit.v1':
        raise ValueError('unsupported source-bound operator report')
    binding = report.get('binding', {})
    if binding.get('capture_run_fingerprints_match') is not True:
        raise ValueError('operator capture fingerprints do not match prediction')
    if not binding.get('source_hashes_verified_against_prediction'):
        raise ValueError('source file bindings are missing')
    if report.get('capture_integrity', {}).get('production_inputs_bitwise_unchanged') is not True:
        raise ValueError('capture mutation guard did not pass')
    rhs = np.asarray(vectors['native_rhs_free_impulse_Ns'], dtype=np.float64)
    if rhs.shape != (A.shape[0],) or not np.isfinite(rhs).all() or np.linalg.norm(rhs) == 0:
        raise ValueError('captured center RHS is unavailable, malformed, or zero')
    free_nodes = np.asarray(vectors['free_node_ids'], dtype=np.int64)
    if free_nodes.size * 3 != A.shape[0]:
        raise ValueError('free-node ordering does not match A dimensions')

    # The nonsymmetric source matrix is used as captured; no symmetrization.
    csc = A.tocsc()
    factor = splu(csc)
    if not np.isfinite(factor.U.data).all() or not np.isfinite(factor.L.data).all():
        raise ValueError('sparse LU contains nonfinite factors')
    perm_r = np.asarray(factor.perm_r, dtype=np.int64)
    perm_c = np.asarray(factor.perm_c, dtype=np.int64)
    if not (np.array_equal(np.sort(perm_r), np.arange(A.shape[0])) and
            np.array_equal(np.sort(perm_c), np.arange(A.shape[1]))):
        raise ValueError('sparse LU permutation vectors are not complete permutations')
    diagonal = np.asarray(factor.U.diagonal(), dtype=np.float64)
    if not np.isfinite(diagonal).all() or np.any(diagonal == 0.0):
        raise ValueError('sparse LU has zero/nonfinite pivots')
    x = factor.solve(rhs)
    residual = rhs - A @ x
    rhs2 = float(np.linalg.norm(rhs))
    residual2 = float(np.linalg.norm(residual))
    backward = float(np.linalg.norm(residual, ord=np.inf) /
                     (float(np.asarray(abs(A).sum(axis=1)).max()) * float(np.linalg.norm(x, ord=np.inf)) +
                      float(np.linalg.norm(rhs, ord=np.inf))))
    if not np.isfinite(x).all() or not math.isfinite(residual2) or not math.isfinite(backward):
        raise ValueError('direct solve produced nonfinite solution or residual')
    norm1_A = float(np.asarray(abs(A).sum(axis=0)).max())
    inverse_op = LinearOperator(A.shape, matvec=factor.solve,
        rmatvec=lambda y: factor.solve(y, trans='T'), dtype=np.float64)
    est_inverse_norm1 = float(onenormest(inverse_op, t=2, itmax=5))
    cond1_estimate = norm1_A * est_inverse_norm1

    # Reuse the reviewed bounded graph-patch builder/factor/apply code.  These
    # are local CPU experiments only; no native preconditioner history is read.
    topology_data = {'nodes': np.asarray(topology['nodes']),
                     'node_count': int(np.asarray(topology['node_count']).reshape(-1)[0])}
    rhs_norm = float(np.linalg.norm(rhs))
    schwarz = []
    for patch_nodes, anchor_stride in ((16, 8), (32, 16)):
        patches, membership = STUDY.build_graph_patches(topology_data, free_nodes,
                                                        patch_nodes, anchor_stride)
        apply, detail = STUDY.schwarz_preconditioner(A, free_nodes, patches, membership)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', RuntimeWarning)
            gmres = STUDY.restarted_right_gmres(A, rhs, apply, restart=16,
                                                max_iterations=64, relative_tolerance=1.0e-8)
        runtime_warning_rows = [{'message': str(w.message), 'category': w.category.__name__}
                                for w in caught]
        exact_partition = [p.astype(int).tolist() for p in patches]
        membership_blob = json.dumps({'max_patch_nodes': patch_nodes,
            'anchor_stride': anchor_stride, 'patches': exact_partition,
            'per_node_patch_coverage': membership.astype(int).tolist()},
            sort_keys=True, separators=(',', ':')).encode()
        schwarz.append({
            'max_patch_nodes': patch_nodes,
            'anchor_stride': anchor_stride,
            'patch_count': int(len(patches)),
            'membership_sha256': hashlib.sha256(membership_blob).hexdigest(),
            'patches': exact_partition,
            'per_node_patch_coverage': membership.astype(int).tolist(),
            'dense_factor_bytes_estimate': detail['dense_factor_bytes_estimate'],
            'estimated_factorization_flops': detail['estimated_factorization_flops'],
            'gmres_restart': gmres['restart'],
            'gmres_iteration_cap': gmres['max_iterations'],
            'gmres_relative_tolerance': gmres['relative_tolerance'],
            'gmres_status': gmres['status'],
            'gmres_true_relative_residual': gmres['final_true_relative_residual'],
            'gmres_residual_history': gmres['true_residual_history'],
            'gmres_operator_matvecs': gmres['true_operator_matvec_count'],
            'gmres_preconditioner_applications': gmres['preconditioner_application_count'],
            'runtime_warnings': runtime_warning_rows,
        })
        if time.monotonic() - started > MAX_SECONDS:
            raise TimeoutError('CPU exploratory analysis exceeded 120 s guard')

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
      'schema': 'numi.cardboard.source175-cpu-exploratory-conditioning.v1',
      'classification': 'CPU diagnostic only; not a native solver run, not physical qualification, no RAP export consumed',
      'operator_report': str(report_path),
      'operator_report_sha256': sha(report_path),
      'operator_sidecar_sha256': operator_hashes,
      'native_binding': {
        'prediction_sha256': binding['prediction_sha256'],
        'receipt_sha256': binding['receipt_sha256'],
        'stderr_sha256': binding['stderr_sha256'],
        'operator_capture_sha256': binding['capture_sha256'],
        'source_snapshot': binding['source_snapshot'],
        'selector': report['capture_selection'],
        'world_fingerprint': binding['operator_fd_iteration']['world_fingerprint'],
        'execution_fingerprint': binding['operator_fd_iteration']['execution_fingerprint'],
      },
      'operator': {
        'dimension': int(A.shape[0]), 'nnz': int(A.nnz),
        'assembly': report['operator']['assembly'],
        'relative_skew_norm': report['operator']['relative_skew_norm'],
        'measured_native_action_relative_error': report['operator']['action_parity']['relative_l2_error'],
        'captured_rhs_l2_Ns': rhs2,
        'factorization': 'SciPy sparse SuperLU, source A as captured, no symmetrization',
      },
      'direct_sparse_lu': {
        'solution_finite': bool(np.isfinite(x).all()),
        'true_residual_l2_Ns': residual2,
        'true_relative_residual_l2': residual2 / rhs2,
        'normwise_backward_error_inf': backward,
        'U_pivot_count': int(diagonal.size),
        'row_and_column_permutations_are_complete': True,
        'permutation_sha256': hashlib.sha256(perm_r.astype('<i8').tobytes() + perm_c.astype('<i8').tobytes()).hexdigest(),
        'U_min_abs_diagonal': float(np.min(np.abs(diagonal))),
        'U_max_abs_diagonal': float(np.max(np.abs(diagonal))),
        'U_diagonal_ratio_min_over_max': float(np.min(np.abs(diagonal)) / np.max(np.abs(diagonal))),
        'one_norm_A': norm1_A,
        'inverse_one_norm_estimate': est_inverse_norm1,
        'condition_one_norm_estimate': cond1_estimate,
        'condition_estimate_method': 'onenormest with itmax=5 on sparse LU inverse; estimate only, not rigorous bound',
      },
      'schwarz_only': {
        'rhs_source': 'native captured center residual vector, direct right-preconditioned solve of audited A',
        'algorithm': 'reviewed offline graph-patch additive Schwarz with partition-of-unity node weights',
        'convergence_tolerance': 1.0e-8,
        'case_results': schwarz,
      },
      'limits': [
        'The native operator capture is from rejected root 3 / Newton 23, not an accepted physical step.',
        'A is reconstructed offline from production cached local tangents and exact captured geometry; the direct solve does not repeat a native transaction.',
        'The 175 rejected-conditioning export was not used; its environment-0 AP is mass-only and fails exact A*P comparison.',
        'The CPU Schwarz preconditioners are experimental and not equivalent to Matter RAP, production FGMRES, contact, or transaction semantics.',
        'The condition estimate is iterative and is not an upper bound or qualification gate.',
      ],
      'elapsed_seconds': time.monotonic() - started,
    }
    result_path = out_dir / 'cpu-exploratory-conditioning.json'
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    np.savez_compressed(out_dir / 'direct-solution.npz', x=x, rhs=rhs, true_residual=residual)
    payload['direct_solution_sidecar_sha256'] = sha(out_dir / 'direct-solution.npz')
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--operator-report', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    args = p.parse_args()
    try:
        result = run(args.operator_report, args.output_dir)
    except Exception as exc:
        print(f'CPU exploratory analysis rejected: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    print(json.dumps({'result': str(args.output_dir / 'cpu-exploratory-conditioning.json'),
                      'direct_true_relative_residual': result['direct_sparse_lu']['true_relative_residual_l2'],
                      'condition_1_estimate': result['direct_sparse_lu']['condition_one_norm_estimate'],
                      'schwarz': [{'max_patch_nodes': x['max_patch_nodes'], 'status': x['gmres_status'],
                                   'true_relative_residual': x['gmres_true_relative_residual']} for x in result['schwarz_only']['case_results']]},
                     sort_keys=True))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
