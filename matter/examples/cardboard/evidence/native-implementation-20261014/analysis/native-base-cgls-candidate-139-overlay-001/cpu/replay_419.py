#!/usr/bin/env python3
"""Source-bound CPU-only CGLS reference replay for capture 419.

This measures a proposed unregularized truncated least-squares direction only;
it does not establish GPU recurrence parity or authorize a Newton certificate.
"""
from __future__ import annotations
import hashlib, importlib.util, json
from pathlib import Path
import numpy as np
from scipy import sparse
from cgls_reference import run_cgls_fp32

BASE = Path('/Users/home/cardboard-implementation-20261014')
CAPTURE = BASE / 'solver/two-stage-first-failure-rich-fd-419/operator-capture'
INPUT = BASE / 'analysis/captured-base-state-cauchy-419-001'
SOURCE = BASE / 'snapshots/source-status5-vtccd-diagnostics-125'
ASSEMBLER = SOURCE / 'matter/tools/cardboard_cached_operator_audit.py'
EXPECTED_CAPTURE = 'f2d42314cfee0428ecd285ebb9dbd5da1556624e3e92af1d12240fd8106b309a'
EXPECTED_ASSEMBLER = '2ce80f9b3c989efd058bdbcc29661d10701061cfde459d64cca06c0d6caeee38'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    report = json.loads((INPUT / 'report.json').read_text())
    if report['input_bindings']['capture_sha256'] != EXPECTED_CAPTURE:
        raise RuntimeError('419 report does not bind the expected source capture')
    if sha(CAPTURE) != EXPECTED_CAPTURE:
        raise RuntimeError('source capture hash mismatch')
    if sha(ASSEMBLER) != EXPECTED_ASSEMBLER:
        raise RuntimeError('source125 operator assembler hash mismatch')
    if int(report['selection']['root']) != 1 or int(report['selection']['newton_iteration']) != 48:
        raise RuntimeError('capture selector differs from the sealed 419 source case')

    module_spec = importlib.util.spec_from_file_location('source125_cached_operator_audit', ASSEMBLER)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError('cannot load the bound source125 assembler')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    # The primary source125 audit reader does not preserve JSON negative zero;
    # use the source-bound experiment's strict reader for exact captured words.
    experiment_path = INPUT / 'run_experiment.py'
    exp_spec = importlib.util.spec_from_file_location('capture419_experiment', experiment_path)
    if exp_spec is None or exp_spec.loader is None:
        raise RuntimeError('cannot load the bound capture reader')
    experiment = importlib.util.module_from_spec(exp_spec)
    exp_spec.loader.exec_module(experiment)
    header, nodes, tets, integrity = experiment.strict_capture(CAPTURE)
    _, _, source_A, _, metadata = module.validate_and_assemble(
        header, nodes, tets, integrity, environment=0)
    matrix_file = sparse.load_npz(INPUT / 'A_free_csr.npz').tocsr()
    source_A = source_A.tocsr()
    difference = (matrix_file - source_A).tocsr()
    difference.eliminate_zeros()
    if matrix_file.shape != (2538, 2538) or matrix_file.nnz != 71046:
        raise RuntimeError('sealed 419 CSR dimensions/nonzero count changed')
    if difference.nnz:
        max_diff = float(np.max(np.abs(difference.data)))
        if not np.allclose(matrix_file.toarray(), source_A.toarray(), rtol=1e-13, atol=1e-14):
            raise RuntimeError(f'package CSR does not match captured assembly; max diff {max_diff}')
    if metadata['free_node_count'] != 846 or metadata['fixed_node_count'] != 36:
        raise RuntimeError('source capture free/fixed node counts changed')
    free_nodes = np.asarray(metadata['free_nodes'], dtype=np.uint32)
    rhs64 = np.fromfile(INPUT / 'base_rhs.float64le.bin', dtype='<f8')
    if rhs64.shape != (2538,) or not np.isfinite(rhs64).all():
        raise RuntimeError('419 source RHS shape or finiteness mismatch')
    rhs_from_capture = np.asarray(metadata['native_rhs_free_impulse_Ns'], dtype=np.float64)
    if rhs_from_capture.shape != rhs64.shape or not np.array_equal(rhs_from_capture, rhs64):
        if not np.allclose(rhs_from_capture, rhs64, rtol=0.0, atol=0.0):
            raise RuntimeError('sealed RHS is not exactly the assembled captured center RHS')

    results = []
    for cap in (32, 64):
        result = run_cgls_fp32(matrix_file, rhs64.astype(np.float32), free_nodes,
                               node_count=int(header['node_count_per_environment']), cap=cap)
        x_free = np.concatenate([result.direction[int(n), :3] for n in free_nodes])
        residual = rhs64.astype(np.float64) - source_A @ x_free.astype(np.float64)
        results.append({
            'cap': cap,
            'status': result.status,
            'iterations': result.iterations,
            'error': result.error,
            'fp32_reference_direction_l2_m_per_s': float(np.linalg.norm(x_free.astype(np.float64))),
            'independent_fp64_linear_residual_l2_Ns': float(np.linalg.norm(residual)),
            'initial_rhs_l2_Ns': float(np.linalg.norm(rhs64)),
            'reduction_ratio': float(np.linalg.norm(residual) / np.linalg.norm(rhs64)),
            'gamma_terminal': result.gamma,
        })
    payload = {
        'schema': 'numi.cardboard.base-cgls-cpu-replay-419.v1',
        'classification': 'CPU FP32 recurrence reference against source125 captured unsymmetric free-DOF A; diagnostic candidate only, no native parity or certificate claim',
        'bindings': {
            'capture_sha256': EXPECTED_CAPTURE,
            'source125_assembler_sha256': EXPECTED_ASSEMBLER,
            'input_report_sha256': sha(INPUT / 'report.json'),
            'A_free_csr_sha256': sha(INPUT / 'A_free_csr.npz'),
            'base_rhs_sha256': sha(INPUT / 'base_rhs.float64le.bin'),
            'shape': list(source_A.shape),
            'nnz': int(source_A.nnz),
            'free_nodes': len(free_nodes),
            'node_count': int(header['node_count_per_environment']),
            'dt_seconds': float(metadata['dt_seconds']),
            'matrix_semantics': metadata['assembly'],
        },
        'results': results,
        'limitations': [
            'No GPU execution was performed.',
            'The CPU recurrence uses FP32 matrix actions and is not claimed bitwise equivalent to the production matrix-free Metal actions.',
            'CGLS directions are alternative total corrections from the captured Newton center and do not inherit FGMRES convergence/certification.',
            'This replay does not include contact, rigid, field, cohesive, or physical nonlinear trial behavior.',
        ],
    }
    out = Path(__file__).resolve().parents[1] / 'cpu/replay_419.json'
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    print(json.dumps(payload, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
