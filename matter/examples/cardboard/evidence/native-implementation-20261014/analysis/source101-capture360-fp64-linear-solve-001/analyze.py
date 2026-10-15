"""CPU-only solve of the retained cached matrix; no constitutive/native claim."""
from pathlib import Path
import datetime
import hashlib
import json
import platform
import shutil
import warnings

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, onenormest, splu

BASE = Path('/Users/home/cardboard-implementation-20261014')
INPUT = BASE / 'analysis/source101-control2-newton23-audit-004/result-360'
OUT = BASE / 'analysis/source101-capture360-fp64-linear-solve-001'
OUT.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
names = ['environment-0-operator-audit.operator.npz',
         'environment-0-operator-audit.vectors.npz',
         'environment-0-operator-audit.json']
bindings = {str(INPUT / name): sha(INPUT / name) for name in names}
for path in [BASE / 'solver/reset-augmentation-first-failure-capture-360' / name
             for name in ['prediction.json', 'receipt.json', 'operator-capture']]:
    bindings[str(path)] = sha(path)
v = np.load(INPUT / names[1])
A = sparse.load_npz(INPUT / names[0]).tocsc()
b = v['native_rhs_free_impulse_Ns']
x_native = v['direction']
assert A.shape == (2538, 2538) and A.nnz == 71046
assert np.isfinite(A.data).all() and np.isfinite(b).all() and np.isfinite(x_native).all()
assert np.allclose(A @ x_native, v['assembled_action_Ns'], rtol=0, atol=1e-15)
with warnings.catch_warnings(record=True) as captured_warnings:
    warnings.simplefilter('always')
    lu = splu(A, permc_spec='COLAMD', diag_pivot_thresh=1.0,
              options={'Equil': True})
    solution = lu.solve(b)
    residual64 = b - A @ solution
    quantized = solution.astype(np.float32).astype(np.float64)
    residual_quantized = b - A @ quantized
    residual_native = b - A @ x_native
    inverse = LinearOperator(A.shape, dtype=np.float64,
        matvec=lambda q: lu.solve(q), rmatvec=lambda q: lu.solve(q, trans='T'),
        matmat=lambda q: lu.solve(q))
    np.random.seed(1729)
    inverse_norm1_estimate = float(onenormest(inverse, t=3, itmax=8))
norm = lambda q: float(np.linalg.norm(q))
norm1 = float(sparse.linalg.norm(A, 1))
rhs_norm = norm(b)
forcing_target = 0.015625 * rhs_norm
assert norm(residual64) / rhs_norm < 1e-8
assert norm(residual_native) > forcing_target
assert all(sha(Path(p)) == digest for p, digest in bindings.items())
np.savez_compressed(OUT / 'linear-vectors.npz', fp64_solution=solution,
    fp32_quantized_solution=quantized, fp64_residual=residual64,
    fp32_quantized_residual=residual_quantized, captured_native_residual=residual_native)
report = dict(schema='numi.cardboard.captured-cached-fp64-linear-solve.v1',
    source_snapshot='source-boundary-reset-order-101', control_step=2, newton_iteration=23,
    environment=0, operator_scope='FP64 reconstruction of the captured native cached-tangent matrix; exact unsymmetric CSR retained',
    matrix_shape=list(A.shape), matrix_nnz=A.nnz,
    algorithm=dict(library='SciPy SuperLU', permc_spec='COLAMD', diag_pivot_thresh=1.0,
                   equilibrate=True, condition_estimator='onenormest inverse via LU solves and transpose solves',
                   condition_estimator_t=3, condition_estimator_itmax=8, numpy_seed=1729),
    metrics=dict(rhs_l2_Ns=rhs_norm, unchanged_forcing=0.015625,
        forcing_target_Ns=forcing_target, fp64_solution_l2_m_per_s=norm(solution),
        captured_native_solution_l2_m_per_s=norm(x_native),
        captured_native_solution_relative_distance_from_fp64=norm(x_native-solution)/norm(solution),
        fp64_residual_l2_Ns=norm(residual64), fp64_residual_relative_to_rhs=norm(residual64)/rhs_norm,
        fp64_residual_backward_error_norm2_numerator_over_norm1_bound=norm(residual64)/(norm1*norm(solution)+rhs_norm),
        fp32_quantized_fp64_solution_residual_l2_Ns=norm(residual_quantized),
        captured_native_solution_residual_l2_Ns=norm(residual_native),
        maximum_fp64_solution_component_m_per_s=float(np.max(np.abs(solution))),
        operator_norm1=norm1, inverse_norm1_iterative_lower_estimate=inverse_norm1_estimate,
        condition1_iterative_lower_estimate=norm1*inverse_norm1_estimate),
    software=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
    warnings=[str(w.message) for w in captured_warnings], inputs_sha256=bindings,
    limits=['Zero native or physical transactions executed.',
        'This solves captured cached matrices; it does not independently validate the paper constitutive law or its derivative.',
        'FP64 matrix reconstruction is not the same arithmetic as a native Metal matvec.',
        'The residual of one FP32-quantized solution is a sensitivity check, not a lower bound over all representable FP32 solutions or a measured noise floor.',
        'The iterative condition estimate is a lower estimate, not an exact singular-value condition number.',
        'No tolerance, physical parameter, accepted state or runtime source is changed.'],
    physical_validation=False, native_transactions=0,
    finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
(OUT / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
shutil.copy2(__file__, OUT / 'analyze.py')
(OUT / 'manifest.json').write_text(json.dumps(dict(schema='numi.cardboard.analysis-package.v1',
    files={p.name: sha(p) for p in sorted(OUT.iterdir()) if p.is_file()}), indent=2) + '\n')
print(json.dumps(report['metrics'], indent=2))
