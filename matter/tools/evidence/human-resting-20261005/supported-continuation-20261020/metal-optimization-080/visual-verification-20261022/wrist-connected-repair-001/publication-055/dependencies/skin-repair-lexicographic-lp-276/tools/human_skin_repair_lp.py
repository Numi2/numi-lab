"""Offline skin-repair LP feasibility before shape regularization.

No physical state or simulation acceptance authority lives here. Callers must
certify the emitted Float32 geometry with the existing anatomical owners.
"""
from __future__ import annotations
import time
import numpy as np


def solve_contact_lp_lexicographic(c, A, b, bounds, slack_start, slack_count,
                                   *, time_limit=60., slack_sum_tolerance=1e-7):
    """Minimize contact slack first, then the supplied shape objective.

    Contact slack variables must be nonnegative, occupy one explicit contiguous
    range and retain their original constraints/bounds. Their supplied objective
    coefficients are ignored, avoiding large artificial penalties. No row or
    coefficient is dropped; max-norm row scaling is algebraically equivalent.
    Returns the final scipy OptimizeResult and telemetry for both phases.
    Positive optimal slack still means the geometric constraints are infeasible.
    """
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix, diags, vstack
    objective = np.asarray(c, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    matrix = csr_matrix(A, dtype=np.float64)
    if objective.ndim != 1 or rhs.ndim != 1 or matrix.shape != (len(rhs), len(objective)):
        raise ValueError('LP matrix/vector shape mismatch')
    if not (np.isfinite(objective).all() and np.isfinite(rhs).all()
            and np.isfinite(matrix.data).all()):
        raise ValueError('LP contains nonfinite coefficients')
    if (isinstance(slack_start, bool) or isinstance(slack_count, bool)
            or int(slack_start) != slack_start or int(slack_count) != slack_count
            or slack_start < 0 or slack_count <= 0
            or slack_start + slack_count > len(objective)):
        raise ValueError('invalid explicit contact slack domain')
    if len(bounds) != len(objective):
        raise ValueError('LP bounds do not cover variables')
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError('time limit must be finite and positive')
    if not np.isfinite(slack_sum_tolerance) or slack_sum_tolerance < 0:
        raise ValueError('slack tolerance must be finite and nonnegative')
    start, count = int(slack_start), int(slack_count)
    bound_rows = []
    for index, pair in enumerate(bounds):
        if len(pair) != 2:
            raise ValueError('each variable requires a lower/upper bound pair')
        lower, upper = pair
        if lower is not None and (not np.isfinite(lower)):
            raise ValueError('nonfinite lower bound; use None for unbounded')
        if upper is not None and (not np.isfinite(upper)):
            raise ValueError('nonfinite upper bound; use None for unbounded')
        if lower is not None and upper is not None and lower > upper:
            raise ValueError('reversed variable bounds')
        if start <= index < start + count and (lower is None or lower < 0):
            raise ValueError('contact slack must have a nonnegative lower bound')
        bound_rows.append((lower, upper))
    scale = np.asarray(abs(matrix).max(axis=1).toarray()).ravel()
    scale[scale == 0] = 1.
    if np.any(~np.isfinite(1. / scale)):
        raise ValueError('LP row scale exceeds representable range')
    normalized = diags(1. / scale) @ matrix
    normalized_rhs = rhs / scale
    if not np.isfinite(normalized_rhs).all():
        raise ValueError('LP normalized right-hand side overflow')
    primary = np.zeros_like(objective)
    primary[start:start + count] = 1.
    began = time.monotonic()
    options = {'time_limit': float(time_limit), 'presolve': True,
               'primal_feasibility_tolerance': 1e-9,
               'dual_feasibility_tolerance': 1e-9}
    first = linprog(primary, A_ub=normalized, b_ub=normalized_rhs,
                    bounds=bound_rows, method='highs', options=options)
    info = {'phase_one_status': int(first.status),
            'phase_one_message': str(first.message),
            'phase_one_seconds': time.monotonic() - began,
            'rows': matrix.shape[0], 'variables': matrix.shape[1],
            'row_max_scale_range': [float(scale.min(initial=1)),
                                    float(scale.max(initial=1))],
            'slack_sum_tolerance': float(slack_sum_tolerance)}
    if not first.success:
        info['phase_two_run'] = False
        return first, info
    optimum = max(0., float(first.x[start:start + count].sum()))
    info['minimum_contact_slack_sum'] = optimum
    elapsed = time.monotonic() - began
    remaining = float(time_limit) - elapsed
    if remaining <= 0:
        # Do not silently present the feasibility solution as shape-optimized.
        info['phase_two_run'] = False
        info['time_budget_exhausted_after_phase_one'] = True
        first.success = False
        first.status = 1
        first.message = 'Time limit after contact-feasibility phase; shape phase not run'
        return first, info
    secondary = objective.copy()
    secondary[start:start + count] = 0.
    # Unit scaling of an entire objective preserves its minimizers.
    objective_scale = float(np.max(np.abs(secondary), initial=1.))
    secondary /= objective_scale
    phase_two_matrix = vstack((normalized, csr_matrix(primary[None, :])), format='csr')
    phase_two_rhs = np.r_[normalized_rhs, optimum + slack_sum_tolerance]
    options['time_limit'] = remaining
    t = time.monotonic()
    final = linprog(secondary, A_ub=phase_two_matrix, b_ub=phase_two_rhs,
                    bounds=bound_rows, method='highs', options=options)
    info.update(phase_two_run=True, phase_two_status=int(final.status),
                phase_two_message=str(final.message),
                phase_two_seconds=time.monotonic()-t,
                shape_objective_scale=objective_scale)
    if final.success:
        residual = np.asarray(matrix @ final.x - rhs)
        info.update(contact_slack_sum=float(final.x[start:start+count].sum()),
                    contact_slack_max=float(final.x[start:start+count].max(initial=0)),
                    maximum_original_constraint_violation=float(residual.max(initial=0)))
        final.fun = float(objective @ final.x)
    return final, info
