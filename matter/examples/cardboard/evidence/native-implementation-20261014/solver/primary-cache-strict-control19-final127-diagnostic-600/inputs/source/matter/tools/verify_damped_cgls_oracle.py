#!/usr/bin/env python3
"""Independent 2x2 spectral oracle for the full-zero damped CGLS recurrence.

This is a CPU contract test, not a Metal/native execution or physics qualification.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def matvec(a, x):
    return [math.fsum(a[i][j] * x[j] for j in range(2)) for i in range(2)]


def transpose_matvec(a, x):
    return [math.fsum(a[i][j] * x[i] for i in range(2)) for j in range(2)]


def add(a, b):
    return [a[i] + b[i] for i in range(2)]


def sub(a, b):
    return [a[i] - b[i] for i in range(2)]


def scale(a, value):
    return [value * x for x in a]


def norm(a):
    return math.sqrt(dot(a, a))


def cgls_full_zero(a, b, mu, iterations):
    """Solve min ||Ax-b||^2 + mu^2||x||^2, starting from x=0."""
    x = [0.0, 0.0]
    r = list(b)
    s = transpose_matvec(a, r)
    gamma = dot(s, s)
    p = list(s)
    for _ in range(iterations):
        q = matvec(a, p)
        denominator = dot(q, q) + mu * mu * dot(p, p)
        if not math.isfinite(denominator) or denominator <= 0.0:
            return x, r, "breakdown"
        alpha = gamma / denominator
        x = add(x, scale(p, alpha))
        r = sub(r, scale(q, alpha))
        s = sub(transpose_matvec(a, r), scale(x, mu * mu))
        next_gamma = dot(s, s)
        beta = next_gamma / gamma
        p = add(s, scale(p, beta))
        gamma = next_gamma
    return x, r, "iteration_cap"


def legacy_cgls_unregularized(a, b, iterations):
    """Separate old recurrence used to check the mu=0 branch."""
    x = [0.0, 0.0]
    r = list(b)
    s = transpose_matvec(a, r)
    gamma = dot(s, s)
    p = list(s)
    for _ in range(iterations):
        q = matvec(a, p)
        denominator = dot(q, q)
        if not math.isfinite(denominator) or denominator <= 0.0:
            return x, r, "breakdown"
        alpha = gamma / denominator
        x = add(x, scale(p, alpha))
        r = sub(r, scale(q, alpha))
        s = transpose_matvec(a, r)
        next_gamma = dot(s, s)
        beta = next_gamma / gamma
        p = add(s, scale(p, beta))
        gamma = next_gamma
    return x, r, "iteration_cap"


def spectral_tikhonov_2x2(a, b, mu):
    """Independent right-singular-vector expansion from eig(A^T A)."""
    c00 = math.fsum(a[i][0] * a[i][0] for i in range(2))
    c01 = math.fsum(a[i][0] * a[i][1] for i in range(2))
    c11 = math.fsum(a[i][1] * a[i][1] for i in range(2))
    radius = math.hypot(c00 - c11, 2.0 * c01)
    eigenvalues = [(c00 + c11 + radius) / 2.0,
                   (c00 + c11 - radius) / 2.0]
    if abs(c01) > 1e-30:
        v0 = [c01, eigenvalues[0] - c00]
        length = norm(v0)
        v0 = scale(v0, 1.0 / length)
    elif c00 >= c11:
        v0 = [1.0, 0.0]
    else:
        v0 = [0.0, 1.0]
    vectors = [v0, [-v0[1], v0[0]]]
    result = [0.0, 0.0]
    for sigma2, v in zip(eigenvalues, vectors):
        av = matvec(a, v)
        coefficient = dot(av, b) / (sigma2 + mu * mu)
        result = add(result, scale(v, coefficient))
    return result


def assert_close(a, b, tol):
    error = norm(sub(a, b))
    if error > tol:
        raise AssertionError(f"vector error {error} exceeds {tol}: {a} vs {b}")
    return error


def main():
    # Nonsymmetric, full-rank A exercises A and A^T orientation distinctly.
    a = [[2.0, -0.75], [0.5, 3.0]]
    b = [1.25, -2.0]
    results = []

    for mu in (0.0, 1.0e-3, 2.0e-2):
        cgls, residual, status = cgls_full_zero(a, b, mu, 2)
        spectral = spectral_tikhonov_2x2(a, b, mu)
        error = assert_close(cgls, spectral, 2.0e-12)
        # A separate legacy implementation must be exactly the same arithmetic
        # path for mu=0, not just a numerically similar result.
        if mu == 0.0:
            legacy, legacy_r, legacy_status = legacy_cgls_unregularized(a, b, 2)
            if cgls != legacy or residual != legacy_r or status != legacy_status:
                raise AssertionError("mu=0 recurrence differs from legacy recurrence")
        objective_residual = sub(matvec(a, cgls), b)
        objective = dot(objective_residual, objective_residual) + mu * mu * dot(cgls, cgls)
        results.append({
            "mu": mu, "candidate": cgls, "spectral_candidate": spectral,
            "candidate_vs_spectral_l2": error,
            "physical_residual_l2": norm(objective_residual),
            "regularized_objective": objective, "status": status,
        })

    capped, capped_r, capped_status = cgls_full_zero(a, b, 1.0e-3, 1)
    full, _, _ = cgls_full_zero(a, b, 1.0e-3, 2)
    if capped_status != "iteration_cap" or not norm(capped_r) > norm(sub(matvec(a, full), b)):
        raise AssertionError("iteration-limited full-total direction was misclassified")
    if norm(capped) <= 0.0:
        raise AssertionError("CGLS candidate must be the total correction from zero")

    # Bind the implementation terms so this oracle cannot silently outlive the
    # Metal recurrence it is intended to check.
    metal = Path(__file__).resolve().parents[1] / "src/metal/base_cgls.metalinc"
    runtime = Path(__file__).resolve().parents[1] / "src/runtime.mm"
    m = metal.read_text()
    r = runtime.read_text()
    required_metal = (
        "dampingSquared * dot(activeP, activeP)",
        "value.xyz - dampingSquared * xi.xyz",
    )
    for token in required_metal:
        if token not in m:
            raise AssertionError(f"Metal recurrence binding is missing {token!r}")
    if "atIndex:8u" not in r or "atIndex:9u" not in r:
        raise AssertionError("runtime does not bind the added CGLS kernel arguments")
    if "NM_FEM_BASE_CGLS_DAMPING_KG" not in r:
        raise AssertionError("runtime lacks the source-bound damping selector")

    print(json.dumps({
        "schema": "numi.cardboard.damped-cgls-svd-oracle.v1",
        "passed": True,
        "scope": "CPU recurrence plus independent 2x2 spectral Tikhonov oracle",
        "native_metal_executed": False,
        "full_step_starts_from_zero": True,
        "iteration_cap_case_status": capped_status,
        "iteration_cap_candidate_norm": norm(capped),
        "cases": results,
    }, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
