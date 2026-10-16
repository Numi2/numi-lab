"""Optional DD wrapper for the explicitly loaded paper equation reference."""
from __future__ import annotations

import numpy as np

import cardboard_compensated_fp64_base as v1
from cardboard_dd_fp64 import DD, add, div, mul, neg, sqrt, sub


def fb_row(p, old, z):
    """Compensated FB row with correct masked quotient scatter for mixed batches."""
    nrow = len(p)
    q, flow = v1.hill_dd(p, z[:, 6:12])
    dl = sub(DD.from_value(z[:, 12]), DD.from_value(old[:, 12]))
    eps = DD.from_value(p[:, 26])
    hard1 = div(mul(mul(DD.from_value(p[:, 17]), DD.from_value(p[:, 19])),
                    DD.from_value(p[:, 21])), DD.from_value(p[:, 10]))
    hard2 = div(mul(mul(DD.from_value(p[:, 18]), DD.from_value(p[:, 20])),
                    DD.from_value(p[:, 22])), DD.from_value(p[:, 10]))
    hard1 = mul(hard1, DD.from_value(z[:, 13]))
    hard2 = mul(hard2, DD.from_value(z[:, 14]))
    y = sub(add(add(DD.from_value(np.ones(nrow)), hard1), hard2), q)
    root = sqrt(add(add(mul(dl, dl), mul(y, y)), mul(eps, eps)))
    denominator = add(add(root, dl), y)
    use_quotient = denominator.rounded() > 1.0e-14
    numerator = sub(mul(eps, eps), mul(DD.from_value(2.0), mul(dl, y)))
    values = sub(sub(root, dl), y).rounded().copy()
    ids = np.flatnonzero(use_quotient)
    if ids.size:
        quotient = div(DD(numerator.hi[ids], numerator.lo[ids]),
                       DD(denominator.hi[ids], denominator.lo[ids])).rounded()
        values[ids] = quotient
    if not np.all(np.isfinite(values)):
        raise ArithmeticError('nonfinite FB value in compensated FP64 reference v2')
    branches = {'q': q.rounded(), 'q_low': q.lo.copy(), 'yield_y': y.rounded(),
                'yield_y_low': y.lo.copy(), 'dl': dl.rounded(), 'dl_low': dl.lo.copy(),
                'fb_denominator': denominator.rounded(), 'fb_quotient_branch': use_quotient}
    return values, (q, flow, dl, y, eps, root), branches


def jacobian(base, p, f, old, z):
    """Source166003 analytic Jacobian with a corrected DD FB row scatter."""
    p = v1._real_array(p, 'parameters')
    f = v1._real_array(f, 'deformation')
    old = v1._real_array(old, 'accepted state')
    z = v1._real_array(z, 'candidate state')
    j = base.jacobian(p, f, old, z)
    _, parts, _ = fb_row(p, old, z)
    q, flow, dl, y, eps, root = parts
    one = DD.from_value(np.ones(len(p)))
    coeff_dl = sub(div(dl, root), one).rounded()
    coeff_y = sub(div(y, root), one).rounded()
    positive_dl = dl.rounded() > 0.0
    positive_y = y.rounded() > 0.0
    if np.any(positive_dl):
        ids = np.flatnonzero(positive_dl)
        dl_p = DD(dl.hi[ids], dl.lo[ids]); y_p = DD(y.hi[ids], y.lo[ids])
        e_p = DD(eps.hi[ids], eps.lo[ids]); r_p = DD(root.hi[ids], root.lo[ids])
        d = mul(r_p, add(r_p, dl_p))
        coeff_dl[ids] = neg(div(add(mul(y_p, y_p), mul(e_p, e_p)), d)).rounded()
    if np.any(positive_y):
        ids = np.flatnonzero(positive_y)
        dl_p = DD(dl.hi[ids], dl.lo[ids]); y_p = DD(y.hi[ids], y.lo[ids])
        e_p = DD(eps.hi[ids], eps.lo[ids]); r_p = DD(root.hi[ids], root.lo[ids])
        d = mul(r_p, add(r_p, y_p))
        coeff_y[ids] = neg(div(add(mul(dl_p, dl_p), mul(e_p, e_p)), d)).rounded()
    row = np.zeros((len(p), 15), dtype=np.float64)
    row[:, 12] = coeff_dl
    for k in range(3): row[:, 6+k] = -mul(DD.from_value(coeff_y), flow[k]).rounded()
    for k in range(3, 6): row[:, 6+k] = -mul(DD.from_value(2.0*coeff_y), flow[k]).rounded()
    h13 = div(mul(mul(DD.from_value(p[:,17]), DD.from_value(p[:,19])),
                  DD.from_value(p[:,21])), DD.from_value(p[:,10]))
    h14 = div(mul(mul(DD.from_value(p[:,18]), DD.from_value(p[:,20])),
                  DD.from_value(p[:,22])), DD.from_value(p[:,10]))
    row[:,13] = mul(DD.from_value(coeff_y), h13).rounded()
    row[:,14] = mul(DD.from_value(coeff_y), h14).rounded()
    j[:,12,:] = row
    if not np.all(np.isfinite(j)):
        raise ArithmeticError('nonfinite DD Jacobian')
    return j


def residual(base, p, f, old, z):
    """Same source166003 equations; DD evaluation of q-dependent residuals."""
    r = base.residual(p, f, old, z)
    _, flow, dl, _, _, _ = fb_row(p, old, z)[1]
    dl_dd = dl
    z0 = DD.from_value(z[:, :6])
    old0 = DD.from_value(old[:, :6])
    for k in range(6):
        zk = DD(z0.hi[:, k], z0.lo[:, k])
        ok = DD(old0.hi[:, k], old0.lo[:, k])
        value = sub(sub(zk, ok), mul(dl_dd, flow[k]))
        r[:, k] = value.rounded()
    fb, _, _ = fb_row(p, old, z)
    r[:, 12] = fb
    if not np.all(np.isfinite(r)):
        raise ArithmeticError('nonfinite residual in compensated FP64 reference v2')
    return r


STRICT_RESIDUAL = 2.0e-12
STRICT_DYNAMIC_CORRECTION = 2.0e-11


def project_with_metrics(ref, p, f, old, maxiter=80):
    """Run the source173-qualified projection and return both strict gates.

    The qualified path first uses source173's safeguarded projection/refinement,
    then its bounded continuation until both the unchanged scaled-residual
    (2e-12) and dynamic-correction (2e-11) gates pass.  The ceilings and row
    scales come from the hash-bound equation source; callers cannot loosen them.
    """
    p = v1._real_array(p, 'parameters')
    f = v1._real_array(f, 'deformation')
    old = v1._real_array(old, 'accepted state')
    if maxiter != 80:
        raise ValueError('strict v2 projection uses the fixed qualified 80-iteration seed ceiling')
    class _SeedProjection:
        # project_centers calls project_resilient, whose seed calls project.
        # Keep that seed on the documented initial safeguarded path without
        # recursively re-entering this strict wrapper. Residual/Jacobian remain
        # the compensated v2 equations throughout.
        residual = ref.residual
        jacobian = ref.jacobian

        @staticmethod
        def project(sp, sf, sold, maxiter=80):
            return v1.project(ref, sp, sf, sold, maxiter=maxiter)

    initial = ref.base.project_centers(_SeedProjection, p, f, old, p[:, 26])
    return ref.base.continue_projection_tight(ref, p, f, old, p[:, 26], initial)


class Reference:
    def __init__(self, base):
        self.base = base
        self.residual = lambda p, f, old, z: residual(base, p, f, old, z)
        self.jacobian = lambda p, f, old, z: jacobian(base, p, f, old, z)

    def project_with_metrics(self, p, f, old):
        return project_with_metrics(self, p, f, old)

    def project(self, p, f, old, maxiter=80):
        z, iterations, scaled_residual, correction, ok, reasons, provenance = \
            project_with_metrics(self, p, f, old, maxiter=maxiter)
        if not np.all(ok):
            bad = np.flatnonzero(~ok)
            details = [(int(i), str(reasons[i]), float(scaled_residual[i]),
                        float(correction[i])) for i in bad[:8]]
            raise RuntimeError(f'strict v2 projection gates failed for rows {details}')
        return z, iterations, scaled_residual


def tangent(base, p, f, old, z, h):
    """Implicit tangent using DD Jacobian and exact complex-step F derivative."""
    p = v1._real_array(p, 'parameters')
    f = v1._real_array(f, 'deformation')
    old = v1._real_array(old, 'accepted state')
    z = v1._real_array(z, 'candidate state')
    h = v1._real_array(h, 'deformation direction')
    if h.shape != f.shape:
        raise ValueError('tangent direction shape mismatch')
    step = 1e-25
    fc = f.astype(np.complex128) + 1j * step * h
    rf = np.imag(base.residual(p, fc, old, z)) / step
    j = jacobian(base, p, f, old, z)
    dz = np.linalg.solve(j, -rf[..., None])[..., 0]
    zc = z.astype(np.complex128) + 1j * step * dz
    return np.imag(base.stress(p, fc, zc)) / step
