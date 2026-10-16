"""Experimental error-compensated FP64 reference for source166003.

Only the Hill invariant / flow, yield gap, and Fischer--Burmeister row are
evaluated with two-word binary64 expansions. All words and intermediate
operations remain IEEE binary64. The authored equations and strict gates are
unchanged. This is a CPU diagnostic reference, not production code.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from cardboard_dd_fp64 import DD, add, div, mul, neg, sqrt, sub



def import_reference(path):
    if path is None:
        raise ValueError('an explicitly hash-bound source reference path is required')
    spec = importlib.util.spec_from_file_location('source166003_base_reference', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _real_array(x, name):
    a = np.asarray(x)
    if np.iscomplexobj(a):
        raise TypeError(f'{name}: complex inputs are unsupported by the DD reference')
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError(f'{name}: nonfinite input to the DD reference')
    return a


def _constant(x):
    return DD.from_value(x)


def _take(a: DD, mask):
    return DD(a.hi[mask], a.lo[mask])


def _mul3(a, b, c):
    return mul(mul(a, b), c)


def hill_dd(p, stress):
    """Return the authored Hill q and flow as binary64-rounded DD values."""
    p = _real_array(p, 'parameters')
    s = _real_array(stress, 'stress')
    if p.ndim != 2 or p.shape[1] < 27 or s.shape != (len(p), 6):
        raise ValueError('hill_dd expects parameters[N,>=27] and stress[N,6]')
    f, g, h, l, m, n = (_constant(p[:, k]) for k in range(11, 17))
    a, b, c, d, e, v = (_constant(s[:, k]) for k in range(6))
    eps = _constant(p[:, 25])
    bc = sub(b, c)
    ca = sub(c, a)
    ab = sub(a, b)
    terms = [
        mul(f, mul(bc, bc)),
        mul(g, mul(ca, ca)),
        mul(h, mul(ab, ab)),
        mul(_constant(2.0), mul(l, mul(d, d))),
        mul(_constant(2.0), mul(m, mul(e, e))),
        mul(_constant(2.0), mul(n, mul(v, v))),
        mul(eps, eps),
    ]
    q2 = terms[0]
    for term in terms[1:]:
        q2 = add(q2, term)
    q = sqrt(q2)
    if np.any(q.hi <= 0.0):
        raise ArithmeticError('nonpositive Hill norm in DD reference')
    numerators = [
        add(mul(g, sub(a, c)), mul(h, sub(a, b))),
        add(mul(f, sub(b, c)), mul(h, sub(b, a))),
        add(mul(f, sub(c, b)), mul(g, sub(c, a))),
        mul(l, d), mul(m, e), mul(n, v),
    ]
    flow = [div(vv, q) for vv in numerators]
    return q, flow


def _fb_row(p, old, z):
    """Compute only the FB row, retaining q/y low words through evaluation."""
    nrow = len(p)
    q, flow = hill_dd(p, z[:, 6:12])
    dl = sub(_constant(z[:, 12]), _constant(old[:, 12]))
    eps = _constant(p[:, 26])
    hard1 = div(_mul3(_constant(p[:, 17]), _constant(p[:, 19]),
                      _constant(p[:, 21])), _constant(p[:, 10]))
    hard2 = div(_mul3(_constant(p[:, 18]), _constant(p[:, 20]),
                      _constant(p[:, 22])), _constant(p[:, 10]))
    hard1 = mul(hard1, _constant(z[:, 13]))
    hard2 = mul(hard2, _constant(z[:, 14]))
    base = add(add(_constant(np.ones(nrow, dtype=np.float64)), hard1), hard2)
    y = sub(base, q)
    root = sqrt(add(add(mul(dl, dl), mul(y, y)), mul(eps, eps)))
    den = add(add(root, dl), y)
    # Match source166003's representation branch; only the arithmetic within
    # each algebraically equivalent branch is more accurate.
    use_quotient = den.rounded() > 1.0e-14
    numerator = sub(mul(eps, eps), mul(_constant(2.0), mul(dl, y)))
    direct = sub(sub(root, dl), y)
    values = direct.rounded().copy()
    if np.any(use_quotient):
        quotient = div(_take(numerator, use_quotient), _take(den, use_quotient)).rounded()
        values[use_quotient] = quotient[use_quotient]
    if not np.all(np.isfinite(values)):
        raise ArithmeticError('nonfinite FB value in DD reference')
    branches = {
        'q': q.rounded(), 'q_low': q.lo.copy(),
        'yield_y': y.rounded(), 'yield_y_low': y.lo.copy(),
        'dl': dl.rounded(), 'dl_low': dl.lo.copy(),
        'fb_denominator': den.rounded(), 'fb_quotient_branch': use_quotient,
    }
    return values, (q, flow, dl, y, eps, root), branches


def residual_dd(base, p, f, old, z):
    p = _real_array(p, 'parameters')
    f = _real_array(f, 'deformation')
    old = _real_array(old, 'accepted state')
    z = _real_array(z, 'candidate state')
    if p.ndim != 2 or f.shape != (len(p), 3, 3) or old.shape != z.shape or z.shape != (len(p), 15):
        raise ValueError('residual_dd shape mismatch')
    r = base.residual(p, f, old, z)
    fb, _, _ = _fb_row(p, old, z)
    r[:, 12] = fb
    if not np.all(np.isfinite(r)):
        raise ArithmeticError('nonfinite residual in DD reference')
    return r


def jacobian_dd(base, p, f, old, z):
    p = _real_array(p, 'parameters')
    f = _real_array(f, 'deformation')
    old = _real_array(old, 'accepted state')
    z = _real_array(z, 'candidate state')
    if p.ndim != 2 or f.shape != (len(p), 3, 3) or old.shape != z.shape or z.shape != (len(p), 15):
        raise ValueError('jacobian_dd shape mismatch')
    j = base.jacobian(p, f, old, z)
    _, parts, _ = _fb_row(p, old, z)
    q, flow, dl, y, eps, root = parts
    one = _constant(np.ones(len(p), dtype=np.float64))
    coeff_dl = sub(div(dl, root), one).rounded()
    coeff_y = sub(div(y, root), one).rounded()
    positive_dl = dl.rounded() > 0.0
    positive_y = y.rounded() > 0.0
    if np.any(positive_dl):
        dl_pos, y_pos, eps_pos, root_pos = (_take(v, positive_dl) for v in (dl, y, eps, root))
        den_dl = mul(root_pos, add(root_pos, dl_pos))
        coeff_dl[positive_dl] = neg(div(add(mul(y_pos, y_pos), mul(eps_pos, eps_pos)), den_dl)).rounded()
    if np.any(positive_y):
        dl_pos, y_pos, eps_pos, root_pos = (_take(v, positive_y) for v in (dl, y, eps, root))
        den_y = mul(root_pos, add(root_pos, y_pos))
        coeff_y[positive_y] = neg(div(add(mul(dl_pos, dl_pos), mul(eps_pos, eps_pos)), den_y)).rounded()
    row = np.zeros((len(p), 15), dtype=np.float64)
    row[:, 12] = coeff_dl
    for k in range(3):
        row[:, 6 + k] = -mul(_constant(coeff_y), flow[k]).rounded()
    for k in range(3, 6):
        row[:, 6 + k] = -mul(_constant(2.0 * coeff_y), flow[k]).rounded()
    h13 = div(_mul3(_constant(p[:, 17]), _constant(p[:, 19]),
                    _constant(p[:, 21])), _constant(p[:, 10]))
    h14 = div(_mul3(_constant(p[:, 18]), _constant(p[:, 20]),
                    _constant(p[:, 22])), _constant(p[:, 10]))
    row[:, 13] = mul(_constant(coeff_y), h13).rounded()
    row[:, 14] = mul(_constant(coeff_y), h14).rounded()
    j[:, 12, :] = row
    if not np.all(np.isfinite(j)):
        raise ArithmeticError('nonfinite Jacobian in DD reference')
    return j


def project(ref, p, f, old, maxiter=80):
    """Source166003 safeguarded Newton loop with compensated residual/Jacobian."""
    p = _real_array(p, 'parameters')
    f = _real_array(f, 'deformation')
    old = _real_array(old, 'accepted state')
    z = old.copy()
    z[:, 6:12] = np.einsum('nij,nj->ni', ref.base.stiffness(p),
                           ref.base.strain(f) - old[:, :6])
    counts = np.zeros(len(p), dtype=np.int32)
    scale = np.array([1e4] * 6 + [1.0] * 6 + [1e4] * 3, dtype=np.float64)
    for iteration in range(maxiter):
        r = ref.residual(p, f, old, z)
        norm = np.max(np.abs(r) * scale, axis=1)
        active = norm > 2e-10
        if not active.any():
            return z, counts, norm
        ids = np.flatnonzero(active)
        pa, fa, oa, za = p[ids], f[ids], old[ids], z[ids]
        try:
            j = ref.jacobian(pa, fa, oa, za)
            d = np.linalg.solve(j, -r[ids, :, None])[:, :, 0]
        except Exception:
            if len(ids) == 1:
                raise
            # Let source166003's resilient wrapper split the batch.
            raise RuntimeError('batched DD Newton linear solve failed')
        alpha = np.ones(len(ids), dtype=np.float64)
        pending = np.ones(len(ids), dtype=bool)
        trial = za.copy()
        for _ in range(30):
            candidate = za + alpha[:, None] * d
            trial_norm = np.max(np.abs(ref.residual(pa, fa, oa, candidate)) * scale, axis=1)
            accepted = pending & np.isfinite(trial_norm) & (trial_norm < norm[ids])
            trial[accepted] = candidate[accepted]
            pending[accepted] = False
            if not np.any(pending):
                break
            alpha[pending] *= 0.5
        if np.any(pending):
            raise RuntimeError(f'DD FP64 Newton Armijo stalled at iteration {iteration}; rows={ids[pending][:8].tolist()}')
        z[ids] = trial
        counts[ids] += 1
    raise RuntimeError('DD FP64 Newton iteration ceiling')


class Reference:
    def __init__(self, base=None):
        self.base = base or import_reference()
        self.residual = lambda p, f, old, z: residual_dd(self.base, p, f, old, z)
        self.jacobian = lambda p, f, old, z: jacobian_dd(self.base, p, f, old, z)
        self.project = lambda p, f, old, maxiter=80: project(self, p, f, old, maxiter=maxiter)

    def hill(self, p, s):
        q, flow = hill_dd(p, s)
        return q.rounded(), np.stack([x.rounded() for x in flow], axis=-1)

