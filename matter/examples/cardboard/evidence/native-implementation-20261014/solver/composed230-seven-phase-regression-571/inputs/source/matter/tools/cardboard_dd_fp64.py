"""Small binary64 double-double primitives used by the isolated reference.

Each value is an unevaluated pair of ordinary IEEE-754 binary64 numbers.
There is no wider floating type or arbitrary-precision arithmetic in this
module. Unsupported nonfinite, complex, overflow, and underflow cases fail
closed instead of silently changing the claimed arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

_SPLITTER = np.float64(134217729.0)  # 2**27 + 1, for a 53-bit significand.


def _as_f64(x):
    a = np.asarray(x)
    if np.iscomplexobj(a):
        raise TypeError("complex inputs are unsupported by the FP64 DD path")
    a = np.asarray(a, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError("nonfinite input to FP64 DD arithmetic")
    return a


@dataclass(frozen=True)
class DD:
    hi: np.ndarray
    lo: np.ndarray

    @classmethod
    def from_value(cls, value):
        a = _as_f64(value)
        return cls(a, np.zeros_like(a))

    def rounded(self):
        out = self.hi + self.lo
        if not np.all(np.isfinite(out)):
            raise ArithmeticError("nonfinite rounded FP64 DD result")
        return out


def _two_sum(a, b):
    s = a + b
    bp = s - a
    e = (a - (s - bp)) + (b - bp)
    return s, e


def _normalise(a, b):
    h, l = _two_sum(a, b)
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(l)):
        raise ArithmeticError("FP64 DD addition overflow")
    return DD(h, l)


def add(a: DD, b: DD) -> DD:
    s, e = _two_sum(a.hi, b.hi)
    t, f = _two_sum(a.lo, b.lo)
    e = e + t
    h, l = _two_sum(s, e)
    l = l + f
    return _normalise(h, l)


def neg(a: DD) -> DD:
    return DD(-a.hi, -a.lo)


def sub(a: DD, b: DD) -> DD:
    return add(a, neg(b))


def _two_product(a, b):
    """Return p,e with p+e the exact product when it is representable.

    Splitting normalized significands avoids overflow in splitter*x for large
    but otherwise ordinary operands. A product outside the normal binary64
    range is rejected because an unevaluated pair cannot recover its lost
    underflow bits on this path.
    """
    ma, ea = np.frexp(a)
    mb, eb = np.frexp(b)
    k = ea + eb
    ca = _SPLITTER * ma
    cb = _SPLITTER * mb
    ah = ca - (ca - ma)
    bh = cb - (cb - mb)
    al = ma - ah
    bl = mb - bh
    pm = ma * mb
    em = ((ah * bh - pm) + ah * bl + al * bh) + al * bl
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        p = np.ldexp(pm, k)
        e = np.ldexp(em, k)
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(e)):
        raise ArithmeticError("FP64 DD product overflow")
    nonzero = (a != 0.0) & (b != 0.0)
    if np.any(nonzero & (p == 0.0)):
        raise ArithmeticError("FP64 DD product underflow")
    if np.any(nonzero & (np.abs(p) < np.finfo(np.float64).tiny)):
        raise ArithmeticError("subnormal FP64 DD product is unsupported")
    return p, e


def mul(a: DD, b: DD) -> DD:
    p, e = _two_product(a.hi, b.hi)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        cross = a.hi * b.lo + a.lo * b.hi
        low_low = a.lo * b.lo
        e = e + cross + low_low
    if np.any(~np.isfinite(e)):
        raise ArithmeticError("FP64 DD product correction overflow")
    return _normalise(p, e)


def div(a: DD, b: DD) -> DD:
    if np.any(b.hi == 0.0):
        raise ZeroDivisionError("zero denominator in FP64 DD division")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        q = DD(a.hi / b.hi, np.zeros_like(a.hi))
    if not np.all(np.isfinite(q.hi)):
        raise ArithmeticError("nonfinite initial FP64 DD quotient")
    # Two residual corrections give roughly 106 useful significand bits.
    for _ in range(2):
        rem = sub(a, mul(b, q))
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            qi = (rem.hi + rem.lo) / b.hi
        if not np.all(np.isfinite(qi)):
            raise ArithmeticError("nonfinite FP64 DD quotient correction")
        q = add(q, DD(qi, np.zeros_like(qi)))
    return q


def sqrt(a: DD) -> DD:
    if np.any(a.hi < 0.0) or np.any((a.hi == 0.0) & (a.lo < 0.0)):
        raise ValueError("negative radicand in FP64 DD square root")
    if np.any(a.hi == 0.0):
        if np.any(a.lo != 0.0):
            raise ArithmeticError("unnormalized near-zero FP64 DD radicand")
        return DD(np.zeros_like(a.hi), np.zeros_like(a.lo))
    x = DD(np.sqrt(a.hi), np.zeros_like(a.hi))
    if not np.all(np.isfinite(x.hi)) or np.any(x.hi == 0.0):
        raise ArithmeticError("unsupported FP64 DD square-root scale")
    for _ in range(2):
        rem = sub(a, mul(x, x))
        correction = div(rem, DD(2.0 * x.hi, np.zeros_like(x.hi)))
        x = add(x, correction)
    if np.any(x.hi < 0.0) or not np.all(np.isfinite(x.hi + x.lo)):
        raise ArithmeticError("invalid FP64 DD square-root result")
    return x


def scaled_norm_error(value: DD, reference: float) -> float:
    """Return the absolute rounded error against a supplied binary64 value."""
    return float(np.max(np.abs(value.rounded() - np.float64(reference))))
