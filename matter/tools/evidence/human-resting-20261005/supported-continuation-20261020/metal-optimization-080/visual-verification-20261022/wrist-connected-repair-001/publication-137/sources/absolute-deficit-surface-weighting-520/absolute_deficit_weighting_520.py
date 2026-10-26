"""Opt-in fixed-family weights that penalize absolute positive clearance deficits."""
from __future__ import annotations
import copy
import numpy as np


def reweight_surface_rows(result, *, total_weight: float):
    """Return a shallow result copy with only positive-row objective weights changed.

    For the normalized LP `w_i * slack_i` with `C_i d + h_i slack_i >= h_i`,
    setting `w_i` proportional to `h_i` makes the per-millimetre penalty
    `w_i / h_i` constant within this surface family. The family's total weight
    remains fixed. Row matrices, bounds, guards, and diagnostics are untouched.
    """
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("surface family weight must be finite and positive")
    h = np.asarray(result.get("h"), dtype=np.float64)
    old = np.asarray(result.get("weights"), dtype=np.float64)
    if h.ndim != 1 or old.shape != h.shape or not len(h):
        raise ValueError("positive surface h/weight rows must be nonempty matching vectors")
    if not np.isfinite(h).all() or np.any(h <= 0.0):
        raise ValueError("surface target distances must be positive and finite")
    if not np.isfinite(old).all() or np.any(old < 0.0):
        raise ValueError("baseline surface weights must be finite and nonnegative")
    if not np.isclose(float(old.sum()), total_weight, rtol=0.0, atol=1e-9):
        raise ValueError("baseline surface rows do not have the declared family mass")
    hsum = float(h.sum())
    if not np.isfinite(hsum) or hsum <= 0.0:
        raise ValueError("surface target-distance sum must be positive and finite")
    new = total_weight * h / hsum
    if not np.isclose(float(new.sum()), total_weight, rtol=0.0, atol=1e-12):
        raise ArithmeticError("reweighted surface rows changed their family mass")
    out = dict(result)
    out["weights"] = new
    metadata = result.get("metadata")
    if metadata is not None:
        if len(metadata) != len(new):
            raise ValueError("surface metadata row count differs from objective rows")
        out["metadata"] = [dict(item, normalized_weight=float(weight))
                           for item, weight in zip(metadata, new)]
    return out
