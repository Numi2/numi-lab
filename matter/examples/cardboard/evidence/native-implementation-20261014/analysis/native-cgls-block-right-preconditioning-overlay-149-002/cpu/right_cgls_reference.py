"""Small FP32 reference for right-preconditioned bounded CGLS.

This models the recurrence and exact row masks, not the production FEM
operator or its per-node tangent block construction.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

RUNNING, EXHAUSTED, ZERO_NORMAL, BREAKDOWN, INVALID = range(1, 6)

@dataclass
class Result:
    status: int
    iterations: int
    y: np.ndarray
    x: np.ndarray
    residual: np.ndarray
    normal_residual: np.ndarray
    history: list[dict[str, float]]
    error: str | None = None


def validate_tags_and_blocks(tags, masses, mask, blocks, selected):
    tags = np.asarray(tags, dtype=np.float32)
    masses = np.asarray(masses, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    blocks = np.asarray(blocks, dtype=np.float32)
    selected = np.asarray(selected, dtype=np.uint32)
    e_count, n_count = tags.shape
    if masses.shape != tags.shape or mask.shape != (e_count, n_count, 4):
        return False, "shape"
    if blocks.shape != (e_count, n_count, 3, 4) or selected.shape != (e_count,):
        return False, "shape"
    if np.any(selected > 1):
        return False, "selection"
    for e in range(e_count):
        if not selected[e]:
            continue
        for n in range(n_count):
            tag = float(tags[e, n])
            m = mask[e, n]
            rows = blocks[e, n]
            if tag not in (0.0, 1.0, 3.0):
                return False, "unsupported-constraint"
            if not np.isfinite(m).all() or m[3] != 0.0:
                return False, "mask"
            if tag == 0.0:
                if not np.isfinite(masses[e, n]) or masses[e, n] <= 0:
                    return False, "mass"
                if not np.array_equal(m[:3], np.ones(3, np.float32)):
                    return False, "free-mask"
                if not np.isfinite(rows).all() or rows[0, 3] != 1.0 or \
                        rows[1, 3] != 0.0 or rows[2, 3] != 0.0:
                    return False, "invalid-block"
            else:
                if np.any(m != 0.0) or np.any(rows != 0.0):
                    return False, "fixed-row"
    return True, None


def apply_right_scale(vector, mask, row_environment, selected, blocks,
                      node_count, transpose=False):
    """Apply S or S^T to full U=2*E*N float4 rows, fail-closed."""
    v = np.asarray(vector, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)
    row_environment = np.asarray(row_environment, dtype=np.uint32)
    selected = np.asarray(selected, dtype=np.uint32)
    blocks = np.asarray(blocks, dtype=np.float32)
    env_count = selected.size
    u_count = 2 * env_count * node_count
    if v.shape != (u_count, 4) or m.shape != v.shape or \
            row_environment.shape != (u_count,) or blocks.shape != (env_count, node_count, 3, 4):
        raise ValueError("bad full-vector layout")
    out = np.zeros_like(v)
    mechanical_rows = env_count * node_count
    for i in range(u_count):
        local = i if i < mechanical_rows else i - mechanical_rows
        env = local // node_count
        if env >= env_count or int(row_environment[i]) != env or selected[env] > 1:
            raise ValueError("bad environment map")
        if not np.isfinite(v[i]).all() or not np.isfinite(m[i]).all() or \
                v[i, 3] != 0 or m[i, 3] != 0:
            raise ValueError("nonfinite or nonzero w lane")
        if not selected[env] or i >= mechanical_rows or np.all(m[i, :3] == 0):
            if np.any(v[i] != 0):
                raise ValueError("nonzero quiet/fixed/field row")
            continue
        if np.any(m[i, :3] != 1):
            raise ValueError("nonbinary active mask")
        node = local - env * node_count
        block = blocks[env, node, :, :3]
        if transpose:
            out[i, :3] = np.asarray(block.T @ v[i, :3], dtype=np.float32)
        else:
            out[i, :3] = np.asarray(block @ v[i, :3], dtype=np.float32)
        if not np.isfinite(out[i]).all():
            raise ValueError("nonfinite scaled row")
    return out


def run_cgls_fp32(A, rhs, *, cap=32, scale_blocks=None,
                  use_transpose=True):
    """Run unregularized CGLS for A, optionally using right transform S.

    `scale_blocks` is the block diagonal S, not the source A_nn blocks.
    Directions are returned as physical x=S*y when scaling is enabled.
    """
    A = np.asarray(A, dtype=np.float32)
    b = np.asarray(rhs, dtype=np.float32)
    if cap not in (32, 64) or A.ndim != 2 or A.shape[0] != A.shape[1] or \
            b.shape != (A.shape[0],) or not np.isfinite(A).all() or not np.isfinite(b).all():
        return Result(INVALID, 0, np.zeros(0, np.float32), np.zeros(0, np.float32),
                      np.zeros(0, np.float32), np.zeros(0, np.float32), [], "invalid input")
    n = b.size
    if scale_blocks is None:
        S = np.eye(n, dtype=np.float32)
    else:
        S = np.asarray(scale_blocks, dtype=np.float32)
        if S.shape != A.shape or not np.isfinite(S).all():
            return Result(INVALID, 0, np.zeros(n, np.float32), np.zeros(n, np.float32),
                          b.copy(), np.zeros(n, np.float32), [], "invalid scale matrix")
    AT = A.T.copy()
    y = np.zeros(n, np.float32)
    r = b.copy()
    s0 = np.asarray(AT @ r, dtype=np.float32)
    s = np.asarray((S.T if use_transpose else S) @ s0, dtype=np.float32)
    p = s.copy()
    gamma = np.float32(s @ s)
    status = RUNNING if gamma > 0 and np.isfinite(gamma) else ZERO_NORMAL
    history = []
    iters = 0
    for _ in range(cap):
        if status != RUNNING:
            break
        physical_p = np.asarray(S @ p, dtype=np.float32)
        q = np.asarray(A @ physical_p, dtype=np.float32)
        denom = np.float32(q @ q)
        if not np.isfinite(denom) or denom <= 0 or gamma <= 0:
            status = BREAKDOWN
            break
        alpha = np.float32(gamma / denom)
        if not np.isfinite(alpha) or alpha <= 0:
            status = BREAKDOWN
            break
        y = np.asarray(y + alpha * p, dtype=np.float32)
        r = np.asarray(r - alpha * q, dtype=np.float32)
        s0 = np.asarray(AT @ r, dtype=np.float32)
        s = np.asarray((S.T if use_transpose else S) @ s0, dtype=np.float32)
        next_gamma = np.float32(s @ s)
        iters += 1
        history.append({"iteration": float(iters), "alpha": float(alpha),
                        "residual_norm": float(np.linalg.norm(r.astype(np.float64))),
                        "normal_norm": float(np.linalg.norm(s.astype(np.float64)))})
        if not np.isfinite(next_gamma) or next_gamma < 0:
            status = BREAKDOWN
            break
        if next_gamma == 0:
            gamma = next_gamma
            status = ZERO_NORMAL
            break
        beta = np.float32(next_gamma / gamma)
        if not np.isfinite(beta) or beta <= 0:
            status = BREAKDOWN
            break
        gamma = next_gamma
        if iters == cap:
            status = EXHAUSTED
            break
        p = np.asarray(s + beta * p, dtype=np.float32)
    if status == RUNNING:
        status = EXHAUSTED
    x = np.asarray(S @ y, dtype=np.float32)
    residual = np.asarray(b - A @ x, dtype=np.float32)
    normal = np.asarray(AT @ residual, dtype=np.float32)
    return Result(int(status), iters, y, x, residual, normal, history)
