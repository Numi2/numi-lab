"""Witness-local proposal step scaling for connected surface backtracking.

This helper changes only a per-node proposal multiplier. Exact geometry gates
remain the caller's responsibility after every trial.
"""
from __future__ import annotations
import math
import numpy as np


def _index_array(value, name, ndim):
    a = np.asarray(value)
    if a.size == 0 and a.ndim == ndim:
        return np.empty(a.shape, dtype=np.int64)
    if a.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}")
    if a.dtype.kind not in "iu" or a.dtype.kind == "b":
        raise ValueError(f"{name} must contain integer indices")
    return a.astype(np.int64, copy=False)


def reduce_witnessed_step_scales(
    faces,
    step_scales,
    bad_face_rows,
    self_pair_face_rows=None,
    anchor_rows=(),
    halving_counts=None,
    *,
    factor=0.5,
    max_halvings=6,
):
    """Reduce proposal scales only on vertices touched by exact-gate witnesses.

    ``step_scales[i]`` multiplies the caller's proposed displacement for node i.
    Bad face rows and both faces of each exact self-pair mark affected vertices.
    A marked node is halved at each retry; after ``max_halvings`` halvings, a
    subsequent witness sets only that node's scale to exactly zero. Anchor rows
    are always set to zero. No neighboring-node smoothing is applied.

    Returns ``(scales, halving_counts, report)``. This does not certify geometry.
    """
    f = _index_array(faces, "faces", 2)
    if f.shape[1:] != (3,):
        raise ValueError("faces must have shape (M,3)")
    s = np.asarray(step_scales)
    if s.ndim != 1 or s.dtype.kind not in "fiu" or s.dtype.kind == "b":
        raise ValueError("step_scales must be a numeric vector")
    s = s.astype(np.float64, copy=False)
    if len(s) == 0 or not np.all(np.isfinite(s)) or np.any(s < 0.0) or np.any(s > 1.0):
        raise ValueError("step_scales must be nonempty, finite, and within [0,1]")
    if f.size and (int(f.min()) < 0 or int(f.max()) >= len(s)):
        raise ValueError("face node index out of range")
    if len(f) == 0 or any(len(set(map(int, row))) != 3 for row in f):
        raise ValueError("faces must be nonempty with three distinct vertices each")

    bad = _index_array(bad_face_rows, "bad_face_rows", 1)
    if len(set(map(int, bad))) != len(bad):
        raise ValueError("bad_face_rows must be unique")
    if np.any(bad < 0) or np.any(bad >= len(f)):
        raise ValueError("bad face row out of range")

    if self_pair_face_rows is None:
        pairs = np.empty((0, 2), dtype=np.int64)
    else:
        if np.asarray(self_pair_face_rows).size == 0:
            pairs = np.empty((0, 2), dtype=np.int64)
        else:
            pairs = _index_array(self_pair_face_rows, "self_pair_face_rows", 2)
        if pairs.shape[1:] != (2,):
            raise ValueError("self_pair_face_rows must have shape (K,2)")
        if len(pairs):
            if np.any(pairs < 0) or np.any(pairs >= len(f)) or np.any(pairs[:, 0] == pairs[:, 1]):
                raise ValueError("invalid self-pair face rows")
            canonical = [tuple(sorted(map(int, row))) for row in pairs]
            if len(set(canonical)) != len(canonical):
                raise ValueError("self_pair_face_rows contain duplicate pairs")

    if len(bad) == 0 and len(pairs) == 0:
        raise ValueError("at least one bad face or self-pair witness is required")

    anchors = np.asarray(anchor_rows)
    if anchors.size == 0:
        anchors = np.empty(0, dtype=np.int64)
    if anchors.ndim != 1 or anchors.dtype.kind not in "iu" or anchors.dtype.kind == "b":
        raise ValueError("anchor_rows must be a 1D integer index vector")
    anchors = anchors.astype(np.int64, copy=False)
    if len(set(map(int, anchors))) != len(anchors):
        raise ValueError("anchor_rows contain duplicates")
    if np.any(anchors < 0) or np.any(anchors >= len(s)):
        raise ValueError("anchor row out of range")

    if not math.isfinite(float(factor)) or not (0.0 < float(factor) < 1.0):
        raise ValueError("factor must be finite and in (0,1)")
    if isinstance(max_halvings, (bool, np.bool_)) or int(max_halvings) != max_halvings or int(max_halvings) < 1:
        raise ValueError("max_halvings must be a positive integer")
    max_halvings = int(max_halvings)

    affected_faces = np.unique(np.r_[bad, pairs.ravel()]).astype(np.int64)
    affected_nodes = np.unique(f[affected_faces].ravel()).astype(np.int64)
    if halving_counts is None:
        counts = np.zeros(len(s), dtype=np.int64)
    else:
        counts_in = np.asarray(halving_counts)
        if counts_in.shape != s.shape or counts_in.dtype.kind not in "iu" or counts_in.dtype.kind == "b":
            raise ValueError("halving_counts must be an integer vector matching step_scales")
        counts = counts_in.astype(np.int64, copy=True)
        if np.any(counts < 0):
            raise ValueError("halving_counts cannot be negative")

    out = s.copy()
    zero_fallback = []
    halved = []
    for node in affected_nodes:
        i = int(node)
        if counts[i] >= max_halvings:
            out[i] = 0.0
            zero_fallback.append(i)
        elif out[i] != 0.0:
            out[i] *= float(factor)
            counts[i] += 1
            halved.append(i)
    out[anchors] = 0.0
    untouched = np.ones(len(s), dtype=bool)
    untouched[affected_nodes] = False
    untouched[anchors] = False
    if not np.array_equal(out[untouched], s[untouched]):
        raise AssertionError("unaffected scales changed")
    if np.any(out[anchors] != 0.0):
        raise AssertionError("anchors are not fixed")

    report = {
        "status": "PROPOSAL_SCALES_REDUCED_ONLY",
        "construction_only": True,
        "geometry_certified": False,
        "bad_face_rows": bad.tolist(),
        "self_pair_face_rows": pairs.tolist(),
        "affected_faces": affected_faces.tolist(),
        "affected_node_rows": affected_nodes.tolist(),
        "halved_node_rows": halved,
        "exact_zero_fallback_node_rows": zero_fallback,
        "anchor_rows_forced_zero": anchors.tolist(),
        "unaffected_scales_bitwise_unchanged": bool(np.array_equal(out[untouched], s[untouched])),
        "max_halvings": max_halvings,
        "factor": float(factor),
    }
    return out, counts, report


def reduce_witness_node_scales(faces, step_scales, bad_face_rows, self_pair_face_rows, *, anchors=(), zero_below=1.0 / 128.0):
    """Runner-compatible one-round wrapper with a bounded exact-zero fallback.

    It halves only witness-incident nodes. If a halved multiplier is at or below
    ``zero_below``, that node's proposal multiplier is set to exactly zero.
    This is still only proposal bookkeeping; callers must rerun every exact gate.
    """
    threshold = float(zero_below)
    if not math.isfinite(threshold) or threshold < 0.0 or threshold >= 1.0:
        raise ValueError("zero_below must be finite and in [0,1)")
    out, counts, report = reduce_witnessed_step_scales(
        faces, step_scales, bad_face_rows, self_pair_face_rows,
        anchor_rows=anchors, max_halvings=32,
    )
    affected = np.asarray(report["affected_node_rows"], dtype=np.int64)
    fallback = affected[(out[affected] > 0.0) & (out[affected] <= threshold)]
    out[fallback] = 0.0
    report["threshold_zero_fallback_node_rows"] = fallback.tolist()
    report["zero_below"] = threshold
    report["halving_counts"] = counts.tolist()
    report["geometry_certified"] = False
    return out, report
