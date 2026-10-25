"""183-compatible witness-local scaling with the existing edge-gradient bound.

This is proposal bookkeeping only. Every emitted trial still needs the full
exact geometry gates used by its caller.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np

_BASE_183 = Path(__file__).resolve().parents[1] / "local-witness-backtracking-183" / "final-a644" / "local_witness_backtracking.py"
_BASE_183_SHA256 = "a644de24e2f23e90807eb75578309618d89c1a3f9d95d4f6a6f045f4d935c31a"


def _load_183():
    digest = hashlib.sha256(_BASE_183.read_bytes()).hexdigest()
    if digest != _BASE_183_SHA256:
        raise RuntimeError(f"pinned 183 helper changed: {digest}")
    spec = importlib.util.spec_from_file_location("local_witness_backtracking_183_pinned", _BASE_183)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pinned 183 helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _integer_matrix(value, name: str, columns: int, *, allow_empty: bool):
    a = np.asarray(value)
    if a.ndim != 2 or a.shape[1] != columns:
        raise ValueError(f"{name} must have shape (N,{columns})")
    if a.dtype.kind not in "iu" or a.dtype.kind == "b":
        raise ValueError(f"{name} must contain integer indices")
    if not allow_empty and len(a) == 0:
        raise ValueError(f"{name} must not be empty")
    return a.astype(np.int64, copy=False)


def mesh_unique_edges(faces):
    """Return sorted, canonical unique undirected edges from triangular faces."""
    f = _integer_matrix(faces, "faces", 3, allow_empty=False)
    if np.any(f < 0) or np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])):
        raise ValueError("faces contain negative or repeated node indices")
    edge_rows = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]), axis=0)
    return np.unique(np.sort(edge_rows, axis=1), axis=0)


def _validate_scale_inputs(base_points_f32, proposed_displacements, step_scales, edges):
    base = np.asarray(base_points_f32)
    if base.dtype != np.dtype(np.float32) or base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("base_points_f32 must be a stored Float32 (N,3) array")
    if len(base) == 0 or not np.isfinite(base).all():
        raise ValueError("base_points_f32 must be nonempty and finite")
    delta = np.asarray(proposed_displacements)
    if delta.ndim != 2 or delta.shape != base.shape or delta.dtype.kind not in "fiu" or delta.dtype.kind == "b":
        raise ValueError("proposed_displacements must be a numeric (N,3) array")
    delta = delta.astype(np.float64, copy=False)
    if not np.isfinite(delta).all():
        raise ValueError("proposed_displacements must be finite")
    scales = np.asarray(step_scales)
    if scales.ndim != 1 or scales.shape != (len(base),) or scales.dtype.kind not in "fiu" or scales.dtype.kind == "b":
        raise ValueError("step_scales must be an (N,) numeric array")
    scales = scales.astype(np.float64, copy=True)
    if not np.isfinite(scales).all() or np.any(scales < 0.0) or np.any(scales > 1.0):
        raise ValueError("step_scales must be finite and within [0,1]")
    edge_rows = _integer_matrix(edges, "edges", 2, allow_empty=True)
    if len(edge_rows):
        if np.any(edge_rows < 0) or np.any(edge_rows >= len(base)) or np.any(edge_rows[:, 0] == edge_rows[:, 1]):
            raise ValueError("edges contain out-of-range or repeated node indices")
        edge_rows = np.unique(np.sort(edge_rows, axis=1), axis=0)
        lengths = np.linalg.norm(base[edge_rows[:, 0]].astype(np.float64) - base[edge_rows[:, 1]].astype(np.float64), axis=1)
        if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
            raise ValueError("reference edges must have finite positive Float32 lengths")
    else:
        lengths = np.empty(0, dtype=np.float64)
    return base, delta, scales, edge_rows, lengths


def _emit(base, delta, scales):
    # Match the connected backtracker: scale the raw proposal from the fixed base,
    # then emit once to Float32. The gate is evaluated on these emitted bytes.
    return (base.astype(np.float64) + scales[:, None] * delta).astype(np.float32)


def _edge_ratios(base, trial, edges, lengths):
    if len(edges) == 0:
        return np.empty(0, dtype=np.float64)
    emitted_delta = trial.astype(np.float64) - base.astype(np.float64)
    component_difference = np.abs(emitted_delta[edges[:, 0]] - emitted_delta[edges[:, 1]])
    return np.max(component_difference, axis=1) / lengths


def _single_edge_ratio(base, delta, scales, i, j, length):
    base_i = base[i].astype(np.float64)
    base_j = base[j].astype(np.float64)
    emitted_i = (base_i + scales[i] * delta[i]).astype(np.float32).astype(np.float64)
    emitted_j = (base_j + scales[j] * delta[j]).astype(np.float32).astype(np.float64)
    change_i = emitted_i - base_i
    change_j = emitted_j - base_j
    return float(np.max(np.abs(change_i - change_j)) / length)


def enforce_emitted_edge_gradient(
    base_points_f32,
    proposed_displacements,
    step_scales,
    edges,
    *,
    max_component_gradient=0.5,
    max_sweeps=64,
):
    """Couple endpoint scales until emitted-F32 displacement gradients pass.

    Enforces componentwise ``abs(d_i[k]-d_j[k]) <= cap * |base_i-base_j|``.
    A violating edge reduces both endpoint scales by the same factor, then a
    deterministic full recheck catches violations propagated to adjacent edges.
    If bounded sweeps do not pass, the candidate is ``None`` and must not be
    used.
    """
    base, delta, scales, edge_rows, lengths = _validate_scale_inputs(
        base_points_f32, proposed_displacements, step_scales, edges
    )
    cap = float(max_component_gradient)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError("max_component_gradient must be finite and positive")
    if isinstance(max_sweeps, (bool, np.bool_)) or int(max_sweeps) != max_sweeps or int(max_sweeps) < 1:
        raise ValueError("max_sweeps must be a positive integer")
    max_sweeps = int(max_sweeps)

    initial_trial = _emit(base, delta, scales)
    initial_ratios = _edge_ratios(base, initial_trial, edge_rows, lengths)
    initial_bad = np.flatnonzero(initial_ratios > cap)
    adjusted_nodes = set()
    sweep_count = 0
    edge_updates = 0

    for sweep in range(max_sweeps):
        trial = _emit(base, delta, scales)
        ratios = _edge_ratios(base, trial, edge_rows, lengths)
        bad = np.flatnonzero(ratios > cap)
        if len(bad) == 0:
            sweep_count = sweep
            break
        sweep_count = sweep + 1
        for edge_row in bad:
            i, j = map(int, edge_rows[int(edge_row)])
            # Earlier edges may already have reduced either endpoint.
            ratio_now = _single_edge_ratio(base, delta, scales, i, j, float(lengths[edge_row]))
            if ratio_now <= cap:
                continue
            bound = cap * float(lengths[edge_row])
            continuous_peak = float(np.max(np.abs(scales[i] * delta[i] - scales[j] * delta[j])))
            if continuous_peak > bound:
                factor = (bound / continuous_peak) * (1.0 - 1e-10)
            else:
                # Only F32 rounding violates the bound; halving both scales is
                # deterministic and eventually emits the unchanged base edge.
                factor = 0.5
            factor = min(1.0 - 1e-12, max(0.0, factor))
            scales[i] *= factor
            scales[j] *= factor
            adjusted_nodes.add(i)
            adjusted_nodes.add(j)
            edge_updates += 1
        trial = _emit(base, delta, scales)
        if not len(np.flatnonzero(_edge_ratios(base, trial, edge_rows, lengths) > cap)):
            break

    trial = _emit(base, delta, scales)
    ratios = _edge_ratios(base, trial, edge_rows, lengths)
    unresolved = np.flatnonzero(ratios > cap)
    passed = len(unresolved) == 0
    report = {
        "status": "EMITTED_F32_EDGE_GRADIENT_PASS" if passed else "EDGE_GRADIENT_UNRESOLVED_FAIL_CLOSED",
        "construction_only": True,
        "geometry_certified": False,
        "accepted_physical_steps": 0,
        "max_component_gradient": cap,
        "semantics": "componentwise absolute displacement difference per coordinate divided by the Float32 base edge length",
        "reference_edge_domain": "base_points_f32 supplied to this scale update",
        "edge_count": int(len(edge_rows)),
        "initial_violating_edge_count": int(len(initial_bad)),
        "maximum_initial_gradient_ratio": float(initial_ratios.max(initial=0.0)),
        "maximum_emitted_gradient_ratio": float(ratios.max(initial=0.0)),
        "maximum_emitted_gradient_excess": float(max(0.0, ratios.max(initial=0.0) - cap)),
        "sweeps": int(sweep_count),
        "edge_scale_updates": int(edge_updates),
        "adjusted_node_rows": sorted(adjusted_nodes),
        "unresolved_edge_rows": unresolved[:64].tolist(),
        "unresolved_edge_count": int(len(unresolved)),
        "scales_monotone_nonincreasing": bool(np.all(scales <= np.asarray(step_scales, dtype=np.float64))),
        "edge_constraints_checked_after_float32_emission": True,
        "candidate_usable": bool(passed),
    }
    return scales, (trial if passed else None), report


def reduce_witness_node_scales_with_edge_gradient(
    faces,
    step_scales,
    bad_face_rows,
    self_pair_face_rows,
    *,
    base_points_f32,
    proposed_displacements,
    anchors=(),
    zero_below=1.0 / 128.0,
    max_component_gradient=0.5,
    max_edge_sweeps=64,
):
    """Apply frozen 183 witness reduction, then restore the edge-gradient cap.

    Returns ``(scales, emitted_trial_or_None, report)``. Exact collision, point,
    area, source-cap and ownership gates remain the caller's responsibility.
    """
    base183 = _load_183()
    f = np.asarray(faces)
    if f.ndim != 2 or f.shape[1] != 3 or f.dtype.kind not in "iu" or f.dtype.kind == "b":
        raise ValueError("faces must be an integer (F,3) array")
    f = f.astype(np.int64, copy=False)
    witness_scales, witness_report = base183.reduce_witness_node_scales(
        f, step_scales, bad_face_rows, self_pair_face_rows,
        anchors=anchors, zero_below=zero_below,
    )
    adjusted, emitted, gradient_report = enforce_emitted_edge_gradient(
        base_points_f32, proposed_displacements, witness_scales, mesh_unique_edges(f),
        max_component_gradient=max_component_gradient,
        max_sweeps=max_edge_sweeps,
    )
    report = {
        "status": gradient_report["status"],
        "construction_only": True,
        "geometry_certified": False,
        "accepted_physical_steps": 0,
        "pinned_183_helper_path": str(_BASE_183),
        "pinned_183_helper_sha256": _BASE_183_SHA256,
        "witness_scale_update": witness_report,
        "edge_gradient_update": gradient_report,
        "candidate_usable": bool(gradient_report["candidate_usable"]),
    }
    return adjusted, emitted, report
