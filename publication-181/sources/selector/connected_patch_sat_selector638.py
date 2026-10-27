"""Experimental per-patch finite-triangle SAT feasibility selector.

This is a box-only construction diagnostic. It does not check triangle self
intersections, containment against all banks, fixed-boundary constraints,
area, edge gradients, or any physical acceptance gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class AxisOption:
    pair_index: int
    normal: np.ndarray
    rhs_mm: np.ndarray
    source: str
    axis_index: int
    axis_sign: int
    source_order: int
    sign_order: int
    extra_sign: int
    max_unmet_mm: float = 0.0


def partition_pair_components(pair_nodes: np.ndarray) -> list[np.ndarray]:
    """Return deterministic pair-index components under exact shared nodes."""
    rows = np.asarray(pair_nodes, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise ValueError("pair_nodes must have shape (P,3)")
    if np.any(rows < 0) or np.any(np.diff(np.sort(rows, axis=1), axis=1) == 0):
        raise ValueError("pair rows need three distinct nonnegative nodes")
    count = len(rows)
    parent = np.arange(count, dtype=np.int64)
    rank = np.zeros(count, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    owner: dict[int, int] = {}
    for i, row in enumerate(rows):
        for node in row:
            node = int(node)
            old = owner.get(node)
            if old is None:
                owner[node] = i
            else:
                union(i, old)
    groups: dict[int, list[int]] = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)
    result = [np.asarray(v, dtype=np.int64) for v in groups.values()]
    result.sort(key=lambda v: (int(v.min()), len(v), tuple(v.tolist())))
    return result


def _validate_bounds(
    node_rows: np.ndarray, lower_by_node: dict[int, np.ndarray], upper_by_node: dict[int, np.ndarray],
    anchors: Iterable[int] = (), tol: float = 1e-10,
) -> None:
    anchor_set = {int(x) for x in anchors}
    for raw in np.unique(np.asarray(node_rows, dtype=np.int64)):
        node = int(raw)
        if node not in lower_by_node or node not in upper_by_node:
            raise ValueError(f"missing directional box for node {node}")
        lo = np.asarray(lower_by_node[node], dtype=np.float64)
        hi = np.asarray(upper_by_node[node], dtype=np.float64)
        if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite(lo).all() or not np.isfinite(hi).all():
            raise ValueError(f"invalid box for node {node}")
        if np.any(lo > hi + tol) or np.any(lo > tol) or np.any(hi < -tol):
            raise ValueError(f"box must be ordered and contain zero for node {node}")
        if node in anchor_set and (np.any(np.abs(lo) > tol) or np.any(np.abs(hi) > tol)):
            raise ValueError(f"anchor node {node} is not fixed at zero")


def _row_big_m(normal: np.ndarray, rhs: float, lo: np.ndarray, hi: np.ndarray,
               relaxation_pad_mm: float = 1e-8) -> tuple[float, float]:
    """Return finite row-specific M and the exact box minimum of n·d."""
    n = np.asarray(normal, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if n.shape != (3,) or lo.shape != (3,) or hi.shape != (3,):
        raise ValueError("normal and box vectors must have length three")
    if not np.isfinite(n).all() or not np.isfinite([rhs, relaxation_pad_mm]).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("non-finite row data")
    if relaxation_pad_mm < 0:
        raise ValueError("relaxation pad must be nonnegative")
    if np.any(lo > hi):
        raise ValueError("inverted box")
    min_dot = float(np.sum(np.where(n >= 0.0, n * lo, n * hi)))
    required = max(0.0, float(rhs) - min_dot)
    return required + float(relaxation_pad_mm), min_dot


def build_component_milp(
    pair_indices: np.ndarray,
    pair_nodes: np.ndarray,
    identities: np.ndarray,
    axis_options: list[list[AxisOption]],
    lower_by_node: dict[int, np.ndarray],
    upper_by_node: dict[int, np.ndarray],
    anchors: Iterable[int] = (),
    relaxation_pad_mm: float = 1e-8,
):
    """Build one exact pair-node-incidence component MILP.

    Variables are shared node displacements (mm) and one binary per feasible
    signed SAT axis. Selected pair-axis binaries enforce all three corner rows.
    """
    pidx = np.asarray(pair_indices, dtype=np.int64)
    nodes_all = np.asarray(pair_nodes, dtype=np.int64)
    ids = np.asarray(identities, dtype=np.int64)
    if pidx.ndim != 1 or np.any(pidx < 0) or np.any(pidx >= len(nodes_all)):
        raise ValueError("invalid component pair indices")
    if len(ids) != len(nodes_all) or len(axis_options) != len(nodes_all):
        raise ValueError("pair inventory length mismatch")
    local_pairs = pidx.tolist()
    node_rows = nodes_all[pidx]
    _validate_bounds(node_rows, lower_by_node, upper_by_node, anchors)
    nodes = np.unique(node_rows.reshape(-1))
    node_pos = {int(node): i for i, node in enumerate(nodes)}
    axis_vars: list[tuple[int, AxisOption]] = []
    for pi in local_pairs:
        for op in axis_options[pi]:
            if op.pair_index != pi:
                raise ValueError("axis option belongs to another pair")
            if op.normal.shape != (3,) or op.rhs_mm.shape != (3,) or not np.isfinite(op.normal).all() or not np.isfinite(op.rhs_mm).all():
                raise ValueError("invalid axis option")
            if not np.isclose(np.linalg.norm(op.normal), 1.0, atol=2e-8):
                raise ValueError("SAT normal must be unit length")
            axis_vars.append((pi, op))
    axis_vars.sort(key=lambda z: (
        tuple(ids[z[0]].tolist()), z[1].source_order, z[1].axis_index,
        z[1].sign_order, z[1].extra_sign, tuple(np.round(z[1].normal, 14).tolist())
    ))
    ncoord = 3 * len(nodes)
    nbin = len(axis_vars)
    nvar = ncoord + nbin
    axis_col = {(pi, id(op)): ncoord + j for j, (pi, op) in enumerate(axis_vars)}
    lo_var = np.empty(nvar, dtype=np.float64)
    hi_var = np.empty(nvar, dtype=np.float64)
    for node, j in node_pos.items():
        lo_var[3*j:3*j+3] = lower_by_node[node]
        hi_var[3*j:3*j+3] = upper_by_node[node]
    if nbin:
        lo_var[ncoord:] = 0.0
        hi_var[ncoord:] = 1.0
    c = np.zeros(nvar, dtype=np.float64)
    # Primary objective is pair count. The total deterministic axis tie-break
    # is bounded below 0.2, so it cannot trade away one selected pair.
    if nbin:
        tie_den = max(1, nbin + 1)
        pair_den = max(1, len(local_pairs))
        for j in range(nbin):
            c[ncoord+j] = -1.0 + 0.2 * (j + 1) / tie_den / pair_den

    rr: list[int] = []
    cc: list[int] = []
    vv: list[float] = []
    lb_rows: list[float] = []
    ub_rows: list[float] = []
    row = 0
    local_pair_set = set(local_pairs)
    # At most one axis can be selected for a pair.
    for pi in local_pairs:
        cols = [axis_col[(ppi, id(op))] for ppi, op in axis_vars if ppi == pi]
        if cols:
            for col in cols:
                rr.append(row); cc.append(col); vv.append(1.0)
            lb_rows.append(-np.inf); ub_rows.append(1.0); row += 1
    big_m_receipts = []
    for pi, op in axis_vars:
        col_y = axis_col[(pi, id(op))]
        for corner, node_raw in enumerate(nodes_all[pi]):
            node = int(node_raw); j = node_pos[node]
            lo = np.asarray(lower_by_node[node], dtype=np.float64)
            hi = np.asarray(upper_by_node[node], dtype=np.float64)
            n = np.asarray(op.normal, dtype=np.float64)
            rhs = float(op.rhs_mm[corner])
            big_m, min_dot = _row_big_m(n, rhs, lo, hi, relaxation_pad_mm)
            # -n·d + M*y <= M-rhs; when y=0 this is box-redundant.
            for k in range(3):
                if n[k] != 0.0:
                    rr.append(row); cc.append(3*j+k); vv.append(-float(n[k]))
            if big_m != 0.0:
                rr.append(row); cc.append(col_y); vv.append(big_m)
            lb_rows.append(-np.inf); ub_rows.append(big_m-rhs); row += 1
            big_m_receipts.append({
                "pair_identity": ids[pi].tolist(), "corner": corner, "node": node,
                "axis_source": op.source, "axis_index": op.axis_index,
                "axis_sign": op.axis_sign, "normal": n.tolist(),
                "rhs_mm": rhs, "box_min_dot_mm": min_dot, "big_m_mm": big_m,
            })
    matrix = coo_matrix((np.asarray(vv, dtype=np.float64),
                         (np.asarray(rr, dtype=np.int64), np.asarray(cc, dtype=np.int64))),
                        shape=(row, nvar)).tocsc()
    if matrix.shape != (len(lb_rows), nvar):
        raise AssertionError("constraint matrix row accounting mismatch")
    return {
        "c": c,
        "integrality": np.r_[np.zeros(ncoord, dtype=np.int8), np.ones(nbin, dtype=np.int8)],
        "bounds": Bounds(lo_var, hi_var),
        "constraints": LinearConstraint(matrix, np.asarray(lb_rows), np.asarray(ub_rows)),
        "nodes": nodes,
        "node_pos": node_pos,
        "pair_indices": pidx,
        "pair_nodes": nodes_all,
        "axis_vars": axis_vars,
        "axis_col": axis_col,
        "coordinate_variable_count": ncoord,
        "binary_variable_count": nbin,
        "constraint_count": row,
        "big_m_receipts": big_m_receipts,
        "matrix": matrix,
        "identities": ids,
    }


def solve_component(model: dict[str, Any], time_limit_s: float, node_limit: int):
    seconds = float(time_limit_s)
    nodes_cap = int(node_limit)
    if not np.isfinite(seconds) or seconds <= 0 or seconds > 5.0:
        raise ValueError("component time limit must be in (0,5] seconds")
    if nodes_cap <= 0 or nodes_cap > 25000:
        raise ValueError("node limit must be in [1,25000]")
    if model["binary_variable_count"] == 0:
        return {"has_incumbent": True, "selected": [],
                "node_displacements_mm": np.zeros((len(model["nodes"]),3)),
                "status": 0, "message": "No box-feasible SAT axes in this component",
                "elapsed_s": 0.0, "objective": 0.0, "mip_gap": 0.0, "mip_node_count": 0}
    start = time.monotonic()
    result = milp(
        c=model["c"],
        integrality=model["integrality"],
        bounds=model["bounds"],
        constraints=model["constraints"],
        options={"time_limit": seconds, "node_limit": nodes_cap, "mip_rel_gap": 0.0, "presolve": True},
    )
    elapsed = time.monotonic() - start
    x = getattr(result, "x", None)
    if x is None or not np.isfinite(x).all():
        return {"has_incumbent": False, "selected": [], "node_displacements_mm": np.zeros((len(model["nodes"]),3)),
                "status": int(result.status), "message": str(result.message), "elapsed_s": elapsed,
                "mip_gap": getattr(result, "mip_gap", None), "mip_node_count": getattr(result, "mip_node_count", None)}
    tol = 1e-6
    if np.any(x < model["bounds"].lb-tol) or np.any(x > model["bounds"].ub+tol):
        raise RuntimeError("solver incumbent violates variable bounds")
    d = x[:model["coordinate_variable_count"]].reshape(-1,3)
    selected_by_pair: dict[int, list[tuple[AxisOption,int]]] = {}
    for pi, op in model["axis_vars"]:
        col = model["axis_col"][(pi,id(op))]
        val = float(x[col])
        if abs(val-round(val)) > 1e-5:
            raise RuntimeError("solver incumbent has fractional axis indicator")
        if val > 0.5:
            selected_by_pair.setdefault(pi, []).append((op,col))
    if any(len(v) != 1 for v in selected_by_pair.values()):
        raise RuntimeError("solver incumbent selected multiple axes for one pair")
    selected = []
    for pi in model["pair_indices"]:
        pi = int(pi)
        choices = selected_by_pair.get(pi, [])
        if not choices:
            continue
        op, col = choices[0]
        # Check all three rows against the shared displacement vector.
        row_nodes = np.asarray(model["pair_nodes"][pi], dtype=np.int64)
        for corner,node in enumerate(row_nodes):
            lhs = float(np.dot(op.normal, d[model["node_pos"][int(node)]]))
            if lhs < float(op.rhs_mm[corner]) - tol:
                raise RuntimeError("solver incumbent does not satisfy selected full-corner SAT rows")
        selected.append({"pair_index":pi,"axis":op,"node_rows":row_nodes.copy()})
    return {
        "has_incumbent": True,
        "selected": selected,
        "node_displacements_mm": d,
        "status": int(result.status),
        "message": str(result.message),
        "elapsed_s": elapsed,
        "objective": float(result.fun),
        "mip_gap": getattr(result, "mip_gap", None),
        "mip_node_count": getattr(result, "mip_node_count", None),
    }
