"""Exact-anchor harmonic displacement prolongation on a fixed triangle mesh.

This is a construction diagnostic. It does not certify collision, material,
biomechanical, or native-runtime acceptance.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.linalg import lstsq


def _validate_surface(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all():
        raise ValueError("vertices must be finite Nx3")
    if f.ndim != 2 or f.shape[1] != 3 or f.size == 0:
        raise ValueError("faces must be nonempty Mx3")
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError("face index outside vertex array")
    if np.any((f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])):
        raise ValueError("face has repeated vertex index")
    return v, f


def _dual_adjacency(faces):
    m = len(faces)
    edge_rows = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edge_rows.sort(axis=1)
    face_ids = np.tile(np.arange(m, dtype=np.int64), 3)
    order = np.lexsort((face_ids, edge_rows[:, 1], edge_rows[:, 0]))
    edges = edge_rows[order]
    owners = face_ids[order]
    starts = np.r_[0, np.flatnonzero(np.any(edges[1:] != edges[:-1], axis=1)) + 1]
    ends = np.r_[starts[1:], len(edges)]
    rr, cc = [], []
    for a, b in zip(starts, ends):
        own = owners[a:b]
        if len(own) == 2:
            rr.extend((int(own[0]), int(own[1])))
            cc.extend((int(own[1]), int(own[0])))
        elif len(own) > 2:
            raise ValueError("nonmanifold edge in input surface")
    return sparse.csr_matrix((np.ones(len(rr), dtype=np.uint8), (rr, cc)), shape=(m, m))


def _expand(mask, adjacency, rings):
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(rings)):
        frontier = np.asarray(adjacency @ out.astype(np.uint8)).ravel() > 0
        out |= frontier
    return out


def _boundary_vertices(faces, selected):
    sf = faces[selected]
    if len(sf) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    e = np.concatenate((sf[:, [0, 1]], sf[:, [1, 2]], sf[:, [2, 0]]))
    e.sort(axis=1)
    ue, counts = np.unique(e, axis=0, return_counts=True)
    bedges = ue[counts == 1]
    if len(bedges) == 0:
        return np.empty(0, dtype=np.int64), ue
    return np.unique(bedges), ue


def prolongate_surface_displacement(
    vertices_m,
    faces,
    direct_vertex_ids,
    direct_displacements_m,
    bary_face_ids,
    bary_weights,
    bary_displacements_m,
    *,
    core_rings=2,
    collar_rings=1,
    maximum_expansion_rings=8,
    anchor_tolerance_m=2e-8,
):
    """Prolong coarse point deltas while fixing exact anchors and an unchanged collar.

    The minimizer is inverse-edge-length graph-harmonic in the free patch nodes.
    Direct source-vertex rows are hard Dirichlet constraints. Barycentric rows
    are exact linear constraints on their source triangles. The support is the
    union of affected anchor stars expanded by core_rings and collar_rings.
    """
    v, f = _validate_surface(vertices_m, faces)
    did = np.asarray(direct_vertex_ids, dtype=np.int64)
    dd = np.asarray(direct_displacements_m, dtype=np.float64)
    bfi = np.asarray(bary_face_ids, dtype=np.int64)
    bw = np.asarray(bary_weights, dtype=np.float64)
    bd = np.asarray(bary_displacements_m, dtype=np.float64)
    if did.ndim != 1 or dd.shape != (len(did), 3):
        raise ValueError("direct anchor shape mismatch")
    if bfi.ndim != 1 or bw.shape != (len(bfi), 3) or bd.shape != (len(bfi), 3):
        raise ValueError("barycentric anchor shape mismatch")
    if not np.isfinite(dd).all() or not np.isfinite(bw).all() or not np.isfinite(bd).all():
        raise ValueError("nonfinite anchor value")
    if len(did) and (did.min() < 0 or did.max() >= len(v)):
        raise ValueError("direct anchor index outside surface")
    if len(bfi) and (bfi.min() < 0 or bfi.max() >= len(f)):
        raise ValueError("barycentric face outside surface")
    if len(bfi) and (np.any(bw < -1e-10) or np.any(bw > 1 + 1e-10) or not np.allclose(bw.sum(axis=1), 1.0, atol=1e-10, rtol=0)):
        raise ValueError("invalid barycentric weights")
    if len(np.unique(did)) != len(did):
        order = np.argsort(did)
        sd, sx = did[order], dd[order]
        unique, first, counts = np.unique(sd, return_index=True, return_counts=True)
        for i, n in zip(first, counts):
            if n > 1 and not np.allclose(sx[i:i+n], sx[i], atol=1e-12, rtol=0):
                raise ValueError("conflicting duplicate direct anchors")
        keep = np.r_[True, sd[1:] != sd[:-1]]
        did, dd = sd[keep], sx[keep]
    if len(did) == 0 and len(bfi) == 0:
        return {"displacements_m": np.zeros_like(v), "report": {"support_faces": 0, "free_vertices": 0}}

    # Affected direct vertices and all barycentric supports seed a connected face patch.
    changed = np.linalg.norm(dd, axis=1) > 1e-14
    seed = np.zeros(len(f), dtype=bool)
    if changed.any():
        seed[np.any(np.isin(f, did[changed]), axis=1)] = True
    seed[bfi] = True
    if not seed.any():
        raise ValueError("no anchor support faces")
    dual = _dual_adjacency(f)
    core = _expand(seed, dual, core_rings)
    support = core.copy()
    collar = np.empty(0, dtype=np.int64)
    expansion = 0
    for expansion in range(maximum_expansion_rings + 1):
        support = _expand(core, dual, collar_rings + expansion)
        collar, _ = _boundary_vertices(f, support)
        bad = np.intersect1d(collar, did[changed], assume_unique=False)
        if len(bad) == 0:
            break
    else:
        raise ValueError("could not place changed direct anchors inside fixed collar")
    sv = np.unique(f[support])
    direct_in = np.isin(did, sv)
    if np.any(changed & ~direct_in):
        raise ValueError("changed direct anchor escaped support")
    local = np.full(len(v), -1, dtype=np.int64)
    local[sv] = np.arange(len(sv), dtype=np.int64)
    sf = local[f[support]]
    if np.any(sf < 0):
        raise AssertionError("support face vertex omitted")

    # Inverse-length weighted graph Laplacian over support edges.
    er = np.concatenate((sf[:, [0, 1]], sf[:, [1, 2]], sf[:, [2, 0]]))
    er.sort(axis=1)
    ue, counts = np.unique(er, axis=0, return_counts=True)
    ga = v[sv[ue[:, 0]]]
    gb = v[sv[ue[:, 1]]]
    lengths = np.linalg.norm(ga - gb, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 1e-12):
        raise ValueError("degenerate support edge")
    w = 1.0 / lengths
    i, j = ue[:, 0], ue[:, 1]
    rows = np.r_[i, j, i, j]
    cols = np.r_[i, j, j, i]
    vals = np.r_[w, w, -w, -w]
    L = sparse.coo_matrix((vals, (rows, cols)), shape=(len(sv), len(sv))).tocsc()

    collar_local = local[collar]
    collar_local = collar_local[collar_local >= 0]
    direct_local = local[did[direct_in]]
    direct_vals = dd[direct_in]
    fixed_local = np.unique(np.r_[collar_local, direct_local])
    fixed_values = np.zeros((len(fixed_local), 3), dtype=np.float64)
    fixed_map = np.full(len(sv), -1, dtype=np.int64)
    fixed_map[fixed_local] = np.arange(len(fixed_local))
    if len(direct_local):
        for li, value in zip(direct_local, direct_vals):
            fixed_values[fixed_map[li]] = value
    free_local = np.setdiff1d(np.arange(len(sv), dtype=np.int64), fixed_local, assume_unique=True)
    if len(free_local) == 0:
        out = np.zeros_like(v)
        out[sv] = fixed_values[fixed_map] if len(fixed_local) else 0
        # Recheck exact constraints below; normally synthetic tiny cases may have no free rows.
        for vid, val in zip(did, dd):
            if np.linalg.norm(out[vid] - val) > anchor_tolerance_m:
                raise ValueError("direct anchors infeasible with empty free patch")
        for face, weights, val in zip(bfi, bw, bd):
            if np.linalg.norm(weights @ out[f[face]] - val) > anchor_tolerance_m:
                raise ValueError("bary anchors infeasible with empty free patch")
        return {"displacements_m": out, "report": {"support_faces": int(support.sum()), "support_vertices": int(len(sv)), "free_vertices": 0, "collar_vertices": int(len(collar)), "anchor_residual_m": 0.0}}

    Lff = L[free_local, :][:, free_local].tocsc()
    Lfk = L[free_local, :][:, fixed_local]
    rhs = -Lfk @ fixed_values
    lu = splu(Lff)
    base = lu.solve(np.asarray(rhs))

    # Exact barycentric rows, including fixed-anchor contributions.
    ar, ac, av, target = [], [], [], []
    fixed_position = {int(x): k for k, x in enumerate(fixed_local)}
    free_position = {int(x): k for k, x in enumerate(free_local)}
    for row, (face_id, weights, value) in enumerate(zip(bfi, bw, bd)):
        tri_local = local[f[int(face_id)]]
        residual_target = value.copy()
        for node_local, weight in zip(tri_local, weights):
            if node_local in fixed_position:
                residual_target -= weight * fixed_values[fixed_position[int(node_local)]]
            elif node_local in free_position:
                ar.append(row); ac.append(free_position[int(node_local)]); av.append(float(weight))
        target.append(residual_target)
    target = np.asarray(target, dtype=np.float64).reshape((-1, 3))
    A = sparse.coo_matrix((av, (ar, ac)), shape=(len(bfi), len(free_local))).tocsr()
    if len(bfi):
        residual = target - A @ base
        if A.nnz == 0:
            if np.max(np.linalg.norm(residual, axis=1), initial=0.0) > anchor_tolerance_m:
                raise ValueError("barycentric rows conflict with fixed anchors")
            solution = base
        else:
            Z = lu.solve(A.T.toarray())
            S = np.asarray(A @ Z)
            multipliers, *_ = lstsq(S, residual, cond=1e-12, lapack_driver="gelsd")
            solution = base + Z @ multipliers
            err = A @ solution - target
            if np.max(np.linalg.norm(err, axis=1), initial=0.0) > anchor_tolerance_m:
                raise ValueError("barycentric anchors are inconsistent with direct anchors/collar")
    else:
        solution = base

    out = np.zeros_like(v)
    out[sv] = 0.0
    out[sv[fixed_local]] = fixed_values
    out[sv[free_local]] = solution
    direct_err = np.linalg.norm(out[did] - dd, axis=1)
    bary_err = np.array([np.linalg.norm(wt @ out[f[fi]] - dv) for fi, wt, dv in zip(bfi, bw, bd)])
    if np.max(direct_err, initial=0.0) > anchor_tolerance_m or np.max(bary_err, initial=0.0) > anchor_tolerance_m:
        raise ValueError("postsolve anchor reproduction exceeded tolerance")
    report = {
        "support_faces": int(support.sum()),
        "support_vertices": int(len(sv)),
        "free_vertices": int(len(free_local)),
        "collar_vertices": int(len(collar)),
        "collar_expansion_rings": int(expansion),
        "direct_anchor_count": int(len(did)),
        "bary_anchor_count": int(len(bfi)),
        "direct_max_residual_m": float(direct_err.max(initial=0.0)),
        "bary_max_residual_m": float(bary_err.max(initial=0.0)),
        "outside_nonzero_vertices": int(np.count_nonzero(np.linalg.norm(out[np.setdiff1d(np.arange(len(v)), sv)], axis=1) > 0.0)),
    }
    return {"displacements_m": out, "report": report}
