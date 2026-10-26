"""Pure same-count retriangulation proposals for two adjacent interior vertices.

No geometry is executed and no coordinates or candidate data are modified.
"""
from collections import defaultdict
import importlib.util
from pathlib import Path
import sys

import numpy as np

STAR569_PATH = Path(__file__).resolve().parent.parent / "connected-star-569" / "star_candidates569.py"
_spec = importlib.util.spec_from_file_location("star_candidates569_for_patch570", STAR569_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"cannot load pinned 569 chart helpers from {STAR569_PATH}")
star569 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = star569
_spec.loader.exec_module(star569)


def _boundary_edges(faces):
    edge_dirs = defaultdict(list)
    for tri in np.asarray(faces):
        a, b, c = map(int, tri)
        for u, v in ((a, b), (b, c), (c, a)):
            edge_dirs[(min(u, v), max(u, v))].append((u, v))
    boundary = []
    for edge, directions in edge_dirs.items():
        if len(directions) > 2:
            raise ValueError(f"nonmanifold patch edge {edge}")
        if len(directions) == 2:
            if directions[0] != directions[1][::-1]:
                raise ValueError(f"inconsistent internal edge orientation {edge}")
        else:
            boundary.append(directions[0])
    return tuple(sorted(boundary))


def _ring_from_boundary(boundary):
    if len(boundary) < 3:
        raise ValueError("patch boundary has fewer than three edges")
    outgoing = {}
    incoming = {}
    for u, v in boundary:
        if u == v or u in outgoing or v in incoming:
            raise ValueError("patch boundary is open, branched, or repeated")
        outgoing[u] = v
        incoming[v] = u
    if set(outgoing) != set(incoming):
        raise ValueError("patch boundary is open")
    start = min(outgoing)
    ring = [start]
    while True:
        nxt = outgoing[ring[-1]]
        if nxt == start:
            break
        if nxt in ring or len(ring) >= len(boundary):
            raise ValueError("patch boundary contains multiple cycles")
        ring.append(nxt)
    if len(ring) != len(boundary) or len(ring) != len(outgoing):
        raise ValueError("patch boundary is not one simple cycle")
    return tuple(ring)


def _patch_data(points, reference, faces, rows, centers, max_boundary):
    pts = np.asarray(points)
    ref = np.asarray(reference)
    all_faces = np.asarray(faces)
    row_ids = np.asarray(rows)
    centers = tuple(map(int, centers))
    if pts.ndim != 2 or pts.shape[1] != 3 or ref.shape != pts.shape:
        raise ValueError("points/reference must be matching Nx3 arrays")
    if not np.isfinite(pts).all() or not np.isfinite(ref).all():
        raise ValueError("non-finite coordinates")
    if all_faces.ndim != 2 or all_faces.shape[1] != 3 or all_faces.dtype.kind not in "iu":
        raise ValueError("faces must be integer Fx3")
    if row_ids.ndim != 1 or row_ids.dtype.kind not in "iu" or not len(row_ids):
        raise ValueError("rows must be a nonempty integer vector")
    if len(set(map(int, row_ids))) != len(row_ids) or np.any(row_ids < 0) or np.any(row_ids >= len(all_faces)):
        raise ValueError("invalid or repeated patch rows")
    if len(centers) != 2 or centers[0] == centers[1]:
        raise ValueError("exactly two distinct interior centers are required")
    if min(centers) < 0 or max(centers) >= len(pts):
        raise ValueError("center index is outside points")
    if isinstance(max_boundary, bool) or int(max_boundary) < 3:
        raise ValueError("max_boundary must be an integer >= 3")

    expected = set(map(int, np.flatnonzero(np.any(np.isin(all_faces, centers), axis=1))))
    if set(map(int, row_ids)) != expected:
        raise ValueError("rows must be the complete union of both vertex stars")
    old = all_faces[row_ids.astype(np.int64)]
    if len(old) < 5 or any(len(set(map(int, t))) != 3 for t in old):
        raise ValueError("degenerate or undersized patch")
    boundary = _boundary_edges(old)
    ring = _ring_from_boundary(boundary)
    k = len(ring)
    if k > int(max_boundary):
        raise ValueError("boundary degree exceeds configured maximum")
    if len(old) != k + 2:
        raise ValueError("two-interior disk must have k+2 faces")
    used = set(map(int, old.ravel()))
    interior = used - set(ring)
    if len(interior) != 2 or interior != set(centers):
        raise ValueError("patch must have exactly the two requested interior vertices")
    # Adjacent interior vertices share an interior edge with two incident faces.
    center_edge = tuple(sorted(centers))
    edge_count = sum(
        center_edge[0] in tri and center_edge[1] in tri
        for tri in old
    )
    if edge_count != 2:
        raise ValueError("the two interior vertices must share a two-face edge")
    if len({tuple(sorted(map(int, t))) for t in old}) != len(old):
        raise ValueError("duplicate old face")
    return pts, ref, all_faces, row_ids.astype(np.int64), old, ring, centers, used


def _chart_for(points, ring, centers):
    p0, c0, eps0 = star569.chart(points, ring, centers[0])
    p1, c1, eps1 = star569.chart(points, ring, centers[1])
    if not np.allclose(p0, p1, rtol=0.0, atol=1e-12) or abs(eps0 - eps1) > 1e-14:
        raise ValueError("inconsistent boundary PCA chart")
    coords = {int(node): p0[i] for i, node in enumerate(ring)}
    coords[centers[0]] = np.asarray(c0)
    coords[centers[1]] = np.asarray(c1)
    return p0, coords, float(max(eps0, eps1))


def _insert_vertex(triangles, vertex, coords, eps):
    containing = []
    q = coords[int(vertex)]
    for idx, tri in enumerate(triangles):
        a, b, c = (coords[int(v)] for v in tri)
        signs = (star569.orient(a, b, q), star569.orient(b, c, q), star569.orient(c, a, q))
        if min(signs) > eps:
            containing.append(idx)
    if len(containing) != 1:
        return None
    idx = containing[0]
    a, b, c = map(int, triangles[idx])
    replacement = [(a, b, int(vertex)), (b, c, int(vertex)), (c, a, int(vertex))]
    return triangles[:idx] + replacement + triangles[idx + 1:]


def _positive_orientation(points, old, new):
    xyz = np.asarray(points, dtype=np.float64)
    oldv = np.cross(xyz[old[:, 1]] - xyz[old[:, 0]], xyz[old[:, 2]] - xyz[old[:, 0]])
    refv = oldv.sum(axis=0)
    norm = float(np.linalg.norm(refv))
    scale = max(float(np.max(np.linalg.norm(oldv, axis=1))), norm, 1e-300)
    if not np.isfinite(norm) or norm <= 64 * np.finfo(np.float64).eps * scale:
        return False
    newv = np.cross(xyz[new[:, 1]] - xyz[new[:, 0]], xyz[new[:, 2]] - xyz[new[:, 0]])
    dots = newv @ refv
    return bool(np.isfinite(dots).all() and np.all(dots > 64 * np.finfo(np.float64).eps * scale * norm))


def _oriented_signature(faces):
    # Face order and cyclic start do not affect topology; reversal remains distinct.
    out = []
    for tri in np.asarray(faces):
        a, b, c = map(int, tri)
        out.append(min((a, b, c), (b, c, a), (c, a, b)))
    return tuple(sorted(out))


def patch_candidates(points, reference, faces, rows, centers, max_boundary=9):
    """Return deterministic same-count patches with the same oriented boundary.

    Candidate construction enumerates simple boundary triangulations and both
    insertion orders for the two existing interior vertices. Points are fixed.
    """
    pts, ref, all_faces, row_ids, old, ring, centers, used = _patch_data(
        points, reference, faces, rows, centers, max_boundary
    )
    old_boundary = _boundary_edges(old)
    old_signature = _oriented_signature(old)
    candidates = {}
    for geometry in (pts, ref):
        try:
            p, coords, eps = _chart_for(geometry, ring, centers)
        except ValueError:
            continue
        boundary_tris = star569.polygon_triangulations(p, eps)
        for triangulation in boundary_tris:
            initial = [tuple(ring[i] for i in tri) for tri in triangulation]
            for order in (centers, centers[::-1]):
                work = initial
                for center in order:
                    work = _insert_vertex(work, center, coords, eps)
                    if work is None:
                        break
                if work is None:
                    continue
                nf = np.asarray(work, dtype=all_faces.dtype)
                if nf.shape != (len(row_ids), 3):
                    continue
                if any(len(set(map(int, t))) != 3 for t in nf):
                    continue
                if len({_face_key(t) for t in nf}) != len(nf):
                    continue
                if set(map(int, nf.ravel())) != used:
                    continue
                try:
                    if _boundary_edges(nf) != old_boundary:
                        continue
                except ValueError:
                    continue
                if _oriented_signature(nf) == old_signature:
                    continue
                if not _positive_orientation(pts, old, nf) or not _positive_orientation(ref, old, nf):
                    continue
                key = _oriented_signature(nf)
                candidates[key] = nf
    return [candidates[key] for key in sorted(candidates)]


def _face_key(tri):
    return tuple(sorted(map(int, tri)))


def external_edge_conflict(faces, rows, new_faces, edge_map):
    """Reject new local diagonals that already belong to an outside face."""
    row_set = set(map(int, rows))
    old_edges = set()
    for tri in np.asarray(faces)[list(rows)]:
        for a, b in zip(tri, np.roll(tri, -1)):
            old_edges.add(tuple(sorted((int(a), int(b)))))
    for tri in np.asarray(new_faces):
        for a, b in zip(tri, np.roll(tri, -1)):
            edge = tuple(sorted((int(a), int(b))))
            if edge not in old_edges and any(int(v[0]) not in row_set for v in edge_map.get(edge, ())):
                return True
    return False

