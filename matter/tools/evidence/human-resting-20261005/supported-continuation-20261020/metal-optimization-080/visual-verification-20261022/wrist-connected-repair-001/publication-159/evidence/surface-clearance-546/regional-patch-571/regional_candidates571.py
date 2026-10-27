"""Fixed-coordinate same-count Delaunay proposals for a simple disk patch.

This helper proposes face arrays only. The root workflow applies provenance and
runs its existing exact quality, self-intersection, and typed-bank checks.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import Delaunay, QhullError

HERE = Path(__file__).resolve().parent
D_ROOT = HERE.parents[1]
N_ROOT = D_ROOT / "surface-clearance-546"
PATCH570_PATH = N_ROOT / "connected-patch-570" / "patch_candidates570.py"
PATCH570_SHA = "5e309fd361ac99de50f1d3f9aeca8d5f5711a406969525c09a1a80481bf0861d"
STAR569_PATH = N_ROOT / "connected-star-569" / "star_candidates569.py"
STAR569_SHA = "81b33966ba0c2182c86d2a6d66862064105cf48e91c2efe0cfcc0ef6359d1b2c"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=1)
def _owners():
    for path, expected in ((PATCH570_PATH, PATCH570_SHA), (STAR569_PATH, STAR569_SHA)):
        actual = _sha(path)
        if actual != expected:
            raise ValueError(f"pinned owner SHA mismatch for {path}: {actual}")
    loaded = []
    for name, path in (("patch_candidates570_for571", PATCH570_PATH),
                       ("star_candidates569_for571", STAR569_PATH)):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load pinned helper {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        loaded.append(module)
    return tuple(loaded)


def _canonical(faces):
    return tuple(sorted(tuple(sorted(map(int, tri))) for tri in np.asarray(faces)))


def _boundary_and_vertices(points, faces, rows, max_boundary):
    patch_owner, _ = _owners()
    mesh_faces = np.asarray(faces)
    row_ids = np.asarray(rows)
    xyz = np.asarray(points)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.dtype.kind != "f" or not np.isfinite(xyz).all():
        raise ValueError("points must be finite floating Nx3")
    if mesh_faces.ndim != 2 or mesh_faces.shape[1] != 3 or mesh_faces.dtype.kind not in "iu":
        raise ValueError("faces must be integer Fx3")
    if row_ids.ndim != 1 or row_ids.dtype.kind not in "iu" or not len(row_ids):
        raise ValueError("rows must be nonempty integer face rows")
    row_ids = row_ids.astype(np.int64)
    if len(np.unique(row_ids)) != len(row_ids) or row_ids.min() < 0 or row_ids.max() >= len(mesh_faces):
        raise ValueError("rows contain duplicates or out-of-range face IDs")
    old = mesh_faces[row_ids]
    if any(len(set(map(int, tri))) != 3 for tri in old):
        raise ValueError("input patch contains degenerate triangles")
    if len({_canonical([tri]) for tri in old}) != len(old):
        raise ValueError("input patch contains duplicate triangles")

    boundary_edges = patch_owner._boundary_edges(old)
    boundary = patch_owner._ring_from_boundary(boundary_edges)
    if len(boundary) > max_boundary:
        raise ValueError(f"boundary count {len(boundary)} exceeds max_boundary={max_boundary}")

    edge_incidence = defaultdict(list)
    face_adjacency = [[] for _ in old]
    for local, tri in enumerate(old):
        a, b, c = map(int, tri)
        for u, v in ((a, b), (b, c), (c, a)):
            edge_incidence[tuple(sorted((u, v)))].append((local, u, v))
    for edge, incident in edge_incidence.items():
        if len(incident) == 2:
            i, u, v = incident[0]
            j, x, y = incident[1]
            if (u, v) != (y, x):
                raise ValueError(f"inconsistent internal winding at edge {edge}")
            face_adjacency[i].append(j)
            face_adjacency[j].append(i)
        elif len(incident) != 1:
            raise ValueError(f"nonmanifold patch edge {edge}")
    reached = {0}
    pending = [0]
    while pending:
        i = pending.pop()
        for j in face_adjacency[i]:
            if j not in reached:
                reached.add(j)
                pending.append(j)
    if len(reached) != len(old):
        raise ValueError("patch faces are disconnected")

    vertices = np.unique(old).astype(np.int64)
    if len(vertices) - len(edge_incidence) + len(old) != 1:
        raise ValueError("patch is not a topological disk")
    if len(old) != len(boundary) + 2 * (len(vertices) - len(boundary)) - 2:
        raise ValueError("patch face count does not match a triangulated disk")
    return row_ids, old, boundary, vertices


def _chart(points, boundary, vertices):
    _, star = _owners()
    ids = list(map(int, vertices))
    polygon = None
    xy = {}
    eps = None
    for node in ids:
        p, center, node_eps = star.chart(points, boundary, node)
        if polygon is None:
            polygon = np.asarray(p, dtype=np.float64)
            eps = float(node_eps)
        elif not np.allclose(polygon, p, rtol=0.0, atol=max(eps, node_eps)):
            raise ValueError("pinned PCA chart changed across vertices")
        xy[node] = np.asarray(center, dtype=np.float64)
    return polygon, xy, float(eps)


def _inside_triangle_domain(tri_xy, polygon, eps):
    _, star = _owners()
    if not star.inside_polygon(np.mean(tri_xy, axis=0), polygon, eps):
        return False
    if not all(star.inside_polygon(pt, polygon, eps) for pt in tri_xy):
        return False
    for a, b in zip(tri_xy, np.roll(tri_xy, -1, axis=0)):
        for c, d in zip(polygon, np.roll(polygon, -1, axis=0)):
            if any(np.linalg.norm(x - y) <= eps for x in (a, b) for y in (c, d)):
                continue
            if star.segments_cross(a, b, c, d, eps):
                return False
    return True


def _boundary_matches(candidate, boundary):
    edges = defaultdict(list)
    for tri in np.asarray(candidate, dtype=np.int64):
        for u, v in zip(tri, np.roll(tri, -1)):
            u, v = int(u), int(v)
            edges[tuple(sorted((u, v)))].append((u, v))
    actual = []
    for incident in edges.values():
        if len(incident) > 2:
            return False
        if len(incident) == 1:
            actual.append(incident[0])
        elif incident[0] != incident[1][::-1]:
            return False
    wanted = {(int(boundary[i]), int(boundary[(i + 1) % len(boundary)]))
              for i in range(len(boundary))}
    return len(actual) == len(wanted) and set(actual) == wanted


def _valid_candidate(candidate, old, vertices, boundary, chart_data):
    _, star = _owners()
    if candidate.shape != old.shape or len({_canonical([tri]) for tri in candidate}) != len(candidate):
        return False
    if set(map(int, candidate.ravel())) != set(map(int, vertices)):
        return False
    if not _boundary_matches(candidate, boundary):
        return False
    area_sum = []
    for polygon, xy, eps in chart_data:
        signed_polygon_area = sum(star.orient(np.zeros(2), polygon[i], polygon[(i + 1) % len(polygon)])
                                  for i in range(len(polygon)))
        tri_area_sum = 0.0
        for tri in candidate:
            a, b, c = map(int, tri)
            signed = star.orient(xy[a], xy[b], xy[c])
            if signed <= eps:
                return False
            tri_xy = np.asarray((xy[a], xy[b], xy[c]))
            if not _inside_triangle_domain(tri_xy, polygon, eps):
                return False
            tri_area_sum += signed
        if abs(signed_polygon_area - tri_area_sum) > max(eps * len(candidate) * 8, abs(signed_polygon_area) * 1e-9):
            return False
        area_sum.append(tri_area_sum)
    return True


def patch_candidates(points, reference, faces, rows, centers=None, max_boundary=48):
    """Return deterministic same-count replacement face arrays for a disk patch.

    All unique used vertices and the directed boundary remain fixed. Delaunay is
    run in the current and reference PCA charts at three anisotropic metrics;
    concavity-exterior triangles are discarded before the full disk checks.
    """
    points = np.asarray(points)
    reference = np.asarray(reference)
    faces = np.asarray(faces)
    row_ids, old, boundary, vertices = _boundary_and_vertices(points, faces, rows, max_boundary)
    if reference.shape != points.shape or reference.dtype.kind != "f" or not np.isfinite(reference).all():
        raise ValueError("reference must be finite floating and match points")
    if isinstance(max_boundary, bool) or int(max_boundary) < 3:
        raise ValueError("max_boundary must be an integer >= 3")
    if centers is not None:
        center_ids = list(map(int, centers))
        interior = set(map(int, vertices)) - set(map(int, boundary))
        if len(center_ids) != len(set(center_ids)) or not set(center_ids).issubset(interior):
            raise ValueError("centers must be unique interior nodes")

    current_poly, current_xy, current_eps = _chart(points, boundary, vertices)
    ref_poly, ref_xy, ref_eps = _chart(reference, boundary, vertices)
    _, star = _owners()
    if not star.inside_polygon(current_poly[0], current_poly, current_eps):
        raise ValueError("invalid current PCA polygon")
    if not star.inside_polygon(ref_poly[0], ref_poly, ref_eps):
        raise ValueError("invalid reference PCA polygon")
    for polygon, xy, eps in ((current_poly, current_xy, current_eps),
                             (ref_poly, ref_xy, ref_eps)):
        for node in set(map(int, vertices)) - set(map(int, boundary)):
            if not star.inside_polygon(xy[node], polygon, eps):
                raise ValueError(f"interior node {node} lies outside PCA boundary")
    chart_data = ((current_poly, current_xy, current_eps),
                  (ref_poly, ref_xy, ref_eps))
    base_key = _canonical(old)
    proposals = {}
    expected_faces = len(row_ids)
    boundary_nodes = np.asarray(boundary, dtype=np.int64)
    for chart_name, polygon, xy, eps in (
        ("current", current_poly, current_xy, current_eps),
        ("reference", ref_poly, ref_xy, ref_eps),
    ):
        cloud = np.asarray([xy[int(node)] for node in vertices], dtype=np.float64)
        for scale in (0.5, 1.0, 2.0):
            for angle in (0.0, 45.0, 90.0):
                radians = np.deg2rad(angle)
                c, s = float(np.cos(radians)), float(np.sin(radians))
                rotation = np.asarray(((c, -s), (s, c)))
                metric = rotation @ np.diag((scale, 1.0)) @ rotation.T
                try:
                    tess = Delaunay(cloud @ metric.T)
                except QhullError:
                    continue
                if len(getattr(tess, "coplanar", ())):
                    continue
                if tess.simplices.ndim != 2 or tess.simplices.shape[1] != 3:
                    continue
                if tess.simplices.size and (tess.simplices.min() < 0 or tess.simplices.max() >= len(vertices)):
                    continue
                candidate = np.asarray(vertices[tess.simplices], dtype=np.int64)
                # Delaunay covers the convex hull; discard triangles outside a
                # concave boundary before testing the complete patch inventory.
                keep = []
                for tri in candidate:
                    tri_xy = np.asarray([xy[int(node)] for node in tri])
                    if _inside_triangle_domain(tri_xy, polygon, eps):
                        keep.append(tri)
                candidate = np.asarray(keep, dtype=np.int64).reshape((-1, 3))
                if len(candidate) != expected_faces:
                    continue
                for tri in candidate:
                    if star.orient(xy[int(tri[0])], xy[int(tri[1])], xy[int(tri[2])]) < 0:
                        tri[1], tri[2] = tri[2], tri[1]
                if _canonical(candidate) == base_key:
                    continue
                key = _canonical(candidate)
                if key in proposals:
                    continue
                if not _valid_candidate(candidate, old, vertices, boundary, chart_data):
                    continue
                proposals[key] = candidate.astype(faces.dtype, copy=False)
    return [proposals[key] for key in sorted(proposals)]


def external_edge_conflict(faces, rows, new_faces, edge_map):
    """Use the pinned 569 exact edge conflict predicate."""
    _, star = _owners()
    return bool(star.external_edge_conflict(faces, rows, new_faces, edge_map))
