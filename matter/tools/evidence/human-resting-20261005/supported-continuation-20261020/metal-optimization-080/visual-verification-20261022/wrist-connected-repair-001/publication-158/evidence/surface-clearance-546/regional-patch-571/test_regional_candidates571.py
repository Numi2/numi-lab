import numpy as np
import pytest
from scipy.spatial import Delaunay

import regional_candidates571 as m


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _fan(poly, center):
    xy = np.vstack((poly, center))
    center_id = len(poly)
    faces = []
    for i in range(len(poly)):
        tri = [center_id, i, (i + 1) % len(poly)]
        if _orient(*xy[tri]) < 0:
            tri[1], tri[2] = tri[2], tri[1]
        faces.append(tri)
    return np.c_[xy, np.zeros(len(xy))].astype(np.float64), np.asarray(faces, dtype=np.int64)


def _convex_patch():
    angle = np.arange(8) * (2 * np.pi / 8)
    boundary = np.c_[np.cos(angle), np.sin(angle)]
    xy = np.vstack((boundary, [-0.22, 0.08], [0.27, -0.12]))
    points = np.c_[xy, np.zeros(len(xy))].astype(np.float64)
    faces = Delaunay(xy).simplices.copy()
    for tri in faces:
        if _orient(*xy[tri]) < 0:
            tri[1], tri[2] = tri[2], tri[1]
    return points, faces


def test_convex_patch_returns_same_domain_and_oriented_boundary():
    points, faces = _convex_patch()
    rows = np.arange(len(faces))
    candidates = m.patch_candidates(points, points.copy(), faces, rows)
    assert candidates
    owner, _ = m._owners()
    boundary = owner._ring_from_boundary(owner._boundary_edges(faces))
    for candidate in candidates:
        assert candidate.shape == faces.shape
        assert set(candidate.ravel()) == set(faces.ravel())
        assert m._canonical(candidate) != m._canonical(faces)
        assert m._boundary_matches(candidate, boundary)


def test_concave_disk_filters_convex_hull_exterior_triangles():
    polygon = np.asarray([[0, 0], [3, 0], [3, 3], [2, 2.3], [1, 3], [0, 3]], dtype=float)
    points, faces = _fan(polygon, np.asarray([1.5, 0.7]))
    candidates = m.patch_candidates(points, points.copy(), faces, np.arange(len(faces)))
    assert candidates
    owner, _ = m._owners()
    boundary = owner._ring_from_boundary(owner._boundary_edges(faces))
    poly, xy, eps = m._chart(points, boundary, np.unique(faces))
    for candidate in candidates:
        assert candidate.shape == faces.shape
        assert m._boundary_matches(candidate, boundary)
        assert all(m._inside_triangle_domain(np.asarray([xy[int(v)] for v in tri]), poly, eps)
                   for tri in candidate)


def test_disconnected_or_open_patch_is_rejected():
    points, faces = _convex_patch()
    with pytest.raises(ValueError):
        m.patch_candidates(points, points.copy(), faces, np.asarray([0, len(faces) - 1]))


def test_external_edge_conflict_uses_pinned_detector():
    points, faces = _convex_patch()
    rows = np.arange(len(faces))
    candidate = m.patch_candidates(points, points.copy(), faces, rows)[0]
    old_edges = {tuple(sorted((int(a), int(b))))
                 for tri in faces for a, b in zip(tri, np.roll(tri, -1))}
    new_edge = next(tuple(sorted((int(a), int(b))))
                    for tri in candidate for a, b in zip(tri, np.roll(tri, -1))
                    if tuple(sorted((int(a), int(b)))) not in old_edges)
    assert m.external_edge_conflict(faces, rows, candidate, {new_edge: [(999, 0, 0)]})
