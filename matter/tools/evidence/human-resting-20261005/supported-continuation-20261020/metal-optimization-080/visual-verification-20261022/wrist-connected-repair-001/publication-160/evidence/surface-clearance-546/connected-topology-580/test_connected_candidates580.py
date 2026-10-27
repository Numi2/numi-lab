import numpy as np
from scipy.spatial import Delaunay

import connected_candidates580 as unified


def _boundary(faces):
    return unified._owners()["regional"]._boundary_and_vertices(
        np.zeros((int(np.max(faces)) + 1, 3), dtype=np.float64),
        faces,
        np.arange(len(faces)),
        128,
    )[2]


def _assert_unchanged(points, reference, faces, pbytes, rbytes, fbytes):
    assert points.tobytes() == pbytes
    assert reference.tobytes() == rbytes
    assert faces.tobytes() == fbytes


def _assert_patch(points, reference, faces, candidates):
    assert candidates
    owner = unified._owners()["regional"]
    old_boundary = owner._boundary_and_vertices(points, faces, np.arange(len(faces)), 128)[2]
    used = set(map(int, faces.ravel()))
    for candidate in candidates:
        assert candidate.shape == faces.shape
        assert set(map(int, candidate.ravel())) == used
        assert owner._boundary_and_vertices(points, candidate, np.arange(len(candidate)), 128)[2] == old_boundary
        assert len({_face_key(t) for t in candidate}) == len(candidate)


def _face_key(tri):
    return tuple(sorted(map(int, tri)))


def _one_star():
    k = 6
    theta = np.arange(k) * 2 * np.pi / k
    points = np.vstack((np.zeros((1, 3)), np.c_[np.cos(theta), np.sin(theta), np.zeros(k)]))
    faces = np.asarray([[0, i + 1, (i + 1) % k + 1] for i in range(k)], dtype=np.int64)
    return points, faces


def _two_star(k=5):
    theta = np.arange(k, dtype=np.float64) * (2 * np.pi / k)
    ring = np.c_[2 * np.cos(theta), 2 * np.sin(theta), np.zeros(k)]
    points = np.vstack((np.zeros((1, 3)), ring, np.zeros((1, 3))))
    center_b = k + 1
    points[center_b] = (points[0] + points[1] + points[2]) / 3
    fan = [[0, i + 1, (i + 1) % k + 1] for i in range(k)]
    faces = [[0, 1, center_b], [1, 2, center_b], [2, 0, center_b]] + fan[1:]
    return points, np.asarray(faces, dtype=np.int64)


def _curved_disk():
    angle = np.arange(7) * 2 * np.pi / 7
    p2 = np.vstack((np.c_[np.cos(angle), np.sin(angle)], [[0, 0], [0.3, 0.1]]))
    points = np.c_[p2, np.r_[np.zeros(7), 1.0, -1.0]].astype(np.float64)
    faces = Delaunay(p2).simplices.copy()
    for tri in faces:
        a, b, c = tri
        orient = (p2[b, 0] - p2[a, 0]) * (p2[c, 1] - p2[a, 1]) - (p2[b, 1] - p2[a, 1]) * (p2[c, 0] - p2[a, 0])
        if orient < 0:
            tri[1], tri[2] = tri[2], tri[1]
    return points, faces.astype(np.int64)


def test_zero_interior_two_face_quad_and_no_mutation():
    points = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    reference = points.copy()
    pbytes, rbytes, fbytes = points.tobytes(), reference.tobytes(), faces.tobytes()
    candidates = unified.patch_candidates(points, reference, faces, np.arange(2), max_boundary=9)
    _assert_patch(points, reference, faces, candidates)
    assert any({frozenset(map(int, tri)) for tri in c} == {
        frozenset((0, 1, 3)), frozenset((1, 2, 3))
    } for c in candidates)
    _assert_unchanged(points, reference, faces, pbytes, rbytes, fbytes)


def test_single_interior_star_family_and_no_mutation():
    points, faces = _one_star()
    reference = points.copy()
    pbytes, rbytes, fbytes = points.tobytes(), reference.tobytes(), faces.tobytes()
    candidates = unified.patch_candidates(points, reference, faces, np.arange(len(faces)), centers=(0,), max_boundary=9)
    _assert_patch(points, reference, faces, candidates)
    single = unified._owners()["single"].star_candidates(points, reference, faces, np.arange(len(faces)), 0, max_degree=9)
    assert single
    assert {_sig(c) for c in single}.issubset({_sig(c) for c in candidates})
    _assert_unchanged(points, reference, faces, pbytes, rbytes, fbytes)


def test_two_interior_star_family_and_no_mutation():
    points, faces = _two_star()
    reference = points.copy()
    centers = (0, len(points) - 1)
    pbytes, rbytes, fbytes = points.tobytes(), reference.tobytes(), faces.tobytes()
    candidates = unified.patch_candidates(points, reference, faces, np.arange(len(faces)), centers=centers, max_boundary=9)
    _assert_patch(points, reference, faces, candidates)
    double = unified._owners()["double"].patch_candidates(points, reference, faces, np.arange(len(faces)), centers, max_boundary=9)
    assert double
    assert {_sig(c) for c in double}.issubset({_sig(c) for c in candidates})
    _assert_unchanged(points, reference, faces, pbytes, rbytes, fbytes)


def test_curved_general_disk_includes_harmonic_family_and_no_mutation():
    points, faces = _curved_disk()
    reference = points.copy()
    rows = np.arange(len(faces))
    pbytes, rbytes, fbytes = points.tobytes(), reference.tobytes(), faces.tobytes()
    candidates = unified.patch_candidates(points, reference, faces, rows, max_boundary=9)
    _assert_patch(points, reference, faces, candidates)
    harmonic = unified._owners()["harmonic"].patch_candidates(points, reference, faces, rows, max_boundary=9)
    assert harmonic
    assert {_sig(c) for c in harmonic}.issubset({_sig(c) for c in candidates})
    _assert_unchanged(points, reference, faces, pbytes, rbytes, fbytes)


def _sig(faces):
    return tuple(sorted(min(tuple(map(int, t)), tuple(map(int, t[1:])) + (int(t[0]),),
                            tuple(map(int, t[2:])) + tuple(map(int, t[:2]))) for t in faces))

