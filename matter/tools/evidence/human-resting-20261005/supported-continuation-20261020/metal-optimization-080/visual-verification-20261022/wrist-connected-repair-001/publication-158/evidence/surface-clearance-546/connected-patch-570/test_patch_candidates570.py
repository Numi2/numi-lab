import numpy as np
import pytest

import patch_candidates570 as patch


def two_center_star(k=5):
    theta = np.arange(k, dtype=np.float64) * (2.0 * np.pi / k)
    ring_points = np.column_stack((2.0 * np.cos(theta), 2.0 * np.sin(theta), np.zeros(k)))
    points = np.vstack((np.zeros((1, 3)), ring_points, np.zeros((1, 3))))
    center_a = 0
    center_b = k + 1
    points[center_b] = (points[center_a] + points[1] + points[2]) / 3.0

    fan = [[center_a, i + 1, (i + 1) % k + 1] for i in range(k)]
    # Insert center_b into the first fan triangle. The resulting complete union
    # of both interior vertex stars has k+2 faces and the original boundary.
    faces = [[center_a, 1, center_b], [1, 2, center_b], [2, center_a, center_b]] + fan[1:]
    return points, np.asarray(faces, dtype=np.int64), (center_a, center_b)


def boundary(faces):
    return patch._boundary_edges(faces)


def test_convex_k5_two_center_patch_preserves_ids_boundary_count_and_orientation():
    points, faces, centers = two_center_star(5)
    rows = np.arange(len(faces), dtype=np.int64)
    candidates = patch.patch_candidates(points, points.copy(), faces, rows, centers, max_boundary=9)
    assert candidates

    old_boundary = boundary(faces)
    old_normal = np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                          points[faces[:, 2]] - points[faces[:, 0]]).sum(axis=0)
    for new in candidates:
        assert new.shape == faces.shape
        assert set(map(int, new.ravel())) == set(map(int, faces.ravel()))
        assert boundary(new) == old_boundary
        normals = np.cross(points[new[:, 1]] - points[new[:, 0]],
                           points[new[:, 2]] - points[new[:, 0]])
        assert np.all(normals @ old_normal > 0)


def test_open_or_non_disk_patch_is_rejected():
    points, faces, centers = two_center_star(5)
    with pytest.raises(ValueError, match="complete union"):
        patch.patch_candidates(points, points.copy(), faces, np.arange(len(faces) - 1), centers)

    broken = faces.copy()
    broken[0] = broken[1]
    with pytest.raises(ValueError):
        patch.patch_candidates(points, points.copy(), broken, np.arange(len(broken)), centers)

