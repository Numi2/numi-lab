import importlib.util
from pathlib import Path
import unittest

import numpy as np

MODULE_PATH = Path(__file__).with_name("human_skin_tetra_refinement.py")
spec = importlib.util.spec_from_file_location("human_skin_tetra_refinement_tested", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def positive_tets(points, tets):
    out = np.asarray(tets, dtype=np.int64).copy()
    for row in range(len(out)):
        if module._signed_six_volume(points, out[row]) < 0:
            out[row, 0], out[row, 1] = out[row, 1], out[row, 0]
    return out


def oriented_boundary(tets):
    t = np.asarray(tets, dtype=np.int64)
    rows = np.concatenate((t[:, [1, 2, 3]], t[:, [0, 3, 2]], t[:, [0, 1, 3]], t[:, [0, 2, 1]]), axis=0)
    incidence = {}
    for face in rows:
        key = tuple(sorted(map(int, face)))
        r = int(np.argmin(face))
        directed = tuple(int(face[(r + i) % 3]) for i in range(3))
        incidence.setdefault(key, []).append(directed)
    if any(len(v) > 2 for v in incidence.values()):
        raise AssertionError("nonmanifold test fixture")
    for key, directed in incidence.items():
        if len(directed) == 2 and directed[0] != (directed[1][0], directed[1][2], directed[1][1]):
            raise AssertionError(f"shared face orientation mismatch: {key} {directed}")
    return {key: value[0] for key, value in incidence.items() if len(value) == 1}


def expected_boundary_after_edge_splits(boundary, original_node_count, edges):
    edge_midpoint = {tuple(sorted(edge)): original_node_count + row for row, edge in enumerate(sorted(edges))}
    expected = {}
    for face in boundary.values():
        edge = next((e for e in edge_midpoint if set(e).issubset(face)), None)
        triangles = [face]
        if edge is not None:
            cycle = None
            for k in range(3):
                candidate = tuple(face[(k + j) % 3] for j in range(3))
                if {candidate[0], candidate[1]} == set(edge):
                    cycle = candidate
                    break
            if cycle is None:
                raise AssertionError("marked edge is not on its selected boundary face")
            u, v, w = cycle
            m = edge_midpoint[tuple(sorted((u, v)))]
            triangles = [(u, m, w), (m, v, w)]
        for tri in triangles:
            k = int(np.argmin(tri))
            expected[tuple(sorted(tri))] = tuple(tri[(k + j) % 3] for j in range(3))
    return expected


class HumanSkinTetraRefinementTests(unittest.TestCase):
    def _three_tet_edge_ring(self):
        points = np.asarray([
            [0, 0, 0], [0, 0, 1], [1, 0, .5], [-.5, .8660254, .5], [-.5, -.8660254, .5],
        ], dtype=np.float32)
        tets = positive_tets(points, np.asarray([[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]]))
        return points, tets

    def test_shared_internal_edge_is_split_once_and_conformingly(self):
        points, tets = self._three_tet_edge_ring()
        before = oriented_boundary(tets)
        n = len(points)
        current = points + np.asarray([.25, -.5, .75], dtype=np.float32)
        ptr = np.arange(n + 1, dtype=np.int64)
        ids = np.arange(100, 100 + n, dtype=np.int64)
        weights = np.ones(n, dtype=np.float64)
        materials = np.asarray([2, 5, 5], dtype=np.int16)
        source_cells = np.asarray([91, 92, 93], dtype=np.int64)

        out = module.refine_tetra_edges(
            points, tets, [(1, 0), (0, 1)],
            point_fields={"current": current},
            material_ids=materials,
            parent_cell_ids=source_cells,
            source_vertex_indptr=ptr,
            source_vertex_indices=ids,
            source_vertex_weights=weights,
        )

        self.assertEqual(out["reference_points_f32"].shape, (n + 1, 3))
        np.testing.assert_array_equal(out["reference_points_f32"][:n], points)
        np.testing.assert_array_equal(out["point_fields"]["current"][:n], current)
        np.testing.assert_array_equal(out["point_fields"]["current"][n], (current[0] + current[1]) * np.float32(.5))
        self.assertEqual(out["tetrahedra"].shape, (6, 4))
        self.assertEqual(out["new_node_parent_edges"].tolist(), [[0, 1]])
        np.testing.assert_array_equal(out["new_node_rows"], [n])
        self.assertFalse(np.any(np.any(out["tetrahedra"] == 0, axis=1) & np.any(out["tetrahedra"] == 1, axis=1)))
        self.assertEqual(oriented_boundary(out["tetrahedra"]), before)
        self.assertTrue(np.all(module._volumes_and_quality(out["reference_points_f32"], out["tetrahedra"])[0] > 0))
        self.assertEqual(out["child_parent_cell_ids"].tolist(), [91, 91, 92, 92, 93, 93])
        np.testing.assert_array_equal(out["material_ids"], [2, 2, 5, 5, 5, 5])
        self.assertEqual(out["child_reference_volume_m3"].shape, (6,))
        self.assertEqual(out["parent_reference_volume_before_m3"].shape, (3,))
        self.assertEqual(out["parent_reference_volume_after_m3"].shape, (3,))
        self.assertEqual(out["parent_reference_volume_relative_drift"].shape, (3,))
        np.testing.assert_allclose(np.bincount(out["child_parent_rows"], weights=out["parent_reference_volume_fraction"], minlength=3), [1, 1, 1], atol=2e-15, rtol=0)
        np.testing.assert_array_equal(out["source_vertex_ancestry_indptr"][:n + 1], ptr)
        np.testing.assert_array_equal(out["source_vertex_ancestry_indices"][:n], ids)
        np.testing.assert_allclose(out["source_vertex_ancestry_weights"][n:], [.5, .5])
        self.assertLess(out["report"]["maximum_parent_reference_volume_relative_drift"], 1e-6)

    def test_multiple_marked_edges_in_one_cell_update_incidence(self):
        points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        tets = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        before = oriented_boundary(tets)
        out = module.refine_tetra_edges(points, tets, [(2, 3), (0, 1)])
        self.assertEqual(out["tetrahedra"].shape, (4, 4))
        self.assertEqual(out["new_node_parent_edges"].tolist(), [[0, 1], [2, 3]])
        self.assertEqual(out["child_parent_rows"].tolist(), [0, 0, 0, 0])
        expected = expected_boundary_after_edge_splits(before, len(points), [(0, 1), (2, 3)])
        self.assertEqual(oriented_boundary(out["tetrahedra"]), expected)
        for edge in ((0, 1), (2, 3)):
            self.assertFalse(any(edge[0] in tet and edge[1] in tet for tet in out["tetrahedra"]))
        self.assertAlmostEqual(float(out["parent_reference_volume_fraction"].sum()), 1.0, places=14)

    def test_missing_or_invalid_edges_fail_closed(self):
        points, tets = self._three_tet_edge_ring()
        absent_points = np.vstack((points, np.asarray([[7, 7, 7]], dtype=np.float32)))
        with self.assertRaisesRegex(ValueError, "absent"):
            module.refine_tetra_edges(absent_points, tets, [(0, 5)])
        with self.assertRaisesRegex(ValueError, "invalid marked edge"):
            module.refine_tetra_edges(points, tets, [(0, 99)])
        with self.assertRaisesRegex(ValueError, "invalid marked edge"):
            module.refine_tetra_edges(points, tets, [(0, 0)])
        with self.assertRaisesRegex(ValueError, "positive finite signed"):
            module.refine_tetra_edges(points, tets[:, [1, 0, 2, 3]], [])

    def test_float32_collapsed_midpoint_and_quality_floor_are_reported(self):
        a = np.float32(1.0)
        b = np.nextafter(a, np.float32(2.0), dtype=np.float32)
        points = np.asarray([[a, 0, 0], [b, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=np.float32)
        tets = positive_tets(points, np.asarray([[0, 1, 2, 3]]))
        with self.assertRaisesRegex(ValueError, "midpoint collapsed"):
            module.refine_tetra_edges(points, tets, [(0, 1)])

        points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        tets = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "below required"):
            module.refine_tetra_edges(points, tets, [(0, 1)], minimum_normalized_quality=.5)

    def test_parent_and_material_arrays_must_match_input_domain(self):
        points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        tets = np.asarray([[0, 1, 2, 3]], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "one value per input tetrahedron"):
            module.refine_tetra_edges(points, tets, [(0, 1)], material_ids=np.asarray([1, 2]))
        with self.assertRaisesRegex(ValueError, "one value per input tetrahedron"):
            module.refine_tetra_edges(points, tets, [(0, 1)], parent_cell_ids=np.asarray([1, 2]))

    def test_float32_midpoint_parent_volume_drift_fails_closed_before_normalization(self):
        # At this coordinate magnitude, the midpoint rounds in x while the
        # other two coordinates remain distinct, moving it off the source edge.
        points = np.asarray([
            [100_000_000, 2, 5],
            [100_000_024, 0, 5],
            [100_000_016, 5, -5],
            [100_000_000, 1, -2],
        ], dtype=np.float32)
        tets = np.asarray([[1, 0, 2, 3]], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "per-parent reference-volume closure failed"):
            module.refine_tetra_edges(points, tets, [(0, 1)])

        diagnostic = module.refine_tetra_edges(
            points, tets, [(0, 1)], maximum_parent_reference_volume_relative_drift=None
        )
        self.assertGreater(
            float(diagnostic["parent_reference_volume_relative_drift"][0]), 1e-4
        )
        self.assertGreater(
            float(diagnostic["child_reference_volume_fraction_of_input_parent"].sum()), 1.0
        )
        self.assertAlmostEqual(
            float(diagnostic["parent_reference_volume_fraction"].sum()), 1.0, places=14
        )
        self.assertIsNone(diagnostic["report"]["parent_reference_volume_relative_drift_limit"])


if __name__ == "__main__":
    unittest.main()
