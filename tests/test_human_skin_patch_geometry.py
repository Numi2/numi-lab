"""Regression checks for connected local skin patch boundary construction."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location(
    "human_skin_patch_geometry",
    Path(__file__).resolve().parents[1] / "tools/human_skin_patch_geometry.py")
geom = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(geom)


class TriangleDeformationMetricsTests(unittest.TestCase):
    def test_corner_axis_area_and_signed_orientation(self):
        points = np.array([[17., -3, 9], [18, -3, 9], [17, -1, 9], [50, 20, -10]])
        faces = np.array([[0, 1, 2]])
        tri = points[faces]
        area, cosine, oriented = geom.triangle_deformation_metrics(tri, tri.copy())
        np.testing.assert_allclose([area, cosine, oriented], [[1], [1], [1]])
        area, cosine, oriented = geom.triangle_deformation_metrics(tri, tri[:, [0, 2, 1]])
        np.testing.assert_allclose([area, cosine, oriented], [[1], [-1], [-1]])
        candidate = tri[:, :1]+2*(tri-tri[:, :1])
        area, cosine, oriented = geom.triangle_deformation_metrics(tri, candidate)
        np.testing.assert_allclose([area, cosine, oriented], [[4], [1], [4]])

    def test_collapse_empty_and_invalid_reference(self):
        tri = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0]]])
        a, c, o = geom.triangle_deformation_metrics(tri, np.zeros_like(tri))
        np.testing.assert_array_equal([a, c, o], [[0], [0], [0]])
        self.assertEqual(geom.triangle_deformation_metrics(
            np.empty((0, 3, 3)), np.empty((0, 3, 3)))[0].shape, (0,))
        for ref, candidate in [(tri[0], tri[0]), (np.zeros_like(tri), tri),
                               (tri, np.full_like(tri, np.nan))]:
            with self.assertRaises(ValueError):
                geom.triangle_deformation_metrics(ref, candidate)


class LinearTetFractionTests(unittest.TestCase):
    def fraction(self, values, level):
        return geom.p1_tetrahedral_sublevel_fractions(
            values, np.array([[0, 1, 2, 3]]), level)[0]

    def test_repeated_knots_analytic(self):
        for t in np.linspace(0, 1, 21):
            self.assertAlmostEqual(self.fraction([0, 1, 1, 1], t), t**3)
            self.assertAlmostEqual(self.fraction([0, 0, 0, 1], t), 1-(1-t)**3)
            self.assertAlmostEqual(self.fraction([0, 0, 1, 1], t), 3*t*t-2*t**3)
        self.assertEqual(self.fraction([1, 1, 1, 1], 1), 1)
        self.assertEqual(self.fraction([1, 1, 1, 1], .5), 0)

    def test_distinct_knots_divided_difference_reference(self):
        values = np.array([-2., -.3, .7, 3.])
        for t in np.linspace(-2, 3, 47):
            expected = sum(max(t-a, 0)**3 / np.prod(np.delete(values, i)-a)
                           for i, a in enumerate(values))
            self.assertAlmostEqual(self.fraction(values, t), expected, places=12)

    def test_permutation_affine_complement_and_monotonicity(self):
        import itertools
        values = np.array([-.8, -.1, -.1, 1.7])
        expected = self.fraction(values, .3)
        for order in itertools.permutations(range(4)):
            self.assertAlmostEqual(self.fraction(values[list(order)], .3), expected)
        self.assertAlmostEqual(self.fraction(values*7+13, .3*7+13), expected)
        self.assertAlmostEqual(self.fraction(-values, -.3), 1-expected)
        fractions = [self.fraction(values, t) for t in np.linspace(-1, 2, 100)]
        self.assertTrue(np.all(np.diff(fractions) >= 0))

    def test_batch_and_empty(self):
        values = np.array([0, 1, 1, 1, 0, 0, 0, 1.])
        actual = geom.p1_tetrahedral_sublevel_fractions(
            values, np.arange(8).reshape(2, 4), .5)
        np.testing.assert_allclose(actual, [.125, .875])
        self.assertEqual(geom.p1_tetrahedral_sublevel_fractions(
            [], np.empty((0, 4), dtype=np.int64), 0).shape, (0,))

    def test_invalid_inputs(self):
        good = np.array([[0, 1, 2, 3]])
        for values, cells, level in [
                ([0, 1, np.nan, 2], good, 0),
                ([[0, 1, 2, 3]], good, 0),
                ([0, 1, 2, 3], good.astype(float), 0),
                ([0, 1, 2, 3], [[0, 1, 2, 4]], 0),
                ([0, 1, 2, 3], [[0, 1, 2, -1]], 0),
                ([0, 1, 2, 3], [[0, 1, 1, 3]], 0),
                ([0, 1, 2, 3], good, np.inf),
                ([0, 1, 2, 3], good, [0])]:
            with self.subTest(values=values, cells=cells, level=level):
                with self.assertRaises(ValueError):
                    geom.p1_tetrahedral_sublevel_fractions(values, cells, level)


class LinearTetInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.t = np.array([[0, 1, 2, 3]])

    def test_ordinary_cut_preserves_edge_ancestry_and_orientation(self):
        phi = np.array([0., 1, 1, 1])
        x, faces, edges, alpha = geom.extract_p1_tetrahedral_interface(self.x, self.t, phi, .5)
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(faces.shape, (1, 3))
        np.testing.assert_allclose(x, (1-alpha[:, None])*self.x[edges[:, 0]]+alpha[:, None]*self.x[edges[:, 1]])
        np.testing.assert_allclose((1-alpha)*phi[edges[:, 0]]+alpha*phi[edges[:, 1]], .5)
        p = x[faces[0]].astype(float)
        self.assertGreater(np.dot(np.cross(p[1]-p[0], p[2]-p[0]), [1, 1, 1]), 0)

    def test_exact_threshold_vertices_and_quad(self):
        x, f, edge, alpha = geom.extract_p1_tetrahedral_interface(self.x, self.t, [-1, 0, 0, 1], 0)
        self.assertEqual(f.shape, (1, 3))
        for node in (1, 2):
            row = np.flatnonzero(np.all(edge == [node, node], axis=1))[0]
            np.testing.assert_array_equal(x[row], self.x[node].astype(np.float32))
            self.assertEqual(alpha[row], 0)
        x, f, edge, alpha = geom.extract_p1_tetrahedral_interface(self.x, self.t, [-1, -1, 1, 1], 0)
        self.assertEqual(f.shape, (2, 3))
        self.assertEqual(len(x), 4)

    def test_shared_level_face_has_one_owner_at_crossing(self):
        x = np.vstack([self.x, [0, 0, -1.]])
        t = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
        points, faces, edges, alpha = geom.extract_p1_tetrahedral_interface(x, t, [0, 0, 0, 1, -1], 0)
        self.assertEqual(len(faces), 1)
        self.assertTrue(np.all(edges[:, 0] == edges[:, 1]))
        # Both cells below: the internal shared face cancels exactly.
        _, faces, _, _ = geom.extract_p1_tetrahedral_interface(x, t, [0, 0, 0, -1, -1], 0)
        self.assertEqual(len(faces), 0)

    def test_cell_order_and_orientation_do_not_change_surface(self):
        a = geom.extract_p1_tetrahedral_interface(self.x, self.t, [-2, -.3, .7, 3], .1)
        b = geom.extract_p1_tetrahedral_interface(self.x, self.t[:, [3, 1, 0, 2]], [-2, -.3, .7, 3], .1)
        def coordinates(result):
            x, faces, _, _ = result
            return sorted(tuple(sorted(tuple(p) for p in x[f])) for f in faces)
        self.assertEqual(coordinates(a), coordinates(b))

    def test_plateau_and_float32_collapse_rejected(self):
        with self.assertRaisesRegex(ValueError, "constant threshold"):
            geom.extract_p1_tetrahedral_interface(self.x, self.t, [0, 0, 0, 0], 0)
        with self.assertRaisesRegex(ValueError, "Float32"):
            geom.extract_p1_tetrahedral_interface(self.x*1e-12+1, self.t, [0, 1, 1, 1], .5)

    def test_empty(self):
        x, faces, edges, alpha = geom.extract_p1_tetrahedral_interface(
            np.empty((0, 3)), np.empty((0, 4), dtype=np.int64), [], 0)
        self.assertEqual(x.shape, (0, 3)); self.assertEqual(faces.shape, (0, 3))


class AnnularPatchTests(unittest.TestCase):
    def setUp(self):
        a = np.arange(25)*2*np.pi/25
        b = np.arange(154)*2*np.pi/154 + .37
        self.x = np.r_[np.c_[np.cos(a), np.sin(a), a*0],
                       np.c_[.98*np.cos(b), .98*np.sin(b), b*0+.05]]
        self.a = np.arange(25)
        self.b = np.arange(25, 179)

    def test_unequal_loops_form_oriented_annulus(self):
        faces, report = geom.zipper_annulus(
            self.x, self.a, self.b, max_edge=.32)
        self.assertEqual(len(faces), 179)
        self.assertTrue(report["oriented_boundary_exact"])
        self.assertFalse(report["physical_admission"])
        edges = {}
        for face in faces:
            for i, j in zip(face, np.roll(face, -1)):
                edges.setdefault(tuple(sorted((int(i), int(j)))), []).append((int(i), int(j)))
        self.assertEqual(len(self.x)-len(edges)+len(faces), 0)
        actual = {v[0] for v in edges.values() if len(v) == 1}
        expected = set(zip(self.a, np.roll(self.a, -1)))
        expected |= set(zip(np.roll(self.b, -1), self.b))
        self.assertEqual(actual, expected)
        self.assertGreater(report["minimum_quality"], .18)
        self.assertLessEqual(report["maximum_edge_m"], .32)

    def test_cyclic_reindexing_has_same_optimum(self):
        f, r = geom.zipper_annulus(self.x, self.a, self.b, max_edge=.32)
        g, s = geom.zipper_annulus(self.x, self.a, np.roll(self.b, 17), max_edge=.32)
        self.assertAlmostEqual(r["score"], s["score"], places=12)
        canonical = lambda faces: {min(tuple(np.roll(t, k)) for k in range(3)) for t in faces}
        self.assertEqual(canonical(f), canonical(g))

    def test_impossible_edge_cap_rejected(self):
        with self.assertRaisesRegex(ValueError, "no strip"):
            geom.zipper_annulus(self.x, self.a, self.b, max_edge=.001)

    def test_invalid_indices_rejected(self):
        for a, b in [([0, 1, 1], [25, 26, 27]),
                     ([0, 1, 2], [2, 3, 4]),
                     ([0., 1., 2.], [25, 26, 27]),
                     ([-1, 1, 2], [25, 26, 27]),
                     ([0, 1, 999], [25, 26, 27])]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                geom.zipper_annulus(self.x, a, b)

    def test_invalid_geometry_and_limits_rejected(self):
        bad = self.x.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            geom.zipper_annulus(bad, self.a, self.b)
        for limit in [-1, 0, np.inf, np.nan]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                geom.zipper_annulus(self.x, self.a, self.b, max_edge=limit)

    def test_collinear_surface_rejected(self):
        x = np.c_[np.arange(6), np.zeros((6, 2))]
        with self.assertRaisesRegex(ValueError, "no strip"):
            geom.zipper_annulus(x, [0, 1, 2], [3, 4, 5])


class LayerPrismTests(unittest.TestCase):
    def test_all_permutations_match_native_sorted_id_signs(self):
        import itertools
        faces = np.asarray(list(itertools.permutations((0, 1, 2))), dtype=np.int64)
        nodes, layers, signs = geom.ordered_layer_prism_tets(faces)
        np.testing.assert_array_equal(nodes[:, 0], nodes[:, 1])
        np.testing.assert_array_equal(layers[:, 1], layers[:, 0] + 1)
        expected = np.array([-1, 1, 1, -1, -1, 1])
        np.testing.assert_array_equal(signs[:, 0, 0], expected)
        np.testing.assert_array_equal(signs[:, 0], signs[:, 1])
        # Compare against a direct native-template decomposition, not a second
        # inversion-count implementation.
        o = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
        native_nodes = np.array([[0, 1, 2, 2], [0, 1, 1, 2], [0, 0, 1, 2]])
        for i, face in enumerate(faces):
            inward = np.cross(o[face[1]]-o[face[0]], o[face[2]]-o[face[0]])
            xyz = np.stack((o, o-.3*inward, o-.9*inward))
            p = xyz[layers[i], nodes[i]]
            det, _ = geom.tetra_determinants_and_gradients(p)
            self.assertTrue(np.all(signs[i]*det > 0))
            np.testing.assert_array_equal(nodes[i, 0], native_nodes)
            self.assertAlmostEqual(float(np.abs(det[0]).sum()/6), .15)
            self.assertAlmostEqual(float(np.abs(det[1]).sum()/6), .30)

    def test_empty_and_invalid_faces(self):
        n, l, s = geom.ordered_layer_prism_tets(np.empty((0, 3), dtype=np.int64))
        self.assertEqual(n.shape, (0, 2, 3, 4))
        self.assertEqual(l.shape, n.shape)
        self.assertEqual(s.shape, (0, 2, 3))
        for f in [[[0, 1, 1]], [[-1, 2, 3]], [[0., 1., 2.]], [[0, 1]]]:
            with self.subTest(f=f), self.assertRaises(ValueError):
                geom.ordered_layer_prism_tets(f)

    def test_batch_signed_and_absolute_volume_gradients(self):
        rng = np.random.default_rng(20261023)
        p = rng.normal(size=(24, 4, 3))*.07
        d = rng.normal(size=p.shape)
        d /= np.linalg.norm(d, axis=(1, 2))[:, None, None]
        det, gradient = geom.tetra_determinants_and_gradients(p)
        h = 2e-7
        plus, _ = geom.tetra_determinants_and_gradients(p+h*d)
        minus, _ = geom.tetra_determinants_and_gradients(p-h*d)
        np.testing.assert_allclose((plus-minus)/(2*h), np.sum(gradient*d, axis=(1, 2)),
                                   rtol=2e-6, atol=1e-12)
        np.testing.assert_allclose((np.abs(plus)-np.abs(minus))/(12*h),
                                   np.sign(det)*np.sum(gradient*d, axis=(1, 2))/6,
                                   rtol=2e-6, atol=1e-12)
        np.testing.assert_allclose(gradient.sum(axis=-2), 0., atol=1e-17)
        for invalid in [np.zeros((3, 3)), np.full((4, 3), np.nan)]:
            with self.assertRaises(ValueError):
                geom.tetra_determinants_and_gradients(invalid)

class SurfaceEdgeSplitTests(unittest.TestCase):
    def test_shared_edge_is_conforming_and_untouched_face_exact(self):
        x=np.array([[0.,0,0],[1,0,0],[0,1,0],[1,1,0],[2,0,0],[2,1,0]])
        f=np.array([[0,1,2],[1,3,2],[1,4,3]])
        y,g,parent,edge,w=geom.split_surface_edges(x,f,{(1,2):.3})
        self.assertEqual(len(y),7)
        self.assertEqual(len(g),5)
        np.testing.assert_array_equal(g[parent==2],f[2:3])
        np.testing.assert_array_equal(edge,[[1,2]])
        np.testing.assert_allclose(y[6],w[0]@x[edge[0]])
        incidence={}
        for tri in g:
            for a,b in zip(tri,np.roll(tri,-1)):
                incidence.setdefault(tuple(sorted((int(a),int(b)))),[]).append((int(a),int(b)))
        self.assertEqual(len(incidence[(1,6)]),2)
        self.assertEqual(len(incidence[(2,6)]),2)
        for rows in incidence.values():
            self.assertLessEqual(len(rows),2)
            if len(rows)==2:self.assertEqual(rows[0],rows[1][::-1])

    def test_one_two_and_three_edge_cuts_preserve_signed_area(self):
        x=np.array([[0.,0,0],[1.7,.2,.3],[.1,1.2,-.4]])
        normal=np.cross(x[1]-x[0],x[2]-x[0])
        for cuts in [{(0,1):.31},{(0,1):.31,(1,2):.62},
                     {(0,1):.31,(1,2):.62,(2,0):.43}]:
            y,g,parent,edges,w=geom.split_surface_edges(x,np.array([[0,1,2]]),cuts)
            self.assertEqual(len(g),1+len(cuts))
            v=y[g];n=np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0])
            self.assertTrue(np.all(n@normal>0))
            np.testing.assert_allclose(n.sum(0),normal,rtol=1e-14,atol=1e-14)
            np.testing.assert_allclose(w.sum(1),1.,atol=1e-15)
            np.testing.assert_allclose(y[3:],np.sum(x[edges]*w[:,:,None],axis=1))
            self.assertTrue(np.all(parent==0))

    def test_reversed_parameters_and_invalid_edges(self):
        x=np.array([[0.,0,0],[1,0,0],[0,1,0],[2,2,0]])
        f=np.array([[0,1,2]])
        a=geom.split_surface_edges(x,f,{(1,0):.7})
        b=geom.split_surface_edges(x,f,{(0,1):.3})
        for i in range(5):np.testing.assert_allclose(a[i],b[i])
        for cuts in [{(0,1):0.},{(0,1):1.},{(0,1):np.nan},{(0,9):.5},
                     {(0,0):.5},{(2,3):.5},{(0,1):.3,(1,0):.3}]:
            with self.subTest(cuts=cuts),self.assertRaises(ValueError):
                geom.split_surface_edges(x,f,cuts)
        y,g,parent,edges,w=geom.split_surface_edges(x,f,{})
        np.testing.assert_array_equal(y,x)
        np.testing.assert_array_equal(g,f)
        self.assertEqual(edges.shape,(0,2))


class TransverseSectionTests(unittest.TestCase):
    @staticmethod
    def tetra():
        return (np.array([[0.,0,1.],[-1.,-1.,-1.],[1.,-1.,-1.],[0.,1.,-1.]]),
                np.array([[0,2,1],[0,3,2],[0,1,3],[1,2,3]]))

    def test_closed_and_open_sections_use_edge_identity(self):
        x,f=self.tetra()
        r=geom.transverse_surface_section(x,f,0.)
        self.assertEqual(r["closed_loops"],1)
        self.assertEqual(len(r["components"]),1)
        self.assertEqual(r["degree_histogram"],{2:3})
        self.assertEqual(len(r["source_face_rows"]),3)
        np.testing.assert_array_equal(r["points"][:,2],0.)
        r=geom.transverse_surface_section(x,f[1:],0.)
        self.assertEqual(r["closed_loops"],0)
        self.assertEqual(r["degree_histogram"],{1:2,2:1})

    def test_coincident_disconnected_shells_are_not_welded(self):
        x,f=self.tetra()
        r=geom.transverse_surface_section(np.vstack((x,x)),np.vstack((f,f+4)),0.)
        self.assertEqual(r["closed_loops"],2)
        self.assertEqual(len(r["components"]),2)
        self.assertEqual(len(r["points"]),6)
        # Arbitrarily tiny coordinates cannot lose topology to quantization.
        r=geom.transverse_surface_section(x*1e-15,f,0.)
        self.assertEqual(r["closed_loops"],1)

    def test_empty_and_ambiguous_planes(self):
        x,f=self.tetra()
        self.assertEqual(geom.transverse_surface_section(x,f,2.)["closed_loops"],0)
        for value,axis in [(1.,2),(np.nan,2),(0.,3)]:
            with self.assertRaises(ValueError):
                geom.transverse_surface_section(x,f,value,axis=axis)



class TetrahedralRegionTests(unittest.TestCase):
    def test_parent_identity_join_and_complete_refined_parent(self):
        result = geom.select_tetrahedra_by_surface_parent(
            np.array([48, 0, 142, 48]), np.array([0]),
            np.array([0, 0, 48, 142, 48]))
        np.testing.assert_array_equal(result, [2, 4])
        for parents, rows, tets in [
            ([48], [1], [48]), ([48], [0], [0]), ([48.0], [0], [48])]:
            with self.assertRaises(ValueError):
                geom.select_tetrahedra_by_surface_parent(
                    np.array(parents), np.array(rows), np.array(tets))

    def test_two_material_positive_tets_preserve_interface(self):
        t = np.array([[0, 1, 2, 3], [0, 2, 1, 4]])
        r = geom.tetrahedral_region_boundary(t, np.array([1, 2]))
        self.assertEqual(len(r["exterior_faces"]), 6)
        np.testing.assert_array_equal(r["interface_faces"], [[0, 2, 1]])
        np.testing.assert_array_equal(r["interface_tet_pairs"], [[0, 1]])
        self.assertEqual(len(geom.tetrahedral_region_boundary(t, np.array([1, 1]))["interface_faces"]), 0)

    def test_bad_volume_topology_is_rejected(self):
        for t in [
            [[0, 1, 2, 3], [0, 1, 2, 4]],
            [[0, 1, 2, 3], [0, 2, 1, 4], [0, 1, 2, 5]],
            [[0, 1, 2, 3], [0, 1, 2, 3]],
            [[0, 0, 2, 3]],
            [[0, 1, 2, 3], [0, 1, 4, 5]],
            [[0, 1, 2, 3], [0, 4, 5, 6]]]:
            with self.assertRaises(ValueError):
                geom.tetrahedral_region_boundary(np.array(t), np.ones(len(t), dtype=int))

    def test_empty_region(self):
        r = geom.tetrahedral_region_boundary(np.empty((0, 4),dtype=int), np.empty(0,dtype=int))
        self.assertEqual(r["exterior_faces"].shape, (0, 3))
        self.assertEqual(r["interface_tet_pairs"].shape, (0, 2))

partition = geom.partition_tetrahedral_repair_batch

class RepairBatchTests(unittest.TestCase):
    def test_fixed_stitch_keeps_independent_regions(self):
        t = np.array([[0,1,2,3], [0,1,4,5], [6,7,8,9]])
        groups, shared = partition(t, [[0], [1], [2]], [[2], [4], [6]])
        self.assertEqual([g["selected_global_tet_ids"].tolist() for g in groups], [[0],[1],[2]])
        self.assertEqual(shared[0]["fixed_global_node_ids"].tolist(), [0,1])
        self.assertEqual(len(shared), 1)

    def test_moved_vertex_claims_full_star_and_merges_transitively(self):
        t = np.array([[0,1,2,3], [0,4,5,6], [4,7,8,9], [10,11,12,13]])
        groups, _ = partition(t, [[0], [2], [3]], [[0], [4], [10]])
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["selected_global_tet_ids"].tolist(), [0,1,2])
        self.assertEqual(groups[0]["seed_components"].tolist(), [0,1])
        self.assertEqual(groups[0]["movable_global_node_ids"].tolist(), [0,4])

    def test_topology_uses_ids_without_geometric_welding(self):
        t = np.array([[20,21,22,23], [24,25,26,27]])
        groups, shared = partition(t, [[1], [0]], [[27], [20]])
        self.assertEqual([g["selected_global_tet_ids"].tolist() for g in groups], [[0],[1]])
        self.assertEqual(groups[0]["seed_components"].tolist(), [1])
        self.assertFalse(shared)

    def test_empty_movable_regions_and_empty_batch(self):
        t = np.array([[0,1,2,3]])
        groups, _ = partition(t, [[0]], [[]])
        self.assertEqual(groups[0]["movable_global_node_ids"].size, 0)
        self.assertEqual(partition(t, [], []), ([], []))

    def test_invalid_identity_is_rejected(self):
        t = np.array([[0,1,2,3]])
        for selected, moved in [([[1]], [[0]]), ([[0]], [[4]]), ([[0.0]], [[0]]),
                                 ([[0]], [[-1]]), ([[0]], [[2**64-1]]),
                                 ([[0]], [[0],[1]]), ([[]], [[]])]:
            with self.subTest(selected=selected, moved=moved):
                with self.assertRaises(ValueError):
                    partition(t, selected, moved)
        with self.assertRaises(ValueError):
            partition(np.array([[0,1,1,3]]), [[0]], [[0]])




class TetBoundaryCeilingTests(unittest.TestCase):
    def test_ceiling_bounds_all_four_incident_face_tets(self):
        rng = np.random.default_rng(200)
        for _ in range(100):
            x = rng.normal(size=(4,3))
            det = abs(np.linalg.det((x[1:]-x[0]).T))
            longest = max(np.linalg.norm(x[a]-x[b]) for a in range(4) for b in range(a))
            actual = det/longest**3
            faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
            ceiling = geom.boundary_tetrahedron_quality_ceiling(x,faces)
            self.assertTrue(np.all(actual <= ceiling+1e-14))

    def test_local_scale_metric_and_sliver_impossibility(self):
        x = np.array([[0.,0.,0.],[1.,0.,0.],[.5,1e-8,0.]])
        face = np.array([[0,1,2]])
        ceiling = geom.boundary_tetrahedron_quality_ceiling(x,face)
        self.assertLess(ceiling[0],1e-5)
        np.testing.assert_allclose(
            geom.boundary_tetrahedron_quality_ceiling(x*1000,face),ceiling,rtol=1e-14)
        # A remote large triangle cannot alter the local bound.
        y = np.r_[x,[[0,0,10],[100,0,10],[0,100,10]]]
        np.testing.assert_array_equal(
            geom.boundary_tetrahedron_quality_ceiling(y,np.r_[face,[[3,4,5]]])[:1],ceiling)

    def test_invalid_and_collapsed_faces(self):
        x = np.array([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
        self.assertEqual(geom.boundary_tetrahedron_quality_ceiling(x,np.array([[0,1,2]]))[0],0)
        self.assertEqual(len(geom.boundary_tetrahedron_quality_ceiling(x,np.empty((0,3),int))),0)
        for f in (np.array([[0.,1.,2.]]),np.array([[-1,1,2]]),np.array([[0,1,3]])):
            with self.assertRaises(ValueError):
                geom.boundary_tetrahedron_quality_ceiling(x,f)


if __name__ == "__main__":
    unittest.main()
