"""Exact predicate regressions; no simulator or physical stepping."""
from __future__ import annotations
import copy
from fractions import Fraction
import math
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from numilab_human import cardiac_cavity_intersections as audit
from numilab_human.cardiac_cavity_geometry import extract_cavity_surfaces
from numilab_human.model import ImportError as HumanImportError


def tetra(name, offset=(0., 0., 0.), scale=1.):
    vertices = [[offset[k]+scale*p[k] for k in range(3)] for p in ((0,0,0),(1,0,0),(0,1,0),(0,0,1))]
    return {'source_id': name, 'exact_coordinate_quotient': {'vertices_m': vertices,
            'triangles': [[0,2,1], [0,1,3], [0,3,2], [1,2,3]]}}


def run(*chambers):
    return audit.audit_cavity_intersections({'chambers': list(chambers)})


class PredicateTests(unittest.TestCase):
    def test_prepared_point_location_enforces_resource_bound_before_indexing(self):
        records = audit._records([(0,0,0),(1,0,0),(0,1,0),(0,0,1)],
                                 [(0,2,1),(0,1,3),(0,3,2),(1,2,3)])
        with patch.object(audit, "MAX_PREPARED_POINT_LOCATION_FACES", 4):
            prepared = audit.prepare_point_location(records)
            self.assertEqual(audit.point_location_prepared_indexed(
                (Fraction(1,8),)*3, prepared)["location"], "inside")
            with self.assertRaisesRegex(HumanImportError, "bounded face count"):
                audit.prepare_point_location(records + [None])
        self.assertGreaterEqual(audit.MAX_PREPARED_POINT_LOCATION_FACES, 220_672)
        self.assertEqual(audit.MAX_PREPARED_POINT_LOCATION_FACES, 262_144)

    def test_missing_prepared_records_cannot_pass_as_empty_geometry(self):
        for first, second in ((None, None), (None, []), ([], None)):
            with self.subTest(first=first, second=second):
                with self.assertRaisesRegex(HumanImportError, "missing triangle records"):
                    audit._audit_pair(first, second, same_surface=True)
        self.assertEqual(audit._audit_pair([], [], same_surface=True)["count"], 0)

    def pair(self, vertices, faces):
        records = audit._records(vertices, faces)
        return audit._audit_pair(records, records, same_surface=True)

    def test_selected_faces_match_full_audit_for_either_record_order(self):
        vertices = [(0, 0, 0), (4, 0, 0), (0, 4, 0),
                    (1, 1, -1), (1, 1, 1), (2, 1, 0),
                    (20, 0, 0), (21, 0, 0), (20, 1, 0)]
        records = audit._records(vertices, [(0, 1, 2), (3, 4, 5), (6, 7, 8)])
        records = [(*row[:3], face_id, row[4])
                   for row, face_id in zip(records, (3323, 7167, 9900))]
        full = audit._audit_pair(records, records, same_surface=True)
        self.assertEqual(full["triangle_pairs"], [[3323, 7167]])
        for order in (records, records[::-1]):
            for selected in ({3323}, {7167}, {3323, 7167}, {9900}, set()):
                with self.subTest(selected=selected, reversed=order is not records):
                    expected = [pair for pair in full["triangle_pairs"]
                                if selected.intersection(pair)]
                    result = audit._audit_self_pairs_touching_faces(order, selected)
                    self.assertEqual(result["triangle_pairs"], expected)
                    self.assertEqual(result["count"], len(expected))

    def test_selected_faces_preserve_shared_vertex_identity(self):
        vertices = [(0, 0, 0), (2, 0, 0), (0, 2, 0),
                    (0, -2, 0), (1, 1, 0)]
        records = audit._records(vertices, [(0, 1, 2), (1, 0, 3), (1, 0, 4)])
        full = audit._audit_pair(records, records, same_surface=True)
        for selected in ({0}, {1}, {2}, {0, 1, 2}):
            expected = [pair for pair in full["triangle_pairs"]
                        if selected.intersection(pair)]
            result = audit._audit_self_pairs_touching_faces(records, selected)
            self.assertEqual(result["triangle_pairs"], expected)
        allowed = audit._audit_self_pairs_touching_faces(records[:2], {0})
        self.assertEqual(allowed["count"], 0)
        self.assertEqual(allowed["allowed_shared_vertex_or_edge_pairs"], 1)

    def test_selected_faces_reject_missing_or_ambiguous_records(self):
        with self.assertRaisesRegex(HumanImportError, "missing triangle records"):
            audit._audit_self_pairs_touching_faces(None, {0})
        with self.assertRaisesRegex(HumanImportError, "selected face ID missing"):
            audit._audit_self_pairs_touching_faces([], {0})
        records = audit._records([(0, 0, 0), (1, 0, 0), (0, 1, 0)], [(0, 1, 2)])
        with self.assertRaisesRegex(HumanImportError, "duplicate face IDs"):
            audit._audit_self_pairs_touching_faces(records + records, {0})
        self.assertEqual(audit._audit_self_pairs_touching_faces([], set())["count"], 0)

    def test_selected_scope_must_cover_required_witnesses_before_indexing(self):
        records = audit._records([(0,0,0),(4,0,0),(0,4,0),
                                  (1,1,-1),(1,1,1),(2,1,0),
                                  (20,0,0),(21,0,0),(20,1,0)],
                                 [(0,1,2),(3,4,5),(6,7,8)])
        with patch.object(audit, "_prepare_surface_aabb", side_effect=AssertionError):
            with self.assertRaisesRegex(HumanImportError, "witness omitted"):
                audit._audit_self_pairs_touching_faces(
                    records, {2}, required_witness_pairs=[(0,1)])
        for selected in ({0}, {1}):
            r = audit._audit_self_pairs_touching_faces(
                records, selected, required_witness_pairs=[(0,1),(1,0)])
            self.assertEqual(r["triangle_pairs"], [[0,1]])
            self.assertFalse(r["complete_surface_audit"])
            self.assertEqual(r["selected_face_count"], 1)
            self.assertEqual(r["complete_surface_face_count"], 3)
            self.assertEqual(r["covered_required_witness_pairs"], 1)
        self.assertTrue(audit._audit_self_pairs_touching_faces(
            records, {0,1,2})["complete_surface_audit"])

    def test_required_witness_ids_must_be_valid_even_for_clear_geometry(self):
        records = audit._records([(0,0,0),(1,0,0),(0,1,0),
                                  (10,0,0),(11,0,0),(10,1,0)],
                                 [(0,1,2),(3,4,5)])
        # A repaired witness may now be clear: coverage never forces a hit.
        result = audit._audit_self_pairs_touching_faces(
            records, {0}, required_witness_pairs=[(0,1)])
        self.assertEqual(result["count"], 0)
        for bad in ([(0,3)], [(0,0)], [(0,)], [(0,1,2)], [(False,1)],
                    [(0.,1)], [None]):
            with self.subTest(bad=bad), self.assertRaises(HumanImportError):
                audit._audit_self_pairs_touching_faces(
                    records, {0}, required_witness_pairs=bad)

    def test_shared_edge_contact_allowed_but_same_side_overlap_rejected(self):
        vertices = [(0,0,0),(2,0,0),(0,2,0),(0,-2,0),(1,1,0)]
        self.assertEqual(self.pair(vertices, [(0,1,2),(1,0,3)])['count'], 0)
        self.assertEqual(self.pair(vertices, [(0,1,2),(1,0,4)])['triangle_pairs'], [[0,1]])

    def test_shared_vertex_is_not_blanket_overlap_exemption(self):
        vertices = [(0,0,0),(2,0,0),(0,2,0),(-2,0,0),(0,-2,0),(2,1,0),(1,2,0)]
        self.assertEqual(self.pair(vertices, [(0,1,2),(0,3,4)])['count'], 0)
        self.assertEqual(self.pair(vertices, [(0,1,2),(0,5,6)])['count'], 1)

    def test_duplicate_or_unrelated_coincident_faces_rejected(self):
        vertices = [(0,0,0),(2,0,0),(0,2,0)]
        self.assertEqual(self.pair(vertices, [(0,1,2),(0,2,1)])['count'], 1)
        self.assertEqual(self.pair(vertices+vertices, [(0,1,2),(3,4,5)])['count'], 1)

    def test_transverse_and_coplanar_crossing_without_contained_vertices(self):
        first = ((0,0,0),(4,0,0),(0,4,0))
        transverse = ((1,1,-1),(1,1,1),(2,1,0))
        self.assertTrue(audit.triangle_intersection_points(first, transverse))
        # Star-of-David triangles: intersection polygon comes from edge crossings.
        a = ((0,3,0),(-3,-2,0),(3,-2,0))
        b = ((0,-3,0),(-3,2,0),(3,2,0))
        self.assertEqual(len(audit.triangle_intersection_points(a, b)), 6)

    def test_pair_local_lattice_preserves_huge_integer_and_fraction_intersections(self):
        first = ((0, 0, 0), (4, 0, 0), (0, 4, 0))
        second = ((1, 1, -1), (1, 1, 1), (2, 1, 0))
        expected = audit.triangle_intersection_points(first, second)
        self.assertTrue(expected)

        scale = 1 << 2860
        offset = (1 << 2900, -(1 << 2899), 1 << 2888)
        def huge(point):
            return tuple(offset[axis] + scale * point[axis] for axis in range(3))
        huge_first, huge_second = tuple(map(huge, first)), tuple(map(huge, second))
        actual = audit.triangle_intersection_points(huge_first, huge_second)
        legacy = audit._triangle_intersection_points_integer(huge_first, huge_second)
        self.assertEqual(actual, legacy)
        self.assertEqual(actual, [tuple(offset[k] + scale * point[k] for k in range(3))
                                  for point in expected])
        normalized, _transform = audit._integer_lattice_normalization(huge_first + huge_second)
        self.assertLess(max(abs(value).bit_length()
                            for point in normalized for value in point), 5)

        rational_scale = Fraction(5, 13)
        rational_offset = (Fraction(1, 7), Fraction(-2, 9), Fraction(3, 11))
        def rational(point):
            return tuple(rational_offset[axis] + rational_scale * point[axis]
                         for axis in range(3))
        rational_actual = audit.triangle_intersection_points(
            tuple(map(rational, first)), tuple(map(rational, second)))
        self.assertEqual(rational_actual,
                         [tuple(rational_offset[k] + rational_scale * point[k]
                                for k in range(3)) for point in expected])

    def test_pair_local_lattice_preserves_coplanar_and_disjoint_results(self):
        first = ((0, 3, 0), (-3, -2, 0), (3, -2, 0))
        second = ((0, -3, 0), (-3, 2, 0), (3, 2, 0))
        expected = audit.triangle_intersection_points(first, second)
        scale = 1 << 2870
        offset = (1 << 2900, 1 << 2899, -(1 << 2880))
        transform = lambda point: tuple(offset[k] + scale * point[k] for k in range(3))
        huge_first, huge_second = tuple(map(transform, first)), tuple(map(transform, second))
        actual = audit.triangle_intersection_points(huge_first, huge_second)
        self.assertEqual(actual,
                         audit._triangle_intersection_points_integer(huge_first, huge_second))
        self.assertEqual(actual, [tuple(offset[k] + scale * point[k] for k in range(3))
                                  for point in expected])
        disjoint = tuple(tuple(point[k] + (100 if k == 2 else 0) for k in range(3))
                         for point in first)
        self.assertEqual(audit.triangle_intersection_points(first, disjoint), [])

    def test_exact_one_bit_separation_is_not_tolerance_contact(self):
        first = ((0,0,0),(1,0,0),(0,1,0))
        second = ((0,0,1),(1,0,1),(0,1,1))
        self.assertFalse(audit.triangle_intersection_points(first, second))
        a, b = tetra('a'), tetra('b', (math.nextafter(1., math.inf),0.,0.))
        self.assertTrue(run(a,b)['all_domains_disjoint'])
        self.assertFalse(run(a,tetra('b',(1.,0.,0.)))['all_domains_disjoint'])

    def test_float32_shared_face_keys_use_one_exact_integer_lattice(self):
        f32 = lambda value: struct.unpack('<f', struct.pack('<f', value))[0]
        vertices = [(f32(.1),f32(.2),f32(.3)), (f32(.8),f32(.2),f32(.3)),
                    (f32(.1),f32(.9),f32(.3)), (f32(.5),f32(.5),f32(.3))]
        forward = audit.float32_triangle_lattice_key(vertices, (0,1,2))
        reverse = audit.float32_triangle_lattice_key(vertices, (2,1,0))
        self.assertEqual(forward, reverse)
        second_mesh = [vertices[index] for index in (2,0,1)]
        self.assertEqual(forward, audit.float32_triangle_lattice_key(second_mesh, (0,1,2)))
        self.assertNotEqual(forward, audit.float32_triangle_lattice_key(vertices, (0,1,3)))
        with self.assertRaisesRegex(HumanImportError, 'exact Float32'):
            audit.float32_point_lattice_key((.1,.2,.3))

    def test_float32_patch_membership_uses_one_scalar_denominator(self):
        f32 = lambda value: struct.unpack('<f', struct.pack('<f', value))[0]
        vertices = [(f32(0),f32(0),f32(0)), (f32(1),f32(0),f32(0)),
                    (f32(0),f32(1),f32(0))]
        triangle = tuple(audit.float32_point_lattice_key(vertices[index]) for index in (0,1,2))
        lattice = 1 << 149
        self.assertTrue(audit.float32_lattice_point_on_triangle((lattice,lattice,0),4,triangle))
        self.assertFalse(audit.float32_lattice_point_on_triangle((3*lattice,3*lattice,0),4,triangle))
        with self.assertRaisesRegex(HumanImportError, 'scalar'):
            audit.float32_lattice_point_on_triangle((lattice,lattice,0),(4,4,4),triangle)

    def test_exact_ray_retries_vertex_hit_and_reports_boundary(self):
        t = tetra('test')['exact_coordinate_quotient']
        records = audit._records([tuple(map(int,p)) for p in t['vertices_m']],t['triangles'])
        outside = audit.point_location((-1,-1,-1), records)
        self.assertEqual(outside['location'], 'outside')
        self.assertGreater(outside['attempts'], 1)
        self.assertEqual(audit.point_location((0,0,0), records)['location'], 'boundary')
        self.assertEqual(audit.point_location((Fraction(1,8),)*3, records)['location'], 'inside')

    def test_degenerate_triangles_are_not_admitted(self):
        with self.assertRaisesRegex(HumanImportError,'degenerate'):
            audit.triangle_intersection_points(((0,0,0),(1,0,0),(2,0,0)), ((0,0,1),(1,0,1),(0,1,1)))



class PreparedPointLocationTests(unittest.TestCase):
    @staticmethod
    def tetra_records(scale=12):
        vertices = ((0, 0, 0), (scale, 0, 0), (0, scale, 0), (0, 0, scale))
        faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        return audit._records(vertices, faces)

    def assert_legacy_parity(self, records, points):
        prepared = audit.prepare_point_location(records)
        for point in points:
            with self.subTest(point=point):
                self.assertEqual(audit.point_location(point, records),
                                 audit.point_location_prepared(point, prepared))
        return prepared

    def test_full_result_inside_outside_boundary_edge_and_vertex(self):
        records = self.tetra_records()
        edge = (Fraction(6), Fraction(6), Fraction(0))
        prepared = self.assert_legacy_parity(
            records, ((1, 1, 1), (-1, -1, -1), (0, 0, 0), edge, (12, 0, 0)))
        self.assertGreater(audit.point_location((-1, -1, -1), records)["attempts"], 1)
        self.assertIn(2, prepared._fallback_ray_data)

    def test_coplanar_ambiguity_behind_forward_aabb_is_preserved(self):
        # The first face lies in x=y, contains the origin in its plane, and is
        # wholly behind the query point. The primary (1,1,1) ray is coplanar.
        # Legacy point_location retries that ray even though no triangle AABB
        # is reachable in the positive direction.
        vertices = ((-1, -1, -1), (-2, -2, -1), (-1, -1, -2), (-3, -4, -3))
        faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        records = audit._records(vertices, faces)
        self.assertTrue(all(any(bound < 0 for bound in row[2]) for row in records))
        prepared = audit.prepare_point_location(records)
        expected = audit.point_location((0, 0, 0), records)
        self.assertEqual(expected["attempts"], 2)
        self.assertEqual(audit.point_location_prepared((0, 0, 0), prepared), expected)
        self.assertEqual(audit.point_location_prepared_indexed((0, 0, 0), prepared), expected)
        self.assertIn(2, prepared._fallback_ray_data)


    def test_all_sixty_four_coplanar_rays_reach_indeterminate(self):
        records = []
        for attempt in range(1, 65):
            triangle = ((-1, -attempt, 0), (-2, -2 * attempt, 0),
                        (-1, -attempt, -1))
            lower = tuple(min(point[axis] for point in triangle) for axis in range(3))
            upper = tuple(max(point[axis] for point in triangle) for axis in range(3))
            vertex_base = 3 * (attempt - 1)
            records.append((triangle, lower, upper, attempt - 1,
                            (vertex_base, vertex_base + 1, vertex_base + 2)))
        prepared = audit.prepare_point_location(records)
        expected = audit.point_location((0, 0, 0), records)
        self.assertEqual(expected, {"location": "indeterminate", "ray": None,
                                    "crossings": None, "attempts": 64})
        indexed_prepared = audit.prepare_point_location(records)
        self.assertEqual(audit.point_location_prepared((0, 0, 0), prepared), expected)
        self.assertEqual(audit.point_location_prepared_indexed((0, 0, 0), indexed_prepared), expected)
        self.assertEqual(set(prepared._fallback_ray_data), set(range(2, 65)))
        self.assertEqual(set(indexed_prepared._fallback_ray_data), set(range(2, 65)))


    def test_large_float32_lattice_integers_remain_exact(self):
        scale = 1 << 149
        records = self.tetra_records(scale)
        points = ((scale // 8, scale // 8, scale // 8),
                  (scale, 0, 0), (2 * scale, 2 * scale, 2 * scale))
        self.assert_legacy_parity(records, points)

    def test_local_lattices_bound_huge_surface_and_fraction_query_arithmetic(self):
        scale = 1 << 2800
        offset = (1 << 2900, -(1 << 2898), 1 << 2887)
        base_vertices = ((0, 0, 0), (12, 0, 0), (0, 12, 0), (0, 0, 12))
        vertices = tuple(tuple(offset[axis] + scale * point[axis] for axis in range(3))
                         for point in base_vertices)
        faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        records = audit._records(vertices, faces)
        prepared = audit.prepare_point_location(records)
        self.assertLess(max(abs(value).bit_length()
                            for triangle in prepared.triangles
                            for point in triangle for value in point), 5)
        for point, expected in (
                (tuple(offset[k] + scale for k in range(3)), "inside"),
                (tuple(offset[k] + 20 * scale for k in range(3)), "outside"),
                (offset, "boundary"),
                (tuple(Fraction(offset[k]) + Fraction(scale, 8) for k in range(3)), "inside")):
            with self.subTest(expected=expected):
                result = audit.point_location_prepared(point, prepared)
                self.assertEqual(result["location"], expected)
        self.assertEqual(
            audit.point_location(tuple(offset[k] + scale for k in range(3)), records)["location"],
            "inside")

    def test_snapshot_is_immutable_after_caller_mutates_records(self):
        records = self.tetra_records()
        expected = audit.point_location((1, 1, 1), records)
        mutable_records = [
            ([list(point) for point in row[0]], list(row[1]), list(row[2]), row[3], list(row[4]))
            for row in records
        ]
        prepared = audit.prepare_point_location(mutable_records)
        identity = prepared.record_identity_sha256
        mutable_records[0][0][0][0] = 500
        mutable_records[0][1][0] = -500
        mutable_records[0][4][0] = 99
        self.assertEqual(audit.point_location_prepared((1, 1, 1), prepared), expected)
        self.assertEqual(prepared.record_identity_sha256, identity)
        self.assertIsInstance(prepared.records, tuple)
        self.assertIsInstance(prepared.records[0][0], tuple)
        self.assertIsInstance(prepared.records[0][0][0], tuple)
        self.assertEqual(prepared.records, tuple(records))

    def test_empty_degenerate_malformed_and_inexact_inputs_reject(self):
        with self.assertRaisesRegex(HumanImportError, "non-empty"):
            audit.prepare_point_location([])
        degenerate = [(((0, 0, 0), (1, 0, 0), (2, 0, 0)),
                       (0, 0, 0), (2, 0, 0), 0, (0, 1, 2))]
        with self.assertRaisesRegex(HumanImportError, "degenerate"):
            audit.prepare_point_location(degenerate)
        records = self.tetra_records()
        malformed_bounds = list(records)
        row = records[0]
        malformed_bounds[0] = (row[0], (row[1][0] - 1, *row[1][1:]), row[2], row[3], row[4])
        with self.assertRaisesRegex(HumanImportError, "bounds"):
            audit.prepare_point_location(malformed_bounds)
        prepared = audit.prepare_point_location(records)
        with self.assertRaisesRegex(HumanImportError, "exact integer or Fraction"):
            audit.point_location_prepared((0.25, 0, 0), prepared)
        with self.assertRaisesRegex(HumanImportError, "snapshot"):
            audit.point_location_prepared((0, 0, 0), records)





class IndexedPreparedPointLocationTests(unittest.TestCase):
    @staticmethod
    def records():
        scale = 1 << 900
        offset = (1 << 2900, -(1 << 2890), 1 << 2880)
        base = ((0, 0, 0), (12, 0, 0), (0, 12, 0), (0, 0, 12))
        vertices = tuple(tuple(offset[axis] + scale * point[axis] for axis in range(3))
                         for point in base)
        faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        return audit._records(vertices, faces), offset, scale

    def test_exact_aabb_candidates_match_exhaustive_closed_box_oracle(self):
        records, offset, scale = self.records()
        extra_records = []
        for index in range(12):
            x = 24 + 2 * index
            triangle = tuple(
                tuple(offset[axis] + scale * point[axis] for axis in range(3))
                for point in ((x, 0, 0), (x + 1, 0, 0), (x, 1, 0)))
            lower = tuple(min(point[axis] for point in triangle) for axis in range(3))
            upper = tuple(max(point[axis] for point in triangle) for axis in range(3))
            extra_records.append((triangle, lower, upper, 4 + index,
                                  (4 + 3 * index, 5 + 3 * index, 6 + 3 * index)))
        records = tuple(records) + tuple(extra_records)
        prepared = audit.prepare_point_location(records)
        points = (
            offset,
            tuple(offset[axis] + scale for axis in range(3)),
            tuple(offset[axis] + 20 * scale for axis in range(3)),
            tuple(Fraction(offset[axis]) + Fraction(scale, 3) for axis in range(3)),
            tuple(Fraction(offset[axis]) + Fraction(scale * 4, 3) for axis in range(3)),
        )
        for point in points:
            with self.subTest(point=point):
                boundary, ray = audit._point_aabb_candidate_indices(point, prepared)
                expected_boundary = tuple(index for index, row in enumerate(records)
                    if all(row[1][axis] <= point[axis] <= row[2][axis] for axis in range(3)))
                expected_ray = tuple(index for index, row in enumerate(records)
                    if all(row[2][axis] >= point[axis] for axis in range(3)))
                self.assertEqual(boundary, expected_boundary)
                self.assertEqual(ray, expected_ray)
                self.assertEqual(boundary, tuple(sorted(set(boundary))))
                self.assertEqual(ray, tuple(sorted(set(ray))))
        self.assertGreater(prepared.aabb_node_count, 1)

    def test_exact_ray_aabb_candidates_match_exhaustive_closed_box_oracle(self):
        records, offset, scale = self.records()
        extra_records = []
        for index in range(12):
            x = 24 + 2 * index
            triangle = tuple(
                tuple(offset[axis] + scale * point[axis] for axis in range(3))
                for point in ((x, 0, 0), (x + 1, 0, 0), (x, 1, 0)))
            lower = tuple(min(point[axis] for point in triangle) for axis in range(3))
            upper = tuple(max(point[axis] for point in triangle) for axis in range(3))
            extra_records.append((triangle, lower, upper, 4 + index,
                                  (4 + 3 * index, 5 + 3 * index, 6 + 3 * index)))
        records = tuple(records) + tuple(extra_records)
        prepared = audit.prepare_point_location(records)
        points = (
            offset,
            tuple(Fraction(offset[axis]) + Fraction(scale, 3) for axis in range(3)),
            tuple(offset[axis] + 20 * scale for axis in range(3)),
            tuple(Fraction(offset[axis]) + Fraction(scale * 4, 3) for axis in range(3)),
        )
        directions = ((1, 1, 1), (1, 2, 4), (1, 3, 9))

        def intersects(lower, upper, point, direction):
            entry, exit = Fraction(0), None
            for axis, component in enumerate(direction):
                first = (Fraction(lower[axis]) - point[axis]) / component
                last = (Fraction(upper[axis]) - point[axis]) / component
                if first > last:
                    first, last = last, first
                if last < 0:
                    return False
                entry = max(entry, first)
                exit = last if exit is None else min(exit, last)
                if entry > exit:
                    return False
            return True

        self.assertGreater(prepared.aabb_node_count, 1)
        for point in points:
            for direction in directions:
                with self.subTest(point=point, direction=direction):
                    actual = audit._ray_aabb_candidate_indices(point, prepared, direction)
                    expected = tuple(index for index, row in enumerate(records)
                                     if intersects(row[1], row[2], point, direction))
                    self.assertEqual(actual, expected)
                    self.assertEqual(actual, tuple(sorted(set(actual))))

    def test_indexed_normalized_classifier_matches_full_scan(self):
        records, offset, scale = self.records()
        prepared = audit.prepare_point_location(records)
        points = (
            tuple(offset[axis] + scale for axis in range(3)),
            tuple(offset[axis] + 20 * scale for axis in range(3)),
            offset,
            tuple(Fraction(offset[axis]) + Fraction(scale, 8) for axis in range(3)),
            tuple(Fraction(offset[axis]) + Fraction(scale * 4, 3) for axis in range(3)),
        )
        for point in points:
            with self.subTest(point=point):
                full = audit.point_location_prepared(point, prepared)
                indexed = audit.point_location_prepared_indexed(point, prepared)
                self.assertEqual(indexed, full)


class PreparedSignedWindingTests(unittest.TestCase):
    @staticmethod
    def tetra_records(offset=(0, 0, 0), scale=12, reverse=False):
        base = ((0, 0, 0), (scale, 0, 0), (0, scale, 0), (0, 0, scale))
        vertices = tuple(tuple(offset[axis] + point[axis] for axis in range(3))
                         for point in base)
        faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        if reverse:
            faces = tuple((a, c, b) for a, b, c in faces)
        return vertices, faces, audit._records(vertices, faces)

    def test_outward_and_reversed_tetra_have_signed_unit_winding(self):
        _vertices, _faces, outward = self.tetra_records()
        prepared = audit.prepare_signed_winding(outward)
        self.assertEqual(prepared.face_component_count, 1)
        result = audit.signed_winding_number((1, 1, 1), prepared)
        self.assertEqual(result["status"], "resolved")
        self.assertEqual(result["winding_number"], 1)
        self.assertEqual(result["crossings"], 1)

        _vertices, _faces, inward = self.tetra_records(reverse=True)
        reversed_result = audit.signed_winding_number(
            (1, 1, 1), audit.prepare_signed_winding(inward))
        self.assertEqual(reversed_result["winding_number"], -1)

    def test_overlapping_closed_tetrahedra_sum_to_winding_two(self):
        outer_vertices, outer_faces, _outer = self.tetra_records(scale=12)
        inner_vertices, inner_faces, _inner = self.tetra_records(
            offset=(2, 2, 2), scale=12)
        faces = outer_faces + tuple(tuple(index + 4 for index in face)
                                    for face in inner_faces)
        records = audit._records(outer_vertices + inner_vertices, faces)
        first_records = audit._records(outer_vertices, outer_faces)
        second_records = audit._records(inner_vertices, inner_faces)
        self.assertGreater(audit._audit_pair(first_records, second_records,
                                             same_surface=False)["count"], 0)
        prepared = audit.prepare_signed_winding(records)
        self.assertEqual(prepared.face_component_count, 2)
        self.assertEqual(audit.signed_winding_number((3, 3, 3), prepared)["winding_number"], 2)

    def test_nested_opposite_shell_is_zero_in_cavity_and_one_in_material(self):
        outer_vertices, outer_faces, _outer = self.tetra_records(scale=12)
        inner_vertices, inner_faces, _inner = self.tetra_records(
            offset=(1, 1, 1), scale=4, reverse=True)
        faces = outer_faces + tuple(tuple(index + 4 for index in face)
                                    for face in inner_faces)
        records = audit._records(outer_vertices + inner_vertices, faces)
        prepared = audit.prepare_signed_winding(records)
        self.assertEqual(prepared.face_component_count, 2)
        self.assertEqual(audit.signed_winding_number((2, 2, 2), prepared)["winding_number"], 0)
        self.assertEqual(audit.signed_winding_number((6, 1, 1), prepared)["winding_number"], 1)

    def test_boundary_has_no_winding_and_primary_vertex_ray_retries(self):
        _vertices, _faces, records = self.tetra_records()
        prepared = audit.prepare_signed_winding(records)
        boundary = audit.signed_winding_number((0, 0, 0), prepared)
        self.assertEqual(boundary["status"], "boundary")
        self.assertIsNone(boundary["winding_number"])
        outside = audit.signed_winding_number((-1, -1, -1), prepared)
        self.assertEqual(outside["status"], "resolved")
        self.assertEqual(outside["winding_number"], 0)
        self.assertGreater(outside["attempts"], 1)

    def test_local_lattice_signed_winding_keeps_fractional_queries_exact(self):
        scale = 1 << 1800
        offset = (1 << 1900, -(1 << 1898), 1 << 1887)
        vertices = tuple(tuple(offset[axis] + scale * point[axis] for axis in range(3))
                         for point in ((0, 0, 0), (12, 0, 0), (0, 12, 0), (0, 0, 12)))
        faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        prepared = audit.prepare_signed_winding(audit._records(vertices, faces))
        self.assertLess(max(abs(value).bit_length()
                            for triangle in prepared.point_locator.triangles
                            for point in triangle for value in point), 5)
        inside = tuple(Fraction(offset[k]) + Fraction(scale, 12) for k in range(3))
        outside = tuple(Fraction(offset[k]) - scale for k in range(3))
        boundary = tuple(Fraction(offset[k]) + scale * (12 if k == 0 else 0) for k in range(3))
        self.assertEqual(audit.signed_winding_number(inside, prepared)["winding_number"], 1)
        self.assertEqual(audit.signed_winding_number(outside, prepared)["winding_number"], 0)
        self.assertEqual(audit.signed_winding_number(boundary, prepared)["status"], "boundary")

    def test_open_inconsistently_oriented_and_degenerate_shells_are_rejected(self):
        _vertices, faces, records = self.tetra_records()
        with self.assertRaisesRegex(HumanImportError, "exactly two incident faces"):
            audit.prepare_signed_winding(records[:-1])
        inconsistent_faces = list(faces)
        inconsistent_faces[0] = tuple(reversed(inconsistent_faces[0]))
        inconsistent = audit._records(_vertices, inconsistent_faces)
        with self.assertRaisesRegex(HumanImportError, "consistently oriented"):
            audit.prepare_signed_winding(inconsistent)
        duplicate = records + [(records[0][0], records[0][1], records[0][2],
                                len(records), records[0][4])]
        with self.assertRaisesRegex(HumanImportError, "duplicate faces"):
            audit.prepare_signed_winding(duplicate)

        vertices_a, faces_a, _ = self.tetra_records(scale=4)
        vertices_b, faces_b, _ = self.tetra_records(
            offset=(0, 0, 0), scale=-4)
        shared_vertex_mesh = vertices_a + vertices_b[1:]
        second_map = (0, 4, 5, 6)
        shared_vertex_faces = faces_a + tuple(
            tuple(second_map[index] for index in face) for face in faces_b)
        pinch = audit._records(shared_vertex_mesh, shared_vertex_faces)
        with self.assertRaisesRegex(HumanImportError, "multiple cycles"):
            audit.prepare_signed_winding(pinch)

        degenerate = [(((0, 0, 0), (1, 0, 0), (2, 0, 0)),
                       (0, 0, 0), (2, 0, 0), 0, (0, 1, 2))]
        with self.assertRaisesRegex(HumanImportError, "degenerate"):
            audit.prepare_signed_winding(degenerate)

    def test_unresolved_ray_returns_no_winding(self):
        _vertices, _faces, records = self.tetra_records()
        prepared = audit.prepare_signed_winding(records)
        # Force the exact parallel/coplanar retry path for every deterministic
        # direction while the query lies in the z=0 plane, outside the face.
        cached = (tuple(0 for _ in prepared.point_locator.normals), (0,))
        with patch.object(audit, "_prepared_ray_data", return_value=cached):
            result = audit.signed_winding_number((13, 1, 0), prepared)
        self.assertEqual(result["status"], "indeterminate")
        self.assertIsNone(result["winding_number"])
        self.assertEqual(result["attempts"], 64)


class ClosedDomainTests(unittest.TestCase):
    def test_disjoint_nested_and_intersecting_solids_are_distinguished(self):
        disjoint = run(tetra('a'),tetra('b',(2.,0.,0.)))
        self.assertTrue(disjoint['all_surfaces_embedded'])
        self.assertTrue(disjoint['all_domains_disjoint'])
        nested = run(tetra('outer',scale=2.),tetra('inner',(.25,.25,.25),.25))
        pair = nested['per_pair'][0]
        self.assertEqual(pair['count'], 0)
        self.assertFalse(nested['all_domains_disjoint'])
        self.assertEqual(pair['containment']['first_in_second']['location'], 'inside')
        crossing = run(tetra('a'),tetra('b',(.25,.25,.25)))
        self.assertGreater(crossing['per_pair'][0]['count'],0)
        self.assertEqual(crossing['per_pair'][0]['containment']['status'],'not_checked_intersecting_surfaces')

    def test_cached_topology_claim_cannot_hide_open_or_disconnected_surface(self):
        open_surface = tetra('open')
        open_surface['exact_coordinate_quotient']['triangles'].pop()
        open_surface['exact_coordinate_quotient']['topology'] = {'closed_oriented_manifold_candidate':True}
        self.assertFalse(run(open_surface)['all_domains_disjoint'])
        a, b = tetra('disconnected'), tetra('unused',(2.,0.,0.))
        q, r = a['exact_coordinate_quotient'], b['exact_coordinate_quotient']
        q['vertices_m'] += r['vertices_m']
        q['triangles'] += [[k+4 for k in face] for face in r['triangles']]
        self.assertFalse(run(a)['all_surfaces_embedded'])

    def test_unused_first_vertex_cannot_turn_nested_domains_into_disjoint_solids(self):
        outer = tetra('outer', scale=2.)
        inner = tetra('inner', (.25, .25, .25), .25)
        nested = run(outer, inner)
        self.assertTrue(nested['all_surfaces_embedded'])
        self.assertFalse(nested['all_domains_disjoint'])
        self.assertEqual(nested['per_pair'][0]['containment']['first_in_second']['location'], 'inside')
        q = inner['exact_coordinate_quotient']
        q['vertices_m'].insert(0, [9., 9., 9.])
        q['triangles'] = [[index + 1 for index in face] for face in q['triangles']]
        # The actual inner surface is unchanged, but vertex zero now lies
        # outside the outer shell. It must never serve as a containment witness.
        result = run(outer, inner)
        self.assertEqual(result['per_pair'][0]['count'], 0)
        self.assertFalse(result['per_surface']['inner']['embedded_closed_surface'])
        self.assertFalse(result['all_surfaces_embedded'])
        self.assertFalse(result['all_domains_disjoint'])
        self.assertEqual(result['per_pair'][0]['containment']['status'], 'not_checked_invalid_surface')

    def test_translation_and_reversed_orientation_preserve_disjointness(self):
        a,b=tetra('a'),tetra('b',(2.,0.,0.))
        expected=run(a,b)
        for chamber in (a,b):
            q=chamber['exact_coordinate_quotient']
            q['vertices_m']=[[p[0]+1024,p[1]-2048,p[2]+4096] for p in q['vertices_m']]
            q['triangles']=[list(reversed(face)) for face in q['triangles']]
        actual=run(a,b)
        self.assertEqual(expected['all_domains_disjoint'],actual['all_domains_disjoint'])
        self.assertEqual([r['triangle_pairs'] for r in expected['per_pair']], [r['triangle_pairs'] for r in actual['per_pair']])

    def test_duplicate_ids_nonfinite_and_wrong_arity_rejected(self):
        with self.assertRaisesRegex(HumanImportError,'duplicate'):
            run(tetra('a'),tetra('a'))
        for value in (float('nan'),float('inf'),True,10**1000):
            bad=tetra('a');bad['exact_coordinate_quotient']['vertices_m'][0][0]=value
            with self.assertRaises(HumanImportError):run(bad)
        bad=tetra('a');bad['exact_coordinate_quotient']['triangles'][0]=[0,1,2,3]
        with self.assertRaises(HumanImportError):run(bad)


class SourceGeometryTests(unittest.TestCase):
    def test_pinned_source_is_embedded_but_right_chambers_intersect(self):
        cavities=extract_cavity_surfaces()
        original=copy.deepcopy(cavities)
        result=audit.audit_cavity_intersections(cavities)
        self.assertEqual(cavities,original)
        self.assertTrue(result['all_surfaces_embedded'])
        self.assertFalse(result['all_domains_disjoint'])
        self.assertTrue(all(row['count']==0 for row in result['per_surface'].values()))
        pairs={(p['first'],p['second']):p for p in result['per_pair']}
        overlap=pairs[('right_atrium','right_ventricle')]
        self.assertEqual(overlap['count'],42)
        self.assertIn([748,1916],overlap['triangle_pairs'])
        self.assertTrue(all(p['disjoint_closed_domains'] for key,p in pairs.items() if key!=('right_atrium','right_ventricle')))
        self.assertEqual(result['coordinate_semantics'],'exact_rational_value_of_published_binary64_metres')
        self.assertFalse(result['source_coordinates_modified'])
        self.assertFalse(result['overlap_repair'])


if __name__ == '__main__':
    unittest.main()
