#!/usr/bin/env python3
"""Compile the retained BodyParts3D cavity overlap into the existing NHANAT5.

This is a bounded asset-authoring step. It does not modify CVSim, add cardiac
mass, or select a biological valve plane. The source meshes are registered to
one torso body frame; only their duplicated RA/RV interior is assigned to one
source surface so the two closed presentation volumes have disjoint interiors.
The NHA payload ABI stays unchanged and its existing source receipt is updated.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np


HEADER = struct.Struct("<8s5I32s")
RECORD = struct.Struct("<8I")
VERTEX = struct.Struct("<6f")
U32 = struct.Struct("<I")
RA_ID, RV_ID, LA_ID, LV_ID = 318, 319, 320, 321
CARDIAC_MOTION_MODEL = "source_centroid_radial_freewall_v2"
RA_RV_TAPER_DEGREES = 20.0
RA_CENTER_OFFSET_M = 0.008
LEFT_ATRIUM_CENTER_OFFSET_M = 0.012
LEFT_VENTRICLE_CENTER_OFFSET_M = 0.015


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("cardiac geometry binding: " + message)


def read_payload(path: Path):
    raw = path.read_bytes()
    require(len(raw) >= HEADER.size, "truncated NHANAT5 header")
    magic, abi, count, vertex_count, index_count, registration, source_sha = HEADER.unpack_from(raw)
    require(magic == b"NHANAT1\0" and abi == 5, "expected the existing NHANAT5 payload")
    expected = HEADER.size + count * RECORD.size + vertex_count * VERTEX.size + index_count * U32.size
    require(len(raw) == expected and index_count % 3 == 0, "NHANAT5 byte counts disagree")
    at = HEADER.size
    records = [RECORD.unpack_from(raw, at + i * RECORD.size) for i in range(count)]
    at += count * RECORD.size
    vertices = np.frombuffer(raw, dtype="<f4", count=vertex_count * 6, offset=at).reshape(-1, 6).copy()
    at += vertex_count * VERTEX.size
    indices = np.frombuffer(raw, dtype="<u4", count=index_count, offset=at).copy()
    return raw, (magic, abi, registration, source_sha), records, vertices, indices


def surface_arrays(records, vertices, indices, stable_id, expected_layer=9):
    matches = [record for record in records if record[5] == stable_id]
    require(len(matches) == 1, f"stable ID {stable_id} is not unique")
    body, first_vertex, vertex_count, first_index, index_count, _, layer, reserved = matches[0]
    require(layer == expected_layer and reserved == 0, f"stable ID {stable_id} has an unexpected source layer")
    points = vertices[first_vertex:first_vertex + vertex_count, :3].astype(np.float64)
    faces = (indices[first_index:first_index + index_count].reshape(-1, 3) - first_vertex).astype(np.int64)
    require(np.all(faces >= 0) and np.all(faces < vertex_count), f"stable ID {stable_id} has an invalid face")
    return body, points, faces


def closed_centroid(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    origin = points[0].astype(np.float64)
    volume = 0.0
    moment = np.zeros(3, dtype=np.float64)
    for face in faces:
        a, b, c = points[face].astype(np.float64) - origin
        signed = float(np.dot(a, np.cross(b, c)) / 6.0)
        volume += signed
        moment += signed * (a + b + c) / 4.0
    require(abs(volume) > 1e-12, "cardiac phase audit found a zero-volume cavity")
    return (origin + moment / volume).astype(np.float32)


def closest_triangle(point, a, b, c):
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0 and d2 <= 0:
        return a
    bp = point - b
    d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
    if d3 >= 0 and d4 <= d3:
        return b
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        return a + (d1 / (d1 - d3)) * ab
    cp = point - c
    d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
    if d6 >= 0 and d5 <= d6:
        return c
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        return a + (d2 / (d2 - d6)) * ac
    va = d3 * d6 - d5 * d4
    if va <= 0 and d4 - d3 >= 0 and d5 - d6 >= 0:
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b)
    denom = 1.0 / (va + vb + vc)
    return a + (vb * denom) * ab + (vc * denom) * ac


class TriangleIndex:
    """Small offline counterpart of the load-time cavity triangle BVH."""
    def __init__(self, points: np.ndarray, faces: np.ndarray):
        self.points = points.astype(np.float64)
        self.triangles = faces.astype(np.int64)
        tri = self.points[self.triangles]
        self.lower, self.upper = tri.min(axis=1), tri.max(axis=1)
        self.centers = tri.mean(axis=1)
        self.order = np.arange(len(faces), dtype=np.int64)
        self.nodes = []
        self._build(0, len(self.order))

    def _build(self, begin: int, end: int) -> int:
        ids = self.order[begin:end]
        lo, hi = self.lower[ids].min(axis=0), self.upper[ids].max(axis=0)
        index = len(self.nodes)
        self.nodes.append(None)
        if end - begin <= 8:
            self.nodes[index] = (lo, hi, begin, end, -1, -1)
            return index
        center = self.centers[ids]
        axis = int(np.argmax(center.max(axis=0) - center.min(axis=0)))
        middle = begin + (end - begin) // 2
        local = np.argpartition(center[:, axis], middle - begin)
        self.order[begin:end] = ids[local]
        left = self._build(begin, middle)
        right = self._build(middle, end)
        self.nodes[index] = (lo, hi, 0, 0, left, right)
        return index

    @staticmethod
    def _box_distance_squared(lo, hi, point):
        delta = np.maximum(np.maximum(lo - point, point - hi), 0.0)
        return float(np.dot(delta, delta))

    def nearest(self, point: np.ndarray):
        best = [float("inf"), None, None, None]

        def query(index):
            lo, hi, begin, end, left, right = self.nodes[index]
            if self._box_distance_squared(lo, hi, point) >= best[0]:
                return
            if left < 0:
                for row in self.order[begin:end]:
                    face = self.triangles[row]
                    q = closest_triangle(point, *self.points[face])
                    delta = point - q
                    distance = float(np.dot(delta, delta))
                    if distance < best[0]:
                        a, b, c = self.points[face]
                        ab, ac, aq = b - a, c - a, q - a
                        d00, d01, d11 = float(np.dot(ab, ab)), float(np.dot(ab, ac)), float(np.dot(ac, ac))
                        d20, d21 = float(np.dot(aq, ab)), float(np.dot(aq, ac))
                        denom = d00 * d11 - d01 * d01
                        v = (d11 * d20 - d01 * d21) / denom
                        w = (d00 * d21 - d01 * d20) / denom
                        best[:] = (distance, q.copy(), face.copy(), np.asarray((1.0 - v - w, v, w)))
                return
            dleft = self._box_distance_squared(self.nodes[left][0], self.nodes[left][1], point)
            dright = self._box_distance_squared(self.nodes[right][0], self.nodes[right][1], point)
            if dleft <= dright:
                query(left)
                query(right)
            else:
                query(right)
                query(left)

        query(0)
        return tuple(best)

    def nearest_distance_squared(self, point: np.ndarray) -> float:
        return self.nearest(point)[0]


def phase_rows(path: Path, phase_selection: str = "representative-cycle"):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"time_s", "step", "q_ra", "q_rv", "q_la", "q_lv",
                "ra_target_ml", "rv_target_ml", "la_target_ml", "lv_target_ml"}
    require(len(rows) >= 8 and required.issubset(rows[0]),
            "cardiac phase trace needs at least eight accepted rows with time, step and four q values")
    q_names = ("q_ra", "q_rv", "q_la", "q_lv")
    for row in rows:
        row["time_s"] = float(row["time_s"])
        row["step"] = int(float(row["step"]))
        row["q"] = np.asarray([float(row[key]) for key in q_names], dtype=np.float32)
        volume_names = ("ra_target_ml", "rv_target_ml", "la_target_ml", "lv_target_ml")
        row["target_volume_m3"] = np.asarray([float(row[key]) for key in volume_names], dtype=np.float32) * np.float32(1e-6)
        require(np.isfinite(row["time_s"]) and np.all(np.isfinite(row["q"])) and
                np.all(np.isfinite(row["target_volume_m3"])) and np.all(row["target_volume_m3"] > 0),
                "cardiac phase trace has nonfinite accepted coordinates")
    if phase_selection == "initial-and-lv-max":
        selected = {0, int(np.argmax([row["q"][3] for row in rows]))}
    elif phase_selection == "representative-cycle":
        selected = {round(i * (len(rows) - 1) / 7) for i in range(8)}
        for chamber in range(4):
            selected.add(int(np.argmin([row["q"][chamber] for row in rows])))
            selected.add(int(np.argmax([row["q"][chamber] for row in rows])))
    else:
        raise ValueError(f"unsupported cardiac phase selection: {phase_selection}")
    return rows, [rows[i] for i in sorted(selected)]


def exact_large_surface_audits(surfaces, predicates, *, stop_after_first_unqualified=False):
    """Apply NumiLab's exact predicates to passive meshes just over its public size cap.

    The retained exact auditor limits public inputs to 20,000 vertices. One
    source wall has 21,053, so use its same exact lower-level integer triangle
    predicates after validating topology and constructing one shared rational
    coordinate scale. No faces are dropped or approximated.
    """
    vertices_by_id, faces_by_id, rational_by_id, topology_by_id = {}, {}, {}, {}
    denominator = 1
    for surface in surfaces:
        name = surface["source_id"]
        quotient = surface["exact_coordinate_quotient"]
        raw_vertices = np.asarray(quotient["vertices_m"], dtype=np.float32)
        raw_faces = np.asarray(quotient["triangles"], dtype=np.int64)
        raw_topology = predicates.analyze_topology(raw_vertices.astype(float).tolist(), raw_faces.astype(int).tolist())
        exact_ids, exact_vertices = {}, []
        source_to_exact = []
        for point in raw_vertices:
            key = tuple(float(value) for value in point)
            if key not in exact_ids:
                exact_ids[key] = len(exact_vertices)
                exact_vertices.append(point.copy())
            source_to_exact.append(exact_ids[key])
        exact_faces = np.asarray([[source_to_exact[int(i)] for i in face] for face in raw_faces], dtype=np.int64)
        used = sorted(set(int(i) for i in exact_faces.reshape(-1)))
        compact = {old: new for new, old in enumerate(used)}
        vertices = np.asarray([exact_vertices[i] for i in used], dtype=np.float32).astype(float).tolist()
        faces = [[compact[int(i)] for i in face] for face in exact_faces]
        topology = predicates.analyze_topology(vertices, faces)
        if int(name) in (RA_ID, RV_ID, LA_ID, LV_ID):
            require(topology["closed_oriented_manifold_candidate"] and topology["face_component_count"] == 1,
                    f"deformed cardiac cavity {name} lost closed connected topology after exact coordinate quotient")
        rational = [tuple(Fraction.from_float(float(x)) for x in point) for point in vertices]
        for point in rational:
            for coordinate in point:
                denominator = max(denominator, coordinate.denominator)
        vertices_by_id[name], faces_by_id[name] = vertices, faces
        rational_by_id[name], topology_by_id[name] = rational, {
            "raw": raw_topology,
            "exact_coordinate_quotient": topology,
            "exact_duplicate_vertex_count": len(raw_vertices) - len(exact_vertices),
            "unused_source_vertex_count": len(raw_vertices) - len(set(int(i) for i in raw_faces.reshape(-1))),
        }

    records_by_id = {}
    collision_face_counts = {}
    for name in sorted(vertices_by_id):
        integer = [tuple(x.numerator * (denominator // x.denominator) for x in point)
                   for point in rational_by_id[name]]
        collision_faces = []
        degenerate_count = 0
        for face in faces_by_id[name]:
            a, b, c = (integer[int(index)] for index in face)
            if any(predicates._cross(predicates._sub(b, a), predicates._sub(c, a))):
                collision_faces.append(face)
            else:
                degenerate_count += 1
        require(int(name) not in (RA_ID, RV_ID, LA_ID, LV_ID) or degenerate_count == 0,
                f"deformed cardiac cavity {name} contains an exact degenerate triangle")
        collision_face_counts[name] = {"tested_nonzero_area_face_count": len(collision_faces),
                                       "excluded_exact_zero_area_face_count": degenerate_count}
        records_by_id[name] = predicates._records(integer, collision_faces)

    surface_reports = {}
    pair_reports = []
    names = sorted(records_by_id)
    for offset, first in enumerate(names):
        for second in names[offset + 1:]:
            if int(first) not in (1, 23, 24) and int(second) not in (1, 23, 24):
                continue
            report = predicates._audit_pair(
                records_by_id[first], records_by_id[second], same_surface=False,
                stop_after_first_intersection=stop_after_first_unqualified)
            row = {"first": int(first), "second": int(second), "intersection_triangle_pair_count": report["count"],
                   "aabb_candidate_pairs": report["aabb_candidate_pairs"]}
            if report.get("audit_complete") is False:
                row.update({"relation": "intersecting_unclassified",
                            "intersection_count_is_lower_bound": True,
                            "triangle_pair_witnesses": report["triangle_pairs"]})
                pair_reports.append(row)
                return surface_reports, pair_reports, False
            first_closed = topology_by_id[first]["exact_coordinate_quotient"]["closed_oriented_manifold_candidate"] and \
                topology_by_id[first]["exact_coordinate_quotient"]["face_component_count"] == 1
            second_closed = topology_by_id[second]["exact_coordinate_quotient"]["closed_oriented_manifold_candidate"] and \
                topology_by_id[second]["exact_coordinate_quotient"]["face_component_count"] == 1
            if report["count"]:
                row["relation"] = "intersecting_unclassified"
            elif not first_closed or not second_closed:
                row["relation"] = "nonintersecting_passive_surface_with_unclosed_source_topology"
                row["containment"] = "unavailable_for_open_or_disconnected_or_nonmanifold_source_surface"
            else:
                a = [tuple(x.numerator * (denominator // x.denominator) for x in point)
                     for point in rational_by_id[first]]
                b = [tuple(x.numerator * (denominator // x.denominator) for x in point)
                     for point in rational_by_id[second]]
                first_in_second = predicates.point_location(a[0], records_by_id[second])
                second_in_first = predicates.point_location(b[0], records_by_id[first])
                row["containment"] = {"first_in_second": first_in_second, "second_in_first": second_in_first}
                if (first_in_second["location"] == "outside" and second_in_first["location"] == "outside"):
                    row["relation"] = "disjoint"
                elif int(first) in (1, 24) and int(second) in (318, 319, 320, 321) and \
                        second_in_first["location"] == "inside":
                    row["relation"] = "expected_cavity_enclosed_by_passive_outer_heart"
                elif int(second) in (1, 24) and int(first) in (318, 319, 320, 321) and \
                        first_in_second["location"] == "inside":
                    row["relation"] = "expected_cavity_enclosed_by_passive_outer_heart"
                elif int(first) == 23 and int(second) in (319, 321) and second_in_first["location"] == "inside":
                    row["relation"] = "expected_ventricular_cavity_enclosed_by_passive_ventricular_wall"
                elif int(second) == 23 and int(first) in (319, 321) and first_in_second["location"] == "inside":
                    row["relation"] = "expected_ventricular_cavity_enclosed_by_passive_ventricular_wall"
                else:
                    row["relation"] = "nested_surface_without_registered_anatomical_interface"
            pair_reports.append(row)
            if stop_after_first_unqualified and row["relation"] not in (
                    "disjoint", "expected_cavity_enclosed_by_passive_outer_heart",
                    "expected_ventricular_cavity_enclosed_by_passive_ventricular_wall",
                    "nonintersecting_passive_surface_with_unclosed_source_topology"):
                return surface_reports, pair_reports, False

    # A single unqualified wall pair is already sufficient to reject the
    # passive-wall gate. Check these pairs before the more expensive per-wall
    # self audits in fail-fast source admission; if every pair is qualified,
    # continue with all self audits before reporting a pass.
    for name in sorted(records_by_id):
        if int(name) not in (1, 23, 24):
            continue
        report = predicates._audit_pair(
            records_by_id[name], records_by_id[name], same_surface=True,
            stop_after_first_intersection=stop_after_first_unqualified)
        topology = topology_by_id[name]
        quotient_topology = topology["exact_coordinate_quotient"]
        surface_reports[name] = {"raw_topology": topology["raw"],
                                 "exact_coordinate_quotient_topology": quotient_topology,
                                 "exact_duplicate_vertex_count": topology["exact_duplicate_vertex_count"],
                                 "unused_source_vertex_count": topology["unused_source_vertex_count"],
                                 **collision_face_counts[name],
                                 "closed_connected_oriented": quotient_topology["closed_oriented_manifold_candidate"] and
                                     quotient_topology["face_component_count"] == 1,
                                 "self_intersection_triangle_pair_count": report["count"],
                                 "self_intersection_audit_complete": report.get("audit_complete", True),
                                 "self_intersection_count_is_lower_bound": report.get("count_is_lower_bound", False),
                                 "aabb_candidate_pairs": report["aabb_candidate_pairs"]}
        if stop_after_first_unqualified and not report.get("audit_complete", True):
            return surface_reports, pair_reports, False
    return surface_reports, pair_reports, True


def audit_dynamic_phases(records, vertices, indices, rows, samples, interface_points,
                         interface_faces, predicates, certificate):
    ids = (RA_ID, RV_ID, LA_ID, LV_ID)
    source = [surface_arrays(records, vertices, indices, stable_id)[1:] for stable_id in ids]
    points = [row[0].astype(np.float32) for row in source]
    faces = [row[1] for row in source]
    centers = [closed_centroid(p, f) for p, f in zip(points, faces, strict=True)]

    def shifted_center(chamber, from_chamber, away_from_chamber, distance):
        direction = (centers[from_chamber].astype(np.float64) -
                     centers[away_from_chamber].astype(np.float64))
        length = float(np.linalg.norm(direction))
        require(np.isfinite(length) and length > 1e-9, "cardiac registration offset has an undefined direction")
        centers[chamber] = (centers[chamber].astype(np.float64) + direction * (distance / length)).astype(np.float32)

    shifted_center(0, 0, 1, RA_CENTER_OFFSET_M)
    left_direction = centers[2].astype(np.float64) - centers[3].astype(np.float64)
    left_length = float(np.linalg.norm(left_direction))
    require(np.isfinite(left_length) and left_length > 1e-9, "left cardiac registration offset has an undefined direction")
    centers[2] = (centers[2].astype(np.float64) + left_direction *
                  (LEFT_ATRIUM_CENTER_OFFSET_M / left_length)).astype(np.float32)
    centers[3] = (centers[3].astype(np.float64) + left_direction *
                  (LEFT_VENTRICLE_CENTER_OFFSET_M / left_length)).astype(np.float32)

    indices_by_chamber = [TriangleIndex(p, f) for p, f in zip(points, faces, strict=True)]
    weights = []
    seam_points = np.unique(interface_points.astype(np.float32), axis=0).astype(np.float64)
    seam_keys = {tuple(point.astype(np.float32).view(np.uint32).tolist()) for point in seam_points}
    for chamber, p in enumerate(points):
        if chamber >= 2:
            weight = np.ones(len(p), dtype=np.float32)
        else:
            radial = p - centers[chamber]
            require(np.all(np.isfinite(radial)),
                    f"chamber {ids[chamber]} has nonfinite source radial vectors")
            radial_length = np.linalg.norm(radial, axis=1)
            require(np.all(np.isfinite(radial_length)),
                    f"chamber {ids[chamber]} has nonfinite source radial lengths")
            radial = np.divide(radial, radial_length[:, None], out=np.zeros_like(radial),
                                where=radial_length[:, None] > 0.0)
            seam = seam_points - centers[chamber]
            require(np.all(np.isfinite(seam)),
                    f"chamber {ids[chamber]} has nonfinite RA/RV seam vectors")
            seam_length = np.linalg.norm(seam, axis=1)
            require(np.all(np.isfinite(seam_length)),
                    f"chamber {ids[chamber]} has nonfinite RA/RV seam lengths")
            seam = seam[seam_length > 0.0] / seam_length[seam_length > 0.0, None]
            require(len(seam) > 0, f"chamber {ids[chamber]} has no directed RA/RV seam samples")
            radial64 = radial.astype(np.float64)
            directional_cosines = (radial64[:, None, 0] * seam[None, :, 0] +
                                   radial64[:, None, 1] * seam[None, :, 1] +
                                   radial64[:, None, 2] * seam[None, :, 2])
            require(np.all(np.isfinite(directional_cosines)),
                    f"chamber {ids[chamber]} has nonfinite RA/RV seam direction cosines")
            nearest_direction_cosine = np.clip(directional_cosines, -1.0, 1.0).max(axis=1)
            interface_cosine = np.cos(np.radians(RA_RV_TAPER_DEGREES))
            t = np.clip((1.0 - nearest_direction_cosine) / (1.0 - interface_cosine), 0.0, 1.0)
            weight = (t * t * (3.0 - 2.0 * t)).astype(np.float32)
            for vertex, point in enumerate(p):
                if tuple(point.astype(np.float32).view(np.uint32).tolist()) in seam_keys:
                    weight[vertex] = 0.0
        weights.append(weight)

    # Match the native load-time cubic: evaluate the candidate's own source
    # surface and free-wall weights at q = 0, 1, -1, 2, then cast its four
    # coefficients to FP32 before solving the same accepted hydraulic target.
    polynomials = []
    for chamber in range(4):
        center = centers[chamber].astype(np.float64)

        def volume_at(q):
            scale = 1.0 + float(q) * weights[chamber].astype(np.float64)
            mapped = center + scale[:, None] * (points[chamber].astype(np.float64) - center)
            tri = mapped[faces[chamber]] - center
            volume = np.einsum("ij,ij->i", tri[:, 0],
                               np.cross(tri[:, 1], tri[:, 2])).sum(dtype=np.float64) / 6.0
            return abs(float(volume))

        f0, f1, fm1, f2 = (volume_at(q) for q in (0.0, 1.0, -1.0, 2.0))
        a2 = (f1 + fm1 - 2.0 * f0) / 2.0
        s1 = (f1 - fm1) / 2.0
        a3 = (f2 - f0 - 4.0 * a2 - 2.0 * s1) / 6.0
        a1 = s1 - a3
        coefficient = np.asarray((f0, a1, a2, a3), dtype=np.float32)
        require(np.all(np.isfinite(coefficient)) and coefficient[0] > 0,
                f"candidate chamber {ids[chamber]} has invalid native volume coefficients")
        polynomials.append(coefficient)

    def solve_native_q(chamber, target):
        coefficient = polynomials[chamber]

        def value(q):
            return np.float32(np.float32(np.float32(np.float32(coefficient[3] * q + coefficient[2]) * q +
                                   coefficient[1]) * q + coefficient[0]))

        low, high = np.float32(-.999), np.float32(1.0)
        require(value(low) <= target <= value(high),
                f"candidate chamber {ids[chamber]} does not cover accepted volume {float(target)} m3")
        for _ in range(28):
            middle = np.float32(.5) * np.float32(low + high)
            if value(middle) < target:
                low = middle
            else:
                high = middle
        return np.float32(.5) * np.float32(low + high)

    wall_ids = (1, 23, 24)
    wall_source = [surface_arrays(records, vertices, indices, stable_id, expected_layer=1)[1:]
                   for stable_id in wall_ids]
    wall_points = [row[0].astype(np.float32) for row in wall_source]
    wall_faces = [row[1] for row in wall_source]
    def build_wall_binding():
        wall_binding = []
        for points_for_wall in wall_points:
            binding = []
            for point in points_for_wall:
                nearest = None
                chamber = -1
                for index, candidate in enumerate(indices_by_chamber):
                    result = candidate.nearest(point.astype(np.float64))
                    if nearest is None or result[0] < nearest[0]:
                        nearest, chamber = result, index
                distance = float(np.sqrt(nearest[0]))
                t = np.clip((distance - 0.040) / (0.080 - 0.040), 0.0, 1.0)
                falloff = 1.0 - t * t * (3.0 - 2.0 * t)
                face_ids = nearest[2]
                freewall = float(np.dot(nearest[3], weights[chamber][face_ids]))
                delta = (nearest[1] - centers[chamber]).astype(np.float32)
                binding.append((chamber, delta, np.float32(falloff * freewall)))
            wall_binding.append(binding)
        return wall_binding

    source_surfaces = [{"source_id": str(stable_id), "exact_coordinate_quotient": {
        "vertices_m": points[c].astype(np.float64).tolist(), "triangles": faces[c].tolist()}}
        for c, stable_id in enumerate(ids)]
    source_surfaces.extend({"source_id": str(stable_id), "exact_coordinate_quotient": {
        "vertices_m": wall_points[i].astype(np.float64).tolist(), "triangles": wall_faces[i].tolist()}}
        for i, stable_id in enumerate(wall_ids))
    source_wall_surface_reports, source_wall_pair_reports, source_wall_audit_complete = (
        exact_large_surface_audits(source_surfaces, predicates, stop_after_first_unqualified=True))
    source_wall_relations = [pair for pair in source_wall_pair_reports
                             if int(pair["first"]) in wall_ids or int(pair["second"]) in wall_ids]
    source_unqualified_pairs = [[pair["first"], pair["second"]] for pair in source_wall_relations
        if pair["relation"] not in ("disjoint", "expected_cavity_enclosed_by_passive_outer_heart",
                                     "expected_ventricular_cavity_enclosed_by_passive_ventricular_wall",
                                     "nonintersecting_passive_surface_with_unclosed_source_topology")]
    source_wall_gate = "fail" if not source_wall_audit_complete or source_unqualified_pairs or any(
        report["self_intersection_triangle_pair_count"] for report in source_wall_surface_reports.values()) else "pass"
    # A failed source-state wall gate already makes every full wall-envelope
    # claim fail closed. Avoid constructing expensive nearest-cavity bindings
    # for tens of thousands of wall vertices when those bindings cannot admit
    # a passing wall result; the four cavity phase audit remains independent.
    wall_binding = build_wall_binding() if source_wall_gate == "pass" else None

    interface = interface_points[interface_faces].astype(np.float64).tolist()
    results = []
    for row in samples:
        native_q = row["q"].copy()
        q = np.asarray([solve_native_q(chamber, row["target_volume_m3"][chamber])
                        for chamber in range(4)], dtype=np.float32)
        deformed = []
        for chamber in range(4):
            chamber_q = q[chamber]
            scale = np.float32(1.0) + chamber_q * weights[chamber]
            mapped = centers[chamber] + scale[:, None] * (points[chamber] - centers[chamber])
            # The registered RA/RV cut is an exact shared interface. Its
            # zero-weight map is mathematically the identity; evaluate that
            # identity directly so FP32 cancellation cannot move a seam
            # vertex by one ULP and invalidate the exact partition witness.
            fixed = weights[chamber] == 0.0
            mapped[fixed] = points[chamber][fixed]
            deformed.append(mapped.astype(np.float32))
        surfaces = [{"source_id": str(stable_id), "exact_coordinate_quotient": {
            "vertices_m": deformed[c].astype(np.float64).tolist(), "triangles": faces[c].tolist()}}
            for c, stable_id in enumerate(ids)]
        deformed_walls = []
        if wall_binding is not None:
            for wall_index, stable_id in enumerate(wall_ids):
                mapped = wall_points[wall_index].copy()
                for vertex, (chamber, delta, weight) in enumerate(wall_binding[wall_index]):
                    mapped[vertex] += delta * q[chamber] * weight
                mapped = mapped.astype(np.float32)
                deformed_walls.append(mapped)
                surfaces.append({"source_id": str(stable_id), "exact_coordinate_quotient": {
                    "vertices_m": mapped.astype(np.float64).tolist(),
                    "triangles": wall_faces[wall_index].tolist()}})
        audit = predicates.audit_cavity_intersections({"chambers": surfaces[:4]})
        if not audit["all_surfaces_embedded"]:
            detail = ";".join(f"{name}:embedded={report['embedded_closed_surface']}:pairs={report['count']}:"
                               f"topology={report['closed_connected_oriented_manifold']}"
                               for name, report in sorted(audit["per_surface"].items()))
            require(False, f"cavity self-intersection during accepted frame {row['step']} q="
                           f"{row['q'].tolist()} [{detail}]")
        ra = surfaces[0]["exact_coordinate_quotient"]
        rv = surfaces[1]["exact_coordinate_quotient"]
        triangle_points = lambda mesh: [[mesh["vertices_m"][int(i)] for i in face] for face in mesh["triangles"]]
        expected_interface = Counter(oriented_face_key(triangle) for triangle in interface)
        reverse_interface = Counter(oriented_face_key((triangle[0], triangle[2], triangle[1]))
                                    for triangle in interface)
        ra_interface = Counter(oriented_face_key(triangle) for triangle in triangle_points(ra))
        rv_interface = Counter(oriented_face_key(triangle) for triangle in triangle_points(rv))
        missing_ra = sum(count for key, count in expected_interface.items() if ra_interface[key] != count)
        missing_rv = sum(count for key, count in reverse_interface.items() if rv_interface[key] != count)
        require(missing_ra == 0 and missing_rv == 0,
                f"phase {row['step']} FP32 shared-face keys differ from the registered RA/RV interface: "
                f"interface_faces={len(interface)}, missing_RA={missing_ra}, missing_reversed_RV={missing_rv}")
        region_triangles = {"right_atrium": triangle_points(ra), "right_ventricle": triangle_points(rv)}
        try:
            interface_audit = certificate.audit_shared_interface_partition(region_triangles, interface)
        except Exception as error:
            witness = shared_interface_failure_witness(region_triangles, interface, certificate, predicates)
            require(False, f"RA/RV exact phase interface audit failed at step {row['step']} "
                           f"time={row['time_s']:.9g}s q={row['q'].tolist()} witness={witness}: {error}")
        require(interface_audit["interiors_disjoint"],
                f"RA/RV interior overlap during accepted frame {row['step']}")
        cavity_pairs = [pair for pair in audit["per_pair"]
                        if int(pair["first"]) in ids and int(pair["second"]) in ids and
                        {pair["first"], pair["second"]} != {str(RA_ID), str(RV_ID)}]
        require(all(pair["disjoint_closed_domains"] for pair in cavity_pairs),
                f"non-RA/RV chamber overlap during accepted frame {row['step']}")
        if wall_binding is not None:
            wall_surface_reports, wall_pair_reports, wall_audit_complete = exact_large_surface_audits(
                surfaces, predicates)
            wall_relations = [pair for pair in wall_pair_reports
                              if int(pair["first"]) in wall_ids or int(pair["second"]) in wall_ids]
            unqualified_wall_pairs = [[pair["first"], pair["second"]] for pair in wall_relations
                if pair["relation"] not in ("disjoint", "expected_cavity_enclosed_by_passive_outer_heart",
                                             "expected_ventricular_cavity_enclosed_by_passive_ventricular_wall",
                                             "nonintersecting_passive_surface_with_unclosed_source_topology")]
            wall_gate = "fail" if not wall_audit_complete or unqualified_wall_pairs or any(
                report["self_intersection_triangle_pair_count"] for report in wall_surface_reports.values()) else "pass"
            wall_check_status = "audited_complete" if wall_audit_complete else "audit_incomplete"
        else:
            wall_surface_reports, wall_relations = {}, []
            unqualified_wall_pairs = source_unqualified_pairs
            wall_gate = "fail"
            wall_check_status = "not_run_source_q0_gate_failed"
        candidate_volume_m3 = np.asarray([
            ((np.float64(polynomials[c][3]) * float(q[c]) + polynomials[c][2]) * float(q[c]) +
             polynomials[c][1]) * float(q[c]) + polynomials[c][0] for c in range(4)], dtype=np.float64)
        target_volume_m3 = row["target_volume_m3"].astype(np.float64)
        results.append({"step": row["step"], "time_s": row["time_s"],
            "native_q": [float(x) for x in native_q],
            "q": [float(x) for x in q],
            "target_volume_ml": (target_volume_m3 * 1e6).tolist(),
            "candidate_volume_ml": (candidate_volume_m3 * 1e6).tolist(),
            "relative_volume_error": ((candidate_volume_m3 - target_volume_m3) / target_volume_m3).tolist(),
            "embedded_chambers": [str(x) for x in ids],
            "RA_RV_shared_interface": interface_audit,
            "other_cavity_pair_audits": cavity_pairs,
            "passive_outer_heart_self_audits": wall_surface_reports,
            "passive_outer_heart_relations": wall_relations,
            "passive_outer_heart_unqualified_pairs": unqualified_wall_pairs,
            "passive_outer_heart_gate": wall_gate,
            "passive_outer_heart_phase_check_status": wall_check_status})
    phase_wall_gate = "fail" if any(row["passive_outer_heart_gate"] == "fail" for row in results) else "pass"
    passive_wall_gate = "fail" if source_wall_gate == "fail" or phase_wall_gate == "fail" else "pass"
    return {"four_chamber_gate": "pass", "passive_outer_heart_gate": passive_wall_gate,
            "overall_geometry_gate": "pass" if passive_wall_gate == "pass" else "unqualified",
            "source_q0_passive_outer_heart_gate": source_wall_gate,
            "source_q0_passive_outer_heart_audit_complete": source_wall_audit_complete,
            "source_q0_passive_outer_heart_self_audits": source_wall_surface_reports,
            "source_q0_passive_outer_heart_relations": source_wall_relations,
            "source_q0_passive_outer_heart_unqualified_pairs": source_unqualified_pairs,
            "phases": results}


def signed_volume(points: np.ndarray, faces: np.ndarray) -> float:
    origin = points[0]
    tri = points[faces] - origin
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum(dtype=np.float64) / 6.0)


def mesh_from_triangles(triangles, partition):
    mesh = partition.indexed_mesh(triangles, convert=float)
    points64 = np.asarray(mesh["vertices_m"], dtype=np.float64)
    points32 = points64.astype(np.float32)
    require(np.all(np.isfinite(points32)), "partition produced nonfinite FP32 vertices")
    require(len(np.unique(points32, axis=0)) == len(points32), "FP32 conversion collapsed distinct partition vertices")
    faces = np.asarray(mesh["triangles"], dtype=np.uint32)
    require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0, "empty partition surface")
    return points32, faces


def vertex_normals(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = points[faces].astype(np.float64)
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(points, dtype=np.float64)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    require(np.all(np.isfinite(lengths)) and np.all(lengths > 1e-12), "partition surface has undefined vertex normals")
    normals /= lengths[:, None]
    return normals.astype(np.float32)


def oriented_face_key(triangle):
    face = tuple(tuple(float(x) for x in point) for point in triangle)
    return min(face, face[1:] + face[:1], face[2:] + face[:2])


def shared_interface_failure_witness(regions, interface, certificate, predicates):
    """Return the first exact RA/RV contact that escapes the named cut."""
    precise = {name: [[tuple(Fraction.from_float(float(x)) for x in point) for point in tri]
                      for tri in triangles] for name, triangles in regions.items()}
    precise_interface = [[tuple(Fraction.from_float(float(x)) for x in point) for point in tri]
                         for tri in interface]
    scale = max(x.denominator for triangles in list(precise.values()) + [precise_interface]
                for tri in triangles for point in tri for x in point)
    integer = lambda triangles: [tuple(tuple(int(x * scale) for x in point) for point in tri)
                                 for tri in triangles]
    prepared = {name: certificate._Prepared(integer(triangles))
                for name, triangles in precise.items()}
    shared = integer(precise_interface)
    shared_keys = {tuple(sorted(triangle)) for triangle in shared}
    interface_vertices = {point for triangle in shared for point in triangle}
    interface_edges = {certificate._edge_key(a, b)
                       for triangle in shared for a, b in zip(triangle, triangle[1:] + triangle[:1])}
    first, second = list(prepared)
    for row in prepared[first].records:
        for other in prepared[second].candidates(row[0]):
            points = predicates.triangle_intersection_points(row[0], other[0])
            if not points:
                continue
            if tuple(sorted(row[0])) == tuple(sorted(other[0])) and tuple(sorted(row[0])) in shared_keys:
                continue
            allowed = (all(point in interface_vertices for point in points) and len(set(points)) == 1) or any(
                all(certificate._on_segment(point, a, b) for point in points) for a, b in interface_edges)
            uncut = all(any(all(certificate._on_segment(point, triangle[i], triangle[(i + 1) % 3])
                                for point in points) for i in range(3))
                        for triangle in (row[0], other[0]))
            if not allowed or not uncut:
                return {"ra_triangle_index": row[3], "rv_triangle_index": other[3],
                        "ra_triangle_m": [[float(Fraction(x, scale)) for x in point] for point in row[0]],
                        "rv_triangle_m": [[float(Fraction(x, scale)) for x in point] for point in other[0]],
                        "intersection_point_count": len(points),
                        "intersection_points_m": [[float(Fraction(x, scale)) for x in point] for point in points],
                        "allowed_declared_interface_contact": allowed,
                        "uncut_face_edge_contact": uncut}
    return None


def pack_payload(header, records, vertex_rows, faces_by_record):
    magic, abi, registration, source_sha = header
    out_records, out_vertices, out_indices = [], [], []
    for record, points_normals, faces in zip(records, vertex_rows, faces_by_record, strict=True):
        body, _, _, _, _, stable_id, layer, reserved = record
        first_vertex = len(out_vertices)
        first_index = len(out_indices)
        points = points_normals[:, :3]
        normals = points_normals[:, 3:]
        require(points.shape == normals.shape and points.shape[1] == 3, f"stable ID {stable_id} vertex rows malformed")
        out_vertices.extend(np.concatenate((points, normals), axis=1).tolist())
        out_indices.extend((faces.reshape(-1).astype(np.uint64) + first_vertex).tolist())
        out_records.append((body, first_vertex, len(points), first_index, int(faces.size), stable_id, layer, reserved))
    vertex_count, index_count = len(out_vertices), len(out_indices)
    blob = bytearray(HEADER.pack(magic, abi, len(out_records), vertex_count, index_count, registration, source_sha))
    for row in out_records:
        blob.extend(RECORD.pack(*row))
    for row in out_vertices:
        blob.extend(VERTEX.pack(*row))
    for index in out_indices:
        blob.extend(U32.pack(index))
    return bytes(blob), out_records, np.asarray(out_vertices, dtype=np.float32), np.asarray(out_indices, dtype=np.uint32)


def compile_binding(input_path: Path, receipt_path: Path, output_path: Path, output_receipt: Path,
                    numilab_source: Path, audit: bool, phase_trace: Path | None = None,
                    phase_selection: str = "representative-cycle"):
    require(not output_path.exists() and not output_receipt.exists(),
            "refusing to overwrite an existing bound anatomy payload or receipt")
    source_root = numilab_source.resolve()
    if (source_root / "src" / "numilab_human").is_dir():
        source_root = source_root / "src"
    require((source_root / "numilab_human").is_dir(),
            "pinned NumiLab source path has no numilab_human package")
    sys.path.insert(0, str(source_root))
    from numilab_human import cardiac_cavity_intersections as predicates
    from numilab_human import cardiac_cavity_partition as partition
    from numilab_human import cardiac_partition_certificate as certificate

    raw, header, records, vertices, indices = read_payload(input_path)
    receipt = json.loads(receipt_path.read_text())
    source_hash = hashlib.sha256(raw).hexdigest()
    require(receipt.get("payload", {}).get("sha256") == source_hash, "input receipt does not bind the NHANAT bytes")
    require(receipt.get("functional_bindings", {}).get("anatomy_payload_sha256") == source_hash,
            "functional bindings do not bind the NHANAT bytes")
    provenance = receipt.get("provenance", {})
    source_map = provenance.get("source_id_map", {})
    ra_source = source_map.get(str(RA_ID), {})
    rv_source = source_map.get(str(RV_ID), {})
    require(ra_source.get("source_member") == "FJ2424" and rv_source.get("source_member") == "FJ2423",
            "cardiac source-member identities differ")

    _, ra_points, ra_faces = surface_arrays(records, vertices, indices, RA_ID)
    _, rv_points, rv_faces = surface_arrays(records, vertices, indices, RV_ID)
    ra_src = {"vertices": [tuple(map(float, p)) for p in ra_points],
              "triangles": [tuple(map(int, face)) for face in ra_faces],
              "source_sha256": ra_source["source_sha256"]}
    rv_src = {"vertices": [tuple(map(float, p)) for p in rv_points],
              "triangles": [tuple(map(int, face)) for face in rv_faces],
              "source_sha256": rv_source["source_sha256"]}

    arrangement, construction = partition.construct_arrangement(
        {"right_atrium": ra_src, "right_ventricle": rv_src})
    ra_triangles = [row["vertices"] for row in arrangement if row["source"] == "right_atrium"]
    shared_ra = [row["vertices"] for row in arrangement
                 if row["source"] == "right_atrium" and row["other_location"] == "inside"]
    rv_triangles = [row["vertices"] for row in arrangement
                    if row["source"] == "right_ventricle" and row["other_location"] == "outside"]
    rv_triangles.extend((triangle[0], triangle[2], triangle[1]) for triangle in shared_ra)
    ra_new_points, ra_new_faces = mesh_from_triangles(ra_triangles, partition)
    rv_new_points, rv_new_faces = mesh_from_triangles(rv_triangles, partition)
    interface_points, interface_faces = mesh_from_triangles(shared_ra, partition)
    require(abs(signed_volume(ra_new_points, ra_new_faces) - signed_volume(ra_points, ra_faces)) < 2e-10,
            "RA-priority partition changed right atrium source volume")
    original_rv_volume = signed_volume(rv_points, rv_faces)
    new_rv_volume = signed_volume(rv_new_points, rv_new_faces)
    require(new_rv_volume > 0 and original_rv_volume > new_rv_volume,
            "RA-priority partition did not remove positive duplicated RV volume")
    shared_volume = abs(original_rv_volume - new_rv_volume)
    require(construction["source_intersecting_triangle_pair_count"] > 0,
            "the registered RA/RV source surfaces no longer intersect")

    # Replace the two intersecting source records with their arrangement
    # meshes. All other anatomy arrays retain their source order and ownership.
    out_records, vertex_rows, faces_by_record = [], [], []
    for record in records:
        body, first_vertex, vertex_count, first_index, index_count, stable_id, layer, reserved = record
        local_points = vertices[first_vertex:first_vertex + vertex_count, :3].copy()
        local_normals = vertices[first_vertex:first_vertex + vertex_count, 3:].copy()
        local_faces = (indices[first_index:first_index + index_count].reshape(-1, 3) - first_vertex).copy()
        if stable_id == RA_ID:
            local_points = ra_new_points
            local_normals = vertex_normals(ra_new_points, ra_new_faces)
            local_faces = ra_new_faces
        elif stable_id == RV_ID:
            local_points = rv_new_points
            local_normals = vertex_normals(rv_new_points, rv_new_faces)
            local_faces = rv_new_faces
        vertex_rows.append(np.concatenate((local_points, local_normals), axis=1).astype(np.float32))
        faces_by_record.append(np.asarray(local_faces, dtype=np.uint32))
        out_records.append(record)
    packed, packed_records, packed_vertices, packed_indices = pack_payload(
        header, out_records, vertex_rows, faces_by_record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(packed)
    output_hash = hashlib.sha256(packed).hexdigest()
    output_receipt_data = json.loads(json.dumps(receipt))
    output_receipt_data["functional_bindings"]["anatomy_payload_sha256"] = output_hash
    payload_row = output_receipt_data["payload"]
    payload_row.update({"path": str(output_path), "sha256": output_hash,
                        "vertex_count": len(packed_vertices), "index_count": len(packed_indices)})
    output_receipt_data.setdefault("provenance", {})["cardiac_geometry_binding"] = {
        "method": "exact_source_face_arrangement_with_RA_priority",
        "input_anatomy_payload_sha256": source_hash,
        "output_anatomy_payload_sha256": output_hash,
        "source_member_bindings": [
            {"stable_id": RA_ID, "source_member": "FJ2424", "fma_id": "FMA11359", "source_sha256": ra_source["source_sha256"]},
            {"stable_id": RV_ID, "source_member": "FJ2423", "fma_id": "FMA9291", "source_sha256": rv_source["source_sha256"]},
            {"stable_id": LA_ID, "source_member": "FJ2425", "fma_id": "FMA9465"},
            {"stable_id": LV_ID, "source_member": "FJ2422", "fma_id": "FMA9466"},
        ],
        "passive_outer_heart_bindings": [
            {"stable_id": 1, "source_member": "FJ2439", "fma_id": "FMA7088",
             "source_sha256": source_map["1"]["source_sha256"]},
            {"stable_id": 23, "source_member": "FJ2428", "fma_id": "FMA13884",
             "source_sha256": source_map["23"]["source_sha256"]},
            {"stable_id": 24, "source_member": "FJ2438", "fma_id": "FMA7088",
             "source_sha256": source_map["24"]["source_sha256"]},
        ],
        "cavity_motion": "source-centroid radial free-wall dilation using the explicit inferred registration parameters below; positive radius and exact accepted-phase embedding are required",
        "cavity_motion_parameters": {
            "model": CARDIAC_MOTION_MODEL,
            "ra_rv_taper_degrees": RA_RV_TAPER_DEGREES,
            "ra_center_offset_m": RA_CENTER_OFFSET_M,
            "ra_center_offset_direction": "unit(source_RA_centroid - source_RV_centroid)",
            "left_center_offsets_m": [LEFT_ATRIUM_CENTER_OFFSET_M, LEFT_VENTRICLE_CENTER_OFFSET_M],
            "left_center_offset_direction": "unit(source_LA_centroid - source_LV_centroid)",
            "ra_rv_shared_interface_weight": 0.0,
            "left_chamber_freewall_weight": 1.0,
            "parameter_status": "inferred_reference_registration_not_measured_subject_geometry",
        },
        "passive_outer_heart_motion": "presentation-only closest-cavity-surface displacement driven by the corresponding accepted CVSim chamber q; exact-coordinate-quotiented source surfaces are audited for self/cavity intersections; no myocardium constitutive mechanics or enclosure claim is inferred from open source boundaries",
        "ownership_choice": "right atrium retains the exact source overlap; right ventricle retains its source-exclusive region and a reversed copy of the RA shared boundary",
        "source_overlap_triangle_pair_count": construction["source_intersecting_triangle_pair_count"],
        "source_overlap_volume_ml": shared_volume * 1e6,
        "right_atrium_volume_m3": signed_volume(ra_new_points, ra_new_faces),
        "right_ventricle_volume_before_m3": original_rv_volume,
        "right_ventricle_volume_after_m3": new_rv_volume,
        "retained_geometry": "all non-RA/RV records, including the source-lobe 306 FP32 patch, are copied in original record order; RA/RV replace only their exact arrangement surfaces",
        "interpretation": "inferred geometric ownership of an existing duplicate source volume; not a measured subject valve plane or a biological annulus",
        "mechanical_mass_or_volume_changed": False,
        "new_payload_format": False,
        "physical_solver_changed": False,
    }

    # Re-read the emitted bytes to make the actual binary32 geometry, not the
    # exact authoring fractions, the admission witness.
    out_header = HEADER.unpack_from(packed)
    require(out_header[2] == len(packed_records), "emitted NHANAT record count changed")
    out_at = HEADER.size + len(packed_records) * RECORD.size
    emitted_vertices = np.frombuffer(packed, dtype="<f4", count=len(packed_vertices) * 6, offset=out_at).reshape(-1, 6)
    out_at += len(packed_vertices) * VERTEX.size
    emitted_indices = np.frombuffer(packed, dtype="<u4", count=len(packed_indices), offset=out_at)
    surfaces = []
    for stable_id in (RA_ID, RV_ID, LA_ID, LV_ID):
        rec = next(row for row in packed_records if row[5] == stable_id)
        _, first_vertex, vertex_count, first_index, index_count, *_ = rec
        points = emitted_vertices[first_vertex:first_vertex + vertex_count, :3].astype(np.float64)
        faces = (emitted_indices[first_index:first_index + index_count].reshape(-1, 3) - first_vertex).astype(np.int64)
        require(abs(signed_volume(points, faces)) > 1e-8, f"emitted cavity {stable_id} has invalid volume")
        surfaces.append({"source_id": str(stable_id), "exact_coordinate_quotient":
                         {"vertices_m": points.tolist(), "triangles": faces.tolist()}})
    pair_audit = None
    if audit:
        pair_audit = predicates.audit_cavity_intersections({"chambers": surfaces})
        require(pair_audit["all_surfaces_embedded"], "emitted four-chamber geometry is not embedded")
        ra_rows = next(row for row in surfaces if row["source_id"] == str(RA_ID))["exact_coordinate_quotient"]
        rv_rows = next(row for row in surfaces if row["source_id"] == str(RV_ID))["exact_coordinate_quotient"]
        region_triangles = lambda mesh: [[mesh["vertices_m"][int(i)] for i in face] for face in mesh["triangles"]]
        ra_triangles_emitted = region_triangles(ra_rows)
        rv_triangles_emitted = region_triangles(rv_rows)
        interface_triangles = interface_points[interface_faces].astype(np.float64).tolist()
        interface_keys = Counter(oriented_face_key(triangle) for triangle in interface_triangles)
        reverse_keys = Counter(oriented_face_key((triangle[0], triangle[2], triangle[1]))
                               for triangle in interface_triangles)
        ra_keys = Counter(oriented_face_key(triangle) for triangle in ra_triangles_emitted)
        rv_keys = Counter(oriented_face_key(triangle) for triangle in rv_triangles_emitted)
        missing_ra = sum(count for key, count in interface_keys.items() if ra_keys[key] != count)
        missing_rv = sum(count for key, count in reverse_keys.items() if rv_keys[key] != count)
        require(missing_ra == 0 and missing_rv == 0,
                f"RA/RV interface face multiplicity mismatch: interface={sum(interface_keys.values())}, "
                f"missing_from_RA={missing_ra}, missing_reversed_from_RV={missing_rv}")
        ra_rv_audit = certificate.audit_shared_interface_partition(
            {"right_atrium": ra_triangles_emitted, "right_ventricle": rv_triangles_emitted},
            interface_triangles)
        require(ra_rv_audit["interiors_disjoint"], "RA-priority emitted shared-interface audit failed")
        other_pairs = [row for row in pair_audit["per_pair"]
                       if {row["first"], row["second"]} != {str(RA_ID), str(RV_ID)}]
        require(all(row["disjoint_closed_domains"] for row in other_pairs),
                "emitted cavity domains overlap away from the explicitly allowlisted RA/RV source interface")
        output_receipt_data["provenance"]["cardiac_geometry_binding"]["emitted_four_cavity_audit"] = {
            "all_surfaces_embedded": pair_audit["all_surfaces_embedded"],
            "pair_audits": pair_audit["per_pair"],
            "RA_RV_shared_interface": ra_rv_audit,
            "allowlisted_interface": "exact common RA source-surface patch used to close the RA-priority partition; inferred geometric cut, not an anatomical valve annulus",
            "exact_predicate": "numilab_human.cardiac_cavity_intersections on emitted FP32 mesh coordinates",
        }

    phase_audits = None
    if phase_trace is not None:
        require(audit and pair_audit is not None,
                "accepted cardiac phase audit requires the exact static cavity audit")
        source_rows, sampled_rows = phase_rows(phase_trace, phase_selection)
        phase_audits = audit_dynamic_phases(packed_records, packed_vertices, packed_indices,
            source_rows, sampled_rows, interface_points, interface_faces, predicates, certificate)
        output_receipt_data["provenance"]["cardiac_geometry_binding"]["accepted_cycle_geometry_audit"] = {
            "accepted_trace_sha256": hashlib.sha256(phase_trace.read_bytes()).hexdigest(),
            "accepted_trace_row_count": len(source_rows),
            "phase_sample_count": len(phase_audits["phases"]),
            "phase_selection": phase_selection,
            "sampling": ("initial accepted state and observed LV q maximum" if phase_selection == "initial-and-lv-max"
                         else "eight time-spaced accepted frames plus observed per-chamber q minima and maxima"),
            "q_mapping": "native FP32 cubic and 28-iteration bisection reconstructed from same-frame accepted chamber volume columns; geometry uses the receipt-bound inferred 20-degree/8mm/12mm/15mm source-centroid map",
            "dynamic_surface_source": "emitted FP32 NHANAT5 cavity vertices and faces",
            "predicate": "NumiLab exact cavity embeddedness, pair intersection and RA/RV shared-interface certificate",
            "four_chamber_gate": phase_audits["four_chamber_gate"],
            "passive_outer_heart_gate": phase_audits["passive_outer_heart_gate"],
            "source_q0_passive_outer_heart_gate": phase_audits["source_q0_passive_outer_heart_gate"],
            "overall_geometry_gate": phase_audits["overall_geometry_gate"],
            "source_q0_passive_outer_heart_audit_complete": phase_audits[
                "source_q0_passive_outer_heart_audit_complete"],
            "source_q0_passive_outer_heart_self_audits": phase_audits["source_q0_passive_outer_heart_self_audits"],
            "source_q0_passive_outer_heart_relations": phase_audits["source_q0_passive_outer_heart_relations"],
            "source_q0_passive_outer_heart_unqualified_pairs": phase_audits["source_q0_passive_outer_heart_unqualified_pairs"],
            "phases": phase_audits["phases"],
        }

    output_receipt.write_text(json.dumps(output_receipt_data, sort_keys=True, separators=(",", ":")) + "\n")
    return {
        "input_payload_sha256": source_hash,
        "output_payload_sha256": output_hash,
        "output_receipt_sha256": hashlib.sha256(output_receipt.read_bytes()).hexdigest(),
        "input_surface_count": len(records),
        "output_surface_count": len(packed_records),
        "input_vertex_count": len(vertices),
        "output_vertex_count": len(packed_vertices),
        "input_index_count": len(indices),
        "output_index_count": len(packed_indices),
        "RA_RV_source_intersection_triangle_pairs": construction["source_intersecting_triangle_pair_count"],
        "source_shared_volume_ml": shared_volume * 1e6,
        "right_atrium_volume_ml": signed_volume(ra_new_points, ra_new_faces) * 1e6,
        "right_ventricle_volume_before_ml": original_rv_volume * 1e6,
        "right_ventricle_volume_after_ml": new_rv_volume * 1e6,
        "four_cavity_embeddedness_audit": pair_audit,
        "accepted_cycle_phase_count": len(phase_audits["phases"]) if phase_audits is not None else 0,
        "accepted_cycle_outer_wall_gate": phase_audits["passive_outer_heart_gate"] if phase_audits is not None else "not_audited",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="merged registered NHANAT5 payload")
    parser.add_argument("--input-receipt", type=Path, required=True, help="existing source-bound anatomy receipt")
    parser.add_argument("--output", type=Path, required=True, help="emitted NHANAT5 payload path")
    parser.add_argument("--output-receipt", type=Path, required=True, help="updated existing-format anatomy receipt path")
    parser.add_argument("--numilab-source", type=Path, required=True, help="pinned NumiLab source tree for exact partition predicates")
    parser.add_argument("--phase-trace", type=Path, help="accepted resting-surface-audit.csv to audit sampled cardiac geometry through the recorded cycle")
    parser.add_argument("--phase-selection", choices=("representative-cycle", "initial-and-lv-max"),
                        default="representative-cycle", help="select a bounded initial/LV-maximum check before the full representative-cycle audit")
    parser.add_argument("--skip-exact-audit", action="store_true", help="skip exact four-cavity collision admission")
    args = parser.parse_args(argv)
    result = compile_binding(args.input, args.input_receipt, args.output, args.output_receipt,
                             args.numilab_source, audit=not args.skip_exact_audit,
                             phase_trace=args.phase_trace, phase_selection=args.phase_selection)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
