#!/usr/bin/env python3
"""Exact-rational audit of authored FP32 mortar maps captured in run 318.

This script audits input geometry only. It does not run the solver, emulate the
Metal contact kernel, or qualify any collision/CCD result.
"""
import argparse
import hashlib
import json
import struct
from fractions import Fraction
from pathlib import Path

RECORD_BYTES = 384
SIDE_A_WEIGHTS_OFFSET = 224
SIDE_B_WEIGHTS_OFFSET = 272
PARENT = [(Fraction(1), Fraction(0)),
          (Fraction(0), Fraction(1)),
          (Fraction(0), Fraction(0))]


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def signed_area(poly):
    if len(poly) < 3:
        return Fraction(0)
    twice = sum(poly[i][0] * poly[(i + 1) % len(poly)][1]
                - poly[(i + 1) % len(poly)][0] * poly[i][1]
                for i in range(len(poly)))
    return twice / 2


def area(poly):
    return abs(signed_area(poly))


def clip_convex(subject, clipper):
    """Exact Sutherland-Hodgman clipping for convex polygons over Fractions."""
    orientation = 1 if signed_area(clipper) >= 0 else -1
    out = list(subject)
    for i, a in enumerate(clipper):
        b = clipper[(i + 1) % len(clipper)]
        inp, out = out, []
        if not inp:
            break
        for j, e in enumerate(inp):
            s = inp[j - 1]
            ds = orientation * cross(a, b, s)
            de = orientation * cross(a, b, e)
            s_inside, e_inside = ds >= 0, de >= 0
            if s_inside != e_inside:
                denom = ds - de
                if denom:
                    t = ds / denom
                    out.append((s[0] + t * (e[0] - s[0]),
                                s[1] + t * (e[1] - s[1])))
            if e_inside:
                out.append(e)
    # Remove consecutive exact duplicate vertices to keep subsequent passes tidy.
    clean = []
    for point in out:
        if not clean or point != clean[-1]:
            clean.append(point)
    if len(clean) > 1 and clean[0] == clean[-1]:
        clean.pop()
    return clean


def frac_text(value):
    return f"{value.numerator}/{value.denominator}"


def f32_fraction(value):
    # struct.unpack('<f') returns the exactly represented binary32 value widened
    # to binary64, so Fraction.from_float recovers that exact dyadic rational.
    return Fraction.from_float(float(value))


def parse_capture(path, wanted):
    raw = Path(path).read_bytes()
    for line_number, raw_line in enumerate(raw.splitlines(), 1):
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if record.get("case") != wanted:
            continue
        fields = {field["name"]: bytes.fromhex(field["hex"])
                  for field in record["input_buffers"]["fields"]}
        faces = fields["cohesive_faces_buffer_9"]
        if len(faces) % RECORD_BYTES:
            raise ValueError("captured cohesive-face bytes are not 384-byte records")
        polygons = []
        rows3 = []
        for index in range(len(faces) // RECORD_BYTES):
            rec = faces[index * RECORD_BYTES:(index + 1) * RECORD_BYTES]
            weights_a = struct.unpack_from("<12f", rec, SIDE_A_WEIGHTS_OFFSET)
            weights_b = struct.unpack_from("<12f", rec, SIDE_B_WEIGHTS_OFFSET)
            if weights_a != weights_b:
                raise ValueError(f"face {index}: side-A/B maps differ in captured FP32 input")
            rows = []
            for vertex in range(3):
                row = tuple(f32_fraction(weights_a[vertex * 4 + lane])
                            for lane in range(3))
                rows.append(row)
            rows3.append(rows)
            polygons.append([(row[0], row[1]) for row in rows])
        return raw, record, polygons, rows3, line_number
    raise ValueError(f"capture has no case named {wanted!r}")


def audit_case(path, name):
    raw, record, polygons, rows3, line = parse_capture(path, name)
    parent_area = area(PARENT)
    each_area = [area(poly) for poly in polygons]
    pair_intersections = []
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            intersection = clip_convex(polygons[i], polygons[j])
            pair_intersections.append({"faces": [i, j],
                                       "intersection_area": frac_text(area(intersection))})
    overlap_total = sum((area(clip_convex(polygons[i], polygons[j]))
                         for i in range(len(polygons))
                         for j in range(i + 1, len(polygons))), Fraction(0))
    total_area = sum(each_area, Fraction(0))
    xyz_sums = [sum(row, Fraction(0)) for group in rows3 for row in group]
    mapped_within = all(x >= 0 and y >= 0 and x + y <= 1
                        for poly in polygons for x, y in poly)
    # Parent containment follows every vertex lying in the parent simplex;
    # each mapped face is convex.
    result = {
        "case": name,
        "source_log_line": line,
        "face_count": len(polygons),
        "captured_maps_side_a_equals_side_b_bitwise": True,
        "all_mapped_vertices_within_parent_simplex_exactly": mapped_within,
        "parent_xy_barycentric_area": frac_text(parent_area),
        "patch_areas": [frac_text(a) for a in each_area],
        "patch_area_sum": frac_text(total_area),
        "sum_area_equals_parent": total_area == parent_area,
        "sum_pairwise_intersection_areas": frac_text(overlap_total),
        "pairwise_intersections": pair_intersections,
        "xyz_vertex_weight_sums": [frac_text(s) for s in xyz_sums],
        "all_xyz_weight_sums_equal_one": all(s == 1 for s in xyz_sums),
        "exact_map_conclusion": (
            "zero planar complement and no positive-area overlap in exact rational "
            "interpretation of captured binary32 x/y maps"
            if mapped_within and total_area == parent_area and overlap_total == 0
            else "map area/containment audit did not prove a complete nonoverlapping partition"
        ),
        "input_status": record.get("actual"),
        "input_record_sha256": hashlib.sha256(json.dumps(
            record, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "stdout_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--case", action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = {"schema": "numi.cardboard.mortar_exact_map_audit.v1",
              "scope": "authored input maps only; no Metal partition or CCD claim",
              "cases": [audit_case(args.capture, case) for case in args.case]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
