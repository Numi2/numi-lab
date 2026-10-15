#!/usr/bin/env python3
"""Decode native private-sweep inputs and exact-check their mapped patch tilings."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


SCALE = 1 << 25
FACE_BYTES = 384
PRIMITIVE_BYTES = 64
FACE_NODES = 0
FACE_SIDE_B_NODES = 64
SIDE_A_WEIGHTS = 224
SIDE_B_WEIGHTS = 272


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def u32(data: bytes, offset: int, count: int) -> tuple[int, ...]:
    return struct.unpack_from("<" + "I" * count, data, offset)


def f32(data: bytes, offset: int, count: int) -> tuple[float, ...]:
    return struct.unpack_from("<" + "f" * count, data, offset)


def q25(value: float) -> int | None:
    numerator, denominator = value.as_integer_ratio()
    scaled = numerator * SCALE
    if scaled % denominator:
        return None
    return scaled // denominator


def orient(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def separated(axis_a: tuple[int, int], axis_b: tuple[int, int],
               sign: int, tri: list[tuple[int, int]]) -> bool:
    return all(orient(axis_a, axis_b, p) * sign <= 0 for p in tri)


def no_positive_overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> bool:
    area_a = orient(*a)
    area_b = orient(*b)
    if area_a == 0 or area_b == 0:
        return False
    for tri, other, sign in ((a, b, 1 if area_a > 0 else -1),
                             (b, a, 1 if area_b > 0 else -1)):
        for edge in range(3):
            if separated(tri[edge], tri[(edge + 1) % 3], sign, other):
                break
        else:
            return False
    return True


def exact_cover(raw_triangles: list[list[tuple[float, float]]]) -> dict:
    coordinates: list[list[tuple[int, int]]] = []
    for raw in raw_triangles:
        converted = []
        for x, y in raw:
            ix, iy = q25(x), q25(y)
            if ix is None or iy is None or ix < 0 or iy < 0 or ix + iy > SCALE:
                return {"valid": False, "reason": "not exact Q25/simplex"}
            converted.append((ix, iy))
        area = orient(*converted)
        if area == 0:
            return {"valid": False, "reason": "degenerate triangle"}
        if area < 0:
            converted[1], converted[2] = converted[2], converted[1]
        coordinates.append(converted)

    edges = []
    for triangle in coordinates:
        edges.extend((triangle[i], triangle[(i + 1) % 3]) for i in range(3))
    matched = [False] * len(edges)
    boundary = [[], [], []]
    for index, (a, b) in enumerate(edges):
        if matched[index]:
            continue
        reverse = next((j for j in range(index + 1, len(edges))
                        if not matched[j] and edges[j] == (b, a)), None)
        if reverse is not None:
            matched[index] = matched[reverse] = True
            continue
        ax, ay = a
        bx, by = b
        if ay == 0 and by == 0 and ax < bx:
            side, lo, hi = 0, ax, bx
        elif ax == 0 and bx == 0 and ay > by:
            side, lo, hi = 1, by, ay
        elif ax + ay == SCALE and bx + by == SCALE and ay < by:
            side, lo, hi = 2, ay, by
        else:
            return {"valid": False, "reason": "unmatched interior directed edge",
                    "edge": [list(a), list(b)]}
        boundary[side].append((lo, hi))
        matched[index] = True

    for side, intervals in enumerate(boundary):
        intervals.sort()
        if not intervals or intervals[0][0] != 0 or intervals[-1][1] != SCALE:
            return {"valid": False, "reason": "parent boundary endpoint missing",
                    "side": side, "intervals": intervals}
        if any(intervals[i][1] != intervals[i + 1][0]
               for i in range(len(intervals) - 1)):
            return {"valid": False, "reason": "parent boundary gap or overlap",
                    "side": side, "intervals": intervals}

    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            if not no_positive_overlap(coordinates[i], coordinates[j]):
                return {"valid": False, "reason": "pairwise positive-area overlap",
                        "pair": [i, j]}
    return {
        "valid": True,
        "proof": "exact Q25 directed-edge cancellation and parent-boundary tiling; pairwise SAT",
        "q25_triangles": [[[x, y] for x, y in tri] for tri in coordinates],
    }


def mapped_patches(fields: dict[str, bytes], side: str, patch_count: int) -> tuple[list, list]:
    faces = fields["cohesive_faces_buffer_9"]
    primitives = fields["primitives_buffer_2"]
    grid_node_count = u32(fields["dispatch_buffer_0"], 28, 1)[0]
    if len(faces) != patch_count * FACE_BYTES:
        raise ValueError(f"face buffer has {len(faces)} bytes, expected {patch_count * FACE_BYTES}")
    primitive_index = 0 if side == "A" else 4
    primitive_nodes = u32(primitives, primitive_index * PRIMITIVE_BYTES, 3)
    node_offset = FACE_NODES if side == "A" else FACE_SIDE_B_NODES
    weight_offset = SIDE_A_WEIGHTS if side == "A" else SIDE_B_WEIGHTS
    triangles = []
    normalization = []
    for patch in range(patch_count):
        face = faces[patch * FACE_BYTES:(patch + 1) * FACE_BYTES]
        face_nodes = u32(face, node_offset, 3)
        mapped = []
        for vertex in range(3):
            source = f32(face, weight_offset + vertex * 16, 3)
            reordered = [None, None, None]
            for corner, node in enumerate(face_nodes):
                unified_node = grid_node_count + node
                if unified_node not in primitive_nodes:
                    raise ValueError("face map node absent from parent primitive")
                reordered[primitive_nodes.index(unified_node)] = source[corner]
            if any(value is None for value in reordered):
                raise ValueError("incomplete parent barycentric map")
            x, y, z = reordered
            reconstructed_z = struct.unpack("<f", struct.pack("<f", struct.unpack("<f", struct.pack("<f", 1.0 - x))[0] - y))[0]
            normalization.append({
                "patch": patch,
                "vertex": vertex,
                "source_z_f32": z,
                "xy_reconstructed_z_f32": reconstructed_z,
                "source_minus_xy_reconstructed_z": z - reconstructed_z,
            })
            mapped.append((x, y))
        triangles.append(mapped)
    return triangles, normalization


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases_jsonl", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    selected = {"three_patch_covered_crossing", "four_patch_matched_crossing",
                "three_patch_boundary_common_translation", "three_patch_boundary_separating"}
    results = []
    for line in args.cases_jsonl.read_text().splitlines():
        case = json.loads(line)
        if case.get("case") not in selected:
            continue
        fields = {item["name"]: bytes.fromhex(item["hex"])
                  for item in case["input_buffers"]["fields"]}
        count = 4 if case["case"] == "four_patch_matched_crossing" else 3
        sides = {}
        for side in ("A", "B"):
            triangles, normalization = mapped_patches(fields, side, count)
            sides[side] = {
                "triangles_xy_f32": [[[x, y] for x, y in triangle]
                                     for triangle in triangles],
                "exact_cover": exact_cover(triangles),
                "source_normalization_residuals": normalization,
            }
        results.append({"case": case["case"], "native_expected": case["expected"],
                        "native_actual": case["actual"], "sides": sides})
    if len(results) != len(selected):
        raise SystemExit(f"expected {len(selected)} selected cases, found {len(results)}")
    output = {
        "schema": "numi.matter.mortar_exact_tiling_input_audit.v1",
        "classification": "CPU exact-input geometry audit; not native re-execution or qualification",
        "input_cases_jsonl": str(args.cases_jsonl),
        "input_cases_jsonl_sha256": sha256(args.cases_jsonl),
        "face_record_bytes": FACE_BYTES,
        "source_field_offsets_bytes": {"nodesAndFirst": 0, "separatedNodes": 64,
                                       "mortarSideAWeights": 224,
                                       "mortarSideBWeights": 272},
        "coordinate_contract": "Q25 is applied to captured FP32 XY values exactly; z normalization residuals are reported, not hidden",
        "cases": results,
    }
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.output_json),
                      "cases": len(results),
                      "all_left_right_covers_exact": all(
                          c["sides"][side]["exact_cover"]["valid"]
                          for c in results for side in ("A", "B"))}, sort_keys=True))
    return 0 if all(c["sides"][side]["exact_cover"]["valid"]
                    for c in results for side in ("A", "B")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
