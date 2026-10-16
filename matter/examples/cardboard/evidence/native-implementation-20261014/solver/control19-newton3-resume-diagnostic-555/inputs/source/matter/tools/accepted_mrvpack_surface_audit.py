#!/usr/bin/env python3
"""Exact triangle-pair checks on selected surfaces in an accepted MRVPACK2."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import mmap
import json
import struct
import sys
import heapq
from pathlib import Path

HEADER = struct.Struct("<8sIIQQ32s24s")
DIRECTORY = struct.Struct("<IIQQQII32s")


def read_pack(path: Path):
    stream = path.open("rb")
    mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    header = HEADER.unpack_from(mapped, 0)
    if header[0] != b"MRVPACK2" or header[1] != 2:
        raise ValueError(f"unsupported pack header: {header[:3]}")
    sections = [DIRECTORY.unpack_from(mapped, HEADER.size + i * DIRECTORY.size)
                for i in range(header[2])]
    by_type = {row[0]: row for row in sections}
    vertices_section, indices_section, primitives_section = (
        by_type[2], by_type[3], by_type[4])
    if vertices_section[5] != 80 or indices_section[5] != 4 or primitives_section[5] != 64:
        raise ValueError("unexpected MRVPACK2 section stride")
    surfaces = {}
    for i in range(primitives_section[4]):
        raw_offset = primitives_section[2] + i * primitives_section[5]
        geom = struct.unpack_from("<4I", mapped, raw_offset)
        identity = struct.unpack_from("<4I", mapped, raw_offset + 16)
        first_index, index_count = geom[:2]
        semantic, instance, link, stable_id = identity
        if index_count % 3:
            raise ValueError(f"non-triangle index count in primitive {identity}")
        key = (semantic, stable_id)
        if key in surfaces:
            raise ValueError(f"ambiguous repeated primitive identity {key}")
        if first_index + index_count > indices_section[4]:
            raise ValueError(f"index range out of bounds for {key}")
        faces = []
        for index in range(first_index, first_index + index_count, 3):
            faces.append(struct.unpack_from("<3I", mapped, indices_section[2] + index * 4))
        if faces and max(max(face) for face in faces) >= vertices_section[4]:
            raise ValueError(f"vertex index out of range for {key}")
        surfaces[key] = {"instance": instance, "link": link, "faces": faces}
    return mapped, stream, vertices_section[2], surfaces


def predicate_module(path: Path):
    """Load the predicate and its relative imports from the exact supplied tree."""
    path = path.resolve(strict=True)
    package_dir = path.parent
    package_init = package_dir / "__init__.py"
    if not package_init.is_file():
        raise ValueError(f"predicate package initializer is absent: {package_init}")
    namespace = "_accepted_mrvpack_predicate_" + hashlib.sha256(
        str(package_dir).encode("utf-8")).hexdigest()[:16]
    package_spec = importlib.util.spec_from_file_location(
        namespace, package_init, submodule_search_locations=[str(package_dir)])
    if package_spec is None or package_spec.loader is None:
        raise ValueError(f"cannot load predicate package from {package_dir}")
    package = importlib.util.module_from_spec(package_spec)
    sys.modules[namespace] = package
    package_spec.loader.exec_module(package)

    module_name = f"{namespace}.{path.stem}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"cannot load predicate source {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    loaded_path = Path(module.__file__).resolve(strict=True)
    if loaded_path != path:
        raise ValueError(f"loaded predicate source mismatch: {loaded_path} != {path}")
    for symbol in ("_records", "_audit_pair", "triangle_intersection_points"):
        if not callable(getattr(module, symbol, None)):
            raise ValueError(f"predicate source lacks callable {symbol}: {path}")
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_accepted_receipt(pack_path: Path, receipt_path: Path, requested_step: int,
                              mapped, vertex_offset: int, surfaces):
    """Fail closed unless the receipt identifies this exact accepted pack and vertices."""
    pack_path = pack_path.resolve(strict=True)
    receipt_path = receipt_path.resolve(strict=True)
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    receipt_pack_path = Path(receipt.get("accepted_pack_path", ""))
    if not receipt_pack_path.is_absolute() or receipt_pack_path.resolve() != pack_path:
        raise ValueError("accepted receipt pack path does not identify the supplied MRVPACK")
    if receipt.get("accepted_step") != requested_step:
        raise ValueError("accepted receipt step does not match the requested audit step")
    root_fingerprint = receipt.get("accepted_root_fingerprint")
    if not isinstance(root_fingerprint, int) or isinstance(root_fingerprint, bool):
        raise ValueError("accepted receipt has no accepted root fingerprint")
    transaction = receipt.get("accepted_transaction_fingerprint")
    if transaction is not None and (not isinstance(transaction, int) or isinstance(transaction, bool)):
        raise ValueError("accepted transaction fingerprint is not an integer")

    pack_sha256 = sha256_file(pack_path)
    if receipt.get("accepted_pack_file_sha256"):
        if receipt["accepted_pack_file_sha256"] != pack_sha256:
            raise ValueError("accepted receipt pack SHA-256 does not match supplied MRVPACK")
    elif receipt_pack_path != pack_path:
        # resolve() matched above, but a non-canonical alias is not useful provenance
        # when the receipt has no independent whole-pack digest.
        raise ValueError("accepted receipt path alias is ambiguous without a pack digest")

    section_rows = _pack_sections(mapped)
    vertex_row_count = next((row[4] for row in section_rows if row[0] == 2), None)
    index_row_count = next((row[4] for row in section_rows if row[0] == 3), None)
    if vertex_row_count is None or index_row_count is None:
        raise ValueError("accepted MRVPACK lacks vertex or index sections")
    if receipt.get("vertex_count") != vertex_row_count:
        raise ValueError("accepted receipt vertex count does not match MRVPACK")
    if receipt.get("index_count") != index_row_count:
        raise ValueError("accepted receipt index count does not match MRVPACK")
    vertex_bytes = mapped[vertex_offset:vertex_offset + vertex_row_count * 80]
    vertex_sha256 = hashlib.sha256(vertex_bytes).hexdigest()
    if receipt.get("captured_vertex_buffer_sha256") != vertex_sha256:
        raise ValueError("accepted receipt vertex-buffer SHA-256 does not match MRVPACK")
    return {
        "path": str(receipt_path),
        "sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "accepted_step": requested_step,
        "accepted_time_s": receipt.get("accepted_time_s"),
        "accepted_root_fingerprint": root_fingerprint,
        "accepted_root_fingerprint_hex": receipt.get("accepted_root_fingerprint_hex"),
        "accepted_transaction_fingerprint": transaction,
        "accepted_body_state_sha256": receipt.get("accepted_body_state_sha256"),
        "accepted_respiration_state_sha256": receipt.get("accepted_respiration_state_sha256"),
        "base_pack_content_hash": receipt.get("base_pack_content_hash"),
        "base_pack_file_sha256": receipt.get("base_pack_file_sha256"),
        "anatomy_layer_index": receipt.get("anatomy_layer_index"),
        "captured_vertex_buffer_sha256": vertex_sha256,
    }


def _pack_sections(mapped):
    header = HEADER.unpack_from(mapped, 0)
    return [DIRECTORY.unpack_from(mapped, HEADER.size + i * DIRECTORY.size)
            for i in range(header[2])]


def coordinate_lattice_scale(values):
    """Return an exact common integer scale for finite binary32 coordinates."""
    scale = 1
    for value in values:
        denominator = value.as_integer_ratio()[1]
        if denominator & (denominator - 1):
            raise ValueError(f"coordinate denominator is not a power of two: {denominator}")
        scale = max(scale, denominator)
    return scale


def lattice_integer(value, scale):
    numerator, denominator = value.as_integer_ratio()
    quotient, remainder = divmod(scale, denominator)
    if remainder:
        raise ValueError("coordinate is not exactly representable on the common lattice")
    return numerator * quotient


def vector_arg(value, count=3):
    result = tuple(float(item) for item in value.split(","))
    if len(result) != count:
        raise ValueError(f"expected {count} comma-separated values: {value}")
    return result


def rotate(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


def surface_points(mapped, vertex_offset, faces, *, scale_xyz=(1., 1., 1.),
                   translate=(0., 0., 0.), center=(0., 0., 0.), body_pose=None):
    source_indices = sorted({int(index) for face in faces for index in face})
    points = {}
    rational_values = []
    for source in source_indices:
        xyz = struct.unpack_from("<3f", mapped, vertex_offset + source * 80)
        if body_pose is not None:
            position, quaternion = body_pose
            local = rotate((-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]),
                           tuple(xyz[k] - position[k] for k in range(3)))
            local = tuple(center[k] + (local[k] - center[k]) * scale_xyz[k] + translate[k]
                          for k in range(3))
            xyz = tuple(position[k] + rotate(quaternion, local)[k] for k in range(3))
        else:
            xyz = tuple(center[k] + (xyz[k] - center[k]) * scale_xyz[k] + translate[k]
                        for k in range(3))
        xyz = tuple(struct.unpack("<f", struct.pack("<f", value))[0] for value in xyz)
        points[source] = xyz
        rational_values.extend(xyz)
    scale = coordinate_lattice_scale(rational_values)
    return source_indices, points, scale


def make_records(surface, scale, faces, predicates):
    source_indices, points, _ = surface
    remap = {source: target for target, source in enumerate(source_indices)}
    vertices = [tuple(lattice_integer(value, scale) for value in points[source])
                for source in source_indices]
    local_faces = [tuple(remap[int(index)] for index in face) for face in faces]
    try:
        return predicates._records(vertices, local_faces)
    except ValueError as error:
        if "degenerate" not in str(error).lower():
            raise
        zero_area = []
        for face_index, face in enumerate(local_faces):
            a, b, c = (vertices[index] for index in face)
            ab = tuple(b[axis] - a[axis] for axis in range(3))
            ac = tuple(c[axis] - a[axis] for axis in range(3))
            cross = (ab[1] * ac[2] - ab[2] * ac[1],
                     ab[2] * ac[0] - ab[0] * ac[2],
                     ab[0] * ac[1] - ab[1] * ac[0])
            if not any(cross):
                zero_area.append(face_index)
        raise ValueError(f"exact predicate rejected zero-area triangle rows {zero_area[:32]}"
                         f" (total {len(zero_area)}); no faces were omitted") from error


def first_intersection(first, second, predicates):
    """Exact fail-fast witness search for bounded affine sensitivity arms."""
    first_lo = tuple(min(row[1][axis] for row in first) for axis in range(3))
    first_hi = tuple(max(row[2][axis] for row in first) for axis in range(3))
    second_lo = tuple(min(row[1][axis] for row in second) for axis in range(3))
    second_hi = tuple(max(row[2][axis] for row in second) for axis in range(3))
    if any(first_hi[axis] < second_lo[axis] or second_hi[axis] < first_lo[axis]
           for axis in range(3)):
        return {"triangle_pairs": [], "count": 0, "aabb_candidate_pairs": 0,
                "audit_complete": True, "count_is_lower_bound": False}
    ordered = sorted(second, key=lambda row: row[1][0])
    first_ordered = sorted(first, key=lambda row: row[1][0])
    candidates = 0
    cursor = 0
    active = {}
    expiry = []
    for row in first_ordered:
        while expiry and expiry[0][0] < row[1][0]:
            _, index = heapq.heappop(expiry)
            active.pop(index, None)
        while cursor < len(ordered) and ordered[cursor][1][0] <= row[2][0]:
            other = ordered[cursor]
            if other[2][0] >= row[1][0]:
                active[other[3]] = other
                heapq.heappush(expiry, (other[2][0], other[3]))
            cursor += 1
        for other in active.values():
            if other[1][0] > row[2][0]:
                continue
            if row[2][0] < other[1][0] or other[2][0] < row[1][0]:
                continue
            if any(row[2][axis] < other[1][axis] or other[2][axis] < row[1][axis]
                   for axis in (1, 2)):
                continue
            candidates += 1
            points = predicates.triangle_intersection_points(row[0], other[0])
            if points:
                return {"triangle_pairs": [(row[3], other[3])], "count": 1,
                        "aabb_candidate_pairs": candidates, "audit_complete": False,
                        "count_is_lower_bound": True}
    return {"triangle_pairs": [], "count": 0, "aabb_candidate_pairs": candidates,
            "audit_complete": True, "count_is_lower_bound": False}


def shared_source_boundaries(mapped, vertex_offset, surfaces, source_meshes, pairs):
    """Compare copies of exact source points after native motion; never use a tolerance.

    Face order is the existing NHANAT-to-viewer contract. Verify a bijection of
    source and captured vertex indices before using that correspondence.
    """
    import math
    by_source = {}
    for key in {item for pair in pairs for item in pair}:
        if key not in surfaces or key not in source_meshes:
            raise ValueError(f"source/captured surface identity absent: {key}")
        points, source_faces = source_meshes[key]
        captured_faces = surfaces[key]["faces"]
        if len(source_faces) != len(captured_faces):
            raise ValueError(f"source/captured triangle counts differ: {key}")
        forward, reverse = {}, {}
        for source_face, captured_face in zip(source_faces, captured_faces):
            if len(source_face) != 3 or len(captured_face) != 3:
                raise ValueError(f"non-triangular shared-boundary source: {key}")
            for local, captured in zip(source_face, captured_face):
                local, captured = int(local), int(captured)
                if not 0 <= local < len(points):
                    raise ValueError(f"source vertex index out of bounds: {key}")
                if forward.setdefault(local, captured) != captured or reverse.setdefault(captured, local) != local:
                    raise ValueError(f"source/captured vertex correspondence is not bijective: {key}")
        groups = {}
        for local, captured in forward.items():
            point = tuple(float(value) for value in points[local])
            world = struct.unpack_from("<3f", mapped, vertex_offset + captured * 80)
            if len(point) != 3 or not all(math.isfinite(x) for x in (*point, *world)):
                raise ValueError(f"nonfinite shared-boundary source/capture: {key}")
            groups.setdefault(point, []).append((local, world))
        by_source[key] = groups
    rows = []
    for left, right in pairs:
        if left == right:
            raise ValueError("shared-boundary comparison requires distinct surfaces")
        first, second = by_source[left], by_source[right]
        common = sorted(first.keys() & second.keys())
        different, comparisons, maximum, squared = 0, 0, 0.0, 0.0
        witnesses = []
        for point in common:
            for local_a, world_a in first[point]:
                for local_b, world_b in second[point]:
                    comparisons += 1
                    gap = math.dist(world_a, world_b)
                    squared += gap * gap
                    maximum = max(maximum, gap)
                    if world_a != world_b:
                        different += 1
                        if len(witnesses) < 16:
                            witnesses.append({"source_point_m": point,
                                              "first_local_vertex": local_a,
                                              "second_local_vertex": local_b,
                                              "first_world_m": world_a, "second_world_m": world_b,
                                              "gap_m": gap})
        rows.append({"first": list(left), "second": list(right),
                     "source_shared_coordinate_count": len(common),
                     "native_vertex_copy_comparisons": comparisons,
                     "native_different_vertex_copies": different,
                     "max_world_gap_m": maximum,
                     "rms_world_gap_m": math.sqrt(squared / comparisons) if comparisons else None,
                     "status": ("no_shared_source_boundary" if not common else
                                "different_native_boundary_copies" if different else "pass"),
                     "witnesses": witnesses})
    return rows


def load_source_boundary_meshes(path, expected_sha256, surfaces, keys):
    """Read the existing NHANAT5 owner format, retaining its explicit identity."""
    if len(expected_sha256) != 64 or sha256_file(path) != expected_sha256:
        raise ValueError("source anatomy SHA-256 differs from the supplied identity")
    owner_path = Path(__file__).with_name("cardiac_geometry_binding.py")
    spec = importlib.util.spec_from_file_location("_accepted_boundary_nhanat_owner", owner_path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    raw, _, records, vertices, indices = owner.read_payload(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("source anatomy changed while loading")
    meshes = {}
    for key in keys:
        if key[0] not in (51010, 51023, 51024, 51025):
            raise ValueError(f"shared-boundary mode supports NHANAT anatomy surfaces only: {key}")
        found = [row for row in records if row[5] == key[1]]
        if len(found) != 1:
            raise ValueError(f"source anatomy stable ID is absent or ambiguous: {key}")
        record = found[0]
        body, points, faces = owner.surface_arrays(records, vertices, indices,
                                                   key[1], expected_layer=record[6])
        if surfaces[key]["link"] != body:
            raise ValueError(f"source/captured body identity differs: {key}")
        meshes[key] = (points, faces)
    return meshes, {"path": str(path.resolve()), "sha256": expected_sha256,
                    "reader": str(owner_path.resolve()), "reader_sha256": sha256_file(owner_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pack", type=Path)
    parser.add_argument("predicate_source", type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--pairs",
                        help="comma-separated semantic:id,semantic:id pairs")
    parser.add_argument("--pair-set", choices=("respiratory-versus-thorax-bones",
                                                "respiratory-versus-thorax-clearance"),
                        help="predefined exact pair inventory for the resting thorax")
    parser.add_argument("--thorax-audit-json", type=Path,
                        help="derive checked pairs from a retained thorax exact-frame audit")
    parser.add_argument("--audit-filter", choices=("all", "candidate", "hit"), default="all",
                        help="when deriving pairs, optionally restrict to broadphase candidates or retained hits")
    parser.add_argument("--only-surfaces",
                        help="comma-separated source IDs to retain from the derived thorax-audit pairs")
    parser.add_argument("--first-scale", default="1,1,1",
                        help="hypothetical affine scale of each first surface")
    parser.add_argument("--first-translate", default="0,0,0",
                        help="hypothetical translation of each first surface, metres")
    parser.add_argument("--center", default="0,0,0",
                        help="center for first-surface affine, metres")
    parser.add_argument("--frame-local-transform", action="store_true",
                        help="apply affine in torso body-20 coordinates using the accepted receipt pose")
    parser.add_argument("--receipt", type=Path, required=True,
                        help="receipt for this exact accepted MRVPACK and vertex buffer")
    parser.add_argument("--stop-after-first", action="store_true")
    parser.add_argument("--shared-boundaries-only", action="store_true",
                        help="compare exact source-shared vertex copies in the accepted native frame")
    parser.add_argument("--source-anatomy", type=Path, help="NHANAT5 source for shared-boundary mode")
    parser.add_argument("--source-anatomy-sha256", help="expected NHANAT5 SHA-256 from the run inputs")
    args = parser.parse_args()

    args.pack = args.pack.resolve(strict=True)
    args.receipt = args.receipt.resolve(strict=True)
    args.predicate_source = args.predicate_source.resolve(strict=True)
    predicate_sha256 = sha256_file(args.predicate_source)
    predicates = predicate_module(args.predicate_source)
    if sha256_file(args.predicate_source) != predicate_sha256:
        raise ValueError("predicate source changed while loading the exact module")
    mapped, stream, vertex_offset, surfaces = read_pack(args.pack)
    receipt_provenance = validate_accepted_receipt(
        args.pack, args.receipt, args.step, mapped, vertex_offset, surfaces)
    pairs = []
    keys = set()
    selectors = sum(bool(value) for value in (args.pairs, args.thorax_audit_json, args.pair_set))
    if selectors != 1:
        parser.error("supply exactly one of --pairs, --pair-set or --thorax-audit-json")
    if args.pairs:
        requested = []
        for item in args.pairs.split(","):
            left_text, right_text = item.split("-")
            requested.append((tuple(map(int, left_text.split(":"))),
                              tuple(map(int, right_text.split(":")))))
    elif args.thorax_audit_json:
        frame_data = json.loads(args.thorax_audit_json.read_text())
        frame = next((row for row in frame_data["frames"] if row["step"] == args.step), None)
        if frame is None:
            raise ValueError(f"step {args.step} is absent from thorax audit")
        def semantic_for_surface(stable_id):
            if 305 <= stable_id <= 309:
                return 51023
            if stable_id == 310:
                return 51024
            if stable_id in (1, 23, 24, 311):
                return 51010
            if 318 <= stable_id <= 321:
                return 51025
            raise ValueError(f"unknown thorax source identity {stable_id}")
        rows = frame["thorax_rib_heart_clearance"]
        if args.audit_filter == "candidate":
            rows = [row for row in rows if row.get("aabb_candidate_triangle_pairs", 0) > 0]
        elif args.audit_filter == "hit":
            rows = [row for row in rows if row.get("one_or_more_intersecting_triangle_pairs", 0) > 0 or
                    row.get("exact_intersecting_triangle_pairs", 0) > 0 or
                    row.get("status") == "exact_surface_intersection_found"]
        if args.only_surfaces:
            allowed_ids = {int(value) for value in args.only_surfaces.split(",")}
            rows = [row for row in rows if row["surface"] in allowed_ids]
        requested = [((semantic_for_surface(row["surface"]), row["surface"]),
                      (51004, row["bone"]))
                     for row in rows]
    else:
        respiratory = [(51023, stable_id) for stable_id in range(305, 310)] + [(51010, 311)]
        if args.pair_set == "respiratory-versus-thorax-clearance":
            respiratory.insert(-1, (51024, 310))
        bones = [(51004, stable_id) for stable_id in range(26, 60)]
        externals = list(bones)
        if args.pair_set == "respiratory-versus-thorax-clearance":
            externals += [(51007, 1), (51010, 1), (51010, 23), (51010, 24)]
            externals += [(51025, stable_id) for stable_id in range(318, 322)]
        requested = [(organ, external) for organ in respiratory for external in externals]
    for left, right in requested:
        if left not in surfaces or right not in surfaces:
            raise ValueError(f"surface identity absent from pack: {left} or {right}")
        pairs.append((left, right))
        keys.update((left, right))

    if args.shared_boundaries_only:
        if not args.source_anatomy or not args.source_anatomy_sha256:
            parser.error("--shared-boundaries-only requires --source-anatomy and --source-anatomy-sha256")
        if (vector_arg(args.first_scale) != (1., 1., 1.) or
                vector_arg(args.first_translate) != (0., 0., 0.) or args.frame_local_transform):
            parser.error("shared-boundary mode audits the native capture without hypothetical transforms")
        meshes, source_identity = load_source_boundary_meshes(
            args.source_anatomy, args.source_anatomy_sha256, surfaces, keys)
        rows = shared_source_boundaries(mapped, vertex_offset, surfaces, meshes, pairs)
        print(json.dumps({"schema": "accepted-mrvpack-pair-audit.v2",
                          "audit_mode": "source_shared_boundary_correspondence",
                          "step": args.step, "pack": str(args.pack),
                          "pack_sha256": sha256_file(args.pack),
                          "accepted_receipt": receipt_provenance,
                          "source_anatomy": source_identity,
                          "driver_sha256": sha256_file(Path(__file__)),
                          "pairs": rows,
                          "status": "pass" if rows and all(row["status"] == "pass" for row in rows) else "rejected",
                          "limitation": "Exact correspondence of declared source-shared coordinates only. The supplied source hash must come from the retained run input identity. This does not certify triangle intersections, solid containment, a full trajectory, or physiology."}, indent=2))
        mapped.close()
        stream.close()
        if not rows or any(row["status"] != "pass" for row in rows):
            raise SystemExit(2)
        return
    if args.source_anatomy or args.source_anatomy_sha256:
        parser.error("--source-anatomy options require --shared-boundaries-only")

    affine_scale = vector_arg(args.first_scale)
    translate = vector_arg(args.first_translate)
    center = vector_arg(args.center)
    body_pose = None
    body_pose_json = None
    if args.frame_local_transform:
        receipt = json.loads(args.receipt.read_text())
        pose = next((row for row in receipt.get("accepted_registered_body_poses", [])
                     if row.get("body_index") == 20), None)
        if pose is None:
            raise ValueError("receipt has no exact accepted body-20 pose")
        body_pose = (tuple(float(v) for v in pose["position_m"]),
                     tuple(float(v) for v in pose["quaternion_xyzw"]))
        body_pose_json = pose
    first_keys = {left for left, _ in pairs}
    exact_surfaces = {key: surface_points(mapped, vertex_offset, surfaces[key]["faces"],
                                          scale_xyz=affine_scale if key in first_keys else (1., 1., 1.),
                                          translate=translate if key in first_keys else (0., 0., 0.),
                                          center=center if key in first_keys else (0., 0., 0.),
                                          body_pose=body_pose if key in first_keys else None)
                      for key in keys}
    scale = max(surface[2] for surface in exact_surfaces.values())
    cache = {key: make_records(exact_surfaces[key], scale, surfaces[key]["faces"], predicates)
             for key in exact_surfaces}
    pair_rows = []
    for left, right in pairs:
        if args.stop_after_first:
            report = first_intersection(cache[left], cache[right], predicates)
        else:
            kwargs = {"same_surface": False}
            if "stop_after_first_intersection" in inspect.signature(predicates._audit_pair).parameters:
                kwargs["stop_after_first_intersection"] = False
            report = predicates._audit_pair(cache[left], cache[right], **kwargs)
        witnesses = []
        for first_triangle, second_triangle in report["triangle_pairs"]:
            first_record = cache[left][first_triangle]
            second_record = cache[right][second_triangle]
            points = predicates.triangle_intersection_points(first_record[0], second_record[0])
            witnesses.append({"first_triangle": first_triangle,
                              "second_triangle": second_triangle,
                              "intersection_points_m": [[float(c) / scale for c in p]
                                                         for p in points]})
        pair_rows.append({"first": list(left), "second": list(right),
                          "surface_triangle_counts": [len(surfaces[left]["faces"]),
                                                      len(surfaces[right]["faces"])],
                          "aabb_candidate_pairs": report["aabb_candidate_pairs"],
                          "intersecting_triangle_pairs": report["count"],
                          "audit_complete": report.get("audit_complete", True),
                          "count_is_lower_bound": report.get("count_is_lower_bound", False),
                          "witnesses": witnesses})
    out = {"schema": "accepted-mrvpack-pair-audit.v2", "step": args.step,
           "driver": str(Path(__file__).resolve()),
           "driver_sha256": sha256_file(Path(__file__).resolve()),
           "pack": str(args.pack), "pack_sha256": sha256_file(args.pack),
           "accepted_receipt": receipt_provenance,
           "predicate_source": str(args.predicate_source),
           "predicate_sha256": predicate_sha256,
           "coordinate_scale_per_m": scale,
           "first_surface_affine": {"scale": affine_scale, "translation_m": translate,
                                     "center_m": center,
                                     "coordinate_frame": "body20_torso_local" if body_pose else "world",
                                     "body20_pose": body_pose_json,
                                     "volume_ratio_for_closed_first_surface": affine_scale[0] * affine_scale[1] * affine_scale[2]},
           "pairs": pair_rows,
           "limitation": "This exact triangle-pair audit uses an accepted post-motion MRVPACK and its matching receipt. Any first-surface affine is applied to that already-deformed captured geometry, so it is a post-motion sensitivity diagnostic, not a static source-registration candidate, a native simulation result, penetration depth, or biological severity."}
    print(json.dumps(out, indent=2))
    mapped.close()
    stream.close()


if __name__ == "__main__":
    main()
