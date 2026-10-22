#!/usr/bin/env python3
"""Rebuild only the left APL origin envelope with the pinned owner algorithm.

All source route/site coordinates and endpoint identities remain unchanged.
The script appends one force-map envelope to a new NHTENDON3 payload and
updates only that endpoint's envelope index and compiler metrics. The source
payload and all original evidence are read-only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import sys

REPO = Path("/Users/n/numi-human-aortic-contact-opt-20261019")
EVIDENCE = REPO / "matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/hand-actuation-qualification-001"
SOURCE_BASE = Path("/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004")
MUSCLE_PATH = SOURCE_BASE / "myosim-fullbody-muscle-reference.nhmyo"
TENDON_PATH = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/compiled-tendon/numi-human-tendon-attachments.nhtendon")
BONE_DIR = Path("/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/full-anatomy-delivery-audit-1378/rope-swing-eight-views-20261018-001/paired-bone49-precision/bone-step-4")
MODEL_PY = Path("/Users/n/numi-human-anatomy-finish-1378/src/numilab_human/model.py")
SOURCE_MANIFEST = Path("/Users/n/numi-human-separation-20261018/C/publication-source-001/docs/media/numi-human-passive-fit-v4/source-manifest.json")
SOURCE_SHA = "280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975"
EXPECTED = {
    str(MUSCLE_PATH): "e5bb8a8168706bb3b23cf42b1d849659569ba2e6027bcc85729c3110fdbb8c4b",
    str(TENDON_PATH): "d0f5da8c63d5e3335eef72e7e79e8e5780ae43f0cf254e840dd7ad604b72da48",
    str(BONE_DIR / "bodyparts3d-myosim-major-bones.nhbones"): "43a31ef4771e2a383ae2c7caab1b994f0c28076c2a43501860cd1ba3fe7708c0",
    str(SOURCE_MANIFEST): "c794687f4b60dab86b510658365fba74b29bcf749b1665f0ec232d4b6e451054",
}
TARGET_INDEX = 322
TARGET_ENDPOINT = 0
TARGET_BINDING = TARGET_INDEX * 2 + TARGET_ENDPOINT
BODY_INDEX = 93
BONE_STABLE_ID = 17
BONE_MEMBER_ID = "FJ3277"
AMPLIFICATION_CAP = 2.2
SURFACE_DISTANCE_CAP_M = 0.004
PATCH_RADIUS_CAP_M = 0.015
MOMENT_RESIDUAL_CAP_M = 1.0e-9
OUTPUT_DIR = EVIDENCE / "results" / "left-apl-envelope-under-2p2"
AUDIT_PATH = EVIDENCE / "thumb-endpoint-bound-audit.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    for filename, expected in EXPECTED.items():
        actual = sha256(Path(filename))
        if actual != expected:
            raise RuntimeError(f"pinned source drift: {filename}: {actual} != {expected}")

    import numilab_human.model as owner_model

    original = TENDON_PATH.read_bytes()
    muscle = MUSCLE_PATH.read_bytes()
    header = struct.unpack_from("<8s10I32s32s32s", original)
    magic, abi, body_count, muscle_count, site_count, endpoint_count, envelope_count, bone_count, fingerprint, reserved0, reserved1, source_sha, muscle_sha, bone_sha = header
    if (
        magic != b"NHTEND3\0" or abi != 3 or body_count != 157 or
        muscle_count != 416 or endpoint_count != 832 or envelope_count != 366 or
        source_sha.hex() != SOURCE_SHA or muscle_sha.hex() != EXPECTED[str(MUSCLE_PATH)] or
        bone_sha.hex() != EXPECTED[str(BONE_DIR / "bodyparts3d-myosim-major-bones.nhbones")]
    ):
        raise RuntimeError("pinned NHTENDON3 source identities or dimensions differ")

    manifest = json.loads(SOURCE_MANIFEST.read_text())
    if manifest.get("source", {}).get("archive_sha256") != SOURCE_SHA:
        raise RuntimeError("source manifest archive identity differs")
    expected_endpoints = {
        256: ([42, 54], [14, 94]), 257: ([43, 53], [16, 93]),
        258: ([43, 54], [16, 94]), 259: ([43, 52], [16, 88]),
        319: ([92, 104], [15, 113]), 320: ([93, 103], [17, 112]),
        321: ([93, 104], [17, 113]), 322: ([93, 102], [17, 107]),
    }
    table = []
    for muscle_index, (bodies, bone_ids) in expected_endpoints.items():
        muscle_name = manifest["muscles"][muscle_index]["name"]
        for endpoint in (0, 1):
            record_index = muscle_index * 2 + endpoint
            binding = struct.unpack_from("<8I8f", original, 144 + 64 * record_index)
            ints, vals = binding[:8], binding[8:]
            row = {
                "record": record_index, "source_actuator_index": muscle_index,
                "muscle_name": muscle_name, "endpoint_ordinal": endpoint,
                "route_node_index": ints[2], "source_site_index": ints[3],
                "body_index": ints[4], "mode": ints[5], "envelope_index": ints[6],
                "bone_stable_id": ints[7], "source_local_point_m": list(vals[:3]),
                "surface_distance_m": vals[3], "force_amplification": vals[4],
                "patch_radius_m": vals[5], "compiled_moment_residual_m": vals[6],
                "endpoint_migration_m": vals[7], "expected_body_index": bodies[endpoint],
                "expected_bone_stable_id": bone_ids[endpoint],
            }
            row["failed_checks"] = []
            if ints[4] != bodies[endpoint]: row["failed_checks"].append("body_index")
            if ints[7] != bone_ids[endpoint]: row["failed_checks"].append("bone_stable_id")
            if ints[5] != 2: row["failed_checks"].append("distributed_envelope_mode")
            if vals[7] > 1.0e-12: row["failed_checks"].append("endpoint_migration")
            if vals[3] > SURFACE_DISTANCE_CAP_M: row["failed_checks"].append("surface_distance")
            if not 0.004 <= vals[5] <= PATCH_RADIUS_CAP_M: row["failed_checks"].append("patch_radius")
            if vals[4] > AMPLIFICATION_CAP: row["failed_checks"].append("force_amplification")
            if vals[6] > MOMENT_RESIDUAL_CAP_M: row["failed_checks"].append("compiled_moment_residual")
            table.append(row)

    target_row = next(row for row in table if row["record"] == TARGET_BINDING)
    if (
        target_row["muscle_name"] != "APL_l" or target_row["body_index"] != BODY_INDEX or
        target_row["bone_stable_id"] != BONE_STABLE_ID or
        target_row["mode"] != 2 or target_row["endpoint_migration_m"] != 0.0
    ):
        raise RuntimeError("left APL source endpoint identity is not the pinned expected owner")
    if target_row["failed_checks"] != ["force_amplification"]:
        raise RuntimeError(f"left APL endpoint failure set changed: {target_row['failed_checks']}")

    muscle_header = struct.unpack_from("<8s9I32s", muscle)
    muscle_magic, muscle_abi, myo_body_count, myo_muscle_count, myo_site_count, wrap_count, route_count, tendon_count, arch0, arch1, myo_source_sha = muscle_header
    if myo_source_sha.hex() != SOURCE_SHA or myo_muscle_count != 416:
        raise RuntimeError("pinned NHMYO source identity differs")
    site_offset = struct.calcsize("<8s9I32s")
    sites = [
        struct.unpack_from("<I3f", muscle, site_offset + 16 * i)
        for i in range(myo_site_count)
    ]
    route_offset = site_offset + 16 * myo_site_count + 64 * wrap_count
    routes = [
        struct.unpack_from("<4I", muscle, route_offset + 16 * i)
        for i in range(route_count)
    ]
    muscle_offset = route_offset + 16 * route_count
    muscle_record = struct.unpack_from(
        "<4I37f", muscle, muscle_offset + 164 * TARGET_INDEX
    )
    route_index, route_nodes = muscle_record[1], muscle_record[2]
    route = routes[route_index]
    if route[0] != owner_model._MYOSIM_ROUTE_SITE:
        raise RuntimeError("left APL origin is not the expected source-site route node")
    source_site_index = route[1]
    source_site = sites[source_site_index]
    source_body = int(source_site[0])
    source_point = [float(v) for v in source_site[1:]]
    if source_body != BODY_INDEX or route_nodes != 8:
        raise RuntimeError("left APL source-site/body mapping differs from pinned route")
    if any(abs(a - b) > 1.0e-7 for a, b in zip(source_point, target_row["source_local_point_m"])):
        raise RuntimeError("NHTENDON3 row does not preserve the exact NHMYO source site")

    surfaces_by_body, bone_descriptor, bone_payload_path = (
        owner_model._numi_human_bone_envelope_surfaces(BONE_DIR, SOURCE_SHA)
    )
    surfaces = [
        surface for surface in surfaces_by_body.get(BODY_INDEX, [])
        if surface["stable_id"] == BONE_STABLE_ID and
        surface["member_id"] == BONE_MEMBER_ID
    ]
    if len(surfaces) != 1:
        raise RuntimeError("expected single exact FJ3277 surface is missing or ambiguous")
    surface = surfaces[0]
    surface["body_index"] = BODY_INDEX
    envelope, reason = owner_model._numi_human_tendon_surface_envelope(
        source_point, surface,
        SURFACE_DISTANCE_CAP_M, PATCH_RADIUS_CAP_M, AMPLIFICATION_CAP,
        False,
    )
    if envelope is None:
        raise RuntimeError(f"owner compiler did not admit the endpoint: {reason}")
    metrics = {
        "surface_distance_m": float(envelope["surface_distance_m"]),
        "force_amplification": float(envelope["sampled_total_force_amplification"]),
        "patch_radius_m": float(envelope["patch_radius_m"]),
        "compiled_moment_residual_m": float(envelope["moment_residual_m"]),
        "force_residual": float(envelope["force_residual"]),
        "l2_force_amplification": float(envelope["l2_force_amplification"]),
        "source_triangle_index": int(envelope["source_triangle_index"]),
        "surface_patch_method": envelope["surface_patch_method"],
    }
    if (
        metrics["surface_distance_m"] > SURFACE_DISTANCE_CAP_M or
        not 0.004 <= metrics["patch_radius_m"] <= PATCH_RADIUS_CAP_M or
        metrics["force_amplification"] > AMPLIFICATION_CAP or
        metrics["compiled_moment_residual_m"] > MOMENT_RESIDUAL_CAP_M or
        metrics["force_residual"] > 2.0e-6
    ):
        raise RuntimeError(f"candidate envelope failed unchanged mechanical bounds: {metrics}")

    endpoint_count_actual = endpoint_count
    target_offset = 144 + 64 * TARGET_BINDING
    target_ints = list(struct.unpack_from("<8I", original, target_offset))
    target_floats = list(struct.unpack_from("<8f", original, target_offset + 32))
    if target_ints[6] >= envelope_count:
        raise RuntimeError("target binding references an invalid original envelope")
    new_index = envelope_count
    target_ints[6] = new_index
    target_floats[3:8] = [
        metrics["surface_distance_m"], metrics["force_amplification"],
        metrics["patch_radius_m"], metrics["compiled_moment_residual_m"], 0.0,
    ]

    node_values = [
        component for node in envelope["node_local_points_m"]
        for component in (*node, 0.0)
    ]
    map_values = [
        component for matrix in envelope["force_maps"]
        for row in matrix for component in (*row, 0.0)
    ]
    if len(node_values) != 16 or len(map_values) != 48:
        raise RuntimeError("owner force-map envelope has unexpected dimensions")
    encoded_envelope = struct.pack(
        "<4I68f", BODY_INDEX, BONE_STABLE_ID,
        metrics["source_triangle_index"], 4,
        *node_values, *map_values,
        metrics["surface_distance_m"], metrics["patch_radius_m"],
        metrics["force_amplification"], metrics["l2_force_amplification"],
    )
    output_bytes = bytearray(original)
    struct.pack_into("<I", output_bytes, 8 + 5 * 4, envelope_count + 1)
    struct.pack_into("<8I8f", output_bytes, target_offset, *target_ints, *target_floats)
    output_bytes.extend(encoded_envelope)

    allowed = set(range(8 + 5 * 4, 8 + 6 * 4))
    allowed.update(range(target_offset + 24, target_offset + 28))
    allowed.update(range(target_offset + 44, target_offset + 64))
    changed = {
        i for i, (old, new) in enumerate(zip(original, output_bytes))
        if old != new
    }
    if not changed.issubset(allowed):
        raise RuntimeError(f"unexpected edits outside target envelope record: {sorted(changed - allowed)[:20]}")

    if OUTPUT_DIR.exists():
        raise RuntimeError(f"refusing to overwrite prior staged payload: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)
    output_payload = OUTPUT_DIR / TENDON_PATH.name
    output_payload.write_bytes(output_bytes)
    output_sha = sha256(output_payload)
    audit = {
        "schema": "numi.human.left-apl-single-envelope-regeneration.v1",
        "status": "STAGED_CPU_OWNER_COMPILER",
        "scope": "only left APL origin binding and one appended distributed surface envelope; no source route, source point, muscle, skeleton, accepted payload, or certificate bound changed",
        "inputs": {
            str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (MUSCLE_PATH, TENDON_PATH, bone_payload_path, SOURCE_MANIFEST, MODEL_PY)
        },
        "source_archive_sha256": SOURCE_SHA,
        "owner_algorithm": "numilab_human.model._numi_human_tendon_surface_envelope (connected topology-aware exact surface patch)",
        "owner_algorithm_file_sha256": sha256(MODEL_PY),
        "owner_compiler_output": "single endpoint envelope built with source point fixed and maximum_force_amplification=2.2; no full-body recompilation",
        "bounds_unchanged": {
            "surface_distance_max_m": SURFACE_DISTANCE_CAP_M,
            "patch_radius_min_m": 0.004,
            "patch_radius_max_m": PATCH_RADIUS_CAP_M,
            "force_amplification_max": AMPLIFICATION_CAP,
            "compiled_moment_residual_max_m": MOMENT_RESIDUAL_CAP_M,
            "force_residual_max": 2.0e-6,
        },
        "decoded_thumb_endpoint_table": table,
        "offending_original_endpoint": target_row,
        "source_owner": {
            "muscle_index": TARGET_INDEX, "muscle_name": "APL_l",
            "endpoint_ordinal": TARGET_ENDPOINT, "route_node_index": route_index,
            "source_site_index": source_site_index, "body_index": source_body,
            "source_local_point_m": source_point,
            "bone_stable_id": BONE_STABLE_ID, "bone_member_id": BONE_MEMBER_ID,
        },
        "replacement_endpoint": {
            **metrics, "envelope_index": new_index,
            "body_index": BODY_INDEX, "bone_stable_id": BONE_STABLE_ID,
            "bone_member_id": BONE_MEMBER_ID,
            "endpoint_migration_m": 0.0,
            "source_local_point_m": source_point,
            "source_point_unchanged": True,
            "semantic_identity_unchanged": True,
        },
        "payload_change_audit": {
            "original_envelope_count": envelope_count,
            "new_envelope_count": envelope_count + 1,
            "endpoint_record_index": TARGET_BINDING,
            "changed_existing_byte_ranges_allowed": [
                [8 + 5 * 4, 8 + 6 * 4],
                [target_offset + 24, target_offset + 28],
                [target_offset + 44, target_offset + 64],
            ],
            "unexpected_existing_payload_byte_changes": 0,
            "appended_bytes": len(encoded_envelope),
            "output_sha256": output_sha,
            "output_bytes": len(output_bytes),
        },
        "output_payload": str(output_payload),
        "gpu_or_native_execution": False,
    }
    (OUTPUT_DIR / "qualification.json").write_text(json.dumps(audit, indent=2) + "\n")
    AUDIT_PATH.write_text(json.dumps({
        "schema": "numi.human.thumb-endpoint-bound-audit.v1",
        "source_tendon_sha256": sha256(TENDON_PATH),
        "source_muscle_sha256": sha256(MUSCLE_PATH),
        "bone_payload_sha256": sha256(bone_payload_path),
        "expected_endpoint_table": table,
        "failed_check_rows": [row for row in table if row["failed_checks"]],
        "candidate_regeneration_receipt": str(OUTPUT_DIR / "qualification.json"),
    }, indent=2) + "\n")
    print(json.dumps({
        "status": audit["status"],
        "original_left_apl": target_row,
        "replacement": audit["replacement_endpoint"],
        "output_payload": str(output_payload),
        "output_sha256": output_sha,
        "audit": str(AUDIT_PATH),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, "/Users/n/numi-human-anatomy-finish-1378/src")
    raise SystemExit(main())
