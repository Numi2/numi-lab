#!/usr/bin/env python3
"""Rebase the selected passive-organ NHANAT1 surfaces to a common torso frame.

This authoring step preserves the existing NHANAT1 ABI, topology, source
measurements, and neutral world pose. It changes only the body owner and local
position/normal bytes of the declared passive-viscera surfaces, then records
the inferred torso-to-pelvis presentation convention in the existing receipt.
It does not add tissue mechanics, forces, mass, or physiology.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import sys

import numpy as np


HEADER = struct.Struct("<8s5I32s")
RECORD = struct.Struct("<8I")
VERTEX = struct.Struct("<6f")
SELECTED_IDS = tuple(range(2, 6)) + tuple(range(13, 23)) + tuple(range(398, 464))
PELVIC_SOURCE_IDS = (459, 462, 463)
UPPER_ANCHOR_IDS = (2, 3, 4, 5, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 461)
PELVIC_ANCHOR_IDS = (462, 463)
CARDIAC_IDS = (1, 23, 24, 318, 319, 320, 321)
MODEL = "common_respiratory_field_with_torso_pelvis_blend_v1"


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError("passive viscera geometry rebase: " + message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def normalized_quaternion(value) -> np.ndarray:
    q = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    require(q.shape == (4,) and math.isfinite(norm) and abs(norm - 1.0) < 1e-5,
            "initial body quaternion is malformed or not unit length")
    return q / norm


def rotate(q: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate row-vector arrays using normalized xyzw quaternions."""
    axis = q[:3]
    values = np.asarray(vectors, dtype=np.float64)
    return values + 2.0 * np.cross(axis, np.cross(axis, values) + q[3] * values)


def surface_digest(blob: bytes | bytearray, records_by_id: dict[int, tuple[int, ...]], ids) -> str:
    h = hashlib.sha256()
    for stable_id in ids:
        body, first_vertex, vertex_count, first_index, index_count, sid, layer, reserved = records_by_id[stable_id]
        h.update(struct.pack("<I", sid))
        h.update(blob[record_offsets[sid]:record_offsets[sid] + RECORD.size])
        h.update(blob[vertex_base + first_vertex * VERTEX.size:
                      vertex_base + (first_vertex + vertex_count) * VERTEX.size])
        h.update(blob[index_base + first_index * 4:index_base + (first_index + index_count) * 4])
    return h.hexdigest()


def main() -> int:
    global record_offsets, vertex_base, index_base
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-payload", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--initial-registration", type=Path, required=True)
    parser.add_argument("--output-payload", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()

    require(not args.output_payload.exists() and not args.output_receipt.exists() and not args.evidence.exists(),
            "refusing to overwrite an existing candidate or evidence directory")
    source_bytes = args.input_payload.read_bytes()
    receipt = json.loads(args.input_receipt.read_text())
    registration_bytes = args.initial_registration.read_bytes()
    registration = json.loads(registration_bytes)
    require(len(source_bytes) >= HEADER.size, "truncated NHANAT1 payload")
    magic, abi, surface_count, vertex_count, index_count, registration_fingerprint, source_sha = HEADER.unpack_from(source_bytes)
    require(magic == b"NHANAT1\0" and abi == 5, "expected existing NHANAT1 ABI 5")
    expected_bytes = HEADER.size + surface_count * RECORD.size + vertex_count * VERTEX.size + index_count * 4
    require(len(source_bytes) == expected_bytes and index_count % 3 == 0, "NHANAT1 byte counts disagree")
    source_hash = sha256(source_bytes)
    require(receipt.get("payload", {}).get("sha256") == source_hash and
            receipt.get("functional_bindings", {}).get("anatomy_payload_sha256") == source_hash,
            "input receipt does not bind source payload")
    cardiac = receipt.get("provenance", {}).get("cardiac_geometry_binding")
    require(isinstance(cardiac, dict) and cardiac.get("output_anatomy_payload_sha256") == source_hash,
            "input cardiac binding does not bind the source payload")

    record_base = HEADER.size
    vertex_base = record_base + surface_count * RECORD.size
    index_base = vertex_base + vertex_count * VERTEX.size
    records = [RECORD.unpack_from(source_bytes, record_base + i * RECORD.size)
               for i in range(surface_count)]
    records_by_id = {row[5]: row for row in records}
    record_offsets = {row[5]: record_base + i * RECORD.size for i, row in enumerate(records)}
    require(len(records_by_id) == len(records), "stable surface IDs are not unique")
    require(len(SELECTED_IDS) == 80 and all(sid in records_by_id for sid in SELECTED_IDS),
            "selected passive surface set is incomplete")
    require(all(records_by_id[sid][0] == (128 if sid in PELVIC_SOURCE_IDS else 7)
                for sid in SELECTED_IDS), "selected surface source owners differ from retained provenance")
    source_id_map = receipt.get("provenance", {}).get("source_id_map", {})
    require(all(str(sid) in source_id_map and source_id_map[str(sid)].get("body_index") == records_by_id[sid][0]
                for sid in SELECTED_IDS),
            "surface source-owner map disagrees with NHANAT1")

    poses = {}
    for row in registration.get("body_poses", []):
        body = int(row["body_index"])
        poses[body] = (np.asarray(row["position_m"], dtype=np.float64),
                       normalized_quaternion(row["quaternion_xyzw"]))
    require(all(body in poses for body in (7, 20, 128)), "native initial body registration omits a source owner")
    require(registration.get("torso_body_index") == 20, "registration does not declare torso body 20")
    position20, quaternion20 = poses[20]
    inverse20 = quaternion20.copy()
    inverse20[:3] *= -1.0

    source_vertices = np.frombuffer(source_bytes, dtype="<f4", count=vertex_count * 6,
                                    offset=vertex_base).reshape(vertex_count, 6).copy()
    output_bytes = bytearray(source_bytes)
    source_owners = {}
    point_errors = []
    normal_errors = []
    rebased_bounds = {}
    modifications = []
    for stable_id in SELECTED_IDS:
        body, first_vertex, count, first_index, index_count_for_surface, sid, layer, reserved = records_by_id[stable_id]
        source_owners[str(stable_id)] = body
        old_position, old_quaternion = poses[body]
        source_rows = source_vertices[first_vertex:first_vertex + count].astype(np.float64)
        old_local = source_rows[:, :3]
        old_normals = source_rows[:, 3:]
        world_positions = old_position + rotate(old_quaternion, old_local)
        torso_local = rotate(inverse20, world_positions - position20)
        world_normals = rotate(old_quaternion, old_normals)
        torso_normals = rotate(inverse20, world_normals)
        normal_lengths = np.linalg.norm(torso_normals, axis=1)
        require(np.isfinite(torso_local).all() and np.isfinite(torso_normals).all() and
                float(normal_lengths.min()) > 1e-8,
                f"stable ID {stable_id} contains invalid source geometry")
        torso_rows = np.concatenate((torso_local, torso_normals), axis=1).astype("<f4")
        restored = position20 + rotate(quaternion20, torso_rows[:, :3].astype(np.float64))
        point_errors.extend(np.linalg.norm(restored - world_positions, axis=1).tolist())
        restored_normals = rotate(quaternion20, torso_rows[:, 3:].astype(np.float64))
        normalized_expected = world_normals / np.linalg.norm(world_normals, axis=1)[:, None]
        normalized_restored = restored_normals / np.linalg.norm(restored_normals, axis=1)[:, None]
        normal_errors.extend(np.linalg.norm(normalized_restored - normalized_expected, axis=1).tolist())

        record_offset = record_offsets[stable_id]
        struct.pack_into("<I", output_bytes, record_offset, 20)
        modifications.append((record_offset, record_offset + 4))
        for local_index, vertex_row in enumerate(torso_rows):
            absolute_index = first_vertex + local_index
            byte_offset = vertex_base + absolute_index * VERTEX.size
            VERTEX.pack_into(output_bytes, byte_offset, *map(float, vertex_row))
        modifications.append((vertex_base + first_vertex * VERTEX.size,
                              vertex_base + (first_vertex + count) * VERTEX.size))
        rebased_bounds[str(stable_id)] = {
            "minimum_m": torso_rows[:, :3].min(axis=0).astype(float).tolist(),
            "maximum_m": torso_rows[:, :3].max(axis=0).astype(float).tolist(),
            "source_body_index": body,
            "target_body_index": 20,
            "vertex_count": count,
        }

    # Prove the registration dump describes the same source points and pose.
    registration_bounds = {int(row["stable_id"]): row for row in registration.get("surfaces", [])}
    bound_errors = []
    for stable_id in UPPER_ANCHOR_IDS + PELVIC_ANCHOR_IDS:
        source_row = registration_bounds.get(stable_id)
        require(source_row is not None, f"registration dump has no selected anchor {stable_id}")
        ours = rebased_bounds[str(stable_id)]
        for key, expected_key in (("minimum_m", "torso_local_min_m"), ("maximum_m", "torso_local_max_m")):
            bound_errors.extend(np.abs(np.asarray(ours[key]) - np.asarray(source_row[expected_key])).tolist())
    max_bound_error = max(bound_errors)
    require(max_bound_error <= 1e-6, f"native registration bounds differ by {max_bound_error:.9g} m")

    # All bytes outside the declared owner/vertex ranges must remain identical.
    modifications.sort()
    merged = []
    for begin, end in modifications:
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    cursor = 0
    for begin, end in merged:
        require(source_bytes[cursor:begin] == output_bytes[cursor:begin],
                "candidate changed bytes outside selected passive records/vertices")
        cursor = end
    require(source_bytes[cursor:] == output_bytes[cursor:],
            "candidate changed bytes after selected passive records/vertices")
    require(source_bytes[:HEADER.size] == output_bytes[:HEADER.size], "NHANAT1 header changed")
    require(source_bytes[index_base:] == output_bytes[index_base:], "NHANAT1 source topology/index bytes changed")
    out_records = [RECORD.unpack_from(output_bytes, record_base + i * RECORD.size)
                   for i in range(surface_count)]
    out_by_id = {row[5]: row for row in out_records}
    for stable_id in SELECTED_IDS:
        require(out_by_id[stable_id][0] == 20, f"selected stable ID {stable_id} was not rebound to torso 20")
    for stable_id in records_by_id.keys() - set(SELECTED_IDS):
        require(records_by_id[stable_id] == out_by_id[stable_id],
                f"nonselected stable ID {stable_id} record changed")

    source_cardiac_digest = surface_digest(source_bytes, records_by_id, CARDIAC_IDS)
    candidate_cardiac_digest = surface_digest(output_bytes, out_by_id, CARDIAC_IDS)
    require(source_cardiac_digest == candidate_cardiac_digest,
            "cardiac source subset changed during passive-geometry rebase")
    output_hash = sha256(output_bytes)
    output_receipt = json.loads(json.dumps(receipt))
    bindings = output_receipt["functional_bindings"]
    bindings["anatomy_payload_sha256"] = output_hash
    upper_minimum_y = min(rebased_bounds[str(sid)]["minimum_m"][1] for sid in UPPER_ANCHOR_IDS)
    pelvic_maximum_y = max(rebased_bounds[str(sid)]["maximum_m"][1] for sid in PELVIC_ANCHOR_IDS)
    require(pelvic_maximum_y < upper_minimum_y, "passive reference transition endpoints are inverted")
    bindings["passive_viscera_geometry_binding"] = {
        "motion_model": MODEL,
        "parameter_status": "inferred_reference_attachment_not_measured_subject_motion",
        "functional_role": "passive_geometry_no_independent_forces_mass_or_physiology",
        "stable_ids": list(SELECTED_IDS),
        "torso_body_index": 20,
        "pelvic_body_index": 128,
        "transition_superior_coordinates_m": [pelvic_maximum_y, upper_minimum_y],
        "source_body_index_by_stable_id": source_owners,
        "upper_anchor_stable_ids": list(UPPER_ANCHOR_IDS),
        "pelvic_anchor_stable_ids": list(PELVIC_ANCHOR_IDS),
    }
    payload_row = output_receipt["payload"]
    payload_row["path"] = str(args.output_payload)
    payload_row["sha256"] = output_hash
    payload_row["input_payload_sha256"] = source_hash

    cardiac_out = output_receipt["provenance"]["cardiac_geometry_binding"]
    old_cardiac_output = cardiac_out["output_anatomy_payload_sha256"]
    require(old_cardiac_output == source_hash, "cardiac proof source payload changed before composition")
    cardiac_binding_hash_before = canonical_hash(cardiac_out)
    cardiac_out["output_anatomy_payload_sha256"] = output_hash
    cardiac_binding_for_hash = json.loads(json.dumps(cardiac_out))
    cardiac_binding_for_hash["output_anatomy_payload_sha256"] = old_cardiac_output
    require(canonical_hash(cardiac_binding_for_hash) == cardiac_binding_hash_before,
            "cardiac binding proof changed beyond its composed-payload identity pointer")
    rebase = {
        "schema": "numi.human.passive-viscera-geometry-rebase.v1",
        "method": "preserve_neutral_world_pose_rebase_selected_surface_frames_to_torso20",
        "source_payload_path": str(args.input_payload),
        "source_payload_sha256": source_hash,
        "output_payload_path": str(args.output_payload),
        "output_payload_sha256": output_hash,
        "source_payload_header_source_sha256": source_sha.hex(),
        "native_initial_registration_path": str(args.initial_registration),
        "native_initial_registration_sha256": sha256(registration_bytes),
        "source_body_index_by_stable_id": source_owners,
        "target_torso_body_index": 20,
        "lower_pelvic_body_index": 128,
        "selected_stable_ids": list(SELECTED_IDS),
        "neutral_pose_max_world_position_error_m": float(max(point_errors)),
        "neutral_pose_rms_world_position_error_m": float(np.sqrt(np.mean(np.square(point_errors)))),
        "neutral_pose_max_world_normal_direction_error": float(max(normal_errors)),
        "native_registration_anchor_aabb_max_error_m": float(max_bound_error),
        "transition_endpoints_m": [pelvic_maximum_y, upper_minimum_y],
        "transition_lower_anchor_stable_id": max(PELVIC_ANCHOR_IDS,
            key=lambda sid: rebased_bounds[str(sid)]["maximum_m"][1]),
        "transition_upper_anchor_stable_id": min(UPPER_ANCHOR_IDS,
            key=lambda sid: rebased_bounds[str(sid)]["minimum_m"][1]),
        "cardiac_prior_output_anatomy_payload_sha256": old_cardiac_output,
        "cardiac_binding_proof_sha256_before": cardiac_binding_hash_before,
        "cardiac_binding_proof_sha256_after_replacing_output_identity_only": canonical_hash(cardiac_binding_for_hash),
        "cardiac_surface_subset_sha256_before": source_cardiac_digest,
        "cardiac_surface_subset_sha256_after": candidate_cardiac_digest,
        "unselected_records_byte_identical": True,
        "unselected_vertices_byte_identical": True,
        "all_index_bytes_byte_identical": True,
        "header_bytes_identical": True,
        "source_geometry_and_topology_changed": False,
        "physical_mass_or_volume_changed": False,
        "interpretation": "inferred shared passive torso-pelvis kinematic reference; not measured attachment or visceral tissue mechanics",
    }
    output_receipt.setdefault("provenance", {})["passive_viscera_geometry_rebase"] = rebase
    output_receipt.setdefault("qualification", {})["passive_viscera_geometry_binding"] = \
        "inferred_reference_kinematic_attachment_only; no independent visceral mechanics"

    args.output_payload.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.mkdir(parents=True, exist_ok=False)
    args.output_payload.write_bytes(output_bytes)
    args.output_receipt.write_text(json.dumps(output_receipt, sort_keys=True, indent=2) + "\n")
    (args.evidence / "rebase-summary.json").write_text(json.dumps({
        "source_payload_sha256": source_hash,
        "output_payload_sha256": output_hash,
        "output_receipt_sha256": sha256(args.output_receipt.read_bytes()),
        "selected_surface_count": len(SELECTED_IDS),
        "selected_stable_ids": list(SELECTED_IDS),
        "source_body_index_by_stable_id": source_owners,
        "neutral_pose_max_world_position_error_m": float(max(point_errors)),
        "neutral_pose_rms_world_position_error_m": float(np.sqrt(np.mean(np.square(point_errors)))),
        "neutral_pose_max_world_normal_direction_error": float(max(normal_errors)),
        "native_registration_anchor_aabb_max_error_m": float(max_bound_error),
        "transition_endpoints_m": [pelvic_maximum_y, upper_minimum_y],
        "cardiac_surface_subset_sha256_before": source_cardiac_digest,
        "cardiac_surface_subset_sha256_after": candidate_cardiac_digest,
        "unselected_records_byte_identical": True,
        "unselected_vertices_byte_identical": True,
        "all_index_bytes_byte_identical": True,
        "header_bytes_identical": True,
        "rebase_receipt": rebase,
    }, sort_keys=True, indent=2) + "\n")
    (args.evidence / "README.md").write_text(
        "# Passive visceral geometry rebase\n\n"
        "This candidate rebases only the 80 explicitly listed passive organ surfaces into torso body 20's neutral frame, "
        "then records an inferred common torso-to-pelvis presentation convention. The original NHANAT1 header, topology, "
        "index bytes, source archive identity, surface order, and all nonselected surface data are preserved. Each selected "
        "source point is transformed through its exact accepted initial source-owner pose into world coordinates and back "
        "through the torso pose. The output is an existing NHANAT1 payload and its matching existing anatomy receipt.\n\n"
        "This is not a measurement of one individual, not a measured visceral attachment map, and not a visceral tissue solver. "
        "The new two-body binding carries passive geometry only; it adds no force, mass, or physiology. See `rebase-summary.json` "
        "and the generated anatomy receipt for source/output identities and byte-preservation checks.\n")
    print(json.dumps({"source_payload_sha256": source_hash, "output_payload_sha256": output_hash,
                      "output_receipt": str(args.output_receipt), "output_payload": str(args.output_payload),
                      "evidence": str(args.evidence), "selected_surface_count": len(SELECTED_IDS),
                      "neutral_pose_max_world_position_error_m": max(point_errors),
                      "neutral_pose_max_world_normal_direction_error": max(normal_errors),
                      "native_registration_anchor_aabb_max_error_m": max_bound_error,
                      "transition_endpoints_m": [pelvic_maximum_y, upper_minimum_y],
                      "cardiac_surface_subset_sha256": source_cardiac_digest}, sort_keys=True))
    return 0


# Populated by main() immediately before calls to surface_digest.
record_offsets: dict[int, int] = {}
vertex_base = 0
index_base = 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
