#!/usr/bin/env python3
"""Bind one archived FEBio XPLT checkpoint to the frozen Open Knee FEM order.

The output is a diagnostic initial state for Matter, not an equilibrium claim.
Requires NumPy. All source/archive identities and the exact plot mesh ordering
are checked before writing the seed and its provenance manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


TISSUES = ("QAT", "TBC-L", "PCL", "PTC", "ACL", "MCL", "PTL",
           "MNS-L", "MNS-M", "LCL", "TBC-M", "FMC")
RIGID = (1, 2, 3, 4, 17, 18, 19, 20, 21)


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def plot_mesh_nodes(path: Path) -> tuple[np.ndarray, str]:
    with path.open("rb") as stream:
        def read(start: int, count: int) -> bytes:
            stream.seek(start)
            result = stream.read(count)
            if len(result) != count:
                raise ValueError("truncated XPLT chunk")
            return result

        def chunks(start: int, end: int):
            while start < end:
                tag, count = struct.unpack("<II", read(start, 8))
                payload = start + 8
                if payload + count > end:
                    raise ValueError("XPLT chunk exceeds parent")
                yield tag, payload, payload + count
                start = payload + count

        if read(0, 4) != b"BEF\0":
            raise ValueError("XPLT signature differs from archived format")
        root_tag, root_count = struct.unpack("<II", read(4, 8))
        if root_tag != 0x01000000:
            raise ValueError("XPLT root is missing")
        sections = {tag: (lo, hi) for tag, lo, hi in chunks(12, root_count + 12)}
        mesh_lo, mesh_hi = sections[0x01040000]
        mesh_hash = hashlib.sha256(read(mesh_lo, mesh_hi - mesh_lo)).hexdigest()
        mesh_sections = {tag: (lo, hi) for tag, lo, hi in chunks(mesh_lo, mesh_hi)}
        _, node_lo, node_hi = next(chunks(*mesh_sections[0x01041000]))
        raw = read(node_lo, node_hi - node_lo)
        if len(raw) % 12:
            raise ValueError("XPLT mesh node payload is not float32[3]")
        return np.frombuffer(raw, dtype="<f4").reshape(-1, 3).copy(), mesh_hash


def fit_pose(reference: np.ndarray, current: np.ndarray,
             source_center: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center_a = reference.mean(axis=0)
    center_b = current.mean(axis=0)
    u, _, vt = np.linalg.svd((reference - center_a).T @ (current - center_b))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    moved = (reference - center_a) @ rotation.T + center_b
    rms = float(np.sqrt(np.mean(np.sum((moved - current) ** 2, axis=1))))
    center = (source_center - center_a) @ rotation.T + center_b
    # Positive-trace branch is valid for the retained near-neutral checkpoint.
    trace = float(np.trace(rotation))
    if trace <= 0:
        raise ValueError("checkpoint pose is outside the supported quaternion branch")
    qw = np.sqrt(1.0 + trace) / 2.0
    quaternion = np.array([
        (rotation[2, 1] - rotation[1, 2]) / (4.0 * qw),
        (rotation[0, 2] - rotation[2, 0]) / (4.0 * qw),
        (rotation[1, 0] - rotation[0, 1]) / (4.0 * qw), qw,
    ])
    return center, quaternion, rms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanics", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--rigid-ties", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--xplt", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mechanics = json.loads(args.mechanics.read_text())
    baseline = json.loads(args.reference.read_text())
    geometry_hash = digest(args.geometry)
    expected_geometry = mechanics["source_geometry_resolution"]["binary_storage"]["sha256"]
    if geometry_hash != expected_geometry:
        raise ValueError("source geometry binary hash mismatch")
    archive_hash = digest(args.xplt)
    if archive_hash != baseline["contact_archive"]["source_xplt_sha256"]:
        raise ValueError("archived XPLT hash mismatch")
    with np.load(args.checkpoint) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata_json"]))
        if metadata["reference_plot_sha256"] != archive_hash:
            raise ValueError("checkpoint belongs to a different XPLT")
        displacement = np.array(checkpoint["node_01"], dtype="<f4").reshape(-1, 3)
        field = next(row for row in metadata["fields"]
                     if row["class"] == "node" and row["field"] == "displacement")
        if hashlib.sha256(displacement.tobytes()).hexdigest() != field["float32_array_sha256"]:
            raise ValueError("checkpoint displacement hash mismatch")
    plot_reference, mesh_hash = plot_mesh_nodes(args.xplt)
    if metadata["source_mesh_sha256"] != mesh_hash or plot_reference.shape != displacement.shape:
        raise ValueError("checkpoint plot mesh identity or node count mismatch")

    groups = mechanics["source_geometry_resolution"]["node_coordinate_groups"]
    geometry = args.geometry.read_bytes()
    node_index: dict[int, int] = {}
    group_indices: dict[str, np.ndarray] = {}
    source_coordinates = np.empty(plot_reference.shape, dtype=np.float64)
    cursor = 0
    for name, group in sorted(groups.items(), key=lambda row: row[1]["binary_offset_bytes"]):
        first = cursor
        for local in range(group["node_count"]):
            node_id, x, y, z = struct.unpack_from(
                "<I3d", geometry, group["binary_offset_bytes"] + 28 * local)
            if node_id in node_index:
                raise ValueError("source geometry node ID is duplicated")
            node_index[node_id] = cursor
            source_coordinates[cursor] = (x, y, z)
            cursor += 1
        group_indices[name] = np.arange(first, cursor)
    if cursor != len(plot_reference) or not np.array_equal(
            source_coordinates.astype("<f4"), plot_reference):
        raise ValueError("source geometry order differs from archived XPLT mesh")
    current = source_coordinates + displacement.astype(np.float64)

    ties = args.rigid_ties.read_bytes()
    if ties[:8] != b"NHTIES1\0":
        raise ValueError("rigid-tie program has an unexpected header")
    tie_count = struct.unpack_from("<I", ties, 16)[0]
    tied_qso: list[int] = []
    for row in range(tie_count):
        node_id, _, body_id, _ = struct.unpack_from("<4I", ties, 88 + 16 * row)
        if body_id == 21:
            tied_qso.append(node_index[node_id])

    body_program = {row["material_id"]: row for row in mechanics["rigid_graph"]["bodies"]}
    archive_checkpoint = next(
        (row for row in baseline["reference_run"]["checkpoints"]
         if abs(row["continuation_time"] - metadata["continuation_time"]) < 1e-8), None)
    if archive_checkpoint is None:
        raise ValueError("archived rigid checkpoint is unavailable")
    archived_bodies = {row["material_id"]: row
                       for row in archive_checkpoint["rigid_bodies"].values()}
    if set(archived_bodies) != set(RIGID):
        raise ValueError("archived rigid body set differs from source graph")

    poses = []
    pose_rms = {}
    for material_id in RIGID:
        archived = archived_bodies[material_id]
        center = np.array(archived["center_of_mass"], dtype=np.float64)
        quaternion = np.array(archived["rotation_quaternion_xyzw"], dtype=np.float64)
        points = group_indices.get(body_program[material_id]["name"])
        if material_id == 21:
            points = np.array(tied_qso)
        if points is not None and len(points) >= 3:
            source_center = np.array(body_program[material_id]["center_of_mass"])
            center, quaternion, rms = fit_pose(
                source_coordinates[points], current[points], source_center)
            if rms > 1e-6:
                raise ValueError(f"rigid material {material_id} is not rigid in XPLT: {rms} mm")
            pose_rms[str(material_id)] = rms
        poses.append((material_id, center * 0.001, quaternion))

    tissue_indices = np.concatenate([group_indices[name] for name in TISSUES])
    positions = (current[tissue_indices] * 0.001).astype("<f4")
    # Preserve the archived displacement independently of the rounded world
    # position. Tiny source tetrahedra lose stress-significant edge bits when
    # a current position is stored as one absolute float32 metre coordinate.
    tissue_displacements_mm = displacement[tissue_indices].astype("<f4")
    if len(positions) != 194729 or not np.isfinite(positions).all():
        raise ValueError("source tissue position payload is invalid")
    output = bytearray(b"NOKSEED3")
    output += struct.pack("<IIIId", 3, len(positions), len(poses),
                          metadata["state_index"], metadata["continuation_time"])
    output += bytes.fromhex(baseline["source_deck_sha256"])
    output += bytes.fromhex(baseline["source_geometry_sha256"])
    output += bytes.fromhex(archive_hash)
    output += positions.tobytes()
    output += tissue_displacements_mm.tobytes()
    for material_id, center, quaternion in poses:
        output += struct.pack("<I3f4f", material_id, *center, *quaternion)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    manifest = {
        "schema": "numi.matter.open-knee-archived-checkpoint-seed.v3",
        "status": "diagnostic_initial_state_not_equilibrium",
        "state_index": metadata["state_index"],
        "continuation_time": metadata["continuation_time"],
        "source_deck_sha256": baseline["source_deck_sha256"],
        "source_geometry_binary_sha256": geometry_hash,
        "source_xplt_sha256": archive_hash,
        "source_xplt_mesh_sha256": mesh_hash,
        "checkpoint_sha256": digest(args.checkpoint),
        "seed_sha256": hashlib.sha256(output).hexdigest(),
        "rigid_fit_rms_mm": pose_rms,
        "tissue_node_count": len(positions),
        "tissue_displacement_units": "mm_float32_original_xplt",
        "rigid_body_count": len(poses),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
