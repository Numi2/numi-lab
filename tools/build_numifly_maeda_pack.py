#!/usr/bin/env python3
"""Build Numifly's bilateral, torso-local Maeda wing artifact.

The input is BirdFlow's provenance-preserving conversion of Maeda et al.'s
measured right-wing surface.  This tool does not call the resulting geometry
measured at robot scale: the manifest records the mirror, spatial scale,
robot wingbeat frequency, and torso mount as explicit Numifly design steps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import struct


SOURCE_DATASET = "maeda-2017-hovering-right-wing-surface-v1"
SOURCE_ARCHIVE_SHA256 = (
    "5e32b920bfff91ab4ef8c8d872662ce4c6f45180421e42bcaab7729290499375"
)
SOURCE_JSON_SHA256 = (
    "5de3e1d9377ad652ab88d2f460287affd6055c69691e32f120d74cdf79628887"
)
ROBOT_LINEAR_SCALE = 0.09
SPATIAL_SCALE = 2.5
ROBOT_CYCLE_HZ = 28.8
TORSO_MOUNT_METERS = (-0.12 * ROBOT_LINEAR_SCALE, 0.0,
                      0.18 * ROBOT_LINEAR_SCALE)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def triangles(span_count: int, chord_count: int) -> list[tuple[int, int, int]]:
    result: list[tuple[int, int, int]] = []
    for span in range(span_count - 1):
        for chord in range(chord_count - 1):
            a = span * chord_count + chord
            b = (span + 1) * chord_count + chord
            c = b + 1
            d = a + 1
            result.extend(((a, b, c), (a, c, d)))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_json_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_json_sha256 != SOURCE_JSON_SHA256:
        raise ValueError("BirdFlow Maeda JSON fingerprint changed")
    source = json.loads(source_bytes)
    if source.get("datasetIdentifier") != SOURCE_DATASET:
        raise ValueError("unexpected Maeda dataset identifier")
    if source.get("scientificTier") != "measured-wing-only":
        raise ValueError("Maeda input is not the measured-wing-only artifact")
    if source.get("completeBirdReplayReady") is not False:
        raise ValueError("Maeda input changed its incomplete-bird boundary")
    if source.get("source", {}).get("sha256") != SOURCE_ARCHIVE_SHA256:
        raise ValueError("Maeda source archive fingerprint changed")
    if float(source.get("frequencyHz", 0.0)) != ROBOT_CYCLE_HZ:
        raise ValueError("Maeda measured wingbeat frequency changed")
    registration = source.get("coordinateRegistration", {})
    if registration.get("axes") != "BirdFlow forward/left/up":
        raise ValueError("Maeda coordinate registration changed")

    chord_count = int(source["chordCount"])
    span_count = int(source["spanCount"])
    phases = [float(value) for value in source["phases"]]
    source_positions = [float(value) for value in source["verticesMeters"]]
    source_vertex_count = chord_count * span_count
    frame_count = len(phases)
    if len(source_positions) != frame_count * source_vertex_count * 3:
        raise ValueError("Maeda position dimensions are inconsistent")
    if frame_count < 3 or any(
        not phases[index] > phases[index - 1]
        for index in range(1, frame_count)
    ):
        raise ValueError("Maeda phases are not strictly increasing")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    positions_path = output / "positions.f32le"
    triangles_path = output / "triangles.u16le"
    visual_path = output / "numifly-maeda-wings-phase0.stl"
    manifest_path = output / "manifest.json"

    bilateral: list[float] = []
    for frame in range(frame_count):
        first = frame * source_vertex_count * 3
        source_frame = source_positions[first:first + source_vertex_count * 3]
        # Left is derived solely by reflecting the measured right wing about
        # the robot sagittal plane. Components stay contiguous in every frame.
        for mirrored in (True, False):
            for offset in range(0, len(source_frame), 3):
                x, y, z = source_frame[offset:offset + 3]
                if mirrored:
                    y = -y
                bilateral.extend((
                    TORSO_MOUNT_METERS[0] + SPATIAL_SCALE * x,
                    TORSO_MOUNT_METERS[1] + SPATIAL_SCALE * y,
                    TORSO_MOUNT_METERS[2] + SPATIAL_SCALE * z,
                ))
    positions_path.write_bytes(struct.pack(f"<{len(bilateral)}f", *bilateral))

    right_triangles = triangles(span_count, chord_count)
    indices: list[int] = []
    # Reflection changes handedness, so reverse the derived left winding.
    for a, b, c in right_triangles:
        indices.extend((a, c, b))
    for a, b, c in right_triangles:
        indices.extend((
            source_vertex_count + a,
            source_vertex_count + b,
            source_vertex_count + c,
        ))
    if max(indices) > 0xFFFF:
        raise ValueError("Numifly wing topology exceeds the u16 index contract")
    triangles_path.write_bytes(struct.pack(f"<{len(indices)}H", *indices))

    frame_vertex_count = 2 * source_vertex_count
    phase_zero = bilateral[:frame_vertex_count * 3]
    with visual_path.open("wb") as stream:
        stream.write(b"Numifly bilateral Maeda wings phase 0".ljust(80, b"\0"))
        stream.write(struct.pack("<I", len(indices) // 3))
        for offset in range(0, len(indices), 3):
            points = []
            for index in indices[offset:offset + 3]:
                base = index * 3
                points.append(phase_zero[base:base + 3])
            ab = [points[1][axis] - points[0][axis] for axis in range(3)]
            ac = [points[2][axis] - points[0][axis] for axis in range(3)]
            normal = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ]
            length = math.sqrt(sum(value * value for value in normal))
            if length > 0:
                normal = [value / length for value in normal]
            stream.write(struct.pack("<12fH", *(normal + sum(points, [])), 0))

    manifest = {
        "format": "numi.measured-surface-pack.v1",
        "id": "numifly-maeda-bilateral-wing-v1",
        "dataset_identifier": SOURCE_DATASET,
        "scientific_tier": "measured-right-wing-derived-robot-morphology",
        "source": {
            "article_doi": source["source"]["articleDOI"],
            "dataset_doi": source["source"]["datasetDOI"],
            "license": source["source"]["license"],
            "archive_sha256": source["source"]["sha256"],
            "birdflow_json_sha256": source_json_sha256,
            "measured_frequency_hz": float(source["frequencyHz"]),
            "axes": registration["axes"],
            "origin": registration["origin"],
        },
        "numifly_design": {
            "right_wing": "measured Maeda right-wing surface",
            "left_wing": "sagittal reflection of right wing with reversed winding",
            "spatial_scale": SPATIAL_SCALE,
            "unitree_g1_linear_scale": ROBOT_LINEAR_SCALE,
            "unitree_g1_mass_scale": ROBOT_LINEAR_SCALE ** 3,
            "unitree_g1_inertia_scale": ROBOT_LINEAR_SCALE ** 5,
            "robot_cycle_hz": ROBOT_CYCLE_HZ,
            "temporal_mapping": (
                "17 measured phases mapped uniformly onto one periodic "
                "28.8 Hz cycle; the source wrap interval is time-reparameterized"
            ),
            "torso_mount_meters_forward_left_up": list(TORSO_MOUNT_METERS),
            "surface_contact": "aerodynamic force surface; no wing collision thickness claimed",
        },
        "layout": {
            "frame_count": frame_count,
            "source_vertex_count_per_wing": source_vertex_count,
            "vertex_count": frame_vertex_count,
            "triangle_count_per_wing": len(right_triangles),
            "triangle_count": len(indices) // 3,
            "chord_count": chord_count,
            "span_count": span_count,
            "phase_boundary": "wrap",
            "sample_rate_hz": ROBOT_CYCLE_HZ * (frame_count - 1),
            "action_count": 20,
        },
        "payloads": {
            "positions": positions_path.name,
            "positions_sha256": sha256(positions_path),
            "triangles": triangles_path.name,
            "triangles_sha256": sha256(triangles_path),
            "phase_zero_visual": visual_path.name,
            "phase_zero_visual_sha256": sha256(visual_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "positions_sha256": sha256(positions_path),
        "triangles_sha256": sha256(triangles_path),
        "visual_sha256": sha256(visual_path),
    }, indent=2))


if __name__ == "__main__":
    main()
