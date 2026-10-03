#!/usr/bin/env python3
"""Compare full-assembly candidate contact fields with retained XPLT state 1.

The Matter field is read from a rejected Newton candidate at an imported
FEBio pose. This diagnostic cannot qualify a native Matter checkpoint.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np


SOURCE_HASHES = (
    "f0d72ed6fdb7fced00a4656e53876791ffd7c995e803bceec7238d6fc4695181",
    "e8aa8771e80fc2a479ae80d38ee742272c50ac4fd54fffbb959cf2ebddbbb5e6",
    "de0ab6ec1de1a421187027beca0153b9b5768af14f90d19802c794e72cc10fb4",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_contact", type=Path)
    parser.add_argument("source_baseline", type=Path)
    parser.add_argument("xplt_state_1_npz", type=Path)
    parser.add_argument("matter_candidate_contact_fields", type=Path)
    args = parser.parse_args()
    source_paths = (args.source_contact, args.source_baseline,
                    args.xplt_state_1_npz)
    hashes = tuple(digest(path) for path in source_paths)
    if hashes != SOURCE_HASHES:
        parser.error("source contact, baseline, or XPLT state-1 hash changed")

    source = args.source_contact.read_bytes()
    if source[:8] != b"NHCNTP1\0" or struct.unpack_from("<4I", source, 8) != (
            1, 18, 36, 345070):
        parser.error("source contact topology changed")
    surfaces = []
    for index in range(36):
        offset = 156 + index * 76
        name = source[offset:offset + 32].split(b"\0", 1)[0].decode("ascii")
        first, count = struct.unpack_from("<2I", source, offset + 36)
        surfaces.append((name, first, count))
    if surfaces[0][1] != 0 or surfaces[-1][1] + surfaces[-1][2] != 345070:
        parser.error("source contact face ranges changed")

    baseline = json.loads(args.source_baseline.read_text())
    xplt_names = [row["name"] for row in
                  baseline["contact_archive"]["checkpoints"][0]["surfaces"]
                  if row["field"] == "contact gap"]
    if len(xplt_names) != 36 or set(xplt_names) != {row[0] for row in surfaces}:
        parser.error("XPLT source surface names changed")
    xplt = np.load(args.xplt_state_1_npz)

    field_bytes = args.matter_candidate_contact_fields.read_bytes()
    if field_bytes[:8] != b"NOKCFD2\0" or len(field_bytes) != 16 + 345070 * 16 or \
            struct.unpack_from("<2I", field_bytes, 8) != (2, 345070):
        parser.error("Matter candidate contact field layout changed")
    field = np.frombuffer(field_bytes, dtype="<f4", offset=16).reshape(-1, 4)
    if not np.isfinite(field).all() or np.any(field[:, 2] < 0) or \
            np.any(field[:, 2] > 3):
        parser.error("Matter candidate contact field is invalid")

    results = []
    pressure_differences = []
    reference_pressures = []
    total_source_active = 0
    total_matter_active = 0
    total_support_difference = 0
    for name, first, count in surfaces:
        xplt_index = xplt_names.index(name) + 1
        source_gap = xplt[f"surface_01_surface_{xplt_index:02d}"].astype(np.float64)
        source_pressure = xplt[f"surface_02_surface_{xplt_index:02d}"].astype(np.float64)
        if len(source_gap) != count or len(source_pressure) != count:
            parser.error(f"XPLT face count changed for {name}")
        matter_own_pressure = field[first:first + count, 0].astype(np.float64)
        matter_pressure = field[first:first + count, 3].astype(np.float64)
        matter_gap = field[first:first + count, 1].astype(np.float64)
        source_active = source_pressure > 0
        matter_active = field[first:first + count, 2] > 0
        union = source_active | matter_active
        total_source_active += int(source_active.sum())
        total_matter_active += int(matter_active.sum())
        total_support_difference += int(np.count_nonzero(
            source_active ^ matter_active))
        pressure_differences.append((matter_pressure - source_pressure)[union])
        reference_pressures.append(source_pressure[union])
        results.append({
            "name": name,
            "faces": count,
            "xplt_active_faces": int(source_active.sum()),
            "matter_candidate_active_faces": int(matter_active.sum()),
            "active_face_symmetric_difference": int(np.count_nonzero(
                source_active ^ matter_active)),
            "xplt_max_pressure_mpa": float(source_pressure.max()),
            "matter_candidate_max_own_pass_pressure_mpa":
                float(matter_own_pressure.max()),
            "matter_candidate_max_face_pressure_mpa": float(matter_pressure.max()),
            "pressure_difference_rms_mpa_on_union":
                rms((matter_pressure - source_pressure)[union]) if union.any() else None,
            "gap_difference_rms_mm_on_union":
                rms((matter_gap - source_gap)[union]) if union.any() else None,
        })
    pressure_differences = np.concatenate(pressure_differences)
    reference_pressures = np.concatenate(reference_pressures)
    print(json.dumps({
        "status": "rejected_root_diagnostic_only",
        "source_checkpoint": "FEBio XPLT state 1 at continuation time 0.05",
        "matter_state": "rejected coupled Newton candidate from imported XPLT pose",
        "source_hashes": hashes,
        "matter_candidate_contact_fields_sha256": digest(
            args.matter_candidate_contact_fields),
        "aggregate": {
            "xplt_active_faces": total_source_active,
            "matter_candidate_active_faces": total_matter_active,
            "active_face_symmetric_difference": total_support_difference,
            "pressure_difference_rms_mpa_on_union": rms(pressure_differences),
            "xplt_pressure_rms_mpa_on_union": rms(reference_pressures),
            "relative_pressure_rms_difference_on_union":
                rms(pressure_differences) / rms(reference_pressures),
        },
        "surfaces": results,
    }, indent=2))


if __name__ == "__main__":
    main()
