#!/usr/bin/env python3
"""Compare an accepted Matter root's nodal displacement with a pinned XPLT state.

This checks the common tissue-node displacement field only. It does not qualify
rigid wrenches, tissue stress, prestrain, contact, or source equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from open_knee_reference_state import load_displacement_reference


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checkpoint(path: Path):
    raw = path.read_bytes()
    if len(raw) < 60 or raw[:8] != b"NOKACPT1":
        raise ValueError("invalid accepted Matter checkpoint header")
    version, accepted = struct.unpack_from("<2I", raw, 8)
    source_time = struct.unpack_from("<d", raw, 16)[0]
    counts = struct.unpack_from("<9I", raw, 24)
    (node_count, rigid_count, reaction_count, fem_field_count,
     material_state_count, source_contact_count, candidate_contact_count,
     prestrain_count, material_state_stride) = counts
    if version != 1 or accepted != 1 or not np.isfinite(source_time) or source_time < 0:
        raise ValueError("checkpoint is not a finite accepted root")
    expected_bytes = (
        60 + node_count * 28 + rigid_count * 32 + reaction_count * 40
        + fem_field_count * 32 + material_state_count * 4
        + (source_contact_count + candidate_contact_count) * 48
        + prestrain_count * 32
    )
    if len(raw) != expected_bytes:
        raise ValueError("accepted Matter checkpoint length does not match its header")
    records = np.frombuffer(raw, dtype=np.dtype([
        ("source_id", "<u4"), ("xplt_index", "<u4"),
        ("cooked_node", "<u4"), ("object", "<u4"),
        ("displacement_m", "<f4", (3,)),
    ]), count=node_count, offset=60)
    if node_count == 0 or not np.isfinite(records["displacement_m"]).all():
        raise ValueError("accepted Matter checkpoint has no finite tissue displacements")
    if np.any(records["source_id"] == 0):
        raise ValueError("accepted Matter checkpoint has a zero source node identity")
    if len(np.unique(records["cooked_node"])) != node_count:
        raise ValueError("accepted Matter checkpoint repeats cooked node identities")
    if len(np.unique(records["source_id"])) != node_count:
        raise ValueError("accepted Matter checkpoint repeats source node identities")
    if len(np.unique(records["xplt_index"])) != node_count:
        raise ValueError("accepted Matter checkpoint repeats XPLT node indices")
    if not np.array_equal(records["xplt_index"], records["source_id"] - 1):
        raise ValueError("checkpoint source-node to XPLT-row mapping changed")
    return raw, source_time, records, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matter_checkpoint", type=Path)
    parser.add_argument("xplt_state_npz", type=Path)
    parser.add_argument("--expected-reference-sha256", required=True,
                        help="NPZ SHA256 from the independently frozen source extraction record")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    matter_raw, matter_time, records, counts = read_checkpoint(args.matter_checkpoint)
    metadata, source_displacement_mm = load_displacement_reference(
        args.xplt_state_npz, args.expected_reference_sha256)
    source_time = metadata["continuation_time"]
    source_state_index = metadata["state_index"]
    if abs(matter_time - source_time) > 2.0e-6:
        parser.error("Matter and XPLT continuation times differ")
    rows = records["xplt_index"].astype(np.int64)
    if rows.max(initial=-1) >= len(source_displacement_mm):
        parser.error("Matter checkpoint points outside the archived XPLT node table")

    matter_displacement_m = records["displacement_m"].astype(np.float64)
    reference_displacement_m = source_displacement_mm[rows] * 1.0e-3
    delta = matter_displacement_m - reference_displacement_m
    vector_error = np.linalg.norm(delta, axis=1)
    vector_reference = np.linalg.norm(reference_displacement_m, axis=1)
    reference_rms = float(np.sqrt(np.mean(np.square(vector_reference))))
    error_rms = float(np.sqrt(np.mean(np.square(vector_error))))
    result = {
        "schema": "numi.matter.open-knee-accepted-displacement-comparison.v1",
        "status": "accepted_root_displacement_comparison_only",
        "comparison_scope": "common continuum tissue-node displacement",
        "units": "metres",
        "source_xplt_state_index": source_state_index,
        "continuation_time": source_time,
        "source_xplt_sha256": metadata["source_xplt_sha256"],
        "matter_checkpoint_sha256": hashlib.sha256(matter_raw).hexdigest(),
        "xplt_state_npz_sha256": args.expected_reference_sha256,
        "reference_digest_admitted": True,
        "reference_digest_authority": "caller independently frozen source extraction record",
        "matter_checkpoint_counts": {
            "tissue_nodes": int(counts[0]),
            "rigid_bodies": int(counts[1]),
            "reaction_records": int(counts[2]),
            "fem_fields": int(counts[3]),
            "material_state_values": int(counts[4]),
            "accepted_contact_projections": int(counts[5]),
            "candidate_contact_projections": int(counts[6]),
            "prestrain_records": int(counts[7]),
            "material_state_stride": int(counts[8]),
        },
        "reference_xplt_node_count": int(len(source_displacement_mm)),
        "compared_node_count": int(len(rows)),
        "uncompared_xplt_nodes": int(len(source_displacement_mm) - len(rows)),
        "vector_displacement_error_rms_m": error_rms,
        "vector_displacement_error_max_m": float(vector_error.max()),
        "reference_vector_displacement_rms_m": reference_rms,
        "relative_vector_displacement_error_rms":
            error_rms / reference_rms if reference_rms > 0 else None,
        "component_error_rms_m": [
            float(np.sqrt(np.mean(np.square(delta[:, axis]))))
            for axis in range(3)
        ],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
