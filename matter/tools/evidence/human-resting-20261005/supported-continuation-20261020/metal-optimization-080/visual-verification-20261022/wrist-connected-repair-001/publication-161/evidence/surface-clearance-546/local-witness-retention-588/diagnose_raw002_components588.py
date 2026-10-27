#!/usr/bin/env python3
"""Reproduce the array-only mutable-component census for saved 584 raw002."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from connected_witness_damping588 import boundary_nodes588, mutable_components588

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
BASE = ROOT / "D/surface-clearance-546/connected-topology-580/run-003/lineage/candidate.npz"
RAW = ROOT / "D/surface-clearance-546/connected-continuation-584/run-001/round-001/emitted-fit/fit/raw-joint-capture-563/raw-direction-002/candidate.npz"
SURFACE = ROOT / "D/surface-clearance-546/connected-topology-580/run-003/full-check/audit-159/report.json"
BASE_SHA256 = "8d5ac3dddb8bfebb77ef608fefcb484e8c648625ff5d7e84f876cd6fe0536d78"
RAW_SHA256 = "ae456cebd0f320956bbb139b32adcba8f941bd57ed8a6d256655dad1c27fb439"
SURFACE_SHA256 = "83cc6f6ee8839e1d5ca0991b0cb16854336f78c499603128ba8b62b2e44cd80e"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def diagnose():
    for path, expected in ((BASE, BASE_SHA256), (RAW, RAW_SHA256), (SURFACE, SURFACE_SHA256)):
        actual = sha(path)
        if actual != expected:
            raise ValueError(f"input pin mismatch for {path}: {actual}")
    with np.load(BASE, allow_pickle=False) as archive:
        base_points = archive["points_world_m"].copy()
        faces = archive["source_O_faces"].copy()
        anchors = archive["typed_coarse_anchor_output_rows_018"].copy()
    with np.load(RAW, allow_pickle=False) as archive:
        target_points = archive["points_world_m"].copy()
        target_faces = archive["source_O_faces"].copy()
    if base_points.dtype != np.float32 or target_points.dtype != np.float32 or target_points.shape != base_points.shape:
        raise ValueError("raw point domain differs from base")
    if not np.array_equal(faces, target_faces):
        raise ValueError("raw direction changed the face domain")
    mutable = np.any(base_points != target_points, axis=1)
    _, components = mutable_components588(faces, len(base_points), mutable)
    boundary = boundary_nodes588(faces, len(base_points))
    report159 = json.loads(SURFACE.read_text())
    if report159.get("candidate_sha256") != BASE_SHA256 or report159.get("complete") is not True:
        raise ValueError("baseline 159 report does not certify the pinned base")
    if report159.get("bank_count") != 185 or report159.get("expected_bank_count") != 185:
        raise ValueError("baseline 159 report lacks all 185 bank results")
    base_pairs = sum(int((bank.get("result") or {}).get("count", 0)) for bank in report159["banks"])
    return {
        "scope": "array-only mutable-node connectivity; not an audit or candidate check",
        "base_candidate_path": str(BASE),
        "base_candidate_sha256": BASE_SHA256,
        "raw_candidate_path": str(RAW),
        "raw_candidate_sha256": RAW_SHA256,
        "base_159_report_path": str(SURFACE),
        "base_159_report_sha256": SURFACE_SHA256,
        "base_surface_pair_count_from_existing_159": int(base_pairs),
        "node_count": int(len(base_points)),
        "face_count": int(len(faces)),
        "mutable_node_count": int(mutable.sum()),
        "mutable_component_count": int(len(components)),
        "mutable_component_sizes_descending": sorted((int(len(c)) for c in components), reverse=True),
        "boundary_node_count": int(len(boundary)),
        "mutable_boundary_node_count": int(mutable[boundary].sum()),
        "anchors_bitwise_unchanged": bool(np.array_equal(base_points[anchors].view(np.uint8), target_points[anchors].view(np.uint8))),
        "candidate_face_domain_unchanged": True,
        "exact_geometry_audit_run": False,
        "self_or_bone_clearance_claim": False,
        "native_admission": False,
        "accepted_physical_steps": 0,
        "script_sha256": sha(Path(__file__)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = diagnose()
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
