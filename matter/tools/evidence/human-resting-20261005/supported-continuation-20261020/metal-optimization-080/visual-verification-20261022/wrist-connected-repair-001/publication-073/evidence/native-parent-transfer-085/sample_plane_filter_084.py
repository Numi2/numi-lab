#!/usr/bin/env python3
"""Bounded 084 sample for the conservative 073 plane rejection."""
import csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
from plane_filter_085 import (
    INSIDE_TOL, _expanded_tetra_vertices, halfspaces,
    separated_by_owner_planes, tetra_intersection_volume,
)

ROOT = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
CAND = Path("/Users/n/Work/numi-human-connected-surface-069/whole-reference-tetgen-084-bounded/whole-skin-fat-reference-084-candidate.npz")
NODES = ROOT / "connected-skin-parent-mass-transfer-067/source-contribution-export-072/run-003/source-preflight/compiled-skin-fem-nodes.csv"
TETS = ROOT / "connected-skin-parent-mass-transfer-067/source-contribution-export-072/run-003/source-preflight/compiled-skin-fem-tets.csv"
OUT = Path(__file__).resolve().parent / "sample-128-report.json"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read_csv(p):
    with Path(p).open(newline="") as f:
        return list(csv.DictReader(f))

def main():
    t0 = time.perf_counter()
    z = np.load(CAND, allow_pickle=False)
    nodes = read_csv(NODES)
    tets = [r for r in read_csv(TETS) if int(r["identity_active_bit"]) == 1]
    native_global = np.asarray([int(r["global_node"]) for r in nodes], dtype=np.int64)
    native_rest = np.asarray([[float(r[k]) for k in ("rest_x", "rest_y", "rest_z")] for r in nodes])
    old_t = np.asarray([[int(r[f"source_node{j}"]) for j in range(4)] for r in tets], dtype=np.int64)
    old_xyz = native_rest[old_t]
    new_t = z["tetrahedra_zero_based"].astype(np.int64)
    new_xyz = z["reference_points_m"][new_t]
    if len(nodes) != 7233 or len(tets) != 28632 or len(new_t) != 32801:
        raise RuntimeError("pinned input dimensions changed")
    if not np.array_equal(z["source_global_node_ids"].astype(np.int64), native_global):
        raise RuntimeError("084 source global-node prefix does not equal compiled source rows")
    if not np.array_equal(z["reference_points_m"][:len(nodes)], native_rest):
        raise RuntimeError("084 source reference prefix is not bitwise native F32-promoted-F64")

    # Keep a material-stratified, deterministic 128-child sample.
    sample_ids = np.unique(np.linspace(0, len(new_t) - 1, 128, dtype=np.int64))
    old_lo, old_hi = old_xyz.min(axis=1), old_xyz.max(axis=1)
    counts = {"sample_children": int(len(sample_ids)), "aabb_pairs": 0,
              "plane_rejected": 0, "exact_survivors": 0, "positive_overlap": 0,
              "unculled_positive_overlap": 0}
    tclip = time.perf_counter()
    for ci in sample_ids:
        a = new_xyz[ci]
        lo, hi = a.min(axis=0), a.max(axis=0)
        candidates = np.flatnonzero(np.all(old_hi >= lo - INSIDE_TOL, axis=1) &
                                    np.all(old_lo <= hi + INSIDE_TOL, axis=1))
        counts["aabb_pairs"] += len(candidates)
        for oi in candidates:
            b = old_xyz[oi]
            rejected = separated_by_owner_planes(a, b)
            # Diagnostic parity: even rejected sample pairs are passed to the
            # exact frozen-owner clip once, then compared against filter result.
            reference = tetra_intersection_volume(a, b)
            if rejected:
                counts["plane_rejected"] += 1
                value = 0.0
                if reference != 0.0:
                    raise RuntimeError(f"unsafe plane reject child={ci} parent={oi} volume={reference}")
            else:
                counts["exact_survivors"] += 1
                value = reference
            counts["unculled_positive_overlap"] += int(reference > max(min(
                abs(float(np.linalg.det(a[1:]-a[0]))) / 6.0,
                abs(float(np.linalg.det(b[1:]-b[0]))) / 6.0) * 1e-13, 1e-25))
            counts["positive_overlap"] += int(value > max(min(
                abs(float(np.linalg.det(a[1:]-a[0]))) / 6.0,
                abs(float(np.linalg.det(b[1:]-b[0]))) / 6.0) * 1e-13, 1e-25))
    elapsed = time.perf_counter() - tclip
    report = {
        "status": "sample_complete",
        "method": "expanded-unit-halfspace-face rejection before exact frozen-073 triple-plane ConvexHull clip; every sampled rejected pair replayed through exact clip",
        "input_sha256": {"candidate_084": sha(CAND), "native_nodes": sha(NODES),
                         "native_tets": sha(TETS)},
        "dimensions": {"candidate_tets": int(len(new_t)), "native_active_tets": int(len(old_t)),
                       "candidate_nodes": int(len(z["reference_points_m"]))},
        "reference_frame": "candidate 084 F64 reference_points_m; first 7,233 rows bitwise equal to native compiled rest F32 promoted to F64",
        "sample_ids": sample_ids.tolist(),
        "counts": counts,
        "parity": {"unsafe_rejects": 0,
                   "filtered_positive_overlap_equals_unculled": counts["positive_overlap"] == counts["unculled_positive_overlap"]},
        "exact_pair_loop_seconds": elapsed,
        "total_seconds": time.perf_counter() - t0,
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()

