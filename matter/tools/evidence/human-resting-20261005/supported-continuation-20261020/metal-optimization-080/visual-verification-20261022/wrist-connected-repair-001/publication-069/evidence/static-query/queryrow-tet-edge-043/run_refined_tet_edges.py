#!/usr/bin/env python3
"""Run the pinned 037006 refined-mesh edge audit in query-row identity space."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import time

import numpy as np

BUNDLE = pathlib.Path(__file__).resolve().parent
HELPER = BUNDLE / "matter/tools/human_skin_fat_bone_audit.py"
EXACT_HELPER_ROOT = BUNDLE / "matter"
ROOT = pathlib.Path(
    "/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/"
    "human-resting-20261005/supported-continuation-20261020/metal-optimization-080/"
    "visual-verification-20261022/wrist-connected-repair-001"
)
CANDIDATE = ROOT / "E/connected-volume-refinement-037/run-006/candidate.npz"
CANDIDATE_SHA = "52c81dbe323eab2240bc315f7b7bde605a201fa8e928206f27000de7e5d7a492"
CANDIDATE_REPORT = ROOT / "E/connected-volume-refinement-037/run-006/report.json"
CANDIDATE_REPORT_SHA = "05d26d04485c6d767e0805932e048cae6e7b3bb2e29117e8e44a032e78a8e693"
POINTS_CSV = ROOT / "E/generic-static-query-042/run-037006-current-001/query-points.csv"
POINTS_CSV_SHA = "3dc530517f52b960c5a207fadbce92de18ec19f3c55286b07a2741fcefb71a2f"
BANK_DIR = ROOT / "bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
OUTPUT_DIR = ROOT / "E/generic-static-query-043/run-037006-query-row-edges-001"
OUTPUT_REPORT = OUTPUT_DIR / "tet-edge-report.json"
OUTPUT_RUN = OUTPUT_DIR / "run-manifest.json"
QUERY_COUNT = 9320
ORIGINAL_PREFIX_ROWS = 7233


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path: pathlib.Path, expected: str, label: str) -> str:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: expected {expected}, got {actual}")
    return actual


def load_helper():
    spec = importlib.util.spec_from_file_location("human_skin_fat_bone_audit_043", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load existing edge-audit owner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"refusing to overwrite frozen audit output: {OUTPUT_DIR}")
    started = time.monotonic()
    helper = load_helper()
    if helper.sha256(HELPER) != "902269457740f70387267fce437d4198b1545a2e4f8d95fdbbdecc666a23f4c2":
        raise ValueError("edge-audit helper is not the pinned counter-corrected owner")
    verify(CANDIDATE, CANDIDATE_SHA, "037006 candidate")
    verify(CANDIDATE_REPORT, CANDIDATE_REPORT_SHA, "037006 construction report")
    verify(POINTS_CSV, POINTS_CSV_SHA, "042 refined query points")

    with np.load(CANDIDATE, allow_pickle=False) as data:
        points = np.asarray(data["current_points_f32"], dtype=np.float32)
        tets = np.asarray(data["tetrahedra"], dtype=np.int64)
        materials = np.asarray(data["material_ids"], dtype=np.int64)
        prefix_ids = np.asarray(data["global_node_ids_original_prefix"], dtype=np.int64)
    if points.shape != (QUERY_COUNT, 3) or tets.ndim != 2 or tets.shape[1] != 4:
        raise ValueError(f"candidate shape mismatch: points={points.shape}, tetrahedra={tets.shape}")
    if materials.shape != (len(tets),) or tets.min(initial=0) < 0 or tets.max(initial=-1) >= len(points):
        raise ValueError("candidate tetrahedron/material rows do not align to candidate points")
    if not np.isfinite(points).all():
        raise ValueError("candidate world positions contain nonfinite Float32 values")
    if prefix_ids.shape != (ORIGINAL_PREFIX_ROWS,):
        raise ValueError("037006 original global-ID prefix has wrong shape")

    # Generic-query input and exact native 039 point prefix are independently
    # checked in Float32. Appended rows remain query-row identities only.
    with POINTS_CSV.open(newline="") as f:
        point_rows = list(csv.DictReader(f))
    if len(point_rows) != QUERY_COUNT or [int(r["query_row"]) for r in point_rows] != list(range(QUERY_COUNT)):
        raise ValueError("042 query-points CSV row order/count differs from 037006")
    csv_points = np.asarray([[float(r[c]) for c in ("x_m", "y_m", "z_m")] for r in point_rows],
                            dtype=np.float32)
    csv_bits = np.asarray([[int(r[c]) for c in ("x_bits_u32", "y_bits_u32", "z_bits_u32")]
                           for r in point_rows], dtype=np.uint32)
    if not np.array_equal(csv_points.view(np.uint32), points.view(np.uint32)):
        raise ValueError("042 query-point Float32 coordinates differ from 037006 candidate")
    if not np.array_equal(csv_bits, points.view(np.uint32)):
        raise ValueError("042 explicit query-point Float32 bits differ from candidate")

    triangles, ids, bank_gids, bodies, matrices, offsets, positions, bankkeys, bank_pins, exact = (
        helper.read_bank_inputs(BANK_DIR, EXACT_HELPER_ROOT))
    if len(bankkeys) != 185 or len(np.unique(ids[:, 3])) == 0:
        raise ValueError("pinned 039 typed bank inventory is incomplete")
    if not np.array_equal(prefix_ids, bank_gids):
        raise ValueError("037006 original node identity prefix differs from exact 039 query rows")
    nodes = np.loadtxt(BANK_DIR / "exact-emitted.nodes.csv", delimiter=",", skiprows=1)
    native_prefix = nodes[:, 5:8].astype(np.float32)
    if native_prefix.shape != (ORIGINAL_PREFIX_ROWS, 3):
        raise ValueError("039 exact emitted prefix shape changed")
    if not np.array_equal(points[:ORIGINAL_PREFIX_ROWS].view(np.uint32), native_prefix.view(np.uint32)):
        raise ValueError("037006 original world Float32 prefix differs from 039 exact emitted nodes")

    # Use the pinned native raw-q world-to-body matrix and COM offset. The
    # existing owner validated matrix/quaternion parity for every used bank.
    body_local_points = {
        int(body): (points.astype(np.float64) - positions[int(body)]) @ matrices[int(body)].T
                 - offsets[int(body)]
        for body in np.unique(ids[:, 0])
    }
    query_rows = np.arange(len(points), dtype=np.int64)
    report = helper.audit_unique_tet_edges(
        query_rows, tets, ids, triangles, body_local_points,
        exact["classify_segment_triangle"], node_identity_domain="query_row")
    report["candidate"] = {
        "path": str(CANDIDATE), "sha256": CANDIDATE_SHA,
        "construction_report_path": str(CANDIDATE_REPORT),
        "construction_report_sha256": CANDIDATE_REPORT_SHA,
        "query_rows": len(points), "tetrahedra": len(tets),
        "material_ids_present": sorted(int(x) for x in np.unique(materials)),
        "prefix_global_ids_used_for_validation_only": True,
        "identity_domain": "query_row",
    }
    report["inputs"] = {
        str(CANDIDATE): CANDIDATE_SHA,
        str(CANDIDATE_REPORT): CANDIDATE_REPORT_SHA,
        str(POINTS_CSV): POINTS_CSV_SHA,
        **{str(BANK_DIR / name): pin for name, pin in bank_pins.items()},
        str(BANK_DIR / "all-bank-containment.csv.banks.csv"):
            sha256(BANK_DIR / "all-bank-containment.csv.banks.csv"),
        str(HELPER): sha256(HELPER),
        str(EXACT_HELPER_ROOT / "root_exact_segment_witness.py"):
            sha256(EXACT_HELPER_ROOT / "root_exact_segment_witness.py"),
    }
    report["frame_validation"] = {
        "generic_query_world_f32_bits_equal_candidate": True,
        "native_039_prefix_world_f32_bits_equal_candidate": True,
        "native_039_original_global_id_prefix_equal": True,
        "body_raw_quaternion_matrix_parity": "passed by pinned read_bank_inputs; max < 1e-12",
        "body_transform": "native raw world_to_body matrix and exact COM offset from 039",
        "appended_rows_native_global_identity": False,
    }
    report["limitations"] = [
        "Exact narrowphase is unique FEM tetrahedral edge segments against pinned 039 current-pose per-bone triangles.",
        "This is an edge witness audit, not a complete FEM surface/tetrahedron intersection or clearance certificate.",
        "Endpoints are query_row identities; no native compiled global node IDs are assigned to appended rows.",
        "No rest-pose audit, mass transfer/admission, or physical simulation.",
    ]
    elapsed = time.monotonic() - started
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    report_bytes = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    OUTPUT_REPORT.write_bytes(report_bytes)
    run = {
        "schema": "numi.human.query-row-tet-edge-run.v1",
        "status": report["status"],
        "candidate_sha256": CANDIDATE_SHA,
        "candidate_point_field": "current_points_f32",
        "identity_domain": "query_row",
        "helper_sha256": sha256(HELPER),
        "test_path": str(BUNDLE / "matter/tools/test_human_skin_fat_bone_audit.py"),
        "test_sha256": sha256(BUNDLE / "matter/tools/test_human_skin_fat_bone_audit.py"),
        "report_path": str(OUTPUT_REPORT),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "counts": report["counts"],
        "elapsed_seconds_including_input_validation": elapsed,
    }
    OUTPUT_RUN.write_text(json.dumps(run, sort_keys=True, indent=2) + "\n")
    print(json.dumps(run, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
