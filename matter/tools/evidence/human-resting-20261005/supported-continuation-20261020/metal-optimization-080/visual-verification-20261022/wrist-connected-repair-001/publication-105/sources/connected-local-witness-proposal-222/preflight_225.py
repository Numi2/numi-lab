#!/usr/bin/env python3
"""Read-only target/candidate compatibility preflight for the 222 builder."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import hashlib
import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
BUILDER = ROOT / "D/connected-local-witness-proposal-222/propose_joint_clearance_witnesses_222.py"
PREP = ROOT / "barycentric-connected-clearance-170/whole-connected-prep-225-run001"
CANDIDATE = ROOT / "barycentric-connected-clearance-170/whole-connected-halfstep-223-run001/pass-023/candidate.npz"
POINT_NPZ = PREP / "proposal/targets/nearest-union-exits.npz"
POINT_REPORT = PREP / "proposal/targets/nearest-union-exits.json"
FACE_NPZ = PREP / "targets/targets.npz"
FACE_REPORT = PREP / "targets/report.json"
OUT = ROOT / "D/connected-local-witness-proposal-222/preflight-225-run002.json"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

spec = importlib.util.spec_from_file_location("proposal222_preflight", BUILDER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {BUILDER}")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
with np.load(CANDIDATE, allow_pickle=False) as z:
    points = np.asarray(z["points_world_m"], dtype=np.float32)
    faces = np.asarray(z["source_O_faces"], dtype=np.int64)
    anchors = np.asarray(z["typed_coarse_anchor_output_rows_018"], dtype=np.int64)
pt, ft = m.load_clearance_target_inputs(
    POINT_NPZ, POINT_REPORT, FACE_NPZ, FACE_REPORT, CANDIDATE, points, faces
)
support = np.unique(np.r_[pt["point_rows"], ft["face_nodes"].ravel()]).astype(np.int64)
bary_owner = m.load_module("barycentric_preflight_091", m.DEFAULT_BARY_OWNER)
C, h, weights, metadata = m.build_clearance_rows(
    points.astype(np.float64) * 1000.0, faces, support, anchors, pt, ft, bary_owner
)
record = {
    "status": "PASS_TARGET_INPUT_COMPATIBILITY_PREFLIGHT_ONLY",
    "classification": "read-only schema/hash/affine-row check; no LP, geometry edit, exact audit, or physical step",
    "builder": {"path": str(BUILDER), "sha256": sha(BUILDER)},
    "candidate": {"path": str(CANDIDATE), "sha256": sha(CANDIDATE),
                  "points": int(len(points)), "faces": int(len(faces))},
    "point_targets": {"npz": str(POINT_NPZ), "npz_sha256": sha(POINT_NPZ),
                      "report": str(POINT_REPORT), "report_sha256": sha(POINT_REPORT),
                      "rows": int(len(pt["point_rows"]))},
    "face_targets": {"npz": str(FACE_NPZ), "npz_sha256": sha(FACE_NPZ),
                     "report": str(FACE_REPORT), "report_sha256": sha(FACE_REPORT),
                     "rows": int(len(ft["face_rows"]))},
    "barycentric_owner": {"path": str(m.DEFAULT_BARY_OWNER), "sha256": sha(m.DEFAULT_BARY_OWNER)},
    "clearance_target_nodes": int(len(support)),
    "sparse_clearance_rows": {"shape": list(C.shape), "nnz": int(C.nnz),
                              "minimum_positive_projected_target_mm": float(h.min()),
                              "normalized_weight_sum": float(weights.sum()),
                              "metadata_rows": int(len(metadata))},
    "full_self_area_point185_and_surface159_gates": "NOT_RUN_REQUIRED",
    "native_admission": False,
    "physical_steps": 0,
}
with OUT.open("x") as f:
    json.dump(record, f, indent=2, sort_keys=True)
    f.write("\n")
print(json.dumps(record, indent=2, sort_keys=True))

