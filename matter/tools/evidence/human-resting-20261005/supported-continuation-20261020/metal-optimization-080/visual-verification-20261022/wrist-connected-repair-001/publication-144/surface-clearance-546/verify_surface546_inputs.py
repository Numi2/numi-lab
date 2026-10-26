from __future__ import annotations
import hashlib, importlib.util, json
from pathlib import Path
import numpy as np

D = Path("/Users/numi/Work/numi-human-connected-surface-069")
P = D / "D/surface-clearance-546/round-001"
candidate = D / "D/contained-clearance-545/last-exit-022/direct/candidate.npz"
audit = D / "D/contained-clearance-545/last-exit-022/check/audit-159/report.json"
face_npz = P / "surface-targets/targets.npz"
face_report = P / "surface-targets/report.json"
point_npz = P / "point-targets/nearest-union-exits.npz"
point_report = P / "point-targets/nearest-union-exits.json"
config = P / "matched-inputs.json"
depth_report = P / "depth/365-compatible-report.json"
fit = D / "D/D/component-lp-decomposition-490/run_matched_component_fit_490.py"
owner_path = D / "D/surface-witness-basis-408/propose_joint_clearance_witnesses_408.py"
out = P / "adapter-input-preflight-546.json"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

matched = load("matched490_preflight_546", fit)
validated = matched.validate_matched_inputs(config, {
    "candidate": candidate,
    "audit159": audit,
    "face_target_npz": face_npz,
    "point_target_npz": point_npz,
})
source_lock = matched.validate_source_lock()
owner = load("frozen408_preflight_546", owner_path)
with np.load(candidate, allow_pickle=False) as z:
    points = z["points_world_m"].copy()
    faces = z["source_O_faces"].copy()
point_targets, face_targets = owner.load_clearance_target_inputs(
    point_npz, point_report, face_npz, face_report, candidate, points, faces
)
depth = json.loads(depth_report.read_text())
surface = json.loads(face_report.read_text())
point = json.loads(point_report.read_text())
assert len(point_targets["point_rows"]) == 0
assert len(face_targets["face_rows"]) == 2861
assert depth["candidate_sha256"] == validated["inputs"]["candidate"]["sha256"]
assert depth["outside_row_count"] == 6339 and depth["inside_row_count"] == 0
assert surface["inputs"]["candidate_sha256"] == validated["inputs"]["candidate"]["sha256"]
assert surface["matched_159_complete"] is True
assert point["status"] == "PASS_EMPTY_POINT_TARGETS_ZERO_CONTAINMENT"
assert point["query_rows"] == point["passing_targets"] == point["inside_node_bank_pairs"] == 0
result = {
    "status": "PASS_INPUT_PRECHECKS_ONLY",
    "construction_only": True,
    "fit_or_geometry_audit_run": False,
    "candidate": {"path": str(candidate), "sha256": sha(candidate)},
    "matched_inputs": {"path": str(config), "sha256": sha(config),
                       "schema": validated["schema"],
                       "source_lock_count": len(source_lock)},
    "point_input": {"npz": str(point_npz), "sha256": sha(point_npz),
                    "report": str(point_report), "report_sha256": sha(point_report),
                    "status": point["status"], "rows": 0},
    "surface_input": {"npz": str(face_npz), "sha256": sha(face_npz),
                      "report": str(face_report), "report_sha256": sha(face_report),
                      "rows": len(face_targets["face_rows"]),
                      "matched_159_complete": surface["matched_159_complete"]},
    "outside_depth": {"report": str(depth_report), "report_sha256": sha(depth_report),
                      "outside_rows": depth["outside_row_count"],
                      "inside_rows": depth["inside_row_count"]},
    "validated_by": {"490_runner": str(fit), "490_runner_sha256": sha(fit),
                     "408_owner": str(owner_path), "408_owner_sha256": sha(owner_path)},
}
out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, sort_keys=True))
