#!/usr/bin/env python3
import csv, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT = ROOT / "bone-clearance-fit-041-exact-material-routes"
REPORT = OUT / "route-report.json"
NODES = ROOT / "bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight/exact-emitted.nodes.csv"
OVERRIDE = ROOT / "bone-clearance-fit-038-candidate-query/run-002/candidate-node-override.csv"
NPZ = OUT / "construction-plane-constraints.npz"
RECEIPT = OUT / "construction-plane-adapter-receipt.json"
REPORT_SHA = "c631c390dc8c0acb5f5673d2d7c3efddcbe8d312d41427167eaf272d42fb8124"
NODES_SHA = "99e58d6afb68e0939670f69d4132bfa730cf3e988a7f70a179a41a89eac3802c"
OVERRIDE_SHA = "058db5c144a48d84670963081e2b1a6cfab826d79ec098e19eefd07d14841681"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

for p, expected in ((REPORT, REPORT_SHA), (NODES, NODES_SHA), (OVERRIDE, OVERRIDE_SHA)):
    if sha(p) != expected:
        raise RuntimeError(f"input pin mismatch: {p}")

report = json.loads(REPORT.read_text())
if report.get("status") != "EXACT_NATIVE_GEOMETRY_ROUTE_WITNESSES_NOT_FULL_TISSUE_CLEARANCE":
    raise RuntimeError("unexpected route report status")
with NODES.open(newline="") as f:
    node_rows = list(csv.DictReader(f))
if len(node_rows) != 7233:
    raise RuntimeError("native node dump count mismatch")
node = {}
for r in node_rows:
    layer, row, gid = int(r["layer"]), int(r["row"]), int(r["global_node"])
    xyz = np.asarray([np.float32(r[k]) for k in ("x_m", "y_m", "z_m")], dtype=np.float32).astype(np.float64)
    key = (layer, row)
    if key in node:
        raise RuntimeError(f"duplicate layer/row node {key}")
    node[key] = (gid, xyz)
with OVERRIDE.open(newline="") as f:
    override_rows = list(csv.DictReader(f))
if len(override_rows) != 7233:
    raise RuntimeError("candidate override count mismatch")
for r in override_rows:
    layer, row = int(r["layer"]), int(r["row"])
    got = np.asarray([np.float32(r[k]) for k in ("x_m", "y_m", "z_m")], dtype=np.float32)
    if int(r["global_node"]) != node[(layer, row)][0] or not np.array_equal(got.view(np.uint32), node[(layer, row)][1].astype(np.float32).view(np.uint32)):
        raise RuntimeError(f"native node/candidate mismatch at {(layer, row)}")

witness_by_key = {}
for w in report["route_witnesses"]:
    key = (int(w["row"]), int(w["segment"]), int(w["body"]), int(w["bone_stable_id"]))
    witness_by_key.setdefault(key, []).append(w)

planes = report["construction_planes_world"]
fields = {k: [] for k in (
    "plane_row", "plane_segment_code", "plane_start_global_node", "plane_end_global_node",
    "plane_body_index", "plane_bone_stable_id", "plane_source_record_index",
    "plane_triangle_stable_id", "plane_t", "plane_barycentric",
    "plane_point_world_m", "plane_outside_normal_world",
    "plane_start_world_f32_m", "plane_end_world_f32_m", "plane_start_classification",
    "plane_end_classification", "plane_target_layer_mask", "plane_margin_m")}
expanded = {k: [] for k in (
    "constraint_plane_index", "constraint_source_hit_row", "constraint_endpoint_node_row",
    "constraint_endpoint_global_node", "constraint_endpoint_layer",
    "constraint_hit_point_world_m", "constraint_outside_normal_world",
    "constraint_margin_m", "constraint_policy", "constraint_current_signed_distance_m",
    "constraint_current_margin_violation_m")}
entry_count = 0
for pi, p in enumerate(planes):
    row, seg = int(p["row"]), int(p["segment"])
    body, bone, tid = int(p["body"]), int(p["bone_stable_id"]), int(p["triangle_stable_id"])
    key = (row, seg, body, bone)
    candidates = witness_by_key.get(key, [])
    matches = [w for w in candidates if int(w["triangle_stable_id"]) == tid and w.get("is_first_route_hit")]
    if len(matches) != 1:
        raise RuntimeError(f"construction plane lacks unique exact first witness {key}, triangle {tid}")
    w = matches[0]
    if w["exact_first_hit"]["classification"] != "proper_interior_hit" or w.get("first_hit_ambiguous"):
        raise RuntimeError(f"plane is not an unambiguous proper hit: {key}")
    if w["start_classification"] not in ("outside", "outside_AABB_proven"):
        raise RuntimeError(f"entry start side not proven outside: {key}")
    expected_mask = 6 if seg == 0 else 4
    if int(p["target_layer_mask"]) != expected_mask:
        raise RuntimeError(f"unexpected target layer mask for {key}")
    n = np.asarray(p["unit_exterior_normal_world"], dtype=np.float64)
    point = np.asarray(p["plane_point_world_m"], dtype=np.float64)
    start = np.asarray(w["segment_start_world_f32_m"], dtype=np.float64)
    end = np.asarray(w["segment_end_world_f32_m"], dtype=np.float64)
    if abs(np.linalg.norm(n) - 1.0) > 1e-10 or float(np.dot(n, start - point)) <= 0:
        raise RuntimeError(f"entry-side normal does not point toward outside endpoint: {key}")
    if not np.allclose(point, np.asarray(w["plane_point_world_m"]), rtol=0, atol=0):
        raise RuntimeError(f"plane point disagrees with exact route witness: {key}")
    start_layer, end_layer = seg, seg + 1
    start_gid = node[(start_layer, row)][0]
    end_gid = node[(end_layer, row)][0]
    if w.get("start_global_node") is not None and (start_gid != int(w["start_global_node"]) or end_gid != int(w["end_global_node"])):
        raise RuntimeError(f"route endpoint ID mismatch: {key}")
    fields["plane_row"].append(row)
    fields["plane_segment_code"].append(seg)
    fields["plane_start_global_node"].append(start_gid)
    fields["plane_end_global_node"].append(end_gid)
    fields["plane_body_index"].append(body)
    fields["plane_bone_stable_id"].append(bone)
    fields["plane_source_record_index"].append(int(p["source_record_index"]))
    fields["plane_triangle_stable_id"].append(tid)
    fields["plane_t"].append(float(w["exact_first_hit"]["t"]))
    fields["plane_barycentric"].append(w["exact_first_hit"]["barycentric"])
    fields["plane_point_world_m"].append(point)
    fields["plane_outside_normal_world"].append(n)
    fields["plane_start_world_f32_m"].append(start)
    fields["plane_end_world_f32_m"].append(end)
    fields["plane_start_classification"].append(w["start_classification"])
    fields["plane_end_classification"].append(w["end_classification"])
    fields["plane_target_layer_mask"].append(expected_mask)
    fields["plane_margin_m"].append(float(p["clearance_m"]))
    policy = "OI_entry_requires_I_and_D" if seg == 0 else "ID_entry_requires_D"
    for target_layer in (1, 2):
        if not (expected_mask & (1 << target_layer)):
            continue
        gid, xyz = node[(target_layer, row)]
        signed = float(np.dot(n, xyz - point))
        expanded["constraint_plane_index"].append(pi)
        expanded["constraint_source_hit_row"].append(row)
        expanded["constraint_endpoint_node_row"].append(row)
        expanded["constraint_endpoint_global_node"].append(gid)
        expanded["constraint_endpoint_layer"].append(target_layer)
        expanded["constraint_hit_point_world_m"].append(point)
        expanded["constraint_outside_normal_world"].append(n)
        expanded["constraint_margin_m"].append(float(p["clearance_m"]))
        expanded["constraint_policy"].append(policy)
        expanded["constraint_current_signed_distance_m"].append(signed)
        expanded["constraint_current_margin_violation_m"].append(max(0.0, float(p["clearance_m"]) - signed))
    entry_count += 1
if len(planes) != 121 or entry_count != 121:
    raise RuntimeError("construction plane count changed from frozen report")

arrays = {
    "plane_row": np.asarray(fields["plane_row"], dtype=np.int32),
    "plane_segment_code": np.asarray(fields["plane_segment_code"], dtype=np.uint8),
    "plane_start_global_node": np.asarray(fields["plane_start_global_node"], dtype=np.int64),
    "plane_end_global_node": np.asarray(fields["plane_end_global_node"], dtype=np.int64),
    "plane_body_index": np.asarray(fields["plane_body_index"], dtype=np.int32),
    "plane_bone_stable_id": np.asarray(fields["plane_bone_stable_id"], dtype=np.int32),
    "plane_source_record_index": np.asarray(fields["plane_source_record_index"], dtype=np.int32),
    "plane_triangle_stable_id": np.asarray(fields["plane_triangle_stable_id"], dtype=np.int64),
    "plane_t": np.asarray(fields["plane_t"], dtype=np.float64),
    "plane_barycentric": np.asarray(fields["plane_barycentric"], dtype=np.float64),
    "plane_point_world_m": np.asarray(fields["plane_point_world_m"], dtype=np.float64),
    "plane_outside_normal_world": np.asarray(fields["plane_outside_normal_world"], dtype=np.float64),
    "plane_start_world_f32_m": np.asarray(fields["plane_start_world_f32_m"], dtype=np.float64),
    "plane_end_world_f32_m": np.asarray(fields["plane_end_world_f32_m"], dtype=np.float64),
    "plane_start_classification": np.asarray(fields["plane_start_classification"], dtype="U32"),
    "plane_end_classification": np.asarray(fields["plane_end_classification"], dtype="U32"),
    "plane_target_layer_mask": np.asarray(fields["plane_target_layer_mask"], dtype=np.uint8),
    "plane_margin_m": np.asarray(fields["plane_margin_m"], dtype=np.float64),
    "constraint_plane_index": np.asarray(expanded["constraint_plane_index"], dtype=np.int32),
    "constraint_source_hit_row": np.asarray(expanded["constraint_source_hit_row"], dtype=np.int32),
    "constraint_endpoint_node_row": np.asarray(expanded["constraint_endpoint_node_row"], dtype=np.int32),
    "constraint_endpoint_global_node": np.asarray(expanded["constraint_endpoint_global_node"], dtype=np.int64),
    "constraint_endpoint_layer": np.asarray(expanded["constraint_endpoint_layer"], dtype=np.uint8),
    "constraint_hit_point_world_m": np.asarray(expanded["constraint_hit_point_world_m"], dtype=np.float64),
    "constraint_outside_normal_world": np.asarray(expanded["constraint_outside_normal_world"], dtype=np.float64),
    "constraint_margin_m": np.asarray(expanded["constraint_margin_m"], dtype=np.float64),
    "constraint_policy": np.asarray(expanded["constraint_policy"], dtype="U28"),
    "constraint_current_signed_distance_m": np.asarray(expanded["constraint_current_signed_distance_m"], dtype=np.float64),
    "constraint_current_margin_violation_m": np.asarray(expanded["constraint_current_margin_violation_m"], dtype=np.float64),
}
np.savez_compressed(NPZ, **arrays)
receipt = {
    "status": "derived_constraint_sidecar_not_physical_acceptance",
    "route_report": str(REPORT), "route_report_sha256": REPORT_SHA,
    "candidate_world_nodes": str(OVERRIDE), "candidate_world_nodes_sha256": OVERRIDE_SHA,
    "native_emitted_nodes": str(NODES), "native_emitted_nodes_sha256": NODES_SHA,
    "constraint_npz": str(NPZ), "constraint_npz_sha256": sha(NPZ),
    "route_planes": len(planes), "expanded_endpoint_constraints": len(expanded["constraint_plane_index"]),
    "policies": {"OI_entry_requires_I_and_D": int(sum(x == "OI_entry_requires_I_and_D" for x in expanded["constraint_policy"])),
                 "ID_entry_requires_D": int(sum(x == "ID_entry_requires_D" for x in expanded["constraint_policy"]))},
    "assertions": ["every plane maps to one exact first-route proper interior witness",
                   "entry starts are outside or AABB-proven outside",
                   "outside normal points toward the start-side endpoint",
                   "target layers exactly follow mask 6 for O-I and mask 4 for I-D",
                   "all target global node IDs match the native 039 Float32 node dump"],
    "current_violation_summary": {
        "positive_margin_violation_constraints": int(sum(x > 0 for x in expanded["constraint_current_margin_violation_m"])),
        "maximum_margin_violation_m": max(expanded["constraint_current_margin_violation_m"], default=0.0),
        "minimum_signed_distance_m": min(expanded["constraint_current_signed_distance_m"], default=0.0)
    },
    "limitations": ["The plane rows encode local route-entry construction constraints only.",
                    "They do not certify tissue surface or tetrahedron clearance and do not admit a physical/native state.",
                    "Ambiguous and nonconstruction route hits remain in the frozen route-report JSON and are not binding rows."]
}
RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, indent=2, sort_keys=True))
