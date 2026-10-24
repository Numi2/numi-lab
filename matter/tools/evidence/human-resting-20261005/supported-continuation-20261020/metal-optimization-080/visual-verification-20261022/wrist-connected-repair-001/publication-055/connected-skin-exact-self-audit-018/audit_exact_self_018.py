from __future__ import annotations
import hashlib, importlib, json, pathlib, sys, time, traceback
from fractions import Fraction
import numpy as np

E = pathlib.Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
OUT = E / "connected-skin-exact-self-audit-018"
OUT.mkdir(parents=True, exist_ok=True)
BASELINE = E / "E/connected-skin-mechanical-transfer-013/boundary-refinement-002/candidate.npz"
CANDIDATES = [
    ("016-run-001", E / "connected-skin-ray-envelope-016/run-001/outer-envelope-proposal.npz"),
    ("017-run-002", E / "connected-skin-safe-flow-017/run-002/connected-rest-proposal.npz"),
    ("017-run-003", E / "connected-skin-safe-flow-017/run-003/connected-rest-proposal.npz"),
]
PINNED = {
    str(BASELINE): "1849d24c14f8c0ddfedf3aa131877bdd38a852f9e48baebcf900a0efadf75db1",
    str(CANDIDATES[0][1]): "ee9f6173c6662d9ee52d624c7b22258b14b62d9e3b0170b1bf4c3eaa1df5f10e",
    str(CANDIDATES[1][1]): "64f40e915703e35950ba2d3ab95b01a5e75681616c316cace25518c3a0ccb054",
    str(CANDIDATES[2][1]): "bacc0f763147b3e534f082b8797a865691b93b2134c1a9f9e86c9ff870ef0d1f",
}
OWNER = E / "collision-free-seam-records-owner-315"
SRC = OWNER / "src"
COMPAT = pathlib.Path("/Users/n/numi-human-anatomy-finish-1378/src/numilab_human")
import types
pkg = types.ModuleType("numilab_human")
pkg.__path__ = [str(SRC / "numilab_human"), str(COMPAT)]
sys.modules["numilab_human"] = pkg
ci = importlib.import_module("numilab_human.cardiac_cavity_intersections")
DEN = 1 << 149

def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()

def atomic_json(path, value):
    tmp = pathlib.Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)

def load_mesh(path):
    with np.load(path, allow_pickle=False) as z:
        points = z["points_world_m"].copy()
        faces = z["physical_outer_faces"].copy()
        declared = z["changed_node_ids"].copy() if "changed_node_ids" in z.files else None
    return points, faces, declared

def degenerate_rows(points, faces):
    tri = points[faces].astype(np.float64)
    cross = np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0])
    return np.flatnonzero(np.all(cross == 0.0, axis=1))

def normal_comparison(base_points, new_points, faces):
    a = base_points[faces].astype(np.float64)
    b = new_points[faces].astype(np.float64)
    na = np.cross(a[:,1]-a[:,0], a[:,2]-a[:,0])
    nb = np.cross(b[:,1]-b[:,0], b[:,2]-b[:,0])
    aa = np.linalg.norm(na, axis=1)
    ab = np.linalg.norm(nb, axis=1)
    good = (aa > 0.0) & (ab > 0.0)
    cosine = np.full(len(faces), np.nan, dtype=np.float64)
    cosine[good] = np.einsum("ij,ij->i", na[good], nb[good]) / (aa[good] * ab[good])
    return {
        "valid_face_count": int(good.sum()),
        "degenerate_baseline_count": int((aa == 0).sum()),
        "degenerate_candidate_count": int((ab == 0).sum()),
        "nonpositive_normal_cosine_count": int(np.sum(cosine[good] <= 0)),
        "minimum_normal_cosine": float(np.min(cosine[good])) if good.any() else None,
    }

def exact_records(points, faces):
    if points.dtype != np.float32:
        raise ValueError("expected emitted Float32 point array")
    records, effective_ids = ci.float32_surface_records(points, faces, source_seam_groups=())
    if tuple(effective_ids) != tuple(range(len(points))):
        raise AssertionError("unexpected effective IDs without seam groups")
    return records

def pair_detail(pair, records):
    a_id, b_id = map(int, pair)
    by_id = {r[3]: r for r in records}
    a, b = by_id[a_id], by_id[b_id]
    points = ci.triangle_intersection_points(a[0], b[0])
    return {
        "face_pair": [a_id, b_id],
        "source_vertex_ids": [list(a[4]), list(b[4])],
        "intersection_vertices_exact_world_m": [
            [str(Fraction(coord, DEN)) for coord in point] for point in points
        ],
    }

started = time.monotonic()
for p, expected in PINNED.items():
    actual = sha(p)
    if actual != expected:
        raise RuntimeError(f"input hash mismatch: {p}: {actual} != {expected}")

base_points, base_faces, _ = load_mesh(BASELINE)
if base_points.dtype != np.float32:
    raise RuntimeError("013 baseline points are not emitted Float32")
if base_faces.ndim != 2 or base_faces.shape[1] != 3:
    raise RuntimeError("unexpected baseline face shape")
base_degenerate = degenerate_rows(base_points, base_faces)
if len(base_degenerate):
    raise RuntimeError(f"baseline has degenerate F32 faces: {base_degenerate[:20].tolist()}")
base_records = exact_records(base_points, base_faces)
t0 = time.monotonic()
base_cache, base_result = ci._prepare_complete_self_pair_baseline(base_records)
base_seconds = time.monotonic() - t0
base_pairs = tuple(map(tuple, base_result["triangle_pairs"]))
base_report = {
    "path": str(BASELINE), "sha256": sha(BASELINE), "point_sha256": hashlib.sha256(base_points.tobytes()).hexdigest(),
    "face_sha256": hashlib.sha256(base_faces.tobytes()).hexdigest(),
    "point_count": len(base_points), "face_count": len(base_faces),
    "degenerate_face_count": int(len(base_degenerate)),
    "exact_self_pair_count": len(base_pairs),
    "exact_pair_set_sha256": hashlib.sha256(json.dumps(base_pairs, separators=(",", ":")).encode()).hexdigest(),
    "full_exact_audit": True,
    "algorithm": ci.ALGORITHM,
    "baseline_audit_seconds": base_seconds,
    "aabb_candidate_pairs": int(base_result.get("aabb_candidate_pairs", 0)),
    "allowed_shared_vertex_or_edge_pairs": int(base_result.get("allowed_shared_vertex_or_edge_pairs", 0)),
    "source_seam_groups": [],
    "shared_feature_policy": "original indexed Float32 vertex IDs only; no coordinate welding",
}
report = {
    "schema": "connected-skin-exact-self-audit-018-v1",
    "status": "RUNNING",
    "baseline": base_report,
    "owner_source": {
        "module": str(OWNER / "src/numilab_human/cardiac_cavity_intersections.py"),
        "module_sha256": sha(OWNER / "src/numilab_human/cardiac_cavity_intersections.py"),
        "algorithm": ci.ALGORITHM,
        "geometry_module_sha256": sha(COMPAT / "cardiac_cavity_geometry.py"),
        "model_module_sha256": sha(COMPAT / "model.py"),
        "physiology_module_sha256": sha(COMPAT / "physiology.py"),
    },
    "method": {
        "comparison_baseline": "same 013v2 emitted Float32 candidate for all rows",
        "candidate_scope": "all faces incident to any candidate vertex whose packed Float32 xyz differs bytewise from baseline; each selected face is tested against the complete surface",
        "incremental_owner": "_update_complete_self_pair_baseline; unchanged-pair reuse only when both complete exact triangle records and source IDs are identical",
        "face_identity": "dense physical_outer_faces row IDs preserved unchanged",
        "interpretation": "negative baseline-to-candidate normal cosine is diagnostic only; exact self-crossing classification comes from owner315 triangle intersections",
    },
    "candidates": {},
    "script_sha256": sha(__file__),
}
atomic_json(OUT / "report.json", report)
print(json.dumps({"phase":"baseline_complete","pairs":len(base_pairs),"seconds":base_seconds,"candidates":len(CANDIDATES)}), flush=True)

for name, path in CANDIDATES:
    t_start = time.monotonic()
    item = {"name": name, "path": str(path), "sha256": sha(path)}
    try:
        points, faces, declared = load_mesh(path)
        if points.dtype != np.float32:
            raise RuntimeError("candidate points are not emitted Float32")
        if points.shape != base_points.shape or faces.shape != base_faces.shape:
            raise RuntimeError(f"point/face shape differs: {points.shape}/{faces.shape}")
        if not np.array_equal(faces, base_faces):
            raise RuntimeError("candidate topology/index rows differ from 013v2")
        changed_nodes = np.flatnonzero(np.any(points.view(np.uint32) != base_points.view(np.uint32), axis=1))
        changed_faces = np.flatnonzero(np.isin(faces, changed_nodes).any(axis=1))
        declared_list = None if declared is None else sorted(map(int, declared.tolist()))
        computed_list = list(map(int, changed_nodes.tolist()))
        item.update({
            "point_sha256": hashlib.sha256(points.tobytes()).hexdigest(),
            "face_sha256": hashlib.sha256(faces.tobytes()).hexdigest(),
            "point_count": len(points), "face_count": len(faces),
            "changed_node_count_bytewise_vs_013v2": len(changed_nodes),
            "changed_node_ids_bytewise_vs_013v2": computed_list,
            "declared_changed_node_count": None if declared is None else int(len(declared)),
            "declared_changed_node_ids_match_bytewise_baseline": None if declared is None else (declared_list == computed_list),
            "changed_incident_face_count": len(changed_faces),
            "changed_incident_face_rows": list(map(int, changed_faces.tolist())),
            "changed_nodes_float32_max_displacement_mm": float(np.linalg.norm(points[changed_nodes].astype(np.float64)-base_points[changed_nodes].astype(np.float64),axis=1).max()*1000) if len(changed_nodes) else 0.0,
            "normal_comparison_vs_013v2": normal_comparison(base_points, points, faces),
        })
        degenerates = degenerate_rows(points, faces)
        item["degenerate_candidate_face_count"] = int(len(degenerates))
        item["degenerate_candidate_face_rows"] = list(map(int, degenerates.tolist()))
        if len(degenerates):
            item["status"] = "FAIL_CANDIDATE_HAS_DEGENERATE_FLOAT32_FACES"
            item["audit_error"] = "315 complete-record builder fails closed on degenerate faces; exact pair screen not run"
            item["elapsed_seconds"] = time.monotonic()-t_start
        else:
            records = exact_records(points, faces)
            record_by_id = {r[3]: r for r in records}
            owner_cache, result = ci._update_complete_self_pair_baseline(base_cache, records)
            pairs = tuple(map(tuple, result["triangle_pairs"]))
            pair_set = set(pairs)
            base_set = set(base_pairs)
            added = sorted(pair_set-base_set)
            removed = sorted(base_set-pair_set)
            calculated_changed = {r[3] for r0,r in zip(base_records, records) if r0 != r}
            if calculated_changed != set(map(int, changed_faces)):
                raise AssertionError("incident-face scope differs from exact-record changed-face set")
            examples = []
            for pair in (added[:24] + removed[:24]):
                src_records = records if pair in added else base_records
                examples.append(pair_detail(pair, src_records))
            item.update({
                "status": "EXACT_COMPLETE_SURFACE_AUDIT_PASS" if not added else "EXACT_SELF_CROSSINGS_ADDED",
                "exact_complete_surface_audit": bool(result.get("complete_surface_audit")),
                "exact_self_pair_count_vs_013v2": len(pairs),
                "baseline_exact_self_pair_count": len(base_pairs),
                "added_exact_self_pairs": [list(p) for p in added],
                "removed_exact_self_pairs": [list(p) for p in removed],
                "added_exact_self_pair_count": len(added),
                "removed_exact_self_pair_count": len(removed),
                "candidate_pair_set_sha256": hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest(),
                "reused_unchanged_crossing_pairs": int(result.get("reused_unchanged_crossing_pairs", 0)),
                "unchanged_face_count": int(result.get("unchanged_face_count", 0)),
                "owner_reaudited_face_count": int(result.get("selected_face_count", 0)),
                "owner_aabb_candidate_pairs_reaudited": int(result.get("aabb_candidate_pairs", 0)),
                "owner_allowed_shared_vertex_or_edge_pairs_reaudited": int(result.get("allowed_shared_vertex_or_edge_pairs", 0)),
                "deterministic_exact_witness_examples_first24_added_then_removed": examples,
                "exact_owner_result_fields": {k:v for k,v in result.items() if k != "triangle_pairs"},
                "elapsed_seconds": time.monotonic()-t_start,
            })
    except Exception as exc:
        item["status"] = "AUDIT_ERROR"
        item["audit_error"] = repr(exc)
        item["traceback"] = traceback.format_exc()
        item["elapsed_seconds"] = time.monotonic()-t_start
    report["candidates"][name] = item
    report["status"] = "PARTIAL" if item["status"] in ("AUDIT_ERROR","FAIL_CANDIDATE_HAS_DEGENERATE_FLOAT32_FACES") else "RUNNING"
    atomic_json(OUT / "report.json", report)
    print(json.dumps({"phase":"candidate_complete","name":name,"status":item["status"],
                      "changed_faces":item.get("changed_incident_face_count"),
                      "baseline_pairs":item.get("baseline_exact_self_pair_count"),
                      "candidate_pairs":item.get("exact_self_pair_count_vs_013v2"),
                      "added":item.get("added_exact_self_pair_count"),
                      "removed":item.get("removed_exact_self_pair_count"),
                      "elapsed_s":item.get("elapsed_seconds")}), flush=True)

report["status"] = "COMPLETE_ALL_CANDIDATE_AUDITS" if all(
    x.get("status") in ("EXACT_COMPLETE_SURFACE_AUDIT_PASS","EXACT_SELF_CROSSINGS_ADDED")
    for x in report["candidates"].values()
) else "COMPLETE_WITH_CANDIDATE_AUDIT_FAILURE"
report["total_elapsed_seconds"] = time.monotonic()-started
atomic_json(OUT / "report.json", report)
print(json.dumps({"phase":"complete","status":report["status"],"total_elapsed_seconds":report["total_elapsed_seconds"]}), flush=True)
