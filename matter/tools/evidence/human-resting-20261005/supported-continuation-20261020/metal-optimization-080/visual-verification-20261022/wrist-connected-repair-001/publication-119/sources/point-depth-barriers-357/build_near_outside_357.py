#!/usr/bin/env python3
"""Build extra baseline point-depth rows in the frozen 352 immediate-Q scope."""
from __future__ import annotations
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/Users/n/Work/numi-human-connected-surface-069")
TASK = ROOT / "D/point-depth-barriers-357"
BASE_REPORT = TASK / "actual-491-v2/report.json"
BASE_CANDIDATE = Path("/Users/n/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/joint-boundary-continuation-322/round-003/attempt-004/global-275/backtrack/pass-002/candidate.npz")
SUPPORT_CANDIDATE = Path("/Users/n/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/coherent-support-continuation-352/round-001/attempt-001/proposal/candidate.npz")
SUPPORT_CONFIG = Path("/Users/n/Work/numi-human-connected-surface-069/barycentric-connected-clearance-170/coherent-support-continuation-352/round-001/attempt-001/global-275/backtrack-config.json")
BASIS_OWNER = ROOT / "D/connected-local-witness-proposal-354/propose_joint_clearance_witnesses_354.py"
BODY_CSV = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight/exact-emitted.bodies.csv")
THRESHOLD_M = 0.0011
OUTDIR = TASK / "actual-near-outside-001"

sys.path.insert(0, str(TASK))
sys.path.insert(0, str(BASIS_OWNER.parent))
from point_depth_barriers_357 import build_point_depth_rows
import propose_joint_clearance_witnesses_354 as basis_owner

def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def sha_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()

def main() -> None:
    if OUTDIR.exists():
        raise FileExistsError(OUTDIR)
    base_report = json.loads(BASE_REPORT.read_text())
    with np.load(BASE_CANDIDATE, allow_pickle=False) as candidate:
        points = candidate["points_world_m"]
        if points.dtype != np.float32 or points.shape != (20889, 3):
            raise ValueError("baseline candidate must be exact F32 (20889,3)")
        support_data = json.loads(SUPPORT_CONFIG.read_text())
        support = np.asarray(support_data["fit_support_rows"], dtype=np.int64)
        if len(support) != 6418 or len(np.unique(support)) != len(support):
            raise ValueError("352 LP support does not match frozen 6418-row set")
        anchor_rows = np.asarray(candidate["typed_coarse_anchor_output_rows_018"])
        if anchor_rows.dtype.kind not in "iu" or anchor_rows.shape != (237,):
            raise ValueError("expected 237 typed output anchors")
        B, control_rows, basis_receipt = basis_owner.build_immediate_parent_basis(candidate, support, anchor_rows)
        scope_rows = np.asarray(basis_receipt["output_scope_rows"], dtype=np.int64)
        if B.shape != (6433, 3971) or len(scope_rows) != 6433 or len(np.unique(scope_rows)) != 6433:
            raise ValueError("352 immediate-Q scope changed from verified 6433 rows")
        if len(np.intersect1d(scope_rows, anchor_rows)) != 0:
            raise ValueError("expanded scope overlaps exact typed anchors")
        # The 352 proposal and 322 baseline must carry the identical immediate-Q map and anchors.
        with np.load(SUPPORT_CANDIDATE, allow_pickle=False) as support_candidate:
            q_keys = ("immediate_refinement_prolongation_indptr", "immediate_refinement_prolongation_node_rows", "immediate_refinement_prolongation_weights", "typed_coarse_anchor_output_rows_018")
            for key in q_keys:
                if candidate[key].dtype != support_candidate[key].dtype or candidate[key].shape != support_candidate[key].shape or candidate[key].tobytes() != support_candidate[key].tobytes():
                    raise ValueError(f"352 Q/anchor source differs from baseline candidate: {key}")
    scope_set = set(map(int, scope_rows))
    support_set = set(map(int, support))
    pairs_csv = Path(base_report["query_sources"][0]["pairs_csv"])
    selected = []
    support_near_count = 0
    min_distance = float("inf")
    max_distance = 0.0
    with pairs_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["classification"] != "outside":
                continue
            distance = float(row["nearest_distance_m"])
            if not np.isfinite(distance) or distance < 0:
                raise ValueError("nonfinite/negative local nearest distance")
            if distance > THRESHOLD_M:
                continue
            query_row = int(row["query_row"])
            if query_row in support_set:
                support_near_count += 1
            if query_row not in scope_set:
                continue
            selected.append([query_row, int(row["bank_body"]), int(row["bank_stable_id"]), int(row["nearest_triangle"])])
            min_distance = min(min_distance, distance)
            max_distance = max(max_distance, distance)
    if support_near_count != 2975 or len(selected) != 2980:
        raise ValueError(f"near-outside counts changed: support={support_near_count}, expanded={len(selected)}")
    if len({tuple(x) for x in selected}) != len(selected):
        raise ValueError("duplicate selected outside typed IDs")

    source_specs = []
    inside_source = next(q for q in base_report["query_sources"] if q["kind"] == "inside")
    outside_source = next(q for q in base_report["query_sources"] if q["kind"] == "outside")
    for q, kind in ((inside_source, "inside"), (outside_source, "outside")):
        spec = {k: q[k] for k in ("kind", "points_csv", "pairs_csv", "nodes_csv", "query_receipt_json")}
        spec["require_points_match"] = True
        if kind == "inside":
            spec["expected_selected_count"] = 489
        else:
            spec["selected_typed_pairs"] = selected
            spec["expected_selected_count"] = 2980
        source_specs.append(spec)
    node_rows, normals, lower, typed, records, report = build_point_depth_rows(
        points, source_specs, BODY_CSV, expected_point_count=20889, expected_bank_count=185,
        baseline_feasibility_tolerance_m=1e-10, reconstruction_tolerance_m=2e-8)
    if len(records) != 3469 or report["inside_row_count"] != 489 or report["outside_row_count"] != 2980:
        raise ValueError("helper-selected row counts do not match preflight")
    if report["maximum_zero_step_violation_m"] != 0.0:
        raise ValueError("exact baseline zero-step gate failed")
    if not all(q["complete_bank_accounting_verified"] and q["bank_count"] == 185 and q["query_rows"] == 20889 and q["ambiguous_rows"] == 0 and q["invalid_rows"] == 0 for q in report["query_sources"]):
        raise ValueError("query sources lack complete 185-bank accounting")
    OUTDIR.mkdir(parents=True, exist_ok=False)
    rows_path = OUTDIR / "point-depth-rows.npz"
    np.savez_compressed(rows_path,
        node_rows=node_rows, normals_world_unit_covector=normals, lower_delta_m=lower, typed_ids=typed,
        base_lp_support_rows=support, expanded_q_scope_rows=scope_rows, q_control_parent_rows=control_rows)
    source_bytes = lambda path: sha_file(path)
    basis_keys = ("immediate_refinement_prolongation_indptr", "immediate_refinement_prolongation_node_rows", "immediate_refinement_prolongation_weights")
    with np.load(BASE_CANDIDATE, allow_pickle=False) as candidate:
        q_hashes = {k: sha_array(candidate[k]) for k in basis_keys}
        q_shapes = {k: list(candidate[k].shape) for k in basis_keys}
        q_dtypes = {k: str(candidate[k].dtype) for k in basis_keys}
        q_values_identical_to_352 = True
        anchor_hash = sha_array(candidate["typed_coarse_anchor_output_rows_018"])
    report.update({
        "candidate_npz": str(BASE_CANDIDATE), "candidate_sha256": source_bytes(BASE_CANDIDATE),
        "candidate_points_sha256": sha_array(points), "rows_npz": str(rows_path), "rows_npz_sha256": source_bytes(rows_path),
        "builder_path": str(TASK / "point_depth_barriers_357.py"), "builder_sha256": source_bytes(TASK / "point_depth_barriers_357.py"),
        "owner_sha256": source_bytes(TASK / "point_depth_barriers_357.py"),
        "selection_runner_path": str(Path(__file__).resolve()), "selection_runner_sha256": source_bytes(Path(__file__)),
        "selection_policy": {
            "kind": "exact query CSV outside rows within immediate-Q scope",
            "distance_field": "nearest_distance_m (raw body-local Euclidean metric)",
            "distance_threshold_m_inclusive": THRESHOLD_M,
            "outside_rows_within_352_support": support_near_count,
            "outside_rows_within_expanded_q_scope": len(selected),
            "outside_rows_added_by_scope_expansion": len(selected) - support_near_count,
            "selected_outside_min_nearest_distance_m": min_distance,
            "selected_outside_max_nearest_distance_m": max_distance,
            "inside_rows_retained": 489,
            "rows_total": len(records),
        },
        "immediate_q_scope": {
            "basis_owner_path": str(BASIS_OWNER), "basis_owner_sha256": source_bytes(BASIS_OWNER),
            "support_config_path": str(SUPPORT_CONFIG), "support_config_sha256": source_bytes(SUPPORT_CONFIG),
            "support_rows_key": "fit_support_rows", "support_rows_count": len(support), "support_rows_sha256": sha_array(support),
            "anchor_rows_key": "typed_coarse_anchor_output_rows_018", "anchor_rows_count": len(anchor_rows), "anchor_rows_sha256": anchor_hash,
            "scope_rows_count": len(scope_rows), "scope_rows_sha256": sha_array(scope_rows),
            "control_parent_rows_count": len(control_rows), "control_parent_rows_sha256": sha_array(control_rows),
            "basis_shape": list(B.shape), "basis_nnz": int(B.nnz), "q_csr_array_sha256": q_hashes,
            "q_csr_array_shapes": q_shapes, "q_csr_array_dtypes": q_dtypes,
            "q_and_anchor_bytes_match_352_proposal": q_values_identical_to_352,
            "source_candidate_352_proposal": str(SUPPORT_CANDIDATE), "source_candidate_352_sha256": source_bytes(SUPPORT_CANDIDATE),
            "source_candidate_baseline_322": str(BASE_CANDIDATE),
        },
        "complete_bank_accounting_verified": True, "complete_bank_accounting_rows": 20889,
        "scope_limit": "Rows outside expanded immediate-Q scope are excluded from the near-outside additions; no support is expanded by this output.",
        "physical_steps": 0, "geometry_admitted": False,
    })
    report_path = OUTDIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "outdir": str(OUTDIR), "rows": len(records), "inside": report["inside_row_count"], "outside": report["outside_row_count"],
        "unique_nodes": report["unique_node_count"], "expanded_scope_rows": len(scope_rows), "support_rows": len(support),
        "inside_max_zero_step_violation_m": report["maximum_zero_step_violation_m"],
        "candidate_sha256": report["candidate_sha256"], "candidate_points_sha256": report["candidate_points_sha256"],
        "rows_npz_sha256": report["rows_npz_sha256"], "helper_sha256": report["builder_sha256"],
        "runner_sha256": report["selection_runner_sha256"], "report_sha256": source_bytes(report_path),
    }, indent=2))

if __name__ == "__main__":
    main()
