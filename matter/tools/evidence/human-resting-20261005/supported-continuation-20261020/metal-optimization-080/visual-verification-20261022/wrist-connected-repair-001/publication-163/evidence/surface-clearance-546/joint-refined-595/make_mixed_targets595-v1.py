#!/usr/bin/env python3
"""Bind the previously certified 591 per-row exits as ordinary 408 point targets.

This adapter does not compute exits or certify new geometry. It selects the exact
per-row records from the pinned 589 source-outward/depth-maximizing reports and
requires byte-identical endpoints in the previously checked 591 coordinate target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

EXPECTED_ORIGINS = [
    "depth-maximizing", "depth-maximizing", "depth-maximizing", "depth-maximizing",
    "source-outward", "depth-maximizing", "depth-maximizing", "source-outward",
]
ARRAY_KEYS = (
    "query_rows", "target_world_f32_m", "proposal_direction_world_unit",
    "source_surface_normal_world_unit", "depth_covector_world_unit",
    "direction_compatibility_mask", "source_normal_available_mask",
    "pass_mask", "target_distance_m", "target_exactly_outside_all185",
)

def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_json(path: Path | str) -> dict:
    obj = json.loads(Path(path).read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"expected object JSON: {path}")
    return obj

def bytes_equal(a: np.ndarray, b: np.ndarray) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes()

def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]).copy() for k in z.files}

def build_mixed_targets(*, base_candidate: Path, point_summary: Path,
                        mixed_candidate: Path, mixed_report_path: Path,
                        source_npz_path: Path, source_report_path: Path,
                        depth_npz_path: Path, depth_report_path: Path,
                        out_dir: Path) -> tuple[Path, Path, dict]:
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    base_sha, summary_sha = sha(base_candidate), sha(point_summary)
    source_sha, depth_sha = sha(source_npz_path), sha(depth_npz_path)
    mixed_sha, mixed_report_sha = sha(mixed_candidate), sha(mixed_report_path)
    source_report, depth_report = load_json(source_report_path), load_json(depth_report_path)
    mixed_report = load_json(mixed_report_path)
    for label, rep in (("source", source_report), ("depth", depth_report)):
        if rep.get("inputs", {}).get("candidate_sha256") != base_sha:
            raise ValueError(f"{label} target report is for another candidate")
        if rep.get("inputs", {}).get("point_summary_sha256") != summary_sha:
            raise ValueError(f"{label} target report is for another point summary")
        if rep.get("status") != "PASS_FIRST_UNION_EXIT_TARGETS_ONLY" or rep.get("physical_steps") != 0:
            raise ValueError(f"{label} target source is not a proposal-only exact exit receipt")
    if source_report.get("direction_mode") != "source-outward":
        raise ValueError("source parent is not source-outward")
    if depth_report.get("direction_mode") != "depth-maximizing":
        raise ValueError("depth parent is not depth-maximizing")
    if mixed_report.get("base_candidate_sha256") != base_sha:
        raise ValueError("591 target report is not bound to the requested baseline")
    if mixed_report.get("candidate_sha256") != mixed_sha:
        raise ValueError("591 candidate/report hash mismatch")
    if mixed_report.get("586_v2_Q_and_037_refinement_validation") != "PASS":
        raise ValueError("591 candidate lacks the pinned Q and fixed-037 refinement validation")
    if mixed_report.get("individual_targets_all185_certified") is not True:
        raise ValueError("591 report does not certify all selected endpoints")
    if mixed_report.get("native_admission") is not False or mixed_report.get("physical_steps") != 0:
        raise ValueError("591 input must remain a zero-step unadmitted coordinate target")
    origins = mixed_report.get("origins")
    rows = [int(v) for v in mixed_report.get("changed_point_rows", [])]
    details = mixed_report.get("rows")
    if origins != EXPECTED_ORIGINS or len(rows) != 8 or len(set(rows)) != 8:
        raise ValueError("591 per-row origin/identity mapping differs from the reviewed pilot")
    if not isinstance(details, list) or [int(x.get("row", -1)) for x in details] != rows:
        raise ValueError("591 detailed row identities disagree")
    base = _load_npz(base_candidate)
    mixed = _load_npz(mixed_candidate)
    source, depth = _load_npz(source_npz_path), _load_npz(depth_npz_path)
    base_points = np.asarray(base.get("points_world_m"))
    mixed_points = np.asarray(mixed.get("points_world_m"))
    if base_points.ndim != 2 or base_points.shape[1] != 3 or mixed_points.shape != base_points.shape:
        raise ValueError("candidate point domains differ")
    if rows != list(range(len(base_points) - len(rows), len(base_points))):
        raise ValueError("591 rows are not exactly the appended midpoint suffix")
    if mixed_report.get("point_domain_count") != len(base_points):
        raise ValueError("591 point-domain count differs")
    if mixed_report.get("all_nonpoint_arrays_preserved") is not True or mixed_report.get("anchors_unchanged") is not True:
        raise ValueError("591 report lacks its nonpoint/anchor preservation checks")
    if not np.array_equal(source["query_rows"], rows) or not np.array_equal(depth["query_rows"], rows):
        raise ValueError("parent target rows do not match the 591 suffix")
    if not all(np.array_equal(source[k], depth[k]) for k in ("source_surface_normal_world_unit", "depth_covector_world_unit")):
        raise ValueError("589 source/depth parent normal domains disagree")
    by_record = {
        "source-outward": {int(x["query_row"]): x for x in source_report.get("targets", [])},
        "depth-maximizing": {int(x["query_row"]): x for x in depth_report.get("targets", [])},
    }
    chosen_arrays = {k: [] for k in ARRAY_KEYS}
    selected_records, row_pins = [], []
    anchors = np.asarray(base.get("typed_coarse_anchor_output_rows_018"), dtype=np.int64)
    if np.intersect1d(rows, anchors).size:
        raise ValueError("591 point-target suffix includes typed anchors")
    for i, (row, origin, detail) in enumerate(zip(rows, origins, details)):
        parent_arrays = source if origin == "source-outward" else depth
        parent_report = source_report if origin == "source-outward" else depth_report
        parent_npz_path = source_npz_path if origin == "source-outward" else depth_npz_path
        parent_report_path = source_report_path if origin == "source-outward" else depth_report_path
        record = by_record[origin].get(row)
        if record is None or record.get("status") != "PASS_FIRST_UNION_EXIT" or record.get("end_outside_all185_by_event_parity") is not True or record.get("target_exactly_outside_all185") is not True:
            raise ValueError(f"parent ray record for row {row} is missing or not all185-certified")
        if detail.get("all185_endpoint_certified") is not True:
            raise ValueError(f"591 selected row {row} is not all185-certified")
        if float(detail.get("actual_displacement_dot_source_outward", 0.0)) <= 0.0 or float(detail.get("actual_displacement_dot_signed_current_outward", 0.0)) <= 0.0:
            raise ValueError(f"591 selected row {row} is not source/current outward")
        start = np.asarray(record["start_world_f32_m"], dtype=np.float32)
        target = np.asarray(parent_arrays["target_world_f32_m"][i], dtype=np.float32)
        candidate_target = np.asarray(mixed_points[row], dtype=np.float32)
        if not bytes_equal(start, np.asarray(base_points[row], dtype=np.float32)):
            raise ValueError(f"row {row} parent ray starts at different candidate coordinates")
        if not bytes_equal(target, candidate_target) or not bytes_equal(target, np.asarray(record["target_world_f32_m"], dtype=np.float32)):
            raise ValueError(f"row {row} selected endpoint differs from the exact 591 coordinate target")
        direction = np.asarray(parent_arrays["proposal_direction_world_unit"][i], dtype=np.float64)
        if not np.allclose(direction, np.asarray(record["proposal_direction_world_unit"], dtype=np.float64), rtol=0, atol=1e-14):
            raise ValueError(f"row {row} parent direction differs from exact ray receipt")
        for key in ARRAY_KEYS:
            chosen_arrays[key].append(parent_arrays[key][i])
        selected_records.append({**record, "mixed_selection_origin": origin,
            "mixed_selection_report_sha256": mixed_report_sha, "parent_target_report_sha256": sha(parent_report_path),
            "parent_target_npz_sha256": sha(parent_npz_path)})
        row_pins.append({
            "query_row": row, "origin": origin,
            "parent_report_sha256": sha(parent_report_path), "parent_npz_sha256": sha(parent_npz_path),
            "actual_displacement_dot_source_outward": float(detail["actual_displacement_dot_source_outward"]),
            "actual_displacement_dot_signed_current_outward": float(detail["actual_displacement_dot_signed_current_outward"]),
            "all185_endpoint_certified": True,
        })
    arrays = {k: np.asarray(v) for k, v in chosen_arrays.items()}
    arrays["query_rows"] = np.asarray(rows, dtype=np.int64)
    arrays["target_world_f32_m"] = np.asarray(arrays["target_world_f32_m"], dtype=np.float32)
    arrays["proposal_direction_world_unit"] = np.asarray(arrays["proposal_direction_world_unit"], dtype=np.float64)
    npz_out = out_dir / "nearest-union-exits.npz"
    report_out = out_dir / "nearest-union-exits.json"
    np.savez_compressed(npz_out, **arrays)
    report = dict(depth_report)
    report.update({
        "schema": "numi.human.point-first-union-exits.per-row-certified-mixed-591.v1",
        "classification": "per-row endpoints selected from pinned 589 all185 ray receipts; proposal targets only, zero physical steps",
        "direction_mode": "per-row-certified-source-or-depth-selection",
        "inputs": {
            "candidate_sha256": base_sha, "point_summary_sha256": summary_sha,
            "source_parent_report_sha256": sha(source_report_path), "source_parent_npz_sha256": source_sha,
            "depth_parent_report_sha256": depth_sha, "depth_parent_npz_sha256": depth_sha,
            "mixed_selection_candidate_sha256": mixed_sha, "mixed_selection_report_sha256": mixed_report_sha,
        },
        "candidate_points": len(base_points),
        "candidate_faces": int(np.asarray(base.get("source_O_faces")).shape[0]),
        "query_rows": len(rows), "inside_query_nodes": rows,
        "inside_node_bank_pairs": len(rows), "all_bank_ray_results": len(rows),
        "passing_targets": len(rows), "physical_steps": 0,
        "status": "PASS_FIRST_UNION_EXIT_TARGETS_ONLY",
        "targets": selected_records,
        "outputs": {
            "npz": str(npz_out), "npz_sha256": sha(npz_out),
            "parent_prefix_node_count": rows[0],
            "point_domain_count": len(base_points),
        },
        "per_row_selected_parent_receipts": row_pins,
        "source_depth_mixed_selection": {
            "candidate": str(mixed_candidate), "candidate_sha256": mixed_sha,
            "report": str(mixed_report_path), "report_sha256": mixed_report_sha,
            "origins": origins, "native_admission": False, "physical_steps": 0,
        },
        "native_admission": False,
    })
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    summary = {"status": "PASS_MIXED_591_TARGETS_REPACKED", "candidate_sha256": base_sha,
        "query_rows": rows, "origins": origins, "npz": str(npz_out), "npz_sha256": sha(npz_out),
        "report": str(report_out), "report_sha256": sha(report_out), "physical_steps": 0}
    return npz_out, report_out, summary

def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("base-candidate", "point-summary", "mixed-candidate", "mixed-report",
                 "source-npz", "source-report", "depth-npz", "depth-report", "out"):
        ap.add_argument("--" + name, required=True, type=Path)
    a = ap.parse_args()
    npz, report, result = build_mixed_targets(
        base_candidate=a.base_candidate, point_summary=a.point_summary,
        mixed_candidate=a.mixed_candidate, mixed_report_path=a.mixed_report,
        source_npz_path=a.source_npz, source_report_path=a.source_report,
        depth_npz_path=a.depth_npz, depth_report_path=a.depth_report, out_dir=a.out)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()

