#!/usr/bin/env python3
"""Emit a zero-row first-union target artifact from a complete exact point query."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

SCHEMA = "numi.human.point-first-union-exits.run034-current.v1"
BANK_COUNT = 185

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def _read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value

def _read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))

def validate_zero_point_audit(candidate_path: Path, summary_path: Path, query_dir: Path) -> tuple[np.ndarray, dict]:
    candidate_path = Path(candidate_path)
    query_dir = Path(query_dir)
    summary_path = Path(summary_path)
    candidate_sha = sha(candidate_path)
    with np.load(candidate_path, allow_pickle=False) as data:
        if "points_world_m" not in data.files or "source_O_faces" not in data.files:
            raise ValueError("candidate lacks the point-query geometry fields")
        points = np.asarray(data["points_world_m"], dtype=np.float32)
        faces = np.asarray(data["source_O_faces"], dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("candidate points are malformed")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("candidate faces are malformed")
    summary = _read_json(summary_path)
    if summary.get("candidate_sha256") != candidate_sha:
        raise ValueError("point query summary candidate hash mismatch")
    if summary.get("complete_surface_audit") is not True or summary.get("exact_PLC_surface_pairs") != 0:
        raise ValueError("zero-target emission requires complete zero-self audit")
    if summary.get("geometry_admitted") is not False or summary.get("accepted_physical_steps") != 0:
        raise ValueError("query summary is not construction-only")
    if summary.get("points") != len(points) or summary.get("faces") != len(faces):
        raise ValueError("query summary geometry counts differ from candidate")
    for field in ("inside_nodes", "inside_node_bank_pairs"):
        if type(summary.get(field)) is not int or summary[field] != 0:
            raise ValueError(f"zero-target query requires {field}=0")
    if type(summary.get("query_pair_count")) is not int or summary["query_pair_count"] < 0:
        raise ValueError("point query pair count is missing or invalid")
    if summary.get("inside_witnesses") != []:
        raise ValueError("zero-target query has inside witnesses")
    query_files = {name: query_dir / f"{name}.csv" for name in ("points", "pairs", "nodes")}
    receipt_path = query_dir / "query-receipt.json"
    if not all(path.is_file() for path in (*query_files.values(), receipt_path)):
        raise ValueError("complete point-query CSVs and receipt are required")
    receipt = _read_json(receipt_path)
    if receipt.get("query_rows") != len(points) or receipt.get("identity_domain") != "mechanics_node_row":
        raise ValueError("query receipt row domain/count mismatch")
    if receipt.get("physical_steps") != 0:
        raise ValueError("query receipt records a physical step")
    for name, path in query_files.items():
        if receipt.get(f"{name}_sha256") != sha(path):
            raise ValueError(f"query receipt hash mismatch for {name}.csv")
    point_rows = _read_csv(query_files["points"])
    node_rows = _read_csv(query_files["nodes"])
    pair_rows = _read_csv(query_files["pairs"])
    if len(point_rows) != len(points) or len(node_rows) != len(points):
        raise ValueError("point or node query row count differs from candidate")
    required_bits = ("x_bits_u32", "y_bits_u32", "z_bits_u32")
    point_bits = points.view(np.uint32)
    for row_id, (point_row, node_row) in enumerate(zip(point_rows, node_rows)):
        if int(point_row["query_row"]) != row_id or int(node_row["query_row"]) != row_id:
            raise ValueError("query row identities are not dense and ordered")
        if any(int(point_row[k]) != int(v) for k, v in zip(required_bits, point_bits[row_id])):
            raise ValueError(f"query point bits differ from candidate row {row_id}")
        counts = {k: int(node_row[k]) for k in (
            "total_bone_banks", "aabb_proven_outside_banks", "parity_outside_banks",
            "inside_banks", "ambiguous_banks", "invalid_banks")}
        if counts["total_bone_banks"] != BANK_COUNT:
            raise ValueError(f"point row {row_id} does not account for all 185 banks")
        if counts["inside_banks"] or counts["ambiguous_banks"] or counts["invalid_banks"]:
            raise ValueError(f"point row {row_id} is not strictly outside all banks")
        if sum(counts[k] for k in ("aabb_proven_outside_banks", "parity_outside_banks",
                                    "inside_banks", "ambiguous_banks", "invalid_banks")) != BANK_COUNT:
            raise ValueError(f"point row {row_id} has incomplete bank accounting")
    if any(row.get("classification") != "outside" or int(row.get("ambiguous", "1")) != 0
           for row in pair_rows):
        raise ValueError("pair CSV contains a non-outside or ambiguous result")
    if summary.get("query_pair_count") != len(pair_rows):
        raise ValueError("query pair CSV count differs from summary")
    if any(row.get("classification") == "inside" for row in pair_rows):
        raise ValueError("pair CSV contains an inside result despite zero inside summary")
    return points, {
        "candidate_sha256": candidate_sha,
        "summary_sha256": sha(summary_path),
        "query_points_sha256": sha(query_files["points"]),
        "query_pairs_sha256": sha(query_files["pairs"]),
        "query_nodes_sha256": sha(query_files["nodes"]),
        "query_receipt_sha256": sha(receipt_path),
        "point_count": len(points),
        "face_count": len(faces),
    }

def emit_empty_point_targets(candidate_path: Path, summary_path: Path, query_dir: Path,
                             npz_path: Path, report_path: Path) -> dict:
    npz_path, report_path = Path(npz_path), Path(report_path)
    if npz_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite zero-target artifact")
    _, pins = validate_zero_point_audit(candidate_path, summary_path, query_dir)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "query_rows": np.empty((0,), dtype=np.int64),
        "target_world_f32_m": np.empty((0, 3), dtype=np.float32),
        "proposal_direction_world_unit": np.empty((0, 3), dtype=np.float64),
        "source_surface_normal_world_unit": np.empty((0, 3), dtype=np.float64),
        "source_normal_available_mask": np.empty((0,), dtype=np.bool_),
        "pass_mask": np.empty((0,), dtype=np.bool_),
    }
    with npz_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    report = {
        "schema": SCHEMA,
        "classification": "construction_target_proposal_not_clearance_admission",
        "inputs": {"candidate_sha256": pins["candidate_sha256"],
                   "query_points_sha256": pins["query_points_sha256"],
                   "query_nodes_sha256": pins["query_nodes_sha256"],
                   "query_pairs_sha256": pins["query_pairs_sha256"],
                   "point_summary_sha256": pins["summary_sha256"],
                   "query_receipt_sha256": pins["query_receipt_sha256"],
                   "candidate_points": pins["point_count"],
                   "candidate_faces": pins["face_count"],
                   "bank_count": BANK_COUNT},
        "query_rows": 0, "inside_query_nodes": 0, "inside_node_bank_pairs": 0,
        "inherited_prefix_inside_nodes": 0, "new_refinement_inside_nodes": 0,
        "source_normal_unavailable_rows": [], "source_normal_errors": {},
        "all_bank_ray_results": 0, "passing_targets": 0,
        "status": "PASS_FIRST_UNION_EXIT_TARGETS_ONLY",
        "normal_method": "No point targets were required: the matched complete exact query classified every candidate row outside all 185 banks; no rays were constructed.",
        "targets": [], "construction_cap_m": 0.012, "physical_steps": 0,
        "zero_target_basis": "complete same-candidate exact 185-bank point query with zero inside, ambiguous, invalid, and self-intersection results",
        "zero_targets_emitted": True,
    }
    report["outputs"] = {"npz_sha256": sha(npz_path)}
    with report_path.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return {"npz": str(npz_path), "npz_sha256": sha(npz_path),
            "report": str(report_path), "report_sha256": sha(report_path),
            "targets": 0, "candidate_sha256": pins["candidate_sha256"]}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(emit_empty_point_targets(args.candidate, args.summary, args.query_dir,
                                              args.npz, args.report), indent=2))

if __name__ == "__main__":
    main()
