#!/usr/bin/env python3
"""Build candidate-matched 357 point-depth rows from complete pinned queries.

No geometry is regenerated. This consumes saved candidate, 130 summary, complete
159 report, and matching 130 query CSVs, then writes the established NPZ/raw
report/401-compatible report/validation receipt bundle.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

D = Path("/Users/numi/Work/numi-human-connected-surface-069")
DEFAULT_OWNER = D / "D/point-depth-barriers-357/point_depth_barriers_357.py"
DEFAULT_VALIDATOR = D / "D/retained-gain-continuation-401/propose_joint_clearance_witnesses_401.py"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def require_hash(path: Path, expected: str, label: str) -> str:
    actual = sha(path)
    if len(expected) != 64 or actual.lower() != expected.lower():
        raise ValueError(f"{label} SHA mismatch: expected {expected}, got {actual}")
    return actual


def validate_point_summary(summary: dict[str, Any], *, candidate_sha: str,
                           point_count: int, face_count: int) -> None:
    if (summary.get("candidate_sha256") != candidate_sha
            or int(summary.get("points", -1)) != point_count
            or int(summary.get("faces", -1)) != face_count
            or summary.get("complete_surface_audit") is not True
            or int(summary.get("exact_PLC_surface_pairs", -1)) != 0
            or summary.get("geometry_admitted") is not False
            or int(summary.get("accepted_physical_steps", -1)) != 0):
        raise ValueError("130 summary is incomplete, admitted, or mismatched to candidate")


def validate_surface_report(report: dict[str, Any], *, candidate_sha: str,
                            point_count: int, face_count: int,
                            expected_bank_count: int = 185) -> list[tuple[int, int]]:
    if (report.get("candidate_sha256") != candidate_sha
            or report.get("status") != "COMPLETE_TYPED_BANK_SURFACE_CROSSING_AUDIT"
            or report.get("complete") is not True
            or int(report.get("bank_count", -1)) != expected_bank_count
            or int(report.get("expected_bank_count", -1)) != expected_bank_count
            or int(report.get("surface_point_count", -1)) != point_count
            or int(report.get("surface_face_count", -1)) != face_count):
        raise ValueError("159 report is incomplete or mismatched to candidate")
    banks = report.get("banks")
    keys = [(int(b["body"]), int(b["stable_bone_id"])) for b in banks or []]
    if len(keys) != expected_bank_count or len(set(keys)) != expected_bank_count:
        raise ValueError(f"159 report must contain {expected_bank_count} unique typed banks")
    return keys


def validate_query_receipt(receipt: dict[str, Any], *, point_count: int,
                           query_dir: Path, expected_bank_count: int = 185) -> tuple[Path, dict[str, Path]]:
    if (int(receipt.get("query_rows", -1)) != point_count
            or int(receipt.get("physical_steps", -1)) != 0
            or receipt.get("identity_domain") != "mechanics_node_row"):
        raise ValueError("130 query receipt identity domain, point count, or step count mismatch")
    argv = receipt.get("argv")
    if not isinstance(argv, list) or len(argv) < 7:
        raise ValueError("130 query receipt lacks exact argv provenance")
    if int(argv[6]) != expected_bank_count:
        raise ValueError("130 query receipt does not pin all expected banks")
    qdir = Path(query_dir).resolve()
    files = {
        "points": qdir / "points.csv",
        "pairs": qdir / "pairs.csv",
        "nodes": qdir / "nodes.csv",
        "receipt": qdir / "query-receipt.json",
    }
    for name, p in files.items():
        if not p.is_file():
            raise FileNotFoundError(p)
    expected = {
        "points": receipt.get("points_sha256"),
        "pairs": receipt.get("pairs_sha256"),
        "nodes": receipt.get("nodes_sha256"),
    }
    for name, p in files.items():
        if name in expected and sha(p) != expected[name]:
            raise ValueError(f"query receipt hash mismatch for {p.name}")
    return Path(argv[2]).resolve(), files


def validate_node_accounting(nodes_path: Path, *, point_count: int,
                             bank_count: int = 185) -> list[dict[str, str]]:
    """Validate the 130 per-node accounting, including AABB-pruned banks."""
    rows = read_rows(nodes_path)
    if len(rows) != point_count:
        raise ValueError(f"nodes CSV row count {len(rows)} != point count {point_count}")
    required = ("query_row", "total_bone_banks", "aabb_proven_outside_banks",
                "parity_outside_banks", "inside_banks", "ambiguous_banks",
                "invalid_banks", "parity_query_banks")
    for i, row in enumerate(rows):
        if any(key not in row for key in required):
            raise ValueError("nodes CSV lacks complete-bank accounting columns")
        try:
            values = {key: int(row[key]) for key in required}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"node accounting row {i} has noninteger counts") from exc
        if values["query_row"] != i:
            raise ValueError("nodes CSV query rows are not contiguous")
        if any(values[key] < 0 for key in required[1:]):
            raise ValueError(f"node accounting row {i} has negative counts")
        accounted = sum(values[key] for key in (
            "aabb_proven_outside_banks", "parity_outside_banks",
            "inside_banks", "ambiguous_banks", "invalid_banks"))
        if (values["total_bone_banks"] != bank_count or accounted != bank_count
                or values["ambiguous_banks"] != 0 or values["invalid_banks"] != 0
                or values["parity_query_banks"] !=
                   values["parity_outside_banks"] + values["inside_banks"]
                or values["parity_query_banks"] > bank_count):
            raise ValueError(f"query row {i} lacks complete valid bank accounting")
    return rows


def validate_sparse_pair_rows(pair_rows: list[dict[str, str]], *,
                              point_count: int,
                              bank_keys: list[tuple[int, int]]) -> None:
    """Validate present narrow-phase rows; completeness comes from nodes.csv."""
    bank_set = set(bank_keys)
    if len(pair_rows) > point_count * len(bank_set):
        raise ValueError("query pair CSV has more rows than the dense domain")
    seen: set[tuple[int, int, int]] = set()
    for row in pair_rows:
        try:
            query_row = int(row["query_row"])
            body = int(row["bank_body"])
            stable_id = int(row["bank_stable_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("query pair CSV has invalid typed row identity") from exc
        key = (query_row, body, stable_id)
        if not 0 <= query_row < point_count or (body, stable_id) not in bank_set:
            raise ValueError(f"query pair row has out-of-domain identity {key}")
        if key in seen:
            raise ValueError(f"duplicate query pair identity {key}")
        seen.add(key)


def validate_zero_feasible_builder_result(report: dict[str, Any], *,
                                           point_count: int,
                                           bank_count: int = 185) -> None:
    if (not np.isfinite(float(report.get("maximum_zero_step_violation_m", np.nan)))
            or float(report["maximum_zero_step_violation_m"]) != 0.0):
        raise ValueError("point-depth rows are not exactly zero-step feasible")
    sources = report.get("query_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("point-depth rows lack query source receipts")
    # The established 357 owner records coverage on each query-source receipt;
    # it does not emit redundant top-level accounting keys.
    if ("complete_bank_accounting_rows" in report
            or "complete_bank_accounting_verified" in report):
        if (int(report.get("complete_bank_accounting_rows", -1)) != point_count
                or report.get("complete_bank_accounting_verified") is not True):
            raise ValueError("point-depth rows lack complete all-bank accounting")
    for source in sources:
        if (int(source.get("bank_count", -1)) != bank_count
                or int(source.get("query_rows", -1)) != point_count
                or source.get("complete_bank_accounting_verified") is not True
                or int(source.get("complete_bank_accounting_rows", -1)) != point_count
                or int(source.get("ambiguous_rows", -1)) != 0
                or int(source.get("invalid_rows", -1)) != 0):
            raise ValueError("a point-depth query source has incomplete bank coverage")


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import pinned Python owner: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build(args: argparse.Namespace) -> dict[str, Any]:
    candidate = args.candidate.resolve()
    point_summary_path = args.point_summary.resolve()
    surface_report_path = args.surface_report.resolve()
    query_dir = args.query_dir.resolve()
    out = args.output_dir.resolve()
    owner = args.owner.resolve()
    validator_path = args.validator.resolve()
    if out.exists():
        raise FileExistsError(out)
    for p in (candidate, point_summary_path, surface_report_path, query_dir / "query-receipt.json", owner, validator_path):
        if not p.exists():
            raise FileNotFoundError(p)

    actual_candidate_sha = require_hash(candidate, args.expected_candidate_sha256, "candidate")
    point_summary_sha = require_hash(point_summary_path, args.expected_point_summary_sha256, "130 summary")
    surface_sha = require_hash(surface_report_path, args.expected_surface_report_sha256, "159 report")
    receipt_path = query_dir / "query-receipt.json"
    receipt_sha = require_hash(receipt_path, args.expected_query_receipt_sha256, "query receipt")
    if args.expected_owner_sha256:
        owner_sha = require_hash(owner, args.expected_owner_sha256, "357 owner")
    else:
        owner_sha = sha(owner)

    with np.load(candidate, allow_pickle=False) as z:
        if "points_world_m" not in z or "source_O_faces" not in z:
            raise ValueError("candidate lacks points_world_m/source_O_faces")
        points = np.asarray(z["points_world_m"]).copy()
        faces = np.asarray(z["source_O_faces"]).copy()
    if (points.dtype != np.float32 or points.ndim != 2 or points.shape[1] != 3
            or not np.isfinite(points).all() or faces.ndim != 2 or faces.shape[1] != 3
            or len(faces) == 0):
        raise ValueError("candidate points/faces fail exact shape, dtype, or finite checks")
    point_count, face_count = len(points), len(faces)

    point_summary = json.loads(point_summary_path.read_text())
    surface = json.loads(surface_report_path.read_text())
    validate_point_summary(point_summary, candidate_sha=actual_candidate_sha,
                           point_count=point_count, face_count=face_count)
    bank_keys = validate_surface_report(surface, candidate_sha=actual_candidate_sha,
                                        point_count=point_count, face_count=face_count)
    receipt = json.loads(receipt_path.read_text())
    body_csv, query_files = validate_query_receipt(
        receipt, point_count=point_count, query_dir=query_dir,
        expected_bank_count=len(bank_keys))
    if not body_csv.is_file():
        raise FileNotFoundError(body_csv)

    qrows = read_rows(query_files["points"])
    query_points = np.asarray([[float(r["x_m"]), float(r["y_m"]), float(r["z_m"])]
                               for r in qrows], dtype=np.float32)
    if query_points.shape != points.shape or not np.array_equal(query_points, points):
        raise ValueError("query Float32 points differ bitwise from candidate coordinates")
    # 130 writes one node summary per queried mechanics row, while the
    # detailed pair table is intentionally sparse after certified AABB rejects.
    validate_node_accounting(query_files["nodes"], point_count=point_count,
                             bank_count=len(bank_keys))
    pair_rows = read_rows(query_files["pairs"])
    validate_sparse_pair_rows(pair_rows, point_count=point_count, bank_keys=bank_keys)
    inside_count = 0
    selected_outside: list[list[int]] = []
    outside_distances: list[float] = []
    for row in pair_rows:
        classification = row.get("classification")
        if classification == "inside":
            inside_count += 1
            if (int(row["ambiguous"]) != 0 or int(row["inside_rays"]) <= 0
                    or int(row["outside_rays"]) != 0):
                raise ValueError("inside row has ambiguous or contradictory ray evidence")
        elif classification == "outside":
            distance = float(row["nearest_distance_m"])
            if not np.isfinite(distance) or distance < 0:
                raise ValueError("outside row has invalid nearest distance")
            if distance <= args.threshold_m:
                selected_outside.append([
                    int(row["query_row"]), int(row["bank_body"]),
                    int(row["bank_stable_id"]), int(row["nearest_triangle"])])
                outside_distances.append(distance)
        else:
            raise ValueError(f"unexpected query classification {classification!r}")
    if len({tuple(row) for row in selected_outside}) != len(selected_outside):
        raise ValueError("duplicate selected outside typed witness")

    inside_spec = {
        "kind": "inside", "points_csv": str(query_files["points"]),
        "pairs_csv": str(query_files["pairs"]), "nodes_csv": str(query_files["nodes"]),
        "query_receipt_json": str(receipt_path), "require_points_match": True,
        "expected_selected_count": inside_count,
    }
    outside_spec = {
        "kind": "outside", "points_csv": str(query_files["points"]),
        "pairs_csv": str(query_files["pairs"]), "nodes_csv": str(query_files["nodes"]),
        "query_receipt_json": str(receipt_path), "require_points_match": True,
        "selected_typed_pairs": selected_outside,
        "expected_selected_count": len(selected_outside),
    }
    owner_module = import_file("point_depth_rows_owner", owner)
    node_rows, normals, lower, typed, records, row_report = owner_module.build_point_depth_rows(
        points, [inside_spec, outside_spec], body_csv,
        expected_point_count=point_count, expected_bank_count=len(bank_keys),
        baseline_feasibility_tolerance_m=1e-10,
        reconstruction_tolerance_m=2e-8)
    if (int(row_report.get("inside_row_count", -1)) != inside_count
            or int(row_report.get("outside_row_count", -1)) != len(selected_outside)):
        raise ValueError("357 row counts do not match selected query rows")
    validate_zero_feasible_builder_result(row_report, point_count=point_count,
                                          bank_count=len(bank_keys))
    k = len(records)
    if (np.asarray(node_rows).shape != (k,) or np.asarray(normals).shape != (k, 3)
            or np.asarray(lower).shape != (k,) or np.asarray(typed).shape != (k, 4)):
        raise ValueError("357 returned arrays are not aligned")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=False)
    npz_path = out / "point-depth-rows.npz"
    np.savez_compressed(npz_path, node_rows=node_rows,
                        normals_world_unit_covector=normals,
                        lower_delta_m=lower, typed_ids=typed)
    row_report.update({
        "status": "PASS_CANDIDATE_MATCHED_POINT_DEPTH_ROWS",
        "candidate_npz": str(candidate), "candidate_sha256": actual_candidate_sha,
        "candidate_points_sha256": array_sha(points),
        "point_summary_path": str(point_summary_path), "point_summary_sha256": point_summary_sha,
        "surface_report_path": str(surface_report_path), "surface_report_sha256": surface_sha,
        "query_receipt_path": str(receipt_path), "query_receipt_sha256": receipt_sha,
        "query_points_csv_sha256": sha(query_files["points"]),
        "query_pairs_csv_sha256": sha(query_files["pairs"]),
        "query_nodes_csv_sha256": sha(query_files["nodes"]),
        "body_csv_sha256": sha(body_csv), "helper_path": str(owner), "helper_sha256": owner_sha,
        "output_npz": str(npz_path), "output_npz_sha256": sha(npz_path),
        "selection_policy": {
            "inside": "all exact inside typed rows",
            "outside": "exact outside rows with nearest_distance_m <= inclusive threshold",
            "distance_metric": "raw bank-local Euclidean nearest distance",
            "threshold_m": float(args.threshold_m), "inside_rows": inside_count,
            "outside_rows": len(selected_outside),
            "outside_min_m": min(outside_distances) if outside_distances else None,
            "outside_max_m": max(outside_distances) if outside_distances else None,
        },
        "input_pins": {
            "candidate_sha256": actual_candidate_sha,
            "point_summary_sha256": point_summary_sha,
            "surface_report_sha256": surface_sha,
            "query_receipt_sha256": receipt_sha,
            "owner_sha256": owner_sha,
        },
        "topology": [point_count, face_count], "typed_banks": len(bank_keys),
        "physical_steps": 0, "geometry_admitted": False,
        "complete_bank_accounting_rows": point_count,
        "complete_bank_accounting_verified": True,
    })
    validate_zero_feasible_builder_result(row_report, point_count=point_count,
                                          bank_count=len(bank_keys))
    raw_path = out / "report.json"
    raw_path.write_text(json.dumps(row_report, indent=2, sort_keys=True, default=str) + "\n")

    compatible = dict(row_report)
    compatible.update({
        "status": "PASS_POINT_DEPTH_ROWS_BASELINE_FEASIBLE",
        "adapter_schema": "numi.human.point-depth-rows-compatible.v1",
        "compatibility_basis": "Derived from candidate-matched raw 357 output and checked by the existing 401 validator.",
        "rows_npz": str(npz_path), "rows_npz_sha256": sha(npz_path),
        "baseline_points_sha256": array_sha(points),
        "baseline_points_dtype": str(points.dtype),
        "baseline_points_shape": list(points.shape),
        "owner_path": str(owner), "owner_sha256": owner_sha,
        "builder_path": str(owner), "builder_sha256": owner_sha,
        "exact_body_csv": str(body_csv), "exact_body_csv_sha256": sha(body_csv),
        "complete_bank_accounting_rows": point_count,
        "complete_bank_accounting_verified": True,
    })
    compat_path = out / "365-compatible-report.json"
    compat_path.write_text(json.dumps(compatible, indent=2, sort_keys=True, default=str) + "\n")
    validator = import_file("point_depth_validator_401", validator_path)
    arrays = {"node_rows": np.asarray(node_rows), "normals_world_unit_covector": np.asarray(normals),
              "lower_delta_m": np.asarray(lower), "typed_ids": np.asarray(typed)}
    validated = validator.validate_point_depth_rows_payload(
        arrays, compatible, base_candidate_sha256=actual_candidate_sha,
        base_points=points, rows_npz_path=npz_path, owner_path=owner,
        candidate_path=candidate)
    val = {"status": "PASS_401_POINT_DEPTH_PAYLOAD_VALIDATION",
           "candidate_path": str(candidate), "candidate_sha256": actual_candidate_sha,
           "raw_report_path": str(raw_path), "raw_report_sha256": sha(raw_path),
           "compat_report_path": str(compat_path), "compat_report_sha256": sha(compat_path),
           "rows_npz_path": str(npz_path), "rows_npz_sha256": sha(npz_path),
           "validator_path": str(validator_path), "validator_sha256": sha(validator_path),
           "bank_count": len(bank_keys), "row_count": len(validated[0]),
           "inside_row_count": inside_count, "outside_row_count": len(selected_outside),
           "unique_node_count": int(row_report["unique_node_count"]),
           "zero_step_violation_m": float(row_report["maximum_zero_step_violation_m"]),
           "complete_bank_accounting_verified": True,
           "validator_receipt": validated[-1]}
    validation_path = out / "365-validation.json"
    validation_path.write_text(json.dumps(val, indent=2, sort_keys=True, default=str) + "\n")
    return {
        "status": "PASS", "candidate": str(candidate), "candidate_sha256": actual_candidate_sha,
        "output_dir": str(out), "rows_npz": str(npz_path), "rows_npz_sha256": sha(npz_path),
        "raw_report": str(raw_path), "raw_report_sha256": sha(raw_path),
        "compatible_report": str(compat_path), "compatible_report_sha256": sha(compat_path),
        "validation": str(validation_path), "validation_sha256": sha(validation_path),
        "inside": inside_count, "outside": len(selected_outside), "row_count": len(records),
        "unique_nodes": int(row_report["unique_node_count"]),
        "max_zero_step_violation_m": float(row_report["maximum_zero_step_violation_m"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--point-summary", type=Path, required=True)
    parser.add_argument("--surface-report", type=Path, required=True)
    parser.add_argument("--query-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--expected-point-summary-sha256", required=True)
    parser.add_argument("--expected-surface-report-sha256", required=True)
    parser.add_argument("--expected-query-receipt-sha256", required=True)
    parser.add_argument("--expected-owner-sha256")
    parser.add_argument("--owner", type=Path, default=DEFAULT_OWNER)
    parser.add_argument("--validator", type=Path, default=DEFAULT_VALIDATOR)
    parser.add_argument("--threshold-m", type=float, default=0.0011)
    args = parser.parse_args(argv)
    if not np.isfinite(args.threshold_m) or args.threshold_m < 0:
        raise ValueError("threshold must be finite and nonnegative")
    print(json.dumps(build(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
