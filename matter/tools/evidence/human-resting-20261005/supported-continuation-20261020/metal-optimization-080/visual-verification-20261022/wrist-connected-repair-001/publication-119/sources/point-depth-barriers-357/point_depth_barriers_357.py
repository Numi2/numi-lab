#!/usr/bin/env python3
"""Build sparse point-depth non-regression rows from matched exact-query CSVs.

Construction guidance only: emitted-F32 exact point and surface audits remain
mandatory. Raw body transforms are used; no orthogonality approximation.
"""
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np

def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f: return list(csv.DictReader(f))

def _vec(row: Mapping[str, str], keys: Sequence[str], label: str) -> np.ndarray:
    value = np.asarray([float(row[key]) for key in keys], dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all(): raise ValueError(f"{label} must contain three finite values")
    return value

def local_halfspace_world_row(world_to_body: np.ndarray, normal_local_unit: np.ndarray,
                               rhs_local_m: float) -> tuple[np.ndarray, float]:
    """Normalize the exact world covector M.T@n and its local-metric RHS together."""
    M = np.asarray(world_to_body, dtype=np.float64)
    nlocal = np.asarray(normal_local_unit, dtype=np.float64)
    rhs = float(rhs_local_m)
    if M.shape != (3, 3) or nlocal.shape != (3,) or not np.isfinite(M).all() or not np.isfinite(nlocal).all() or not np.isfinite(rhs):
        raise ValueError("halfspace inputs must be finite M(3,3), normal(3), rhs")
    covector = M.T @ nlocal
    scale = float(np.linalg.norm(covector))
    if scale <= 1e-15:
        raise ValueError("transformed halfspace covector is degenerate")
    return covector / scale, rhs / scale


def _load_bodies(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = {}
    for r in _rows(path):
        body = int(r["body_index"])
        if body in out: raise ValueError(f"duplicate body_index {body}")
        M = np.asarray([float(r[f"world_to_body_m{i}{j}"]) for i in range(3) for j in range(3)], dtype=np.float64).reshape(3, 3)
        pos = np.asarray([float(np.float32(r[f"position_{c}_m"])) for c in "xyz"], dtype=np.float64)
        off = np.asarray([float(r[f"com_offset_{c}_m"]) for c in "xyz"], dtype=np.float64)
        quat = np.asarray([float(np.float32(r[f"orientation_q{c}"])) for c in "xyzw"], dtype=np.float64)
        if not np.isfinite(M).all() or not np.isfinite(pos).all() or not np.isfinite(off).all() or not np.isfinite(quat).all():
            raise ValueError(f"body {body} has nonfinite pose")
        if abs(float(np.linalg.det(M))) < 1e-12: raise ValueError(f"body {body} transform is singular")
        quat[:3] *= -1.0
        def rotate(v): return v + 2.0 * np.cross(quat[:3], np.cross(quat[:3], v) + quat[3] * v)
        expected = np.column_stack([rotate(v) for v in np.eye(3)])
        if np.max(np.abs(M - expected)) >= 1e-12: raise ValueError(f"body {body} matrix/quaternion parity failure")
        out[body] = (M, pos, off)
    return out

def _load_query(spec: Mapping[str, Any], n: int, bank_count: int, bodies_path: Path):
    required = ("points_csv", "pairs_csv", "nodes_csv", "query_receipt_json", "kind")
    missing = [k for k in required if k not in spec]
    if missing: raise ValueError(f"query source missing fields: {missing}")
    kind = str(spec["kind"])
    if kind not in ("inside", "outside"): raise ValueError("query source kind must be inside or outside")
    points_path, pairs_path, nodes_path = Path(spec["points_csv"]), Path(spec["pairs_csv"]), Path(spec["nodes_csv"])
    receipt_path = Path(spec["query_receipt_json"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    argv = receipt.get("argv", [])
    if int(receipt.get("query_rows", -1)) != n or int(receipt.get("physical_steps", -1)) != 0:
        raise ValueError("query receipt row count or physical_steps mismatch")
    if len(argv) < 7 or int(argv[6]) != bank_count: raise ValueError("query receipt bank count mismatch")
    for path, arg_index, label in ((bodies_path, 2, "bodies"), (points_path, 3, "points"),
                                   (pairs_path, 4, "pairs"), (nodes_path, 5, "nodes")):
        if Path(argv[arg_index]).resolve() != path.resolve(): raise ValueError(f"query receipt argv {label} path mismatch")
    for path, key in ((points_path, "points_sha256"), (pairs_path, "pairs_sha256"), (nodes_path, "nodes_sha256")):
        if _sha(path) != receipt.get(key): raise ValueError(f"query receipt {key} mismatch")
    point_rows = _rows(points_path)
    if len(point_rows) != n: raise ValueError("points CSV row count mismatch")
    qpoints = np.asarray([_vec(r, tuple(f"{c}_m" for c in "xyz"), "query point") for r in point_rows], dtype=np.float64)
    nodes = _rows(nodes_path)
    if len(nodes) != n: raise ValueError("nodes CSV row count mismatch")
    for i, row in enumerate(nodes):
        if int(row["query_row"]) != i: raise ValueError("nodes CSV query rows are not contiguous")
        accounted = sum(int(row[key]) for key in ("aabb_proven_outside_banks", "parity_outside_banks",
                                                       "inside_banks", "ambiguous_banks", "invalid_banks"))
        if (int(row["total_bone_banks"]) != bank_count or accounted != bank_count
                or int(row["ambiguous_banks"]) != 0 or int(row["invalid_banks"]) != 0
                or int(row["parity_query_banks"]) != int(row["parity_outside_banks"]) + int(row["inside_banks"])):
            raise ValueError(f"query row {i} lacks complete valid bank accounting")
    metadata = {
        "kind": kind, "points_csv": str(points_path.resolve()), "points_sha256": _sha(points_path),
        "pairs_csv": str(pairs_path.resolve()), "pairs_sha256": _sha(pairs_path),
        "nodes_csv": str(nodes_path.resolve()), "nodes_sha256": _sha(nodes_path),
        "query_receipt_json": str(receipt_path.resolve()), "query_receipt_sha256": _sha(receipt_path),
        "binary_sha256": receipt.get("binary_sha256"), "body_csv_sha256": _sha(bodies_path),
        "query_rows": n, "bank_count": bank_count,
        "complete_bank_accounting_verified": True,
        "complete_bank_accounting_rows": n,
        "ambiguous_rows": 0, "invalid_rows": 0,
    }
    return _rows(pairs_path), qpoints, metadata

def build_point_depth_rows(
    baseline_points_world_m: np.ndarray,
    query_sources: Sequence[Mapping[str, Any]],
    bodies_csv_path: str | Path,
    *,
    expected_point_count: int | None = None,
    expected_bank_count: int = 185,
    baseline_feasibility_tolerance_m: float = 1e-10,
    reconstruction_tolerance_m: float = 2e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    """Return node rows, unit world covectors, lower bounds, IDs, records, report.

    Each query source has points_csv, pairs_csv, nodes_csv, query_receipt_json,
    kind, and optionally require_points_match/selected_typed_pairs. Inside selects
    every exact inside pair; outside requires (query_row, body, bone[, triangle]).

    Since delta_local=M@delta_world, coefficients are normalized M.T@n_local.
    The RHS is divided by the same norm. Rows preserve the non-regression
    halfspace relative to the saved exact query and the solver baseline.
    """
    p_raw = np.asarray(baseline_points_world_m)
    if p_raw.ndim != 2 or p_raw.shape[1] != 3 or not np.issubdtype(p_raw.dtype, np.floating):
        raise ValueError("baseline_points_world_m must be floating (N,3)")
    if not np.isfinite(p_raw).all(): raise ValueError("baseline points must be finite")
    p = p_raw.astype(np.float64, copy=False)
    n = len(p) if expected_point_count is None else int(expected_point_count)
    if n != len(p) or n <= 0: raise ValueError("expected point count does not match baseline")
    if not query_sources: raise ValueError("at least one query source is required")
    tol, recon_tol = float(baseline_feasibility_tolerance_m), float(reconstruction_tolerance_m)
    if not np.isfinite(tol) or tol < 0 or not np.isfinite(recon_tol) or recon_tol <= 0:
        raise ValueError("invalid feasibility or reconstruction tolerance")
    bodies_path = Path(bodies_csv_path)
    bodies = _load_bodies(bodies_path)
    out_nodes, out_normals, out_lower, out_ids, records, receipts = [], [], [], [], [], []
    seen = set()
    for spec in query_sources:
        kind = str(spec["kind"])
        pair_rows, qpoints, receipt = _load_query(spec, n, expected_bank_count, bodies_path)
        receipts.append(receipt)
        selected = spec.get("selected_typed_pairs")
        if kind == "inside":
            if selected is not None: raise ValueError("inside sources select all inside rows; omit selected_typed_pairs")
            chosen = [r for r in pair_rows if r["classification"] == "inside"]
            for r in chosen:
                if int(r["inside_rays"]) <= 0 or int(r["outside_rays"]) != 0 or int(r["ambiguous"]) != 0:
                    raise ValueError("inside witness has inconsistent exact classifier counts")
        else:
            if selected is None: raise ValueError("outside source requires selected_typed_pairs")
            chosen = []
            for raw in selected:
                item = tuple(int(x) for x in raw)
                if len(item) not in (3, 4): raise ValueError("outside typed IDs require (row,body,bone[,triangle])")
                qr, body, bone = item[:3]; tri = item[3] if len(item) == 4 else None
                matches = [r for r in pair_rows if int(r["query_row"]) == qr and int(r["bank_body"]) == body
                           and int(r["bank_stable_id"]) == bone and (tri is None or int(r["nearest_triangle"]) == tri)]
                if len(matches) != 1: raise ValueError(f"outside typed witness {item} matched {len(matches)} rows")
                r = matches[0]
                if r["classification"] != "outside" or int(r["outside_rays"]) <= 0 or int(r["ambiguous"]) != 0:
                    raise ValueError(f"selected outside witness {item} is not unambiguous")
                chosen.append(r)
        expected = spec.get("expected_selected_count")
        if expected is not None and len(chosen) != int(expected): raise ValueError(f"selected count {len(chosen)} != expected {expected}")
        if bool(spec.get("require_points_match", False)) and not np.array_equal(qpoints.astype(np.float32), p_raw.astype(np.float32)):
            raise ValueError("query points do not match solver baseline F32 coordinates")
        for r in chosen:
            qr, body = int(r["query_row"]), int(r["bank_body"])
            bone, tri = int(r["nearest_bone"]), int(r["nearest_triangle"])
            if not 0 <= qr < n or body not in bodies or bone != int(r["bank_stable_id"]) or bone < 0 or tri < 0:
                raise ValueError("invalid typed node/body/bone/triangle identity")
            key = (qr, body, bone, tri, kind)
            if key in seen: raise ValueError(f"duplicate typed witness {key}")
            seen.add(key)
            M, pos, off = bodies[body]
            qlocal = _vec(r, tuple(f"query_local_{c}" for c in "xyz"), "query local point")
            clocal = _vec(r, tuple(f"closest_{c}" for c in "xyz"), "closest local point")
            qworld_pose = np.linalg.solve(M, qlocal + off) + pos
            cworld = np.linalg.solve(M, clocal + off) + pos
            qworld = qpoints[qr]
            recon = float(np.linalg.norm(qworld_pose - qworld))
            if recon > recon_tol: raise ValueError(f"local/world query mismatch at row {qr}: {recon:g}m")
            depth_local = float(np.linalg.norm(clocal - qlocal))
            if not np.isfinite(depth_local) or depth_local <= 1e-15: raise ValueError(f"degenerate direction for {key}")
            nlocal = (clocal - qlocal) / depth_local if kind == "inside" else (qlocal - clocal) / depth_local
            normal, rhs_scale_check = local_halfspace_world_row(M, nlocal, 1.0)
            scale = 1.0 / rhs_scale_check
            baseline_local = M @ (p[qr] - pos) - off
            same_f32_query_row = np.array_equal(qworld.astype(np.float32), p_raw[qr].astype(np.float32))
            if kind == "inside":
                rhs_local = 0.0 if same_f32_query_row else -float(np.dot(nlocal, baseline_local - qlocal))
            else:
                rhs_local = -depth_local if same_f32_query_row else -float(np.dot(nlocal, baseline_local - clocal))
            rhs = rhs_local / scale
            if not np.isfinite(rhs) or rhs > tol: raise ValueError(f"zero-step baseline violates {key}: rhs={rhs:g}m")
            out_nodes.append(qr); out_normals.append(normal); out_lower.append(rhs); out_ids.append((qr, body, bone, tri))
            records.append({
                "kind": "inside_nonregression" if kind == "inside" else "outside_tangent_nonregression",
                "node_row": qr, "body_index": body, "stable_bone_id": bone, "closest_triangle_id": tri,
                "normal_world_unit_covector": normal.tolist(), "lower_delta_m": rhs, "local_halfspace_rhs_m": rhs_local,
                "local_depth_m": depth_local, "world_query_to_closest_m": float(np.linalg.norm(qworld-cworld)),
                "covector_scale": scale, "query_point_world_m": qworld.tolist(), "closest_point_world_m": cworld.tolist(),
                "solver_baseline_point_world_m": p[qr].tolist(), "pose_reconstruction_residual_m": recon,
                "query_classification": r["classification"], "points_sha256": receipt["points_sha256"],
                "pairs_sha256": receipt["pairs_sha256"],
            })
    if not records: raise ValueError("no selected point-depth rows")
    node_array = np.asarray(out_nodes, dtype=np.int64)
    normal_array = np.asarray(out_normals, dtype=np.float64)
    lower_array = np.asarray(out_lower, dtype=np.float64)
    typed_array = np.asarray(out_ids, dtype=np.int64)
    report = {
        "status": "PASS_POINT_DEPTH_ROWS_BASELINE_FEASIBLE",
        "scope": "construction-only local tangent non-regression; exact emitted-F32 gates remain mandatory",
        "baseline_points_shape": list(p_raw.shape), "baseline_points_dtype": str(p_raw.dtype),
        "baseline_points_sha256": hashlib.sha256(np.ascontiguousarray(p_raw).tobytes()).hexdigest(),
        "bank_count": int(expected_bank_count), "row_count": len(records),
        "inside_row_count": sum(x["kind"] == "inside_nonregression" for x in records),
        "outside_row_count": sum(x["kind"] == "outside_tangent_nonregression" for x in records),
        "unique_node_count": int(len(np.unique(node_array))),
        "maximum_zero_step_violation_m": float(max(0.0, float(lower_array.max(initial=0.0)))),
        "baseline_feasibility_tolerance_m": tol, "exact_body_csv": str(bodies_path.resolve()),
        "exact_body_csv_sha256": _sha(bodies_path), "query_sources": receipts, "records": records,
        "physical_steps": 0, "geometry_admitted": False,
    }
    return node_array, normal_array, lower_array, typed_array, records, report
