#!/usr/bin/env python3
"""Build direct-depth or source-oriented coherent point targets with exact all-bank first exits.

Construction proposals only. Vertex directions are area-weighted from unique
incident candidate faces, with face normals resolved through the pinned 303
registered source ancestry/orientation owner. Every emitted Float32 endpoint is
checked against the complete 185-bank point classifier.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

D = Path("/Users/n/Work/numi-human-connected-surface-069")
EVIDENCE = Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
DETAIL = D / "publication-072/evidence/run-006-direct-detail-p1-075/direct-detail-p1-075.npz"
BASE = D / "full-connected-clearance-150/run-028-current-target-connected-fit-001/candidate.npz"
ORIENT = D / "digit-source-diagnosis-110/source-path-orientation-audit-112.json"
AUDIT = D / "publication-084/canonical/matter/tools/human_skin_fat_bone_audit.py"
BANK = EVIDENCE / "bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight"
BANK_FIX = EVIDENCE / "route-audit-counter-fix-030/canonical/matter"
PRED = EVIDENCE / "incremental-exact-audit-owner-253/src/numilab_human/cardiac_cavity_intersections.py"
SOURCE303 = D / "barycentric-connected-clearance-170/source-outward-proposal-300/build_outward_nearest_union_targets_303.py"
SELECTOR300 = D / "barycentric-connected-clearance-170/source-outward-proposal-300/first_representable_union_target_300.py"
CLEARANCE_M = 5e-5

SOURCE371 = D / "publication-121/sources/connected-coherent-point-targets-371/build_coherent_point_targets_371.py"

PINS = {
    "371_base": "b0ef9f9b7e900ecbf5ae1e08bf450742735550fb6a673a3aa80f06e7ed5d7e7c",
    "303_source": "c8fe5b7be2efb2c6867a2cde178ebc35280814c37f90eec5aea47e964fc17a1f",
    "300_selector": "5e613e21b7bdebcc036dc9712ad21903df7d081de32bfdb83fae0a0c344ac094",
    "035_template": "cfa5071fb03e3b8d637b5be285a96b83aade01a755a663bc1c52a91fc093e240",
    "253_point_owner": "6f04ec7b129f47e810e75262f9f3ad3be0159bbe13dc2611a96e27141132b0fc",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def loadmod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_point_location_owner():
    """Import the pinned predicate as its real package for relative imports."""
    package_root = PRED.parents[1]
    if not package_root.is_dir():
        raise FileNotFoundError(f"pinned point-owner package root is missing: {package_root}")
    package_root_s = str(package_root)
    if package_root_s not in sys.path:
        sys.path.insert(0, package_root_s)
    module_name = "numilab_human.connected_skin_bone_exact_327"
    current = sys.modules.get(module_name)
    if current is not None:
        current_path = Path(getattr(current, "__file__", "")).resolve()
        if current_path != PRED.resolve():
            raise ImportError(f"another module already owns {module_name}: {current_path}")
        return current
    module = loadmod(PRED, module_name)
    if Path(module.__file__).resolve() != PRED.resolve():
        raise ImportError(f"loaded unexpected exact point owner: {module.__file__}")
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def validate_complete_surface_summary(summary: dict) -> None:
    if summary.get("complete_surface_audit") is not True:
        raise ValueError("point summary does not certify a complete surface audit")
    pairs = summary.get("exact_PLC_surface_pairs")
    if type(pairs) is not int or pairs != 0:
        raise ValueError("point summary does not certify zero exact PLC surface pairs")


def validate_candidate_geometry_arrays(candidate: dict) -> tuple[np.ndarray, np.ndarray]:
    raw_points = np.asarray(candidate["points_world_m"])
    raw_faces = np.asarray(candidate["source_O_faces"])
    if raw_points.dtype != np.dtype(np.float32):
        raise ValueError("candidate points_world_m must be stored as Float32")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or not np.issubdtype(raw_faces.dtype, np.integer):
        raise ValueError("candidate source_O_faces must be an integer F x 3 array")
    points = raw_points
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("candidate points must be finite N x 3 Float32")
    faces = raw_faces.astype(np.int64, copy=False)
    if np.any(faces < 0) or np.any(faces >= len(points)):
        raise ValueError("candidate face index outside point rows")
    return points, faces


def validate_query_receipt(receipt: dict, points_path: Path, nodes_path: Path,
                           pairs_path: Path, expected_rows: int) -> None:
    if int(receipt.get("query_rows", -1)) != expected_rows:
        raise ValueError("matched point query row count differs from candidate")
    for key, path, label in (
        ("points_sha256", points_path, "point"),
        ("nodes_sha256", nodes_path, "node"),
        ("pairs_sha256", pairs_path, "pair"),
    ):
        if receipt.get(key) != sha(path):
            raise ValueError(f"matched query {label} CSV SHA differs from query receipt")


def validate_exact_owner_apis(segment_owner: dict, point_owner) -> None:
    if not callable(segment_owner.get("classify_segment_triangle")):
        raise ValueError("pinned segment witness owner lacks classify_segment_triangle")
    required = ("_records", "prepare_point_location", "point_location_prepared_indexed")
    if any(not callable(getattr(point_owner, name, None)) for name in required):
        raise ValueError("pinned exact point owner lacks required point-location API")


def json_safe_selection_record(selection: dict) -> dict:
    """Retain unbounded interval semantics without emitting nonstandard JSON Infinity."""
    result = {k: v for k, v in selection.items() if k != "target_world_f32_m"}

    def encode_gap(record: dict) -> dict:
        encoded = dict(record)
        gap = encoded.get("free_gap_m")
        if isinstance(gap, (float, np.floating)) and math.isinf(float(gap)):
            encoded["free_gap_m"] = None
            encoded["free_gap_unbounded"] = True
        elif "free_gap_m" in encoded:
            encoded["free_gap_unbounded"] = False
        return encoded

    if "free_gap_m" in result:
        result = encode_gap(result)
    if "skipped_intervals" in result:
        result["skipped_intervals"] = [encode_gap(record) for record in result["skipped_intervals"]]
    return result


def ray_box(start_local: np.ndarray, direction_local: np.ndarray,
            lo: np.ndarray, hi: np.ndarray) -> tuple[float, float] | None:
    t0, t1 = -math.inf, math.inf
    for axis in range(3):
        d = float(direction_local[axis])
        q = float(start_local[axis])
        if d == 0.0:
            if q < lo[axis] or q > hi[axis]:
                return None
            continue
        a = (float(lo[axis]) - q) / d
        b = (float(hi[axis]) - q) / d
        if a > b:
            a, b = b, a
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return None
    if t1 < 0.0:
        return None
    return max(0.0, t0), t1


def source_face_area_weights(points_037: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Return positive source-037 reference areas, one per candidate face row."""
    p = np.asarray(points_037, dtype=np.float64)
    f = np.asarray(faces)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("source037 reference points must be finite N x 3")
    if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer):
        raise ValueError("candidate faces must be integer F x 3")
    if np.any(f < 0) or np.any(f >= len(p)):
        raise ValueError("candidate face index outside source037 point rows")
    tri = p[f.astype(np.int64)]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if not np.isfinite(area).all() or np.any(area < 0.0):
        raise ValueError("source037 contains nonfinite or negative face areas")
    return area


def vertex_source_outward_normals(candidate, source303, detail, base, orientation,
                                  target_rows: list[int]) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
    """Area-average unique incident face normals after typed 303 ancestry resolution."""
    points = np.asarray(candidate["points_world_m"], dtype=np.float32)
    faces = np.asarray(candidate["source_O_faces"])
    ref = np.asarray(candidate["area_reference_037_points_world_m_f64"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or faces.shape != (len(faces), 3):
        raise ValueError("candidate point/face arrays are malformed")
    if ref.shape != points.shape or not np.isfinite(ref).all():
        raise ValueError("matched source037 area reference is absent or misaligned")
    if not np.issubdtype(faces.dtype, np.integer) or np.any(faces < 0) or np.any(faces >= len(points)):
        raise ValueError("candidate faces are not valid integer node rows")
    faces = faces.astype(np.int64, copy=False)
    areas = source_face_area_weights(ref, faces)
    incident: dict[int, list[int]] = defaultdict(list)
    for face_row, tri in enumerate(faces):
        for node in set(map(int, tri)):
            incident[node].append(face_row)

    normals: dict[int, np.ndarray] = {}
    provenance: dict[int, dict] = {}
    face_cache: dict[int, tuple[np.ndarray, dict]] = {}
    for node in sorted(set(int(x) for x in target_rows)):
        if not 0 <= node < len(points):
            raise ValueError(f"target node row outside candidate: {node}")
        face_rows = sorted(set(incident.get(node, ())))
        if not face_rows:
            raise ValueError(f"target node {node} has no incident source faces")
        acc = np.zeros(3, dtype=np.float64)
        total_area = 0.0
        face_records = []
        for face_row in face_rows:
            weight = float(areas[face_row])
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(f"incident source037 face {face_row} has degenerate area")
            if face_row not in face_cache:
                n, src = source303.registered_source_outward_face_normal(
                    face_row, candidate, detail, base, orientation
                )
                n = np.asarray(n, dtype=np.float64)
                if n.shape != (3,) or not np.isfinite(n).all():
                    raise ValueError(f"face {face_row} has invalid registered source normal")
                nlen = float(np.linalg.norm(n))
                if not math.isclose(nlen, 1.0, rel_tol=0.0, abs_tol=2e-12):
                    raise ValueError(f"face {face_row} source normal is not unit length")
                face_cache[face_row] = (n, src)
            n, src = face_cache[face_row]
            acc += weight * n
            total_area += weight
            face_records.append({"candidate_face_row": face_row,
                                 "source037_area_m2": weight,
                                 "registered_source_normal_world_unit": n.tolist(),
                                 "source_ancestry": src})
        resultant = float(np.linalg.norm(acc))
        ratio = resultant / total_area if total_area > 0.0 else 0.0
        if not math.isfinite(ratio) or ratio <= 1e-12:
            raise ValueError(f"registered source normals cancel at target node {node}: {ratio}")
        normals[node] = acc / resultant
        provenance[node] = {
            "incident_candidate_face_rows": face_rows,
            "unique_incident_face_count": len(face_rows),
            "source037_area_sum_m2": total_area,
            "resultant_over_area_sum": ratio,
            "source_face_area_weights": face_records,
        }
    return normals, provenance



DEPTH357_OWNER_PATH = Path("/Users/n/Work/numi-human-connected-surface-069/D/point-depth-barriers-357/point_depth_barriers_357.py")
DEPTH357_OWNER_SHA256 = "dc8078515dd12b2edc62c0bc75759a7cb9472c9914e4704ecd41e04c4d5378e1"


def direct_depth_unit_direction(source_outward_world_unit, depth_covector_world_unit, **unused):
    """Use current obstacle depth direction, without a source-ray hemisphere heuristic."""
    w=np.asarray(source_outward_world_unit,dtype=np.float64)
    n=np.asarray(depth_covector_world_unit,dtype=np.float64)
    if w.shape!=(3,) or n.shape!=(3,) or not np.isfinite(w).all() or not np.isfinite(n).all():
        raise ValueError("invalid direct-depth vectors")
    if abs(float(np.linalg.norm(w))-1)>2e-10 or abs(float(np.linalg.norm(n))-1)>2e-10:
        raise ValueError("direct-depth vectors must be unit")
    return n.copy(),dict(method="direct matched depth covector",selected_depth_cosine=1.,selected_source_hemisphere_cosine=float(w@n),source_ray_hemisphere_required=False,source_normal_retained_for_provenance=True,direction_world_unit=n.tolist())


def depth_compatible_unit_direction(source_outward_world_unit, depth_covector_world_unit, *, minimum_depth_cosine=0.05):
    """Closest unit direction in the positive depth cap and source-outward hemisphere."""
    w = np.asarray(source_outward_world_unit, dtype=np.float64)
    n = np.asarray(depth_covector_world_unit, dtype=np.float64)
    cmin = float(minimum_depth_cosine)
    if w.shape != (3,) or n.shape != (3,) or not np.isfinite(w).all() or not np.isfinite(n).all():
        raise ValueError("source normal and depth covector must be finite 3-vectors")
    if not math.isfinite(cmin) or not 0.0 < cmin < 1.0:
        raise ValueError("minimum depth cosine must be finite and in (0,1)")
    wn, nn = float(np.linalg.norm(w)), float(np.linalg.norm(n))
    if wn <= 1e-14 or nn <= 1e-14:
        raise ValueError("source normal or depth covector is degenerate")
    if not math.isclose(wn, 1.0, rel_tol=0.0, abs_tol=2e-10):
        raise ValueError("source outward direction is not unit length")
    if not math.isclose(nn, 1.0, rel_tol=0.0, abs_tol=2e-10):
        raise ValueError("depth covector is not unit length")
    original_dot = float(np.dot(w, n))
    adjusted = original_dot < cmin
    if not adjusted:
        d = w.copy()
    else:
        tangent = w - original_dot * n
        tangent_norm = float(np.linalg.norm(tangent))
        if not math.isfinite(tangent_norm) or tangent_norm <= 1e-14:
            raise ValueError("depth-compatible source hemisphere is infeasible or degenerate")
        c = min(1.0, cmin + 8.0 * np.finfo(np.float64).eps)
        d = c * n + math.sqrt(1.0 - c*c) * tangent / tangent_norm
        dnorm = float(np.linalg.norm(d))
        if not math.isfinite(dnorm) or dnorm <= 1e-14:
            raise ValueError("projected direction is degenerate")
        d /= dnorm
    depth_dot = float(np.dot(n, d))
    source_dot = float(np.dot(w, d))
    if (not np.isfinite(d).all()
            or not math.isclose(float(np.linalg.norm(d)), 1.0, rel_tol=0.0, abs_tol=2e-14)
            or depth_dot < cmin or source_dot <= 1e-12):
        raise ValueError("no numerically certified direction satisfies both hemispheres")
    return d, {
        "adjusted": bool(adjusted),
        "source_outward_dot_depth_covector": original_dot,
        "selected_depth_cosine": depth_dot,
        "selected_source_hemisphere_cosine": source_dot,
        "minimum_depth_cosine": cmin,
        "direction_world_unit": d.tolist(),
        "method": "nearest unit direction in source-outward hemisphere intersected with depth-covector cap",
    }


def depth_maximizing_unit_direction(source_outward_world_unit, depth_covector_world_unit, *, minimum_cosine=0.05):
    """Maximize depth progress with both source-outward and depth cosines bounded."""
    w = np.asarray(source_outward_world_unit, dtype=np.float64)
    n = np.asarray(depth_covector_world_unit, dtype=np.float64)
    cmin = float(minimum_cosine)
    if w.shape != (3,) or n.shape != (3,) or not np.isfinite(w).all() or not np.isfinite(n).all():
        raise ValueError("source normal and depth covector must be finite 3-vectors")
    if not math.isfinite(cmin) or not 0.0 < cmin < 1.0:
        raise ValueError("minimum cosine must be finite and in (0,1)")
    wn, nn = float(np.linalg.norm(w)), float(np.linalg.norm(n))
    if not math.isclose(wn, 1.0, rel_tol=0.0, abs_tol=2e-10):
        raise ValueError("source outward direction is not unit length")
    if not math.isclose(nn, 1.0, rel_tol=0.0, abs_tol=2e-10):
        raise ValueError("depth covector is not unit length")
    dot = float(np.dot(w, n))
    if dot >= cmin:
        d = n.copy()
        active = False
    else:
        tangent = n - dot * w
        tangent_norm = float(np.linalg.norm(tangent))
        if not math.isfinite(tangent_norm) or tangent_norm <= 1e-14:
            raise ValueError("source/depth cosine constraints are infeasible or degenerate")
        # The optimum lies on w·d=cmin when n is outside that spherical cap.
        c = min(1.0, cmin + 64.0 * np.finfo(np.float64).eps)
        d = c * w + math.sqrt(1.0 - c*c) * tangent / tangent_norm
        dnorm = float(np.linalg.norm(d))
        if not math.isfinite(dnorm) or dnorm <= 1e-14:
            raise ValueError("projected direction is degenerate")
        d /= dnorm
        active = True
    depth_cosine = float(np.dot(n, d))
    source_cosine = float(np.dot(w, d))
    if (not np.isfinite(d).all()
            or not math.isclose(float(np.linalg.norm(d)), 1.0, rel_tol=0.0, abs_tol=2e-14)
            or depth_cosine < cmin or source_cosine < cmin):
        raise ValueError("no numerically certified direction satisfies both cosine constraints")
    return d, {
        "adjusted": bool(active),
        "source_outward_dot_depth_covector": dot,
        "selected_depth_cosine": depth_cosine,
        "selected_source_hemisphere_cosine": source_cosine,
        "minimum_depth_cosine": cmin,
        "minimum_source_outward_cosine": cmin,
        "direction_world_unit": d.tolist(),
        "method": "maximize depth-covector cosine subject to source-outward and depth cosine floors",
    }


def load_depth_covectors_357(npz_path, report_path, *, candidate_sha256, points_world_f32, expected_query_rows):
    if sha(DEPTH357_OWNER_PATH) != DEPTH357_OWNER_SHA256:
        raise ValueError("pinned 357 point-depth owner SHA mismatch")
    report = json.loads(Path(report_path).read_text())
    if report.get("owner_sha256") != DEPTH357_OWNER_SHA256 or report.get("builder_sha256") != DEPTH357_OWNER_SHA256:
        raise ValueError("357 report owner/source SHA does not match pinned owner")
    if report.get("status") != "PASS_POINT_DEPTH_ROWS_BASELINE_FEASIBLE":
        raise ValueError("357 report is not a baseline-feasible row table")
    if report.get("bank_count") != 185 or report.get("complete_bank_accounting_verified") is not True:
        raise ValueError("357 report lacks complete 185-bank accounting")
    if report.get("candidate_sha256") != candidate_sha256:
        raise ValueError("357 row table is not matched to candidate SHA")
    point_sha = hashlib.sha256(np.ascontiguousarray(points_world_f32).tobytes()).hexdigest()
    if report.get("candidate_points_sha256") != point_sha or report.get("baseline_points_sha256") != point_sha:
        raise ValueError("357 Float32 point bytes do not match candidate")
    if report.get("rows_npz_sha256") != sha(npz_path):
        raise ValueError("357 report does not pin this point-depth NPZ")
    z = np.load(npz_path, allow_pickle=False)
    rows, normals = np.asarray(z["node_rows"]), np.asarray(z["normals_world_unit_covector"])
    lower, typed = np.asarray(z["lower_delta_m"]), np.asarray(z["typed_ids"])
    nrows = len(rows)
    if rows.ndim != 1 or rows.dtype.kind not in "iu":
        raise ValueError("357 node_rows must be an integer vector")
    if normals.shape != (nrows, 3) or normals.dtype.kind != "f" or not np.isfinite(normals).all():
        raise ValueError("357 covectors must be finite Nx3")
    if lower.shape != (nrows,) or lower.dtype.kind != "f" or not np.isfinite(lower).all():
        raise ValueError("357 lower deltas must be finite N")
    if typed.shape != (nrows, 4) or typed.dtype.kind not in "iu" or not np.array_equal(typed[:, 0], rows):
        raise ValueError("357 typed IDs must be Nx4 and identify node rows")
    inside_indices = np.flatnonzero(lower == 0.0)
    if len(inside_indices) != int(report.get("inside_row_count", -1)):
        raise ValueError("357 inside row count differs from report")
    if set(map(int, rows[inside_indices])) != set(map(int, expected_query_rows)):
        raise ValueError("357 inside rows do not match current query inside rows")
    by_node = defaultdict(list)
    for i in inside_indices:
        node = int(rows[i])
        normal = normals[i].astype(np.float64, copy=True)
        norm = float(np.linalg.norm(normal))
        if not math.isfinite(norm) or not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=2e-10):
            raise ValueError(f"357 covector for node {node} is not unit length")
        by_node[node].append({"normal_world_unit_covector": normal,
                              "typed_id": typed[i].astype(np.int64).tolist(),
                              "source_row": int(i)})
    summary = {
        "npz_path": str(npz_path), "npz_sha256": sha(npz_path),
        "report_path": str(report_path), "report_sha256": sha(report_path),
        "owner_path": str(DEPTH357_OWNER_PATH), "owner_sha256": DEPTH357_OWNER_SHA256,
        "inside_row_count": int(len(inside_indices)),
        "unique_inside_node_count": len(by_node),
        "multirow_inside_node_count": int(sum(len(v) > 1 for v in by_node.values())),
        "max_inside_rows_per_node": max((len(v) for v in by_node.values()), default=0),
        "selected_rows_are_exact_zero_rhs": True,
    }
    return dict(by_node), summary


def single_depth_row_for_node(rows_for_node, node_row: int) -> dict:
    """Fail closed instead of choosing among multiple bone-contact rows."""
    if len(rows_for_node) != 1:
        raise ValueError(f"depth-compatible mode requires exactly one inside depth row for node {node_row}; got {len(rows_for_node)}")
    return rows_for_node[0]


def f32_bits_match(csv_row: dict[str, str], p: np.ndarray) -> bool:
    return all(int(csv_row[k]) == int(p.view(np.uint32)[j])
               for j, k in enumerate(("x_bits_u32", "y_bits_u32", "z_bits_u32")))


def run(args) -> tuple[dict, dict]:
    candidate_path = args.candidate.resolve()
    summary_path = args.point_summary.resolve()
    query_dir = args.query_dir.resolve()
    out_dir = args.output_dir.resolve()
    if out_dir.exists():
        raise FileExistsError(f"output exists: {out_dir}")
    for label, path, expected in (
        ("371 base builder", SOURCE371, PINS["371_base"]),
        ("303 source helper", SOURCE303, PINS["303_source"]),
        ("300 selector", SELECTOR300, PINS["300_selector"]),
        ("035 ray template", Path("/Users/n/Work/numi-human-connected-surface-069/full-connected-clearance-150/run-035-current-point-exit-targets-001/build_run034_first_union_exits.py"), PINS["035_template"]),
        ("253 exact point owner", PRED, PINS["253_point_owner"]),
    ):
        if sha(path) != expected:
            raise ValueError(f"pinned {label} changed: {path}")
    if not all(p.is_file() for p in (candidate_path, summary_path, query_dir / "query-receipt.json",
                                      query_dir / "points.csv", query_dir / "nodes.csv", query_dir / "pairs.csv")):
        raise FileNotFoundError("candidate, summary, or matched query export is missing")

    clearance_m = float(args.clearance_m)
    if not math.isfinite(clearance_m) or clearance_m <= 0.0:
        raise ValueError("clearance-m must be positive and finite")
    run_started_at = time.monotonic()
    candidate_sha = sha(candidate_path)
    point_summary = json.loads(summary_path.read_text())
    if point_summary.get("candidate_sha256") != candidate_sha:
        raise ValueError("point summary is not matched to candidate SHA")
    validate_complete_surface_summary(point_summary)
    z = np.load(candidate_path, allow_pickle=False)
    points, faces = validate_candidate_geometry_arrays(z)
    if int(point_summary.get("points", -1)) != len(points) or int(point_summary.get("faces", -1)) != len(faces):
        raise ValueError("point summary row counts do not match candidate")
    receipt_path = query_dir / "query-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    query_points_path = query_dir / "points.csv"
    query_nodes_path = query_dir / "nodes.csv"
    query_pairs_path = query_dir / "pairs.csv"
    validate_query_receipt(receipt, query_points_path, query_nodes_path, query_pairs_path, len(points))
    qpoints = read_csv(query_points_path)
    nodes = read_csv(query_nodes_path)
    pair_rows = read_csv(query_pairs_path)
    if len(qpoints) != len(points) or len(nodes) != len(points):
        raise ValueError("matched query point/node CSV row counts differ from candidate")
    for row, (csv_point, point) in enumerate(zip(qpoints, points)):
        if int(csv_point["query_row"]) != row or not f32_bits_match(csv_point, point):
            raise ValueError(f"matched query point Float32 identity mismatch at row {row}")
    node_by_row = {int(r["query_row"]): r for r in nodes}
    if len(node_by_row) != len(points) or set(node_by_row) != set(range(len(points))):
        raise ValueError("matched query node row identities are incomplete or duplicated")
    pairs_by_row: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in pair_rows:
        pairs_by_row[int(r["query_row"])].append(r)
    target_rows = sorted({int(r["query_row"]) for r in pair_rows if r["classification"] == "inside"})
    if not target_rows:
        raise ValueError("matched point query contains no inside nodes")

    source303 = loadmod(SOURCE303, "registered_source_normals_303_for_327")
    orientation_report = json.loads(source303.ORIENTATION.read_text())
    orientation_pins = source303.validate_source_orientation_input_pins(orientation_report)
    orientation = source303.validate_source_outward_orientation(orientation_report)
    detail = np.load(source303.DETAIL, allow_pickle=False)
    base = np.load(source303.BASE, allow_pickle=False)
    selector_module = loadmod(SELECTOR300, "first_representable_union_selector_300_for_327")
    select_target = selector_module.first_representable_union_target_300
    ref = np.asarray(z["area_reference_037_points_world_m_f64"], dtype=np.float64)
    source_normals, normal_provenance = vertex_source_outward_normals(
        z, source303, detail, base, orientation, target_rows
    )
    depth_by_node = {}
    depth_summary = None
    target_directions = dict(source_normals)
    direction_diagnostics = {}
    direction_mode = getattr(args, "direction_mode", "source-outward")
    if direction_mode not in ("source-outward", "depth-compatible", "depth-maximizing", "direct-depth", "coherent-depth"):
        raise ValueError(f"unsupported direction mode: {direction_mode}")
    if direction_mode in ("depth-compatible", "depth-maximizing", "direct-depth", "coherent-depth"):
        depth_npz = getattr(args, "point_depth_rows_npz", None)
        depth_report = getattr(args, "point_depth_rows_report", None)
        if depth_npz is None or depth_report is None:
            raise ValueError("depth-compatible mode requires both pinned 357 row inputs")
        depth_by_node, depth_summary = load_depth_covectors_357(
            depth_npz, depth_report, candidate_sha256=candidate_sha,
            points_world_f32=points, expected_query_rows=target_rows
        )
        for row in target_rows:
            depth_row = single_depth_row_for_node(depth_by_node.get(row, []), row)
            direction_builder = (direct_depth_unit_direction if direction_mode in ("direct-depth", "coherent-depth") else
                                 depth_compatible_unit_direction if direction_mode == "depth-compatible"
                                 else depth_maximizing_unit_direction)
            selected, diag = direction_builder(
                source_normals[row], depth_row["normal_world_unit_covector"],
                **({"minimum_depth_cosine": float(getattr(args, "minimum_depth_cosine", 0.05))}
                   if direction_mode == "depth-compatible" else
                   {"minimum_cosine": float(getattr(args, "minimum_depth_cosine", 0.05))})
            )
            target_directions[row] = selected
            direction_diagnostics[row] = {
                **diag, "typed_depth_id": depth_row["typed_id"],
                "depth_source_row": depth_row["source_row"],
            }

    coherent_groups=[]
    if direction_mode == "coherent-depth":
        # Group witness nodes sharing a face or one intermediate mesh neighbor.
        from scipy.optimize import linprog
        target_set=set(target_rows);parent={n:n for n in target_rows}
        def find(n):
            while parent[n]!=n:parent[n]=parent[parent[n]];n=parent[n]
            return n
        adjacent={n:set() for n in range(len(points))}
        for face in faces:
            for n in face:adjacent[int(n)].update(map(int,face))
        for node in target_rows:
            near=set(adjacent[node])
            for n in tuple(near):near.update(adjacent[n])
            for n in sorted(near&target_set):parent[find(n)]=find(node)
        grouped={}
        for node in target_rows:grouped.setdefault(find(node),[]).append(node)
        for group in sorted(grouped.values(),key=lambda q:q[0]):
            N=np.array([depth_by_node[n][0]["normal_world_unit_covector"] for n in group])
            # Maximum common depth progress over a bounded direction cube.
            result=linprog([0,0,0,-1],A_ub=np.column_stack([-N,np.ones(len(N))]),b_ub=np.zeros(len(N)),bounds=[(-1,1)]*3+[(0,None)],method="highs")
            if not result.success or result.x[3]<=1e-8:raise ValueError(f"coherent cluster has no positive common depth direction: {group}")
            direction=result.x[:3]/np.linalg.norm(result.x[:3]);cos=N@direction
            if np.min(cos)<=0:raise ValueError("coherent direction lost positive progress")
            coherent_groups.append(group)
            for node in group:
                target_directions[node]=direction.copy()
                direction_diagnostics[node].update(method="shared maximum-minimum depth progress direction for connected patch",coherent_group_rows=group,minimum_group_depth_cosine=float(cos.min()),selected_depth_cosine=float(np.dot(depth_by_node[node][0]["normal_world_unit_covector"],direction)),selected_source_hemisphere_cosine=float(np.dot(source_normals[node],direction)))

    reader = loadmod(AUDIT, "bone_bank_reader_327")
    triangles, ids, gids, bodies, matrices, offsets, positions, bankkeys, bankpins, exact = reader.read_bank_inputs(BANK, BANK_FIX)
    point_owner = load_point_location_owner()
    bankkeys = tuple((int(b), int(k)) for b, k in bankkeys)
    if len(bankkeys) != 185 or len(set(bankkeys)) != 185 or len(triangles) != len(ids):
        raise ValueError("pinned bone export is not a complete 185-bank inventory")
    validate_exact_owner_apis(exact, point_owner)
    banks: dict[tuple[int, int], dict] = {}
    for body, bone in bankkeys:
        mask = (ids[:, 0] == body) & (ids[:, 1] == bone)
        tri = np.asarray(triangles[mask], dtype=np.float64)
        # Exact bank CSV identity columns are body, bone, member, stable triangle ID.
        stable = np.asarray(ids[mask, 3], dtype=np.int64)
        if not len(tri) or len(np.unique(stable)) != len(stable):
            raise ValueError(f"empty/duplicate stable triangle IDs in bank {body}/{bone}")
        banks[(body, bone)] = {
            "tri": tri, "stable": stable, "lo": tri.min(axis=(0, 1)), "hi": tri.max(axis=(0, 1)),
            "tri_lo": tri.min(axis=1), "tri_hi": tri.max(axis=1),
            "M": np.asarray(matrices[body], dtype=np.float64),
            "pos": np.asarray(positions[body], dtype=np.float64),
            "off": np.asarray(offsets[body], dtype=np.float64), "locator": None,
        }

    def locator(key):
        rec = banks[key]
        if rec["locator"] is None:
            vertices = tuple(tuple(Fraction.from_float(float(x)) for x in p)
                             for p in rec["tri"].reshape(-1, 3))
            faces_local = tuple((3 * i, 3 * i + 1, 3 * i + 2) for i in range(len(rec["tri"])) )
            rec["locator"] = point_owner.prepare_point_location(point_owner._records(vertices, faces_local))
        return rec["locator"]

    def classify_endpoint(point_world: np.ndarray):
        classes = {}
        unresolved = []
        point = np.asarray(point_world, dtype=np.float64)
        for key in bankkeys:
            rec = banks[key]
            local = (point - rec["pos"]) @ rec["M"].T - rec["off"]
            if np.any(local < rec["lo"]) or np.any(local > rec["hi"]):
                classes[key] = "outside_aabb"
                continue
            q = tuple(Fraction.from_float(float(v)) for v in local)
            located = point_owner.point_location_prepared_indexed(q, locator(key))
            classification = located.get("location")
            if classification not in ("inside", "outside"):
                unresolved.append({"bank": list(key), "result": located})
            else:
                classes[key] = classification
        return classes, unresolved

    initial_inside_counts = 0
    results = []
    for row in target_rows:
        node_record = node_by_row[row]
        if int(node_record["total_bone_banks"]) != 185 or int(node_record["ambiguous_banks"]) or int(node_record["invalid_banks"]):
            raise ValueError(f"incomplete matched 185-bank point classes at row {row}")
        start = points[row].astype(np.float64)
        direction = target_directions[row]
        per_bank_class = {
            (int(r["bank_body"]), int(r["bank_stable_id"])): r["classification"]
            for r in pairs_by_row.get(row, ())
        }
        expected_classes = 185 - int(node_record["aabb_proven_outside_banks"])
        if len(per_bank_class) != expected_classes:
            raise ValueError(f"query class accounting mismatch at row {row}")
        initial_inside = {key for key, cls in per_bank_class.items() if cls == "inside"}
        if not initial_inside:
            raise ValueError(f"inside point row {row} has no typed inside-bank witness")
        initial_inside_counts += len(initial_inside)
        events = []
        unresolved = []
        parity_by_bank = {}
        for key in bankkeys:
            rec = banks[key]
            status = per_bank_class.get(key, "outside_aabb")
            if status not in ("inside", "outside", "outside_aabb"):
                raise ValueError(f"invalid start class {status!r} at point {row}, bank {key}")
            q0 = (start - rec["pos"]) @ rec["M"].T - rec["off"]
            dlocal = rec["M"] @ direction
            interval = ray_box(q0, dlocal, rec["lo"], rec["hi"])
            if interval is None:
                if status == "inside":
                    raise ValueError(f"inside point {row} is outside bank {key} local AABB")
                parity_by_bank[key] = {"start": status, "hits": 0, "end_inside": False}
                continue
            ray_enter, ray_exit = interval
            end_distance = ray_exit + clearance_m
            p0, p1 = q0 + ray_enter * dlocal, q0 + end_distance * dlocal
            lo, hi = np.minimum(p0, p1), np.maximum(p0, p1)
            candidates = np.flatnonzero(np.all(rec["tri_hi"] >= lo, axis=1) & np.all(rec["tri_lo"] <= hi, axis=1))
            hit_count = 0
            for ti in candidates:
                hit = exact["classify_segment_triangle"](p0, p1, rec["tri"][int(ti)])
                kind = hit["classification"]
                if kind == "proper_interior_hit":
                    distance = ray_enter + float(hit["t"]) * (end_distance - ray_enter)
                    events.append({"distance_m": distance, "bank": key,
                                   "triangle_stable_id": int(rec["stable"][int(ti)])})
                    hit_count += 1
                elif kind not in ("disjoint", "parallel_disjoint"):
                    unresolved.append({"body": key[0], "stable_bone_id": key[1],
                                       "triangle_stable_id": int(rec["stable"][int(ti)]),
                                       "classification": kind, "exact_result": hit})
            end_inside = (status == "inside") ^ (hit_count % 2 == 1)
            parity_by_bank[key] = {"start": status, "hits": hit_count, "end_inside": bool(end_inside)}
        if unresolved:
            results.append({"query_row": row, "status": "UNRESOLVED_EXACT_CONTACT",
                            "unresolved_count": len(unresolved), "witnesses": unresolved[:16]})
            continue
        remaining = [list(k) for k, value in parity_by_bank.items() if value["end_inside"]]
        if remaining:
            results.append({"query_row": row, "status": "RAY_PARITY_FAILURE", "remaining_inside_banks": remaining})
            continue
        selection = select_target(
            initial_inside, events, start_world_m=start, unit_direction_world=direction,
            clearance_m=clearance_m, classify_endpoint=classify_endpoint, expected_bank_keys=bankkeys,
        )
        if selection["status"] != "PASS_FIRST_CERTIFIED_PROPOSAL_INTERVAL":
            results.append({"query_row": row, "status": selection["status"],
                            "target_selection": {k: v for k, v in selection.items() if k != "target_world_f32_m"},
                            "direction_world_unit": direction.tolist(),
                            "source_surface_normal_world_unit": source_normals[row].tolist(),
                            "direction_compatibility": direction_diagnostics.get(row),
                            "normal_provenance": normal_provenance[row]})
            continue
        target = np.asarray(selection["target_world_f32_m"], dtype=np.float32)
        emitted_distance = float(selection["emitted_target_distance_m"])
        results.append({
            "query_row": row, "status": "PASS_FIRST_UNION_EXIT", "start_world_f32_m": start.tolist(),
            "target_world_f32_m": target.tolist(), "proposal_direction_world_unit": direction.tolist(),
            "source_surface_normal_world_unit": source_normals[row].tolist(),
            "direction_compatibility": direction_diagnostics.get(row),
            "target_distance_m": emitted_distance,
            "union_exit_distance_m": float(selection["exit_distance_m"]),
            "next_union_event_distance_m": selection["next_event_distance_m"],
            "free_interval_margin_m": emitted_distance - float(selection["exit_distance_m"]),
            "clearance_m": clearance_m, "initial_inside_banks": [list(k) for k in sorted(initial_inside)],
            "initial_inside_bank_count": len(initial_inside), "hit_count": len(events),
            "end_outside_all185_by_event_parity": True, "target_exactly_outside_all185": True,
            "exact_endpoint_bank_count": int(selection["classified_bank_count"]),
            "source_normal_provenance": normal_provenance[row],
            "selection": json_safe_selection_record(selection),
        })

    if direction_mode == "coherent-depth":
        result_by_row={r["query_row"]:r for r in results}
        for group in coherent_groups:
            records=[result_by_row[n] for n in group]
            if any(r["status"]!="PASS_FIRST_UNION_EXIT" for r in records):continue
            distance=max(r["target_distance_m"] for r in records)
            for r in records:
                node=r["query_row"];direction=target_directions[node]
                target=(points[node].astype(np.float64)+distance*direction).astype(np.float32)
                classes,unresolved=classify_endpoint(target)
                if unresolved or len(classes)!=185 or any(v!="outside" for v in classes.values()):
                    r.update(status="COHERENT_TRANSLATION_ENDPOINT_REJECTED",coherent_group_rows=group);continue
                r.update(target_world_f32_m=target.astype(float).tolist(),target_distance_m=distance,free_interval_margin_m=distance-r["union_exit_distance_m"],coherent_group_rows=group,coherent_translation_world_m=(distance*direction).tolist(),coherent_exact_endpoint_bank_count=len(classes))
                r["selection"]["coherent_translation_distance_m"]=distance
                r["selection"]["coherent_endpoint_exact_outside_all185"]=True

    passing = [r for r in results if r["status"] == "PASS_FIRST_UNION_EXIT"]
    inside_pair_count = sum(1 for r in pair_rows if r["classification"] == "inside")
    report = {
        "schema": "numi.human.point-first-union-exits.depth-maximizing-390.v1",
        "classification": "proposal-only construction targets; no fit, geometry admission, or physical steps",
        "direction_mode": direction_mode,
        "minimum_depth_cosine": (float(getattr(args, "minimum_depth_cosine", 0.05))
                                 if direction_mode in ("depth-compatible", "depth-maximizing", "direct-depth", "coherent-depth") else None),
        "depth_row_inputs": depth_summary,
        "inputs": {
            "candidate_sha256": candidate_sha, "point_summary_sha256": sha(summary_path),
            "query_receipt_sha256": sha(receipt_path), "query_points_sha256": sha(query_points_path),
            "query_nodes_sha256": sha(query_dir / "nodes.csv"), "query_pairs_sha256": sha(query_dir / "pairs.csv"),
            "source371_base_sha256": sha(SOURCE371), "source303_sha256": sha(SOURCE303), "selector300_sha256": sha(SELECTOR300),
            "template035_sha256": sha(Path("/Users/n/Work/numi-human-connected-surface-069/full-connected-clearance-150/run-035-current-point-exit-targets-001/build_run034_first_union_exits.py")),
            "source_orientation_report_sha256": sha(ORIENT), "orientation_input_pins": orientation_pins,
            "detail_source_sha256": sha(DETAIL), "base_source_candidate_sha256": sha(BASE),
            "bank_reader_sha256": sha(AUDIT), "exact_predicate_sha256": sha(PRED),
            "bank_export_pins": bankpins,
        },
        "candidate_points": len(points), "candidate_faces": len(faces),
        "query_rows": len(target_rows), "inside_query_nodes": len(target_rows),
        "inside_node_bank_pairs": inside_pair_count, "all_bank_ray_results": len(results),
        "passing_targets": len(passing), "physical_steps": 0,
        "status": "PASS_FIRST_UNION_EXIT_TARGETS_ONLY" if len(passing) == len(target_rows) else "INCOMPLETE_OR_REJECTED_TARGETS",
        "normal_method": "For each query node, deduplicate incident candidate face rows, resolve each via 303 typed original-face ancestry and pinned source orientation, and area-average unit normals using matched source037 face areas. Cancellation or missing ancestry fails closed; no nearest-boundary fallback.",
        "direction_method": (
            "Direct matched357 depth covector; source normal retained as provenance only. The source-outward ray hemisphere is a construction heuristic and is not imposed in this mode. All185 exact endpoint and connected-mesh area, edge, anchor and exact intersection gates remain mandatory." if direction_mode in ("direct-depth", "coherent-depth") else
            "Maximize depth-covector cosine subject to source-outward and depth cosine floors; single-row nodes only, multiple rows fail closed. This changes ray proposal direction only; exact endpoint and all-185-bank checks are unchanged."
            if direction_mode == "depth-maximizing" else
            ("Closest unit direction to registered source-outward normal satisfying the single matched depth-covector constraint n dot d >= minimum_depth_cosine, while remaining in the positive source-outward hemisphere. Multirow nodes fail closed. This changes ray proposal direction only; exact endpoint and all-185-bank checks are unchanged."
             if direction_mode == "depth-compatible" else "Registered source-outward direction.")),
        "coherent_group_rows": coherent_groups,
        "coherent_translation_policy": "one shared direction and maximum individual first-exit distance per mesh-connected witness group; synchronized Float32 endpoints reclassified exactly against all185 banks" if coherent_groups else None,
        "ray_and_endpoint_method": "Exact segment events use all 185 banks; selector300 requires a representable first union-empty interval and exact endpoint classification outside every complete bank.",
        "source037_area_reference_array_sha256": hashlib.sha256(np.ascontiguousarray(ref).tobytes()).hexdigest(),
        "clearance_m": clearance_m,
        "clearance_interpretation": "Numerical construction target margin only; not anatomical thickness or physical clearance admission.",
        "targets": results,
        "wall_seconds": time.monotonic() - run_started_at,
    }

    out_dir.mkdir(parents=True, exist_ok=False)
    query_rows = np.asarray(target_rows, dtype=np.int64)
    by_row = {int(r["query_row"]): r for r in results}
    target_array = np.asarray([by_row[row].get("target_world_f32_m", [np.nan] * 3) for row in target_rows], dtype=np.float32).reshape(-1, 3)
    direction_array = np.asarray([target_directions[row] for row in target_rows], dtype=np.float64).reshape(-1, 3)
    source_normal_array = np.asarray([source_normals[row] for row in target_rows], dtype=np.float64).reshape(-1, 3)
    depth_covector_array = np.asarray([
        depth_by_node[row][0]["normal_world_unit_covector"] if direction_mode in ("depth-compatible", "depth-maximizing", "direct-depth", "coherent-depth") else [np.nan]*3
        for row in target_rows
    ], dtype=np.float64).reshape(-1, 3)
    pass_mask = np.asarray([by_row[row]["status"] == "PASS_FIRST_UNION_EXIT" for row in target_rows], dtype=np.bool_)
    npz_path = out_dir / "nearest-union-exits.npz"
    with npz_path.open("xb") as f:
        np.savez_compressed(
            f, query_rows=query_rows, target_world_f32_m=target_array,
            proposal_direction_world_unit=direction_array,
            source_surface_normal_world_unit=source_normal_array,
            depth_covector_world_unit=depth_covector_array,
            direction_compatibility_mask=np.asarray([
                row in direction_diagnostics for row in target_rows
            ], dtype=np.bool_),
            source_normal_available_mask=np.ones(len(target_rows), dtype=np.bool_),
            pass_mask=pass_mask,
            target_distance_m=np.asarray([by_row[row].get("target_distance_m", np.nan) for row in target_rows], dtype=np.float64),
            target_exactly_outside_all185=pass_mask.copy(),
        )
    report_path = out_dir / "nearest-union-exits.json"
    report["outputs"] = {"npz": str(npz_path), "npz_sha256": sha(npz_path)}
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report, {"npz": npz_path, "report": report_path}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--point-summary", required=True, type=Path)
    parser.add_argument("--query-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clearance-m", type=float, default=CLEARANCE_M)
    parser.add_argument("--direction-mode", choices=("source-outward", "depth-compatible", "depth-maximizing", "direct-depth", "coherent-depth"), default="source-outward")
    parser.add_argument("--point-depth-rows-npz", type=Path)
    parser.add_argument("--point-depth-rows-report", type=Path)
    parser.add_argument("--minimum-depth-cosine", type=float, default=0.05)
    args = parser.parse_args(argv)
    report, paths = run(args)
    print(json.dumps({"report": str(paths["report"]), "report_sha256": sha(paths["report"]),
                      "npz": str(paths["npz"]), "npz_sha256": sha(paths["npz"]),
                      "query_rows": report["query_rows"], "passing_targets": report["passing_targets"],
                      "status": report["status"], "seconds": report["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
