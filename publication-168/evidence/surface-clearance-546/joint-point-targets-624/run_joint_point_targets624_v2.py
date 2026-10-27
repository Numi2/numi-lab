#!/usr/bin/env python3
"""Strict joint source/current/depth adapter, JSON-safe revision 2, for pinned 545.

This adapter changes only the proposal direction. The pinned 545 selector and its
complete 185-bank endpoint logic remain the sole target and endpoint owner.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
D = ROOT / "D"
HERE = D / "surface-clearance-546" / "joint-point-targets-624"
BUILDER = D / "surface-clearance-546/direct-depth-targets-546/source-snapshots/build_direct_depth_targets545.py"
BUILDER_SHA256 = "2864d9547d2715c61c9c27b17bcfd11fac250197c47f4fd4870f8f6898cecb60"
EXTENSION = D / "surface-clearance-546/direct-depth-targets-546/source-snapshots/build_outward_targets_multi_source_546.py"
EXTENSION_SHA256 = "dd61f0fb87f12c132166515a1e058e144d493a835cadbb2a47e6a2c8574e1e68"
OWNER500 = D / "source-outward-proposal-500/build_outward_nearest_union_targets_500.py"
OWNER500_SHA256 = "454501303db7c6420ff3744729c62c723c208aaf9348a53534137c62b7e6afbf"
ADAPTER589 = D / "surface-clearance-546/source-outward-targets-589/run_depthmax_targets589_v2.py"
ADAPTER589_SHA256 = "cd64f5280dec578ef8441aa9e44b9c01d44ba703edbd3fde39167ed471c71200"
ANGULAR597 = D / "surface-clearance-546/source-current-targets-597/angular_joint_direction597.py"
ANGULAR597_SHA256 = "4df66319908259efdc9faacdec5774876780fcf47c3f00ec0248a1a79810f092"
MINIMUM_COSINE = 0.05
EXPECTED_BANKS = 185


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_array_bytes(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8).tobytes()).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load pinned module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class JointDirectionFailure(ValueError):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        rows = [str(r["query_row"]) for r in records if r.get("status") != "PASS"]
        super().__init__("joint source/current/depth direction preflight failed for rows " + ",".join(rows))


def validate_parent_prefix(parent_points: np.ndarray, candidate_points: np.ndarray,
                            parent_prefix_node_count: int, expected_sha256: str) -> str:
    parent = np.asarray(parent_points)
    child = np.asarray(candidate_points)
    count = int(parent_prefix_node_count)
    if parent.dtype != np.float32 or child.dtype != np.float32:
        raise ValueError("585 parent and candidate point arrays must both be Float32")
    if parent.ndim != 2 or child.ndim != 2 or parent.shape[1:] != (3,) or child.shape[1:] != (3,):
        raise ValueError("585 parent and candidate point arrays must be Nx3")
    if count <= 0 or len(parent) != count or len(child) < count:
        raise ValueError("585 parent prefix count does not match parent/candidate point arrays")
    parent_prefix = np.ascontiguousarray(parent)
    child_prefix = np.ascontiguousarray(child[:count])
    parent_bits = parent_prefix.view(np.uint32)
    child_bits = child_prefix.view(np.uint32)
    if not np.array_equal(parent_bits, child_bits):
        raise ValueError("candidate point prefix differs bitwise from the 585 parent")
    digest = sha256_array_bytes(parent_prefix)
    if digest != expected_sha256 or sha256_array_bytes(child_prefix) != expected_sha256:
        raise ValueError("585 parent/candidate point-prefix SHA does not match refinement report")
    return digest


def validate_refinement_lineage(candidate_path: Path, refinement_report_path: Path) -> dict[str, Any]:
    candidate_path = candidate_path.resolve()
    refinement_report_path = refinement_report_path.resolve()
    report = json.loads(refinement_report_path.read_text())
    if report.get("status") != "REFINEMENT_CONSTRUCTION_READY":
        raise ValueError("refinement report is not a completed 585 construction")
    if report.get("physical_steps") != 0 or report.get("geometry_admitted") is not False:
        raise ValueError("refinement report has invalid physical/admission state")
    if report.get("existing_node_coordinates_moved") is not False or report.get("old_point_prefix_bitwise_unchanged") is not True:
        raise ValueError("refinement report does not certify its unchanged point prefix")
    candidate_sha = sha256_file(candidate_path)
    if report.get("output_candidate_sha256") != candidate_sha:
        raise ValueError("refinement output candidate SHA does not match --candidate")
    if report.get("candidate_sha256") != report.get("input_candidate_sha256"):
        raise ValueError("refinement parent candidate SHA aliases disagree")
    parent_path_value = report.get("input_candidate_path")
    if not isinstance(parent_path_value, str) or not parent_path_value:
        raise ValueError("refinement report lacks its source parent candidate path")
    parent_path = Path(parent_path_value)
    if not parent_path.is_absolute():
        parent_path = refinement_report_path.parent / parent_path
    parent_path = parent_path.resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"585 parent candidate is missing: {parent_path}")
    parent_sha = sha256_file(parent_path)
    if parent_sha != report.get("input_candidate_sha256"):
        raise ValueError("585 parent candidate SHA does not match refinement report")
    count = int(report.get("parent_prefix_node_count", -1))
    prefix_sha = str(report.get("parent_prefix_points_bytes_sha256", ""))
    if len(prefix_sha) != 64:
        raise ValueError("585 refinement report has no valid parent point-prefix SHA")
    with np.load(candidate_path, allow_pickle=False) as z:
        if "points_world_m" not in z:
            raise ValueError("candidate lacks points_world_m")
        candidate_points = np.asarray(z["points_world_m"])
    with np.load(parent_path, allow_pickle=False) as z:
        if "points_world_m" not in z:
            raise ValueError("585 parent lacks points_world_m")
        parent_points = np.asarray(z["points_world_m"])
    computed_prefix_sha = validate_parent_prefix(parent_points, candidate_points, count, prefix_sha)
    return {
        "refinement_report_path": str(refinement_report_path),
        "refinement_report_sha256": sha256_file(refinement_report_path),
        "refinement_adapter": report.get("adapter"),
        "refinement_adapter_sha256": report.get("adapter_sha256"),
        "refinement_585_owner_sha256": report.get("edge_only_refinement585_sha256"),
        "parent_candidate_path": str(parent_path),
        "parent_candidate_sha256": parent_sha,
        "candidate_path": str(candidate_path),
        "candidate_sha256": candidate_sha,
        "parent_prefix_node_count": count,
        "parent_prefix_points_bytes_sha256": computed_prefix_sha,
        "old_point_prefix_bitwise_unchanged": True,
    }


def compute_joint_directions(
    query_rows: list[int] | np.ndarray,
    source_by_row: dict[int, np.ndarray],
    current_by_row: dict[int, np.ndarray],
    depth_by_node: dict[int, list[dict[str, Any]]],
    angular_helper,
    minimum_cosine: float = MINIMUM_COSINE,
    single_depth_row=None,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    rows = [int(x) for x in query_rows]
    if len(rows) != len(set(rows)):
        raise ValueError("inside query rows contain duplicate identities")
    if not rows:
        raise ValueError("inside query rows are empty")
    floor = float(minimum_cosine)
    if not math.isfinite(floor) or not 0.0 < floor < 1.0:
        raise ValueError("minimum cosine must be finite and in (0,1)")
    directions: dict[int, np.ndarray] = {}
    diagnostics: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for row in rows:
        record: dict[str, Any] = {"query_row": row}
        try:
            if row not in source_by_row:
                raise ValueError("registered source normal is absent")
            if row not in current_by_row:
                raise ValueError("current emitted-F32 normal is unresolved")
            rows_for_node = depth_by_node.get(row, [])
            depth_row = (single_depth_row(rows_for_node, row) if single_depth_row is not None
                         else one_depth_row(rows_for_node, row))
            depth = np.asarray(depth_row["normal_world_unit_covector"], dtype=np.float64)
            source = np.asarray(source_by_row[row], dtype=np.float64)
            current = np.asarray(current_by_row[row], dtype=np.float64)
            direction, helper_diag = angular_helper(source, current, depth, floor)
            direction = np.asarray(direction, dtype=np.float64)
            if direction.shape != (3,) or not np.isfinite(direction).all():
                raise ValueError("angular helper returned an invalid direction")
            source_cosine = float(source @ direction)
            current_cosine = float(current @ direction)
            depth_cosine = float(depth @ direction)
            if min(source_cosine, current_cosine, depth_cosine) < floor - 1e-12:
                raise ValueError("joint direction failed one of the three cosine floors")
            record.update({
                "status": "PASS",
                "typed_depth_id": depth_row.get("typed_id"),
                "depth_source_row": depth_row.get("source_row"),
                "source_outward_normal_world_unit": source.tolist(),
                "current_outward_normal_world_unit": current.tolist(),
                "depth_covector_world_unit": depth.tolist(),
                "direction_world_unit": direction.tolist(),
                "selected_source_hemisphere_cosine": source_cosine,
                "selected_current_cosine": current_cosine,
                "selected_depth_cosine": depth_cosine,
                "minimum_cosine": floor,
                "angular_selection": {
                    **helper_diag,
                    # The 597 helper's second cosine slot is current-normal
                    # cosine. Keep the depth/current labels explicit here.
                    "selected_reference_cosine": float(depth @ direction),
                    "selected_source_hemisphere_cosine": source_cosine,
                    "selected_current_cosine": current_cosine,
                    "selected_depth_cosine": depth_cosine,
                    "direction_world_unit": direction.tolist(),
                },
            })
            directions[row] = direction
            diagnostics[row] = record
        except Exception as exc:
            record.update({"status": "FAIL_CLOSED", "reason": f"{type(exc).__name__}: {exc}"})
            failures.append(record)
            diagnostics[row] = record
    if failures:
        raise JointDirectionFailure([diagnostics[row] for row in rows])
    return directions, diagnostics


def one_depth_row(rows_for_node: list[dict[str, Any]], node_row: int) -> dict[str, Any]:
    if len(rows_for_node) != 1:
        raise ValueError(f"exactly one inside depth row required for node {node_row}; got {len(rows_for_node)}")
    return rows_for_node[0]


def validate_row_identity(query_rows, report_targets, archive_rows) -> None:
    query = np.asarray(query_rows, dtype=np.int64)
    archive = np.asarray(archive_rows, dtype=np.int64)
    target = [int(item["query_row"]) for item in report_targets]
    if query.ndim != 1 or archive.ndim != 1:
        raise ValueError("query row arrays must be one-dimensional")
    if len(query) != len(set(map(int, query))) or len(archive) != len(set(map(int, archive))):
        raise ValueError("query/archive row identities are duplicated")
    if target != list(map(int, query)) or not np.array_equal(archive, query):
        raise ValueError("query, report, and target archive row identities/order differ")


def validate_all185_records(report_targets: list[dict[str, Any]], bank_count: int = EXPECTED_BANKS) -> None:
    if bank_count != EXPECTED_BANKS:
        raise ValueError(f"expected complete {EXPECTED_BANKS}-bank owner, got {bank_count}")
    if not report_targets:
        raise ValueError("target report is empty")
    for item in report_targets:
        selection = item.get("selection", {})
        if (item.get("status") != "PASS_FIRST_UNION_EXIT"
                or item.get("target_exactly_outside_all185") is not True
                or item.get("exact_endpoint_bank_count") != EXPECTED_BANKS
                or item.get("end_outside_all185_by_event_parity") is not True
                or selection.get("target_exactly_outside_all_banks") is not True
                or selection.get("classified_bank_count") != EXPECTED_BANKS):
            raise ValueError(f"query row {item.get('query_row')} lacks exact first-union/all-185 endpoint proof")


def validate_emitted_f32_displacements(
    starts_f32: np.ndarray,
    targets_f32: np.ndarray,
    directions: np.ndarray,
    source_normals: np.ndarray,
    current_normals: np.ndarray,
    depth_covectors: np.ndarray,
    minimum_cosine: float = MINIMUM_COSINE,
) -> dict[str, Any]:
    arrays = [np.asarray(a) for a in
              (starts_f32, targets_f32, directions, source_normals, current_normals, depth_covectors)]
    if any(a.shape != (len(arrays[0]), 3) for a in arrays) or len(arrays[0]) == 0:
        raise ValueError("F32 displacement arrays must have the same nonempty Nx3 shape")
    if arrays[0].dtype != np.float32 or arrays[1].dtype != np.float32:
        raise ValueError("start and emitted target arrays must preserve Float32 coordinates")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("proposal/emitted displacement contains nonfinite values")
    starts, targets = arrays[0], arrays[1]
    dirs, source, current, depth = [a.astype(np.float64, copy=False) for a in arrays[2:]]
    floor = float(minimum_cosine)
    norms = [np.linalg.norm(a, axis=1) for a in (dirs, source, current, depth)]
    if any(np.any(n <= 0.0) or not np.isfinite(n).all() for n in norms):
        raise ValueError("proposal/outward vectors must be nondegenerate")
    dirs, source, current, depth = [a / n[:, None] for a, n in zip((dirs, source, current, depth), norms)]
    proposal_cosines = {
        "source": np.einsum("ij,ij->i", source, dirs),
        "current": np.einsum("ij,ij->i", current, dirs),
        "depth": np.einsum("ij,ij->i", depth, dirs),
    }
    if any(np.any(c < floor - 1e-12) for c in proposal_cosines.values()):
        raise ValueError("proposal direction violates a source/current/depth cosine floor")
    emitted = targets.astype(np.float64) - starts.astype(np.float64)
    emitted_dots = {
        "source": np.einsum("ij,ij->i", source, emitted),
        "current": np.einsum("ij,ij->i", current, emitted),
        "depth": np.einsum("ij,ij->i", depth, emitted),
    }
    if any(np.any(v <= 0.0) for v in emitted_dots.values()):
        raise ValueError("emitted Float32 displacement is not positive for all three outward vectors")
    return {
        "row_count": len(starts),
        "minimum_proposal_cosines": {k: float(v.min()) for k, v in proposal_cosines.items()},
        "minimum_emitted_displacement_dots": {k: float(v.min()) for k, v in emitted_dots.items()},
        "emitted_delta_dtype": "Float64 subtraction of exact emitted Float32 target and start coordinates",
    }


def json_plain(value: Any) -> Any:
    """Convert nested runtime diagnostics into bounded, strict-JSON values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return json_plain(value.tolist())
    if isinstance(value, np.generic):
        return json_plain(value.item())
    if isinstance(value, dict):
        return {str(json_plain(key)): json_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return f"nonfinite:{value!r}"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    raise TypeError(f"unsupported diagnostic value for strict JSON: {type(value).__name__}")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(json_plain(value), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def failure_path(output_dir: Path) -> Path:
    return output_dir.parent / (output_dir.name + ".joint-point-targets-624.failure.json")


def run(args) -> dict[str, Any]:
    candidate_path = args.candidate.resolve()
    point_summary_path = args.point_summary.resolve()
    query_dir = args.query_dir.resolve()
    depth_npz = args.depth_npz.resolve()
    depth_report = args.depth_report.resolve()
    refinement_report_path = args.refinement_report.resolve()
    output_dir = args.output_dir.resolve()
    sidecar = failure_path(output_dir)
    context: dict[str, Any] = {"schema": "numi.joint-point-targets-624.failure.v2",
                               "candidate_path": str(candidate_path)}
    if abs(float(args.minimum_cosine) - MINIMUM_COSINE) > 1e-12:
        raise ValueError("624 uses the pinned .05 source/current/depth cosine floor")
    if output_dir.exists():
        raise FileExistsError(f"output exists: {output_dir}")
    if sidecar.exists():
        raise FileExistsError(f"failure sidecar exists: {sidecar}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        expected = (
            (BUILDER, BUILDER_SHA256, "545 builder"),
            (EXTENSION, EXTENSION_SHA256, "546 multisource extension"),
            (OWNER500, OWNER500_SHA256, "500 source-normal owner"),
            (ADAPTER589, ADAPTER589_SHA256, "589 validation/normal-alignment adapter"),
            (ANGULAR597, ANGULAR597_SHA256, "597 joint angular helper"),
        )
        source_pins = {}
        for path, digest, label in expected:
            actual = sha256_file(path)
            source_pins[label] = {"path": str(path), "sha256": actual}
            if actual != digest:
                raise ValueError(f"pinned {label} changed: {path}")
        context["source_pins"] = source_pins
        lineage = validate_refinement_lineage(candidate_path, refinement_report_path)
        context["lineage"] = lineage
        if not point_summary_path.is_file() or not (query_dir / "query-receipt.json").is_file():
            raise FileNotFoundError("matched point summary or query receipt is missing")
        context["point_summary_sha256"] = sha256_file(point_summary_path)
        context["query_receipt_sha256"] = sha256_file(query_dir / "query-receipt.json")
        context["depth_npz_sha256"] = sha256_file(depth_npz)
        context["depth_report_sha256"] = sha256_file(depth_report)
        angular_mod = load_module(ANGULAR597, "joint_direction_angular597_for_624")
        adapter589 = load_module(ADAPTER589, "joint_direction_normal_adapter589_for_624")
        extension = load_module(EXTENSION, "joint_direction_multisource_extension546_for_624")
        if sha256_file(extension.OWNER500) != OWNER500_SHA256 or extension.OWNER500_SHA != OWNER500_SHA256:
            raise ValueError("546 extension's 500 owner pin does not match expected version")
        builder = load_module(BUILDER, "joint_point_target_builder545_for_624")
        original_loadmod = builder.loadmod
        face_provenance: dict[int, dict[str, Any]] = {}

        def load_with_multisource(path, name):
            module = original_loadmod(path, name)
            if Path(path).resolve() == Path(builder.SOURCE303).resolve():
                adapter589.install_source303_extension(module, extension, face_provenance)
            return module

        builder.loadmod = load_with_multisource
        original_vertex = builder.vertex_source_outward_normals
        original_depth_loader = builder.load_depth_covectors_357
        original_single_depth = builder.single_depth_row_for_node
        context["preflight"] = {"minimum_cosine": float(args.minimum_cosine),
                                "rows": [], "source_current_disagreement_is_diagnostic_only": True}

        def capture_source_and_current(candidate, source303, detail, base, orientation, target_rows):
            normals, provenance = original_vertex(candidate, source303, detail, base, orientation, target_rows)
            alignment = adapter589.current_source_alignment(
                candidate["points_world_m"], candidate["source_O_faces"], target_rows, normals,
                orientation.get("coarse_outward_sign", 0))
            current_normals = {}
            current_issues = []
            for row_record in alignment["rows"]:
                row = int(row_record["query_row"])
                if row_record.get("current_outward_normal_world_unit") is None:
                    current_issues.append({
                        "query_row": row,
                        "status": row_record.get("status"),
                        "reason": "current emitted-F32 normal unresolved/cancelled",
                        "diagnostic": row_record,
                    })
                else:
                    current_normals[row] = np.asarray(row_record["current_outward_normal_world_unit"], dtype=np.float64)
                provenance.setdefault(row, {})["current_source_outward_alignment_diagnostic"] = row_record
            context["current_source_alignment"] = alignment
            context["current_normal_unresolved_rows"] = current_issues
            context["source_normals"] = normals
            context["current_normals"] = current_normals
            context["inside_query_rows"] = [int(x) for x in target_rows]
            return normals, provenance

        def capture_depth_rows(*call_args, **call_kwargs):
            result = original_depth_loader(*call_args, **call_kwargs)
            context["depth_by_node"], context["depth_summary"] = result
            return result

        def preflight_all_rows_if_needed():
            if "joint_directions" in context:
                return
            rows = context.get("inside_query_rows")
            if rows is None or "depth_by_node" not in context:
                raise RuntimeError("joint direction preflight inputs are incomplete")
            try:
                directions, diagnostics = compute_joint_directions(
                    rows,
                    context["source_normals"],
                    context["current_normals"],
                    context["depth_by_node"],
                    angular_mod.nearest_joint_direction597,
                    minimum_cosine=float(args.minimum_cosine),
                    single_depth_row=original_single_depth,
                )
            except JointDirectionFailure as exc:
                context["preflight"]["rows"] = exc.records
                raise
            context["joint_directions"] = directions
            context["joint_direction_diagnostics"] = diagnostics
            context["preflight"]["status"] = "PASS_ALL_INSIDE_ROWS"
            context["preflight"]["rows"] = [diagnostics[int(row)] for row in rows]

        current_row: dict[str, int | None] = {"value": None}

        def record_single_depth_row(rows_for_node, node_row):
            row = int(node_row)
            current_row["value"] = row
            preflight_all_rows_if_needed()
            depth_row = original_single_depth(rows_for_node, row)
            context.setdefault("single_depth_rows", []).append({
                "query_row": row,
                "typed_id": depth_row.get("typed_id"),
                "source_row": depth_row.get("source_row"),
                "normal_world_unit_covector": np.asarray(
                    depth_row["normal_world_unit_covector"], dtype=np.float64).tolist(),
            })
            preflight_row = context["depth_by_node"][row][0]
            if (depth_row.get("typed_id") != preflight_row.get("typed_id")
                    or depth_row.get("source_row") != preflight_row.get("source_row")
                    or not np.array_equal(np.asarray(depth_row["normal_world_unit_covector"]),
                                          np.asarray(preflight_row["normal_world_unit_covector"]))):
                raise ValueError(f"depth-row identity changed between preflight and 545 builder for row {row}")
            return depth_row

        def joint_depth_maximizing_direction(source_outward_world_unit, depth_covector_world_unit, *, minimum_cosine=0.05):
            row = current_row["value"]
            if row is None or row not in context.get("joint_directions", {}):
                raise RuntimeError("545 direction selection has no preflighted query-row identity")
            if abs(float(minimum_cosine) - float(args.minimum_cosine)) > 1e-12:
                raise ValueError("545 depth cosine floor differs from 624 pinned floor")
            expected_source = np.asarray(context["source_normals"][row], dtype=np.float64)
            expected_depth = np.asarray(context["depth_by_node"][row][0]["normal_world_unit_covector"],
                                        dtype=np.float64)
            if not np.array_equal(np.asarray(source_outward_world_unit, dtype=np.float64), expected_source):
                raise ValueError(f"545 source normal differs from preflight for row {row}")
            if not np.array_equal(np.asarray(depth_covector_world_unit, dtype=np.float64), expected_depth):
                raise ValueError(f"545 depth covector differs from preflight for row {row}")
            d = context["joint_directions"][row]
            return d.copy(), dict(context["joint_direction_diagnostics"][row])

        builder.vertex_source_outward_normals = capture_source_and_current
        builder.load_depth_covectors_357 = capture_depth_rows
        builder.single_depth_row_for_node = record_single_depth_row
        builder.depth_maximizing_unit_direction = joint_depth_maximizing_direction
        builder.main([
            "--candidate", str(candidate_path),
            "--point-summary", str(point_summary_path),
            "--query-dir", str(query_dir),
            "--output-dir", str(output_dir),
            "--direction-mode", "depth-maximizing",
            "--point-depth-rows-npz", str(depth_npz),
            "--point-depth-rows-report", str(depth_report),
            "--minimum-depth-cosine", str(args.minimum_cosine),
            "--clearance-m", str(args.clearance_m),
        ])
        report_path = output_dir / "nearest-union-exits.json"
        npz_path = output_dir / "nearest-union-exits.npz"
        if not report_path.is_file() or not npz_path.is_file():
            raise RuntimeError("pinned 545 owner did not emit its complete report and NPZ")
        raw_report_bytes = report_path.read_bytes()
        raw_report_copy = output_dir / "nearest-union-exits.545-raw.json"
        raw_report_copy.write_bytes(raw_report_bytes)
        refinement_copy = output_dir / "refinement-report.585-input.json"
        refinement_bytes = refinement_report_path.read_bytes()
        refinement_copy.write_bytes(refinement_bytes)
        report = json.loads(raw_report_bytes)
        with np.load(npz_path, allow_pickle=False) as archive:
            query_rows = np.asarray(archive["query_rows"], dtype=np.int64)
            target_rows = report.get("targets")
            validate_row_identity(context["inside_query_rows"], target_rows, query_rows)
            validate_all185_records(target_rows)
            validation = adapter589.validate_545_depth_maximizing_report(report, archive,
                                                                          float(args.minimum_cosine))
            row_map = {int(item["query_row"]): item for item in target_rows}
            directions = np.asarray(archive["proposal_direction_world_unit"], dtype=np.float64)
            source_normals = np.asarray(archive["source_surface_normal_world_unit"], dtype=np.float64)
            depth_normals = np.asarray(archive["depth_covector_world_unit"], dtype=np.float64)
            current_normals = np.asarray([context["current_normals"][int(r)] for r in query_rows],
                                         dtype=np.float64)
            starts = np.asarray([row_map[int(r)]["start_world_f32_m"] for r in query_rows], dtype=np.float32)
            targets = np.asarray(archive["target_world_f32_m"], dtype=np.float32)
            displacement_validation = validate_emitted_f32_displacements(
                starts, targets, directions, source_normals, current_normals, depth_normals,
                float(args.minimum_cosine))
        report["raw_545_report"] = {
            "path": str(raw_report_copy),
            "sha256": sha256_file(raw_report_copy),
            "preserved_byte_for_byte": raw_report_copy.read_bytes() == raw_report_bytes,
        }
        report["current_source_outward_alignment_diagnostic_624"] = context["current_source_alignment"]
        report["joint_point_target_adapter_624"] = {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(__file__),
            "pinned_545_builder_path": str(BUILDER),
            "pinned_545_builder_sha256": BUILDER_SHA256,
            "pinned_546_extension_path": str(EXTENSION),
            "pinned_546_extension_sha256": EXTENSION_SHA256,
            "pinned_500_owner_path": str(OWNER500),
            "pinned_500_owner_sha256": OWNER500_SHA256,
            "pinned_589_adapter_path": str(ADAPTER589),
            "pinned_589_adapter_sha256": ADAPTER589_SHA256,
            "pinned_597_angular_helper_path": str(ANGULAR597),
            "pinned_597_angular_helper_sha256": ANGULAR597_SHA256,
            "direction_method": "maximum depth-reference cosine over the 597 two-cap feasible set; source and current each have cosine floor .05",
            "selected_depth_cosine_definition": "depth_covector_world_unit dot emitted proposal direction",
            "selected_current_cosine_definition": "current_outward_normal_world_unit dot emitted proposal direction",
            "current_source_disagreement": "retained as a diagnostic; it is not accepted as a source-normal substitute or used to waive either cosine floor",
            "all_inside_rows_joint_preflighted": True,
            "inside_query_rows": context["inside_query_rows"],
            "single_depth_rows_recorded_before_direction_selection": context["single_depth_rows"],
            "joint_direction_preflight": context["preflight"],
            "source_depth_direction_validation": validation,
            "emitted_f32_displacement_validation": displacement_validation,
            "exact_545_first_union_selector_and_all185_bank_owner_unchanged": True,
            "native_admission": False,
            "accepted_physical_steps": 0,
        }
        report["outputs"]["parent_prefix_node_count"] = lineage["parent_prefix_node_count"]
        report["outputs"]["parent_prefix_points_bytes_sha256"] = lineage["parent_prefix_points_bytes_sha256"]
        report["outputs"]["parent_candidate_sha256"] = lineage["parent_candidate_sha256"]
        report["outputs"]["refinement_report_path"] = str(refinement_copy)
        report["outputs"]["refinement_report_sha256"] = sha256_file(refinement_copy)
        report["outputs"]["refinement_report_preserved_byte_for_byte"] = (
            refinement_copy.read_bytes() == refinement_bytes)
        report["lineage_585"] = lineage
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        context["status"] = "PASS_JOINT_SOURCE_CURRENT_DEPTH_TARGETS"
        context["report_path"] = str(report_path)
        context["report_sha256"] = sha256_file(report_path)
        context["target_npz_path"] = str(npz_path)
        context["target_npz_sha256"] = sha256_file(npz_path)
        return context
    except Exception as exc:
        context.setdefault("status", "FAIL_CLOSED")
        context["error"] = f"{type(exc).__name__}: {exc}"
        if "preflight" in context:
            context["preflight"].setdefault("status", "FAIL_CLOSED")
        if output_dir.exists():
            # Preserve a raw 545 artifact when available, but never leave a
            # rejected result presented as an accepted joint-target report.
            report_path = output_dir / "nearest-union-exits.json"
            if report_path.is_file():
                try:
                    report = json.loads(report_path.read_text())
                    if report.get("status") == "PASS_FIRST_UNION_EXIT_TARGETS_ONLY":
                        report["status"] = "REJECTED_JOINT_POINT_TARGETS_624"
                    report["joint_point_target_adapter_624_failure"] = {
                        "status": "FAIL_CLOSED",
                        "error": context["error"],
                        "diagnostics": context.get("preflight", {}),
                    }
                    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                except Exception as rewrite_exc:
                    context["report_rewrite_error"] = f"{type(rewrite_exc).__name__}: {rewrite_exc}"
        if not sidecar.exists():
            _atomic_json(sidecar, context)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--point-summary", required=True, type=Path)
    parser.add_argument("--query-dir", required=True, type=Path)
    parser.add_argument("--depth-npz", required=True, type=Path)
    parser.add_argument("--depth-report", required=True, type=Path)
    parser.add_argument("--refinement-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-cosine", type=float, default=MINIMUM_COSINE)
    parser.add_argument("--clearance-m", type=float, default=5e-5)
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(json_plain(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
