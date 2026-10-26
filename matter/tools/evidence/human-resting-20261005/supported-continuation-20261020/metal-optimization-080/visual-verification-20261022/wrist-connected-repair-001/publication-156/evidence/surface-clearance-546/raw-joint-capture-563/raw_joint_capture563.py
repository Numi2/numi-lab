"""Capture successful joint LP directions for existing 549 component projection.

This is an inert construction diagnostic. It records the actual source/objective
receipt supplied by the caller, the exact emitted-coordinate basis, and each
successful pre-global-alpha LP delta. It does not run geometry checks or admit a
candidate.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
CAPTURE546_PATH = ROOT / "D/surface-clearance-546/raw-lp-direction-capture-546/capture_raw_lp_directions546.py"
CAPTURE546_SHA256 = "850893a0cc3271fee42e8342e8d73c786cc396eb2f1df4d267c28556c2ace073"
RAW_SCHEMA = "numi.human.raw_joint_lp_direction_capture563.v1"
PATCH_SCHEMA = "numi.human.raw_joint_lp_direction_patch563.v1"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load546():
    if sha(CAPTURE546_PATH) != CAPTURE546_SHA256:
        raise ValueError("pinned 546 raw-capture helper SHA mismatch")
    name = "capture_raw_lp_directions546_for563"
    spec = importlib.util.spec_from_file_location(name, CAPTURE546_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load pinned 546 capture helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_CAPTURE546 = _load546()


def _jsonable(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"receipt contains non-JSON value: {type(value).__name__}")


def validate_fit_receipt(receipt: dict) -> dict:
    """Require explicit caller-owned source, objective, goal, row, workload, policy pins."""
    if not isinstance(receipt, dict):
        raise ValueError("fit receipt must be a dict")
    required = (
        "fit_entry_path", "fit_entry_sha256", "objective_contract", "goal_labels",
        "matched_input_pins", "row_receipt", "workload_receipt", "policy_receipt", "delta_semantics",
    )
    missing = [key for key in required if key not in receipt]
    if missing:
        raise ValueError(f"fit receipt missing required provenance: {missing}")
    entry_path = Path(receipt["fit_entry_path"])
    expected = str(receipt["fit_entry_sha256"])
    if not entry_path.is_file() or sha(entry_path) != expected:
        raise ValueError("fit entry source hash does not match caller receipt")
    if receipt["delta_semantics"] != "absolute_base_relative_millimeters":
        raise ValueError("joint solver delta must be explicitly base-relative absolute millimeters")
    if not isinstance(receipt["objective_contract"], str) or not receipt["objective_contract"].strip():
        raise ValueError("actual objective contract label is required")
    goals = receipt["goal_labels"]
    if not isinstance(goals, (list, tuple)) or not goals or any(not isinstance(x, str) or not x.strip() for x in goals):
        raise ValueError("actual nonempty goal labels are required")
    for key in ("matched_input_pins", "row_receipt", "workload_receipt", "policy_receipt"):
        if not isinstance(receipt[key], dict) or not receipt[key]:
            raise ValueError(f"fit receipt {key} must be a nonempty dict")
    for role, pin in receipt["matched_input_pins"].items():
        if isinstance(pin, dict) and set(pin) == {"path", "sha256"}:
            path, digest = pin["path"], pin["sha256"]
        elif isinstance(pin, str):
            path, digest = role, pin
        else:
            raise ValueError("matched input pins must contain path/SHA records or path-to-SHA entries")
        if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("matched input pins must identify paths and SHA-256 strings")
        if any(c not in "0123456789abcdef" for c in digest.lower()):
            raise ValueError("invalid SHA-256 in matched input pins")
        pin_path = Path(path)
        if not pin_path.is_file() or sha(pin_path) != digest:
            raise ValueError(f"matched input source pin mismatch: {path}")
    return _jsonable(receipt)


def capture_basis(base_candidate_path, output_scope_rows, anchor_rows, control_parent_rows, *, basis_matrix=None):
    """Snapshot the exact full-output basis and node-row mapping before the LP call.

    ``basis_matrix`` is the node-level Q from ``full_basis``. For the existing
    560/561 full-coordinate adapter it is an identity injection from controls to
    output-scope rows. If omitted, that exact injection is derived from the
    recorded scope/control identities and explicitly labelled as derived.
    """
    base_path = Path(base_candidate_path).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    if sha(CAPTURE546_PATH) != CAPTURE546_SHA256:
        raise ValueError("pinned 546 capture helper changed")
    with np.load(base_path, allow_pickle=False) as archive:
        base_arrays = {k: np.asarray(archive[k]).copy() for k in archive.files}
    if "points_world_m" not in base_arrays:
        raise ValueError("base candidate lacks points_world_m")
    snapshot = _CAPTURE546.capture_basis_snapshot(
        base_arrays, output_scope_rows, anchor_rows, control_parent_rows
    )
    base = snapshot["base_points_world_m_f32"]
    declared_anchors = base_arrays.get("typed_coarse_anchor_output_rows_018")
    if declared_anchors is None or not np.array_equal(
        np.asarray(declared_anchors, dtype=np.int64), snapshot["anchor_rows"]
    ):
        raise ValueError("captured anchor rows do not match typed base anchors")
    scope = snapshot["output_scope_rows"]
    anchors = snapshot["anchor_rows"]
    controls = snapshot["control_parent_rows"]
    scope_positions = {int(node): i for i, node in enumerate(scope)}
    control_positions = np.asarray([scope_positions[int(node)] for node in controls], dtype=np.int64)
    anchor_scope_positions = np.asarray(
        [scope_positions[int(node)] for node in anchors if int(node) in scope_positions], dtype=np.int64
    )
    classified = np.zeros(len(scope), dtype=bool)
    classified[control_positions] = True
    classified[anchor_scope_positions] = True
    if not classified.all():
        raise ValueError("every output-scope row must be a control or fixed anchor")
    if basis_matrix is not None:
        try:
            from scipy import sparse
            matrix = sparse.csr_matrix(basis_matrix, dtype=np.float64)
        except Exception as exc:
            raise ValueError("basis_matrix must be convertible to finite CSR") from exc
        if matrix.shape != (len(scope), len(controls)) or not np.isfinite(matrix.data).all():
            raise ValueError("basis_matrix shape or values do not match output/control rows")
        expected = sparse.csr_matrix(
            (np.ones(len(controls)), (control_positions, np.arange(len(controls)))),
            shape=(len(scope), len(controls)),
        )
        diff = matrix - expected
        if diff.nnz and np.any(diff.data != 0):
            raise ValueError("captured full-coordinate Q is not the declared identity injection")
        basis = {
            "shape": list(matrix.shape),
            "data": matrix.data.copy(),
            "indices": matrix.indices.copy(),
            "indptr": matrix.indptr.copy(),
            "representation": "exact_csr_from_full_basis_hook",
        }
    else:
        # Derived CSR is valid only when every output row is either a control
        # or a fixed anchor, with controls exactly in their recorded positions.
        if len(scope) != len(controls) + len(anchor_scope_positions):
            raise ValueError("cannot derive basis CSR: scope has unclassified output rows")
        per_row = np.zeros(len(scope), dtype=np.int32)
        control_column_by_position = {}
        for column, row_position in enumerate(control_positions):
            per_row[row_position] = 1
            control_column_by_position[int(row_position)] = int(column)
        indices = np.asarray(
            [control_column_by_position[row] for row in sorted(control_column_by_position)],
            dtype=np.int32,
        )
        basis = {
            "shape": [int(len(scope)), int(len(controls))],
            "data": np.ones(len(controls), dtype=np.float64),
            "indices": indices,
            "indptr": np.concatenate(([0], np.cumsum(per_row, dtype=np.int32))),
            "representation": "derived_identity_injection_from_scope_control_mapping",
        }
    return {
        "base_candidate": str(base_path),
        "base_sha256": sha(base_path),
        "capture_helper_sha256": sha(Path(__file__)),
        "base_points_world_m_f32": base,
        "output_scope_rows": snapshot["output_scope_rows"],
        "anchor_rows": snapshot["anchor_rows"],
        "control_parent_rows": snapshot["control_parent_rows"],
        "control_scope_positions": control_positions,
        "anchor_scope_positions": anchor_scope_positions,
        "basis_csr": basis,
        "output_dimension": int(3 * len(scope)),
        "control_dimension": int(3 * len(controls)),
        "units": {"base_points": "meters_float32", "solver_delta": "millimeters"},
    }


def capture_solver_return(result, basis_token, *, fit_receipt, stage="joint_lp_raw_before_area_alpha", component=None, records=None):
    """Record the successful full-output LP return; return the original result object."""
    if records is None:
        raise ValueError("records list is required so the wrapper retains every LP return")
    receipt = validate_fit_receipt(fit_receipt)
    if not isinstance(basis_token, dict):
        raise ValueError("basis token is required")
    if basis_token.get("units", {}).get("solver_delta") != "millimeters":
        raise ValueError("joint LP delta unit must be explicitly millimeters")
    if basis_token.get("capture_helper_sha256") != sha(Path(__file__)):
        raise ValueError("563 capture helper changed after basis snapshot")
    before = len(records)
    _CAPTURE546.capture_successful_solver_result(result, basis_token, records)
    record = records[-1] if len(records) > before else None
    if record is None:
        raise AssertionError("546 capture helper did not append the solver return")
    record["stage"] = str(stage)
    record["component"] = None if component is None else int(component)
    record["fit_receipt"] = receipt
    if isinstance(result, dict):
        solver_fields = {k: v for k, v in result.items()
                         if k not in {"delta", "basis_coordinates", "normalized_slack"}}
        record["solver_receipt"] = _jsonable(solver_fields)
    if record.get("delta_captured"):
        expected = basis_token["output_dimension"]
        if record["delta_mm"].shape != (expected,):
            record["delta_captured"] = False
            record["status"] = "SUCCESSFUL_SOLVER_RETURN_WITH_WRONG_OUTPUT_DIMENSION"
            record.pop("delta_mm", None)
        elif len(basis_token["anchor_scope_positions"]):
            delta3 = record["delta_mm"].reshape((-1, 3))
            anchor_delta = delta3[basis_token["anchor_scope_positions"]]
            if np.max(np.abs(anchor_delta), initial=0.0) > 1e-12:
                record["delta_captured"] = False
                record["status"] = "SUCCESSFUL_SOLVER_RETURN_MOVES_FIXED_ANCHOR_BASIS_ROWS"
                record["raw_anchor_delta_max_mm"] = float(np.max(np.abs(anchor_delta), initial=0.0))
                record.pop("delta_mm", None)
        basis_coords = result.get("basis_coordinates") if isinstance(result, dict) else None
        if basis_coords is not None and record.get("delta_captured"):
            bc = np.asarray(basis_coords, dtype=np.float64)
            if bc.shape != (basis_token["control_dimension"],) or not np.isfinite(bc).all():
                record["delta_captured"] = False
                record["status"] = "SUCCESSFUL_SOLVER_RETURN_WITH_MALFORMED_BASIS_COORDINATES"
                record.pop("delta_mm", None)
            else:
                from scipy import sparse
                csr = basis_token["basis_csr"]
                Q = sparse.csr_matrix((csr["data"], csr["indices"], csr["indptr"]), shape=csr["shape"])
                expanded = sparse.kron(Q, sparse.eye(3, format="csr"), format="csr")
                reconstructed = np.asarray(expanded @ bc).reshape(-1)
                residual = float(np.max(np.abs(reconstructed - record["delta_mm"]), initial=0.0))
                record["basis_coordinate_reconstruction_residual_max_abs_mm"] = residual
                if residual > 1e-7:
                    record["delta_captured"] = False
                    record["status"] = "SUCCESSFUL_SOLVER_RETURN_DELTA_DOES_NOT_RECONSTRUCT_FROM_BASIS"
                    record.pop("delta_mm", None)
                else:
                    record["basis_coordinates"] = bc.copy()
        record["basis_snapshot"] = {
            "base_candidate": basis_token["base_candidate"],
            "base_sha256": basis_token["base_sha256"],
            "capture_helper_sha256": basis_token["capture_helper_sha256"],
            "output_scope_rows": basis_token["output_scope_rows"],
            "anchor_rows": basis_token["anchor_rows"],
            "control_parent_rows": basis_token["control_parent_rows"],
            "control_scope_positions": basis_token["control_scope_positions"],
            "anchor_scope_positions": basis_token["anchor_scope_positions"],
            "basis_csr": basis_token["basis_csr"],
            "output_dimension": basis_token["output_dimension"],
            "control_dimension": basis_token["control_dimension"],
            "units": basis_token["units"],
            "_base_points_world_m_f32": basis_token["base_points_world_m_f32"].copy(),
        }
    return result


def _write_basis(path: Path, snapshot: dict, delta_mm: np.ndarray, basis_coordinates=None):
    csr = snapshot["basis_csr"]
    arrays = {
        "output_scope_rows": np.asarray(snapshot["output_scope_rows"], dtype=np.int64),
        "anchor_rows": np.asarray(snapshot["anchor_rows"], dtype=np.int64),
        "control_parent_rows": np.asarray(snapshot["control_parent_rows"], dtype=np.int64),
        "control_scope_positions": np.asarray(snapshot["control_scope_positions"], dtype=np.int64),
        "anchor_scope_positions": np.asarray(snapshot["anchor_scope_positions"], dtype=np.int64),
        "basis_shape": np.asarray(csr["shape"], dtype=np.int64),
        "basis_data": np.asarray(csr["data"], dtype=np.float64),
        "basis_indices": np.asarray(csr["indices"], dtype=np.int64),
        "basis_indptr": np.asarray(csr["indptr"], dtype=np.int64),
        "delta_mm": np.asarray(delta_mm, dtype=np.float64),
        "base_points_world_m_f32": np.asarray(snapshot["_base_points_world_m_f32"], dtype=np.float32),
    }
    if basis_coordinates is not None:
        arrays["basis_coordinates"] = np.asarray(basis_coordinates, dtype=np.float64)
    np.savez_compressed(path, **arrays)


def finalize_capture(base_candidate_path, output_dir, records, *, fit_receipt):
    """Write 549-compatible raw proposal dirs for each successful captured LP return."""
    receipt = validate_fit_receipt(fit_receipt)
    base_path = Path(base_candidate_path).resolve()
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite raw joint capture: {out}")
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    base_digest = sha(base_path)
    with np.load(base_path, allow_pickle=False) as archive:
        base_arrays = {k: np.asarray(archive[k]).copy() for k in archive.files}
    base_points = np.asarray(base_arrays["points_world_m"])
    anchors_from_base = np.asarray(base_arrays["typed_coarse_anchor_output_rows_018"], dtype=np.int64)
    out.mkdir(parents=True)
    emitted_records = []
    successful = 0
    for call_index, record in enumerate(records, 1):
        if not record.get("delta_captured"):
            emitted_records.append({k: _jsonable(v) for k, v in record.items() if k not in ("delta_mm", "basis_snapshot", "basis_coordinates")})
            continue
        snapshot = record["basis_snapshot"]
        if snapshot["base_candidate"] != str(base_path) or snapshot["base_sha256"] != base_digest:
            raise ValueError("captured solver basis does not match final base candidate")
        if snapshot.get("capture_helper_sha256") != sha(Path(__file__)):
            raise ValueError("563 capture helper changed after basis snapshot")
        if "_base_points_world_m_f32" not in snapshot or not _CAPTURE546._same_f32_bits(
            base_points, snapshot["_base_points_world_m_f32"]
        ):
            raise ValueError("captured basis base points differ from pinned base candidate")
        scope = np.asarray(snapshot["output_scope_rows"], dtype=np.int64)
        anchor_rows = np.asarray(snapshot["anchor_rows"], dtype=np.int64)
        if not np.array_equal(anchor_rows, anchors_from_base):
            raise ValueError("captured anchors differ from base typed anchors")
        points, free_nodes, raw_anchor_delta_max_mm = _CAPTURE546.emitted_raw_points(
            base_points, scope, anchor_rows, record["delta_mm"]
        )
        payload = _CAPTURE546.candidate_payload_with_raw_points(base_arrays, points)
        proposal_dir = out / f"raw-direction-{call_index:03d}"
        proposal_dir.mkdir()
        candidate_path = proposal_dir / "candidate.npz"
        np.savez_compressed(candidate_path, **payload)
        basis_path = proposal_dir / "basis-and-delta.npz"
        _write_basis(basis_path, snapshot, record["delta_mm"], record.get("basis_coordinates"))
        delta_bytes = np.asarray(record["delta_mm"], dtype="<f8").reshape(-1).tobytes(order="C")
        proposal_receipt = record.get("fit_receipt", receipt)
        report = {
            "schema": PATCH_SCHEMA,
            "status": "RAW_JOINT_LP_DIRECTION_FOR_EXISTING_549_PROJECTION",
            # Required by component_area_backtracking549.py; the report remains
            # compatible without changing 549's parsing or gates.
            "base_candidate": str(base_path),
            "base_sha256": base_digest,
            "candidate_path": str(candidate_path.resolve()),
            "candidate_sha256": sha(candidate_path),
            "free_nodes": np.asarray(free_nodes, dtype=np.int64).tolist(),
            "raw_delta_units": "millimeters",
            "raw_delta_semantics": proposal_receipt["delta_semantics"],
            "raw_delta_sha256": hashlib.sha256(delta_bytes).hexdigest(),
            "basis_coordinate_reconstruction_residual_max_abs_mm": record.get(
                "basis_coordinate_reconstruction_residual_max_abs_mm"
            ),
            "basis_snapshot_path": str(basis_path.resolve()),
            "basis_snapshot_sha256": sha(basis_path),
            "solver_call_index": int(call_index),
            "solver_stage": str(record.get("stage", "unspecified_joint_lp_stage")),
            "solver_component": record.get("component"),
            "solver_status": record.get("solver_status"),
            "solver_message": record.get("solver_message", ""),
            "solver_objective": record.get("solver_objective"),
            "solver_wall_seconds": record.get("solver_wall_seconds"),
            "output_scope_count": int(len(scope)),
            "output_scope_rows": scope.tolist(),
            "fixed_anchor_rows": anchor_rows.tolist(),
            "control_parent_rows": np.asarray(snapshot["control_parent_rows"], dtype=np.int64).tolist(),
            "anchor_scope_positions": np.asarray(snapshot["anchor_scope_positions"], dtype=np.int64).tolist(),
            "output_dimension": int(snapshot["output_dimension"]),
            "control_dimension": int(snapshot["control_dimension"]),
            "units": snapshot["units"],
            "source_objective_receipt": proposal_receipt,
            "solver_receipt": record.get("solver_receipt", {}),
            "objective_source_labels_are_caller_supplied": True,
            "objective_is_joint_as_recorded_by_source_receipt": True,
            "candidate_arrays_copied_from_base_except_points_world_m": True,
            "outside_scope_bitwise_unchanged": True,
            "anchors_bitwise_unchanged": True,
            "raw_anchor_delta_max_mm": float(raw_anchor_delta_max_mm),
            "native_admission": False,
            "accepted_physical_steps": 0,
            "geometry_audits_run": False,
        }
        report_path = proposal_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        emitted_records.append({
            "solver_call_index": int(call_index),
            "status": "RAW_JOINT_LP_DIRECTION_WRITTEN_UNADMITTED",
            "candidate_path": str(candidate_path),
            "candidate_sha256": sha(candidate_path),
            "proposal_report_path": str(report_path),
            "proposal_report_sha256": sha(report_path),
            "solver_stage": report["solver_stage"],
            "solver_component": record.get("component"),
            "source_objective_receipt": proposal_receipt,
            "solver_receipt": record.get("solver_receipt", {}),
        })
        successful += 1
    summary = {
        "schema": RAW_SCHEMA,
        "status": "RAW_JOINT_LP_DIRECTIONS_CAPTURED_UNADMITTED" if successful else "NO_SUCCESSFUL_JOINT_LP_DELTA_CAPTURED",
        "fit_receipt": receipt,
        "capture_helper_path": str(Path(__file__).resolve()),
        "capture_helper_sha256": sha(Path(__file__)),
        "capture546_helper_path": str(CAPTURE546_PATH),
        "capture546_helper_sha256": CAPTURE546_SHA256,
        "base_candidate": str(base_path),
        "base_candidate_sha256": base_digest,
        "solver_return_count": int(len(records)),
        "successful_delta_count": int(successful),
        "solver_returns": emitted_records,
        "raw_proposals_are_pre_global_alpha_and_unadmitted": True,
        "native_admission": False,
        "accepted_physical_steps": 0,
        "geometry_audits_run": False,
    }
    summary_path = out / "report.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary_path, summary
