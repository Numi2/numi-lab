#!/usr/bin/env python3
"""Capture successful pre-area-backtrack LP deltas from the frozen 546 fit.

A capture is an unadmitted construction diagnostic. The original fit entry and
all owners remain frozen; exact geometry/admission is deferred to existing
quality projection and full checkers.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path("/Users/numi/Work/numi-human-connected-surface-069")
FIT = ROOT / "D/surface-clearance-546/run_barycentric_surface_fit546.py"
FIT_SHA256 = "16699bbe0162397edbac8ade4c9d7d8e29196221f492a45f9fa77ca762b9c45b"
ACTIVE = ROOT / "D/contained-clearance-545/component_lp_active545.py"
ACTIVE_SHA256 = "37c8a7e22f240be1a669e0f29712b0dfb068861fd89c9b6b2a8955c8ffdf39ed"
CAPTURE_ENV = "NUMI_RAW_LP_DIRECTION_CAPTURE_DIR_546"


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _same_f32_bits(a: np.ndarray, b: np.ndarray) -> bool:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return aa.shape == bb.shape and np.array_equal(aa.view(np.uint32), bb.view(np.uint32))


def capture_basis_snapshot(base_archive: Any, output_scope_rows, anchor_rows, control_rows):
    base = np.asarray(base_archive["points_world_m"])
    if base.dtype != np.float32 or base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("captured fit basis base points must be an N x 3 Float32 array")
    scope = np.asarray(output_scope_rows)
    anchors = np.asarray(anchor_rows)
    controls = np.asarray(control_rows)
    for label, value in (("output scope", scope), ("anchors", anchors), ("control rows", controls)):
        if value.ndim != 1 or value.dtype.kind not in "iu":
            raise ValueError(f"captured {label} must be a one-dimensional integer array")
    scope = scope.astype(np.int64, copy=True)
    anchors = anchors.astype(np.int64, copy=True)
    controls = controls.astype(np.int64, copy=True)
    if (len(np.unique(scope)) != len(scope) or np.any(scope < 0) or np.any(scope >= len(base))
            or len(np.unique(anchors)) != len(anchors) or np.any(anchors < 0) or np.any(anchors >= len(base))
            or len(np.unique(controls)) != len(controls) or np.any(controls < 0) or np.any(controls >= len(base))):
        raise ValueError("captured fit basis rows are duplicated or out of bounds")
    if not np.isin(controls, scope).all() or np.intersect1d(controls, anchors).size:
        raise ValueError("captured free controls escape output scope or touch anchors")
    return {
        "base_points_world_m_f32": base.copy(),
        "output_scope_rows": scope,
        "anchor_rows": anchors,
        "control_parent_rows": controls,
    }


def capture_successful_solver_result(result, basis_snapshot, records):
    """Copy a successful raw delta into diagnostics, returning the same result object."""
    if not isinstance(result, dict) or result.get("success") is not True:
        records.append({
            "status": "SOLVER_DID_NOT_RETURN_SUCCESS",
            "solver_status": result.get("status") if isinstance(result, dict) else None,
            "solver_message": result.get("message") if isinstance(result, dict) else "non-dict solver result",
            "delta_captured": False,
        })
        return result
    delta = result.get("delta")
    if basis_snapshot is None:
        records.append({
            "status": "SUCCESSFUL_SOLVER_RETURN_WITHOUT_BASIS_SNAPSHOT",
            "delta_captured": False,
        })
        return result
    try:
        delta = np.asarray(delta, dtype=np.float64)
    except (TypeError, ValueError):
        delta = np.empty(0, dtype=np.float64)
    scope = np.asarray(basis_snapshot["output_scope_rows"], dtype=np.int64)
    if delta.shape != (3 * len(scope),) or not np.isfinite(delta).all():
        records.append({
            "status": "SUCCESSFUL_SOLVER_RETURN_WITH_MALFORMED_DELTA",
            "delta_shape": list(delta.shape),
            "expected_shape": [3 * len(scope)],
            "delta_captured": False,
        })
        return result
    records.append({
        "status": "SUCCESSFUL_RAW_LP_DELTA_CAPTURED",
        "delta_mm": delta.copy(),
        "basis_snapshot": basis_snapshot,
        "solver_status": int(result.get("status", 0)),
        "solver_message": str(result.get("message", "")),
        "solver_objective": result.get("objective"),
        "solver_wall_seconds": result.get("wall_seconds"),
        "delta_shape": list(delta.shape),
        "delta_captured": True,
    })
    return result


def emitted_raw_points(base_points_world_m_f32, output_scope_rows, anchor_rows, delta_mm):
    """Emit scope coordinates as Float32 after converting the returned mm delta."""
    base = np.asarray(base_points_world_m_f32)
    if base.dtype != np.float32 or base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("base points must be an N x 3 Float32 array")
    scope = np.asarray(output_scope_rows)
    anchors = np.asarray(anchor_rows)
    if scope.ndim != 1 or scope.dtype.kind not in "iu":
        raise ValueError("output scope must be a one-dimensional integer array")
    if anchors.ndim != 1 or anchors.dtype.kind not in "iu":
        raise ValueError("anchors must be a one-dimensional integer array")
    scope = scope.astype(np.int64, copy=False)
    anchors = anchors.astype(np.int64, copy=False)
    if len(np.unique(scope)) != len(scope) or np.any(scope < 0) or np.any(scope >= len(base)):
        raise ValueError("output scope rows are duplicated or out of bounds")
    if len(np.unique(anchors)) != len(anchors) or np.any(anchors < 0) or np.any(anchors >= len(base)):
        raise ValueError("anchor rows are duplicated or out of bounds")
    delta = np.asarray(delta_mm, dtype=np.float64)
    if delta.shape != (3 * len(scope),) or not np.isfinite(delta).all():
        raise ValueError("raw LP delta must be finite and match output scope in millimeters")
    delta = delta.reshape((len(scope), 3))
    scope_anchor_mask = np.isin(scope, anchors)
    if np.any(delta[scope_anchor_mask] != 0.0):
        raise ValueError("raw LP delta moves an anchor row")
    emitted = base.copy()
    emitted[scope] = (base[scope].astype(np.float64) + delta / 1000.0).astype(np.float32)
    if not _same_f32_bits(emitted[np.setdiff1d(np.arange(len(base)), scope)], base[np.setdiff1d(np.arange(len(base)), scope)]):
        raise AssertionError("raw LP emission changed rows outside captured output scope")
    if not _same_f32_bits(emitted[anchors], base[anchors]):
        raise AssertionError("raw LP emission changed fixed anchor coordinates")
    free_nodes = scope[~scope_anchor_mask]
    return emitted, free_nodes


def candidate_payload_with_raw_points(base_arrays, emitted_points):
    points = np.asarray(base_arrays.get("points_world_m"))
    emitted = np.asarray(emitted_points)
    if points.dtype != np.float32 or emitted.dtype != np.float32 or emitted.shape != points.shape:
        raise ValueError("raw candidate points must match base Float32 point array")
    if set(base_arrays) == set() or "source_O_faces" not in base_arrays:
        raise ValueError("base candidate payload lacks source_O_faces")
    payload = {key: np.asarray(value).copy() for key, value in base_arrays.items()}
    payload["points_world_m"] = emitted.copy()
    return payload


def _load_original_fit_source():
    if sha(FIT) != FIT_SHA256:
        raise ValueError("frozen 546 fit entry SHA mismatch")
    source = FIT.read_text()
    solver_needle = "    m.solve_by_components=active.solve_by_components;receipt['active_component_adapter_source_sha256']=sha(extra)\n"
    solver_hook = """    original_active_solve=active.solve_by_components
    def capture_active_solve(*args,**kwargs):
     result=original_active_solve(*args,**kwargs)
     CAPTURE["solver_return_count"] += 1
     basis_snapshot=CAPTURE["basis_snapshots"][-1] if CAPTURE["basis_snapshots"] else None
     capture_successful_solver_result(result,basis_snapshot,CAPTURE["solver_returns"])
     return result
    active.solve_by_components=capture_active_solve
    m.solve_by_components=active.solve_by_components;receipt['active_component_adapter_source_sha256']=sha(extra)
"""
    basis_needle = "     return Q,scope[free],br\n"
    basis_hook = """     CAPTURE["basis_snapshots"].append(capture_basis_snapshot(base,scope,anchors,scope[free]))
     return Q,scope[free],br
"""
    if source.count(solver_needle) != 1 or source.count(basis_needle) != 1:
        raise ValueError("frozen fit hook locations differ from pinned source")
    patched = source.replace(solver_needle, solver_hook).replace(basis_needle, basis_hook)
    return patched


def _cli_path(flag):
    try:
        return Path(sys.argv[sys.argv.index(flag) + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise ValueError(f"capture wrapper requires {flag} in original 546 fit argv") from exc


def _finalize_capture(capture_dir, base_path, records, capture):
    capture_dir.mkdir(parents=True, exist_ok=False)
    base_sha = sha(base_path)
    with np.load(base_path, allow_pickle=False) as z:
        base_arrays = {k: np.asarray(z[k]).copy() for k in z.files}
    base_points = np.asarray(base_arrays["points_world_m"])
    anchors = np.asarray(base_arrays["typed_coarse_anchor_output_rows_018"], dtype=np.int64)
    emitted_records = []
    for call_index, record in enumerate(records, 1):
        if not record.get("delta_captured"):
            emitted_records.append({k: v for k, v in record.items() if k != "delta_mm" and k != "basis_snapshot"})
            continue
        basis = record["basis_snapshot"]
        scope = np.asarray(basis["output_scope_rows"], dtype=np.int64)
        anchor_rows = np.asarray(basis["anchor_rows"], dtype=np.int64)
        if not np.array_equal(anchor_rows, anchors):
            raise ValueError("captured basis anchors differ from typed base candidate anchors")
        if not _same_f32_bits(base_points, basis["base_points_world_m_f32"]):
            raise ValueError("captured basis points differ from pinned base candidate")
        points, free_nodes = emitted_raw_points(base_points, scope, anchor_rows, record["delta_mm"])
        payload = candidate_payload_with_raw_points(base_arrays, points)
        subdir = capture_dir / f"raw-direction-{call_index:03d}"
        subdir.mkdir(parents=False, exist_ok=False)
        candidate_path = subdir / "candidate.npz"
        np.savez_compressed(candidate_path, **payload)
        emitted_records.append({
            "solver_call_index": call_index,
            "status": "RAW_LP_DIRECTION_WRITTEN_UNADMITTED",
            "solver_status": record["solver_status"],
            "solver_message": record["solver_message"],
            "solver_objective": record["solver_objective"],
            "solver_wall_seconds": record["solver_wall_seconds"],
            "delta_units": "millimeters",
            "delta_shape": record["delta_shape"],
            "candidate_path": str(candidate_path),
            "candidate_sha256": sha(candidate_path),
            "output_scope_count": int(len(scope)),
            "output_scope_rows": scope.tolist(),
            "free_nodes": free_nodes.tolist(),
            "fixed_anchor_count": int(len(anchor_rows)),
            "fixed_anchor_rows": anchor_rows.tolist(),
            "outside_scope_bitwise_unchanged": True,
            "anchors_bitwise_unchanged": True,
            "candidate_arrays_copied_from_base_except_points_world_m": True,
            "quality_projector_owner": str(ROOT / "D/contained-clearance-545/project_patch_quality545.py"),
        })
    successful = sum(bool(x.get("delta_captured")) for x in records)
    report = {
        "schema": "numi.human.raw_lp_direction_capture546.v1",
        "status": "RAW_LP_DIRECTIONS_CAPTURED_UNADMITTED" if successful else "NO_SUCCESSFUL_LP_DELTA_CAPTURED",
        "fit_entry_path": str(FIT),
        "fit_entry_sha256": FIT_SHA256,
        "capture_wrapper_path": str(Path(__file__).resolve()),
        "capture_wrapper_sha256": sha(Path(__file__)),
        "active_solver_path": str(ACTIVE),
        "active_solver_sha256": ACTIVE_SHA256,
        "base_candidate": str(base_path),
        "base_candidate_sha256": base_sha,
        "fit_output_dir": str(_cli_path("--output")),
        "solver_return_count": int(capture["solver_return_count"]),
        "successful_delta_count": successful,
        "basis_snapshot_count": len(capture["basis_snapshots"]),
        "solver_returns": emitted_records,
        "semantics": "Each candidate copies the pinned base NPZ arrays and replaces only points_world_m with the solver's successful pre-area-backtrack delta, in millimeters, emitted as Float32 over the captured output scope. It is a raw construction direction for existing 545 quality projection, not an accepted fit.",
        "unchanged_fit_solver_and_downstream_selection": True,
        "geometry_audits_run": False,
        "native_admission": False,
        "accepted_physical_steps": 0,
    }
    report_path = capture_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report_path, report


def main():
    if sha(FIT) != FIT_SHA256 or sha(ACTIVE) != ACTIVE_SHA256:
        raise ValueError("pinned fit or active component solver changed")
    raw_capture_dir = Path(os.environ[CAPTURE_ENV]).resolve() if os.environ.get(CAPTURE_ENV) else None
    if raw_capture_dir is None:
        raise ValueError(f"set {CAPTURE_ENV} to a new capture directory")
    if raw_capture_dir.exists():
        raise FileExistsError(f"refusing to overwrite raw direction capture: {raw_capture_dir}")
    base_path = _cli_path("--base-candidate")
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    capture = {"basis_snapshots": [], "solver_returns": [], "solver_return_count": 0}
    source = _load_original_fit_source()
    fit_globals = {
        "__name__": "_frozen_fit546_with_inert_raw_delta_capture",
        "__file__": str(FIT),
        "__package__": None,
        "CAPTURE": capture,
        "capture_basis_snapshot": capture_basis_snapshot,
        "capture_successful_solver_result": capture_successful_solver_result,
    }
    sys.path.insert(0, str(FIT.parent))
    exec(compile(source, str(FIT), "exec"), fit_globals)
    fit_error = None
    try:
        fit_globals["main"]()
    except BaseException as exc:
        fit_error = exc
    report_path, report = _finalize_capture(raw_capture_dir, base_path, capture["solver_returns"], capture)
    if fit_error is not None:
        print(json.dumps({"capture_report": str(report_path), "capture_status": report["status"],
                          "fit_exception_type": type(fit_error).__name__, "fit_exception": str(fit_error)}, indent=2))
        raise fit_error
    print(json.dumps({"capture_report": str(report_path), "capture_status": report["status"],
                      "successful_delta_count": report["successful_delta_count"],
                      "solver_return_count": report["solver_return_count"],
                      "native_admission": False, "accepted_physical_steps": 0}, indent=2))


if __name__ == "__main__":
    main()

