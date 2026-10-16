#!/usr/bin/env python3
"""Fail-closed analysis and planning for full-stage shipping-box exports.

This module reads retained Matter probe outputs only. It never launches physics,
repairs a failed run, or turns a constitutive degradation index into geometric
crack/debond area. Every report binds the exact input files it inspected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "numi.cardboard.stage-resolution-analysis.v1"
PLAN_SCHEMA = "numi.cardboard.stage-resolution-plan.v1"
STAGE_FILES = (
    "manifest.json", "result.json", "box-native-result.json",
    "box-native-steps.csv", "box-actions.csv", "box-score-angles.csv",
)
OPTIONAL_FILES = (
    "shipping-compression.csv", "shipping-contact-observations.csv",
    "shipping-gravity-transfer.csv", "shipping-impact-cases.csv",
    "shipping-impact-dynamics.csv", "shipping-history-observables.csv",
    "shipping-compression-nominal-curve.csv",
    "box-grip-reactions.csv", "box-score-springback.csv",
)
AXES = ("mesh_nx", "mesh_ny", "mesh_thickness", "time_step", "score_depth",
        "score_speed", "impact_time_step")


class StageStudyError(Exception):
    pass


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StageStudyError(f"{path.name} must contain a JSON object")
    return value


def _rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise StageStudyError(f"{path.name} has no CSV header")
        return list(reader.fieldnames), list(reader)


def _finite(row: dict[str, str], key: str, where: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise StageStudyError(f"{where}: missing or nonnumeric {key}") from exc
    if not math.isfinite(value):
        raise StageStudyError(f"{where}: nonfinite {key}")
    return value


def _bool(value: Any) -> bool:
    return value is True or value == 1 or value == "1" or value == "true"


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


@dataclass
class StageRun:
    path: Path
    manifest: dict[str, Any]
    result: dict[str, Any] | None
    native: dict[str, Any] | None
    files: dict[str, str]
    source_bindings: dict[str, dict[str, Any]]
    admission: dict[str, Any]
    steps: list[dict[str, str]]
    actions: list[dict[str, str]]
    score_angles: list[dict[str, str]]
    optional_rows: dict[str, list[dict[str, str]]]


def load_run(directory: Path) -> StageRun:
    """Read one probe output directory and fail closed without losing diagnosis."""
    path = directory.expanduser().resolve()
    files: dict[str, str] = {}
    reasons: list[str] = []
    parsed: dict[str, dict[str, Any]] = {}
    csv_data: dict[str, list[dict[str, str]]] = {}
    missing = []
    for name in STAGE_FILES:
        source = path / name
        if not source.is_file():
            missing.append(name)
            continue
        raw = source.read_bytes()
        files[name] = _sha(raw)
        if name.endswith(".json"):
            try:
                parsed[name] = _read_json(source)
            except (OSError, json.JSONDecodeError, StageStudyError) as exc:
                reasons.append(f"invalid_{name}:{exc}")
        else:
            try:
                _, csv_data[name] = _rows(source)
            except (OSError, csv.Error, StageStudyError) as exc:
                reasons.append(f"invalid_{name}:{exc}")
    if "manifest.json" not in parsed:
        reasons.append("missing_or_invalid_manifest")
    else:
        manifest = parsed["manifest.json"]
        if manifest.get("specimen") != "fefco0201_flat_blank":
            reasons.append("not_a_shipping_box_export")
    for name in OPTIONAL_FILES:
        source = path / name
        if source.is_file():
            raw = source.read_bytes()
            files[name] = _sha(raw)
            try:
                _, csv_data[name] = _rows(source)
            except (OSError, csv.Error, StageStudyError) as exc:
                reasons.append(f"invalid_{name}:{exc}")

    manifest = parsed.get("manifest.json", {})
    result = parsed.get("result.json")
    native = parsed.get("box-native-result.json")
    steps = csv_data.get("box-native-steps.csv", [])
    actions = csv_data.get("box-actions.csv", [])
    score_angles = csv_data.get("box-score-angles.csv", [])
    optional_rows = {name: csv_data.get(name, []) for name in OPTIONAL_FILES}

    source_bindings: dict[str, dict[str, Any]] = {}
    source_errors: list[str] = []
    material_table = manifest.get("materials", {})
    if not isinstance(material_table, dict):
        material_table = {}
        source_errors.append("invalid_material_source_table")
    for role in ("liner", "medium", "glue"):
        source_path = material_table.get(f"{role}_path")
        expected_sha = material_table.get(f"{role}_sha256")
        if not isinstance(source_path, str) or not source_path or not isinstance(expected_sha, str):
            source_errors.append(f"missing_material_source_identity:{role}")
            source_bindings[role] = {"path": source_path, "expected_sha256": expected_sha,
                                     "status": "missing_path_or_digest"}
            continue
        resolved = Path(source_path).expanduser().resolve()
        if not resolved.is_file():
            source_errors.append(f"material_source_file_unavailable:{role}")
            source_bindings[role] = {"path": str(resolved), "expected_sha256": expected_sha,
                                     "status": "source_file_unavailable"}
            continue
        actual_sha = _sha(resolved.read_bytes())
        files[f"source_material_{role}"] = actual_sha
        state = "verified" if actual_sha == expected_sha else "digest_mismatch"
        source_bindings[role] = {"path": str(resolved), "expected_sha256": expected_sha,
                                 "actual_sha256": actual_sha, "status": state}
        if state != "verified":
            source_errors.append(f"material_source_digest_mismatch:{role}")

    for feature_name, path_key, digest_key in (
        ("retained_matrix_damage_overlay", "liner_parameter_file", "liner_parameter_file_sha256"),
        ("retained_matrix_damage_overlay", "medium_parameter_file", "medium_parameter_file_sha256"),
        ("corrugator_starch_interface", "parameter_file", "parameter_file_sha256"),
        ("shipping_hma_closure", "parameter_file", "parameter_file_sha256"),
    ):
        feature = manifest.get(feature_name, {})
        if not isinstance(feature, dict) or feature.get("enabled") is not True:
            continue
        source_path, expected_sha = feature.get(path_key), feature.get(digest_key)
        binding_key = f"{feature_name}:{path_key}"
        if not isinstance(source_path, str) or not source_path or not isinstance(expected_sha, str) or not expected_sha:
            source_errors.append(f"missing_enabled_law_source_identity:{binding_key}")
            source_bindings[binding_key] = {"path": source_path, "expected_sha256": expected_sha,
                                            "status": "missing_path_or_digest"}
            continue
        resolved = Path(source_path).expanduser().resolve()
        if not resolved.is_file():
            source_errors.append(f"enabled_law_source_file_unavailable:{binding_key}")
            source_bindings[binding_key] = {"path": str(resolved), "expected_sha256": expected_sha,
                                            "status": "source_file_unavailable"}
            continue
        actual_sha = _sha(resolved.read_bytes())
        files[f"source_{binding_key}"] = actual_sha
        state = "verified" if actual_sha == expected_sha else "digest_mismatch"
        source_bindings[binding_key] = {"path": str(resolved), "expected_sha256": expected_sha,
                                        "actual_sha256": actual_sha, "status": state}
        if state != "verified":
            source_errors.append(f"enabled_law_source_digest_mismatch:{binding_key}")

    if missing:
        reasons.append("missing_required_files:" + ",".join(missing))
    reasons.extend(source_errors)
    if result is not None and result.get("status") == "native_preflight_only_no_physics_step":
        disposition = "preflight_only"
        reasons.append("preflight_contains_no_native_physics_steps")
    elif not steps:
        disposition = "incomplete_missing_step_history"
        reasons.append("no_native_step_rows")
    else:
        disposition = "incomplete_or_rejected"

    preflight_only = result is not None and result.get("status") == "native_preflight_only_no_physics_step"
    if result is not None and native is not None:
        native_status = str(native.get("status", ""))
        if native_status.startswith("failed") or result.get("status") in (
            "failed_native", "failed_prefix", "shape_fit_rejected",
        ):
            disposition = "failed_prefix"
            reasons.append("native_result_reports_failure")
        elif (native_status.startswith(("completed_native_", "completed_released_assembly")) or
              native_status == "stage_cutoff_held_boundary_state"):
            disposition = "completed_candidate"
        else:
            reasons.append(f"native_result_not_complete:{native_status or 'missing'}")

    step_fields = set(steps[0]) if steps else set()
    required_step_fields = {
        "native_step", "stage", "accepted", "status_env0", "fgmres_env0",
        "certificate_valid_env0", "certificate_residual_env0",
        "certificate_correction_env0", "certificate_volume_env0",
        "certificate_pressure_env0", "simulation_time_s", "dt_s",
    }
    missing_cert = sorted(required_step_fields - step_fields)
    if steps and missing_cert:
        disposition = "inconclusive_missing_certificates"
        reasons.append("step_csv_missing_fields:" + ",".join(missing_cert))
    if steps and not missing_cert:
        solver = manifest.get("solver", {})
        tolerances = {
            "certificate_residual_env0": solver.get("relative_residual_tolerance"),
            "certificate_correction_env0": solver.get("relative_residual_tolerance"),
            "certificate_volume_env0": solver.get("volume_tolerance"),
            "certificate_pressure_env0": solver.get("pressure_tolerance"),
        }
        expected_steps = 0
        failed_step_seen = False
        previous_index = -1
        previous_time = -math.inf
        for ordinal, row in enumerate(steps):
            label = f"box-native-steps.csv row {ordinal + 2}"
            try:
                index = int(row["native_step"])
                accepted = row["accepted"] == "1"
                status = int(row["status_env0"])
                status1_text = row.get("status_env1", "NA")
                cert_valid = int(row["certificate_valid_env0"])
                sim_time = _finite(row, "simulation_time_s", label)
                dt = _finite(row, "dt_s", label)
                if index != ordinal:
                    disposition = "inconclusive_invalid_step_row"
                    reasons.append(f"native_step_index_not_zero_based_contiguous:{index}:expected:{ordinal}")
                if not accepted:
                    disposition = "failed_prefix"
                    failed_step_seen = True
                    reasons.append(f"rejected_native_step:{index}")
                if status != 0:
                    disposition = "failed_prefix"
                    failed_step_seen = True
                    reasons.append(f"nonzero_native_status:{index}:{status}")
                if status1_text not in ("", "NA") and int(status1_text) != 0:
                    disposition = "failed_prefix"
                    failed_step_seen = True
                    reasons.append(f"nonzero_native_status_env1:{index}:{status1_text}")
                if cert_valid != 1:
                    disposition = "inconclusive_invalid_certificate"
                    reasons.append(f"invalid_native_certificate:{index}")
                if dt <= 0.0 or (previous_index >= 0 and sim_time <= previous_time):
                    disposition = "inconclusive_invalid_step_row"
                    reasons.append(f"invalid_native_clock:{index}")
                previous_index = index
                previous_time = sim_time
                for field, tolerance in tolerances.items():
                    value = _finite(row, field, label)
                    if not isinstance(tolerance, (int, float)) or not math.isfinite(float(tolerance)):
                        reasons.append(f"manifest_missing_tolerance:{field}")
                        continue
                    if value < 0.0 or value > float(tolerance) * (1.0 + 1e-6):
                        disposition = "inconclusive_invalid_certificate"
                        reasons.append(f"certificate_outside_tolerance:{index}:{field}")
                expected_steps += 1
            except (ValueError, KeyError, StageStudyError) as exc:
                disposition = "inconclusive_invalid_step_row"
                reasons.append(f"invalid_step_row:{ordinal}:{exc}")
        if native is not None:
            attempted = native.get("native_steps_attempted")
            accepted = native.get("native_steps_accepted")
            if attempted != len(steps) or accepted != len(steps):
                disposition = "failed_prefix"
                reasons.append("native_step_count_does_not_match_all_accepted_rows")
        if failed_step_seen:
            disposition = "failed_prefix"

    if result is not None and native is not None:
        if int(result.get("compiled_world_fingerprint", -1)) != int(native.get("compiled_world_fingerprint", -2)):
            if disposition != "failed_prefix":
                disposition = "inconclusive_fingerprint_mismatch"
            reasons.append("result_and_native_fingerprints_differ")

    # Failure evidence takes precedence over missing modern observables. A failed
    # prefix must stay classified as a failed prefix; a clean-looking candidate
    # is not admitted until every declared source file is present and hash-bound.
    explicit_native_failure = bool(
        (result is not None and str(result.get("status", "")).startswith("failed")) or
        (native is not None and str(native.get("status", "")).startswith("failed"))
    )
    for row in steps:
        try:
            if row.get("accepted") != "1" or int(row.get("status_env0", "0")) != 0:
                explicit_native_failure = True
            status1 = row.get("status_env1", "NA")
            if status1 not in (None, "", "NA") and int(status1) != 0:
                explicit_native_failure = True
        except ValueError:
            explicit_native_failure = True
    if native is not None and steps:
        try:
            native_attempted = int(native.get("native_steps_attempted", -1))
            native_accepted = int(native.get("native_steps_accepted", -1))
            row_accepted = sum(row.get("accepted") == "1" for row in steps)
            if native_attempted != len(steps) or native_accepted != row_accepted:
                explicit_native_failure = True
        except (TypeError, ValueError):
            explicit_native_failure = True
    if explicit_native_failure:
        disposition = "failed_prefix"
    elif preflight_only:
        disposition = "preflight_only"
    elif disposition != "failed_prefix" and source_errors:
        disposition = "inconclusive_source_binding"

    return StageRun(path, manifest, result, native, files, source_bindings,
                    {"status": disposition, "reasons": reasons},
                    steps, actions, score_angles, optional_rows)


def _material_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    materials = manifest.get("materials", {})
    keys = ("liner_sha256", "medium_sha256", "glue_sha256")
    result = {key: materials.get(key) for key in keys}
    allowed = {
        "retained_matrix_damage_overlay": (
            "enabled", "candidate_only", "source_grade_calibrated",
            "liner_parameter_file_sha256", "liner_set_id", "liner_evidence_class",
            "medium_parameter_file_sha256", "medium_set_id", "medium_evidence_class",
            "parameter_contract",
        ),
        "corrugator_starch_interface": (
            "enabled", "parameter_file_sha256", "source_class", "activation_event",
            "activation_gap_tolerance_m", "material_grade_calibrated",
        ),
        "shipping_hma_closure": (
            "enabled", "parameter_file_sha256", "source_class", "source_citation",
            "transfer_uncertainty_fraction", "source_grade_calibrated",
        ),
    }
    for key, fields in allowed.items():
        feature = manifest.get(key)
        if isinstance(feature, dict):
            result[key] = {name: feature.get(name) for name in fields}
    return result


def physical_identity(run: StageRun) -> dict[str, Any]:
    manifest = run.manifest
    geometry = manifest.get("geometry", {})
    solver = manifest.get("solver", {})
    tooling = manifest.get("tooling", {})
    identity = {
        "specimen": manifest.get("specimen"),
        "preset": manifest.get("preset"),
        "geometry": {key: geometry.get(key) for key in (
            "source_dimensions", "length_m", "width_m", "total_height_m", "pitch_m",
            "liner_thickness_m", "medium_normal_thickness_m", "medium_centerline_height_m",
            "medium_present", "glue_minimum_gap_m", "bond_width_m", "upper_glue_gap_m",
            "upper_bond_width_m", "contact_tie_assumption",
        )},
        "materials": _material_identity(manifest),
        "tooling_physics": {key: tooling.get(key) for key in (
            "mode", "punch_radius_m", "channel_width_m", "support_rail_width_m",
            "die_to_channel_clearance_calipers", "initial_nose_clearance_m",
            "initial_anvil_clearance_m",
        )},
        "score_command": manifest.get("shipping_score_command"),
        "gravity": solver.get("gravity_m_s2"),
        "solver_controls": {key: solver.get(key) for key in (
            "deformable_self_contact", "contact_slop_m", "deformation_gradient",
            "local_material_newton_iterations", "newton_iteration_budget", "fgmres_restart",
            "fgmres_iteration_budget", "fgmres_preconditioner", "line_search_steps",
            "relative_residual_tolerance", "volume_tolerance", "pressure_tolerance",
            "maximum_rate_exponent",
        )},
    }
    return identity


def _axis_values(run: StageRun, axis: str) -> Any:
    manifest = run.manifest
    mesh = manifest.get("mesh", {})
    solver = manifest.get("solver", {})
    score = manifest.get("shipping_score_command", {})
    result = run.native or {}
    if axis == "mesh_nx": return mesh.get("nx_per_pitch")
    if axis == "mesh_ny": return mesh.get("ny")
    if axis == "mesh_thickness": return mesh.get("thickness_slices")
    if axis == "time_step": return solver.get("dt_s")
    if axis == "score_depth": return score.get("selected_depth_ratio_of_caliper")
    if axis == "score_speed": return score.get("tool_speed_m_per_s")
    if axis == "impact_time_step": return result.get("shipping_impact_timestep_levels_s")
    raise StageStudyError(f"unknown comparison axis: {axis}")


def _varying_identity(identity: dict[str, Any], axis: str) -> dict[str, Any]:
    result = json.loads(_canonical(identity))
    if axis in ("mesh_nx", "mesh_ny", "mesh_thickness"):
        pass  # mesh lives outside physical identity by design
    elif axis == "time_step":
        result["solver_controls"].pop("time_step", None)
        command = result.get("score_command")
        if isinstance(command, dict):
            # The probe's score-command digest includes the requested solver dt
            # and compiled-program fingerprint. Those are expected to change
            # in a timestep study; tool speed, depth, and path stay physical.
            command.pop("source_command_sha256", None)
    elif axis == "score_depth":
        command = result.get("score_command")
        if isinstance(command, dict):
            command.pop("selected_depth_ratio_of_caliper", None)
            command.pop("source_command_sha256", None)
    elif axis == "score_speed":
        command = result.get("score_command")
        if isinstance(command, dict):
            command.pop("tool_speed_m_per_s", None)
            command.pop("source_command_sha256", None)
    elif axis == "impact_time_step":
        pass
    return result


def _scope_gate(run: StageRun, scope: str, *, require_rate_pass: bool = True) -> list[str]:
    reasons: list[str] = []
    if run.admission["status"] != "completed_candidate":
        reasons.append("run_not_admitted_as_complete_native_candidate")
        return reasons
    native = run.native or {}
    native_status = str(native.get("status", ""))
    through_stage = str(native.get("shipping_through_stage", ""))
    if scope == "scoring":
        if native.get("score_line_count") != 12:
            reasons.append("twelve_score_lines_not_reported")
        if native.get("score_tool_and_anvil_withdrawn_before_folding") is not True:
            reasons.append("score_tools_not_withdrawn_before_folding")
        if _safe_int(native.get("shipping_score_approach_steps_accepted")) <= 0:
            reasons.append("no_accepted_scoring_approach_steps_reported")
        if _safe_int(native.get("shipping_score_retract_steps_accepted")) <= 0:
            reasons.append("no_accepted_scoring_withdrawal_steps_reported")
        if native_status == "stage_cutoff_held_boundary_state" and through_stage != "scoring":
            reasons.append("stage_cutoff_is_not_after_scoring")
        elif not native_status.startswith("completed_native_") and native_status != "stage_cutoff_held_boundary_state":
            reasons.append("scoring_stage_not_complete")
        return reasons
    if scope in ("assembly", "compression", "impact", "all"):
        required = {
            "released_assembly_settled": True,
            "released_clamp_zero_active_gate_passed": True,
            "released_assembly_temporary_tool_contact_free": True,
            "released_assembly_simulation_process_complete": True,
        }
        for key, expected in required.items():
            if native.get(key) is not expected:
                reasons.append(f"assembly_gate_not_passed:{key}")
    if scope in ("compression", "impact", "all"):
        if native.get("native_slow_compression_completed") is not True:
            reasons.append("compression_not_completed")
        if native.get("native_half_speed_compression_completed") is not True:
            reasons.append("half_speed_compression_not_completed")
        sensitivity = native.get("compression_rate_sensitivity", {})
        if require_rate_pass and (not isinstance(sensitivity, dict) or sensitivity.get("passed") is not True):
            reasons.append("compression_rate_sensitivity_not_passed")
    if scope in ("impact", "all"):
        planned = native.get("native_finite_mass_drop_planned_cases")
        completed = native.get("native_finite_mass_drop_completed_cases")
        if not isinstance(planned, int) or planned <= 0 or completed != planned:
            reasons.append("all_drop_cases_not_completed")
        if native.get("native_gravity_transfer_all_cases_completed") is not True:
            reasons.append("all_gravity_transfer_cases_not_completed")
    if scope == "all":
        if native.get("score_tool_and_anvil_withdrawn_before_folding") is not True:
            reasons.append("score_tools_not_withdrawn_before_folding")
    if scope in ("assembly", "all"):
        if len(run.actions) != 15:
            reasons.append("assembly_action_ledger_not_complete")
        elif any(not row.get("status", "").startswith("completed") for row in run.actions):
            reasons.append("assembly_action_not_completed")
    if scope == "assembly" and not native_status.startswith(("completed_released_assembly", "completed_native_")):
        reasons.append("assembly_stage_not_complete")
    if scope == "compression" and not native_status.startswith(("completed_native_compression", "completed_native_gravity_transfer_compression")):
        reasons.append("compression_stage_not_complete")
    if scope == "impact" and not native_status.startswith("completed_native_gravity_transfer_compression_and_drop"):
        reasons.append("impact_stage_not_complete")
    if scope == "all" and not native_status.startswith("completed_native_gravity_transfer_compression_and_drop"):
        reasons.append("full_shipping_status_not_complete")
    return reasons


def _compression_curve(run: StageRun, phase: str = "nominal") -> tuple[list[tuple[float, float]], dict[str, Any]]:
    name = "shipping-compression.csv"
    path = run.path / name
    if not path.is_file():
        return [], {"status": "inconclusive_missing_force_work_curve"}
    fields, rows = _rows(path)
    needed = {"phase", "accepted", "stroke_m", "platen_mean_force_y_N", "cumulative_platen_work_J"}
    if not needed.issubset(fields):
        return [], {"status": "inconclusive_missing_columns", "missing": sorted(needed - set(fields))}
    selected = [row for row in rows if row.get("phase", "").startswith(phase + "_")]
    if not selected:
        return [], {"status": "inconclusive_missing_phase", "phase": phase}
    if any(row.get("accepted") != "1" for row in selected):
        return [], {"status": "failed_or_rejected_stage_row"}
    curve: list[tuple[float, float]] = []
    reported_work = None
    for index, row in enumerate(selected):
        stroke = _finite(row, "stroke_m", f"{name} row {index + 2}")
        force = _finite(row, "platen_mean_force_y_N", f"{name} row {index + 2}")
        if force < 0.0:
            return [], {"status": "invalid_negative_compression_force"}
        reported_work = _finite(row, "cumulative_platen_work_J", f"{name} row {index + 2}")
        if curve and stroke <= curve[-1][0]:
            if stroke == curve[-1][0]:
                curve[-1] = (stroke, force)
                continue
            return [], {"status": "invalid_nonmonotonic_stroke"}
        curve.append((stroke, force))
    if len(curve) < 2 or curve[0][0] > 1e-12:
        return [], {"status": "inconclusive_incomplete_stroke_curve"}
    return curve, {"status": "available", "samples": len(curve),
                   "maximum_stroke_m": curve[-1][0],
                   "reported_final_work_J": reported_work}


def _interp(curve: list[tuple[float, float]], x: float) -> float:
    if x <= curve[0][0]: return curve[0][1]
    for left, right in zip(curve, curve[1:]):
        if left[0] <= x <= right[0]:
            t = (x - left[0]) / (right[0] - left[0])
            return left[1] + t * (right[1] - left[1])
    return curve[-1][1]


def _curve_work(curve: list[tuple[float, float]], stop: float) -> float:
    x_values = sorted({0.0, stop, *(x for x, _ in curve if 0.0 < x < stop)})
    return sum((x1 - x0) * (y0 + y1) * 0.5
               for x0, x1 in zip(x_values, x_values[1:])
               for y0, y1 in [(_interp(curve, x0), _interp(curve, x1))])


def _compare_force_curves(curve_a: list[tuple[float, float]],
                          curve_b: list[tuple[float, float]],
                          left_reported_work: float | None = None,
                          right_reported_work: float | None = None) -> dict[str, Any]:
    common = min(curve_a[-1][0], curve_b[-1][0])
    knots = sorted({0.0, common, *(x for x, _ in curve_a if 0 < x < common),
                    *(x for x, _ in curve_b if 0 < x < common)})
    values_a = [_interp(curve_a, x) for x in knots]
    values_b = [_interp(curve_b, x) for x in knots]
    peak_a, peak_b = max(values_a), max(values_b)
    rms_scale = math.sqrt(sum(v*v for v in values_a) / max(len(values_a), 1))
    rms = math.sqrt(sum((a-b)**2 for a,b in zip(values_a, values_b)) / max(len(knots), 1))
    wa, wb = _curve_work(curve_a, common), _curve_work(curve_b, common)
    return {
        "status": "available", "common_stroke_m": common,
        "force_unit": "N",
        "load_normalization": "raw platen force; normalized curve RMS is divided by the left-run RMS force and not by board width",
        "left_peak_force_N": peak_a, "right_peak_force_N": peak_b,
        "relative_peak_force_difference": abs(peak_b-peak_a) / max(abs(peak_a), 1e-30),
        "normalized_force_curve_rms": rms / max(rms_scale, 1e-30),
        "left_work_J": wa, "right_work_J": wb,
        "relative_work_difference": abs(wb-wa) / max(abs(wa), 1e-30),
        "left_reported_final_stroke_m": curve_a[-1][0],
        "right_reported_final_stroke_m": curve_b[-1][0],
        "comparison_is_complete_endpoint": math.isclose(curve_a[-1][0], curve_b[-1][0], rel_tol=1e-9, abs_tol=1e-12),
        "left_native_reported_work_J": left_reported_work,
        "right_native_reported_work_J": right_reported_work,
    }


def _stage_durations(run: StageRun) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], float] = {}
    for row in run.steps:
        key = (row.get("stage", ""), row.get("action_index", ""))
        dt = _finite(row, "dt_s", "box-native-steps.csv")
        totals[key] = totals.get(key, 0.0) + dt
    return totals


def _same_stage_durations(left: StageRun, right: StageRun) -> bool:
    a, b = _stage_durations(left), _stage_durations(right)
    if set(a) != set(b) or not a:
        return False
    return all(math.isclose(a[key], b[key], rel_tol=1.0e-6, abs_tol=1.0e-9) for key in a)


def _compression_branch_step_stats(run: StageRun) -> tuple[dict[str, Any], list[str]]:
    """Summarize actual accepted native substeps, not macro increment labels."""
    stats: dict[str, Any] = {}
    reasons: list[str] = []
    compression_path = run.path / "shipping-compression.csv"
    compression_rows = run.optional_rows.get("shipping-compression.csv", [])
    compression_fields = set()
    if compression_path.is_file():
        try:
            compression_fields = set(_rows(compression_path)[0])
        except (OSError, csv.Error, StageStudyError):
            pass
    required_branch_fields = {
        "native_step", "phase", "accepted", "macro_observation_duration_s",
        "control_step_count", "control_step_dt_min_s", "control_step_dt_max_s",
    }
    if not compression_path.is_file() or not required_branch_fields.issubset(compression_fields):
        missing = sorted(required_branch_fields - compression_fields)
        return {}, ["compression_branch_record_missing_actual_dt_or_count:" +
                    (",".join(missing) if missing else "shipping-compression.csv")]
    for branch in ("nominal", "half_speed"):
        step_rows = [row for row in run.steps if row.get("stage") in (
            f"slow_compression_{branch}",
            f"compression_contact_approach_{branch}")]
        branch_rows = [row for row in compression_rows
                       if row.get("phase", "").startswith(branch + "_") and
                       not row.get("phase", "").endswith("contact_origin")]
        if not step_rows or not branch_rows:
            reasons.append(f"missing_actual_compression_substeps_or_branch_record:{branch}")
            continue
        dts_by_native_step: dict[int, float] = {}
        times: dict[int, float] = {}
        actual_stage_by_native_step: dict[int, str] = {}
        for ordinal, row in enumerate(run.steps):
            try:
                stage = row.get("stage", "")
                if stage not in (f"slow_compression_{branch}",
                                 f"compression_contact_approach_{branch}"):
                    continue
                if row.get("accepted") != "1":
                    reasons.append(f"rejected_actual_compression_substep:{branch}")
                    continue
                step = int(row["native_step"])
                label = f"box-native-steps.csv row {ordinal + 2}"
                dt = _finite(row, "dt_s", label)
                time_s = _finite(row, "simulation_time_s", label)
                if dt <= 0.0 or step in dts_by_native_step:
                    reasons.append(f"invalid_or_duplicate_actual_compression_dt:{branch}")
                    continue
                previous_time = 0.0 if ordinal == 0 else _finite(
                    run.steps[ordinal - 1], "simulation_time_s",
                    f"box-native-steps.csv row {ordinal + 1}")
                if not math.isclose(time_s - previous_time, dt,
                                    rel_tol=1.0e-6, abs_tol=1.0e-10):
                    reasons.append(f"native_clock_increment_does_not_match_dt:{branch}:{step}")
                dts_by_native_step[step] = dt
                times[step] = time_s
                actual_stage_by_native_step[step] = stage
            except (ValueError, KeyError, StageStudyError) as exc:
                reasons.append(f"missing_actual_compression_dt:{branch}:{exc}")
        branch_steps: set[int] = set()
        for ordinal, row in enumerate(branch_rows):
            label = f"shipping-compression.csv row {ordinal + 2}"
            try:
                if row.get("accepted") != "1":
                    raise ValueError("macro observation is not accepted")
                phase = row["phase"]
                if phase == f"{branch}_slow_compression":
                    expected_stage = f"slow_compression_{branch}"
                elif phase == f"{branch}_contact_approach":
                    expected_stage = f"compression_contact_approach_{branch}"
                else:
                    raise ValueError(f"unsupported compression branch phase {phase}")
                endpoint = int(row["native_step"])
                count = int(row["control_step_count"])
                macro_duration = _finite(row, "macro_observation_duration_s", label)
                summary_min = _finite(row, "control_step_dt_min_s", label)
                summary_max = _finite(row, "control_step_dt_max_s", label)
                if count <= 0 or macro_duration <= 0.0 or summary_min <= 0.0 or \
                        summary_max < summary_min:
                    raise ValueError("invalid macro duration or accepted control-step summary")
                indices = list(range(endpoint - count + 1, endpoint + 1))
                if any(index not in dts_by_native_step or
                       actual_stage_by_native_step.get(index) != expected_stage
                       for index in indices):
                    raise ValueError("control-step count does not bind to consecutive native branch rows")
                actual_dts = [dts_by_native_step[index] for index in indices]
                if not math.isclose(sum(actual_dts), macro_duration,
                                    rel_tol=1.0e-6, abs_tol=1.0e-9):
                    raise ValueError("macro duration differs from summed accepted native substeps")
                if not math.isclose(min(actual_dts), summary_min,
                                    rel_tol=1.0e-6, abs_tol=1.0e-12) or \
                        not math.isclose(max(actual_dts), summary_max,
                                         rel_tol=1.0e-6, abs_tol=1.0e-12):
                    raise ValueError("macro min/max dt differs from accepted native substeps")
                if branch_steps.intersection(indices):
                    raise ValueError("compression branch record overlaps native control steps")
                branch_steps.update(indices)
            except (ValueError, KeyError, StageStudyError) as exc:
                reasons.append(f"invalid_actual_compression_branch_record:{branch}:{exc}")
        if branch_steps != set(dts_by_native_step):
            reasons.append(f"compression_branch_record_native_step_coverage_mismatch:{branch}")
        dts = [dts_by_native_step[step] for step in sorted(branch_steps)]
        step_times = [times[step] for step in sorted(branch_steps)]
        if dts and step_times:
            try:
                declared_dt = float(run.manifest["solver"]["dt_s"])
            except (KeyError, TypeError, ValueError):
                declared_dt = math.nan
            if not math.isfinite(declared_dt) or declared_dt <= 0.0:
                reasons.append(f"missing_or_invalid_declared_solver_dt:{branch}")
            elif max(dts) > declared_dt * (1.0 + 1.0e-6) + 1.0e-12:
                reasons.append(f"accepted_compression_dt_exceeds_declared_solver_cap:{branch}")
            stats[branch] = {
                "stages": [f"compression_contact_approach_{branch}", f"slow_compression_{branch}"],
                "accepted_native_substep_count": len(dts),
                "accepted_duration_s": sum(dts),
                "maximum_accepted_dt_s": max(dts),
                "minimum_accepted_dt_s": min(dts),
                "first_step_end_time_s": min(step_times),
                "last_step_end_time_s": max(step_times),
            }
    return stats, reasons


def _time_step_refinement_admission(left: StageRun, right: StageRun,
                                    left_dt: Any, right_dt: Any) -> dict[str, Any]:
    """Require actual finer accepted compression substeps with equal duration."""
    left_stats, left_reasons = _compression_branch_step_stats(left)
    right_stats, right_reasons = _compression_branch_step_stats(right)
    reasons = [f"left:{reason}" for reason in left_reasons]
    reasons.extend(f"right:{reason}" for reason in right_reasons)
    if len(left_stats) != 2 or len(right_stats) != 2:
        return {"status": "inconclusive_missing_actual_branch_dt",
                "reasons": reasons, "left": left_stats, "right": right_stats}
    coarse, fine = (left_stats, right_stats) if float(left_dt) > float(right_dt) else (right_stats, left_stats)
    for branch in ("nominal", "half_speed"):
        a, b = coarse[branch], fine[branch]
        if not math.isclose(a["accepted_duration_s"], b["accepted_duration_s"],
                            rel_tol=1.0e-6, abs_tol=1.0e-9):
            reasons.append(f"compression_branch_duration_changed:{branch}")
        if b["maximum_accepted_dt_s"] >= a["maximum_accepted_dt_s"] * (1.0 - 1.0e-9):
            reasons.append(f"compression_branch_max_dt_did_not_decrease:{branch}")
        if b["accepted_native_substep_count"] <= a["accepted_native_substep_count"]:
            reasons.append(f"compression_branch_substep_count_did_not_increase:{branch}")
    return {"status": "available" if not reasons else "incompatible_actual_timestep_schedule",
            "coarse_axis_dt_s": max(float(left_dt), float(right_dt)),
            "fine_axis_dt_s": min(float(left_dt), float(right_dt)),
            "coarse": coarse, "fine": fine, "reasons": reasons,
            "contract": "accepted native steps in each slow_compression_{nominal,half_speed} stage; equal branch duration, lower maximum dt, and higher accepted substep count required"}


def _interp_series(series: list[tuple[float, float]], x: float) -> float:
    if x <= series[0][0]:
        return series[0][1]
    for left, right in zip(series, series[1:]):
        if left[0] <= x <= right[0]:
            span = right[0] - left[0]
            if span <= 0.0:
                return right[1]
            ratio = (x - left[0]) / span
            return left[1] + ratio * (right[1] - left[1])
    return series[-1][1]


def _accepted_step_times(run: StageRun) -> dict[int, float]:
    result: dict[int, float] = {}
    for ordinal, row in enumerate(run.steps):
        try:
            step = int(row["native_step"])
            if row.get("accepted") != "1":
                continue
            time_s = _finite(row, "simulation_time_s",
                             f"box-native-steps.csv row {ordinal + 2}")
        except (ValueError, KeyError, StageStudyError):
            continue
        result[step] = time_s
    return result


def _grip_moment_series(run: StageRun) -> tuple[dict[tuple[str, str, str, str], list[tuple[float, tuple[float, float, float]]]], dict[str, Any]]:
    name = "box-grip-reactions.csv"
    rows = run.optional_rows.get(name, [])
    path = run.path / name
    if not path.is_file():
        return {}, {"status": "inconclusive_missing_export"}
    fields, _ = _rows(path)
    needed = {"native_step", "time_s", "stage", "assembly_action_index",
              "score_line_index", "environment", "step_accepted",
              "reaction_valid", "active_grip_node_count", "total_moment_x_Nm",
              "total_moment_y_Nm", "total_moment_z_Nm"}
    if not needed.issubset(fields):
        return {}, {"status": "inconclusive_missing_columns",
                    "missing": sorted(needed - set(fields))}
    grouped: dict[tuple[str, str, str, str], list[tuple[float, tuple[float, float, float]]]] = {}
    seen_samples: set[tuple[tuple[str, str, str, str], str]] = set()
    accepted_times = _accepted_step_times(run)
    for ordinal, row in enumerate(rows):
        label = f"{name} row {ordinal + 2}"
        if row.get("step_accepted") != "1":
            return {}, {"status": "failed_or_rejected_reaction_row", "row": ordinal + 2}
        try:
            active_count = int(row["active_grip_node_count"])
            reaction_valid = int(row["reaction_valid"])
            if active_count < 0 or reaction_valid not in (0, 1):
                raise ValueError("invalid active count or reaction flag")
            if active_count == 0:
                if reaction_valid != 0:
                    raise ValueError("reaction marked valid without active grip nodes")
                continue
            if reaction_valid != 1:
                raise ValueError("active grip nodes have no accepted reaction wrench")
            time_s = _finite(row, "time_s", label)
            step_index = int(row["native_step"])
            step = str(step_index)
            if step_index not in accepted_times or not math.isclose(
                    time_s, accepted_times[step_index], rel_tol=1.0e-9,
                    abs_tol=1.0e-12):
                raise ValueError("reaction sample does not bind to an accepted native step/time")
            moment = tuple(_finite(row, key, label) for key in (
                "total_moment_x_Nm", "total_moment_y_Nm", "total_moment_z_Nm"))
            key = (row["stage"], row["assembly_action_index"],
                   row["score_line_index"], row["environment"])
            sample_key = (key, step)
            if sample_key in seen_samples:
                return {}, {"status": "ambiguous_duplicate_active_grip_sample",
                            "key": list(key), "native_step": step}
            seen_samples.add(sample_key)
            grouped.setdefault(key, []).append((time_s, moment))
        except (ValueError, KeyError, StageStudyError) as exc:
            return {}, {"status": "inconclusive_invalid_reaction_row",
                        "row": ordinal + 2, "reason": str(exc)}
    if not grouped:
        return {}, {"status": "inconclusive_no_active_grip_moment_samples"}
    for series in grouped.values():
        series.sort(key=lambda sample: sample[0])
        if any(right[0] <= left[0] for left, right in zip(series, series[1:])):
            return {}, {"status": "inconclusive_nonmonotonic_reaction_time"}
    return grouped, {"status": "available", "active_grip_series_count": len(grouped),
                    "active_grip_sample_count": sum(map(len, grouped.values()))}


def _compare_grip_moment_curves(left: StageRun, right: StageRun) -> dict[str, Any]:
    left_series, left_status = _grip_moment_series(left)
    right_series, right_status = _grip_moment_series(right)
    if left_status.get("status") != "available" or right_status.get("status") != "available":
        return {"status": "inconclusive_missing_or_invalid_accepted_moment_series",
                "left": left_status, "right": right_status}
    if set(left_series) != set(right_series):
        return {"status": "inconclusive_grip_series_coverage_mismatch",
                "left_series": [list(key) for key in sorted(left_series)],
                "right_series": [list(key) for key in sorted(right_series)]}
    differences: list[float] = []
    left_norms: list[float] = []
    peak_left: list[float] = []
    peak_right: list[float] = []
    common_duration_by_group: dict[str, float] = {}
    for key in sorted(left_series):
        a, b = left_series[key], right_series[key]
        common_start = max(a[0][0], b[0][0])
        common_end = min(a[-1][0], b[-1][0])
        if common_end <= common_start or len(a) < 2 or len(b) < 2:
            return {"status": "inconclusive_insufficient_common_moment_curve",
                    "key": list(key), "left_samples": len(a), "right_samples": len(b)}
        knots = sorted({common_start, common_end,
                        *(time_s for time_s, _ in a if common_start < time_s < common_end),
                        *(time_s for time_s, _ in b if common_start < time_s < common_end)})
        mag_a = []
        mag_b = []
        for time_s in knots:
            vector_a = tuple(_interp_series([(t, m[axis]) for t, m in a], time_s)
                             for axis in range(3))
            vector_b = tuple(_interp_series([(t, m[axis]) for t, m in b], time_s)
                             for axis in range(3))
            mag_a.append(math.sqrt(sum(value * value for value in vector_a)))
            mag_b.append(math.sqrt(sum(value * value for value in vector_b)))
            left_norms.append(sum(value * value for value in vector_a))
            differences.append(sum((x - y) ** 2 for x, y in zip(vector_a, vector_b)))
        peak_left.append(max(mag_a))
        peak_right.append(max(mag_b))
        common_duration_by_group["|".join(key)] = common_end - common_start
    rms_difference = math.sqrt(sum(differences) / max(1, len(differences)))
    left_rms_moment = math.sqrt(sum(left_norms) / max(1, len(left_norms)))
    if left_rms_moment <= 1.0e-30:
        return {"status": "inconclusive_zero_reference_moment_scale",
                "left": left_status, "right": right_status}
    max_left = max(peak_left)
    max_right = max(peak_right)
    return {
        "status": "available",
        "quantity": "accepted total clamp wrench moment vector, mechanical plus contact",
        "unit": "N m",
        "load_normalization": "raw total moment; not divided by board width; curve difference divided by left-run RMS moment magnitude",
        "normalized_total_moment_vector_curve_rms": rms_difference / left_rms_moment,
        "absolute_total_moment_vector_curve_rms_Nm": rms_difference,
        "left_rms_moment_Nm": left_rms_moment,
        "left_peak_moment_magnitude_Nm": max_left,
        "right_peak_moment_magnitude_Nm": max_right,
        "relative_peak_moment_difference": abs(max_right - max_left) / max(abs(max_left), 1.0e-30),
        "series_count": len(left_series),
        "left_sample_count": left_status["active_grip_sample_count"],
        "right_sample_count": right_status["active_grip_sample_count"],
        "common_duration_by_series_s": common_duration_by_group,
        "left_series_status": left_status,
        "right_series_status": right_status,
        "pivot_contract": "CSV moments are reported about each accepted active grip patch pivot; compare only matching stage/action/score-line/environment groups",
    }


def _springback_observations(run: StageRun) -> tuple[dict[tuple[int, int], dict[str, float]], dict[str, Any]]:
    name = "box-score-springback.csv"
    path = run.path / name
    native = run.native or {}
    if not path.is_file():
        return {}, {"status": "inconclusive_missing_export"}
    if native.get("springback_observation_file") != name:
        return {}, {"status": "inconclusive_native_export_binding_mismatch"}
    rows = run.optional_rows.get(name, [])
    fields, _ = _rows(path)
    needed = {"environment", "score_line_index", "held_native_step", "held_time_s",
              "released_native_step", "released_time_s", "released_settled",
              "held_turn_deg", "released_turn_deg", "springback_turn_deg",
              "held_fit_rms_m", "released_fit_rms_m", "fit_bound_m",
              "held_fit_coverage", "released_fit_coverage", "sample_count",
              "held_fit_pass", "released_fit_pass", "springback_observable", "reason"}
    if not needed.issubset(fields):
        return {}, {"status": "inconclusive_missing_columns",
                    "missing": sorted(needed - set(fields))}
    if not rows:
        return {}, {"status": "inconclusive_empty_springback_export"}
    if native.get("released_assembly_settled") is not True or \
            native.get("released_clamp_zero_active_gate_passed") is not True or \
            _safe_int(native.get("remaining_active_temporary_clamp_node_count")) != 0 or \
            _safe_int(native.get("released_settle_steps_accepted")) <= 0:
        return {}, {"status": "inconclusive_release_state_not_admitted"}
    if native.get("shape_fit_bound_is_published_tolerance") is not False:
        return {}, {"status": "inconclusive_fit_bound_provenance_not_declared_engineering"}
    try:
        min_coverage = float(native["shape_fit_coverage_minimum"])
    except (KeyError, TypeError, ValueError):
        return {}, {"status": "inconclusive_missing_native_shape_coverage_bound"}
    if not math.isfinite(min_coverage) or not 0.0 < min_coverage <= 1.0:
        return {}, {"status": "inconclusive_invalid_native_shape_coverage_bound"}
    expected_rows = _safe_int(native.get("springback_observation_rows"))
    expected_observable = _safe_int(native.get("springback_observable_rows"))
    if expected_rows != len(rows) or expected_observable != len(rows):
        return {}, {"status": "inconclusive_springback_count_mismatch",
                    "csv_rows": len(rows), "native_rows": expected_rows,
                    "native_observable_rows": expected_observable}
    observations: dict[tuple[int, int], dict[str, float]] = {}
    accepted_times = _accepted_step_times(run)
    for ordinal, row in enumerate(rows):
        label = f"{name} row {ordinal + 2}"
        try:
            env = int(row["environment"])
            score = int(row["score_line_index"])
            held_step = int(row["held_native_step"])
            released_step = int(row["released_native_step"])
            sample_count = int(row["sample_count"])
            if env < 0 or not 0 <= score < 12 or held_step < 0 or released_step < held_step or sample_count <= 0:
                raise ValueError("invalid score-line identity or endpoint ordering")
            if row["released_settled"] != "1" or row["held_fit_pass"] != "1" or \
                    row["released_fit_pass"] != "1" or row["springback_observable"] != "1":
                raise ValueError("native shape/coverage/release fit gate did not pass")
            held_time = _finite(row, "held_time_s", label)
            released_time = _finite(row, "released_time_s", label)
            if released_time <= held_time:
                raise ValueError("release endpoint does not follow held endpoint")
            if held_step not in accepted_times or released_step not in accepted_times or \
                    not math.isclose(held_time, accepted_times[held_step],
                                     rel_tol=1.0e-9, abs_tol=1.0e-12) or \
                    not math.isclose(released_time, accepted_times[released_step],
                                     rel_tol=1.0e-9, abs_tol=1.0e-12):
                raise ValueError("springback endpoints do not bind to accepted native steps and times")
            held_rms = _finite(row, "held_fit_rms_m", label)
            released_rms = _finite(row, "released_fit_rms_m", label)
            fit_bound = _finite(row, "fit_bound_m", label)
            held_coverage = _finite(row, "held_fit_coverage", label)
            released_coverage = _finite(row, "released_fit_coverage", label)
            if fit_bound <= 0.0 or held_rms > fit_bound or released_rms > fit_bound or \
                    not min_coverage <= held_coverage <= 1.0 or \
                    not min_coverage <= released_coverage <= 1.0:
                raise ValueError("exported shape residual or coverage fails its native gate")
            observation = {
                "held_turn_deg": _finite(row, "held_turn_deg", label),
                "released_turn_deg": _finite(row, "released_turn_deg", label),
                "springback_turn_deg": _finite(row, "springback_turn_deg", label),
                "held_fit_rms_m": held_rms, "released_fit_rms_m": released_rms,
                "fit_bound_m": fit_bound, "held_fit_coverage": held_coverage,
                "released_fit_coverage": released_coverage,
                "sample_count": float(sample_count),
            }
            key = (env, score)
            if key in observations:
                raise ValueError("duplicate environment/score-line row")
            observations[key] = observation
        except (ValueError, KeyError, StageStudyError) as exc:
            return {}, {"status": "inconclusive_invalid_springback_observation",
                        "row": ordinal + 2, "reason": str(exc)}
    environments = sorted({key[0] for key in observations})
    expected_keys = {(env, score) for env in environments for score in range(12)}
    if set(observations) != expected_keys or not environments or \
            len(observations) != expected_rows:
        return {}, {"status": "inconclusive_incomplete_score_line_coverage",
                    "observed_rows": len(observations), "expected_rows": expected_rows,
                    "environment_ids": environments}
    return observations, {
        "status": "available", "row_count": len(observations),
        "environment_count": len(environments), "score_lines_per_environment": 12,
        "minimum_native_fit_coverage": min_coverage,
        "minimum_observed_fit_coverage": min(min(
            item["held_fit_coverage"], item["released_fit_coverage"])
            for item in observations.values()),
        "maximum_fit_rms_to_bound_ratio": max(max(
            item["held_fit_rms_m"], item["released_fit_rms_m"])
            / item["fit_bound_m"] for item in observations.values()),
        "all_endpoint_shape_and_coverage_gates_passed": True,
        "fit_bound_is_published_tolerance": native.get("shape_fit_bound_is_published_tolerance"),
    }


def _compare_released_springback(left: StageRun, right: StageRun) -> dict[str, Any]:
    left_obs, left_status = _springback_observations(left)
    right_obs, right_status = _springback_observations(right)
    if left_status.get("status") != "available" or right_status.get("status") != "available":
        return {"status": "inconclusive_missing_or_invalid_released_springback",
                "left": left_status, "right": right_status}
    if set(left_obs) != set(right_obs):
        return {"status": "inconclusive_springback_score_line_coverage_mismatch",
                "left_keys": [list(key) for key in sorted(left_obs)],
                "right_keys": [list(key) for key in sorted(right_obs)]}
    springback = [right_obs[key]["springback_turn_deg"] -
                  left_obs[key]["springback_turn_deg"] for key in sorted(left_obs)]
    released_turn = [right_obs[key]["released_turn_deg"] -
                     left_obs[key]["released_turn_deg"] for key in sorted(left_obs)]
    return {
        "status": "available",
        "score_line_count": 12,
        "environment_count": left_status["environment_count"],
        "mean_absolute_springback_difference_deg": sum(map(abs, springback)) / len(springback),
        "rms_springback_difference_deg": math.sqrt(sum(value * value for value in springback) / len(springback)),
        "maximum_absolute_springback_difference_deg": max(map(abs, springback)),
        "rms_released_turn_difference_deg": math.sqrt(sum(value * value for value in released_turn) / len(released_turn)),
        "left_minimum_fit_coverage": left_status["minimum_observed_fit_coverage"],
        "right_minimum_fit_coverage": right_status["minimum_observed_fit_coverage"],
        "left_maximum_fit_rms_to_bound_ratio": left_status["maximum_fit_rms_to_bound_ratio"],
        "right_maximum_fit_rms_to_bound_ratio": right_status["maximum_fit_rms_to_bound_ratio"],
        "shape_fit_coverage_gate_passed": True,
        "shape_fit_residual_gate_passed": True,
        "native_shape_fit_bound_is_published_tolerance": False,
        "left_validation": left_status,
        "right_validation": right_status,
        "units": "degrees; paired by environment and authored score-line index",
    }


def _history_metrics(run: StageRun) -> dict[str, Any]:
    path = run.path / "shipping-history-observables.csv"
    if not path.is_file():
        return {"status": "inconclusive_missing_history_observables"}
    fields, rows = _rows(path)
    if not rows:
        return {"status": "inconclusive_empty_history_observables"}
    if "accepted" not in fields or any(row.get("accepted") != "1" for row in rows):
        return {"status": "failed_or_rejected_history_row"}
    terminal = [row for row in rows if row.get("stage") == "final_accepted_state"]
    if not terminal:
        return {"status": "inconclusive_missing_terminal_accepted_state"}
    row = terminal[-1]
    result: dict[str, Any] = {"status": "available_model_damage_indices_not_geometric_area"}
    for key in (
        "paper_status", "paper_area_weighted_damage_fraction",
        "paper_reference_plane_area_m2", "starch_status",
        "starch_area_weighted_cohesive_damage_fraction", "starch_active_reference_area_m2",
        "hma_status", "hma_area_weighted_cohesive_damage_fraction", "hma_active_reference_area_m2",
    ):
        result[key] = row.get(key) or None
    result["geometric_crack_area_fraction"] = None
    result["geometric_debond_area_fraction"] = None
    result["area_fraction_status"] = "inconclusive_not_exported_by_probe"

    fully_failed_fields = {
        "starch": ("starch_fully_failed_status", "starch_fully_failed_cohesive_area_m2",
                   "starch_fully_failed_cohesive_area_fraction", "starch_active_reference_area_m2"),
        "hma": ("hma_fully_failed_status", "hma_fully_failed_cohesive_area_m2",
                "hma_fully_failed_cohesive_area_fraction", "hma_active_reference_area_m2"),
    }
    required = {field for fields in fully_failed_fields.values() for field in fields}
    if not required.issubset(fields):
        result["fully_failed_cohesive_capacity_area_status"] = "inconclusive_columns_not_exported"
        result["fully_failed_cohesive_capacity_area"] = None
        return result
    capacities: dict[str, Any] = {}
    for role, (status_key, area_key, fraction_key, denominator_key) in fully_failed_fields.items():
        status = row.get(status_key, "")
        denominator_raw = row.get(denominator_key, "")
        area_raw = row.get(area_key, "")
        fraction_raw = row.get(fraction_key, "")
        if status == "not_enabled":
            capacities[role] = {"status": "not_enabled"}
            continue
        if status != "available_quadrature_zero_tensile_shear_capacity_fraction_not_geometric_open_area":
            capacities[role] = {"status": (
                "inconclusive_legacy_face_maximum_not_quadrature_area"
                if status == "available_zero_tensile_shear_capacity_fraction_not_geometric_open_area"
                else status or "inconclusive_missing_status")}
            continue
        try:
            denominator, area, fraction = map(float, (denominator_raw, area_raw, fraction_raw))
        except (TypeError, ValueError):
            capacities[role] = {"status": "inconclusive_missing_numeric_area"}
            continue
        if not all(math.isfinite(value) for value in (denominator, area, fraction)) or \
                denominator <= 0.0 or area < 0.0 or area > denominator * (1.0 + 1e-9) or \
                fraction < 0.0 or fraction > 1.0 or \
                not math.isclose(fraction, area / denominator, rel_tol=1e-6, abs_tol=1e-9):
            capacities[role] = {"status": "inconclusive_inconsistent_area_accounting"}
            continue
        capacities[role] = {
            "status": "available_zero_tensile_shear_capacity_not_geometric_open_area",
            "fully_failed_cohesive_area_m2": area,
            "activated_reference_area_denominator_m2": denominator,
            "fully_failed_cohesive_area_fraction": fraction,
            "definition": "sum of active reference face area times the fraction of the three equal quadrature weights with accepted point damage exactly 1; divided by activated reference area",
        }
    result["fully_failed_cohesive_capacity_area"] = capacities
    active_capacity = [value for value in capacities.values()
                       if value.get("status") == "available_zero_tensile_shear_capacity_not_geometric_open_area"]
    enabled_capacity = [value for value in capacities.values()
                        if value.get("status") != "not_enabled"]
    result["fully_failed_cohesive_capacity_area_status"] = (
        "available_zero_tensile_shear_capacity_not_geometric_open_area"
        if active_capacity and len(active_capacity) == len(enabled_capacity)
        else "inconclusive_missing_or_invalid_capacity_area")
    return result



def _capacity_loss_debond_gate(history_a: dict[str, Any], history_b: dict[str, Any],
                                differences: dict[str, Any]) -> dict[str, Any]:
    """Irreversible failed reference area includes closed, contact-bearing faces."""
    roles: dict[str, Any] = {}
    available_status = "available_zero_tensile_shear_capacity_not_geometric_open_area"
    for role in ("starch", "hma"):
        left = (history_a.get("fully_failed_cohesive_capacity_area") or {}).get(role, {})
        right = (history_b.get("fully_failed_cohesive_capacity_area") or {}).get(role, {})
        if left.get("status") == right.get("status") == "not_enabled":
            roles[role] = {"status": "not_enabled"}
            continue
        item = differences.get(role, {})
        difference = item.get("absolute_difference")
        if left.get("status") != available_status or right.get("status") != available_status or \
                item.get("status") != "available" or \
                not isinstance(difference, (float, int)) or not math.isfinite(difference):
            roles[role] = {"status": "inconclusive_missing_or_incompatible_accepted_reference_area"}
            continue
        roles[role] = {
            "status": "pass" if 0.0 <= difference <= 0.02 else "fail",
            "absolute_fraction_difference": difference,
            "absolute_fraction_limit": 0.02,
        }
    states = [item["status"] for item in roles.values()]
    status = ("fail" if "fail" in states else
              "inconclusive" if any(value not in ("pass", "not_enabled") for value in states) else
              "pass" if "pass" in states else "not_enabled")
    return {
        "status": status, "per_interface": roles,
        "definition": "reference quadrature area with accepted irreversible damage D exactly 1, divided by matching activated reference interface area",
        "threshold": "D == 1; no tensile or shear capacity remains",
        "quadrature": "three equal reference-area weights per mixed-mode face",
        "closed_failed_interfaces_count_as_debonded": True,
        "geometric_open_area_is_a_separate_observable": True,
    }


def _capacity_history_gates(report: dict[str, Any]) -> dict[str, Any]:
    damage = report.get("metrics", {}).get("damage_and_debond", {})
    left, right = damage.get("left", {}), damage.get("right", {})
    # The existing paper index normalizes by a mesh-dependent sum of all tet
    # cut-plane areas. It is not an objective crack-area fraction. A force-only
    # match must not let an enabled but unmeasured localization field qualify.
    paper = {
        "status": ("not_enabled" if left.get("paper_status") == right.get("paper_status") == "not_enabled"
                   else "inconclusive_missing_mesh_objective_paper_damage_fraction"),
        "absolute_fraction_limit": 0.02,
        "constitutive_damage_index_substituted_for_area": False,
    }
    debond = damage.get("constitutive_debond_fraction_gate") or {"status": "inconclusive"}
    return {"paper_damage_fraction": paper, "irreversible_debond_fraction": debond}


def _impact_histories(run: StageRun) -> tuple[dict[str, list[dict[str, float]]], dict[str, Any]]:
    path = run.path / "shipping-impact-dynamics.csv"
    if not path.is_file():
        return {}, {"status": "inconclusive_missing_impact_dynamics"}
    fields, rows = _rows(path)
    needed = {"case_id", "time_s", "dt_s", "accepted", "body_y_m",
              "matter_mean_force_y_n", "rigid_mechanical_energy_relative_j"}
    if not needed.issubset(fields):
        return {}, {"status": "inconclusive_missing_columns", "missing": sorted(needed - set(fields))}
    histories: dict[str, list[dict[str, float]]] = {}
    for index, row in enumerate(rows):
        label = f"shipping-impact-dynamics.csv row {index + 2}"
        if row.get("accepted") != "1":
            return {}, {"status": "failed_or_rejected_impact_row", "row": index + 2}
        key = row.get("case_id", "")
        if not key:
            return {}, {"status": "invalid_missing_impact_case_id", "row": index + 2}
        sample = {
            "time_s": _finite(row, "time_s", label),
            "dt_s": _finite(row, "dt_s", label),
            "body_y_m": _finite(row, "body_y_m", label),
            "force_N": _finite(row, "matter_mean_force_y_n", label),
            "energy_J": _finite(row, "rigid_mechanical_energy_relative_j", label),
        }
        if sample["time_s"] < 0.0 or sample["dt_s"] <= 0.0:
            return {}, {"status": "invalid_impact_clock", "row": index + 2}
        histories.setdefault(key, []).append(sample)
    for key, samples in histories.items():
        samples.sort(key=lambda item: item["time_s"])
        if len(samples) < 2 or any(b["time_s"] <= a["time_s"] for a, b in zip(samples, samples[1:])):
            return {}, {"status": "incomplete_or_nonmonotonic_impact_history", "case_id": key}
    return histories, {"status": "available", "case_count": len(histories)}


def _impact_interp(samples: list[dict[str, float]], key: str, time_s: float) -> float:
    if time_s <= samples[0]["time_s"]:
        return samples[0][key]
    for left, right in zip(samples, samples[1:]):
        if left["time_s"] <= time_s <= right["time_s"]:
            fraction = (time_s - left["time_s"]) / (right["time_s"] - left["time_s"])
            return left[key] + fraction * (right[key] - left[key])
    return samples[-1][key]


def _compare_impact_histories(left: StageRun, right: StageRun) -> dict[str, Any]:
    histories_a, status_a = _impact_histories(left)
    histories_b, status_b = _impact_histories(right)
    if not histories_a or not histories_b:
        return {"status": "inconclusive_missing_or_invalid_impact_history",
                "left": status_a, "right": status_b}
    if set(histories_a) != set(histories_b):
        return {"status": "incompatible_impact_case_set",
                "left_case_ids": sorted(histories_a), "right_case_ids": sorted(histories_b)}
    per_case: dict[str, Any] = {}
    complete = True
    for case_id in sorted(histories_a):
        a, b = histories_a[case_id], histories_b[case_id]
        start = max(a[0]["time_s"], b[0]["time_s"])
        common = min(a[-1]["time_s"], b[-1]["time_s"])
        if common <= start:
            complete = False
            per_case[case_id] = {"status": "inconclusive_no_common_time_interval"}
            continue
        knots = sorted({start, common,
                        *(x["time_s"] for x in a if start < x["time_s"] < common),
                        *(x["time_s"] for x in b if start < x["time_s"] < common)})
        fa = [_impact_interp(a, "force_N", t) for t in knots]
        fb = [_impact_interp(b, "force_N", t) for t in knots]
        ya = [_impact_interp(a, "body_y_m", t) for t in knots]
        yb = [_impact_interp(b, "body_y_m", t) for t in knots]
        ea = [_impact_interp(a, "energy_J", t) for t in knots]
        eb = [_impact_interp(b, "energy_J", t) for t in knots]
        scale = math.sqrt(sum(value*value for value in fa) / max(1, len(fa)))
        rms_force = math.sqrt(sum((x-y)**2 for x, y in zip(fa, fb)) / len(knots))
        rms_y = math.sqrt(sum((x-y)**2 for x, y in zip(ya, yb)) / len(knots))
        rms_energy = math.sqrt(sum((x-y)**2 for x, y in zip(ea, eb)) / len(knots))
        per_case[case_id] = {
            "status": "available", "common_time_s": common,
            "left_peak_contact_force_N": max(map(abs, fa)),
            "right_peak_contact_force_N": max(map(abs, fb)),
            "normalized_force_curve_rms": rms_force / max(scale, 1e-30),
            "body_height_path_rms_m": rms_y,
            "mechanical_energy_path_rms_J": rms_energy,
            "left_endpoint_time_s": a[-1]["time_s"],
            "right_endpoint_time_s": b[-1]["time_s"],
        }
        if not math.isclose(a[-1]["time_s"], b[-1]["time_s"], rel_tol=1e-9, abs_tol=1e-12):
            complete = False
    return {"status": "available" if complete else "inconclusive_duration_mismatch",
            "case_count": len(per_case), "per_case": per_case,
            "moment_curve": "inconclusive_not_exported"}


def compare_runs(left_dir: Path, right_dir: Path, axis: str, scope: str = "all") -> dict[str, Any]:
    if axis not in AXES:
        raise StageStudyError(f"axis must be one of {AXES}")
    left, right = load_run(left_dir), load_run(right_dir)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "axis": axis,
        "scope": scope,
        "left": {"path": str(left.path), "admission": left.admission, "files": left.files,
                 "source_bindings": left.source_bindings,
                 "axis_value": _axis_values(left, axis)},
        "right": {"path": str(right.path), "admission": right.admission, "files": right.files,
                  "source_bindings": right.source_bindings,
                  "axis_value": _axis_values(right, axis)},
        "status": "inconclusive",
        "metrics": {},
        "missing_observables": [],
    }
    gate_reasons = _scope_gate(left, scope) + [f"right:{item}" for item in _scope_gate(right, scope)]
    if gate_reasons:
        report["status"] = "inconclusive_or_failed_run_admission"
        report["reasons"] = gate_reasons
        return report
    left_identity = _varying_identity(physical_identity(left), axis)
    right_identity = _varying_identity(physical_identity(right), axis)
    if _canonical(left_identity) != _canonical(right_identity):
        report["status"] = "incompatible_physical_identity"
        report["reasons"] = ["runs differ outside the declared comparison axis"]
        return report
    report["physical_identity_sha256"] = _sha(_canonical(left_identity).encode())
    left_axis_value, right_axis_value = _axis_values(left, axis), _axis_values(right, axis)
    if left_axis_value is None or right_axis_value is None:
        report["status"] = "inconclusive_missing_axis_metadata"
        report["reasons"] = ["manifest/result does not identify both comparison-axis values"]
        return report
    if left_axis_value == right_axis_value:
        report["status"] = "incompatible_pair"
        report["reasons"] = ["declared refinement axis has identical values"]
        return report
    if axis == "time_step":
        if scope not in ("compression", "all"):
            report["status"] = "incompatible_time_step_scope"
            report["reasons"] = ["timestep refinement admission currently requires the complete nominal and half-speed compression branches"]
            return report
        if not _same_stage_durations(left, right):
            report["status"] = "incompatible_physical_schedule"
            report["reasons"] = ["per-stage accepted physical durations differ; this is not a pure timestep refinement"]
            return report
    if axis == "mesh_nx" and (left.manifest["mesh"].get("ny"), left.manifest["mesh"].get("thickness_slices")) != (right.manifest["mesh"].get("ny"), right.manifest["mesh"].get("thickness_slices")):
        report["status"] = "incompatible_pair"
        report["reasons"] = ["another mesh axis changed with nx"]
        return report
    if axis == "mesh_ny" and (left.manifest["mesh"].get("nx_per_pitch"), left.manifest["mesh"].get("thickness_slices")) != (right.manifest["mesh"].get("nx_per_pitch"), right.manifest["mesh"].get("thickness_slices")):
        report["status"] = "incompatible_pair"
        report["reasons"] = ["another mesh axis changed with ny"]
        return report
    if axis == "mesh_thickness" and (left.manifest["mesh"].get("nx_per_pitch"), left.manifest["mesh"].get("ny")) != (right.manifest["mesh"].get("nx_per_pitch"), right.manifest["mesh"].get("ny")):
        report["status"] = "incompatible_pair"
        report["reasons"] = ["another mesh axis changed with thickness"]
        return report
    try:
        left_numeric, right_numeric = float(left_axis_value), float(right_axis_value)
        if not (math.isfinite(left_numeric) and math.isfinite(right_numeric)):
            raise ValueError
    except (TypeError, ValueError):
        if axis != "impact_time_step" or not isinstance(left_axis_value, list) or not isinstance(right_axis_value, list):
            report["status"] = "inconclusive_invalid_axis_metadata"
            report["reasons"] = ["comparison-axis values are not finite numeric scalars or declared timestep lists"]
            return report

    metrics: dict[str, Any] = {}
    if axis == "time_step":
        refinement = _time_step_refinement_admission(
            left, right, left_axis_value, right_axis_value)
        metrics["actual_compression_timestep_refinement"] = refinement
        if refinement.get("status") != "available":
            report["status"] = "incompatible_physical_schedule"
            report["reasons"] = ["accepted native substep dt/count did not refine both compression branches"] + refinement.get("reasons", [])
            report["metrics"] = metrics
            return report
    if scope in ("compression", "impact", "all"):
        curve_a, status_a = _compression_curve(left)
        curve_b, status_b = _compression_curve(right)
        if not curve_a or not curve_b:
            metrics["compression_force_work"] = {"status": "inconclusive_missing_or_invalid_curve",
                                                  "left": status_a, "right": status_b}
            report["missing_observables"].append("complete nominal compression force-stroke-work curve")
        else:
            metrics["compression_force_work"] = _compare_force_curves(
                curve_a, curve_b, status_a.get("reported_final_work_J"),
                status_b.get("reported_final_work_J"))
    if scope in ("scoring", "assembly", "all"):
        metrics["accepted_grip_moment_curve"] = _compare_grip_moment_curves(left, right)
        if metrics["accepted_grip_moment_curve"].get("status") != "available":
            report["missing_observables"].append(
                "accepted clamp/fixture reaction-moment curves for matching actions")
        metrics["released_springback"] = _compare_released_springback(left, right)
        if metrics["released_springback"].get("status") != "available":
            report["missing_observables"].append(
                "accepted released score-line springback with passing endpoint shape fits")
    if scope in ("impact", "all"):
        impact = _compare_impact_histories(left, right)
        metrics["impact_dynamics"] = impact
        if impact.get("status") != "available":
            report["missing_observables"].append("complete accepted impact force/position/energy histories")
        report["missing_observables"].append("impact moment curve")
    metrics["damage_and_debond"] = {
        "left": _history_metrics(left), "right": _history_metrics(right),
        "geometric_paper_crack_fraction_difference": None,
        "geometric_starch_debond_fraction_difference": None,
        "fully_failed_cohesive_capacity_area": {},
        "status": "inconclusive_geometric_area_not_exported",
    }
    history_a = metrics["damage_and_debond"]["left"]
    history_b = metrics["damage_and_debond"]["right"]
    area_a = history_a.get("fully_failed_cohesive_capacity_area", {})
    area_b = history_b.get("fully_failed_cohesive_capacity_area", {})
    if (history_a.get("fully_failed_cohesive_capacity_area_status") ==
            "available_zero_tensile_shear_capacity_not_geometric_open_area" and
            history_b.get("fully_failed_cohesive_capacity_area_status") ==
            "available_zero_tensile_shear_capacity_not_geometric_open_area"):
        capacity_differences = {}
        for role in ("starch", "hma"):
            left_capacity, right_capacity = area_a.get(role, {}), area_b.get(role, {})
            if left_capacity.get("status") == right_capacity.get("status") == \
                    "available_zero_tensile_shear_capacity_not_geometric_open_area":
                left_fraction = left_capacity["fully_failed_cohesive_area_fraction"]
                right_fraction = right_capacity["fully_failed_cohesive_area_fraction"]
                left_denominator = left_capacity["activated_reference_area_denominator_m2"]
                right_denominator = right_capacity["activated_reference_area_denominator_m2"]
                same_coverage = math.isclose(left_denominator, right_denominator,
                                             rel_tol=1e-6, abs_tol=1e-12)
                capacity_differences[role] = {
                    "status": "available" if same_coverage else "incompatible_activated_reference_area",
                    "left_fraction": left_fraction,
                    "right_fraction": right_fraction,
                    "absolute_difference": abs(right_fraction - left_fraction) if same_coverage else None,
                    "left_activated_reference_area_m2": left_denominator,
                    "right_activated_reference_area_m2": right_denominator,
                    "activation_area_relative_tolerance": 1e-6,
                }
            elif left_capacity.get("status") != right_capacity.get("status"):
                capacity_differences[role] = {"status": "incompatible_interface_activation_or_enablement"}
        metrics["damage_and_debond"]["fully_failed_cohesive_capacity_area"] = capacity_differences
        metrics["damage_and_debond"]["status"] = (
            "incompatible_activated_reference_area"
            if any(value.get("status", "").startswith("incompatible")
                   for value in capacity_differences.values())
            else "capacity_loss_area_available_geometric_open_area_unexported")
    report["missing_observables"].extend(("geometric paper crack area fraction", "geometric starch/HMA debond area fraction"))
    if (history_a.get("fully_failed_cohesive_capacity_area_status") !=
            "available_zero_tensile_shear_capacity_not_geometric_open_area" or
            history_b.get("fully_failed_cohesive_capacity_area_status") !=
            "available_zero_tensile_shear_capacity_not_geometric_open_area"):
        report["missing_observables"].append(
            "accepted fully failed cohesive capacity reference area fraction")
    metrics["damage_and_debond"]["constitutive_debond_fraction_gate"] = _capacity_loss_debond_gate(
        history_a, history_b, metrics["damage_and_debond"]["fully_failed_cohesive_capacity_area"])
    report["metrics"] = metrics
    report["status"] = "descriptive_metrics_available_with_unresolved_observables"
    report["claim_boundary"] = "Numerical comparison of admitted native exports only; no physical qualification or material calibration."
    return report


def analyze_compression_rate(directory: Path) -> dict[str, Any]:
    """Compare nominal and half-speed accepted compression branches in one run."""
    run = load_run(directory)
    reasons = _scope_gate(run, "compression", require_rate_pass=False)
    report: dict[str, Any] = {
        "schema": SCHEMA, "kind": "within-run compression-rate sensitivity",
        "path": str(run.path), "admission": run.admission, "files": run.files,
        "status": "inconclusive", "metrics": {},
    }
    if reasons:
        report["status"] = "inconclusive_or_failed_run_admission"
        report["reasons"] = reasons
        return report
    nominal, nominal_status = _compression_curve(run, "nominal")
    half, half_status = _compression_curve(run, "half_speed")
    if not nominal or not half:
        report["status"] = "inconclusive_missing_or_invalid_curve"
        report["metrics"] = {"nominal": nominal_status, "half_speed": half_status}
        return report
    _, nominal_status = _compression_curve(run, "nominal")
    _, half_status = _compression_curve(run, "half_speed")
    comparison = _compare_force_curves(
        nominal, half, nominal_status.get("reported_final_work_J"),
        half_status.get("reported_final_work_J"))
    comparison["native_rate_gate"] = (run.native or {}).get("compression_rate_sensitivity")
    comparison["nominal_speed_m_per_s"] = (run.native or {}).get("compression_platen_speed_m_per_s")
    comparison["half_speed_m_per_s"] = (run.native or {}).get("compression_half_speed_platen_speed_m_per_s")
    comparison["is_time_refinement"] = False
    report["metrics"] = {"compression_force_work": comparison}
    report["status"] = "descriptive_rate_metrics_available"
    report["claim_boundary"] = "Within-run rate sensitivity only; it is not timestep refinement, full-stage convergence, or physical qualification."
    return report


def _numeric_axis_levels(value: Any) -> list[float] | None:
    raw = value if isinstance(value, list) else [value]
    if not raw:
        return None
    try:
        levels = [float(item) for item in raw]
    except (TypeError, ValueError):
        return None
    return levels if all(math.isfinite(item) for item in levels) else None


def _pair_metric_distances(report: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    compression = report.get("metrics", {}).get("compression_force_work", {})
    if compression.get("status") == "available":
        for field in ("relative_peak_force_difference", "normalized_force_curve_rms",
                      "relative_work_difference"):
            value = compression.get(field)
            if isinstance(value, (float, int)) and math.isfinite(float(value)):
                values[f"compression.{field}"] = float(value)
    impact = report.get("metrics", {}).get("impact_dynamics", {})
    if impact.get("status") == "available":
        for case_id, fields in impact.get("per_case", {}).items():
            if fields.get("status") != "available":
                continue
            for field in ("normalized_force_curve_rms", "body_height_path_rms_m",
                          "mechanical_energy_path_rms_J"):
                value = fields.get(field)
                if isinstance(value, (float, int)) and math.isfinite(float(value)):
                    values[f"impact.{case_id}.{field}"] = float(value)
    moment = report.get("metrics", {}).get("accepted_grip_moment_curve", {})
    if moment.get("status") == "available":
        for field in ("normalized_total_moment_vector_curve_rms",
                      "relative_peak_moment_difference"):
            value = moment.get(field)
            if isinstance(value, (float, int)) and math.isfinite(float(value)):
                values[f"grip_moment.{field}"] = float(value)
    springback = report.get("metrics", {}).get("released_springback", {})
    if springback.get("status") == "available":
        for field in ("rms_springback_difference_deg",
                      "rms_released_turn_difference_deg",
                      "maximum_absolute_springback_difference_deg"):
            value = springback.get(field)
            if isinstance(value, (float, int)) and math.isfinite(float(value)):
                values[f"released_springback.{field}"] = float(value)
    return values


def compare_triplet(coarse_dir: Path, middle_dir: Path, fine_dir: Path,
                    axis: str, scope: str = "all") -> dict[str, Any]:
    """Describe whether available adjacent differences decrease; never certify convergence."""
    if axis in ("score_depth", "score_speed"):
        return {"schema": SCHEMA, "kind": "full-stage three-level comparison",
                "axis": axis, "scope": scope, "status": "incompatible_pair",
                "convergence_claim": False,
                "reasons": ["score depth and speed are sensitivity axes, not refinement axes"]}
    runs = [load_run(coarse_dir), load_run(middle_dir), load_run(fine_dir)]
    levels = [_numeric_axis_levels(_axis_values(run, axis)) for run in runs]
    if any(level is None for level in levels):
        return {"schema": SCHEMA, "kind": "full-stage three-level comparison",
                "axis": axis, "scope": scope, "status": "inconclusive_invalid_axis_metadata",
                "convergence_claim": False,
                "reasons": ["each level needs finite numeric axis metadata"]}
    # Mesh levels refine upward; time steps refine downward. Impact manifests
    # provide a parallel dt list for their mass/location branches.
    direction = -1.0 if axis in ("time_step", "impact_time_step") else 1.0
    assert levels[0] is not None and levels[1] is not None and levels[2] is not None
    if not (len(levels[0]) == len(levels[1]) == len(levels[2])):
        return {"schema": SCHEMA, "kind": "full-stage three-level comparison",
                "axis": axis, "scope": scope, "status": "incompatible_axis_shape",
                "convergence_claim": False,
                "reasons": ["three levels have different numbers of axis values"]}
    for index in range(len(levels[0])):
        a, b, c = levels[0][index], levels[1][index], levels[2][index]
        if not (direction * (b-a) > 0.0 and direction * (c-b) > 0.0):
            return {"schema": SCHEMA, "kind": "full-stage three-level comparison",
                    "axis": axis, "scope": scope, "status": "incompatible_axis_order",
                    "convergence_claim": False,
                    "reasons": ["axis values are not strictly ordered coarse-to-fine"]}

    coarse_middle = compare_runs(coarse_dir, middle_dir, axis, scope)
    middle_fine = compare_runs(middle_dir, fine_dir, axis, scope)
    pair_reports = [coarse_middle, middle_fine]
    if any(item.get("status") != "descriptive_metrics_available_with_unresolved_observables"
           for item in pair_reports):
        return {"schema": SCHEMA, "kind": "full-stage three-level comparison",
                "axis": axis, "scope": scope, "status": "inconclusive_pair_admission",
                "convergence_claim": False, "pair_reports": pair_reports}
    distances = [_pair_metric_distances(item) for item in pair_reports]
    common_keys = set(distances[0]) & set(distances[1])
    if not common_keys:
        return {"schema": SCHEMA, "kind": "full-stage three-level comparison",
                "axis": axis, "scope": scope, "status": "inconclusive_missing_comparable_observables",
                "convergence_claim": False, "pair_reports": pair_reports,
                "reasons": ["no measured force, work, or impact metric is available at both adjacent levels"]}
    comparisons = {}
    for key in sorted(common_keys):
        first, second = distances[0][key], distances[1][key]
        comparisons[key] = {"coarse_to_middle_difference": first,
                            "middle_to_fine_difference": second,
                            "decreased": second < first}
    unresolved = sorted(set(coarse_middle.get("missing_observables", [])) |
                        set(middle_fine.get("missing_observables", [])))
    all_decreased = all(item["decreased"] for item in comparisons.values())
    status = "descriptive_three_level_decrease_only" if all_decreased else "descriptive_no_monotone_decrease"
    return {"schema": SCHEMA, "kind": "full-stage three-level comparison",
            "axis": axis, "scope": scope, "status": status,
            "axis_levels": levels, "pair_metric_differences": comparisons,
            "pair_reports": pair_reports, "unresolved_observables": unresolved,
            "convergence_claim": False,
            "claim_boundary": "A decreasing set of observed adjacent differences is descriptive only; missing full-stage observables prevent convergence qualification."}


CAPACITY_SCHEMA = "numi.cardboard.shipping-capacity-qualification.v2"
CAPACITY_PEAK_TOLERANCE = 0.02
CAPACITY_CURVE_TOLERANCE = 0.05
CAPACITY_WORK_TOLERANCE = 0.02


def _capacity_case(run: StageRun) -> tuple[dict[str, Any] | None, list[str]]:
    reasons = _scope_gate(run, "compression")
    native = run.native or {}
    if run.admission.get("status") != "completed_candidate":
        reasons.append("run_not_complete_and_certificate_admitted")
    if native.get("native_slow_compression_completed") is not True:
        reasons.append("full_nominal_compression_not_complete")
    if native.get("native_compression_rate_sensitivity_gate_passed") is not True:
        reasons.append("nominal_half_speed_sensitivity_gate_not_passed")
    try:
        terminal_fraction = float(native["compression_stroke_fraction_of_inside_height"])
        if not math.isfinite(terminal_fraction) or not math.isclose(
                terminal_fraction, 0.70, rel_tol=0.0, abs_tol=1e-9):
            reasons.append("nominal_curve_does_not_reach_70_percent_postpeak_endpoint")
    except (KeyError, TypeError, ValueError):
        terminal_fraction = math.nan
        reasons.append("missing_nominal_terminal_stroke_fraction")
    curve_path = run.path / "shipping-compression-nominal-curve.csv"
    if not curve_path.is_file():
        reasons.append("missing_exact_nominal_curve_artifact")
        nominal_curve_sha = ""
        curve_count = 0
    else:
        nominal_curve_sha = _sha(curve_path.read_bytes())
        try:
            _, curve_rows = _rows(curve_path)
            curve_count = len(curve_rows)
            if curve_count < 3:
                reasons.append("nominal_curve_has_too_few_accepted_samples")
        except (OSError, csv.Error, StageStudyError, ValueError):
            curve_count = 0
            reasons.append("invalid_nominal_curve_artifact")
    native_curve_sha = native.get("nominal_compression_curve_sha256")
    if not isinstance(native_curve_sha, str) or native_curve_sha != nominal_curve_sha:
        reasons.append("nominal_curve_hash_does_not_match_native_result")
    peak_index = _safe_int(native.get("compression_peak_sample_index"))
    if peak_index < 0 or peak_index >= curve_count:
        reasons.append("compression_peak_index_outside_nominal_curve")
    try:
        expected_terminal_stroke = float(native["compression_stroke_m"])
        if not curve_path.is_file() or not math.isclose(
                _finite(_rows(curve_path)[1][-1], "stroke_m", curve_path.name),
                expected_terminal_stroke, rel_tol=0.0, abs_tol=1e-9):
            reasons.append("nominal_curve_endpoint_differs_from_native_completed_stroke")
    except (KeyError, TypeError, ValueError, IndexError, OSError, StageStudyError):
        reasons.append("missing_or_invalid_native_completed_stroke")
    required_digests = (
        "shipping_source_material_binding_sha256",
        "assembled_checkpoint_sha256_for_capacity_binding",
        "assembled_rigid_sidecar_sha256_for_capacity_binding",
    )
    for key in required_digests:
        value = native.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value):
            reasons.append(f"missing_or_invalid_source_binding:{key}")
    fingerprints = []
    for key in ("source_physics_fingerprint", "device_program_fingerprint",
                "compiled_world_fingerprint"):
        value = native.get(key)
        if not isinstance(value, int) or value <= 0:
            reasons.append(f"missing_or_invalid_fingerprint:{key}")
        fingerprints.append(value)
    if len(fingerprints) == 3 and fingerprints[1] != fingerprints[2]:
        reasons.append("device_program_and_compiled_world_fingerprints_differ")
    if reasons:
        return None, sorted(set(reasons))
    return {
        "run": run,
        "source_physics_fingerprint": fingerprints[0],
        "device_program_fingerprint": fingerprints[1],
        "compiled_world_fingerprint": fingerprints[2],
        "source_binding_sha256": native[required_digests[0]],
        "assembled_matter_checkpoint_sha256": native[required_digests[1]],
        "assembled_rigid_sidecar_sha256": native[required_digests[2]],
        "nominal_curve_sha256": nominal_curve_sha,
        "nominal_peak_sample_index": peak_index,
        "nominal_accepted_sample_count": curve_count,
        "nominal_terminal_stroke_fraction": terminal_fraction,
    }, []


def _capacity_triplet_report(coarse: Path, middle: Path, fine: Path,
                             axis: str) -> tuple[dict[str, Any], dict[str, Any]]:
    report = compare_triplet(coarse, middle, fine, axis, "compression")
    reasons: list[str] = []
    cases = [load_run(path) for path in (coarse, middle, fine)]
    for index, run in enumerate(cases):
        _, case_reasons = _capacity_case(run)
        reasons.extend(f"level_{index}:{reason}" for reason in case_reasons)
    pairs = report.get("pair_reports", [])
    if len(pairs) != 2:
        reasons.append("three-level-study-missing-two-adjacent-pair-reports")
        coarse_metrics = {}
        finest_metrics = {}
    else:
        for pair_index, pair in enumerate(pairs):
            if pair.get("status") != "descriptive_metrics_available_with_unresolved_observables":
                reasons.append(f"adjacent_pair_{pair_index}_not_admitted")
            metrics = pair.get("metrics", {}).get("compression_force_work", {})
            if metrics.get("status") != "available":
                reasons.append(f"adjacent_pair_{pair_index}_missing-compression-metrics")
            if metrics.get("comparison_is_complete_endpoint") is not True:
                reasons.append(f"adjacent_pair_{pair_index}_endpoint_mismatch")
            reference_stroke = float((cases[1].native or {}).get(
                "compression_stroke_m", math.nan))
            for key in ("left_reported_final_stroke_m", "right_reported_final_stroke_m"):
                if not math.isclose(float(metrics.get(key, math.nan)),
                                    reference_stroke,
                                    rel_tol=0.0, abs_tol=1e-9):
                    reasons.append(f"adjacent_pair_{pair_index}_incomplete_stroke_curve")
        coarse_metrics = pairs[0].get("metrics", {}).get("compression_force_work", {})
        finest_metrics = pairs[-1].get("metrics", {}).get("compression_force_work", {})
    history_gates = []
    for pair_index, pair in enumerate(pairs):
        gates = _capacity_history_gates(pair)
        history_gates.append(gates)
        for name, gate in gates.items():
            if gate.get("status") not in ("pass", "not_enabled"):
                reasons.append(f"adjacent_pair_{pair_index}_{name}_{gate.get('status', 'inconclusive')}")
    if len(history_gates) != 2:
        reasons.append("missing_two_adjacent_damage_and_debond_comparisons")
    metric_names = (
        "relative_peak_force_difference",
        "normalized_force_curve_rms",
        "relative_work_difference",
    )
    coarse_values = [float(coarse_metrics.get(name, math.inf))
                     for name in metric_names]
    finest_values = [float(finest_metrics.get(name, math.inf))
                     for name in metric_names]
    finite_pairs = all(math.isfinite(a) and math.isfinite(b) and a >= 0.0 and b >= 0.0
                       for a, b in zip(coarse_values, finest_values))
    adjacent_differences_decrease = finite_pairs and all(
        b < a for a, b in zip(coarse_values, finest_values))
    if not adjacent_differences_decrease:
        reasons.append("adjacent_pair_metric_differences_do_not_decrease")
    threshold_pass = (
        finest_metrics.get("status") == "available" and
        float(finest_metrics.get("relative_peak_force_difference", math.inf)) <= CAPACITY_PEAK_TOLERANCE and
        float(finest_metrics.get("normalized_force_curve_rms", math.inf)) <= CAPACITY_CURVE_TOLERANCE and
        float(finest_metrics.get("relative_work_difference", math.inf)) <= CAPACITY_WORK_TOLERANCE)
    if not threshold_pass:
        reasons.append("finest_adjacent_pair_exceeds_capacity_tolerance")
    return report, {
        "qualified": not reasons,
        "level_count": len(cases),
        "all_cases_complete": not any(reason.startswith("level_") for reason in reasons),
        "finest_pair_within_tolerance": threshold_pass,
        "accepted_damage_and_debond_gates": history_gates,
        "capacity_requires_damage_and_debond_observables": True,
        "adjacent_pair_differences_decrease": adjacent_differences_decrease,
        "coarse_middle_peak_difference": coarse_values[0] if math.isfinite(coarse_values[0]) else None,
        "coarse_middle_curve_rms": coarse_values[1] if math.isfinite(coarse_values[1]) else None,
        "coarse_middle_work_difference": coarse_values[2] if math.isfinite(coarse_values[2]) else None,
        "middle_fine_peak_difference": finest_values[0] if math.isfinite(finest_values[0]) else None,
        "middle_fine_curve_rms": finest_values[1] if math.isfinite(finest_values[1]) else None,
        "middle_fine_work_difference": finest_values[2] if math.isfinite(finest_values[2]) else None,
        "peak_difference": finest_metrics.get("relative_peak_force_difference"),
        "curve_rms": finest_metrics.get("normalized_force_curve_rms"),
        "work_difference": finest_metrics.get("relative_work_difference"),
        "reasons": reasons,
    }


def create_capacity_qualification(reference_dir: Path,
                                  mesh_coarse: Path, mesh_middle: Path, mesh_fine: Path,
                                  time_coarse: Path, time_middle: Path, time_fine: Path,
                                  output_properties: Path) -> dict[str, Any]:
    """Create a bounded Fpeak admission file from completed mesh and time triplets.

    The shared middle case must be the exact reference run, so the emitted
    record binds to the same accepted assembly checkpoint and nominal curve
    that a later probe invocation must reproduce byte-for-byte.
    """
    reference_path = reference_dir.expanduser().resolve()
    if (reference_path != mesh_middle.expanduser().resolve() or
            reference_path != time_middle.expanduser().resolve()):
        raise StageStudyError("reference run must be the shared middle level of both refinement triplets")
    reference = load_run(reference_path)
    identity, identity_reasons = _capacity_case(reference)
    mesh_report, mesh = _capacity_triplet_report(
        mesh_coarse, mesh_middle, mesh_fine, "mesh_nx")
    time_report, time = _capacity_triplet_report(
        time_coarse, time_middle, time_fine, "time_step")
    report = {
        "schema": CAPACITY_SCHEMA,
        "kind": "source-bound numerical Fpeak capacity evidence",
        "scope": "complete accepted 70-percent shipping compression curve and first local peak only",
        "status": "qualified_for_numerical_capacity_only" if identity and mesh["qualified"] and time["qualified"] else "not_qualified",
        "reference_run": str(reference_path),
        "reference_admission": reference.admission,
        "identity_reasons": identity_reasons,
        "mesh_study": mesh,
        "time_study": time,
        "tolerances": {"peak_relative": CAPACITY_PEAK_TOLERANCE,
                       "normalized_full_curve_rms": CAPACITY_CURVE_TOLERANCE,
                       "work_relative": CAPACITY_WORK_TOLERANCE},
        "claim_boundary": "Numerical capacity evidence for Fpeak mass sizing only; no material-grade calibration or physical shipping qualification.",
    }
    output_properties.parent.mkdir(parents=True, exist_ok=True)
    mesh_path = output_properties.with_name(output_properties.stem + ".mesh-study.json")
    time_path = output_properties.with_name(output_properties.stem + ".time-study.json")
    _write_new(mesh_path, mesh_report)
    _write_new(time_path, time_report)
    report["mesh_report_file"] = mesh_path.name
    report["mesh_report_sha256"] = _sha(mesh_path.read_bytes())
    report["time_report_file"] = time_path.name
    report["time_report_sha256"] = _sha(time_path.read_bytes())
    if not identity or not mesh["qualified"] or not time["qualified"]:
        _write_new(output_properties.with_suffix(".analysis.json"), report)
        return report
    assert identity is not None
    values = {
        "schema": CAPACITY_SCHEMA,
        "status": "qualified_for_numerical_capacity_only",
        "board_recipe_id": str(reference.manifest.get("preset", "")),
        "source_physics_fingerprint": str(identity["source_physics_fingerprint"]),
        "device_program_fingerprint": str(identity["device_program_fingerprint"]),
        "compiled_world_fingerprint": str(identity["compiled_world_fingerprint"]),
        "source_binding_sha256": identity["source_binding_sha256"],
        "assembled_matter_checkpoint_sha256": identity["assembled_matter_checkpoint_sha256"],
        "assembled_rigid_sidecar_sha256": identity["assembled_rigid_sidecar_sha256"],
        "nominal_curve_sha256": identity["nominal_curve_sha256"],
        "nominal_peak_sample_index": str(identity["nominal_peak_sample_index"]),
        "nominal_accepted_sample_count": str(identity["nominal_accepted_sample_count"]),
        "nominal_terminal_stroke_fraction": format(identity["nominal_terminal_stroke_fraction"], ".17g"),
        "nominal_full_stroke_completed": "true",
        "nominal_rate_sensitivity_passed": "true",
        "mesh_status": "qualified",
        "mesh_report_file": mesh_path.name,
        "mesh_report_sha256": report["mesh_report_sha256"],
        "mesh_level_count": str(mesh["level_count"]),
        "mesh_all_cases_complete": "true",
        "mesh_finest_pair_within_tolerance": "true",
        "mesh_finest_peak_difference": format(float(mesh["peak_difference"]), ".17g"),
        "mesh_finest_curve_rms": format(float(mesh["curve_rms"]), ".17g"),
        "mesh_finest_work_difference": format(float(mesh["work_difference"]), ".17g"),
        "mesh_coarse_middle_peak_difference": format(float(mesh["coarse_middle_peak_difference"]), ".17g"),
        "mesh_coarse_middle_curve_rms": format(float(mesh["coarse_middle_curve_rms"]), ".17g"),
        "mesh_coarse_middle_work_difference": format(float(mesh["coarse_middle_work_difference"]), ".17g"),
        "time_status": "qualified",
        "time_report_file": time_path.name,
        "time_report_sha256": report["time_report_sha256"],
        "time_level_count": str(time["level_count"]),
        "time_all_cases_complete": "true",
        "time_finest_pair_within_tolerance": "true",
        "time_finest_peak_difference": format(float(time["peak_difference"]), ".17g"),
        "time_finest_curve_rms": format(float(time["curve_rms"]), ".17g"),
        "time_finest_work_difference": format(float(time["work_difference"]), ".17g"),
        "time_coarse_middle_peak_difference": format(float(time["coarse_middle_peak_difference"]), ".17g"),
        "time_coarse_middle_curve_rms": format(float(time["coarse_middle_curve_rms"]), ".17g"),
        "time_coarse_middle_work_difference": format(float(time["coarse_middle_work_difference"]), ".17g"),
    }
    if output_properties.exists():
        raise StageStudyError(f"refusing to overwrite capacity evidence: {output_properties}")
    output_properties.write_text(
        "# Generated only from six complete native three-level comparison cases.\n" +
        "".join(f"{key}={values[key]}\n" for key in sorted(values)),
        encoding="utf-8")
    report["qualification_file"] = str(output_properties.resolve())
    report["qualification_file_sha256"] = _sha(output_properties.read_bytes())
    _write_new(output_properties.with_suffix(".analysis.json"), report)
    return report


def make_plan(reference_dir: Path) -> dict[str, Any]:
    reference = load_run(reference_dir)
    plan = {
        "schema": PLAN_SCHEMA,
        "status": "reference_admitted" if reference.admission["status"] == "completed_candidate" else "reference_not_admitted",
        "reference": {"path": str(reference.path), "admission": reference.admission,
                      "files": reference.files,
                      "source_bindings": reference.source_bindings,
                      "manifest_sha256": reference.files.get("manifest.json")},
        "analysis_source": {"path": str(Path(__file__).resolve()), "sha256": _sha(Path(__file__).read_bytes())},
        "execution": "plan only; this tool never starts native or Metal work",
        "rule": "A failed, rejected, prefix, preflight-only, or certificate-incomplete reference is not a baseline. Each candidate pair must independently pass full accepted-history admission.",
        "comparison_families": [
            {"id": "mesh_nx", "axis": "mesh_nx", "levels": "baseline, 2x nx, 4x nx", "scope": "all", "change_only": "nx_per_pitch", "requires": "same geometry, material/law hashes, stages, physical schedule and solver controls"},
            {"id": "mesh_ny", "axis": "mesh_ny", "levels": "baseline, 2x ny, 4x ny", "scope": "all", "change_only": "ny", "requires": "same geometry, material/law hashes, stages, physical schedule and solver controls"},
            {"id": "mesh_thickness", "axis": "mesh_thickness", "levels": "baseline, 2x and 4x thickness slices", "scope": "all", "change_only": "thickness_slices", "requires": "same geometry, material/law hashes, stages, physical schedule and solver controls"},
            {"id": "time_step", "axis": "time_step", "levels": "dt, dt/2, dt/4", "scope": "all", "change_only": "solver dt with derived control substeps", "requires": "same physical duration and actuator/tool path; accepted native compression dt/count must verify both nominal and half-speed branches, with equal duration and smaller maximum dt at each finer level"},
            {"id": "score_depth_sensitivity", "axis": "score_depth", "levels": [0.1, 0.25, 0.4], "scope": "scoring", "branching": "fresh flat blank for each depth; never cumulative", "not_convergence": True},
            {"id": "score_speed_sensitivity", "axis": "score_speed", "levels": "declared nominal and half speed", "scope": "scoring", "change_only": "tool speed; same dt and stroke", "not_time_refinement": True},
            {"id": "compression_speed_sensitivity", "method": "within-run nominal-versus-half-speed branches", "levels_mm_per_min": [12.5, 6.25], "scope": "compression", "branching": "same released checkpoint", "not_time_refinement": True},
            {"id": "impact_time_step", "axis": "impact_time_step", "levels_s": [0.0005, 0.00025, 0.000125], "scope": "impact", "branching": "restore same released assembly for every mass/location/dt", "requires": "same mass, drop height, geometry, and initial state"},
        ],
        "required_native_observables": {
            "force_curve": "accepted native platen/tool impulse divided by each accepted dt, keyed by stage time or physical stroke; compression macro dt/count summaries are cross-checked against accepted native steps",
            "moment_curve": "box-grip-reactions.csv accepted total clamp/fixture wrench moment in N m, grouped by stage/action/score-line/environment; this is not the score punch contact moment",
            "work": "integral over accepted force/stroke or moment/rotation samples with a declared sign convention",
            "springback": "box-score-springback.csv held-to-released score-line angle changes after accepted clamp release and settle; endpoint shape residual and coverage flags must pass",
            "damage": "accepted paper internal-state damage index with element crack-plane reference areas; not geometric crack area",
            "debond": "accepted cohesive face reference area and damage/opening, with geometric-debond threshold explicitly declared; current CSV exports damage index only",
            "certificates": "accepted native step, status, certificate validity and residual/correction/volume/pressure fields for every step",
        },
        "available_metric_limits": {
            "shipping-compression.csv": "mean platen force in raw N versus accepted stroke and cumulative work, when complete; curve RMS difference is normalized by left-run RMS force, never by board width; actual timestep refinement is derived from accepted native rows and validated against macro observation duration/count/min/max dt",
            "box-grip-reactions.csv": "accepted grip wrench moments in raw N m, not width-normalized; moment-curve difference uses left-run RMS moment magnitude; score punch contact moment is not exported here",
            "box-score-springback.csv": "accepted endpoint shape-fit observations; the probe's engineering fit bound is not a published manufacturing tolerance",
            "shipping_history_observables.csv": "area-weighted constitutive paper/cohesive degradation indices; not crack/debond area fractions",
            "shipping_contact_observations.csv": "contact impulses and mean forces; these are not substituted for a score-tool moment curve",
        },
        "qualification": "A plan or descriptive comparison is not material calibration, physical cardboard validation, or convergence proof. A convergence conclusion requires complete spatial and temporal triples, decreasing force/work, accepted grip-moment and released-springback differences when the scope includes scoring/assembly, full-stage observables, and admitted accepted histories.",
    }
    plan["plan_sha256"] = _sha(_canonical(plan).encode())
    return plan


def _write_new(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise StageStudyError(f"refusing to overwrite existing report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_cmd = commands.add_parser("inspect", help="admit one retained full-box export")
    inspect_cmd.add_argument("--run", type=Path, required=True)
    inspect_cmd.add_argument("--output", type=Path, required=True)
    plan_cmd = commands.add_parser("plan", help="write a full-stage comparison matrix")
    plan_cmd.add_argument("--reference", type=Path, required=True)
    plan_cmd.add_argument("--output", type=Path, required=True)
    compare_cmd = commands.add_parser("compare", help="compare two already retained box runs")
    compare_cmd.add_argument("--left", type=Path, required=True)
    compare_cmd.add_argument("--right", type=Path, required=True)
    compare_cmd.add_argument("--axis", choices=AXES, required=True)
    compare_cmd.add_argument("--scope", choices=("scoring", "assembly", "compression", "impact", "all"), default="all")
    compare_cmd.add_argument("--output", type=Path, required=True)
    triplet_cmd = commands.add_parser("triplet", help="compare three refinement levels without promoting incomplete observables")
    triplet_cmd.add_argument("--coarse", type=Path, required=True)
    triplet_cmd.add_argument("--middle", type=Path, required=True)
    triplet_cmd.add_argument("--fine", type=Path, required=True)
    triplet_cmd.add_argument("--axis", choices=AXES, required=True)
    triplet_cmd.add_argument("--scope", choices=("scoring", "assembly", "compression", "impact", "all"), default="all")
    triplet_cmd.add_argument("--output", type=Path, required=True)
    rate_cmd = commands.add_parser("compression-rate", help="compare nominal and half-speed branches in one complete run")
    rate_cmd.add_argument("--run", type=Path, required=True)
    rate_cmd.add_argument("--output", type=Path, required=True)
    capacity_cmd = commands.add_parser(
        "capacity-qualification",
        help="create source-bound Fpeak evidence only after complete mesh and timestep triplets pass stated tolerances")
    capacity_cmd.add_argument("--reference", type=Path, required=True,
                              help="exact shared middle run for both triplets")
    for prefix in ("mesh", "time"):
        capacity_cmd.add_argument(f"--{prefix}-coarse", type=Path, required=True)
        capacity_cmd.add_argument(f"--{prefix}-middle", type=Path, required=True)
        capacity_cmd.add_argument(f"--{prefix}-fine", type=Path, required=True)
    capacity_cmd.add_argument("--output-properties", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            run = load_run(args.run)
            report = {"schema": SCHEMA, "path": str(run.path), "admission": run.admission,
                      "files": run.files, "source_bindings": run.source_bindings,
                      "claim_boundary": "Admission of retained software evidence only."}
        elif args.command == "plan":
            report = make_plan(args.reference)
        elif args.command == "triplet":
            report = compare_triplet(args.coarse, args.middle, args.fine, args.axis, args.scope)
        elif args.command == "compression-rate":
            report = analyze_compression_rate(args.run)
        elif args.command == "capacity-qualification":
            report = create_capacity_qualification(
                args.reference,
                args.mesh_coarse, args.mesh_middle, args.mesh_fine,
                args.time_coarse, args.time_middle, args.time_fine,
                args.output_properties)
        else:
            report = compare_runs(args.left, args.right, args.axis, args.scope)
        _write_new(args.output, report)
        print(json.dumps({"status": report.get("status"), "output": str(args.output.resolve())}))
        return 0 if report.get("status") in (
            "reference_admitted", "descriptive_metrics_available_with_unresolved_observables",
            "descriptive_three_level_decrease_only", "descriptive_rate_metrics_available",
            "qualified_for_numerical_capacity_only",
        ) else 2
    except (StageStudyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "rejected", "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
