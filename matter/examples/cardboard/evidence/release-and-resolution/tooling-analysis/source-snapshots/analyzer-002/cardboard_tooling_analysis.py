#!/usr/bin/env python3
"""Fail-closed analysis of a Matter cardboard crease-tooling instrument run.

This reads ``manifest.json``, ``result.json``, ``observations.csv`` and
``tool-observations.csv`` from one probe output directory. It never changes
those files. An optional report path is created exclusively and is never
overwritten. A mechanics-instrument pass is not a fold, material, or physical
qualification.

Exit codes: 0 = complete instrument checks passed, 1 = complete run failed a
check or the evidence is malformed, 2 = incomplete/rejected run (inconclusive).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA = "numi.cardboard.tooling-analysis.v1"
MANIFEST_SCHEMA = "numi.cardboard.explicit-strip.v1"
RESULT_SCHEMA = "numi.cardboard.probe-result.v1"
MATERIAL_SHA_FIELDS = (
    ("liner", "liner_path", "liner_sha256"),
    ("medium", "medium_path", "medium_sha256"),
    ("glue", "glue_path", "glue_sha256"),
)
HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")

OBSERVATION_REQUIRED = {
    "step", "time_s", "environment", "arm", "target_angle_deg",
    "status_code", "step_accepted", "certificate_raw_accepted_flag",
    "certificate_residual", "certificate_correction",
    "certificate_volume", "certificate_pressure",
}
TOOL_REQUIRED = {
    "step", "environment", "commanded_end_travel_m", "commanded_speed_m_s",
    "punch_force_z_N", "punch_contact_count", "min_end_punch_node_gap_m",
    "min_anvil_node_gap_m", "step_accepted",
}
OBSERVATION_FLOAT_FIELDS = (
    "time_s", "target_angle_deg", "status_diagnostic_x", "status_diagnostic_y",
    "status_diagnostic_z", "status_diagnostic_w", "min_J", "max_J",
    "max_node_displacement_m", "reaction_x_N", "reaction_y_N", "reaction_z_N",
    "reaction_moment_y_Nm", "max_free_displacement_m", "max_free_speed_m_s",
    "kinetic_energy_J", "max_material_state_change", "certificate_residual",
    "certificate_correction", "certificate_volume", "certificate_pressure",
    "certificate_raw_accepted_flag", "measured_right_grip_angle_deg",
    "free_force_imbalance_l2_N", "free_force_imbalance_max_N",
)
TOOL_FLOAT_FIELDS = (
    "commanded_end_travel_m", "commanded_speed_m_s", "punch_force_z_N",
    "min_end_punch_node_gap_m", "min_anvil_node_gap_m",
)


class EvidenceError(ValueError):
    """The evidence cannot be interpreted under the declared probe schema."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot read {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"{path.name} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_hashes(run_dir: Path) -> dict[str, str]:
    names = (
        "manifest.json", "result.json", "observations.csv",
        "tool-observations.csv", "mesh.json", "mesh.json.gz",
        "compiled.nmatterpack", "compiled.nmatterpack.gz",
    )
    return {name: _sha256(run_dir / name) for name in names
            if (run_dir / name).is_file()}


def _coordinate_roundoff_allowance(geometry: dict[str, Any],
                                   tooling: dict[str, Any]) -> tuple[float, float]:
    """Return the probe-matched FP32 allowance and coordinate scale.

    Tool contact uses FP32 world coordinates. This tolerance accounts for
    coordinate quantization only; the contact barrier slop is deliberately
    not an allowed geometric penetration.
    """
    dimensions = [
        _finite_manifest_number(geometry, "length_m", "geometry", positive=True),
        _finite_manifest_number(geometry, "width_m", "geometry", positive=True),
        _finite_manifest_number(geometry, "total_height_m", "geometry", positive=True),
        _finite_manifest_number(tooling, "punch_radius_m", "tooling", positive=True),
    ]
    scale = max(dimensions)
    return 8.0 * (2.0 ** -23) * scale, scale


def _read_csv(path: Path, required: set[str], incomplete: list[str]) -> list[dict[str, str]] | None:
    if not path.is_file():
        incomplete.append(f"missing {path.name}")
        return None
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.reader(source)
            header = next(reader, None)
            if not header:
                incomplete.append(f"{path.name} is empty")
                return []
            if len(header) != len(set(header)):
                raise EvidenceError(f"{path.name} has duplicate CSV columns")
            missing = sorted(required.difference(header))
            if missing:
                raise EvidenceError(
                    f"{path.name} is missing required columns: {', '.join(missing)}")
            rows: list[dict[str, str]] = []
            for line_number, values in enumerate(reader, start=2):
                if len(values) != len(header):
                    incomplete.append(
                        f"{path.name}:{line_number} has {len(values)} fields; expected {len(header)}")
                    continue
                rows.append(dict(zip(header, values)))
            return rows
    except EvidenceError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        incomplete.append(f"cannot fully read {path.name}: {error}")
        return []


def _int_value(row: dict[str, str], name: str, context: str,
               incomplete: list[str]) -> int | None:
    try:
        return int(row[name], 10)
    except (KeyError, TypeError, ValueError):
        incomplete.append(f"{context}: invalid integer {name}")
        return None


def _float_value(row: dict[str, str], name: str, context: str,
                 nonfinite: list[str], incomplete: list[str]) -> float | None:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError):
        incomplete.append(f"{context}: invalid numeric {name}")
        return None
    if not math.isfinite(value):
        nonfinite.append(f"{context}: {name}={row[name]}")
        return None
    return value


def _finite_manifest_number(section: dict[str, Any], name: str,
                            context: str, *, positive: bool = False) -> float:
    raw = section.get(name)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise EvidenceError(f"{context}.{name} must be numeric")
    value = float(raw)
    if not math.isfinite(value) or (positive and value <= 0.0):
        raise EvidenceError(f"{context}.{name} must be finite" +
                            (" and positive" if positive else ""))
    return value


def _manifest_int(section: dict[str, Any], name: str, context: str) -> int:
    raw = section.get(name)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise EvidenceError(f"{context}.{name} must be a nonnegative integer")
    return raw


def _expected_travel(step: int, loading: int, hold: int,
                     unloading: int, indentation: float) -> float:
    if step < loading:
        return indentation * (step + 1) / loading
    step -= loading
    if step < hold:
        return indentation
    step -= hold
    if step < unloading:
        return indentation * (1.0 - (step + 1) / unloading)
    return 0.0


def _valid_sha(value: Any, label: str, *, optional: bool = False) -> str | None:
    if optional and value == "":
        return None
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise EvidenceError(f"manifest {label} is not a SHA-256 hex digest")
    return value.lower()


def _material_bindings(run_dir: Path, materials: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    bindings: dict[str, Any] = {}
    mismatches: list[str] = []
    for role, path_key, sha_key in MATERIAL_SHA_FIELDS:
        path_text = materials.get(path_key, "")
        digest = _valid_sha(materials.get(sha_key, ""), f"materials.{sha_key}",
                            optional=(role == "glue"))
        file_path = Path(path_text) if isinstance(path_text, str) and path_text else None
        if file_path is not None and not file_path.is_file() and not file_path.is_absolute():
            candidate = run_dir / file_path
            if candidate.is_file():
                file_path = candidate
        actual = _sha256(file_path) if file_path is not None and file_path.is_file() else None
        verified = None if actual is None or digest is None else actual == digest
        if verified is False:
            mismatches.append(f"{role} material file hash differs from manifest")
        bindings[role] = {
            "path": path_text if isinstance(path_text, str) else "",
            "manifest_sha256": digest,
            "file_available": actual is not None,
            "file_sha256": actual,
            "file_hash_verified": verified,
        }
    for name in ("regional_map_sha256", "frame_map_sha256"):
        value = materials.get(name, "")
        _valid_sha(value, f"materials.{name}", optional=True)
    bindings["regional_map_sha256"] = materials.get("regional_map_sha256", "") or None
    bindings["frame_map_sha256"] = materials.get("frame_map_sha256", "") or None
    return bindings, mismatches


def _index_rows(rows: list[dict[str, str]] | None, table_name: str,
                expected_steps: int, incomplete: list[str], nonfinite: list[str],
                is_tool: bool) -> tuple[dict[tuple[int, int], dict[str, Any]], int]:
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    if rows is None:
        return indexed, 0
    parsed_count = 0
    float_fields = TOOL_FLOAT_FIELDS if is_tool else OBSERVATION_FLOAT_FIELDS
    int_fields = ("step", "environment", "punch_contact_count", "step_accepted") if is_tool else (
        "step", "environment", "status_code", "step_accepted")
    for row_number, raw in enumerate(rows, start=2):
        context = f"{table_name}:{row_number}"
        parsed: dict[str, Any] = {}
        malformed = False
        for name in int_fields:
            value = _int_value(raw, name, context, incomplete)
            parsed[name] = value
            malformed = malformed or value is None
        for name in float_fields:
            if name not in raw:
                # Older probes may omit optional force-balance diagnostics.
                continue
            nonfinite_count = len(nonfinite)
            value = _float_value(raw, name, context, nonfinite, incomplete)
            parsed[name] = value
            # A syntactically valid NaN/Inf is a failed measurement, not a
            # missing CSV row. Keep its step key so complete-run checks can
            # fail for the actual nonfinite value instead of misclassifying
            # the whole run as incomplete.
            malformed = malformed or (value is None and len(nonfinite) == nonfinite_count)
        if malformed:
            continue
        step = parsed["step"]
        environment = parsed["environment"]
        if step < 0 or step >= expected_steps or environment not in (0, 1):
            incomplete.append(f"{context}: step/environment is outside the expected two-arm protocol")
            continue
        key = (step, environment)
        if key in indexed:
            incomplete.append(f"{context}: duplicate row for step={step}, environment={environment}")
            continue
        parsed.update({
            "arm": raw.get("arm", ""),
            "phase": raw.get("phase", ""),
        })
        indexed[key] = parsed
        parsed_count += 1
    return indexed, parsed_count


def analyze_run(run_directory: str | Path) -> dict[str, Any]:
    """Analyze one run directory and return a JSON-serializable report."""
    run_dir = Path(run_directory).expanduser().resolve()
    if not run_dir.is_dir():
        raise EvidenceError(f"run directory does not exist: {run_dir}")
    manifest = _read_json(run_dir / "manifest.json")
    result = _read_json(run_dir / "result.json")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise EvidenceError("unsupported cardboard manifest schema")
    if result.get("schema") != RESULT_SCHEMA:
        raise EvidenceError("unsupported cardboard result schema")

    tooling = manifest.get("tooling")
    solver = manifest.get("solver")
    geometry = manifest.get("geometry")
    materials = manifest.get("materials")
    if not isinstance(tooling, dict) or tooling.get("enabled") is not True:
        raise EvidenceError("manifest does not declare crease tooling enabled")
    if not isinstance(solver, dict) or not isinstance(geometry, dict) or not isinstance(materials, dict):
        raise EvidenceError("manifest is missing solver, geometry, or materials metadata")
    timing = tooling.get("timing")
    legacy_timing = isinstance(timing, str) and (
        "start-of-step body pose" in timing and "commanded end pose" in timing)
    native_endpoint_timing = isinstance(timing, str) and (
        "start-of-step pose" in timing and "preDynamics" in timing and
        "realized end pose" in timing and "postCommit" in timing)
    if not (legacy_timing or native_endpoint_timing):
        raise EvidenceError("manifest does not document start/preDynamics and end/postCommit tooling timing")

    dt = _finite_manifest_number(solver, "dt_s", "solver", positive=True)
    slop = _finite_manifest_number(solver, "contact_slop_m", "solver", positive=True)
    indentation = _finite_manifest_number(tooling, "indentation_m", "tooling")
    gap_roundoff_tolerance, mesh_coordinate_scale = _coordinate_roundoff_allowance(geometry, tooling)
    if indentation < 0.0:
        raise EvidenceError("tooling.indentation_m must be nonnegative")
    nose_clearance = _finite_manifest_number(
        tooling, "initial_nose_clearance_m", "tooling")
    anvil_clearance = _finite_manifest_number(
        tooling, "initial_anvil_clearance_m", "tooling")
    loading = _manifest_int(solver, "steps", "solver")
    hold = _manifest_int(solver, "hold_steps", "solver")
    unloading = _manifest_int(solver, "unload_steps", "solver")
    relaxation = _manifest_int(solver, "relax_steps", "solver")
    release = _manifest_int(solver, "release_steps", "solver")
    certificate_tolerances = {
        "certificate_residual": _finite_manifest_number(
            solver, "relative_residual_tolerance", "solver", positive=True),
        "certificate_volume": _finite_manifest_number(
            solver, "volume_tolerance", "solver", positive=True),
        "certificate_pressure": _finite_manifest_number(
            solver, "pressure_tolerance", "solver", positive=True),
    }
    expected_steps = loading + hold + unloading + relaxation + release
    if expected_steps <= 0 or loading <= 0:
        raise EvidenceError("manifest crease protocol has no loading steps")

    manifest_fingerprint = manifest.get("compiled_world_fingerprint")
    result_fingerprint = result.get("compiled_world_fingerprint")
    if isinstance(manifest_fingerprint, bool) or not isinstance(manifest_fingerprint, int) or manifest_fingerprint < 0:
        raise EvidenceError("manifest compiled_world_fingerprint must be a nonnegative integer")
    if result_fingerprint != manifest_fingerprint:
        raise EvidenceError("manifest/result compiled-world fingerprints do not match")

    material_bindings, material_hash_mismatches = _material_bindings(run_dir, materials)
    incomplete: list[str] = []
    nonfinite: list[str] = []
    observation_rows = _read_csv(
        run_dir / "observations.csv", OBSERVATION_REQUIRED, incomplete)
    tool_rows = _read_csv(run_dir / "tool-observations.csv", TOOL_REQUIRED, incomplete)
    observations, observation_count = _index_rows(
        observation_rows, "observations.csv", expected_steps, incomplete, nonfinite, False)
    tools, tool_count = _index_rows(
        tool_rows, "tool-observations.csv", expected_steps, incomplete, nonfinite, True)

    expected_keys = {(step, env) for step in range(expected_steps) for env in (0, 1)}
    for table_name, indexed in (("observations.csv", observations),
                                ("tool-observations.csv", tools)):
        missing = sorted(expected_keys.difference(indexed))
        if missing:
            incomplete.append(f"{table_name} is missing {len(missing)} expected step/environment rows")

    status = result.get("status")
    accepted_steps = result.get("accepted_steps")
    if status != "completed":
        incomplete.append(f"result status is {status!r}, not completed")
    if isinstance(accepted_steps, bool) or not isinstance(accepted_steps, int):
        incomplete.append("result accepted_steps is missing or invalid")
    elif accepted_steps != expected_steps:
        incomplete.append(f"result accepted_steps={accepted_steps}, expected {expected_steps}")

    rejected_rows: list[str] = []
    certificate_errors: list[str] = []
    for (step, environment), row in observations.items():
        if row.get("status_code") != 0 or row.get("step_accepted") != 1:
            rejected_rows.append(f"step={step}, environment={environment}")
        raw_flag = row.get("certificate_raw_accepted_flag")
        accepted_claim = row.get("step_accepted") == 1 and row.get("status_code") == 0
        raw_certificate_claim = row.get("status_code") == 0 and raw_flag is not None and raw_flag >= 0.5
        if accepted_claim and (raw_flag is None or raw_flag < 0.5):
            certificate_errors.append(
                f"step={step}, environment={environment}: accepted row lacks valid raw certificate flag")
        if accepted_claim or raw_certificate_claim:
            for field, tolerance in certificate_tolerances.items():
                value = row.get(field)
                if value is None or value > tolerance:
                    certificate_errors.append(
                        f"step={step}, environment={environment}: {field}={value!r} exceeds {tolerance:.9g}")
    if rejected_rows:
        incomplete.append("rejected observation rows: " + "; ".join(rejected_rows[:8]))

    accepted_mismatch: list[str] = []
    if observations and tools:
        for key in expected_keys.intersection(observations).intersection(tools):
            obs_flag = observations[key].get("step_accepted")
            tool_flag = tools[key].get("step_accepted")
            if obs_flag != tool_flag:
                accepted_mismatch.append(f"step={key[0]}, environment={key[1]}")
    if accepted_mismatch:
        incomplete.append("main/tool acceptance flags disagree: " + "; ".join(accepted_mismatch[:8]))

    arm_mismatches: list[str] = []
    for (step, environment), row in observations.items():
        expected_arm = "indented" if environment == 0 else "stationary_tool_reference"
        if row.get("arm") != expected_arm:
            arm_mismatches.append(f"step={step}, environment={environment}, arm={row.get('arm')!r}")

    travel_errors: list[str] = []
    expected_command: dict[str, dict[str, float]] = {"indented": {}, "stationary_tool_reference": {}}
    for environment, label in ((0, "indented"), (1, "stationary_tool_reference")):
        previous = 0.0
        for step in range(expected_steps):
            row = tools.get((step, environment))
            if row is None:
                continue
            travel = row.get("commanded_end_travel_m")
            speed = row.get("commanded_speed_m_s")
            if travel is None or speed is None:
                continue
            expected_travel = _expected_travel(
                step, loading, hold, unloading, indentation) if environment == 0 else 0.0
            expected_speed = -(expected_travel - previous) / dt
            travel_tol = max(1.0e-12, 1.0e-9 * max(abs(expected_travel), abs(travel)))
            speed_tol = max(1.0e-10, 1.0e-8 * max(abs(expected_speed), abs(speed)))
            if abs(travel - expected_travel) > travel_tol:
                travel_errors.append(
                    f"{label} step={step}: travel {travel:.9g}, expected {expected_travel:.9g} m")
            if abs(speed - expected_speed) > speed_tol:
                travel_errors.append(
                    f"{label} step={step}: speed {speed:.9g}, expected {expected_speed:.9g} m/s")
            expected_command[label][str(step)] = {
                "travel_m": expected_travel,
                "speed_m_s": expected_speed,
            }
            previous = expected_travel

    accepted_keys = {
        key for key, tool_row in tools.items()
        if tool_row.get("step_accepted") == 1
        and key in observations
        and observations[key].get("step_accepted") == 1
        and observations[key].get("status_code") == 0
        and observations[key].get("certificate_raw_accepted_flag") is not None
        and observations[key].get("certificate_raw_accepted_flag") >= 0.5
        and all(observations[key].get(field) is not None and
                observations[key][field] <= tolerance
                for field, tolerance in certificate_tolerances.items())
    }
    moving_rows = [tools[key] for key in sorted(tools) if key[1] == 0]
    control_rows = [tools[key] for key in sorted(tools) if key[1] == 1]
    accepted_moving_rows = [row for key, row in sorted(tools.items())
                            if key in accepted_keys and key[1] == 0]
    accepted_control_rows = [row for key, row in sorted(tools.items())
                             if key in accepted_keys and key[1] == 1]
    moving_contacts = sum(int(row["punch_contact_count"]) for row in accepted_moving_rows)
    control_contacts = sum(int(row["punch_contact_count"]) for row in control_rows)
    moving_forces = [float(row["punch_force_z_N"]) for row in accepted_moving_rows
                     if row.get("punch_force_z_N") is not None]
    gap_records = [
        {
            "step": key[0],
            "environment": key[1],
            "surface": "punch" if name == "min_end_punch_node_gap_m" else "anvil",
            "gap_m": float(row[name]),
            "step_accepted": key in accepted_keys,
        }
        for key, row in sorted(tools.items())
        for name in ("min_end_punch_node_gap_m", "min_anvil_node_gap_m")
        if row.get(name) is not None
    ]
    accepted_gap_records = [item for item in gap_records if item["step_accepted"]]
    rejected_gap_records = [item for item in gap_records if not item["step_accepted"]]
    all_gap_values = [item["gap_m"] for item in gap_records]
    accepted_gaps = [item["gap_m"] for item in accepted_gap_records]
    rejected_gaps = [item["gap_m"] for item in rejected_gap_records]
    accepted_gap_violations = [item for item in accepted_gap_records
                               if item["gap_m"] < -gap_roundoff_tolerance]
    violations = []
    if nonfinite:
        violations.append("nonfinite_measurements")
    if material_hash_mismatches:
        violations.extend("material_hash_mismatch" for _ in material_hash_mismatches)
    if arm_mismatches:
        violations.append("arm_environment_mapping")
    if travel_errors:
        violations.append("command_travel_continuity")
    if certificate_errors:
        violations.append("accepted_solver_certificate_exceeds_manifest_tolerance")
    if not incomplete and moving_contacts == 0:
        violations.append("moving_arm_has_no_native_punch_contacts")
    if control_contacts != 0:
        violations.append("stationary_control_has_punch_contacts")
    if accepted_gaps and min(accepted_gaps) < -slop:
        violations.append("node_penetration_exceeds_contact_slop")
    if accepted_gap_violations:
        violations.append("commanded_end_pose_node_penetration_exceeds_fp32_roundoff")
    if not accepted_gaps and not incomplete:
        violations.append("no_finite_node_gap_measurements")
    moving_peak_abs_force = max((abs(value) for value in moving_forces), default=None)
    if not incomplete and moving_contacts > 0 and (moving_peak_abs_force is None or moving_peak_abs_force <= 0.0):
        violations.append("moving_contact_has_no_resolved_z_force")

    if incomplete:
        verdict = "inconclusive"
        exit_code = 2
    elif violations:
        verdict = "failed"
        exit_code = 1
    else:
        verdict = "instrument_checks_passed"
        exit_code = 0

    def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        counts = [int(row["punch_contact_count"]) for row in rows]
        forces = [float(row["punch_force_z_N"]) for row in rows
                  if row.get("punch_force_z_N") is not None]
        punch = [float(row["min_end_punch_node_gap_m"]) for row in rows
                 if row.get("min_end_punch_node_gap_m") is not None]
        anvil = [float(row["min_anvil_node_gap_m"]) for row in rows
                 if row.get("min_anvil_node_gap_m") is not None]
        travel = [float(row["commanded_end_travel_m"]) for row in rows
                  if row.get("commanded_end_travel_m") is not None]
        speed = [float(row["commanded_speed_m_s"]) for row in rows
                 if row.get("commanded_speed_m_s") is not None]
        return {
            "steps_observed": len(rows),
            "punch_contact_count_total": sum(counts),
            "steps_with_punch_contacts": sum(count > 0 for count in counts),
            "peak_punch_contact_count_per_step": max(counts, default=0),
            "punch_force_z_N_min": min(forces) if forces else None,
            "punch_force_z_N_max": max(forces) if forces else None,
            "punch_force_z_N_peak_absolute": max((abs(value) for value in forces), default=None),
            "minimum_end_punch_node_gap_m": min(punch) if punch else None,
            "minimum_anvil_node_gap_m": min(anvil) if anvil else None,
            "maximum_commanded_travel_m": max(travel) if travel else None,
            "maximum_absolute_commanded_speed_m_s": max((abs(value) for value in speed), default=None),
        }

    def _gap_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
        values = [item["gap_m"] for item in records]
        negative = [item for item in records if item["gap_m"] < 0.0]
        violations_for_records = [item for item in records
                                  if item["gap_m"] < -gap_roundoff_tolerance]
        return {
            "samples": len(records),
            "minimum_node_gap_m": min(values) if values else None,
            "negative_gap_count": len(negative),
            "fp32_roundoff_violation_count": len(violations_for_records),
            "first_fp32_roundoff_violation": violations_for_records[0] if violations_for_records else None,
        }

    def _force_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        forces = [float(row["punch_force_z_N"]) for row in rows
                  if row.get("punch_force_z_N") is not None]
        return {
            "steps_observed": len(rows),
            "punch_contact_count_total": sum(int(row["punch_contact_count"]) for row in rows),
            "punch_force_z_N_min": min(forces) if forces else None,
            "punch_force_z_N_max": max(forces) if forces else None,
            "punch_force_z_N_peak_absolute": max((abs(value) for value in forces), default=None),
        }

    hashes = _artifact_hashes(run_dir)
    return {
        "schema": SCHEMA,
        "verdict": verdict,
        "exit_code": exit_code,
        "fold_qualification": False,
        "physical_validation": False,
        "evidence_boundary": "software mechanics instrumentation only; no cardboard fold or physical qualification",
        "run_directory": str(run_dir),
        "result_status": status,
        "accepted_steps": accepted_steps,
        "expected_steps": expected_steps,
        "contact_slop_m": slop,
        "timestep_s": dt,
        "tooling": {
            "indentation_m": indentation,
            "initial_nose_clearance_m": nose_clearance,
            "initial_anvil_clearance_m": anvil_clearance,
            "timing_contract": timing,
        },
        "checks": {
        "moving_arm_native_punch_contact": "pass" if moving_contacts > 0 else ("inconclusive" if incomplete else "fail"),
            "stationary_control_punch_contact": "pass" if control_contacts == 0 else "fail",
            "finite_measurements": "fail" if nonfinite else "pass",
            "node_penetration_within_contact_slop": "pass" if accepted_gaps and min(accepted_gaps) >= -slop else ("inconclusive" if incomplete else "fail"),
            "commanded_end_pose_node_nonpenetration": "fail" if accepted_gap_violations else ("pass" if accepted_gaps else ("inconclusive" if incomplete else "fail")),
            "command_travel_matches_manifest_and_velocity": "fail" if travel_errors else ("inconclusive" if incomplete else "pass"),
            "moving_contact_has_resolved_z_force": "pass" if moving_contacts > 0 and moving_peak_abs_force is not None and moving_peak_abs_force > 0.0 else ("inconclusive" if incomplete else "fail"),
            "accepted_solver_certificate_within_manifest_tolerance": "fail" if certificate_errors else ("inconclusive" if incomplete else "pass"),
            "material_source_hashes": "fail" if material_hash_mismatches else "pass",
            "arm_environment_mapping": "fail" if arm_mismatches else "pass",
        },
        "summary": {
            "verified_accepted_step_node_gaps": _gap_summary(accepted_gap_records),
            "rejected_or_unverified_step_node_gap_diagnostics": _gap_summary(rejected_gap_records),
            "verified_accepted_indented_punch_force_receipt": _force_summary(accepted_moving_rows),
            "rejected_or_unverified_indented_punch_force_diagnostics": _force_summary(
                [row for key, row in sorted(tools.items()) if key not in accepted_keys and key[1] == 0]),
            "stationary_control_all_step_force_diagnostics": _force_summary(control_rows),
            "indented_environment_all_observed_rows": _arm_summary(moving_rows),
            "stationary_tool_reference_environment_all_observed_rows": _arm_summary(control_rows),
            "minimum_node_gap_any_tool_all_observed_rows_m": min(all_gap_values) if all_gap_values else None,
            "contact_barrier_range_m": slop,
            "minimum_gap_roundoff_tolerance_m": gap_roundoff_tolerance,
            "roundoff_coordinate_scale_m": mesh_coordinate_scale,
            "contact_slop_is_not_an_allowed_physical_penetration": True,
            "certificate_tolerances": certificate_tolerances,
            "certificate_acceptance_errors": certificate_errors,
            "gap_measurement": "CPU signed-distance evaluation at FEM node samples and commanded end pose; not a continuous surface-distance measurement",
            "force_measurement": "z impulse from native punch contact samples divided by timestep; signed force is on the punch and is an instrument receipt, not force or fold qualification",
        },
        "coverage": {
            "observations_rows": observation_count,
            "tool_observations_rows": tool_count,
            "expected_rows_per_table": expected_steps * 2,
            "incomplete_reasons": incomplete,
            "nonfinite_fields": nonfinite,
            "travel_errors": travel_errors,
            "arm_mapping_errors": arm_mismatches,
            "material_hash_mismatches": material_hash_mismatches,
            "certificate_acceptance_errors": certificate_errors,
        },
        "command_curve": expected_command,
        "sha256_bindings": {
            "compiled_world_fingerprint": str(manifest_fingerprint),
            "materials": material_bindings,
            "artifacts": hashes,
        },
    }


def _error_report(error: Exception) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "verdict": "invalid_evidence",
        "exit_code": 1,
        "fold_qualification": False,
        "physical_validation": False,
        "evidence_boundary": "software mechanics instrumentation only; no cardboard fold or physical qualification",
        "error": str(error),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path,
                        help="create a JSON report at this new path; existing files are never replaced")
    args = parser.parse_args(argv)
    try:
        report = analyze_run(args.run_directory)
    except (EvidenceError, OSError) as error:
        report = _error_report(error)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as destination:
                destination.write(serialized)
        except FileExistsError:
            report = _error_report(EvidenceError(f"refusing to overwrite {args.output}"))
            serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
            print(serialized, end="")
            return 1
        except OSError as error:
            report = _error_report(error)
            serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
            print(serialized, end="")
            return 1
    print(serialized, end="")
    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
