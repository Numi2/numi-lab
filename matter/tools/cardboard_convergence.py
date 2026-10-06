#!/usr/bin/env python3
"""Fail-closed descriptive convergence analysis for native cardboard runs.

This tool compares retained run exports. It does not run Matter or qualify a
physical board. A completed result is admitted only after every saved row has
an accepted step certificate and the exported protocol, states, and mesh agree.
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
from typing import Any, Iterable


ANALYSIS_SCHEMA = "numi.cardboard.convergence-analysis.v1"
MANIFEST_SCHEMA = "numi.cardboard.explicit-strip.v1"
RESULT_SCHEMA = "numi.cardboard.probe-result.v1"
MESH_SCHEMA = "numi.cardboard.authored-mesh.v1"
STATE_SCHEMA = "numi.cardboard.accepted-material-state.v1"
EP_NAMES = ("ep11", "ep22", "ep33", "ep23", "ep13", "ep12")
PHASE_ORDER = ("loading", "hold", "unloading", "relaxation", "release")
PHASE_COUNT_KEYS = {
    "loading": "steps",
    "hold": "hold_steps",
    "unloading": "unload_steps",
    "relaxation": "relax_steps",
    "release": "release_steps",
}
FLOAT_FIELDS = (
    "time_s", "target_angle_deg", "reaction_x_N", "reaction_y_N", "reaction_z_N",
    "reaction_moment_y_Nm", "max_free_displacement_m", "max_free_speed_m_s",
    "kinetic_energy_J", "certificate_residual", "certificate_volume", "certificate_pressure",
)
CURVE_FIELDS = (
    "reaction_x_N", "reaction_y_N", "reaction_z_N", "reaction_moment_y_Nm",
    "kinetic_energy_J", "max_free_speed_m_s", "max_free_displacement_m",
)
FORCE_IMBALANCE_FIELDS = ("free_force_imbalance_l2_N", "free_force_imbalance_max_N")
# Probe manifests are emitted with finite decimal precision. Compare the
# normalized tolerance-per-timestep ratio with a 1 ppm relative allowance.
FORCE_MATCH_RATIO_REL_TOL = 1.0e-6
COMMON_SOLVER_FIELDS = (
    "backend", "deformable_self_contact", "contact_slop_m", "local_material_newton_iterations",
    "newton_iteration_budget", "fgmres_restart", "fgmres_iteration_budget", "line_search_steps",
    "relative_residual_tolerance", "volume_tolerance", "pressure_tolerance", "transport_tolerance",
    "maximum_rate_exponent",
)
GEOMETRY_FIELDS = (
    "source_dimensions", "initial_condition", "equation", "length_m", "width_m", "total_height_m",
    "pitch_m", "liner_thickness_m", "medium_normal_thickness_m", "medium_centerline_height_m",
    "medium_present", "glue_minimum_gap_m", "bond_width_m", "upper_glue_gap_m", "upper_bond_width_m",
    "contact_tie_assumption",
)


class EvidenceError(Exception):
    def __init__(self, code: str, message: str, state: str = "inconclusive_evidence") -> None:
        super().__init__(message)
        self.code = code
        self.state = state


class IncompatibleError(EvidenceError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, "incompatible_pair")


class FailedRunError(EvidenceError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, "failed_run")


@dataclass
class RunData:
    path: Path
    manifest: dict[str, Any]
    result: dict[str, Any]
    mesh: dict[str, Any]
    rows: list[dict[str, Any]]
    files: dict[str, str]
    volumes_m3: list[float]
    materials_by_index: list[dict[str, Any]]
    row_by_key: dict[tuple[int, int], dict[str, Any]]
    phases: list[str]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _analyzer_digest() -> str:
    try:
        return _sha256(Path(__file__).read_bytes())
    except OSError as exc:
        raise EvidenceError("unreadable_analyzer", f"cannot hash analyzer source: {exc}") from exc


def _read_json(path: Path, files: dict[str, str]) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError("unreadable_json", f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError("invalid_json_root", f"{path.name} must contain a JSON object")
    files[path.name] = _sha256(raw)
    return value


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError("invalid_number", f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise EvidenceError("nonfinite_number", f"{label} is not finite")
    return result


def _int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceError("invalid_integer", f"{label} is not an integer") from exc
    if isinstance(value, float) and value != result:
        raise EvidenceError("invalid_integer", f"{label} is not an exact integer")
    return result


def _close(a: float, b: float, *, rel: float = 1e-8, abs_tol: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)


def _geometry_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    geometry = manifest.get("geometry")
    if not isinstance(geometry, dict):
        raise EvidenceError("missing_geometry", "manifest.geometry is missing")
    signature: dict[str, Any] = {}
    for key in GEOMETRY_FIELDS:
        if key not in geometry:
            raise EvidenceError("missing_geometry_field", f"manifest.geometry.{key} is missing")
        value = geometry[key]
        signature[key] = _number(value, f"geometry.{key}") if key.endswith("_m") else value
    return signature


def _protocol(solver: dict[str, Any]) -> dict[str, Any]:
    protocol: dict[str, Any] = {}
    for phase, field in PHASE_COUNT_KEYS.items():
        # Old v1 manifests predate free release; they are accepted for clamped
        # studies as release_steps=0, but never imply release evidence.
        value = solver.get(field, 0 if phase == "release" else None)
        if value is None:
            raise EvidenceError("missing_protocol_field", f"manifest.solver.{field} is missing")
        count = _int(value, f"solver.{field}")
        if count < 0:
            raise EvidenceError("invalid_protocol", f"solver.{field} cannot be negative")
        protocol[phase] = count
    dt = _number(solver.get("dt_s"), "solver.dt_s")
    angle = _number(solver.get("bend_angle_deg"), "solver.bend_angle_deg")
    if dt <= 0 or angle < 0:
        raise EvidenceError("invalid_protocol", "timestep must be positive and bend angle nonnegative")
    protocol["dt_s"] = dt
    protocol["bend_angle_deg"] = angle
    protocol["release_policy"] = solver.get("release_policy")
    if protocol["release"] > 0 and protocol["release_policy"] != (
        "right grip free; left grip remains fixed; physical state preserved"
    ):
        raise EvidenceError("unknown_release_policy", "release phase has an unknown or missing policy")
    return protocol


def _expected_phase(protocol: dict[str, Any], step: int) -> tuple[str, float | None]:
    cursor = 0
    angle = protocol["bend_angle_deg"]
    load_count = protocol["loading"]
    for phase in PHASE_ORDER:
        count = protocol[phase]
        if cursor <= step < cursor + count:
            within = step - cursor
            if phase == "loading":
                target = angle * (within + 1) / count
            elif phase == "hold":
                target = angle
            elif phase == "unloading":
                target = angle * (1.0 - (within + 1) / count)
            elif phase == "relaxation":
                target = 0.0
            else:
                target = None  # target angle is explicitly ignored while released
            return phase, target
        cursor += count
    raise EvidenceError("step_outside_protocol", f"CSV step {step} is outside the declared protocol")


def _tetra_volume(points: list[list[float]], tet: list[int], index: int) -> float:
    if len(tet) != 4:
        raise EvidenceError("invalid_mesh", f"tetrahedron {index} must have four node indices")
    if len(set(tet)) != 4 or any(i < 0 or i >= len(points) for i in tet):
        raise EvidenceError("invalid_mesh", f"tetrahedron {index} has invalid node indices")
    p0, p1, p2, p3 = ([ _number(c, f"node {i} coordinate") for c in points[i] ] for i in tet)
    a = [p1[i] - p0[i] for i in range(3)]
    b = [p2[i] - p0[i] for i in range(3)]
    c = [p3[i] - p0[i] for i in range(3)]
    cross = [b[1]*c[2]-b[2]*c[1], b[2]*c[0]-b[0]*c[2], b[0]*c[1]-b[1]*c[0]]
    volume = abs(sum(a[i] * cross[i] for i in range(3))) / 6.0
    if not math.isfinite(volume) or volume <= 0:
        raise EvidenceError("invalid_mesh", f"tetrahedron {index} has nonpositive reference volume")
    return volume


def _load_run(path: Path) -> RunData:
    path = path.resolve()
    if not path.is_dir():
        raise EvidenceError("missing_run_directory", f"run directory does not exist: {path}")
    files: dict[str, str] = {}
    manifest = _read_json(path / "manifest.json", files)
    result = _read_json(path / "result.json", files)
    mesh = _read_json(path / "mesh.json", files)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise EvidenceError("unsupported_manifest_schema", f"unsupported manifest schema {manifest.get('schema')!r}")
    if manifest.get("specimen", "corrugated_strip") != "corrugated_strip":
        raise EvidenceError("unsupported_specimen", "strip convergence cannot admit a box blank")
    if result.get("schema") != RESULT_SCHEMA:
        raise EvidenceError("unsupported_result_schema", f"unsupported result schema {result.get('schema')!r}")
    if mesh.get("schema") != MESH_SCHEMA or mesh.get("units") != "metres":
        raise EvidenceError("unsupported_mesh_schema", "mesh schema or units are not recognized")
    tooling = manifest.get("tooling")
    if tooling is not None:
        if not isinstance(tooling, dict):
            raise EvidenceError("unknown_tooling_configuration", "manifest.tooling must be an object when present")
        if tooling.get("enabled") is True:
            raise EvidenceError("unsupported_loading_mode", "tool-driven crease/indentation loading is not analyzed by this tool")
        if "enabled" in tooling and tooling["enabled"] is not False:
            raise EvidenceError("unknown_tooling_configuration", "manifest.tooling.enabled must be true or false")
    if result.get("status") != "completed":
        if result.get("status") == "failed":
            raise FailedRunError("native_run_failed", str(result.get("failure") or "native run failed"))
        raise EvidenceError("run_not_complete", f"run status is {result.get('status')!r}")

    solver = manifest.get("solver")
    if not isinstance(solver, dict) or solver.get("runtime_execution") is not True:
        raise EvidenceError("not_runtime_evidence", "manifest does not identify a native runtime execution")
    protocol = _protocol(solver)
    expected_steps = sum(protocol[p] for p in PHASE_ORDER)
    if expected_steps <= 0 or _int(result.get("accepted_steps"), "result.accepted_steps") != expected_steps:
        raise FailedRunError("accepted_step_count_mismatch", "completed status and accepted step count disagree with protocol")
    if result.get("compiled_world_fingerprint") != manifest.get("compiled_world_fingerprint"):
        raise EvidenceError("fingerprint_mismatch", "result and manifest compiled-world fingerprints differ")
    mat_manifest = manifest.get("materials", {})
    for result_key, manifest_key in (("material_map_sha256", "regional_map_sha256"),
                                     ("frame_map_sha256", "frame_map_sha256")):
        if not result.get(result_key) or result.get(result_key) != mat_manifest.get(manifest_key):
            raise EvidenceError("map_digest_mismatch", f"result and manifest {result_key} differ or are missing")

    nodes = mesh.get("nodes_m")
    tetrahedra = mesh.get("tetrahedra")
    material_indices = mesh.get("material_indices")
    if not all(isinstance(v, list) for v in (nodes, tetrahedra, material_indices)):
        raise EvidenceError("invalid_mesh", "mesh nodes, tetrahedra, and material indices must be arrays")
    if len(tetrahedra) != len(material_indices):
        raise EvidenceError("invalid_mesh", "tetrahedron and material-index counts differ")
    mesh_info = manifest.get("mesh", {})
    expected_node_count = len(nodes)
    expected_tet_count = len(tetrahedra)
    if (mesh_info.get("nodes") != expected_node_count or mesh_info.get("tetrahedra") != expected_tet_count
            or result.get("nodes") != expected_node_count or result.get("tetrahedra") != expected_tet_count):
        raise EvidenceError("mesh_count_mismatch", "mesh array counts differ from manifest/result counts")
    cells = manifest.get("materials", {}).get("cells")
    if not isinstance(cells, list) or not cells:
        raise EvidenceError("unknown_material_map", "manifest material cells are missing")
    cells_by_index = {int(c["index"]): c for c in cells if isinstance(c, dict) and "index" in c}
    if len(cells_by_index) != len(cells) or sorted(cells_by_index) != list(range(len(cells))):
        raise EvidenceError("unknown_material_map", "material indices are not contiguous")
    if any(not isinstance(mi, int) or mi not in cells_by_index for mi in material_indices):
        raise EvidenceError("unknown_material_map", "mesh contains an unknown material index")
    if mesh.get("medium_present") != manifest.get("geometry", {}).get("medium_present"):
        raise EvidenceError("mesh_geometry_mismatch", "mesh and manifest disagree about medium presence")
    frames = mesh.get("material_frames_xyzw")
    if not isinstance(frames, list) or len(frames) != expected_tet_count:
        raise EvidenceError("invalid_mesh", "material frame count differs from tetrahedron count")
    for i, frame in enumerate(frames):
        if not isinstance(frame, list) or len(frame) != 4:
            raise EvidenceError("invalid_mesh", f"material frame {i} is not an xyzw quaternion")
        for component in frame:
            _number(component, f"material frame {i}")
    density_values = mesh.get("material_densities_kg_m3")
    if not isinstance(density_values, list) or len(density_values) != len(cells_by_index):
        raise EvidenceError("unknown_material_density_map", "mesh material density vector is missing or mismatched")
    declared_counts = [0] * len(cells_by_index)
    for mi in material_indices:
        declared_counts[mi] += 1
    for i, cell in enumerate([cells_by_index[x] for x in range(len(cells_by_index))]):
        if _int(cell.get("tetrahedra"), f"manifest material {i} tetrahedra") != declared_counts[i]:
            raise EvidenceError("material_count_mismatch", f"material {i} tetrahedron count differs from exported mesh")
        _number(density_values[i], f"mesh material density {i}")
    volumes = [_tetra_volume(nodes, tet, i) for i, tet in enumerate(tetrahedra)]
    for key, count in (("nodes", expected_node_count), ("tetrahedra", expected_tet_count)):
        if _int(result.get(key), f"result.{key}") != count:
            raise EvidenceError("mesh_count_mismatch", f"result.{key} differs from exported mesh")

    try:
        raw_csv = (path / "observations.csv").read_bytes()
        files["observations.csv"] = _sha256(raw_csv)
        rows = list(csv.DictReader(raw_csv.decode("utf-8").splitlines()))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise EvidenceError("unreadable_observations", f"cannot read observations.csv: {exc}") from exc
    if not rows:
        raise FailedRunError("missing_observations", "completed run has no per-step observations")
    header = set(rows[0])
    required = {"step", "time_s", "environment", "arm", "target_angle_deg", "status_code", "phase",
                "reaction_x_N", "reaction_y_N", "reaction_z_N", "reaction_moment_y_Nm",
                "max_free_displacement_m", "max_free_speed_m_s", "kinetic_energy_J",
                "certificate_residual", "certificate_volume", "certificate_pressure"}
    missing = required - header
    if missing:
        raise EvidenceError("unsupported_observation_schema", f"CSV lacks required fields: {sorted(missing)}")
    if "step_accepted" not in header and "certificate_accepted" not in header:
        raise EvidenceError("missing_step_acceptance", "CSV has only raw status data; authoritative step acceptance is absent")
    acceptance_field = "step_accepted" if "step_accepted" in header else "certificate_accepted"
    release_columns = {"right_grip_constrained", "measured_right_grip_angle_deg", "current_fixed_nodes"}
    if protocol["release"] > 0 and not release_columns.issubset(header):
        raise EvidenceError("missing_release_observations", "release run lacks explicit boundary-state columns")

    expected_row_count = expected_steps * 2
    if len(rows) != expected_row_count:
        raise FailedRunError("observation_count_mismatch", f"expected {expected_row_count} rows, found {len(rows)}")
    row_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    times_by_step: dict[int, float] = {}
    tolerance = _number(solver.get("relative_residual_tolerance"), "solver.relative_residual_tolerance")
    volume_tolerance = _number(solver.get("volume_tolerance"), "solver.volume_tolerance")
    pressure_tolerance = _number(solver.get("pressure_tolerance"), "solver.pressure_tolerance")
    dt = protocol["dt_s"]
    if tolerance <= 0:
        raise EvidenceError("invalid_solver_tolerance", "solver.relative_residual_tolerance must be positive")
    for raw in rows:
        step = _int(raw.get("step"), "CSV.step")
        env = _int(raw.get("environment"), "CSV.environment")
        if not (0 <= step < expected_steps and env in (0, 1)) or (step, env) in row_by_key:
            raise FailedRunError("observation_key_mismatch", f"invalid or duplicate (step, environment)=({step},{env})")
        if raw.get("arm") != ("bent" if env == 0 else "held_reference"):
            raise EvidenceError("unknown_arm_mapping", f"environment {env} has unexpected arm label")
        values = {key: _number(raw.get(key), f"CSV.{key}") for key in FLOAT_FIELDS}
        for key in FORCE_IMBALANCE_FIELDS:
            if key in header:
                force_value = _number(raw.get(key), f"CSV.{key}")
                if force_value < 0:
                    raise EvidenceError("invalid_force_imbalance", f"CSV.{key} cannot be negative")
                values[key] = force_value
        status_code = _int(raw.get("status_code"), "CSV.status_code")
        accepted = _int(raw.get(acceptance_field), f"CSV.{acceptance_field}")
        if status_code != 0 or accepted != 1:
            raise FailedRunError("step_rejected", f"step {step} environment {env} has status={status_code}, accepted={accepted}")
        if (values["certificate_residual"] > tolerance or values["certificate_volume"] > volume_tolerance
                or values["certificate_pressure"] > pressure_tolerance):
            raise FailedRunError("certificate_outside_tolerance", f"step {step} environment {env} exceeds a declared certificate tolerance")
        expected_time = (step + 1) * dt
        if not _close(values["time_s"], expected_time, rel=2e-7, abs_tol=2e-10):
            raise EvidenceError("time_grid_mismatch", f"step {step} time does not match declared fixed timestep")
        phase, angle = _expected_phase(protocol, step)
        if raw.get("phase") != phase:
            raise EvidenceError("phase_schedule_mismatch", f"step {step} phase differs from manifest protocol")
        expected_target = angle if env == 0 else 0.0
        if angle is not None and not _close(values["target_angle_deg"], expected_target, rel=2e-7, abs_tol=2e-9):
            raise EvidenceError("target_schedule_mismatch", f"step {step} target angle differs from protocol")
        if acceptance_field == "certificate_accepted" and status_code != 0:
            raise FailedRunError("legacy_raw_acceptance_rejected", "raw accepted flag cannot override failed status")
        if release_columns.issubset(header):
            constrained = _int(raw.get("right_grip_constrained"), "CSV.right_grip_constrained")
            expected_constrained = 0 if phase == "release" else 1
            if constrained != expected_constrained:
                raise EvidenceError("release_boundary_mismatch", f"step {step} grip-constraint flag differs from phase")
            measured_angle = _number(raw.get("measured_right_grip_angle_deg"), "CSV.measured_right_grip_angle_deg")
            fixed_nodes = _int(raw.get("current_fixed_nodes"), "CSV.current_fixed_nodes")
            mesh_left = len(mesh.get("left_grip_nodes", []))
            mesh_fixed = len(mesh.get("fixed_nodes", []))
            expected_fixed = mesh_left if phase == "release" else mesh_fixed
            if expected_fixed and fixed_nodes != expected_fixed:
                raise EvidenceError("fixed_node_count_mismatch", f"step {step} fixed-node count differs from release policy")
            if phase == "release" and env == 1 and abs(measured_angle) > 1e-5:
                raise EvidenceError("reference_grip_moved", "held reference right grip changed angle during release")
            values["right_grip_constrained"] = constrained
            values["measured_right_grip_angle_deg"] = measured_angle
            values["current_fixed_nodes"] = fixed_nodes
        values.update({"step": step, "environment": env, "arm": raw["arm"], "phase": phase,
                       "status_code": status_code, "accepted": accepted})
        row_by_key[(step, env)] = values
        if env == 0:
            times_by_step[step] = values["time_s"]
    if len(row_by_key) != expected_row_count:
        raise FailedRunError("observation_key_mismatch", "per-step observations are incomplete")
    for step in range(expected_steps):
        if step not in times_by_step:
            raise FailedRunError("observation_key_mismatch", f"missing step {step}")
    if release_columns.issubset(header):
        fixed_nodes = mesh.get("fixed_nodes")
        left_nodes = mesh.get("left_grip_nodes")
        right_nodes = mesh.get("right_grip_nodes")
        if not all(isinstance(v, list) for v in (fixed_nodes, left_nodes, right_nodes)):
            raise EvidenceError("invalid_grip_map", "mesh fixed and grip node maps are missing")
        for label, indices in (("fixed", fixed_nodes), ("left grip", left_nodes), ("right grip", right_nodes)):
            if len(set(indices)) != len(indices) or any(not isinstance(i, int) or i < 0 or i >= expected_node_count for i in indices):
                raise EvidenceError("invalid_grip_map", f"mesh {label} node map is malformed")
        if set(fixed_nodes) != set(left_nodes) | set(right_nodes) or set(left_nodes) & set(right_nodes):
            raise EvidenceError("invalid_grip_map", "fixed nodes do not equal disjoint left/right grip node sets")

    # Varying mesh resolution may change discretized glue volume/area. These
    # values are retained as measurements rather than pair-admission keys.
    normalized_rows = [row_by_key[(step, env)] for step in range(expected_steps) for env in (0, 1)]
    return RunData(path, manifest, result, mesh, normalized_rows, files, volumes, [cells_by_index[i] for i in range(len(cells_by_index))], row_by_key,
                   [phase for phase in PHASE_ORDER if protocol[phase] > 0])


def _phase_ends(run: RunData) -> dict[str, int]:
    protocol = _protocol(run.manifest["solver"])
    end: dict[str, int] = {}
    cursor = 0
    for phase in PHASE_ORDER:
        cursor += protocol[phase]
        if protocol[phase] > 0:
            end[phase] = cursor
    return end


def _state_stats(run: RunData, phase: str, arm: str) -> dict[str, Any]:
    end = _phase_ends(run).get(phase)
    if end is None:
        raise EvidenceError("missing_requested_phase", f"run has no {phase} phase")
    path = run.path / f"material_state_{end:06d}_{arm}.json"
    state = _read_json(path, run.files)
    if state.get("schema") != STATE_SCHEMA:
        raise EvidenceError("unsupported_state_schema", f"unsupported state schema in {path.name}")
    env = 0 if arm == "bent" else 1
    if _int(state.get("environment"), "state.environment") != env or _int(state.get("status_code"), "state.status_code") != 0:
        raise EvidenceError("invalid_state_snapshot", f"{path.name} is not an accepted state for {arm}")
    mesh_tets = run.mesh["tetrahedra"]
    mesh_materials = run.mesh["material_indices"]
    state_tets = state.get("tetrahedra")
    if not isinstance(state_tets, list) or len(state_tets) != len(mesh_tets):
        raise EvidenceError("state_mesh_mismatch", f"{path.name} tetrahedron count differs from mesh")
    state_materials = state.get("materials")
    if not isinstance(state_materials, list):
        raise EvidenceError("invalid_state_snapshot", f"{path.name} has no material state table")
    mats = {int(item["index"]): item for item in state_materials if isinstance(item, dict) and "index" in item}
    expected_names = {i: cell["name"] for i, cell in enumerate(run.materials_by_index)}
    if set(mats) != set(expected_names):
        raise EvidenceError("state_material_mismatch", f"{path.name} material indices differ from manifest")
    for i, expected in expected_names.items():
        if mats[i].get("name") != expected:
            raise EvidenceError("state_material_mismatch", f"{path.name} material name differs at index {i}")
    accum: dict[int, dict[str, Any]] = {}
    for mi, material in mats.items():
        names = material.get("state_names")
        if not isinstance(names, list):
            raise EvidenceError("invalid_state_snapshot", f"{path.name} state names are missing")
        ep_indices = {name: names.index(name) for name in EP_NAMES if name in names}
        is_paper = material["name"].startswith(("hajali2009_liner", "hajali2009_medium"))
        is_known_elastic_control = material["name"].endswith("_elastic_control")
        if is_paper and not is_known_elastic_control and set(ep_indices) != set(EP_NAMES):
            raise EvidenceError("missing_plastic_state", f"{path.name} paper state lacks one or more six Ep components")
        accum[mi] = {"volume_m3": 0.0, "ep_integral": [0.0] * 6, "abs_ep_integral": [0.0] * 6,
                     "ep_norm_integral": 0.0, "ep_norm2_integral": 0.0, "max_ep_norm": 0.0,
                     "plasticized_volume_m3": 0.0, "tetrahedra": 0}
    threshold = 1.0e-6  # explicit diagnostic threshold, not a paper yield criterion
    for i, item in enumerate(state_tets):
        if _int(item.get("index"), "state tetrahedron index") != i:
            raise EvidenceError("state_mesh_mismatch", f"{path.name} tetrahedron ordering changed")
        mi = _int(item.get("material_index"), "state material index")
        if mi != mesh_materials[i] or mi not in mats:
            raise EvidenceError("state_mesh_mismatch", f"{path.name} material map differs at tetrahedron {i}")
        values = item.get("state")
        names = mats[mi]["state_names"]
        if not isinstance(values, list) or len(values) != len(names):
            raise EvidenceError("state_mesh_mismatch", f"{path.name} state vector length differs at tetrahedron {i}")
        cell = accum[mi]
        volume = run.volumes_m3[i]
        cell["volume_m3"] += volume
        cell["tetrahedra"] += 1
        indices = {name: names.index(name) for name in EP_NAMES if name in names}
        if len(indices) == 6:
            ep = [_number(values[indices[name]], f"{path.name} {name}") for name in EP_NAMES]
            norm2 = sum(ep[j] * ep[j] for j in range(3)) + 2.0 * sum(ep[j] * ep[j] for j in range(3, 6))
            norm = math.sqrt(norm2)
            for j, value in enumerate(ep):
                cell["ep_integral"][j] += volume * value
                cell["abs_ep_integral"][j] += volume * abs(value)
            cell["ep_norm_integral"] += volume * norm
            cell["ep_norm2_integral"] += volume * norm2
            cell["max_ep_norm"] = max(cell["max_ep_norm"], norm)
            if norm > threshold:
                cell["plasticized_volume_m3"] += volume
    per_material: dict[str, Any] = {}
    paper_total = {"volume_m3": 0.0, "ep_integral": [0.0] * 6, "abs_ep_integral": [0.0] * 6,
                   "ep_norm_integral": 0.0, "ep_norm2_integral": 0.0, "plasticized_volume_m3": 0.0,
                   "max_ep_norm": 0.0}
    for mi, cell in accum.items():
        name = mats[mi]["name"]
        if not name.startswith(("hajali2009_liner", "hajali2009_medium")):
            continue
        volume = cell["volume_m3"]
        if volume <= 0:
            raise EvidenceError("missing_paper_volume", f"paper material {name} has no tetrahedron volume")
        per_material[name] = {
            "reference_volume_m3": volume,
            "tetrahedra": cell["tetrahedra"],
            "volume_mean_ep": [x / volume for x in cell["ep_integral"]],
            "volume_mean_abs_ep": [x / volume for x in cell["abs_ep_integral"]],
            "volume_mean_frobenius_ep": cell["ep_norm_integral"] / volume,
            "volume_rms_frobenius_ep": math.sqrt(cell["ep_norm2_integral"] / volume),
            "max_frobenius_ep": cell["max_ep_norm"],
            "plasticized_volume_fraction_ep_norm_gt_1e-6": cell["plasticized_volume_m3"] / volume,
            "volume_integral_abs_ep_m3": cell["ep_norm_integral"],
            "volume_integral_squared_ep_m3": cell["ep_norm2_integral"],
        }
        for key in ("volume_m3", "ep_norm_integral", "ep_norm2_integral", "plasticized_volume_m3"):
            if key == "volume_m3": paper_total[key] += cell[key]
            elif key == "ep_norm_integral": paper_total[key] += cell[key]
            elif key == "ep_norm2_integral": paper_total[key] += cell[key]
            elif key == "plasticized_volume_m3": paper_total[key] += cell[key]
        for j in range(6):
            paper_total["ep_integral"][j] += cell["ep_integral"][j]
            paper_total["abs_ep_integral"][j] += cell["abs_ep_integral"][j]
        paper_total["max_ep_norm"] = max(paper_total["max_ep_norm"], cell["max_ep_norm"])
    if paper_total["volume_m3"] <= 0:
        raise EvidenceError("missing_paper_volume", "snapshot has no liner or medium tetrahedra")
    total_volume = paper_total["volume_m3"]
    return {
        "snapshot": path.name,
        "plastic_state_components": list(EP_NAMES),
        "plastic_norm": "symmetric tensor Frobenius norm; engineering shear components weighted by two",
        "plasticized_threshold": {"ep_frobenius_norm_gt": threshold, "meaning": "numerical diagnostic, not physical yield"},
        "paper_total": {
            "reference_volume_m3": total_volume,
            "volume_mean_ep": [x / total_volume for x in paper_total["ep_integral"]],
            "volume_mean_abs_ep": [x / total_volume for x in paper_total["abs_ep_integral"]],
            "volume_mean_frobenius_ep": paper_total["ep_norm_integral"] / total_volume,
            "volume_rms_frobenius_ep": math.sqrt(paper_total["ep_norm2_integral"] / total_volume),
            "max_frobenius_ep": paper_total["max_ep_norm"],
            "plasticized_volume_fraction_ep_norm_gt_1e-6": paper_total["plasticized_volume_m3"] / total_volume,
            "volume_integral_abs_ep_m3": paper_total["ep_norm_integral"],
            "volume_integral_squared_ep_m3": paper_total["ep_norm2_integral"],
        },
        "by_material": per_material,
    }


def _numeric_equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return _close(float(a), float(b))
    return a == b


def _same_geometry(a: RunData, b: RunData) -> None:
    ga, gb = _geometry_signature(a.manifest), _geometry_signature(b.manifest)
    for key in GEOMETRY_FIELDS:
        if not _numeric_equal(ga[key], gb[key]):
            raise IncompatibleError("geometry_mismatch", f"geometry.{key} differs")
    fa = a.manifest.get("glue_footprints")
    fb = b.manifest.get("glue_footprints")
    if not isinstance(fa, list) or not isinstance(fb, list) or len(fa) != len(fb):
        raise IncompatibleError("glue_footprint_mismatch", "nominal glue footprint table differs")
    normalize = lambda footprints: sorted(
        (bool(item["upper"]), float(item["center_x_m"]), float(item["requested_width_m"]))
        for item in footprints
    )
    if normalize(fa) != normalize(fb):
        raise IncompatibleError("glue_footprint_mismatch", "nominal glue centers or requested widths differ")


def _same_materials(a: RunData, b: RunData) -> None:
    ma, mb = a.manifest.get("materials", {}), b.manifest.get("materials", {})
    keys = ("liner_sha256", "medium_sha256", "glue_sha256")
    for key in keys:
        if ma.get(key) != mb.get(key):
            raise IncompatibleError("material_source_mismatch", f"material source hash {key} differs")
    names_a = [cell["name"] for cell in a.materials_by_index]
    names_b = [cell["name"] for cell in b.materials_by_index]
    if names_a != names_b:
        raise IncompatibleError("material_map_mismatch", "material region names differ")


def _same_solver_controls(a: RunData, b: RunData, *,
                          allow_force_matched_tolerance: bool = False) -> None:
    sa, sb = a.manifest["solver"], b.manifest["solver"]
    # Older manifests predate the immutable-reference arithmetic option.
    # They unambiguously used absolute-position Ds*Dm^-1. Do not admit a
    # precision-method change as an otherwise matched mesh/time comparison.
    gradients = [s.get("deformation_gradient", "absolute_position") for s in (sa, sb)]
    if any(value not in {"absolute_position", "reference_displacement", "persistent_reference_displacement"} for value in gradients):
        raise EvidenceError("unknown_deformation_gradient", "unrecognized FEM arithmetic mode")
    if gradients[0] != gradients[1]:
        raise IncompatibleError("solver_control_mismatch", "deformation gradient arithmetic differs")
    preconditioners = [s.get("fgmres_preconditioner", "scalar_diagonal") for s in (sa, sb)]
    if any(value not in {"scalar_diagonal", "regional_tangent_fem_diagonal"} for value in preconditioners):
        raise EvidenceError("unknown_preconditioner", "unrecognized FGMRES preconditioner")
    if preconditioners[0] != preconditioners[1]:
        raise IncompatibleError("solver_control_mismatch", "FGMRES preconditioner differs")
    for key in COMMON_SOLVER_FIELDS:
        if key not in sa or key not in sb:
            raise EvidenceError("missing_solver_control", f"solver control {key} is missing")
        if key == "relative_residual_tolerance" and allow_force_matched_tolerance:
            continue
        if not _numeric_equal(sa[key], sb[key]):
            raise IncompatibleError("solver_control_mismatch", f"solver control {key} differs")


def _force_tolerance_ratio(run: RunData) -> float:
    solver = run.manifest["solver"]
    tolerance = _number(solver.get("relative_residual_tolerance"),
                        "solver.relative_residual_tolerance")
    dt = _protocol(solver)["dt_s"]
    if tolerance <= 0 or dt <= 0:
        raise EvidenceError("invalid_force_tolerance_ratio",
                            "relative residual tolerance and timestep must be positive")
    return tolerance / dt


def _observed_force_imbalance_bounds(run: RunData) -> dict[str, Any]:
    if not all(any(field in row for row in run.rows) for field in FORCE_IMBALANCE_FIELDS):
        return {
            "available": False,
            "note": "free-node force-imbalance observations are absent from this export",
        }
    by_arm: dict[str, dict[str, float]] = {}
    for arm in ("bent", "held_reference"):
        arm_rows = [row for row in run.rows if row["arm"] == arm]
        by_arm[arm] = {
            field: max((row[field] for row in arm_rows if field in row), default=0.0)
            for field in FORCE_IMBALANCE_FIELDS
        }
    return {
        "available": True,
        "maximum_over_recorded_accepted_steps_by_arm_N": by_arm,
        "note": "observed free-node force residuals from the probe; measurements, not a solver acceptance threshold",
    }


def _force_tolerance_contract(a: RunData, b: RunData, enabled: bool) -> dict[str, Any]:
    def one(run: RunData) -> dict[str, Any]:
        solver = run.manifest["solver"]
        protocol = _protocol(solver)
        tolerance = _number(solver.get("relative_residual_tolerance"),
                            "solver.relative_residual_tolerance")
        ratio = _force_tolerance_ratio(run)
        return {
            "dt_s": protocol["dt_s"],
            "relative_residual_tolerance": tolerance,
            "relative_residual_tolerance_over_dt_s_inverse": ratio,
            "force_equivalent_at_one_Ns_residual_scale_N": ratio,
            "volume_tolerance": _number(solver.get("volume_tolerance"), "solver.volume_tolerance"),
            "pressure_tolerance": _number(solver.get("pressure_tolerance"), "solver.pressure_tolerance"),
            "transport_tolerance": _number(solver.get("transport_tolerance"), "solver.transport_tolerance"),
            "observed_free_force_imbalance": _observed_force_imbalance_bounds(run),
        }

    base, candidate = one(a), one(b)
    return {
        "enabled": enabled,
        "applies_only_to": "temporal",
        "matching_quantity": "relative_residual_tolerance / dt_s",
        "ratio_rounding_allowance": {
            "relative": FORCE_MATCH_RATIO_REL_TOL,
        },
        "baseline": base,
        "candidate": candidate,
        "force_bound_interpretation": {
            "force_equivalent_at_one_Ns_residual_scale_N": base["force_equivalent_at_one_Ns_residual_scale_N"],
            "absolute_force_bound_N": None,
            "absolute_force_bound_unavailable_reason": (
                "the export omits the residual normalization scale needed to turn the relative "
                "solver threshold into an absolute force bound; per-step free-force residuals "
                "above are observed measurements, not the solver gate"
            ),
        },
    }


def _phase_durations(run: RunData) -> dict[str, float]:
    p = _protocol(run.manifest["solver"])
    return {phase: p[phase] * p["dt_s"] for phase in PHASE_ORDER}


def _mesh_resolution(run: RunData) -> tuple[int, int, int]:
    mesh = run.manifest.get("mesh", {})
    return tuple(_int(mesh.get(key), f"mesh.{key}") for key in ("nx_per_pitch", "ny", "thickness_slices"))


def _same_protocol_except_axis(a: RunData, b: RunData, kind: str,
                               *, force_matched_tolerance: bool = False) -> dict[str, Any]:
    pa, pb = _protocol(a.manifest["solver"]), _protocol(b.manifest["solver"])
    if not _close(pa["bend_angle_deg"], pb["bend_angle_deg"], abs_tol=1e-9):
        raise IncompatibleError("bend_angle_mismatch", "peak grip rotation differs")
    da, db = _phase_durations(a), _phase_durations(b)
    ra, rb = _mesh_resolution(a), _mesh_resolution(b)
    if kind == "spatial":
        if not _close(pa["dt_s"], pb["dt_s"], abs_tol=1e-15):
            raise IncompatibleError("timestep_mismatch", "spatial comparison requires identical timestep")
        if any(pa[p] != pb[p] for p in PHASE_ORDER):
            raise IncompatibleError("phase_protocol_mismatch", "spatial comparison requires identical phase step counts")
        if ra == rb:
            raise IncompatibleError("no_spatial_change", "spatial comparison did not change mesh resolution")
    elif kind == "temporal":
        if ra != rb:
            raise IncompatibleError("mesh_resolution_mismatch", "temporal comparison requires identical mesh resolution")
        if _close(pa["dt_s"], pb["dt_s"], abs_tol=1e-15):
            raise IncompatibleError("no_timestep_change", "temporal comparison did not change timestep")
        for phase in PHASE_ORDER:
            if not _close(da[phase], db[phase], rel=1e-9, abs_tol=1e-12):
                raise IncompatibleError("phase_duration_mismatch", f"physical duration of {phase} differs")
        coarse, fine = (pa["dt_s"], pb["dt_s"]) if pa["dt_s"] > pb["dt_s"] else (pb["dt_s"], pa["dt_s"])
        ratio = coarse / fine
        if not _close(ratio, round(ratio), rel=1e-9, abs_tol=1e-9):
            raise IncompatibleError("incommensurate_time_sampling", "finer timestep does not nest the coarse sampling grid")
        if force_matched_tolerance:
            tolerance_ratio_a = _force_tolerance_ratio(a)
            tolerance_ratio_b = _force_tolerance_ratio(b)
            if not math.isclose(
                tolerance_ratio_a, tolerance_ratio_b,
                rel_tol=FORCE_MATCH_RATIO_REL_TOL,
                abs_tol=0.0,
            ):
                raise IncompatibleError(
                    "force_tolerance_ratio_mismatch",
                    "relative_residual_tolerance / dt_s differs between runs "
                    f"({tolerance_ratio_a:.12g} vs {tolerance_ratio_b:.12g} s^-1; "
                    f"allowed relative rounding {FORCE_MATCH_RATIO_REL_TOL:g})",
                )
    elif kind == "load_rate":
        if ra != rb:
            raise IncompatibleError("mesh_resolution_mismatch", "load-rate comparison requires identical mesh resolution")
        if not _close(pa["dt_s"], pb["dt_s"], abs_tol=1e-15):
            raise IncompatibleError("timestep_mismatch", "load-rate comparison requires identical timestep")
        if _close(da["loading"], db["loading"], rel=1e-9, abs_tol=1e-12):
            raise IncompatibleError("no_load_rate_change", "load-rate comparison did not change loading duration")
        for phase in ("hold", "unloading", "relaxation", "release"):
            if not _close(da[phase], db[phase], rel=1e-9, abs_tol=1e-12):
                raise IncompatibleError("phase_duration_mismatch", f"load-rate comparison changes {phase} duration")
    else:
        raise EvidenceError("unknown_comparison_kind", f"unsupported comparison kind {kind!r}")
    return {
        "kind": kind,
        "baseline_phase_durations_s": da,
        "candidate_phase_durations_s": db,
        "baseline_dt_s": pa["dt_s"],
        "candidate_dt_s": pb["dt_s"],
        "baseline_mesh_resolution": list(ra),
        "candidate_mesh_resolution": list(rb),
        "duration_difference_is_declared_axis": kind == "load_rate",
    }


def _request_phases(a: RunData, b: RunData, requested: Iterable[str] | None) -> list[str]:
    if requested is None:
        phases = [p for p in PHASE_ORDER if p in a.phases and p in b.phases]
    else:
        phases = list(dict.fromkeys(requested))
    if not phases:
        raise EvidenceError("no_common_phases", "no common phase endpoints were requested")
    for phase in phases:
        if phase not in PHASE_ORDER:
            raise EvidenceError("unknown_requested_phase", f"unknown phase {phase!r}")
        if phase not in a.phases or phase not in b.phases:
            if phase == "release":
                raise EvidenceError("inconclusive_missing_free_release", "free-release phase was requested but not recorded in both runs")
            raise EvidenceError("inconclusive_missing_phase", f"requested phase {phase} is not present in both runs")
    return phases


def _observation_endpoint(run: RunData, phase: str, arm: int) -> dict[str, Any]:
    end = _phase_ends(run)[phase]
    return run.row_by_key[(end - 1, arm)]


def _mesh_summary(run: RunData) -> dict[str, Any]:
    volumes = [0.0] * len(run.materials_by_index)
    counts = [0] * len(run.materials_by_index)
    for i, mi in enumerate(run.mesh["material_indices"]):
        volumes[mi] += run.volumes_m3[i]
        counts[mi] += 1
    densities = run.mesh.get("material_densities_kg_m3")
    if not isinstance(densities, list) or len(densities) != len(volumes):
        raise EvidenceError("unknown_material_density_map", "mesh material density vector is missing or mismatched")
    by_material = {}
    for i, cell in enumerate(run.materials_by_index):
        density = _number(densities[i], f"mesh material density {i}")
        by_material[cell["name"]] = {
            "tetrahedra": counts[i], "reference_volume_m3": volumes[i],
            "reference_mass_kg": volumes[i] * density,
            "manifest_volume_m3": _number(cell.get("volume_m3"), f"manifest material volume {i}"),
            "manifest_mass_kg": _number(cell.get("mass_kg"), f"manifest material mass {i}"),
        }
    return {
        "nodes": len(run.mesh["nodes_m"]), "tetrahedra": len(run.mesh["tetrahedra"]),
        "resolution": list(_mesh_resolution(run)), "minimum_reference_tet_volume_m3": min(run.volumes_m3),
        "regional_discretized_geometry": by_material,
    }


def _run_endpoint_metrics(run: RunData, phases: list[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for phase in phases:
        arms: dict[str, Any] = {}
        for env, arm in ((0, "bent"), (1, "held_reference")):
            row = _observation_endpoint(run, phase, env)
            stats = _state_stats(run, phase, arm)
            arms[arm] = {
                "step": row["step"], "time_s": row["time_s"], "phase": phase,
                "reaction_force_N": [row["reaction_x_N"], row["reaction_y_N"], row["reaction_z_N"]],
                "reaction_moment_y_Nm": row["reaction_moment_y_Nm"],
                "target_angle_deg": row["target_angle_deg"],
                "measured_right_grip_angle_deg": row.get("measured_right_grip_angle_deg"),
                "right_grip_constrained": row.get("right_grip_constrained"),
                "current_fixed_nodes": row.get("current_fixed_nodes"),
                "max_free_displacement_m": row["max_free_displacement_m"],
                "max_free_speed_m_s": row["max_free_speed_m_s"],
                "kinetic_energy_J": row["kinetic_energy_J"],
                "certificate_residual": row["certificate_residual"],
                "plastic_strain": stats,
            }
        output[phase] = arms
    return output


def _time_key(value: float) -> int:
    return round(value * 1.0e12)


def _phase_coordinate(run: RunData, row: dict[str, Any], phase: str) -> float:
    if phase in ("loading", "unloading"):
        peak = _protocol(run.manifest["solver"])["bend_angle_deg"]
        return 0.0 if peak == 0 else row["target_angle_deg"] / peak
    end_step = _phase_ends(run)[phase]
    count = _protocol(run.manifest["solver"])[phase]
    start_time = (end_step - count) * _protocol(run.manifest["solver"])["dt_s"]
    duration = count * _protocol(run.manifest["solver"])["dt_s"]
    return 0.0 if duration == 0 else (row["time_s"] - start_time) / duration


def _interpolate(points: list[tuple[float, dict[str, Any]]], x: float, field: str) -> float:
    points = sorted(points, key=lambda pair: pair[0])
    if x <= points[0][0]:
        return points[0][1][field]
    if x >= points[-1][0]:
        return points[-1][1][field]
    for (x0, r0), (x1, r1) in zip(points, points[1:]):
        if x0 <= x <= x1:
            if _close(x0, x1, abs_tol=1e-15):
                return r1[field]
            alpha = (x - x0) / (x1 - x0)
            return r0[field] + alpha * (r1[field] - r0[field])
    raise EvidenceError("curve_interpolation_failed", "cannot interpolate within observed phase")


def _paired_curve_metrics(a: RunData, b: RunData, phases: list[str], kind: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for phase in phases:
        output[phase] = {}
        for arm, env in (("bent", 0), ("held_reference", 1)):
            pa = [(row["time_s"] if kind != "load_rate" else _phase_coordinate(a, row, phase), row)
                  for row in a.rows if row["environment"] == env and row["phase"] == phase]
            pb = [(row["time_s"] if kind != "load_rate" else _phase_coordinate(b, row, phase), row)
                  for row in b.rows if row["environment"] == env and row["phase"] == phase]
            if not pa or not pb:
                raise EvidenceError("missing_curve_phase", f"no {phase} curve rows for {arm}")
            if kind == "load_rate":
                coords = sorted(set(x for x, _ in pa) | set(x for x, _ in pb))
                matching = [(x, None, None) for x in coords]
                sampling = "piecewise-linear interpolation by normalized phase angle/time"
            else:
                # Spatial pairs use identical grids; temporal pairs use their
                # nested common times. Every coarse point must exist in the fine run.
                coarse, fine = (pa, pb) if len(pa) <= len(pb) else (pb, pa)
                fine_by_time = {_time_key(x): row for x, row in fine}
                matching = []
                for x, row in coarse:
                    other = fine_by_time.get(_time_key(x))
                    if other is None:
                        raise IncompatibleError("time_sampling_mismatch", f"no exact common sample at {x:.12g}s in {phase}")
                    matching.append((x, row if coarse is pa else other, other if coarse is pa else row))
                sampling = "exact common physical times; coarse samples nested in fine grid"
            arm_metrics: dict[str, Any] = {"sampling": sampling, "matched_samples": []}
            for field in CURVE_FIELDS:
                deltas: list[float] = []
                values_a: list[float] = []
                values_b: list[float] = []
                for x, ra, rb in matching:
                    if kind == "load_rate":
                        va = _interpolate(pa, x, field)
                        vb = _interpolate(pb, x, field)
                    else:
                        assert ra is not None and rb is not None
                        # Reconstruct A/B order after choosing the shorter grid.
                        if len(pa) <= len(pb):
                            va, vb = ra[field], rb[field]
                        else:
                            va, vb = rb[field], ra[field]
                    values_a.append(va)
                    values_b.append(vb)
                    deltas.append(vb - va)
                rms_difference = math.sqrt(sum(x*x for x in deltas) / len(deltas))
                rms_baseline = math.sqrt(sum(x*x for x in values_a) / len(values_a))
                entry = {
                    "max_abs_difference": max(abs(x) for x in deltas),
                    "rms_difference": rms_difference,
                    "relative_rms_difference": (rms_difference / rms_baseline) if rms_baseline > 1.0e-30 else None,
                    "baseline_max_abs": max(abs(x) for x in values_a),
                    "candidate_max_abs": max(abs(x) for x in values_b),
                    "sample_count": len(deltas),
                }
                arm_metrics[field] = entry
            arm_metrics["coordinate_samples"] = [x for x, _, _ in matching]
            output[phase][arm] = arm_metrics
    return output


def _regional_geometry_delta(a: RunData, b: RunData) -> dict[str, Any]:
    ma, mb = _mesh_summary(a), _mesh_summary(b)
    result: dict[str, Any] = {"baseline": ma, "candidate": mb, "by_material": {}}
    for name in ma["regional_discretized_geometry"]:
        x = ma["regional_discretized_geometry"][name]
        y = mb["regional_discretized_geometry"].get(name)
        if y is None:
            raise IncompatibleError("material_region_mismatch", f"material region {name} differs")
        result["by_material"][name] = {
            "volume_difference_m3": y["reference_volume_m3"] - x["reference_volume_m3"],
            "volume_relative_difference": ((y["reference_volume_m3"] - x["reference_volume_m3"]) / x["reference_volume_m3"]
                                           if x["reference_volume_m3"] else None),
            "mass_difference_kg": y["reference_mass_kg"] - x["reference_mass_kg"],
            "tetrahedra_difference": y["tetrahedra"] - x["tetrahedra"],
        }
    result["note"] = "reference-region volume and mass may vary with mesh discretization; reported, not used as a material-source match gate"
    return result


def _release_response(run: RunData) -> dict[str, Any] | None:
    protocol = _protocol(run.manifest["solver"])
    if protocol["release"] <= 0:
        return None
    release_index = PHASE_ORDER.index("release")
    preceding = [phase for phase in PHASE_ORDER[:release_index]
                 if protocol[phase] > 0]
    if not preceding:
        raise EvidenceError("missing_release_reference", "free release has no preceding clamped phase")
    before_phase = preceding[-1]
    before = _observation_endpoint(run, before_phase, 0)
    after = _observation_endpoint(run, "release", 0)
    before_state = _state_stats(run, before_phase, "bent")["paper_total"]
    after_state = _state_stats(run, "release", "bent")["paper_total"]
    return {
        "pre_release_phase": before_phase,
        "boundary_verified": before.get("right_grip_constrained") == 1 and after.get("right_grip_constrained") == 0,
        "before_release_endpoint": {k: before.get(k) for k in ("time_s", "measured_right_grip_angle_deg", "max_free_displacement_m", "max_free_speed_m_s", "kinetic_energy_J")},
        "after_release_endpoint": {k: after.get(k) for k in ("time_s", "measured_right_grip_angle_deg", "max_free_displacement_m", "max_free_speed_m_s", "kinetic_energy_J")},
        "right_grip_angle_change_deg": after["measured_right_grip_angle_deg"] - before["measured_right_grip_angle_deg"],
        "free_displacement_change_m": after["max_free_displacement_m"] - before["max_free_displacement_m"],
        "kinetic_energy_J": after["kinetic_energy_J"],
        "paper_plastic_mean_frobenius_change": after_state["volume_mean_frobenius_ep"] - before_state["volume_mean_frobenius_ep"],
        "note": "release is a distinct free-grip phase; constrained relaxation is not substituted for it",
    }


def _run_report(run: RunData, phases: list[str]) -> dict[str, Any]:
    solver = run.manifest["solver"]
    phase_endpoints = _run_endpoint_metrics(run, phases)
    release_response = _release_response(run)
    return {
        "path": str(run.path),
        "status": "valid_completed_native_export",
        "accepted_steps": run.result["accepted_steps"],
        "physical_validation": False,
        "compiled_world_fingerprint": run.result["compiled_world_fingerprint"],
        "input_sha256": dict(sorted(run.files.items())),
        "materials": {key: run.manifest["materials"].get(key)
                      for key in ("liner_sha256", "medium_sha256", "glue_sha256", "regional_map_sha256", "frame_map_sha256")},
        "protocol": {**_protocol(solver), "phase_durations_s": _phase_durations(run)},
        "mesh": _mesh_summary(run),
        "phase_endpoints": phase_endpoints,
        "free_release_response": release_response,
        "scope": "retained native software mechanics data; not a physical validation",
    }


def _single_run_curves(run: RunData, phases: list[str]) -> dict[str, Any]:
    curves: dict[str, Any] = {}
    for phase in phases:
        curves[phase] = {}
        for arm, env in (("bent", 0), ("held_reference", 1)):
            samples = []
            for row in run.rows:
                if row["environment"] != env or row["phase"] != phase:
                    continue
                samples.append({
                    "step": row["step"], "time_s": row["time_s"],
                    "target_angle_deg": row["target_angle_deg"],
                    "measured_right_grip_angle_deg": row.get("measured_right_grip_angle_deg"),
                    "right_grip_constrained": row.get("right_grip_constrained"),
                    "reaction_force_N": [row["reaction_x_N"], row["reaction_y_N"], row["reaction_z_N"]],
                    "reaction_moment_y_Nm": row["reaction_moment_y_Nm"],
                    "kinetic_energy_J": row["kinetic_energy_J"],
                    "max_free_speed_m_s": row["max_free_speed_m_s"],
                    "max_free_displacement_m": row["max_free_displacement_m"],
                    "certificate_residual": row["certificate_residual"],
                })
            if not samples:
                raise EvidenceError("missing_curve_phase", f"no {phase} curve rows for {arm}")
            curves[phase][arm] = samples
    return curves


def _endpoint_screen(endpoints_a: dict[str, Any], endpoints_b: dict[str, Any], phases: list[str],
                     relative_tolerance: float, plastic_floor: float, moment_floor: float) -> dict[str, Any]:
    thresholds = (relative_tolerance, plastic_floor, moment_floor)
    if not all(math.isfinite(value) for value in thresholds):
        raise EvidenceError("invalid_screen_threshold", "endpoint tolerance and absolute floors must be finite")
    if relative_tolerance < 0 or plastic_floor <= 0 or moment_floor <= 0:
        raise EvidenceError("invalid_screen_threshold", "endpoint tolerance must be nonnegative and absolute floors positive")
    metrics: dict[str, Any] = {}
    all_within = True
    for phase in phases:
        a = endpoints_a[phase]["bent"]
        b = endpoints_b[phase]["bent"]
        plastic_a = a["plastic_strain"]["paper_total"]["volume_rms_frobenius_ep"]
        plastic_b = b["plastic_strain"]["paper_total"]["volume_rms_frobenius_ep"]
        moment_a = a["reaction_moment_y_Nm"]
        moment_b = b["reaction_moment_y_Nm"]
        plastic_rel = abs(plastic_b - plastic_a) / max(abs(plastic_a), abs(plastic_b), plastic_floor)
        moment_rel = abs(moment_b - moment_a) / max(abs(moment_a), abs(moment_b), moment_floor)
        plastic_ok = plastic_rel <= relative_tolerance
        moment_ok = moment_rel <= relative_tolerance
        all_within = all_within and plastic_ok and moment_ok
        metrics[phase] = {
            "plastic_volume_rms_frobenius_ep": {
                "baseline": plastic_a, "candidate": plastic_b, "absolute_difference": abs(plastic_b-plastic_a),
                "relative_difference_with_floor": plastic_rel, "absolute_floor": plastic_floor,
                "within_tolerance": plastic_ok,
            },
            "reaction_moment_y_Nm": {
                "baseline": moment_a, "candidate": moment_b, "absolute_difference": abs(moment_b-moment_a),
                "relative_difference_with_floor": moment_rel, "absolute_floor": moment_floor,
                "within_tolerance": moment_ok,
            },
        }
    return {
        "status": "within_declared_thresholds" if all_within else "exceeds_declared_thresholds",
        "relative_tolerance": relative_tolerance,
        "metrics_by_phase": metrics,
        "interpretation": "narrow numerical screen only; it does not establish convergence, physical fidelity, or qualification",
    }


def analyze_pair(baseline_dir: str | Path, candidate_dir: str | Path, kind: str,
                 requested_phases: Iterable[str] | None = None,
                 endpoint_relative_tolerance: float | None = None,
                 plastic_absolute_floor: float = 1.0e-8,
                 moment_absolute_floor: float = 1.0e-6,
                 force_matched_tolerance: bool = False) -> dict[str, Any]:
    base_path, cand_path = Path(baseline_dir), Path(candidate_dir)
    report: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "analyzer_sha256": _analyzer_digest(),
        "status": "inconclusive_evidence",
        "comparison": {"kind": kind, "baseline": str(base_path), "candidate": str(cand_path)},
        "physical_validation": False,
        "convergence_verdict": "not_assessed; acceptance is not inferred from completed step count",
        "reasons": [],
    }
    try:
        a = _load_run(base_path)
        b = _load_run(cand_path)
        if force_matched_tolerance and kind != "temporal":
            raise IncompatibleError(
                "force_matched_tolerance_requires_temporal",
                "force-matched tolerance is only admitted for temporal comparisons",
            )
        phases = _request_phases(a, b, requested_phases)
        _same_materials(a, b)
        _same_geometry(a, b)
        _same_solver_controls(a, b,
                              allow_force_matched_tolerance=force_matched_tolerance)
        axis = _same_protocol_except_axis(
            a, b, kind, force_matched_tolerance=force_matched_tolerance)
        axis["force_matched_tolerance"] = _force_tolerance_contract(
            a, b, force_matched_tolerance)
        if "release" in phases:
            if a.manifest["solver"].get("release_steps", 0) <= 0 or b.manifest["solver"].get("release_steps", 0) <= 0:
                raise EvidenceError("inconclusive_missing_free_release", "requested free-release comparison lacks release in both runs")
            if a.manifest["solver"].get("release_policy") != b.manifest["solver"].get("release_policy"):
                raise IncompatibleError("release_policy_mismatch", "free-release policies differ")
        endpoints_a = _run_endpoint_metrics(a, phases)
        endpoints_b = _run_endpoint_metrics(b, phases)
        pair_metrics = _paired_curve_metrics(a, b, phases, kind)
        endpoint_deltas: dict[str, Any] = {}
        for phase in phases:
            endpoint_deltas[phase] = {}
            for arm in ("bent", "held_reference"):
                x, y = endpoints_a[phase][arm], endpoints_b[phase][arm]
                endpoint_deltas[phase][arm] = {
                    "reaction_force_difference_N": [y["reaction_force_N"][i] - x["reaction_force_N"][i] for i in range(3)],
                    "reaction_moment_y_difference_Nm": y["reaction_moment_y_Nm"] - x["reaction_moment_y_Nm"],
                    "kinetic_energy_difference_J": y["kinetic_energy_J"] - x["kinetic_energy_J"],
                    "max_free_speed_difference_m_s": y["max_free_speed_m_s"] - x["max_free_speed_m_s"],
                    "max_free_displacement_difference_m": y["max_free_displacement_m"] - x["max_free_displacement_m"],
                    "measured_right_grip_angle_difference_deg": (
                        None if x["measured_right_grip_angle_deg"] is None or y["measured_right_grip_angle_deg"] is None
                        else y["measured_right_grip_angle_deg"] - x["measured_right_grip_angle_deg"]),
                    "plastic_strain_by_material": {},
                }
                for name in x["plastic_strain"]["by_material"]:
                    xp = x["plastic_strain"]["by_material"][name]
                    yp = y["plastic_strain"]["by_material"][name]
                    endpoint_deltas[phase][arm]["plastic_strain_by_material"][name] = {
                        "volume_rms_frobenius_ep_difference": yp["volume_rms_frobenius_ep"] - xp["volume_rms_frobenius_ep"],
                        "plasticized_volume_fraction_difference": yp["plasticized_volume_fraction_ep_norm_gt_1e-6"] - xp["plasticized_volume_fraction_ep_norm_gt_1e-6"],
                        "volume_difference_m3": yp["reference_volume_m3"] - xp["reference_volume_m3"],
                    }
        report.update({
            "status": "analyzed_descriptive",
            "phases": phases,
            "comparison_contract": axis,
            "baseline_run": _run_report(a, phases),
            "candidate_run": _run_report(b, phases),
            "reaction_dynamics_and_energy_curves": pair_metrics,
            "phase_endpoint_differences": endpoint_deltas,
            "regional_mesh_sensitivity": _regional_geometry_delta(a, b),
            "interpretation": "descriptive numerical sensitivity only; no convergence or physical-validity claim without separately declared thresholds and evidence",
        })
        if endpoint_relative_tolerance is not None:
            report["endpoint_resolution_screen"] = _endpoint_screen(
                endpoints_a, endpoints_b, phases, endpoint_relative_tolerance,
                plastic_absolute_floor, moment_absolute_floor)
    except IncompatibleError as exc:
        report["status"] = exc.state
        report["reasons"].append({"code": exc.code, "message": str(exc)})
    except FailedRunError as exc:
        report["status"] = exc.state
        report["reasons"].append({"code": exc.code, "message": str(exc)})
    except EvidenceError as exc:
        report["status"] = exc.state
        report["reasons"].append({"code": exc.code, "message": str(exc)})
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        report["status"] = "inconclusive_evidence"
        report["reasons"].append({"code": "malformed_evidence", "message": str(exc)})
    return report


def analyze_run(run_dir: str | Path, requested_phases: Iterable[str] | None = None) -> dict[str, Any]:
    path = Path(run_dir)
    report: dict[str, Any] = {
        "schema": ANALYSIS_SCHEMA,
        "analyzer_sha256": _analyzer_digest(),
        "analysis_type": "single_run_observation",
        "status": "inconclusive_evidence",
        "run": str(path),
        "physical_validation": False,
        "reasons": [],
    }
    try:
        run = _load_run(path)
        phases = _request_phases(run, run, requested_phases)
        run_report = _run_report(run, phases)
        report.update({
            "status": "analyzed_descriptive",
            "phases": phases,
            "run_evidence": run_report,
            "reaction_dynamics_and_energy_curves": _single_run_curves(run, phases),
            "interpretation": "descriptive native software mechanics observations; no paired sensitivity, convergence, or physical-validity claim",
        })
    except FailedRunError as exc:
        report["status"] = exc.state
        report["reasons"].append({"code": exc.code, "message": str(exc)})
    except EvidenceError as exc:
        report["status"] = exc.state
        report["reasons"].append({"code": exc.code, "message": str(exc)})
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        report["status"] = "inconclusive_evidence"
        report["reasons"].append({"code": "malformed_evidence", "message": str(exc)})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("spatial", "temporal", "load_rate"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--run", type=Path,
                        help="analyze one retained run without a comparison; useful for phase and free-release observations")
    parser.add_argument("--phase", action="append", choices=PHASE_ORDER,
                        help="endpoint phase(s) to require; defaults to all common recorded phases")
    parser.add_argument("--endpoint-relative-tolerance", type=float,
                        help="optional narrow endpoint screen for plastic RMS and y moment")
    parser.add_argument("--plastic-absolute-floor", type=float, default=1.0e-8,
                        help="denominator floor for plastic-strain relative difference")
    parser.add_argument("--moment-absolute-floor", type=float, default=1.0e-6,
                        help="denominator floor for reaction-moment relative difference in N m")
    parser.add_argument("--force-matched-tolerance", action="store_true",
                        help="temporal studies only: allow differing residual tolerances only when tolerance/dt matches")
    parser.add_argument("--output", type=Path, help="write the JSON report to this path; stdout if omitted")
    args = parser.parse_args(argv)
    if args.force_matched_tolerance and args.kind != "temporal":
        parser.error("--force-matched-tolerance is usable only with --kind temporal")
    if args.run is not None:
        if args.kind is not None or args.baseline is not None or args.candidate is not None:
            parser.error("--run is exclusive with --kind, --baseline, and --candidate")
        if args.endpoint_relative_tolerance is not None:
            parser.error("--endpoint-relative-tolerance requires a paired comparison")
        result = analyze_run(args.run, args.phase)
    else:
        if args.kind is None or args.baseline is None or args.candidate is None:
            parser.error("paired analysis requires --kind, --baseline, and --candidate")
        result = analyze_pair(args.baseline, args.candidate, args.kind, args.phase,
                              args.endpoint_relative_tolerance,
                              args.plastic_absolute_floor, args.moment_absolute_floor,
                              args.force_matched_tolerance)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(encoded)
        except FileExistsError:
            parser.error(f"--output already exists; refusing to overwrite: {args.output}")
    else:
        sys.stdout.write(encoded)
    return 0 if result["status"] == "analyzed_descriptive" else 2


if __name__ == "__main__":
    raise SystemExit(main())
