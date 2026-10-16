#!/usr/bin/env python3
"""Rebuild the constrained cached-tangent FEM operator from a rich capture.

The tool is diagnostic only. It does not solve a Matter step, symmetrize the
captured operator, or treat an eigenvalue/condition estimate as solver
qualification. Its native input is the read-only cached dP/dF snapshot emitted
by the production FEM operator-FD diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigsh, svds
import cardboard_solver_environment as solver_environment


CAPTURE_SCHEMA = "numi.cardboard.fem-newton-capture.v1"
REPORT_SCHEMA = "numi.cardboard.cached-operator-audit.v1"
REQUIRED_BOUND_SOURCES = (
    "matter/include/numi/matter/shared.h",
    "matter/src/runtime.mm",
    "matter/src/metal/common.metalinc",
    "matter/src/metal/fem.metalinc",
    "matter/src/metal/fgmres.metalinc",
    "matter/tools/cardboard_probe.mm",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_array(values: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must have finite shape {shape}, got {result.shape}")
    return result


def captured_f32_array(values: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    """Recover captured Metal ``float`` inputs as FP64 values of exact FP32s.

    The native JSON writer uses ``max_digits10`` so each float32 value can be
    round-tripped, but JSON parsing first creates a binary64 value from its
    decimal spelling. Promoting that spelling directly to the offline FP64
    assembly retains a few non-native decimal bits. Round through float32
    before promotion for fields copied from Metal buffers. Authored/external
    FP64 reference data must continue to use ``finite_array`` instead.
    """
    values64 = finite_array(values, shape, label)
    with np.errstate(over="ignore", invalid="ignore"):
        values32 = values64.astype(np.float32)
    if not np.isfinite(values32).all():
        raise ValueError(f"{label} is outside the finite FP32 range")
    return values32.astype(np.float64)


def captured_f32_scalar(value: Any, label: str) -> float:
    return float(captured_f32_array([value], (1,), label)[0])


def rotation_from_xyzw(quaternion: Any) -> np.ndarray:
    q = finite_array(quaternion, (4,), "material frame quaternion")
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 1.0e-30:
        raise ValueError("material frame quaternion has zero/nonfinite norm")
    x, y, z, w = q / norm
    return np.array([
        [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w), 2.0 * (x*z + y*w)],
        [2.0 * (x*y + z*w), 1.0 - 2.0 * (x*x + z*z), 2.0 * (y*z - x*w)],
        [2.0 * (x*z - y*w), 2.0 * (y*z + x*w), 1.0 - 2.0 * (x*x + y*y)],
    ], dtype=np.float64)


def local_tet_stiffness(inverse_rest: Any, volume: float, tangent_local: Any,
                        frame_rotation: Any, frame_active: bool) -> np.ndarray:
    """Return K_e=-d(f_internal)/d(v) in xyz-by-node ordering.

    The cache maps row-major local direction components to row-major local
    dP. FEM geometry uses dF_world = [dv1-dv0, dv2-dv0, dv3-dv0] invRest.
    The material-frame convention is F_local=F_world R, P_world=P_local R^T.
    """
    inv = finite_array(inverse_rest, (3, 3), "inverse-rest matrix")
    tangent = finite_array(tangent_local, (9, 9), "cached local tangent")
    if not math.isfinite(volume) or volume <= 0:
        raise ValueError("tetrahedron volume must be positive and finite")
    # Unframed tets use the source's canonical zero quaternion, so do not
    # normalize or interpret it when the object frame flag is clear.
    rot = (rotation_from_xyzw(frame_rotation) if frame_active
           else np.eye(3, dtype=np.float64))
    stiffness = np.zeros((12, 12), dtype=np.float64)

    for column in range(12):
        dof_node, dof_axis = divmod(column, 3)
        dv = np.zeros((4, 3), dtype=np.float64)
        dv[dof_node, dof_axis] = 1.0
        # Metal constructs float3x3(edge1, edge2, edge3), i.e. these are
        # columns. The captured inverseRest matrix is row-major.
        edge_columns = np.column_stack((dv[1] - dv[0], dv[2] - dv[0], dv[3] - dv[0]))
        dF_world = edge_columns @ inv
        dF_local = dF_world @ rot
        dP_local = (tangent @ dF_local.reshape(9)).reshape(3, 3)
        dP_world = dP_local @ rot.T
        # tetrahedronForce: h=-V*P*invRest^T; h's columns are node 1..3
        # force derivatives; node 0 balances their sum. K is minus this map.
        h = -volume * dP_world @ inv.T
        force_derivative = np.zeros((4, 3), dtype=np.float64)
        force_derivative[1:] = h.T
        force_derivative[0] = -np.sum(force_derivative[1:], axis=0)
        stiffness[:, column] = -force_derivative.reshape(12)
    return stiffness


def capture_records(path: Path) -> tuple[dict, list[dict], list[dict], dict]:
    header = None
    integrity = None
    nodes: list[dict] = []
    tets: list[dict] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"capture line {line_number} is invalid JSON") from error
            kind = record.get("record")
            if kind == "header":
                if header is not None:
                    raise ValueError("capture has multiple headers")
                header = record
            elif kind == "node":
                nodes.append(record)
            elif kind == "tetrahedron":
                tets.append(record)
            elif kind == "integrity":
                if integrity is not None:
                    raise ValueError("capture has multiple integrity records")
                integrity = record
            else:
                raise ValueError(f"capture line {line_number} has unknown record kind")
    if header is None or integrity is None:
        raise ValueError("capture lacks one header or terminal integrity record")
    return header, nodes, tets, integrity


def json_marker_lines(path: Path, prefix: str) -> list[dict]:
    found = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            at = line.find(prefix)
            if at < 0:
                continue
            payload = line[at + len(prefix):].strip()
            try:
                found.append(json.loads(payload))
            except json.JSONDecodeError as error:
                raise ValueError(f"malformed {prefix} telemetry") from error
    return found


def _source_manifest_path(snapshot: Path, relative: str) -> Path:
    """Resolve both legacy source-relative and current `source/...` bindings."""
    normalized = relative.removeprefix("source/")
    return snapshot / normalized


def _manifest_file(relative: str, snapshot: Path, runtime_dir: Path | None,
                   prediction: dict) -> tuple[Path | None, str]:
    """Resolve a prediction input key without weakening its recorded hash.

    Prediction v1 has appeared with source keys in both `matter/...` and
    `source/matter/...` form, and build products under `runtime/...`. External
    inputs are accepted only when the exact path is listed in external_inputs.
    """
    key = Path(relative)
    if key.is_absolute():
        return key, "absolute_input"
    if relative.startswith("source/"):
        return _source_manifest_path(snapshot, relative), "source_snapshot"
    if relative.startswith("runtime/"):
        if runtime_dir is None:
            return None, "runtime_dir"
        return runtime_dir / relative[len("runtime/"):], "runtime_dir"
    if relative.startswith("external/"):
        tail = relative[len("external/"):]
        matches = [Path(value) for value in prediction.get("external_inputs", [])
                   if isinstance(value, str) and Path(value).name == Path(tail).name]
        if len(matches) != 1:
            return None, "external_input"
        return matches[0], "external_input"
    if runtime_dir is not None and relative.startswith(runtime_dir.name + "/"):
        return runtime_dir / relative[len(runtime_dir.name) + 1:], "runtime_dir"
    return snapshot / relative, "source_snapshot"


def bind_inputs(capture: Path, prediction_path: Path, receipt_path: Path,
                stderr_path: Path, environment_index: int,
                source_snapshot_override: Path | None = None) -> dict:
    prediction = json.loads(prediction_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    prediction_schema = prediction.get("schema")
    diagnostic_prediction = (
        prediction_schema == "numi.cardboard.native-diagnostic-prediction.v1")
    if prediction_schema not in {
            "numi.cardboard.native-prediction.v1",
            "numi.cardboard.native-diagnostic-prediction.v1"}:
        raise ValueError("prediction schema is not a supported cardboard native prediction")
    if not isinstance(prediction.get("environment"), dict):
        raise ValueError("prediction has no sealed environment map")
    environment_record = None
    if diagnostic_prediction:
        if prediction.get("run_mode") != "diagnostic":
            raise ValueError("diagnostic prediction has the wrong run mode")
        if receipt.get("run_mode") != "diagnostic":
            raise ValueError("diagnostic receipt does not identify a diagnostic run")
        environment_path = prediction_path.parent / "native-environment.json"
        if not environment_path.is_file():
            raise ValueError("diagnostic prediction lacks its native environment record")
        try:
            environment_record = solver_environment.validate_run_record(
                json.loads(environment_path.read_text()))
            prediction_raw = prediction_path.read_bytes()
            solver_environment.validate_prediction_binding(
                environment_record, prediction_raw)
            if environment_record.get("prediction_sha256") != sha256(prediction_path):
                raise ValueError("diagnostic environment record prediction hash mismatch")
            if environment_record.get("schema") != \
                    solver_environment.DIAGNOSTIC_RUN_SCHEMA:
                raise ValueError("diagnostic prediction is paired with a non-diagnostic environment record")
            contract = environment_record["contract"]
            if prediction.get("effective_native_environment_contract_sha256") != \
                    contract.get("sha256"):
                raise ValueError("diagnostic prediction environment contract hash mismatch")
            fd_capture_prefix = contract.get("capture_prefixes", {}).get(
                "NM_FEM_OPERATOR_FD_CAPTURE_PATH")
            if (not isinstance(fd_capture_prefix, str) or
                    prediction["environment"].get(
                        "NM_FEM_OPERATOR_FD_CAPTURE_PATH") != fd_capture_prefix or
                    capture.resolve(strict=True) != Path(fd_capture_prefix).resolve(strict=True)):
                raise ValueError("diagnostic capture is not the exact operator-FD path in the environment contract")
            diagnostic_snapshot = (source_snapshot_override or
                Path(prediction.get("source_snapshot", "")))
            if not diagnostic_snapshot.is_dir():
                raise ValueError("diagnostic prediction does not identify an available source snapshot")
            solver_environment.validate_source_registry(diagnostic_snapshot)
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(f"invalid diagnostic environment binding: {error}") from error
        diagnostic_inventory_path = prediction_path.parent / "diagnostic-captures.json"
        if not diagnostic_inventory_path.is_file():
            raise ValueError("diagnostic prediction lacks validated capture inventory")
        try:
            inventory = json.loads(diagnostic_inventory_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("diagnostic capture inventory is unreadable") from error
        if (not isinstance(inventory, dict) or
                inventory.get("schema") != "numi.cardboard.diagnostic-capture-files.v1" or
                inventory.get("status") != "validated" or
                inventory.get("environment_contract_sha256") !=
                    environment_record["contract_sha256"] or
                not isinstance(inventory.get("files"), list)):
            raise ValueError("diagnostic capture inventory is not validated for this environment contract")
        receipt_inventory = receipt.get("diagnostic_capture_validation")
        inventory_payload = {key: value for key, value in inventory.items()
                             if key not in {"schema", "environment_contract_sha256"}}
        if receipt_inventory != inventory_payload:
            raise ValueError("diagnostic capture inventory differs from the terminal receipt")
        try:
            capture_relative = str(capture.resolve(strict=True).relative_to(
                prediction_path.parent.resolve(strict=True)))
        except (OSError, ValueError) as error:
            raise ValueError("diagnostic capture is not retained under its source-bound case") from error
        capture_hash = sha256(capture)
        if not any(isinstance(item, dict) and item.get("path") == capture_relative and
                   item.get("sha256") == capture_hash
                   for item in inventory["files"]):
            raise ValueError("diagnostic capture is not listed with matching bytes in the validated inventory")
    elif prediction_schema != "numi.cardboard.native-prediction.v1":
        raise ValueError("prediction schema is not the cardboard native-prediction schema")
    selector = prediction["environment"].get("NM_FEM_OPERATOR_FD_SELECTOR")
    if not isinstance(selector, str) or not re.fullmatch(r"\d+:\d+", selector):
        raise ValueError("prediction does not bind an operator-FD root:iteration selector")
    root, iteration = (int(value) for value in selector.split(":"))

    if receipt.get("timed_out") is True:
        raise ValueError("run receipt says native process timed out")
    if "exit_code" not in receipt or "timed_out" not in receipt:
        raise ValueError("run receipt lacks terminal process status")
    if receipt.get("prediction_sha256") is not None and \
            receipt["prediction_sha256"] != sha256(prediction_path):
        raise ValueError("run receipt prediction hash does not match supplied prediction")
    if receipt.get("capture_sha256") is not None and \
            receipt["capture_sha256"] != sha256(capture):
        raise ValueError("run receipt capture hash does not match supplied capture")

    snapshot_value = prediction.get("source_snapshot")
    if source_snapshot_override is None and (not isinstance(snapshot_value, str) or not snapshot_value):
        raise ValueError("prediction does not identify a retained source snapshot")
    snapshot = source_snapshot_override or Path(snapshot_value)
    if not snapshot.is_dir():
        raise ValueError("sealed source snapshot directory is unavailable")
    inputs = prediction.get("inputs_sha256")
    if not isinstance(inputs, dict):
        raise ValueError("prediction lacks retained input hashes")
    runtime_value = prediction.get("runtime_dir")
    runtime_dir = Path(runtime_value) if isinstance(runtime_value, str) and runtime_value else None
    checked = {}
    checked_runtime = {}
    missing = []
    for relative in REQUIRED_BOUND_SOURCES:
        source_keys = [key for key in (relative, "source/" + relative)
                       if key in inputs]
        if not source_keys:
            raise ValueError(f"source binding is missing {relative}")
        hashes = {inputs[key] for key in source_keys}
        if len(hashes) != 1 or not isinstance(next(iter(hashes)), str):
            raise ValueError(f"source binding is ambiguous or malformed for {relative}")
        expected = next(iter(hashes))
        file_path = _source_manifest_path(snapshot, source_keys[0])
        if not file_path.is_file():
            raise ValueError(f"source binding is missing {relative}")
        actual = sha256(file_path)
        if actual != expected:
            raise ValueError(f"sealed source hash mismatch for {relative}")
        checked[source_keys[0]] = actual
    # Verify the complete sealed input manifest, including the executable and
    # metallibs. Build outputs are rooted in runtime_dir and are commonly not
    # duplicated into the retained source snapshot.
    for relative, expected in inputs.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("prediction input hash manifest has invalid entries")
        candidate, location = _manifest_file(relative, snapshot, runtime_dir, prediction)
        if candidate is None or not candidate.is_file():
            missing.append(relative)
            continue
        actual = sha256(candidate)
        if actual != expected:
            raise ValueError(f"sealed input hash mismatch for {relative}")
        if location == "source_snapshot":
            checked[relative] = actual
        else:
            checked_runtime[relative] = {"sha256": actual, "location": location}
    if missing:
        raise ValueError(f"sealed input files are unavailable: {missing[:5]}")

    enabled = json_marker_lines(stderr_path, "fem_operator_fd_enabled=")
    matches = [row for row in enabled
               if row.get("root") == root and row.get("iteration") == iteration]
    if len(matches) != 1:
        raise ValueError("stderr must contain exactly one matching successful operator-FD scope admission")
    iterations = json_marker_lines(stderr_path, "fem_operator_fd_iteration=")
    iteration_matches = [row for row in iterations
                         if row.get("root") == root and
                         row.get("newton_iteration") == iteration]
    if len(iteration_matches) != 1:
        raise ValueError("stderr must contain exactly one matching completed operator-FD diagnostic")
    diagnostic = iteration_matches[0]
    if diagnostic.get("cached_tangent_snapshot_available") is not True or \
            diagnostic.get("cached_tangent_snapshot_bitwise_guard_included") is not True or \
            diagnostic.get("production_inputs_bitwise_unchanged") is not True:
        raise ValueError("stderr diagnostic did not publish an unchanged guarded tangent-cache capture")
    env_rows = diagnostic.get("environments")
    if not isinstance(env_rows, list) or len(env_rows) <= environment_index or any(
            float(pair.get("input_integrity_flag", 0.0)) != 1.0
            for pair in env_rows[environment_index].get("epsilon_pairs", [])):
        raise ValueError("selected environment does not carry complete input-integrity flags")
    if len(env_rows[environment_index].get("epsilon_pairs", [])) != 2:
        raise ValueError("operator-FD diagnostic does not contain both epsilon-pair records")

    return {
        "prediction_sha256": sha256(prediction_path),
        "receipt_sha256": sha256(receipt_path),
        "stderr_sha256": sha256(stderr_path),
        "capture_sha256": sha256(capture),
        "source_snapshot": str(snapshot),
        "prediction_selector": selector,
        "prediction_environment": prediction["environment"],
        "prediction_run_mode": "diagnostic" if diagnostic_prediction else "qualification",
        "diagnostic_environment_record_sha256": (
            sha256(prediction_path.parent / "native-environment.json")
            if diagnostic_prediction else None),
        "source_hashes_verified_against_prediction": checked,
        "runtime_and_external_input_hashes_verified_against_prediction": checked_runtime,
        "receipt_exit_code": receipt.get("exit_code"),
        "receipt_timed_out": receipt.get("timed_out"),
        "scope_admission": matches[0],
        "operator_fd_iteration": diagnostic,
        "capture_run_fingerprints_match": None,
        "binary_and_metallib_receipt_binding": {
            "executable_sha256": receipt.get("executable_sha256"),
            "metallib_sha256": receipt.get("metallib_sha256"),
            "limitation": "recorded only when present in run receipt; source hashes and captured execution fingerprints remain separate bindings",
        },
    }


def validate_and_assemble(header: dict, nodes: list[dict], tets: list[dict],
                          integrity: dict, environment: int) -> tuple[sparse.csr_matrix,
                                                                       sparse.csr_matrix,
                                                                       np.ndarray,
                                                                       np.ndarray,
                                                                       dict]:
    if header.get("schema") != CAPTURE_SCHEMA:
        raise ValueError("unsupported capture schema")
    if header.get("production_inputs_bitwise_unchanged") is not True:
        raise ValueError("capture header does not attest unchanged production inputs")
    cache = header.get("cached_algorithmic_tangent_capture")
    if not isinstance(cache, dict) or cache.get("policy_enabled") is not True or \
            cache.get("available") is not True or \
            cache.get("snapshot_bitwise_guard_included") is not True:
        raise ValueError("production tangent cache is absent or lacks bitwise snapshot guard")
    if integrity.get("production_inputs_bitwise_unchanged") is not True or \
            integrity.get("cached_tangent_snapshot_bitwise_guard_included") is not True:
        raise ValueError("native input/cache mutation guard did not pass")
    env_count = int(header["environment_count"])
    node_count = int(header["node_count_per_environment"])
    tet_count = int(header["tetrahedron_count_per_environment"])
    if not 0 <= environment < env_count:
        raise ValueError("selected environment is outside captured range")
    if node_count <= 0 or tet_count <= 0:
        raise ValueError("empty FEM topology")

    selected_nodes = [r for r in nodes if int(r["environment"]) == environment]
    selected_tets = [r for r in tets if int(r["environment"]) == environment]
    if len(selected_nodes) != node_count or len(selected_tets) != tet_count:
        raise ValueError("selected environment record count differs from capture header")
    node_by_id = {int(r["node_id"]): r for r in selected_nodes}
    tet_by_id = {int(r["tet_id"]): r for r in selected_tets}
    if len(node_by_id) != node_count or set(node_by_id) != set(range(node_count)):
        raise ValueError("node IDs are not unique dense zero-based indices")
    if len(tet_by_id) != tet_count or set(tet_by_id) != set(range(tet_count)):
        raise ValueError("tet IDs are not unique dense zero-based indices")

    objects = {int(t["object_index"]) for t in selected_tets}
    if len(objects) != 1:
        raise ValueError("operator capture contains multiple FEM object identities")
    if any(not bool(int(t["tet_flags"]) & 1) for t in selected_tets):
        raise ValueError("inactive tetrahedra are outside this complete-cache assembly scope")
    frame_modes = {float(t["center"]["dt_seconds_bulk_modulus_kind_frame_active"][3])
                   for t in selected_tets}
    if not frame_modes <= {0.0, 1.0}:
        raise ValueError("invalid material-frame active flags")
    if len(frame_modes) != 1:
        raise ValueError("one FEM object has inconsistent material-frame activation across tets")
    descriptors = {int(m["index"]): m for m in header.get("materials", [])}
    if not descriptors:
        raise ValueError("capture has no material descriptors")
    parameters_by_environment = header.get("parameters_by_environment")
    if not isinstance(parameters_by_environment, list) or len(parameters_by_environment) != env_count:
        raise ValueError("captured environment material parameters are incomplete")
    for environment_index, parameter_values in enumerate(parameters_by_environment):
        parameter_array = np.asarray(parameter_values, dtype=np.float64)
        if parameter_array.ndim != 1 or not np.isfinite(parameter_array).all():
            raise ValueError(f"environment {environment_index} parameters are invalid")
    for descriptor in descriptors.values():
        offset = int(descriptor.get("parameter_offset", -1))
        names = descriptor.get("parameter_names")
        if offset < 0 or not isinstance(names, list):
            raise ValueError("capture material descriptor has invalid parameter layout")
        for environment_index, values in enumerate(parameters_by_environment):
            if offset + len(names) > len(values):
                raise ValueError(f"material parameter range exceeds environment {environment_index} capture")

    masses = np.zeros(node_count, dtype=np.float64)
    constraints = np.zeros(node_count, dtype=np.int32)
    direction = np.zeros((node_count, 3), dtype=np.float64)
    native_action = np.zeros((node_count, 3), dtype=np.float64)
    center_residual = np.zeros((node_count, 3), dtype=np.float64)
    positions = np.zeros((node_count, 3), dtype=np.float64)
    captured_node_state_f32 = np.zeros((node_count, 20), dtype=np.float32)
    for node_id, row in node_by_id.items():
        center = row["center"]
        masses[node_id] = captured_f32_scalar(center["mass_kg"],
                                              f"node {node_id} mass")
        tag_value = captured_f32_scalar(center["constraint_tag"],
                                        f"node {node_id} constraint tag")
        if not tag_value.is_integer():
            raise ValueError(f"node {node_id} has nonintegral constraint tag")
        constraints[node_id] = int(tag_value)
        if constraints[node_id] not in (0, 1, 3):
            raise ValueError(f"node {node_id} has unsupported constraint tag {constraints[node_id]}")
        positions[node_id] = captured_f32_array(
            center["x_m"], (3,), f"node {node_id} position")
        center_residual[node_id] = captured_f32_array(
            row.get("center_residual_impulse_Ns"), (4,),
            f"node {node_id} center residual")[:3]
        raw_state = finite_array(center.get("raw_state_float4s"), (5, 4),
                                 f"node {node_id} captured state words")
        raw_state_f32 = raw_state.astype(np.float32)
        if not np.isfinite(raw_state_f32).all():
            raise ValueError(f"node {node_id} captured state words are not finite FP32")
        captured_node_state_f32[node_id] = raw_state_f32.reshape(20)
        direction[node_id] = captured_f32_array(
            row["correction_delta_v_m_s"], (3,),
            f"node {node_id} velocity correction")
        native_action[node_id] = captured_f32_array(
            row["production_A_delta_v_impulse_Ns"], (4,),
            f"node {node_id} native operator action")[:3]
        if not math.isfinite(masses[node_id]) or masses[node_id] <= 0.0:
            raise ValueError(f"node {node_id} has nonpositive/nonfinite mass")
    fixed = np.isin(constraints, (1, 3))
    max_fixed_direction = float(np.max(np.abs(direction[fixed]))) if fixed.any() else 0.0
    if max_fixed_direction > 2.0e-7:
        raise ValueError("captured correction has nonzero fixed/gripped node components")
    max_fixed_native_action = float(np.max(np.abs(native_action[fixed]))) if fixed.any() else 0.0

    free_nodes = np.flatnonzero(~fixed)
    if free_nodes.size == 0:
        raise ValueError("no free FEM node rows")
    reduced_map = np.full(node_count, -1, dtype=np.int64)
    reduced_map[free_nodes] = np.arange(free_nodes.size, dtype=np.int64)
    dof_map = np.full((node_count, 3), -1, dtype=np.int64)
    dof_map[free_nodes] = (3 * reduced_map[free_nodes, None] +
                           np.arange(3, dtype=np.int64)[None, :])
    ndof = int(3 * free_nodes.size)

    mass_diag = np.repeat(masses[free_nodes], 3)
    mass_matrix = sparse.diags(mass_diag, format="csr")
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    dt_values = []
    total_local_entries = 0
    volume_consistency = []
    material_indices = set()
    for tet_id, tet in tet_by_id.items():
        cache_row = tet.get("cached_algorithmic_tangent_local")
        if not isinstance(cache_row, dict) or cache_row.get("valid") is not True:
            raise ValueError(f"tet {tet_id} lacks a valid captured production tangent")
        tangent = captured_f32_array(
            cache_row.get("matrix_row_major_local_dP_dF_Pa_per_dimensionless_F"),
            (81,), f"tet {tet_id} cached tangent").reshape(9, 9)
        if not isinstance(tet.get("center"), dict):
            raise ValueError(f"tet {tet_id} lacks center material state")
        descriptor = descriptors.get(int(tet["material_index"]))
        if descriptor is None or descriptor.get("name") != tet.get("material_name"):
            raise ValueError(f"tet {tet_id} material identity does not match capture header")
        projection = finite_array(tet["center"].get("projection"), (4,),
                                  f"tet {tet_id} center projection diagnostics")
        terminal = finite_array(tet["center"].get("flags_terminal_status_parameter_count"), (4,),
                                f"tet {tet_id} terminal diagnostics")
        if projection[3] != 1.0 or terminal[1] != 0.0 or terminal[2] != 0.0:
            raise ValueError(f"tet {tet_id} captured center constitutive evaluation is invalid")
        meta = captured_f32_array(
            tet["center"]["dt_seconds_bulk_modulus_kind_frame_active"],
            (4,), f"tet {tet_id} captured material metadata")
        dt, _, _, frame_active_number = (float(v) for v in meta)
        if dt <= 0.0 or not math.isfinite(dt):
            raise ValueError(f"tet {tet_id} has invalid timestep")
        frame_active = frame_active_number == 1.0
        dt_values.append(dt)
        material_indices.add(int(tet["material_index"]))
        conn_values = finite_array(tet["nodes"], (4,), f"tet {tet_id} connectivity")
        if not np.equal(conn_values, np.floor(conn_values)).all():
            raise ValueError(f"tet {tet_id} connectivity is not integral")
        conn = conn_values.astype(np.int64)
        if len(set(conn.tolist())) != 4 or np.any(conn < 0) or np.any(conn >= node_count):
            raise ValueError(f"tet {tet_id} connectivity is repeated or out of range")
        inv = captured_f32_array(tet["inverse_rest_rows_per_m"], (3, 3),
                                  f"tet {tet_id} inverse-rest")
        volume = captured_f32_scalar(tet["rest_volume_m3"],
                                     f"tet {tet_id} rest volume")
        if not math.isfinite(volume) or volume <= 0.0:
            raise ValueError(f"tet {tet_id} has nonpositive volume")
        inv_det = float(np.linalg.det(inv))
        if not math.isfinite(inv_det) or abs(inv_det) <= 1.0e-30:
            raise ValueError(f"tet {tet_id} inverse-rest matrix is singular")
        geometric_volume = 1.0 / (6.0 * abs(inv_det))
        volume_rel = abs(geometric_volume - volume) / volume
        volume_consistency.append(volume_rel)
        if volume_rel > 2.0e-4:
            raise ValueError(f"tet {tet_id} volume disagrees with inverse-rest determinant")
        quaternion = captured_f32_array(
            tet["material_frame_quaternion_xyzw"], (4,),
            f"tet {tet_id} quaternion")
        k_local = local_tet_stiffness(inv, volume, tangent, quaternion, frame_active)
        total_local_entries += k_local.size
        local_dofs = np.array([dof_map[int(node), axis]
                               for node in conn for axis in range(3)], dtype=np.int64)
        for local_row, global_row in enumerate(local_dofs):
            if global_row < 0:
                continue
            for local_col, global_col in enumerate(local_dofs):
                if global_col < 0:
                    continue
                value = dt * dt * k_local[local_row, local_col]
                if value != 0.0:
                    rows.append(int(global_row))
                    cols.append(int(global_col))
                    data.append(float(value))
    if not dt_values:
        raise ValueError("no active tets to assemble")
    dt_ref = dt_values[0]
    max_dt_rel = max(abs(dt - dt_ref) / dt_ref for dt in dt_values)
    if any(dt != dt_ref for dt in dt_values):
        raise ValueError("one-object FEM capture has inconsistent per-tet timesteps")
    dt_header = header.get("dt_seconds_by_environment", [None] * env_count)[environment]
    if dt_header is None or abs(float(dt_header) - dt_ref) > 2.0e-6 * dt_ref:
        raise ValueError("header timestep differs from captured per-tet timestep")

    stiffness_scaled = sparse.coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()
    stiffness_scaled.sum_duplicates()
    operator = mass_matrix + stiffness_scaled
    if not np.isfinite(operator.data).all():
        raise ValueError("assembled FEM operator contains nonfinite entries")
    relative_skew = sparse.linalg.norm(operator - operator.T) / max(
        sparse.linalg.norm(operator), 1.0e-300)
    free_direction = direction[free_nodes].reshape(-1)
    native_free_action = native_action[free_nodes].reshape(-1)
    free_center_residual = center_residual[free_nodes].reshape(-1)
    assembled_action = np.asarray(operator @ free_direction).reshape(-1)
    difference = assembled_action - native_free_action
    action_scale = float(np.linalg.norm(native_free_action))
    relative_action_error = (float(np.linalg.norm(difference)) / action_scale
                             if action_scale > 0.0 else None)
    max_abs_action_error = float(np.max(np.abs(difference))) if difference.size else 0.0
    mass_action = mass_diag * free_direction
    stiffness_action = np.asarray(stiffness_scaled @ free_direction).reshape(-1)
    max_abs_action = float(np.max(np.abs(native_free_action))) if difference.size else 0.0
    parity = {
        "native_action_l2_Ns": action_scale,
        "assembled_action_l2_Ns": float(np.linalg.norm(assembled_action)),
        "difference_l2_Ns": float(np.linalg.norm(difference)),
        "relative_l2_error": relative_action_error,
        "max_abs_component_error_Ns": max_abs_action_error,
        "native_action_max_abs_component_Ns": max_abs_action,
        "fixed_rows_max_abs_native_action_Ns": max_fixed_native_action,
        "fixed_correction_max_abs_m_per_s": max_fixed_direction,
        "classification": "measured comparison; no noise-floor qualification",
    }
    metadata = {
        "environment": environment,
        "node_count": node_count,
        "tetrahedron_count": tet_count,
        "free_node_count": int(free_nodes.size),
        "free_dof_count": ndof,
        "fixed_node_count": int(fixed.sum()),
        "constraint_tag_counts": {str(tag): int(np.count_nonzero(constraints == tag))
                                  for tag in sorted(set(constraints.tolist()))},
        "one_object_indices": sorted(objects),
        "material_indices": sorted(material_indices),
        "dt_seconds": dt_ref,
        "per_tet_dt_max_relative_spread": max_dt_rel,
        "inverse_rest_volume_max_relative_mismatch": max(volume_consistency),
        "assembly": "A=M+dt^2*K; K=-d(f_internal)/d(v), local full unsymmetric dP/dF cache; fixed rows/columns eliminated",
        "matrix_layout": "global free-DOF CSR, node-major xyz, no symmetrization",
        "global_nnz": int(operator.nnz),
        "raw_local_stiffness_entries_visited": int(total_local_entries),
        "relative_skew_norm": float(relative_skew),
        "action_parity": parity,
        "mass_action_l2_Ns": float(np.linalg.norm(mass_action)),
        "dt2_stiffness_action_l2_Ns": float(np.linalg.norm(stiffness_action)),
        "action_component_arrays": {
            "native_Ns": native_free_action,
            "assembled_Ns": assembled_action,
            "difference_Ns": difference,
        },
        "free_nodes": free_nodes,
        "node_positions_m": positions,
        "native_rhs_free_impulse_Ns": free_center_residual,
        "captured_node_state_f32": captured_node_state_f32,
    }
    return mass_matrix, stiffness_scaled, operator, free_direction, metadata


def spectral_diagnostics(operator: sparse.csr_matrix, metadata: dict,
                         requested_modes: int = 8) -> dict:
    n = operator.shape[0]
    if n < 4:
        raise ValueError("spectral diagnostics require at least four free DOFs")
    k = min(requested_modes, n - 2)
    diagnostics: dict[str, Any] = {
        "method": "SciPy sparse iterative estimates; exact unsymmetric operator retained",
        "requested_small_modes": k,
        "operator_dimension": n,
        "condition_estimate_status": "inconclusive_until_singular_solver_converges",
    }
    try:
        smax = svds(operator, k=1, which="LM", return_singular_vectors=False,
                    tol=1e-8, maxiter=max(1000, 20*n))
        _, smin, vt = svds(operator, k=min(k, n-1), which="SM",
                           return_singular_vectors=True, tol=1e-8,
                           maxiter=max(5000, 50*n))
        order = np.argsort(smin)
        smin = smin[order]
        vt = vt[order]
        largest = float(np.max(smax))
        smallest = float(smin[0])
        diagnostics["singular_values_smallest_to_larger"] = [float(x) for x in smin]
        diagnostics["singular_value_largest_estimate"] = largest
        diagnostics["singular_value_smallest_estimate"] = smallest
        diagnostics["condition_2_estimate"] = largest / smallest if smallest > 0 else None
        diagnostics["condition_estimate_status"] = "iterative_estimate"
        free_nodes = metadata["free_nodes"]
        positions = metadata["node_positions_m"]
        local_modes = []
        for index, vector in enumerate(vt[:min(4, len(vt))]):
            node_norm = np.linalg.norm(vector.reshape(-1, 3), axis=1)
            weights = node_norm * node_norm
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0:
                continue
            weights /= weight_sum
            top = np.argsort(weights)[-min(10, len(weights)):][::-1]
            local_modes.append({
                "singular_value": float(smin[index]),
                "participation_ratio_nodes": float(1.0 / np.sum(weights * weights)),
                "top_free_node_ids_and_weights": [
                    {"node_id": int(free_nodes[i]), "weight": float(weights[i]),
                     "position_m": positions[int(free_nodes[i])].tolist()}
                    for i in top],
                "top_10_weight_fraction": float(np.sum(weights[top])),
            })
        diagnostics["small_right_singular_modes"] = local_modes
    except ArpackNoConvergence as error:
        diagnostics["condition_estimate_status"] = "inconclusive_ARPACK_no_convergence"
        diagnostics["partial_converged_singular_values"] = [float(x) for x in error.eigenvalues]
    except Exception as error:  # preserve sparse estimates when a solver cannot resolve them
        diagnostics["condition_estimate_status"] = "inconclusive_sparse_svd_error"
        diagnostics["error"] = f"{type(error).__name__}: {error}"

    # The symmetric part is reported separately as a diagnostic for coercivity;
    # it is never substituted for A in action, singular-value, or condition work.
    symmetric_part = (operator + operator.T) * 0.5
    try:
        eigenvalues = eigsh(symmetric_part, k=k, which="SA",
                            return_eigenvectors=False, tol=1e-8,
                            maxiter=max(1000, 20*n))
        diagnostics["symmetric_part_smallest_eigenvalues_diagnostic_only"] = \
            [float(x) for x in np.sort(eigenvalues)]
    except Exception as error:
        diagnostics["symmetric_part_eigenvalue_status"] = \
            f"inconclusive_{type(error).__name__}: {error}"
    return diagnostics


def run(args: argparse.Namespace) -> dict:
    capture = Path(args.capture)
    header, nodes, tets, integrity = capture_records(capture)
    binding = bind_inputs(capture, Path(args.prediction), Path(args.receipt),
                          Path(args.stderr), args.environment,
                          Path(args.source_snapshot) if args.source_snapshot else None)
    root = int(header.get("root", -1))
    iteration = int(header.get("newton_iteration", -1))
    selector = binding["prediction_selector"]
    if selector != f"{root}:{iteration}":
        raise ValueError("capture root:iteration differs from sealed prediction selector")
    diagnostic = binding["operator_fd_iteration"]
    try:
        world_match = int(header["world_fingerprint_hex"], 16) == int(diagnostic["world_fingerprint"])
        execution_match = int(header["execution_fingerprint_hex"], 16) == int(diagnostic["execution_fingerprint"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("capture or stderr lacks comparable world/execution fingerprints") from error
    if not world_match or not execution_match:
        raise ValueError("capture fingerprints differ from matching stderr diagnostic")
    binding["capture_run_fingerprints_match"] = True
    cache_guard = header["cached_algorithmic_tangent_capture"]
    if cache_guard.get("matrix_layout") != "row-major 9x9, local material frame" or \
            "dP_local/dF_local" not in cache_guard.get("derivative_contract", ""):
        raise ValueError("cached tangent layout or derivative contract is unsupported")
    mass, stiffness, operator, vector, meta = validate_and_assemble(
        header, nodes, tets, integrity, args.environment)
    action_vectors = meta.pop("action_component_arrays")
    native_rhs = meta.pop("native_rhs_free_impulse_Ns")
    captured_node_state = meta.pop("captured_node_state_f32")
    selected_tets = sorted((t for t in tets
                            if int(t["environment"]) == args.environment),
                           key=lambda row: int(row["tet_id"]))
    spectral = spectral_diagnostics(operator, meta, args.modes) if args.spectral else {
        "status": "not_requested"
    }
    report = {
        "schema": REPORT_SCHEMA,
        "verdict": "source-bound diagnostic operator reconstruction; no physical or solver qualification",
        "binding": binding,
        "capture_selection": {"root": root, "newton_iteration": iteration,
                              "environment": args.environment},
        "capture_integrity": integrity,
        "operator": {key: value for key, value in meta.items()
                     if key not in ("free_nodes", "node_positions_m")},
        "spectral_diagnostics": spectral,
        "limits": [
            "A is reconstructed from the captured production tangent cache and geometry; it is not a fresh production kernel execution.",
            "The capture scope excludes contact and other generalized unknowns; no contact operator or full coupled system is included.",
            "Sparse eigen/singular results are iterative estimates and can be inconclusive.",
            "Small mechanical actions are compared descriptively; no native noise floor is inferred from FP32 epsilon.",
            "The capture may be from a failed/rejected Newton attempt; its matrix is diagnostic at that selected candidate only.",
        ],
    }
    # Persist matrices/vectors for later preconditioner experiments. These are
    # snapshots of the reconstructed diagnostic operator, not native state.
    report["_offline_matrix_payload"] = {
        "mass": mass,
        "dt2_stiffness": stiffness,
        "dt_seconds": meta["dt_seconds"],
        "direction": vector,
        "native_action": action_vectors["native_Ns"],
        "assembled_action": action_vectors["assembled_Ns"],
        "action_difference": action_vectors["difference_Ns"],
        "native_rhs_free_impulse_Ns": native_rhs,
        "free_nodes": meta["free_nodes"],
        "node_positions_m": meta["node_positions_m"],
        "captured_node_state_f32": captured_node_state,
        "tet_ids": np.asarray([int(t["tet_id"]) for t in selected_tets], dtype=np.int64),
        "nodes": np.asarray([t["nodes"] for t in selected_tets], dtype=np.int64),
        "material_indices": np.asarray([int(t["material_index"]) for t in selected_tets], dtype=np.int64),
        "rest_volumes_m3": np.asarray([
            captured_f32_scalar(t["rest_volume_m3"], "tet rest volume")
            for t in selected_tets], dtype=np.float64),
        "inverse_rest_rows_per_m": np.asarray([
            captured_f32_array(t["inverse_rest_rows_per_m"], (3, 3),
                               "tet inverse-rest")
            for t in selected_tets], dtype=np.float64),
        "material_frame_quaternion_xyzw": np.asarray([
            captured_f32_array(t["material_frame_quaternion_xyzw"], (4,),
                               "tet frame quaternion")
            for t in selected_tets], dtype=np.float64),
        "material_frame_active": np.asarray([
            int(float(t["center"]["dt_seconds_bulk_modulus_kind_frame_active"][3]))
            for t in selected_tets], dtype=np.uint8),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--stderr", required=True)
    parser.add_argument("--source-snapshot")
    parser.add_argument("--environment", type=int, default=0)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--spectral", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    matrix_paths = [output.with_name(output.stem + suffix) for suffix in
                    (".mass.npz", ".stiffness.npz", ".dt2-stiffness.npz", ".operator.npz",
                     ".vectors.npz", ".topology.npz")]
    if output.exists() or any(path.exists() for path in matrix_paths):
        print(f"refusing to overwrite existing output: {output}", file=sys.stderr)
        return 2
    if args.modes < 2 or args.modes > 32:
        parser.error("--modes must be in [2, 32]")
    try:
        report = run(args)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = report.pop("_offline_matrix_payload")
        raw_stiffness = payload["dt2_stiffness"] / (payload["dt_seconds"] ** 2)
        matrix_objects = (payload["mass"], raw_stiffness,
                          payload["dt2_stiffness"],
                          payload["mass"] + payload["dt2_stiffness"])
        matrix_labels = ("mass", "stiffness", "dt2-stiffness", "operator")
        matrix_hashes = {}
        for label, matrix in zip(matrix_labels, matrix_objects):
            target = output.with_name(output.stem + f".{label}.npz")
            sparse.save_npz(target, matrix)
            matrix_hashes[target.name] = sha256(target)
        vectors_path = output.with_name(output.stem + ".vectors.npz")
        np.savez_compressed(vectors_path,
            direction=payload["direction"], native_action_Ns=payload["native_action"],
            assembled_action_Ns=payload["assembled_action"],
            action_difference_Ns=payload["action_difference"],
            native_rhs_free_impulse_Ns=payload["native_rhs_free_impulse_Ns"],
            free_node_ids=payload["free_nodes"], node_positions_m=payload["node_positions_m"],
            captured_node_state_f32=payload["captured_node_state_f32"])
        matrix_hashes[vectors_path.name] = sha256(vectors_path)
        topology_path = output.with_name(output.stem + ".topology.npz")
        np.savez_compressed(topology_path, nodes=payload["nodes"],
            material_indices=payload["material_indices"],
            rest_volumes_m3=payload["rest_volumes_m3"],
            inverse_rest_rows_per_m=payload["inverse_rest_rows_per_m"],
            node_count=np.asarray(payload["captured_node_state_f32"].shape[0], dtype=np.int64))
        matrix_hashes[topology_path.name] = sha256(topology_path)
        report["matrix_artifacts"] = {
            "files_sha256": matrix_hashes,
            "relation": "A = mass + dt2-stiffness; stiffness = dt2-stiffness / dt_seconds^2",
            "coordinate_order": "free-node ID ascending, xyz within node",
        }
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except Exception as error:
        print(f"cached operator audit rejected input: {type(error).__name__}: {error}",
              file=sys.stderr)
        return 2
    print(json.dumps({"schema": REPORT_SCHEMA, "output": str(output),
                      "capture_sha256": report["binding"]["capture_sha256"],
                      "action_relative_l2_error": report["operator"]["action_parity"]["relative_l2_error"],
                      "spectral_status": report["spectral_diagnostics"].get("condition_estimate_status",
                                                                            report["spectral_diagnostics"].get("status"))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
