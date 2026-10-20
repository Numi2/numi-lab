#!/usr/bin/env python3
"""Source-bound pair-aware FP64 analysis of a rich native FEM capture.

This tool replays the captured 13-state orthotropic Hill paper points through
the independent CPU reference. It preserves FP32 high/low words and validates
the sealed native material policy. The default requires initializer, compensated
arithmetic, accurate projection and scaled residuals. Explicit ordinary-generic
reference mode records their absence and excludes residual-norm parity. Reports are diagnostic;
the tool has no calibrated-noise gate and cannot qualify physical cardboard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "numi.cardboard.solver-pair-analysis.v1"
CAPTURE_SCHEMA = "numi.cardboard.fem-newton-capture.v1"
PAPER_PARAMETER_NAMES = (
    "density", "c11", "c22", "c33", "c12", "c13", "c23", "g12", "g13",
    "g23", "sigma_ref", "hill_F", "hill_G", "hill_H", "hill_L", "hill_M",
    "hill_N", "q_floor", "fb_epsilon", "residual_scale",
)
MATERIAL_KIND_BY_NAME = {
    "hajali2009_liner_hill_ideal": "liner",
    "hajali2009_medium_hill_ideal": "medium",
}
REQUIRED_TRANSFORMS = (
    "--safeguarded-paper-initializer",
    "--compensated-paper-arithmetic",
    "--accurate-paper-projection",
    "--scaled-paper-residuals",
)
REQUIRED_SOURCE_BINDINGS = (
    "matter/materials/hajali2009_liner_hill_ideal.nmatter",
    "matter/materials/hajali2009_medium_hill_ideal.nmatter",
    "matter/tools/cardboard_probe.mm",
    "matter/tools/cardboard_paper_projection_scaling.hpp",
    "matter/include/numi/matter/paper_compensated_bytecode.h",
    "matter/include/numi/matter/compiler.hpp",
    "matter/src/compiler_impl.cpp",
    "matter/src/constitutive_impl.cpp",
    "matter/src/metal/common.metalinc",
    "matter/src/metal/fem.metalinc",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            block = source.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def f32(value: Any, label: str = "value") -> float:
    try:
        raw = float(value)
        rounded = struct.unpack("<f", struct.pack("<f", raw))[0]
    except (TypeError, ValueError, OverflowError, struct.error) as error:
        raise ValueError(f"{label} is not a finite FP32 value") from error
    if not math.isfinite(raw) or not math.isfinite(rounded):
        raise ValueError(f"{label} is not a finite FP32 value")
    return rounded


def f32_array(value: Any, size: int, label: str) -> list[float]:
    if size == 9 and isinstance(value, list) and len(value) == 3 and all(
            isinstance(row, list) and len(row) == 3 for row in value):
        value = [entry for row in value for entry in row]
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain exactly {size} values")
    return [f32(item, label) for item in value]


def pair_words(state: dict[str, Any], high_key: str, low_key: str,
               size: int, label: str) -> tuple[list[float], list[float]]:
    if high_key not in state or low_key not in state:
        raise ValueError(f"{label} is missing its captured FP32 high or low word")
    high = f32_array(state[high_key], size, f"{label} high")
    low = f32_array(state[low_key], size, f"{label} low")
    return high, low


def pair_sum(high: list[float], low: list[float]) -> list[float]:
    # Each word is independently widened to Python's binary64 before summing.
    return [float(a) + float(b) for a, b in zip(high, low)]


def vector_dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def vector_sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def vector_scale(a: list[float], scale: float) -> list[float]:
    return [x * scale for x in a]


def metric(actual: Any, expected: Any) -> dict[str, Any]:
    if not isinstance(actual, list) or not isinstance(expected, list) or len(actual) != len(expected):
        raise ValueError("comparison vectors have different shapes")
    actual_values = [float(x) for x in actual]
    expected_values = [float(x) for x in expected]
    if not all(math.isfinite(x) for x in actual_values + expected_values):
        raise ValueError("comparison vectors contain nonfinite values")
    abs_error = max((abs(a - b) for a, b in zip(actual_values, expected_values)), default=0.0)
    reference_scale = max((abs(x) for x in expected_values), default=0.0)
    resolved = reference_scale > 0.0
    return {
        "absolute_error": abs_error,
        "reference_scale": reference_scale,
        "relative_error": abs_error / reference_scale if resolved else None,
        "nonzero_reference": resolved,
        "verdict": "descriptive_only_no_measured_noise_floor",
    }


def classify_trial_q(value: float, branch_band: float) -> str:
    if not math.isfinite(value):
        return "invalid"
    if abs(value - 1.0) <= branch_band:
        return "near_yield_unresolved"
    return "plastic_trial" if value > 1.0 else "elastic_trial"


def classify_candidate_q(value: float, branch_band: float) -> str:
    if not math.isfinite(value):
        return "invalid"
    if abs(value - 1.0) <= branch_band:
        return "on_yield_surface_within_diagnostic_band"
    return "inside_yield_surface" if value < 1.0 else "outside_yield_surface"


def validate_run_binding(capture_path: Path, header: dict[str, Any],
                         prediction: dict[str, Any], receipt: dict[str, Any],
                         ordinary_generic_reference: bool = False) -> dict[str, Any]:
    if prediction.get("schema") != "numi.cardboard.native-prediction.v1":
        raise ValueError("prediction file has an unsupported schema")
    if prediction.get("executed_build_verified") is not True:
        raise ValueError("prediction does not attest a verified native build")
    if not isinstance(prediction.get("inputs_sha256"), dict) or not prediction["inputs_sha256"]:
        raise ValueError("prediction has no retained input hash map")
    if not prediction.get("source_snapshot"):
        raise ValueError("prediction has no source snapshot binding")
    source_snapshot = Path(prediction["source_snapshot"])
    if not source_snapshot.is_dir():
        raise ValueError("prediction source snapshot directory is missing")
    input_hashes = prediction["inputs_sha256"]
    bound_sources: dict[str, str] = {}
    for relative in REQUIRED_SOURCE_BINDINGS:
        key = f"source/{relative}"
        expected = input_hashes.get(key)
        path = source_snapshot / relative
        if not isinstance(expected, str) or len(expected) != 64 or not path.is_file():
            raise ValueError(f"prediction omits retained source binding {key}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"retained source hash mismatch for {key}")
        bound_sources[key] = actual
    argv = prediction.get("argv")
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        raise ValueError("prediction argv is missing or malformed")
    missing = [flag for flag in REQUIRED_TRANSFORMS if argv.count(flag) != 1]
    if ordinary_generic_reference and any(argv.count(flag) for flag in REQUIRED_TRANSFORMS):
        raise ValueError("ordinary generic reference requires all four optional builder transforms absent")
    if missing and not ordinary_generic_reference:
        raise ValueError("native prediction must contain each required builder transform exactly once: " +
                         ", ".join(missing))
    if receipt.get("schema") != "numi.cardboard.native-run-receipt.v2":
        raise ValueError("receipt file has an unsupported schema")
    if receipt.get("execution_started") is not True or receipt.get("source_inputs_unchanged") is not True:
        raise ValueError("receipt does not attest an executed run with unchanged inputs")
    environment = prediction.get("environment")
    if not isinstance(environment, dict):
        raise ValueError("prediction environment binding is absent")
    selector = environment.get("NM_FEM_OPERATOR_FD_SELECTOR")
    expected_selector = f"{int(header.get('root', -1))}:{int(header.get('newton_iteration', -1))}"
    if selector != expected_selector:
        raise ValueError(f"capture root/iteration {expected_selector} differs from prediction selector {selector!r}")
    selected_path = environment.get("NM_FEM_OPERATOR_FD_CAPTURE_PATH")
    if not selected_path or Path(selected_path).resolve() != capture_path.resolve():
        raise ValueError("prediction capture path does not identify the supplied capture")
    return {
        "prediction_schema": prediction["schema"],
        "source_snapshot": str(prediction["source_snapshot"]),
        "native_command": argv,
        "required_transforms_present": [] if ordinary_generic_reference else list(REQUIRED_TRANSFORMS),
        "native_policy": "ordinary_generic_FP32" if ordinary_generic_reference else "compensated_scaled_paper",
        "normalized_source_vm_matches_native_residual_policy": not ordinary_generic_reference,
        "input_hash_count": len(prediction["inputs_sha256"]),
        "receipt_exit_code": receipt.get("exit_code"),
        "receipt_timed_out": receipt.get("timed_out"),
        "receipt_wall_seconds": receipt.get("wall_seconds"),
        "capture_selector": selector,
        "verified_source_hashes": bound_sources,
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            result.append(record)
    return result


def validate_capture(records: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    headers = [record for record in records if record.get("record") == "header"]
    integrity = [record for record in records if record.get("record") == "integrity"]
    if len(headers) != 1 or len(integrity) != 1:
        raise ValueError("capture requires exactly one header and one terminal integrity record")
    header = headers[0]
    terminal = integrity[0]
    if header.get("schema") not in {CAPTURE_SCHEMA, "numi.cardboard.fem-newton-capture.v2"}:
        raise ValueError("unsupported FEM capture schema")
    if terminal.get("production_inputs_bitwise_unchanged") is not True:
        raise ValueError("capture integrity reports modified production inputs")
    if header.get("production_inputs_bitwise_unchanged") is not True:
        raise ValueError("capture header does not attest unchanged production inputs")
    word_contract = str(header.get("F_local_word_encoding", "")).lower()
    if "high_float32" not in word_contract or "low_float32" not in word_contract:
        raise ValueError("capture does not declare preserved compensated F high/low words")
    tets = [record for record in records if record.get("record") == "tetrahedron"]
    expected = int(header.get("tetrahedron_count_per_environment", 0)) * int(header.get("environment_count", 0))
    if expected <= 0 or len(tets) != expected:
        raise ValueError(f"incomplete tetrahedron capture: found {len(tets)}, expected {expected}")
    identities = [(record.get("environment"), record.get("tet_id")) for record in tets]
    if len(set(identities)) != expected:
        raise ValueError("capture has duplicate tetrahedron identities")
    return header, tets


def material_descriptors(header: dict[str, Any]) -> dict[int, dict[str, Any]]:
    descriptors: dict[int, dict[str, Any]] = {}
    used_ranges: list[tuple[int, int, int]] = []
    tables = header.get("parameters_by_environment")
    environment_count = int(header.get("environment_count", 0))
    if not isinstance(tables, list) or len(tables) != environment_count:
        raise ValueError("captured parameter table count differs from environment count")
    if not tables or not all(isinstance(table, list) and len(table) == len(tables[0]) for table in tables):
        raise ValueError("environment parameter tables have inconsistent lengths")
    for material in header.get("materials", []):
        if not isinstance(material, dict):
            raise ValueError("material descriptor is malformed")
        index = int(material.get("index", -1))
        offset = int(material.get("parameter_offset", -1))
        names = material.get("parameter_names")
        if index < 0 or index in descriptors or offset < 0 or not isinstance(names, list):
            raise ValueError("material descriptor index, offset, or parameter names are invalid")
        if len(set(names)) != len(names) or not all(isinstance(name, str) for name in names):
            raise ValueError("material parameter names must be unique strings")
        end = offset + len(names)
        if end > len(tables[0]):
            raise ValueError("material parameter range exceeds captured environment table")
        for old_begin, old_end, old_index in used_ranges:
            if max(offset, old_begin) < min(end, old_end):
                raise ValueError(f"material parameter ranges overlap for indices {old_index} and {index}")
        used_ranges.append((offset, end, index))
        descriptors[index] = material
    if not descriptors:
        raise ValueError("capture has no material descriptors")
    return descriptors


def paper_parameter_row(header: dict[str, Any], descriptor: dict[str, Any], environment: int) -> list[float]:
    name = descriptor.get("name")
    if name not in MATERIAL_KIND_BY_NAME:
        raise ValueError(f"unsupported captured paper program identity {name!r}")
    names = descriptor.get("parameter_names")
    if tuple(names) != PAPER_PARAMETER_NAMES:
        raise ValueError(f"captured {name} parameter names/order differ from the supported source-law schema")
    offset = int(descriptor["parameter_offset"])
    values = header["parameters_by_environment"][environment][offset:offset + len(PAPER_PARAMETER_NAMES)]
    if len(values) != len(PAPER_PARAMETER_NAMES):
        raise ValueError(f"captured {name} parameter slice is incomplete")
    return [f32(value, f"{name}.{parameter}") for parameter, value in zip(PAPER_PARAMETER_NAMES, values)]


def make_replay_rows(header: dict[str, Any], tets: list[dict[str, Any]], branch_band: float):
    descriptors = material_descriptors(header)
    environment_count = int(header["environment_count"])
    dt_by_environment = header.get("dt_seconds_by_environment")
    if not isinstance(dt_by_environment, list) or len(dt_by_environment) != environment_count:
        raise ValueError("capture timestep table is incomplete")
    rows: list[str] = []
    sources: dict[str, tuple[dict[str, Any], dict[str, Any], int]] = {}
    metadata_by_material: dict[str, dict[str, Any]] = {}
    for tet in tets:
        material_index = int(tet.get("material_index", -1))
        descriptor = descriptors.get(material_index)
        if descriptor is None or descriptor.get("name") not in MATERIAL_KIND_BY_NAME:
            continue
        if int(tet.get("material_state_count", -1)) != 13 or int(tet.get("material_parameter_count", -1)) != 20:
            raise ValueError("captured supported paper tet does not have the exact 13-state/20-parameter layout")
        environment = int(tet.get("environment", -1))
        if not 0 <= environment < environment_count:
            raise ValueError("tetrahedron environment is out of range")
        kind = MATERIAL_KIND_BY_NAME[descriptor["name"]]
        parameters = paper_parameter_row(header, descriptor, environment)
        state_records = [tet.get("center")] + tet.get("sides", [])
        if not isinstance(state_records[0], dict) or len(state_records) != 5:
            raise ValueError("selected capture requires a center and exactly four side states per paper tet")
        accepted = f32_array(tet.get("center", {}).get("accepted_state"), 13, "accepted state")
        for side_index, state in enumerate(state_records):
            if not isinstance(state, dict):
                raise ValueError("capture side state is malformed")
            f_high, f_low = pair_words(state, "F_local", "F_local_low", 9, "local F")
            candidate_high, candidate_low = pair_words(
                state, "projected_state", "projected_state_low", 13, "projected candidate state")
            direction = f32_array(tet["center"].get("F_direction_local_per_s"), 9,
                                  "local deformation direction")
            native_p = f32_array(state.get("P_local_Pa"), 9, "native local Piola stress")
            row_key = f"e{environment}_t{int(tet['tet_id'])}_s{side_index}"
            packed = [*f_high, *f_low, *direction, *accepted, *candidate_high, *candidate_low, *parameters]
            rows.append(" ".join([row_key, kind] + [format(value, ".17g") for value in packed]))
            sources[row_key] = (tet, state, side_index)
            if side_index == 0:
                projection = state.get("projection")
                if not isinstance(projection, list) or len(projection) != 4:
                    raise ValueError("center projection diagnostics are absent or malformed")
                metadata_by_material[row_key] = {
                    "native_projection_diagnostics": [f32(x, "projection diagnostic") for x in projection],
                    "dt_seconds": f32(dt_by_environment[environment], "timestep"),
                    "direction_local_per_s": direction,
                    "F_center": pair_sum(f_high, f_low),
                    "native_P_center_Pa": native_p,
                    "native_tangent_center_Pa_per_s": f32_array(
                        state.get("tangent_local_Pa_per_s"), 9, "native local tangent"),
                    "branch_band": branch_band,
                    "native_material_index": material_index,
                    "native_material_name": descriptor["name"],
                    "captured_parameter_names": descriptor["parameter_names"],
                    "captured_parameters_FP32": parameters,
                    "native_hill_indicators": f32_array(
                        state.get("hill_q_lambda_a_b"), 4, "native candidate Hill diagnostics"),
                }
    if not rows:
        raise ValueError("capture contains no supported 13-state ideal-Hill paper tets")
    return rows, sources, metadata_by_material


def parse_replay_output(text: str, expected_keys: list[str]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    outputs: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"line {line_number}: invalid replay JSON: {error}")
            continue
        key = row.get("key")
        if not isinstance(key, str) or key in outputs:
            errors.append(f"line {line_number}: missing or duplicate replay key")
            continue
        outputs[key] = row
    if list(outputs) != expected_keys:
        errors.append("replay output keys/order differ from requested exact input rows")
    return outputs, errors


def finite_summary(values: list[Any]) -> dict[str, Any]:
    finite = sorted(float(value) for value in values
                    if isinstance(value, (int, float)) and math.isfinite(float(value)))
    if not finite:
        return {"finite_count": 0, "total_count": len(values), "statistics": None}
    return {
        "finite_count": len(finite),
        "total_count": len(values),
        "statistics": {
            "min": finite[0],
            "median": statistics.median(finite),
            "p95": finite[min(len(finite) - 1, int(0.95 * (len(finite) - 1)))],
            "max": finite[-1],
        },
    }


def analysis_exit_code(summary: dict[str, Any]) -> int:
    """Keep successful replay mechanics distinct from strict-gate admission."""
    replay = summary.get("replay", {})
    if (int(replay.get("invalid_rows", 0)) > 0 or replay.get("output_errors") or
            int(replay.get("returncode", 1)) != 0):
        return 4
    if summary.get("strict_gate") != "qualified":
        return 3
    return 0


def full_history_complete(binding: dict[str, Any]) -> bool:
    return binding.get("receipt_timed_out") is not True and binding.get("receipt_exit_code") == 0


def make_tet_audit(tets: list[dict[str, Any]], sources: dict[str, tuple[dict[str, Any], dict[str, Any], int]],
                   replay: dict[str, dict[str, Any]], metadata: dict[str, dict[str, Any]], branch_band: float):
    paper_audits: list[dict[str, Any]] = []
    by_tet: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    for key, (tet, state, side_index) in sources.items():
        by_tet.setdefault((int(tet["environment"]), int(tet["tet_id"])), {})[side_index] = {
            "native": state,
            "oracle": replay.get(key),
            "key": key,
        }
    for (environment, tet_id), items in sorted(by_tet.items()):
        center_item = items[0]
        center_state = center_item["native"]
        center_oracle = center_item["oracle"]
        base = {
            "environment": environment,
            "tet_id": tet_id,
            "physical_validation": False,
            "gate_verdict": "inconclusive_no_measured_noise_floor",
        }
        if center_oracle is None or center_oracle.get("valid") is not True:
            base["reprojected_reference_status"] = "failed_or_missing"
            base["replay_failure"] = center_oracle
            paper_audits.append(base)
            continue
        detail = metadata[center_item["key"]]
        base["native_material_index"] = detail["native_material_index"]
        base["native_material_name"] = detail["native_material_name"]
        base["captured_parameter_names"] = detail["captured_parameter_names"]
        base["captured_parameters_FP32"] = detail["captured_parameters_FP32"]
        base["native_projection_diagnostics"] = detail["native_projection_diagnostics"]
        base["native_projection_reported_residual"] = detail["native_projection_diagnostics"][0]
        base["native_projection_state_norm"] = detail["native_projection_diagnostics"][1]
        base["native_projection_iterations"] = detail["native_projection_diagnostics"][2]
        base["captured_candidate_yield_surface"] = {
            "native_hill_q": detail["native_hill_indicators"][0],
            "oracle_hill_q": center_oracle["candidate_q"],
            "classification": classify_candidate_q(center_oracle["candidate_q"], branch_band),
            "absolute_difference_native_vs_oracle": abs(
                detail["native_hill_indicators"][0] - center_oracle["candidate_q"]),
            "diagnostic_band": branch_band,
            "physical_tolerance": False,
        }
        base["candidate_state_vs_reprojected_reference"] = metric(
            pair_sum(*pair_words(center_state, "projected_state", "projected_state_low", 13,
                                 "center candidate state")), center_oracle["oracle_reprojected_state"])
        base["native_stress_vs_oracle_at_captured_candidate"] = metric(
            f32_array(center_state["P_local_Pa"], 9, "native center stress"),
            center_oracle["oracle_candidate_P_Pa"])
        base["native_tangent_vs_oracle_at_captured_candidate"] = metric(
            detail["native_tangent_center_Pa_per_s"],
            center_oracle["oracle_tangent_at_captured_candidate_Pa_per_s"])
        base["native_tangent_vs_oracle_at_reprojected_reference"] = metric(
            detail["native_tangent_center_Pa_per_s"],
            center_oracle["oracle_tangent_at_reprojected_state_Pa_per_s"])
        base["independent_source_residual_at_captured_candidate"] = {
            "unscaled_inf": center_oracle["candidate_residual_unscaled_inf"],
            "scaled_inf": center_oracle["candidate_residual_scaled_inf"],
            "scaled_rows": center_oracle["candidate_residual_scaled_rows"],
            "transformed_source_vm_scaled_inf": center_oracle[
                "source_vm_compensated_scaled_residual_inf"],
            "transformed_source_vm_scaled_rows": center_oracle[
                "source_vm_compensated_scaled_residual_rows"],
            "native_reported_residual": detail["native_projection_diagnostics"][0],
            "difference_from_native_reported_residual": abs(
                center_oracle["candidate_residual_scaled_inf"] -
                detail["native_projection_diagnostics"][0]),
            "source_vm_difference_from_native_reported_residual": abs(
                center_oracle["source_vm_compensated_scaled_residual_inf"] -
                detail["native_projection_diagnostics"][0]),
            "interpretation": "FP64 reference and exact transformed-source bytecode are both reported; residual diagnostic parity has no universal qualification threshold",
        }
        center_q = center_oracle["elastic_trial_q"]
        base["trial_yield"] = {
            "q_center": center_q,
            "classification_center": classify_trial_q(center_q, branch_band),
            "diagnostic_band": branch_band,
            "band_is_physical_tolerance": False,
        }
        dt = detail["dt_seconds"]
        direction = detail["direction_local_per_s"]
        direction_norm2 = vector_dot(direction, direction)
        side_records = []
        by_epsilon: dict[float, dict[str, Any]] = {}
        for side_index in range(1, 5):
            item = items.get(side_index)
            if item is None or item["oracle"] is None or item["oracle"].get("valid") is not True:
                side_records.append({"side_index": side_index, "status": "missing_oracle_replay"})
                continue
            native_state = item["native"]
            oracle_state = item["oracle"]
            f_high, f_low = pair_words(native_state, "F_local", "F_local_low", 9, "side F")
            f_side = pair_sum(f_high, f_low)
            f_delta = vector_sub(f_side, detail["F_center"])
            epsilon = f32(native_state.get("epsilon"), "side epsilon")
            alpha = vector_dot(f_delta, direction) / direction_norm2 if direction_norm2 > 0 else 0.0
            sign = "plus" if alpha > 0 else "minus" if alpha < 0 else "zero_or_unresolved"
            branch = classify_trial_q(oracle_state["elastic_trial_q"], branch_band)
            by_epsilon.setdefault(epsilon, {})[sign] = (native_state, oracle_state, alpha)
            side_records.append({
                "side_index": side_index,
                "epsilon": epsilon,
                "realized_sign": sign,
                "realized_scalar_step_s": alpha,
                "nominal_scalar_step_s": (epsilon * dt) * (1.0 if sign == "plus" else -1.0 if sign == "minus" else 0.0),
                "trial_q": oracle_state["elastic_trial_q"],
                "trial_classification": branch,
                "native_P_vs_oracle_candidate_P": metric(
                    f32_array(native_state["P_local_Pa"], 9, "native side stress"),
                    oracle_state["oracle_candidate_P_Pa"]),
                "candidate_residual_scaled_inf": oracle_state["candidate_residual_scaled_inf"],
                "source_vm_compensated_scaled_residual_inf": oracle_state[
                    "source_vm_compensated_scaled_residual_inf"],
                "source_vm_candidate_residual_vs_native_projection": {
                    "native_reported_residual": f32(
                        native_state.get("projection", native_state.get("projection_diagnostics"))[0],
                        "native reported residual"),
                    "source_vm_scaled_inf": oracle_state["source_vm_compensated_scaled_residual_inf"],
                    "absolute_difference": abs(
                        oracle_state["source_vm_compensated_scaled_residual_inf"] -
                        f32(native_state.get("projection", native_state.get("projection_diagnostics"))[0],
                            "native reported residual")),
                    "qualification": "source-expression replay diagnostic; no universal pass tolerance",
                },
            })
        base["side_states"] = side_records
        base["finite_difference_pairs"] = []
        for epsilon, sides in sorted(by_epsilon.items()):
            pair = {"epsilon": epsilon, "diagnostic_only": True,
                    "noise_floor_measured": False,
                    "strict_gate": "inconclusive_no_measured_noise_floor"}
            if "plus" not in sides or "minus" not in sides:
                pair["status"] = "incomplete_or_unresolved_pair"
                pair["unresolved_reason"] = (
                    "captured local deformation direction is zero" if direction_norm2 == 0.0 else
                    "captured plus/minus records do not form a signed pair")
                base["finite_difference_pairs"].append(pair)
                continue
            plus_native, plus_oracle, alpha_plus = sides["plus"]
            minus_native, minus_oracle, alpha_minus = sides["minus"]
            span = alpha_plus - alpha_minus
            q_classes = [classify_trial_q(x[1]["elastic_trial_q"], branch_band)
                         for x in (sides["minus"], (center_state, center_oracle, 0.0), sides["plus"])]
            near_yield = "near_yield_unresolved" in q_classes
            branch_crossing = not near_yield and len(set(q_classes)) > 1
            pair["status"] = "pair_resolved" if span > 0 else "unresolved_realized_direction"
            if direction_norm2 == 0.0:
                pair["unresolved_reason"] = "captured local deformation direction is zero"
            elif span <= 0.0:
                pair["unresolved_reason"] = "plus/minus pair does not span the captured direction"
            pair["trial_q_classes_minus_center_plus"] = q_classes
            pair["near_yield_unresolved"] = near_yield
            pair["branch_crossing"] = branch_crossing
            pair["smooth_branch_comparison_eligible"] = bool(
                span > 0 and not near_yield and not branch_crossing and direction_norm2 > 0)
            if span > 0:
                native_plus_p = f32_array(plus_native["P_local_Pa"], 9, "native plus stress")
                native_minus_p = f32_array(minus_native["P_local_Pa"], 9, "native minus stress")
                oracle_plus_p = plus_oracle["oracle_reprojected_P_Pa"]
                oracle_minus_p = minus_oracle["oracle_reprojected_P_Pa"]
                native_fd = vector_scale(vector_sub(native_plus_p, native_minus_p), 1.0 / span)
                oracle_fd = vector_scale(vector_sub(oracle_plus_p, oracle_minus_p), 1.0 / span)
                pair["native_centered_FD_vs_native_tangent_at_captured_candidate"] = metric(
                    native_fd, detail["native_tangent_center_Pa_per_s"])
                pair["native_centered_FD_vs_oracle_tangent_at_captured_candidate"] = metric(
                    native_fd, center_oracle["oracle_tangent_at_captured_candidate_Pa_per_s"])
                pair["oracle_reprojected_centered_FD_vs_oracle_tangent_at_reprojected_reference"] = metric(
                    oracle_fd, center_oracle["oracle_tangent_at_reprojected_state_Pa_per_s"])
                realized_direction = vector_scale(vector_sub(
                    pair_sum(*pair_words(plus_native, "F_local", "F_local_low", 9, "plus F")),
                    pair_sum(*pair_words(minus_native, "F_local", "F_local_low", 9, "minus F"))),
                    1.0 / span)
                pair["realized_direction_vs_captured_direction"] = metric(realized_direction, direction)
            pair["strict_gate"] = "inconclusive_no_measured_noise_floor"
            base["finite_difference_pairs"].append(pair)
        paper_audits.append(base)
    return paper_audits


def analyze(capture_path: Path, prediction_path: Path, receipt_path: Path,
            replay_path: Path, output_path: Path, branch_band: float,
            reference_derivative_sweep: bool = False,
            ordinary_generic_reference: bool = False) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if not replay_path.is_file() or not os.access(replay_path, os.X_OK):
        raise ValueError("replay executable must exist and be executable")
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    records = read_jsonl(capture_path)
    header, tets = validate_capture(records)
    binding = validate_run_binding(capture_path, header, prediction, receipt,
                                   ordinary_generic_reference)
    rows, sources, metadata = make_replay_rows(header, tets, branch_band)
    output_path.mkdir(parents=True)
    input_path = output_path / "replay-input.tsv"
    replay_output_path = output_path / "replay-output.jsonl"
    replay_stderr_path = output_path / "replay-stderr.txt"
    input_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    snapshot = Path(prediction["source_snapshot"])
    liner_source = snapshot / "matter/materials/hajali2009_liner_hill_ideal.nmatter"
    medium_source = snapshot / "matter/materials/hajali2009_medium_hill_ideal.nmatter"
    replay_command = [str(replay_path.resolve()), str(liner_source), str(medium_source)]
    if reference_derivative_sweep:
        replay_command.append("--derivative-sweep")
    process = subprocess.run(replay_command,
                             input="\n".join(rows) + "\n",
                             text=True, capture_output=True, check=False)
    replay_output_path.write_text(process.stdout, encoding="utf-8")
    replay_stderr_path.write_text(process.stderr, encoding="utf-8")
    expected_keys = [row.split(" ", 1)[0] for row in rows]
    replay_rows, output_errors = parse_replay_output(process.stdout, expected_keys)
    if process.returncode not in (0, 2):
        raise RuntimeError(f"independent replay infrastructure failed with {process.returncode}: {process.stderr[-2000:]}")
    audits = make_tet_audit(tets, sources, replay_rows, metadata, branch_band)
    if ordinary_generic_reference:
        # The independent reference retains the same constitutive zero-set,
        # but its normalized compensated VM is not the native FP32 residual
        # instrument. Never compare differently scaled residual norms as parity.
        for audit in audits:
            residual = audit.get("independent_source_residual_at_captured_candidate", {})
            residual["difference_from_native_reported_residual"] = None
            residual["source_vm_difference_from_native_reported_residual"] = None
            residual["native_residual_comparison"] = "unavailable_different_arithmetic_and_row_scaling"
    (output_path / "tetrahedron-audit.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n" for row in audits),
        encoding="utf-8")
    invalid_replay_rows = sum(1 for value in replay_rows.values() if value.get("valid") is not True)
    source_residual_differences = []
    for key, (tet, state, side_index) in sources.items():
        if side_index != 0 or key not in replay_rows or replay_rows[key].get("valid") is not True:
            continue
        projection = state.get("projection")
        if not ordinary_generic_reference and isinstance(projection, list) and len(projection) == 4:
            source_residual_differences.append(abs(
                replay_rows[key]["source_vm_compensated_scaled_residual_inf"] -
                f32(projection[0], "native reported residual")))
    environment_summaries = []
    for environment in range(int(header["environment_count"])):
        group = [row for row in audits if row.get("environment") == environment]
        counts: dict[str, int] = {}
        for row in group:
            label = row.get("trial_yield", {}).get("classification_center", "missing")
            counts[label] = counts.get(label, 0) + 1
        candidate_stress_errors = [
            row.get("native_stress_vs_oracle_at_captured_candidate", {}).get("relative_error")
            for row in group]
        candidate_tangent_errors = [
            row.get("native_tangent_vs_oracle_at_captured_candidate", {}).get("relative_error")
            for row in group]
        candidate_state_errors = [
            row.get("candidate_state_vs_reprojected_reference", {}).get("relative_error")
            for row in group]
        fp64_residuals = [
            row.get("independent_source_residual_at_captured_candidate", {}).get("scaled_inf")
            for row in group]
        source_vm_native_residual_differences = [
            row.get("independent_source_residual_at_captured_candidate", {}).get(
                "source_vm_difference_from_native_reported_residual")
            for row in group]
        pair_records = [pair for row in group for pair in row.get("finite_difference_pairs", [])]
        candidate_surface_counts: dict[str, int] = {}
        for row in group:
            label = row.get("captured_candidate_yield_surface", {}).get("classification", "missing")
            candidate_surface_counts[label] = candidate_surface_counts.get(label, 0) + 1
        direction_norms = []
        for key, source_entry in sources.items():
            tet, _, side_index = source_entry
            if side_index != 0 or int(tet["environment"]) != environment:
                continue
            direction = metadata[key]["direction_local_per_s"]
            direction_norms.append(math.sqrt(vector_dot(direction, direction)))
        environment_summaries.append({
            "environment": environment,
            "paper_tetrahedra": len(group),
            "elastic_trial_branch_counts": counts,
            "captured_candidate_yield_surface_counts": candidate_surface_counts,
            "nonzero_local_direction_tetrahedra": sum(value > 0.0 for value in direction_norms),
            "zero_local_direction_tetrahedra": sum(value == 0.0 for value in direction_norms),
            "native_stress_vs_candidate_oracle_relative_error": finite_summary(candidate_stress_errors),
            "native_tangent_vs_candidate_oracle_relative_error": finite_summary(candidate_tangent_errors),
            "candidate_state_vs_reprojected_reference_relative_error": finite_summary(candidate_state_errors),
            "independent_candidate_scaled_residual_inf": finite_summary(fp64_residuals),
            "source_vm_vs_native_reported_residual_absolute_difference": finite_summary(
                source_vm_native_residual_differences),
            "finite_difference_pairs": {
                "count": len(pair_records),
                "smooth_branch_eligible": sum(bool(pair.get("smooth_branch_comparison_eligible")) for pair in pair_records),
                "near_yield_unresolved": sum(bool(pair.get("near_yield_unresolved")) for pair in pair_records),
                "branch_crossing": sum(bool(pair.get("branch_crossing")) for pair in pair_records),
                "incomplete_or_unresolved": sum(pair.get("status") != "pair_resolved" for pair in pair_records),
                "zero_captured_direction": sum(
                    pair.get("unresolved_reason") == "captured local deformation direction is zero"
                    for pair in pair_records),
            },
            "strict_gate": "inconclusive_no_measured_noise_floor",
        })
        if reference_derivative_sweep:
            sweep_centers = [value.get("FP64_derivative_sweep", {}) for key, value in replay_rows.items()
                             if key.startswith(f"e{environment}_") and key.endswith("_s0")]
            environment_summaries[-1]["FP64_derivative_sweep"] = {
                "center_count": len(sweep_centers),
                "two_adjacent_smooth_refinements_pass": sum(
                    value.get("verdict") == "two_adjacent_smooth_refinements_pass" for value in sweep_centers),
                "zero_direction_inconclusive": sum(
                    value.get("verdict") == "inconclusive_zero_direction" for value in sweep_centers),
                "no_adjacent_resolved_pass": sum(
                    value.get("verdict") == "no_adjacent_resolved_pass" for value in sweep_centers),
                "missing_sweep": sum(not value for value in sweep_centers),
                "failed_independent_projections": sum(
                    row.get("valid") is not True for value in sweep_centers for row in value.get("rows", [])),
                "Metal_noise_floor_measured": False,
            }
    summary = {
        "schema": SCHEMA,
        "binding": {
            **binding,
            "capture_sha256": sha256(capture_path),
            "prediction_sha256": sha256(prediction_path),
            "receipt_sha256": sha256(receipt_path),
            "replay_executable_sha256": sha256(replay_path),
            "replay_command": replay_command,
            "replay_source_sha256": {
                "cardboard_solver_pair_replay.cpp": sha256(Path(__file__).with_name("cardboard_solver_pair_replay.cpp")),
                "cardboard_solver_reference.hpp": sha256(Path(__file__).with_name("cardboard_solver_reference.hpp")),
                "cardboard_paper_projection_scaling.hpp": sha256(Path(__file__).with_name("cardboard_paper_projection_scaling.hpp")),
            },
            "analyzer_sha256": sha256(Path(__file__)),
            "replay_input_sha256": sha256(input_path),
            "replay_output_sha256": sha256(replay_output_path),
            "replay_source_material_sha256": {
                "liner": sha256(liner_source),
                "medium": sha256(medium_source),
            },
        },
        "selection": {"root": header["root"], "newton_iteration": header["newton_iteration"],
                      "environments": header["environment_count"], "tets_in_capture": len(tets),
                      "paper_tets_replayed": len(sources) // 5,
                      "selected_capture_terminal_integrity": True},
        "run_completion": {
            "receipt_timed_out": binding["receipt_timed_out"],
            "receipt_exit_code": binding["receipt_exit_code"],
            "full_history_complete": full_history_complete(binding),
            "scope_note": "a complete selected Newton capture may be analyzed after timeout; timeout does not establish full-history completion",
        },
        "builder_transforms": {
            "validated_from_sealed_native_argv": binding["required_transforms_present"],
            "source_equation_note": (
                "ordinary FP32 native candidate; normalized compensated source VM is a reference only, not native residual parity; independent FP64 projection solves the same constitutive zero-set"
                if ordinary_generic_reference else
                "initializer and accurate-projection policies produced the captured native candidate; compensated arithmetic and scaled residuals are replayed on transformed source roots; the independent FP64 oracle solves the same zero-set separately"),
            "F_replay_contract": "every captured FP32 F high and low word is widened separately then summed in FP64",
            "candidate_state_replay_contract": "every captured FP32 projected-state high and low word is widened separately then summed in FP64",
            "parameter_contract": "per-tet parameter values are read from the selected environment table and exact native material descriptor; defaults are never substituted",
            "row_scale_contract": "13 row-scale constants are recomputed from the hash-bound material source defaults, which match this capture's per-environment constants; captured residual_scale is evaluated from each environment parameter row",
        },
        "replay": {"returncode": process.returncode, "invalid_rows": invalid_replay_rows,
                   "output_errors": output_errors},
        "near_yield_diagnostic_band": {"q_units": "dimensionless", "width": branch_band,
                                        "physical_tolerance": False},
        "analysis_counts": {
            "tet_audits": len(audits),
            "reference_projection_failures": sum(1 for row in audits if row.get("reprojected_reference_status") == "failed_or_missing"),
            "center_candidate_stress_comparisons": sum(1 for row in audits if "native_stress_vs_oracle_at_captured_candidate" in row),
            "center_candidate_stress_comparisons_with_nonzero_reference": sum(
                row.get("native_stress_vs_oracle_at_captured_candidate", {}).get("nonzero_reference", False)
                for row in audits),
            "center_candidate_tangent_comparisons_with_nonzero_reference": sum(
                row.get("native_tangent_vs_oracle_at_captured_candidate", {}).get("nonzero_reference", False)
                for row in audits),
            "near_yield_centers": sum(1 for row in audits if row.get("trial_yield", {}).get("classification_center") == "near_yield_unresolved"),
        },
        "transformed_source_residual_parity": {
            "compared_center_rows": len(source_residual_differences),
            "maximum_absolute_difference_from_native_projection_residual": max(source_residual_differences, default=None),
            "qualification": "diagnostic only; no threshold is asserted",
        },
        "per_environment_diagnostics": environment_summaries,
        "strict_gate": "inconclusive_no_measured_noise_floor",
        "analysis_exit_code": 0,
        "solver_correctness": "unqualified",
        "physical_validation": False,
        "terminal_failure_explained": False,
    }
    summary["analysis_exit_code"] = analysis_exit_code(summary)
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n",
                                                 encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("replay", type=Path, help="compiled cardboard_solver_pair_replay executable")
    parser.add_argument("output", type=Path)
    parser.add_argument("--near-yield-band", type=float, default=1.0e-6)
    parser.add_argument("--reference-derivative-sweep", action="store_true",
                        help="refine independent FP64 perturbations at every captured center; no Metal noise-floor qualification")
    parser.add_argument("--ordinary-generic-reference", action="store_true",
                        help="explicitly analyze an untransformed FP32 native capture; normalized reference residuals are not native residual parity")
    args = parser.parse_args(argv)
    if not math.isfinite(args.near_yield_band) or args.near_yield_band <= 0:
        parser.error("--near-yield-band must be finite and positive")
    try:
        result = analyze(args.capture, args.prediction, args.receipt,
                         args.replay, args.output, args.near_yield_band,
                         args.reference_derivative_sweep, args.ordinary_generic_reference)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 2
    print(json.dumps({"schema": result["schema"], "output": str(args.output.resolve()),
                      "paper_tets_replayed": result["selection"]["paper_tets_replayed"],
                      "strict_gate": result["strict_gate"],
                      "analysis_exit_code": result["analysis_exit_code"],
                      "physical_validation": False}, allow_nan=False))
    return int(result["analysis_exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
