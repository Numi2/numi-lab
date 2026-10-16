#!/usr/bin/env python3
"""CPU audit of source-bound retained-damage material tangent captures.

This tool does not execute the native binary. It checks sealed run/source
hashes, reconstructs high+low deformation words, replays reported derivative
metrics, and recomputes Hill branch labels. Its strict gate is software
evidence for the synthetic material-point instrument, never physical
qualification.

Exit codes: 0 means every captured material has a passing smooth pair; 1 means
a strict center or smooth-pair gate failed; 2 means the run/capture/source
integrity check failed; 3 means the available pairs are inconclusive.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any


LIMIT = 1.0e-3
EXIT_PASS = 0
EXIT_STRICT_FAILURE = 1
EXIT_INVALID_EVIDENCE = 2
EXIT_INCONCLUSIVE = 3


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def f32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("value is not representable as binary32") from error


def finite_vector(values: Any, length: int, label: str) -> list[float]:
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"{label}: expected {length} values")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label}: nonfinite value")
    return result


def norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def relative_l2(actual: list[float], reference: list[float]) -> float | None:
    denominator = norm(reference)
    if denominator <= 1.0e-12:
        return None
    return norm([a - b for a, b in zip(actual, reference)]) / denominator


def max_abs_difference(a: list[float], b: list[float]) -> float:
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def checked_float_words(high: Any, low: Any, label: str) -> list[float]:
    high_values = finite_vector(high, 9, label + " high")
    low_values = finite_vector(low, 9, label + " low")
    # Capture JSON uses compact decimal formatting for binary32 words, so the
    # parsed decimal need not compare bit-exactly to its binary32 roundtrip.
    # Reconstruct the represented words by rounding each serialized value to
    # binary32 before widening and summing.
    return [float(f32(high_word)) + float(f32(low_word))
            for high_word, low_word in zip(high_values, low_values)]


def state_values(response: dict[str, Any], *, include_low: bool) -> dict[str, float]:
    high = response.get("projected_state")
    if not isinstance(high, dict) or not high:
        raise ValueError("projected state is absent")
    low = response.get("projected_state_low_f32", {}) if include_low else {}
    if include_low and (not isinstance(low, dict) or list(low) != list(high)):
        raise ValueError("candidate high/low state names or order differ")
    state = {}
    for name, value in high.items():
        high_f = f32(value) if include_low else float(value)
        low_f = f32(low[name]) if include_low else 0.0
        state[name] = float(high_f) + float(low_f)
    return state


def branch_label(accepted: dict[str, float], projected: dict[str, float],
                 parameters: dict[str, float]) -> tuple[str, dict[str, float]]:
    def value(state: dict[str, float], name: str) -> float:
        if name not in state:
            raise ValueError(f"yield branch lacks state {name}")
        return state[name]

    p = {name: float(f32(v)) for name, v in parameters.items()}
    s11 = value(projected, "sbar11")
    s22 = value(projected, "sbar22")
    s33 = value(projected, "sbar33")
    s23 = value(projected, "sbar23")
    s13 = value(projected, "sbar13")
    s12 = value(projected, "sbar12")
    q2 = (p["hill_F"] * (s22 - s33) ** 2 +
          p["hill_G"] * (s33 - s11) ** 2 +
          p["hill_H"] * (s11 - s22) ** 2 +
          2.0 * p["hill_L"] * s23 * s23 +
          2.0 * p["hill_M"] * s13 * s13 +
          2.0 * p["hill_N"] * s12 * s12 + p["q_floor"] ** 2)
    hardening = 0.0
    for axis in range(6):
        hardening += p[f"overlay_plastic_hardening_{axis}"] * value(
            projected, f"overlay_plastic_history_{axis}")
    for axis in range(3):
        hardening += p[f"overlay_densification_modulus_pa_axis{axis}"] * value(
            projected, f"overlay_densification_{axis}")
    yield_ratio = 1.0 + hardening / p["sigma_ref"]
    gap = yield_ratio - math.sqrt(max(0.0, q2))
    delta_lambda = (value(projected, "accumulated_multiplier") -
                    value(accepted, "accumulated_multiplier"))
    if delta_lambda > 1.0e-7 and abs(gap) <= 1.0e-4:
        label = "active_plastic"
    elif delta_lambda <= 1.0e-7 and gap > 1.0e-4:
        label = "inactive_elastic"
    else:
        label = "unresolved"
    return label, {"hill_q": math.sqrt(max(0.0, q2)),
                   "yield_ratio": yield_ratio, "yield_gap": gap,
                   "delta_lambda": delta_lambda}


def verify_source_bindings(run_dir: Path, prediction: dict[str, Any],
                           capture_paths: list[Path]) -> dict[str, Any]:
    snapshot = Path(prediction.get("source_snapshot", ""))
    sealed_source_copy = run_dir / "inputs" / "source"
    mismatches: list[dict[str, str]] = []
    checked = 0
    input_hashes = prediction.get("inputs_sha256")
    if not isinstance(input_hashes, dict) or not input_hashes:
        mismatches.append({"path": "inputs_sha256", "reason": "missing sealed input hashes"})
        input_hashes = {}
    for relative, expected in input_hashes.items():
        if (not isinstance(expected, str) or len(expected) != 64 or
                any(char not in "0123456789abcdef" for char in expected.lower())):
            mismatches.append({"path": str(relative), "reason": "malformed expected SHA-256"})
            continue
        if relative.startswith("source/"):
            source_relative = relative.removeprefix("source/")
            archived = sealed_source_copy / source_relative
            path = archived if archived.is_file() else snapshot / source_relative
        elif relative.startswith("runtime/"):
            path = run_dir / "inputs" / "runtime" / relative.removeprefix("runtime/")
        elif relative.startswith("external/"):
            path = run_dir / "inputs" / "external" / relative.removeprefix("external/")
        else:
            mismatches.append({"path": relative, "reason": "unknown input namespace"})
            continue
        checked += 1
        observed = sha256(path)
        if observed != expected:
            mismatches.append({"path": str(path), "expected": expected,
                               "observed": observed or "missing"})

    capture_artifact_checks = []
    runtime_fallback = run_dir / "inputs" / "runtime"
    external_fallback = run_dir / "inputs" / "external"
    for capture_path in capture_paths:
        capture = json.loads(capture_path.read_text())
        source_record = capture.get("source_material", {})
        material_path = Path(source_record.get("path", ""))
        csv_record = capture.get("hardening_dataset", {})
        csv_path = Path(csv_record.get("path", ""))
        def source_fallback(path: Path) -> Path | None:
            try:
                return sealed_source_copy / path.relative_to(snapshot)
            except ValueError:
                return None
        executable_path = Path(prediction.get("argv", [""])[0])
        metallib_path = Path(capture.get("native_artifacts", {}).get("metallib_path", ""))
        checks = {
            "material": (material_path, source_record.get("sha256"),
                         source_fallback(material_path)),
            "hardening_csv": (csv_path, csv_record.get("sha256"),
                              source_fallback(csv_path) or
                              external_fallback / csv_path.name),
            "executable": (executable_path,
                           capture.get("native_artifacts", {}).get("executable_sha256"),
                           runtime_fallback / executable_path.name),
            "metallib": (metallib_path,
                         capture.get("native_artifacts", {}).get("metallib_sha256"),
                         runtime_fallback / "shaders" / metallib_path.name),
        }
        observed = {}
        for name, entry in checks.items():
            path, expected = entry[:2]
            fallback = entry[2] if len(entry) == 3 else None
            chosen_path = path if path.is_file() else fallback
            got = sha256(chosen_path) if chosen_path is not None else None
            observed[name] = {"path": str(path), "verified_path": str(chosen_path),
                              "expected_sha256": expected, "observed_sha256": got,
                              "matches": bool(expected and got == expected)}
            if not observed[name]["matches"]:
                mismatches.append({"path": str(path), "expected": str(expected),
                                   "observed": got or "missing"})
        capture_artifact_checks.append({"capture": str(capture_path), "checks": observed})
    return {"source_snapshot": str(snapshot), "input_hashes_checked": checked,
            "mismatches": mismatches, "all_bindings_match": not mismatches,
            "capture_artifact_checks": capture_artifact_checks}


def audited_capture(path: Path) -> dict[str, Any]:
    capture = json.loads(path.read_text())
    if capture.get("schema") != "numi.cardboard.retained-damage-capture.v3":
        raise ValueError(f"{path.name}: expected capture schema v3")
    if capture.get("evidence_class") != "synthetic_instrument_only_not_calibration":
        raise ValueError(f"{path.name}: evidence class mismatch")
    if capture.get("physical_fem_steps") != 0 or capture.get("physical_calibration") is not False:
        raise ValueError(f"{path.name}: instrument output overstates physical execution")
    if capture.get("compensated_deformation_perturbations_enabled") is not True:
        raise ValueError(f"{path.name}: compensated deformation mode not recorded")
    if capture.get("generic_compensated_implicit_projection") is not True:
        raise ValueError(f"{path.name}: generic compensated projection not recorded")

    f_high = finite_vector(capture.get("F_f32"), 9, "center F high")
    f_low = finite_vector(capture.get("F_low_f32"), 9, "center F low")
    center_f = checked_float_words(f_high, f_low, "center F")
    serialized_center = finite_vector(capture.get("F_sum_f64"), 9, "serialized center F")
    if center_f != serialized_center:
        raise ValueError(f"{path.name}: center high/low sum differs from stored replay F")
    direction = [f32(v) for v in finite_vector(
        capture.get("direction_f32"), 9, "requested direction")]
    if norm(direction) == 0.0:
        raise ValueError(f"{path.name}: requested direction is zero")
    accepted = {name: f32(value) for name, value in
                capture.get("accepted_history_f32", {}).items()}
    parameters = {name: f32(value) for name, value in
                  capture.get("cooked_parameters_f32", {}).items()}
    if not accepted or not parameters:
        raise ValueError(f"{path.name}: accepted history or parameters missing")

    center_native = capture.get("center_native", {})
    center_fp64 = capture.get("center_fp64_source_reprojection", {})
    center_native_valid = (center_native.get("valid") is True and
                           center_native.get("projection_valid") is True)
    center_fp64_valid = (
        math.isfinite(float(center_fp64.get("projection_residual_inf", math.nan))) and
        bool(center_fp64.get("projected_state")))
    if not center_native_valid or not center_fp64_valid:
        return {
            "capture_path": str(path),
            "capture_sha256": sha256(path),
            "material": capture.get("source_material", {}),
            "evidence_class": capture["evidence_class"],
            "physical_fem_steps": capture["physical_fem_steps"],
            "physical_calibration": capture["physical_calibration"],
            "center_projection_valid": {
                "native": center_native_valid, "fp64": center_fp64_valid},
            "strict_gate": {
                "threshold": LIMIT,
                "status": "inconclusive_invalid_center_projection",
                "overall": "inconclusive_invalid_center_projection",
                "passing_epsilons": [], "failing_smooth_pair_epsilons": [],
                "inconclusive_epsilons": []},
        }
    center_native_state = state_values(center_native, include_low=True)
    center_oracle_state = state_values(center_fp64, include_low=False)
    center_native_branch, center_native_terms = branch_label(
        accepted, center_native_state, parameters)
    center_oracle_branch, center_oracle_terms = branch_label(
        accepted, center_oracle_state, parameters)

    rows = []
    for row in capture.get("compensated_deformation_perturbation_sweep", []):
        epsilon = float(row["requested_epsilon"])
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(f"{path.name}: requested epsilon must be finite and positive")
        row_direction = [f32(v) for v in finite_vector(
            row.get("requested_direction_f32"), 9, "row requested direction")]
        if row_direction != direction:
            raise ValueError(f"{path.name}: row requested direction differs from center direction")
        plus_f = checked_float_words(row.get("plus_F_high_f32"),
                                     row.get("plus_F_low_f32"), "plus F")
        minus_f = checked_float_words(row.get("minus_F_high_f32"),
                                      row.get("minus_F_low_f32"), "minus F")
        if plus_f != finite_vector(row.get("plus_F_sum_f64"), 9, "plus F sum"):
            raise ValueError(f"{path.name}: plus F word sum mismatch at epsilon {epsilon}")
        if minus_f != finite_vector(row.get("minus_F_sum_f64"), 9, "minus F sum"):
            raise ValueError(f"{path.name}: minus F word sum mismatch at epsilon {epsilon}")
        realized = [(p - m) / (2.0 * epsilon) for p, m in zip(plus_f, minus_f)]
        recorded_realized = finite_vector(row.get("realized_direction_f64"), 9,
                                          "realized direction")
        if max_abs_difference(realized, recorded_realized) > 5.0e-14:
            raise ValueError(f"{path.name}: recorded realized direction mismatch at {epsilon}")
        direction_error = norm([a - b for a, b in zip(realized, direction)]) / max(
            math.ulp(0.0), norm(direction))
        midpoint = [(p + m) * 0.5 for p, m in zip(plus_f, minus_f)]
        midpoint_offset = max_abs_difference(midpoint, center_f)
        recorded_direction_error = float(row["direction_realization_relative_l2"])
        direction_metric_discrepancy = abs(direction_error - recorded_direction_error)
        if direction_metric_discrepancy > 1.0e-12:
            raise ValueError(f"{path.name}: direction error metric mismatch at {epsilon}")
        recorded_midpoint = float(row["midpoint_offset_inf_f"])
        midpoint_metric_discrepancy = abs(midpoint_offset - recorded_midpoint)
        if midpoint_metric_discrepancy > 5.0e-14:
            raise ValueError(f"{path.name}: midpoint error metric mismatch at {epsilon}")

        native_plus = row.get("native_plus", {})
        native_minus = row.get("native_minus", {})
        oracle_plus = row.get("fp64_plus", {})
        oracle_minus = row.get("fp64_minus", {})
        pairs_valid = (row.get("native_pair_valid") is True and
                       row.get("oracle_pair_valid") is True and
                       native_plus.get("projection_valid") is True and
                       native_minus.get("projection_valid") is True and
                       math.isfinite(float(oracle_plus.get(
                           "projection_residual_inf", math.nan))) and
                       math.isfinite(float(oracle_minus.get(
                           "projection_residual_inf", math.nan))) and
                       bool(oracle_plus.get("projected_state")) and
                       bool(oracle_minus.get("projected_state")))
        if pairs_valid:
            native_plus_stress = finite_vector(native_plus.get("stress_Pa"), 9,
                                               "native plus stress")
            native_minus_stress = finite_vector(native_minus.get("stress_Pa"), 9,
                                                "native minus stress")
            oracle_plus_stress = finite_vector(oracle_plus.get("stress_Pa"), 9,
                                               "FP64 plus stress")
            oracle_minus_stress = finite_vector(oracle_minus.get("stress_Pa"), 9,
                                                "FP64 minus stress")
            native_fd = [(a - b) / (2.0 * epsilon) for a, b in
                         zip(native_plus_stress, native_minus_stress)]
            oracle_fd = [(a - b) / (2.0 * epsilon) for a, b in
                         zip(oracle_plus_stress, oracle_minus_stress)]
            native_ad = finite_vector(row.get("native_ad_action_at_requested_direction_pa_per_s"),
                                      9, "native AD action")
            oracle_ad = finite_vector(row.get("fp64_ad_action_at_requested_direction_pa_per_s"),
                                      9, "FP64 AD action")
            metrics = {
                "native_fd_vs_fp64_fd": relative_l2(native_fd, oracle_fd),
                "native_ad_vs_fp64_fd": relative_l2(native_ad, oracle_fd),
                "fp64_ad_vs_fp64_fd": relative_l2(oracle_ad, oracle_fd),
                "native_ad_vs_native_fd": relative_l2(native_ad, native_fd),
            }
            recorded_vectors = {
                "native_fd": finite_vector(row.get("native_fd_action_pa_per_s"), 9,
                                            "recorded native FD"),
                "fp64_fd": finite_vector(row.get("fp64_fd_action_pa_per_s"), 9,
                                          "recorded FP64 FD"),
                "native_ad": finite_vector(row.get(
                    "native_ad_action_at_requested_direction_pa_per_s"), 9,
                    "recorded native AD"),
                "fp64_ad": finite_vector(row.get(
                    "fp64_ad_action_at_requested_direction_pa_per_s"), 9,
                    "recorded FP64 AD"),
            }
            vector_discrepancies = {
                "native_fd": max_abs_difference(native_fd, recorded_vectors["native_fd"]),
                "fp64_fd": max_abs_difference(oracle_fd, recorded_vectors["fp64_fd"]),
                "native_ad": max_abs_difference(native_ad, recorded_vectors["native_ad"]),
                "fp64_ad": max_abs_difference(oracle_ad, recorded_vectors["fp64_ad"]),
            }
            # The native capture serializes some FP32-derived stresses/actions
            # to nine significant decimal digits. The FD reconstruction can
            # amplify that formatting by 1/epsilon; compare relative vector
            # discrepancies at a bound covering that representation loss.
            recomputed_vectors = {
                "native_fd": native_fd, "fp64_fd": oracle_fd,
                "native_ad": native_ad, "fp64_ad": oracle_ad,
            }
            vector_relative_discrepancies = {}
            for key, recomputed in recomputed_vectors.items():
                discrepancy = relative_l2(recorded_vectors[key], recomputed)
                if discrepancy is None:
                    if max_abs_difference(recorded_vectors[key], recomputed) > 1.0e-10:
                        raise ValueError(f"{path.name}: zero-scale derivative vector mismatch at {epsilon}")
                elif discrepancy > 5.0e-7:
                    raise ValueError(f"{path.name}: derivative vector mismatch at {epsilon}")
                vector_relative_discrepancies[key] = discrepancy
            recorded_metrics = {
                "native_fd_vs_fp64_fd": row.get("native_fd_vs_fp64_fd_relative_l2"),
                "native_ad_vs_fp64_fd": row.get("native_ad_vs_fp64_fd_relative_l2"),
                "fp64_ad_vs_fp64_fd": row.get("fp64_ad_vs_fp64_fd_relative_l2"),
            }
            metric_discrepancies = {}
            for key, recorded in recorded_metrics.items():
                calculated = metrics[key]
                if calculated is None:
                    if recorded is not None:
                        raise ValueError(f"{path.name}: metric availability mismatch at {epsilon}")
                    metric_discrepancies[key] = None
                else:
                    if (not isinstance(recorded, (int, float)) or
                            not math.isfinite(float(recorded))):
                        raise ValueError(f"{path.name}: missing recorded metric at {epsilon}")
                    metric_discrepancies[key] = abs(calculated - float(recorded))
            finite_metric_discrepancies = [value for value in metric_discrepancies.values()
                                           if value is not None]
            if finite_metric_discrepancies and max(finite_metric_discrepancies) > 5.0e-7:
                raise ValueError(f"{path.name}: derivative error metric mismatch at {epsilon}")
            native_plus_branch, native_plus_terms = branch_label(
                accepted, state_values(native_plus, include_low=True), parameters)
            native_minus_branch, native_minus_terms = branch_label(
                accepted, state_values(native_minus, include_low=True), parameters)
            oracle_plus_branch, oracle_plus_terms = branch_label(
                accepted, state_values(oracle_plus, include_low=False), parameters)
            oracle_minus_branch, oracle_minus_terms = branch_label(
                accepted, state_values(oracle_minus, include_low=False), parameters)
            oracle_smooth = (center_oracle_branch != "unresolved" and
                             oracle_plus_branch == center_oracle_branch and
                             oracle_minus_branch == center_oracle_branch)
            native_matches = (center_native_branch == center_oracle_branch and
                              native_plus_branch == oracle_plus_branch and
                              native_minus_branch == oracle_minus_branch)
            if (row.get("oracle_branch_preserved") is not oracle_smooth or
                    row.get("native_branch_matches_fp64") is not native_matches):
                raise ValueError(f"{path.name}: recorded branch classification mismatch at {epsilon}")
            derivative_pass = all(value is not None and value <= LIMIT for value in
                                  (metrics["native_fd_vs_fp64_fd"],
                                   metrics["native_ad_vs_fp64_fd"],
                                   metrics["fp64_ad_vs_fp64_fd"]))
            direction_pass = direction_error <= LIMIT
            if not pairs_valid:
                row_status = "inconclusive_invalid_pair"
            elif not direction_pass or not oracle_smooth or not native_matches:
                row_status = "inconclusive_direction_or_branch"
            elif not row.get("fp64_fd_step_check_gate_1e_3", False):
                row_status = "inconclusive_fp64_fd_step_check"
            elif any(value is None for value in metrics.values()):
                row_status = "inconclusive_below_response_scale"
            elif derivative_pass:
                row_status = "pass"
            else:
                row_status = "fail_smooth_pair_derivative"
        else:
            metrics = {"native_fd_vs_fp64_fd": None, "native_ad_vs_fp64_fd": None,
                       "fp64_ad_vs_fp64_fd": None, "native_ad_vs_native_fd": None}
            native_plus_branch = native_minus_branch = None
            oracle_plus_branch = oracle_minus_branch = None
            native_plus_terms = native_minus_terms = None
            oracle_plus_terms = oracle_minus_terms = None
            oracle_smooth = native_matches = False
            derivative_pass = direction_pass = False
            row_status = "inconclusive_invalid_pair"

        rows.append({
            "epsilon": epsilon,
            "F_plus_sum_f64": plus_f,
            "F_minus_sum_f64": minus_f,
            "realized_direction_f64": realized,
            "direction_relative_l2_error": direction_error,
            "direction_gate_le_1e-3": direction_pass,
            "midpoint_offset_inf": midpoint_offset,
            "recorded_metric_discrepancies": {
                "direction_relative_l2": direction_metric_discrepancy,
                "midpoint_offset_inf": midpoint_metric_discrepancy,
            },
            "branch_labels": {
                "center_native": center_native_branch,
                "center_fp64": center_oracle_branch,
                "plus_native": native_plus_branch,
                "minus_native": native_minus_branch,
                "plus_fp64": oracle_plus_branch,
                "minus_fp64": oracle_minus_branch,
                "fp64_branch_preserved": oracle_smooth,
                "native_branch_matches_fp64": native_matches,
            },
            "branch_terms": {
                "center_native": center_native_terms,
                "center_fp64": center_oracle_terms,
                "plus_native": native_plus_terms,
                "minus_native": native_minus_terms,
                "plus_fp64": oracle_plus_terms,
                "minus_fp64": oracle_minus_terms,
            },
            "relative_derivative_errors": metrics,
            "derivative_replay_discrepancies": {
                "vectors_relative_l2": vector_relative_discrepancies if pairs_valid else None,
                "relative_errors": metric_discrepancies if pairs_valid else None,
            },
            "derivative_gates_le_1e-3": {
                "native_fd_vs_fp64_fd": metrics["native_fd_vs_fp64_fd"] is not None and metrics["native_fd_vs_fp64_fd"] <= LIMIT,
                "native_ad_vs_fp64_fd": metrics["native_ad_vs_fp64_fd"] is not None and metrics["native_ad_vs_fp64_fd"] <= LIMIT,
                "fp64_ad_vs_fp64_fd": metrics["fp64_ad_vs_fp64_fd"] is not None and metrics["fp64_ad_vs_fp64_fd"] <= LIMIT,
            },
            "row_status": row_status,
        })

    if not rows:
        raise ValueError(f"{path.name}: no compensated perturbation rows")
    pass_rows = [row["epsilon"] for row in rows if row["row_status"] == "pass"]
    fail_rows = [row["epsilon"] for row in rows
                 if row["row_status"] == "fail_smooth_pair_derivative"]
    inconclusive_rows = [row["epsilon"] for row in rows
                         if row["row_status"].startswith("inconclusive")]
    center_comparison = capture.get("native_vs_fp64_center", {})
    center_stress_error = center_comparison.get("stress_relative_l2")
    center_tangent_error = center_comparison.get("tangent_relative_l2")
    center_parity = (isinstance(center_stress_error, (int, float)) and
                     isinstance(center_tangent_error, (int, float)) and
                     center_stress_error <= LIMIT and center_tangent_error <= LIMIT)
    if not center_parity:
        status = "failure_center_stress_or_tangent_parity"
    elif pass_rows:
        status = "pass"
    elif fail_rows:
        status = "failure_smooth_requested_pair"
    else:
        status = "inconclusive_no_qualifying_smooth_pair"
    overall = {
        "pass": "strict_gate_pass",
        "failure_center_stress_or_tangent_parity": "strict_gate_failure_center_parity",
        "failure_smooth_requested_pair": "strict_gate_failure_on_smooth_requested_pair",
        "inconclusive_no_qualifying_smooth_pair": "strict_gate_inconclusive",
    }[status]
    return {
        "capture_path": str(path),
        "capture_sha256": sha256(path),
        "material": capture.get("source_material", {}),
        "evidence_class": capture["evidence_class"],
        "physical_fem_steps": capture["physical_fem_steps"],
        "physical_calibration": capture["physical_calibration"],
        "center_F_high_plus_low_f64": center_f,
        "accepted_history_f32": accepted,
        "center_branch_labels": {"native": center_native_branch,
                                 "fp64": center_oracle_branch},
        "center_branch_terms": {"native": center_native_terms,
                                "fp64": center_oracle_terms},
        "center_stress_relative_l2": center_stress_error,
        "center_tangent_relative_l2": center_tangent_error,
        "center_parity_gate_le_1e-3": center_parity,
        "pair_rows": rows,
        "strict_gate": {"threshold": LIMIT, "status": status, "overall": overall,
                        "passing_epsilons": pass_rows,
                        "failing_smooth_pair_epsilons": fail_rows,
                        "inconclusive_epsilons": inconclusive_rows},
    }


def audit_run(run_dir: Path, output: Path | None = None) -> tuple[dict[str, Any], int]:
    run_dir = run_dir.resolve()
    prediction_path = run_dir / "prediction.json"
    receipt_path = run_dir / "receipt.json"
    try:
        prediction = json.loads(prediction_path.read_text())
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        report = {
            "schema": "numi.cardboard.retained-damage-capture-audit.v1",
            "run_dir": str(run_dir), "status": "invalid_sealed_run",
            "error": str(error), "physical_or_calibration_qualification": False,
        }
        _write_report(report, output or run_dir / "analysis" / "retained_damage_capture_audit.json")
        return report, EXIT_INVALID_EVIDENCE

    captures = sorted((run_dir / "capture").glob("*.tangent-capture.json"))
    if not captures:
        report = {
            "schema": "numi.cardboard.retained-damage-capture-audit.v1",
            "run_dir": str(run_dir), "status": "invalid_missing_captures",
            "physical_or_calibration_qualification": False,
        }
        _write_report(report, output or run_dir / "analysis" / "retained_damage_capture_audit.json")
        return report, EXIT_INVALID_EVIDENCE

    try:
        bindings = verify_source_bindings(run_dir, prediction, captures)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        bindings = {"all_bindings_match": False,
                    "mismatches": [{"reason": f"binding audit could not complete: {error}"}],
                    "capture_artifact_checks": []}
    capture_results = []
    capture_errors = []
    for path in captures:
        try:
            capture_results.append(audited_capture(path))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            capture_errors.append({"capture": str(path), "error": str(error)})

    native_run_ok = (receipt.get("exit_code") == 0 and
                     receipt.get("source_inputs_unchanged") is True and
                     not receipt.get("timed_out", False) and
                     prediction.get("executed_build_verified") is True)
    if not native_run_ok or not bindings["all_bindings_match"] or capture_errors:
        status, exit_code = "invalid_source_or_capture_binding", EXIT_INVALID_EVIDENCE
    else:
        statuses = [capture["strict_gate"]["status"] for capture in capture_results]
        if any(item.startswith("failure") for item in statuses):
            status, exit_code = "strict_gate_failure", EXIT_STRICT_FAILURE
        elif all(item == "pass" for item in statuses):
            status, exit_code = "strict_gate_pass", EXIT_PASS
        else:
            status, exit_code = "strict_gate_inconclusive", EXIT_INCONCLUSIVE

    report = {
        "schema": "numi.cardboard.retained-damage-capture-audit.v1",
        "auditor": str(Path(__file__).resolve()),
        "run_dir": str(run_dir),
        "prediction_sha256": sha256(prediction_path),
        "receipt_sha256": sha256(receipt_path),
        "native_run_success_and_build_verified": native_run_ok,
        "source_rebinding": bindings,
        "capture_errors": capture_errors,
        "captures": capture_results,
        "overall": {
            "status": status, "exit_code": exit_code,
            "physical_or_calibration_qualification": False,
            "interpretation": (
                "CPU audit of captured high/low deformation words, accepted histories, "
                "source bindings, smooth-branch labels, FP64 finite-difference stability, "
                "and the 1e-3 software tangent gate; the instrument has zero physical FEM steps."),
        },
    }
    _write_report(report, output or run_dir / "analysis" / "retained_damage_capture_audit.json")
    return report, exit_code


def _write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report, exit_code = audit_run(args.run_dir, args.output)
    output = args.output or (args.run_dir.resolve() / "analysis" /
                             "retained_damage_capture_audit.json")
    print(f"audit_json={output}")
    print(f"status={report.get('overall', {}).get('status', report.get('status'))}")
    print(f"exit_code={exit_code}")
    print(f"native_run_success_and_build_verified={report.get('native_run_success_and_build_verified')}")
    print(f"source_bindings_match={report.get('source_rebinding', {}).get('all_bindings_match')}")
    for result in report.get("captures", []):
        gate = result["strict_gate"]
        print(f"capture={Path(result['capture_path']).name} overall={gate['overall']} "
              f"pass={len(gate['passing_epsilons'])} fail={len(gate['failing_smooth_pair_epsilons'])} "
              f"inconclusive={len(gate['inconclusive_epsilons'])}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
