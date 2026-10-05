#!/usr/bin/env python3
"""Bind, calibrate, and observe a paired respiratory-drive intervention.

The native Matter executable remains the only simulator. This adapter invokes
it once per declared arm, validates its accepted CSV, and emits one JSON record
for the Numi Lab v2 notebook. Its calibration covers CSV parsing/window
arithmetic only; it does not calibrate physiology or clinical measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any


TRACE_COLUMNS = (
    "time_s", "lung_volume_ml", "airflow_ml_s", "PaO2_mmhg", "PaCO2_mmhg",
    "SaO2", "oxygen_balance_error_stpd_ml", "co2_balance_error_stpd_ml",
    "breaths", "tidal_ml", "lv_mmhg", "rv_mmhg", "aorta_mmhg",
    "pulmonary_artery_mmhg", "lv_ml", "rv_ml", "blood_ml",
    "blood_error_ml", "aortic_ejected_ml", "pulmonary_ejected_ml",
    "complete_filling_ejection_cycles", "last_lv_stroke_ml",
)

SOURCE_INPUTS = (
    "matter/CMakeLists.txt",
    "matter/src/matter_kernels.metal",
    "matter/src/runtime.mm",
    "matter/src/human_physiology.mm",
    "matter/src/metal_world.cpp",
    "matter/src/numi_human.mm",
    "matter/src/metal/common.metalinc",
    "matter/src/metal/mpm.metalinc",
    "matter/src/metal/fem.metalinc",
    "matter/src/metal/fgmres.metalinc",
    "matter/src/metal/vascular.metalinc",
    "matter/src/metal/human_equality.metalinc",
    "matter/src/metal/human_limits.metalinc",
    "matter/src/metal/mixed_fem.metalinc",
    "matter/src/metal/topology_mutation.metalinc",
    "matter/src/metal/contact.metalinc",
    "matter/src/metal/adaptive.metalinc",
    "matter/src/metal/identification_scheduler.metalinc",
    "matter/src/metal/numi_human.metalinc",
    "matter/src/metal/accepted_state_proof.metalinc",
    "matter/include/numi/matter/matter.hpp",
    "matter/include/numi/matter/shared.h",
    "matter/include/numi/matter/accepted_state_proof_gpu.h",
    "matter/include/numi/matter/fiber_exp_linear.h",
    "matter/include/numi/matter/human_equality_gpu.h",
    "matter/include/numi/matter/human_limits_gpu.h",
    "matter/include/numi/matter/human_support_precision_gpu.h",
    "include/metalrobo/engine_types.h",
    "include/metalrobo/compensated_translation_gpu.h",
    "include/metalrobo/compensated_geometry_gpu.h",
    "include/metalrobo/rod_gpu_shared.h",
    "matter/tools/human_resting.mm",
    "matter/tools/human_resting_runtime.hpp",
    "matter/tools/human_respiration_parameters.hpp",
    "matter/include/numi/matter/human_respiration.h",
    "matter/src/human_respiration.metal",
    "src/metal/MujocoMuscleReference.metal",
    "include/metalrobo/mujoco_muscle_gpu.h",
    "include/metalrobo/numi_human_stand_gpu.h",
    "include/metalrobo/numi_human_resting_visual_gpu.h",
    "matter/tools/fixtures/cvsim21.native.v3.json",
    "matter/examples/resting-reference-respiration.json",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def finite_float(value: str, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {field}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"nonfinite value for {field}")
    return number


def read_trace(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not set(TRACE_COLUMNS).issubset(reader.fieldnames):
            missing = sorted(set(TRACE_COLUMNS) - set(reader.fieldnames or ()))
            raise ValueError(f"native trace is missing required columns: {missing}")
        rows: list[dict[str, float]] = []
        previous = -math.inf
        for line_number, raw in enumerate(reader, start=2):
            row = {column: finite_float(raw[column], f"{column} line {line_number}")
                   for column in TRACE_COLUMNS}
            if row["time_s"] <= previous:
                raise ValueError(f"native trace time is not strictly increasing at line {line_number}")
            previous = row["time_s"]
            rows.append(row)
    if len(rows) < 2:
        raise ValueError("native trace must contain at least two accepted samples")
    return rows


def window_rows(rows: list[dict[str, float]], start_s: float, end_s: float) -> list[dict[str, float]]:
    selected = [row for row in rows if start_s <= row["time_s"] < end_s]
    if len(selected) < 100:
        raise ValueError(f"window [{start_s}, {end_s}) has too few accepted samples: {len(selected)}")
    return selected


def mean(rows: list[dict[str, float]], key: str) -> float:
    value = math.fsum(row[key] for row in rows) / len(rows)
    if not math.isfinite(value):
        raise ValueError(f"nonfinite window mean for {key}")
    return value


def positive_linear_area(flow0: float, flow1: float, dt: float) -> float:
    """Integral of max(linearly-interpolated flow, 0) in ml."""
    if flow0 >= 0.0 and flow1 >= 0.0:
        return 0.5 * (flow0 + flow1) * dt
    if flow0 <= 0.0 and flow1 <= 0.0:
        return 0.0
    if flow0 > 0.0:
        crossing = dt * flow0 / (flow0 - flow1)
        return 0.5 * flow0 * crossing
    crossing = dt * (-flow0) / (flow1 - flow0)
    return 0.5 * flow1 * (dt - crossing)


def inspiratory_l_min(rows: list[dict[str, float]], start_s: float, end_s: float) -> float:
    """Estimate inspiratory minute ventilation from the accepted flow trace."""
    area_ml = 0.0
    for left, right in zip(rows, rows[1:]):
        t0, t1 = left["time_s"], right["time_s"]
        lo, hi = max(t0, start_s), min(t1, end_s)
        if hi <= lo:
            continue
        span = t1 - t0
        f0 = left["airflow_ml_s"] + (right["airflow_ml_s"] - left["airflow_ml_s"]) * ((lo - t0) / span)
        f1 = left["airflow_ml_s"] + (right["airflow_ml_s"] - left["airflow_ml_s"]) * ((hi - t0) / span)
        area_ml += positive_linear_area(f0, f1, hi - lo)
    return area_ml * 60.0 / ((end_s - start_s) * 1000.0)


def complete_breath_metrics(rows: list[dict[str, float]], start_s: float, end_s: float) -> dict[str, Any]:
    """Use accepted breath transitions to exclude partial breaths at window edges.

    The owner increments `breaths` at the start of inspiration. Interpolate that
    crossing within the retained accepted-flow samples; never infer a prescribed
    respiratory frequency or use the controller's requested ventilation.
    """
    boundaries: list[float] = []
    for left, right in zip(rows, rows[1:]):
        change = right["breaths"] - left["breaths"]
        if change == 0:
            continue
        if change != 1 or left["airflow_ml_s"] > 0 or right["airflow_ml_s"] <= 0:
            raise ValueError("trace does not resolve accepted inspiratory breath transitions")
        f0, f1 = left["airflow_ml_s"], right["airflow_ml_s"]
        crossing = left["time_s"] + (right["time_s"] - left["time_s"]) * (-f0) / (f1 - f0)
        if start_s <= crossing <= end_s:
            boundaries.append(crossing)
    if len(boundaries) < 2:
        return {"available": False, "complete_breath_count": 0,
                "reason": "fewer than two accepted inspiratory boundaries inside the analysis window"}
    first, last = boundaries[0], boundaries[-1]
    return {"available": True, "complete_breath_count": len(boundaries) - 1,
            "window_s": [first, last],
            "inspiratory_minute_ventilation_L_min": inspiratory_l_min(rows, first, last),
            "respiratory_rate_per_min": 60.0 * (len(boundaries) - 1) / (last - first)}


def window_metrics(rows: list[dict[str, float]], start_s: float, end_s: float) -> dict[str, float]:
    samples = window_rows(rows, start_s, end_s)
    out = {
        "mean_PaCO2_mmhg": mean(samples, "PaCO2_mmhg"),
        "mean_PaO2_mmhg": mean(samples, "PaO2_mmhg"),
        "mean_SaO2": mean(samples, "SaO2"),
        "mean_lung_volume_ml": mean(samples, "lung_volume_ml"),
        "inspiratory_minute_ventilation_L_min": inspiratory_l_min(rows, start_s, end_s),
        "breaths_at_window_start": samples[0]["breaths"],
        "breaths_at_window_end": samples[-1]["breaths"],
        "mean_tidal_volume_ml": mean(samples, "tidal_ml"),
        "complete_fill_ejection_cycles_at_end": samples[-1]["complete_filling_ejection_cycles"],
        "last_LV_stroke_volume_ml": samples[-1]["last_lv_stroke_ml"],
        "max_abs_blood_error_ml": max(abs(row["blood_error_ml"]) for row in samples),
        "max_abs_O2_balance_error_STPD_ml": max(abs(row["oxygen_balance_error_stpd_ml"]) for row in samples),
        "max_abs_CO2_balance_error_STPD_ml": max(abs(row["co2_balance_error_stpd_ml"]) for row in samples),
        "accepted_samples": float(len(samples)),
    }
    if any(not math.isfinite(value) for value in out.values()):
        raise ValueError(f"nonfinite window metric in [{start_s}, {end_s})")
    return out


def known_parser_calibration(fixture: Path) -> dict[str, Any]:
    rows = read_trace(fixture)
    last = rows[-1]
    checks = [
        {"id": "native_csv_required_columns_and_monotonic_time", "passed": True,
         "samples": len(rows)},
        {"id": "native_csv_known_last_time", "passed": abs(last["time_s"] - 6.00000028498) <= 2e-7,
         "measured": last["time_s"], "expected": 6.00000028498},
        {"id": "native_csv_known_last_PaCO2", "passed": abs(last["PaCO2_mmhg"] - 39.3035583496) <= 2e-5,
         "measured": last["PaCO2_mmhg"], "expected": 39.3035583496},
        {"id": "native_csv_known_complete_cycles", "passed": last["complete_filling_ejection_cycles"] == 7,
         "measured": last["complete_filling_ejection_cycles"], "expected": 7},
    ]
    synthetic = [
        {"time_s": float(t), "PaCO2_mmhg": value}
        for t, value in ((0, 36.0), (1, 37.0), (2, 38.0), (3, 39.0))
    ]
    known_mean = mean(synthetic, "PaCO2_mmhg")
    checks.append({"id": "known_value_window_mean", "passed": known_mean == 37.5,
                   "measured": known_mean, "expected": 37.5})
    rejected_nonfinite = False
    try:
        finite_float("nan", "calibration")
    except ValueError:
        rejected_nonfinite = True
    checks.append({"id": "reject_nonfinite_trace_values", "passed": rejected_nonfinite})
    if not all(check["passed"] for check in checks):
        raise ValueError("parser calibration failed; see failed checks")
    return {"schema": "numi.science.calibration.v1", "status": "passed", "checks": checks,
            "scope": "Known-value native CSV parsing, monotonic accepted-time validation, finite-value rejection, and window-mean arithmetic only. This does not calibrate physiological sensors, the gas-exchange model, biological parameters, or clinical validity.",
            "observed_units": []}


def model_prediction(model: dict[str, Any]) -> dict[str, float | str]:
    if model.get("schema") != "numi.human-resting-paco2-model.v1":
        raise ValueError("unsupported resting PaCO2 model schema")
    p = model["parameters"]
    baseline = float(p["baseline_PaCO2_mmhg"])
    min_vent = float(p["minimum_effective_alveolar_ventilation_fraction"])
    max_vent = float(p["maximum_effective_alveolar_ventilation_fraction"])
    if (not all(math.isfinite(value) for value in (baseline, min_vent, max_vent)) or
            baseline <= 0.0 or not 0.0 < min_vent <= max_vent <= 1.0):
        raise ValueError("invalid alveolar-ventilation sensitivity parameters")
    # At fixed CO2 production, the alveolar ventilation relation gives
    # PaCO2_treatment / PaCO2_control = 1 / (VA_treatment / VA_control).
    low = baseline * (1.0 / max_vent - 1.0)
    high = baseline * (1.0 / min_vent - 1.0)
    return {"estimand": "paired_difference_mean", "minimum": low, "maximum": high}


def runner_summary(log: str) -> dict[str, Any]:
    runtime_line = next((line for line in log.splitlines() if line.startswith("runtime=")), None)
    summary_line = next((line for line in reversed(log.splitlines()) if line.startswith("accepted_steps=")), None)
    if runtime_line is None or summary_line is None:
        raise ValueError("native runner did not report runtime and accepted-step summaries")
    dense45 = "eligible dense45 vascular solve" in runtime_line
    if not dense45 or "device=Apple" not in runtime_line:
        raise ValueError("runner did not initialize the expected Dense45 solver on physical Apple GPU")
    device_match = re.search(r"\bdevice=(.*?)\s+world_fingerprint=", runtime_line)
    fingerprint_match = re.search(r"\bworld_fingerprint=([0-9]+)", runtime_line)
    if device_match is None or fingerprint_match is None:
        raise ValueError("native runtime line is missing physical device or world identity")
    fields = dict(re.findall(r"([A-Za-z_]+)=([^ ]+)", summary_line))
    try:
        steps = int(fields["accepted_steps"])
        simulated_s = float(fields["simulated_s"])
        wall_s = float(fields["wall_s"])
        gpu_s = float(fields["gpu_s"])
        rtf = float(fields["real_time_factor"])
    except (KeyError, ValueError) as exc:
        raise ValueError("native runner summary fields are malformed") from exc
    if (steps <= 0 or not all(math.isfinite(value) for value in (simulated_s, wall_s, gpu_s, rtf)) or
            fields.get("Brain_control") != "1" or fields.get("no_anatomy_or_resting_claim") != "true"):
        raise ValueError("native runner did not accept all steps under the intended coupled Brain-controlled scope")
    return {"runtime_line": runtime_line, "accepted_steps": steps, "simulated_s": simulated_s,
            "wall_s": wall_s, "gpu_s": gpu_s, "real_time_factor": rtf,
            "brain_control": True, "vascular_dense45": dense45,
            "device": device_match.group(1), "world_fingerprint": fingerprint_match.group(1),
            "whole_body_anatomy_qualified": False}


def validate_windows(args: argparse.Namespace) -> None:
    duration_s, width = args.steps * args.dt, args.window_s
    if (not all(math.isfinite(x) for x in (duration_s, width, args.start_s, args.end_s, args.scale)) or
            width < 5.0 or args.start_s < width or args.end_s <= args.start_s + width or
            args.end_s > duration_s - width or not 0.0 <= args.scale <= 2.0):
        raise ValueError("intervention must leave nonoverlapping analysis windows of at least 5 s")


def execute_arm(args: argparse.Namespace) -> dict[str, Any]:
    work = Path.cwd()
    trace = Path(args.output).resolve()
    if trace.parent != work.resolve():
        raise ValueError("native trace output must stay in the notebook-provided run directory")
    if trace.exists():
        raise ValueError("refusing to overwrite a prior native trace")
    duration_s = args.steps * args.dt
    width = args.window_s
    validate_windows(args)
    checks = ((args.loaded_matter_metallib, args.frozen_matter_metallib),
              (args.loaded_respiration_metallib, args.frozen_respiration_metallib))
    for loaded, frozen in checks:
        if sha256_file(Path(loaded)) != sha256_file(Path(frozen)):
            raise ValueError(f"loaded/frozen metallib identity mismatch: {loaded}")
    command = [args.runner, args.network, args.parameters, str(trace), "--steps", str(args.steps),
               "--dt", repr(args.dt), "--vascular-dense45"]
    if args.arm == "treatment":
        command.extend(("--drive-intervention", repr(args.start_s), repr(args.end_s), repr(args.scale)))
    completed = subprocess.run(command, cwd=work, capture_output=True, text=True, check=False)
    (work / "native.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (work / "native.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise ValueError(f"native owner failed with status {completed.returncode}; logs retained in run directory")
    native = runner_summary(completed.stdout)
    if native["device"] != args.device:
        raise ValueError(f"native device identity mismatch: {native['device']} != {args.device}")
    if native["world_fingerprint"] != args.world_fingerprint:
        raise ValueError("native world fingerprint does not match the preregistered source world")
    if native["accepted_steps"] != args.steps:
        raise ValueError(f"accepted-step count mismatch: {native['accepted_steps']} != {args.steps}")
    expected_seconds = duration_s
    if abs(native["simulated_s"] - expected_seconds) > max(1.0e-4, expected_seconds * 1.0e-7):
        raise ValueError("native runner simulated time does not match declared step count")
    result = observation(args, trace, native, completed.stdout)
    # Verify dynamic library images remained the exact copies during this run.
    for loaded, frozen in checks:
        if sha256_file(Path(loaded)) != sha256_file(Path(frozen)):
            raise ValueError(f"loaded metallib changed during run: {loaded}")
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return result


def observation(args: argparse.Namespace, trace: Path, native: dict[str, Any], log: str) -> dict[str, Any]:
    """One instrument for both native entry points; every value uses accepted state."""
    expected_seconds, width = args.steps * args.dt, args.window_s
    if abs(native["simulated_s"] - expected_seconds) > max(1e-4, expected_seconds * 1e-7):
        raise ValueError("accepted physical time differs from the preregistered duration")
    rows = read_trace(trace)
    if rows[-1]["time_s"] < expected_seconds - 0.05 or rows[-1]["time_s"] > expected_seconds + 1.0e-3:
        raise ValueError("native trace does not cover the full declared simulation duration")
    pre = window_metrics(rows, args.start_s - width, args.start_s)
    dose = window_metrics(rows, args.end_s - width, args.end_s)
    recovery = window_metrics(rows, duration_s - width, duration_s)
    primary = dose["mean_PaCO2_mmhg"] - pre["mean_PaCO2_mmhg"]
    result: dict[str, Any] = {
        "schema": "numi.human-resting.intervention-observation.v1",
        "unit_id": args.unit_id,
        "arm": args.arm,
        "accepted_steps": native["accepted_steps"],
        "simulated_s": native["simulated_s"],
        "device": native["device"],
        "world_fingerprint": native["world_fingerprint"],
        "timestep_s": args.dt,
        "dense45": native["vascular_dense45"],
        "brain_control": native["brain_control"],
        "nominal_duration_s": expected_seconds,
        "duration_valid": abs(native["simulated_s"] - expected_seconds) <= max(1.0e-4, expected_seconds * 1.0e-7),
        "whole_body_anatomy_qualified": False,
        "intervention_applied": args.arm == "treatment",
        "delivered_drive_scale": args.scale if args.arm == "treatment" else 1.0,
        "primary_delta_PaCO2_mmhg": primary,
        "pre_window_s": [args.start_s - width, args.start_s],
        "dose_window_s": [args.end_s - width, args.end_s],
        "recovery_window_s": [expected_seconds - width, expected_seconds],
        "complete_breath_windows": {
            "pre": complete_breath_metrics(rows, args.start_s - width, args.start_s),
            "dose": complete_breath_metrics(rows, args.end_s - width, args.end_s),
            "recovery": complete_breath_metrics(rows, expected_seconds - width, expected_seconds),
        },
        "PaCO2_pre_mean_mmhg": pre["mean_PaCO2_mmhg"],
        "PaCO2_dose_mean_mmhg": dose["mean_PaCO2_mmhg"],
        "PaCO2_recovery_mean_mmhg": recovery["mean_PaCO2_mmhg"],
        "PaO2_pre_mean_mmhg": pre["mean_PaO2_mmhg"],
        "PaO2_dose_mean_mmhg": dose["mean_PaO2_mmhg"],
        "PaO2_recovery_mean_mmhg": recovery["mean_PaO2_mmhg"],
        "SaO2_pre_mean": pre["mean_SaO2"],
        "SaO2_dose_mean": dose["mean_SaO2"],
        "SaO2_recovery_mean": recovery["mean_SaO2"],
        "inspiratory_minute_ventilation_pre_L_min": pre["inspiratory_minute_ventilation_L_min"],
        "inspiratory_minute_ventilation_dose_L_min": dose["inspiratory_minute_ventilation_L_min"],
        "inspiratory_minute_ventilation_recovery_L_min": recovery["inspiratory_minute_ventilation_L_min"],
        "ventilation_delta_pre_to_dose_L_min": dose["inspiratory_minute_ventilation_L_min"] - pre["inspiratory_minute_ventilation_L_min"],
        "ventilation_delta_pre_to_recovery_L_min": recovery["inspiratory_minute_ventilation_L_min"] - pre["inspiratory_minute_ventilation_L_min"],
        "lung_volume_pre_ml": pre["mean_lung_volume_ml"],
        "lung_volume_dose_ml": dose["mean_lung_volume_ml"],
        "lung_volume_recovery_ml": recovery["mean_lung_volume_ml"],
        "breaths_pre_window": pre["breaths_at_window_end"],
        "breaths_dose_window": dose["breaths_at_window_end"],
        "breaths_recovery_window": recovery["breaths_at_window_end"],
        "recovery_minus_pre_PaCO2_mmhg": recovery["mean_PaCO2_mmhg"] - pre["mean_PaCO2_mmhg"],
        "dose_to_recovery_PaCO2_change_mmhg": recovery["mean_PaCO2_mmhg"] - dose["mean_PaCO2_mmhg"],
        "blood_volume_error_abs_max_ml": max(abs(row["blood_error_ml"]) for row in rows),
        "O2_balance_error_abs_max_STPD_ml": max(abs(row["oxygen_balance_error_stpd_ml"]) for row in rows),
        "CO2_balance_error_abs_max_STPD_ml": max(abs(row["co2_balance_error_stpd_ml"]) for row in rows),
        "complete_fill_ejection_cycles": rows[-1]["complete_filling_ejection_cycles"],
        "last_LV_stroke_volume_ml": rows[-1]["last_lv_stroke_ml"],
        "real_time_factor": native["real_time_factor"],
        "trace_sha256": sha256_file(trace),
        "runtime_log_sha256": hashlib.sha256(log.encode()).hexdigest(),
    }
    return result


def native_scene_command(invocation: dict[str, Any], output: Path, args: argparse.Namespace) -> list[str]:
    """Rebind the existing Human launch receipt to one preregistered arm."""
    command = list(invocation["argv"])
    if len(command) < 5 or Path(command[0]).name != "numi-human-native":
        raise ValueError("expected the existing anatomical native Human owner invocation")
    if "--mechanics-only" in command or "--resting-drive-intervention" in command:
        raise ValueError("the paired reference invocation must include the viewer and no intervention")
    for required in ("--persistent-metal-stand", "--resting-scene", "--vascular-dense45",
                     "--resting-anatomy-receipt", "--skin-payload", "--tendon-payload"):
        if command.count(required) != 1:
            raise ValueError(f"native reference invocation is missing/duplicates {required}")
    command[4] = str(output)
    for flag, value in (("--muscle-step-count", str(args.steps)), ("--muscle-step-seconds", repr(args.dt)),
                        ("--resting-movie", str(output / "native-viewer.mov"))):
        if command.count(flag) != 1:
            raise ValueError(f"native reference invocation is missing/duplicates {flag}")
        command[command.index(flag) + 1] = value
    bindings = invocation.get("asset_sha256", {})
    if not bindings:
        raise ValueError("native invocation has no source/binary/asset identities")
    # Every consumed file must be bound. Outputs are the only absolute paths
    # allowed to be absent from the owner's manifest.
    outputs = {str(output), str(output / "native-viewer.mov")}
    for item in command:
        if item.startswith("/") and item not in outputs and item not in bindings:
            raise ValueError(f"native invocation consumes an unbound file: {item}")
    if args.arm == "treatment":
        command.extend(("--resting-drive-intervention", repr(args.start_s), repr(args.end_s), repr(args.scale)))
    return command


def native_scene_summary(log: str) -> dict[str, Any]:
    runtime = next((line for line in log.splitlines() if line.startswith("runtime=")), "")
    terminal = next((line for line in reversed(log.splitlines()) if line.startswith("stand_terminal_state=")), "")
    summary = next((line for line in reversed(log.splitlines()) if line.startswith("resting_integrated_body=completed")), "")
    device = re.search(r"\bdevice=(.*?)\s+world_fingerprint=([0-9]+)", runtime)
    if device is None or "eligible dense45 vascular solve" not in runtime or not device[1].startswith("Apple "):
        raise ValueError("native scene did not report the physical Apple GPU/Dense45 owner")
    if not terminal or "physiology_body_clock=matched root_assistance=false" not in summary:
        raise ValueError("native scene did not complete a shared-clock unassisted body trajectory")
    state = json.loads(terminal.split("=", 1)[1])
    if state.get("root_assistance") is not False or not state.get("step_count", 0) > 0:
        raise ValueError("native body terminal state has invalid step count or root assistance")
    if not all(math.isfinite(x) for key in ("q", "v") for x in state[key]):
        raise ValueError("native terminal body state is nonfinite")
    fields = dict(re.findall(r"([A-Za-z_]+)=([^ ]+)", summary))
    values = {key: finite_float(fields[key], key) for key in ("simulated_s", "wall_s", "real_time_factor")}
    if min(values.values()) <= 0:
        raise ValueError("native execution timing must be positive")
    return {**values, "accepted_steps": state["step_count"], "device": device[1],
            "world_fingerprint": device[2], "vascular_dense45": True, "brain_control": True,
            "whole_body_anatomy_qualified": False}


def execute_native_scene_arm(args: argparse.Namespace) -> dict[str, Any]:
    validate_windows(args)
    work, output = Path.cwd().resolve(), Path(args.output).resolve()
    if output.parent != work or output.exists():
        raise ValueError("native scene output must be a new child of the notebook run directory")
    invocation_path = Path(args.invocation).resolve()
    invocation = json.loads(invocation_path.read_text())
    command = native_scene_command(invocation, output, args)
    bindings = invocation["asset_sha256"]

    def verify_bindings() -> None:
        for path, digest in bindings.items():
            if sha256_file(Path(path)) != digest:
                raise ValueError(f"preregistered native binary/library/asset changed: {path}")

    verify_bindings()
    env = os.environ.copy()
    env.update(invocation.get("environment", {}))
    if any(k.startswith("NUMI_HUMAN_STAND_CPU_") and v == "1" for k, v in env.items()):
        raise ValueError("CPU stepping is not admitted for the native scene")
    output.mkdir()
    write_json(output / "invocation.json", {**invocation, "argv": command,
               "reference_invocation_sha256": sha256_file(invocation_path)})
    log_path = output / "native.log"
    with log_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False)
    verify_bindings()
    if completed.returncode:
        raise ValueError(f"native scene failed with status {completed.returncode}; all output retained")
    log = log_path.read_text()
    native = native_scene_summary(log)
    if native["device"] != args.device or native["accepted_steps"] != args.steps:
        raise ValueError("native scene device or accepted step count differs from registration")
    # The intervention is part of each root program's identity. Match each
    # declared arm independently; pair equality applies to the common assets.
    if native["world_fingerprint"] != args.world_fingerprint:
        raise ValueError("native scene world differs from this arm's preregistered identity")
    result = observation(args, output / "resting-coupled.csv", native, log)
    movie, surfaces = output / "native-viewer.mov", output / "resting-surface-audit.csv"
    if not movie.is_file() or movie.stat().st_size == 0 or not surfaces.is_file():
        raise ValueError("native scene did not retain its continuous movie and surface trace")
    result.update(native_whole_body_executed=True,
                  common_asset_identity=digest_json(bindings),
                  recording_sha256=sha256_file(movie), surface_trace_sha256=sha256_file(surfaces),
                  reference_invocation_sha256=sha256_file(invocation_path))
    write_json(output / "intervention-observation.json", result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return result


def get_git_identity(repository: Path) -> dict[str, str]:
    def git(*argv: str) -> str:
        return subprocess.check_output(["git", "-C", str(repository), *argv], text=True).strip()
    diff = subprocess.check_output(["git", "-C", str(repository), "diff", "HEAD", "--binary"])
    return {"repository": str(repository), "revision": git("rev-parse", "HEAD"),
            "status": git("status", "--porcelain=v1", "--untracked-files=normal"),
            "diff_sha256": hashlib.sha256(diff).hexdigest()}


def prepare(args: argparse.Namespace) -> Path:
    validate_windows(args)
    repository = Path(args.repository).resolve()
    out = Path(args.directory).resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = Path(args.runner).resolve()
    frozen_binary = Path(args.frozen_runner).resolve()
    network = Path(args.network).resolve()
    parameters = Path(args.parameters).resolve()
    loaded_matter = Path(args.loaded_matter_metallib).resolve()
    frozen_matter = Path(args.frozen_matter_metallib).resolve()
    loaded_resp = Path(args.loaded_respiration_metallib).resolve()
    frozen_resp = Path(args.frozen_respiration_metallib).resolve()
    build_receipt = Path(args.build_receipt).resolve()
    fixture = Path(args.parser_fixture).resolve()
    script = Path(__file__).resolve()
    brain_root = Path(args.brain_root).resolve()
    if not binary.is_file() or not frozen_binary.is_file() or not script.is_file():
        raise ValueError("runner, frozen runner, and adapter source must be regular files")
    if sha256_file(binary) != sha256_file(frozen_binary):
        raise ValueError("frozen runner does not match the loaded executable")
    receipt = json.loads(build_receipt.read_text(encoding="utf-8"))
    if receipt.get("schema") != "numi.human-resting.build-receipt.v1":
        raise ValueError("unsupported exact build receipt schema")
    receipt_bindings = receipt.get("files", {})
    expected_receipt = {sha256_file(path) for path in
                        (frozen_binary, frozen_matter, frozen_resp, network, parameters)}
    if not expected_receipt.issubset(set(receipt_bindings.values())):
        raise ValueError("frozen executable, metallib, or config identity does not match the exact build receipt")
    network_obj, parameter_obj = json.loads(network.read_text()), json.loads(parameters.read_text())
    if network_obj.get("compartments") is None or not parameter_obj:
        raise ValueError("expected source CVSim21 network and declared respiratory reference parameters")
    source_paths = [repository / relative for relative in SOURCE_INPUTS]
    source_paths.extend((brain_root / "Sources/NumiBrainABI/include/NumiBrainRespiratoryChemoreflexV1.h",
                         brain_root / "Sources/NumiBrainMetal/Shaders/RespiratoryChemoreflexV1.metal"))
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise ValueError("source identity manifest has missing build inputs: " + ", ".join(missing))
    source_identity = {str(path): sha256_file(path) for path in source_paths}
    if receipt.get("source_files") != source_identity:
        raise ValueError("registered source snapshot differs from the build-time source hash set")
    identity = get_git_identity(repository)
    if receipt.get("owner_source") != identity:
        raise ValueError("owner source revision/dirty patch changed since the native build receipt")
    manifest = {
        "schema": "numi.human-resting.native-build-identity.v1",
        "owner_source": identity,
        "device": args.device,
        "macos": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "binary": {"loaded_path": str(binary), "sha256": sha256_file(binary),
                   "frozen_path": str(frozen_binary), "frozen_sha256": sha256_file(frozen_binary)},
        "matter_metallib": {"loaded_path": str(loaded_matter),
                            "sha256_at_registration": sha256_file(loaded_matter),
                            "expected_sha256": sha256_file(frozen_matter),
                            "frozen_path": str(frozen_matter), "frozen_sha256": sha256_file(frozen_matter),
                            "expected_at_launch": True},
        "respiration_metallib": {"loaded_path": str(loaded_resp),
                                  "sha256_at_registration": sha256_file(loaded_resp),
                                  "expected_sha256": sha256_file(frozen_resp),
                                  "frozen_path": str(frozen_resp), "frozen_sha256": sha256_file(frozen_resp),
                                  "expected_at_launch": True},
        "inputs": {str(network): sha256_file(network), str(parameters): sha256_file(parameters)},
        "parser_calibration_fixture": {str(fixture): sha256_file(fixture)},
        "source_files_at_registration": source_identity,
        "build_receipt_contents": receipt,
        "build_receipt": {str(build_receipt): sha256_file(build_receipt)},
        "configuration": {"steps": args.steps, "dt_s": args.dt,
                          "analysis_window_s": args.window_s,
                          "drive_intervention_start_s": args.start_s,
                          "drive_intervention_end_s": args.end_s,
                          "drive_intervention_scale": args.scale,
                          "solver": "existing Matter CVSim21 laws; eligible opt-in Dense45 direct linear solve",
                          "controller": "existing NumiBrain accepted respiratory chemoreflex"},
        "scope": "Reduced coupled numerical physiology instrument. It is not the anatomical whole-body scene or clinical validation.",
    }
    manifest_path = out / "frozen-build-identity.json"
    write_json(manifest_path, manifest)
    model = {
        "schema": "numi.human-resting-paco2-model.v1",
        "version": "alveolar-ventilation-envelope-v1",
        "statement": "For fixed resting CO2 production, the steady-state alveolar-ventilation relationship predicts a nonnegative PaCO2 change if effective alveolar ventilation under reduced drive remains at or below baseline. If it lies between one-half and all of baseline, the sensitivity envelope is 0 to +40 mmHg from a 40 mmHg baseline. This broad algebraic interval is not a probability interval or clinical prediction; closed-loop finite-time behavior is measured rather than assumed.",
        "parameters": {"baseline_PaCO2_mmhg": 40.0,
                       "minimum_effective_alveolar_ventilation_fraction": 0.5,
                       "maximum_effective_alveolar_ventilation_fraction": 1.0,
                       "assumption": "for the sensitivity envelope only, delivered drive scaling and closed-loop feedback leave effective alveolar ventilation in [0.5, 1.0] of baseline; the intervention result is measured from the accepted native state"},
        "training_units": [],
        "scope": "One deterministic reduced-order simulation condition. Bounds are algebraic sensitivity to the declared ventilation range, not a statistical interval, calibration, physiological validation, or clinical claim.",
    }
    model_path = out / "model.json"
    write_json(model_path, model)
    calibration = known_parser_calibration(fixture)
    instrument_artifacts = [str(path) for path in [script, binary, frozen_binary,
        loaded_matter, frozen_matter, loaded_resp, frozen_resp, network, parameters,
        fixture, build_receipt, manifest_path, *source_paths]]
    calibration["bindings"] = {path: sha256_file(Path(path)) for path in instrument_artifacts}
    calibration["evidence"] = {str(fixture): sha256_file(fixture)}
    calibration_path = out / "calibration.json"
    write_json(calibration_path, calibration)
    unit_input = {"network_sha256": sha256_file(network), "parameters_sha256": sha256_file(parameters),
                  "initialization_source_sha256": source_identity[str(repository / "matter/src/human_physiology.mm")],
                  "dt_s": args.dt, "steps": args.steps, "device": args.device}
    unit_id = "resting-reference-initialization-v1-" + digest_json(unit_input)[:20]
    py = str(Path(sys.executable).resolve())
    common = ["--runner", str(frozen_binary), "--network", str(network), "--parameters", str(parameters),
              "--loaded-matter-metallib", str(loaded_matter), "--frozen-matter-metallib", str(frozen_matter),
              "--loaded-respiration-metallib", str(loaded_resp), "--frozen-respiration-metallib", str(frozen_resp),
              "--unit-id", unit_id, "--world-fingerprint", args.world_fingerprint,
              "--device", args.device,
              "--steps", str(args.steps), "--dt", repr(args.dt), "--start-s", repr(args.start_s),
              "--end-s", repr(args.end_s), "--scale", repr(args.scale), "--window-s", repr(args.window_s)]
    trials = [
        {"id": "reference-rest-baseline", "pair": "reference-rest-v1", "arm": "control",
         "unit": {"unit_id": unit_id},
         "argv": [py, str(script), "run", *common, "--arm", "control", "--output", "{run}/native.csv"],
         "env": {}, "timeout_seconds": 86400},
        {"id": "reference-rest-drive-half", "pair": "reference-rest-v1", "arm": "treatment",
         "unit": {"unit_id": unit_id},
         "argv": [py, str(script), "run", *common, "--arm", "treatment", "--output", "{run}/native.csv"],
         "env": {}, "timeout_seconds": 86400},
    ]
    artifacts = list(dict.fromkeys([*instrument_artifacts, str(model_path), str(calibration_path)]))
    plan = {
        "schema": "numi.science.plan.v2", "purpose": "exploration",
        "question": f"In the existing coupled reduced-order Matter/NumiBrain resting physiology instrument, does scaling delivered respiratory excitation to {args.scale:g} on [{args.start_s:g}, {args.end_s:g}) s reduce inspiratory ventilation and raise PaCO2, with PaO2 and same-state recovery diagnostics moving toward the unchanged baseline?",
        "hypothesis": model["statement"], "owner": "Numi Human/Matter coupled respiratory and circulation instrument",
        "repository": str(repository), "backend": "Apple Metal on Apple M4 Pro; existing Matter CVSim21 plus coupled respiration and NumiBrain chemoreflex",
        "evidence_level": "simulation", "model": model, "model_file": str(model_path),
        "predictor": {"argv": [py, str(script), "predict", "--model", str(model_path)], "env": {}, "timeout_seconds": 30},
        "instrument": {"description": f"Native accepted-state trace adapter: fixed {args.window_s:g} s PaCO2 windows; inspiratory minute ventilation integrated from accepted airflow, with additional complete-breath windows; PaO2/SaO2 and late recovery diagnostics. Parser calibration is known-value/arithmetic only.",
                       "calibration": str(calibration_path), "artifacts": instrument_artifacts},
        "artifacts": artifacts,
        "design": {"intervention": f"Existing NumiBrain respiratory drive continues to sense and regulate; only its delivered diaphragm/intercostal excitation is multiplied by {args.scale:g} on [{args.start_s:g},{args.end_s:g}) s.",
                   "pre_dose_window_s": [args.start_s - args.window_s, args.start_s],
                   "dose_window_s": [args.end_s - args.window_s, args.end_s],
                   "recovery_window_s": [args.steps * args.dt - args.window_s, args.steps * args.dt],
                   "controls": "Matched fresh process from the same deterministic resting reference initialization, same compiled native executable/libraries, network, physiology parameters, device and timing; scale remains 1.0 throughout control.",
                   "experimental_unit": "One deterministic accepted-state initialization and fixed source/configuration identity; the paired arms are trajectories from the same initial state, not independent people or independent ticks.",
                   "allocation": "One exploratory pair, control then treatment, one run per arm; no retries or exclusion; all reads are from native accepted traces.",
                   "unit_paths": {"unit_id": ["unit_id"]}},
        "observable": {"name": f"paired difference-in-differences in PaCO2 from the {args.window_s:g} s pre-dose window to the last {args.window_s:g} s of the drive intervention", "unit": "mmHg", "path": ["primary_delta_PaCO2_mmhg"]},
        "prediction": model_prediction(model),
        "validity": [{"path": ["schema"], "equals": "numi.human-resting.intervention-observation.v1"},
                     {"path": ["accepted_steps"], "equals": args.steps},
                     {"path": ["nominal_duration_s"], "equals": args.steps * args.dt},
                     {"path": ["duration_valid"], "equals": True},
                     {"path": ["dense45"], "equals": True}, {"path": ["brain_control"], "equals": True},
                     {"path": ["whole_body_anatomy_qualified"], "equals": False}],
        "paired_equal": [["unit_id"], ["device"], ["world_fingerprint"], ["timestep_s"], ["accepted_steps"], ["dense45"], ["brain_control"]],
        "trials": trials,
        "limitations": "The primary effect is one deterministic paired difference-in-differences, not a sample/population estimate. Secondary ventilation, oxygenation, and recovery values are descriptive fields in each retained trial output. The alveolar ventilation equation supplies a steady-state sensitivity envelope; finite-time closed-loop behavior may differ. Parser calibration does not calibrate physiology or clinical measurement. The current runner explicitly reports no qualified whole-body anatomy/resting claim; this study does not establish anatomical mechanics or clinical validity.",
    }
    plan_path = out / "plan.json"
    write_json(plan_path, plan)
    print(plan_path)
    return plan_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    predict = sub.add_parser("predict", help="run the v2 executable model")
    predict.add_argument("--model", required=True)
    run = sub.add_parser("run", help="execute one declared native owner arm")
    run.add_argument("--runner", required=True)
    run.add_argument("--network", required=True)
    run.add_argument("--parameters", required=True)
    run.add_argument("--loaded-matter-metallib", required=True)
    run.add_argument("--frozen-matter-metallib", required=True)
    run.add_argument("--loaded-respiration-metallib", required=True)
    run.add_argument("--frozen-respiration-metallib", required=True)
    run.add_argument("--unit-id", required=True)
    run.add_argument("--world-fingerprint", required=True)
    run.add_argument("--device", default="Apple M4 Pro")
    run.add_argument("--steps", type=int, required=True)
    run.add_argument("--dt", type=float, required=True)
    run.add_argument("--arm", choices=("control", "treatment"), required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--start-s", type=float, required=True)
    run.add_argument("--end-s", type=float, required=True)
    run.add_argument("--scale", type=float, required=True)
    run.add_argument("--window-s", type=float, default=5.0)
    native_run = sub.add_parser("run-native", help="execute the existing anatomical native viewer from its frozen launch receipt")
    native_run.add_argument("--invocation", required=True)
    native_run.add_argument("--unit-id", required=True)
    native_run.add_argument("--world-fingerprint", required=True, help="expected fingerprint for this specific arm")
    native_run.add_argument("--device", default="Apple M4 Pro")
    native_run.add_argument("--steps", type=int, required=True)
    native_run.add_argument("--dt", type=float, required=True)
    native_run.add_argument("--arm", choices=("control", "treatment"), required=True)
    native_run.add_argument("--output", required=True, help="new scene directory inside the notebook run directory")
    native_run.add_argument("--start-s", type=float, required=True)
    native_run.add_argument("--end-s", type=float, required=True)
    native_run.add_argument("--scale", type=float, required=True)
    native_run.add_argument("--window-s", type=float, default=30.0)
    prep = sub.add_parser("prepare", help="write the frozen identity, parser calibration, model, and v2 plan")
    prep.add_argument("--repository", required=True)
    prep.add_argument("--directory", required=True)
    prep.add_argument("--runner", required=True)
    prep.add_argument("--frozen-runner", required=True)
    prep.add_argument("--network", required=True)
    prep.add_argument("--parameters", required=True)
    prep.add_argument("--loaded-matter-metallib", required=True)
    prep.add_argument("--frozen-matter-metallib", required=True)
    prep.add_argument("--loaded-respiration-metallib", required=True)
    prep.add_argument("--frozen-respiration-metallib", required=True)
    prep.add_argument("--parser-fixture", required=True)
    prep.add_argument("--build-receipt", required=True)
    prep.add_argument("--brain-root", required=True)
    prep.add_argument("--world-fingerprint", required=True)
    prep.add_argument("--device", default="Apple M4 Pro")
    prep.add_argument("--steps", type=int, default=150000)
    prep.add_argument("--dt", type=float, default=0.002)
    prep.add_argument("--start-s", type=float, default=120.0)
    prep.add_argument("--end-s", type=float, default=150.0)
    prep.add_argument("--scale", type=float, default=0.5)
    prep.add_argument("--window-s", type=float, default=5.0)
    receipt = sub.add_parser("receipt", help="capture exact source, binary, library, input, and repository identity after a native build")
    receipt.add_argument("--repository", required=True)
    receipt.add_argument("--brain-root", required=True)
    receipt.add_argument("--runner", required=True)
    receipt.add_argument("--matter-metallib", required=True)
    receipt.add_argument("--respiration-metallib", required=True)
    receipt.add_argument("--network", required=True)
    receipt.add_argument("--parameters", required=True)
    receipt.add_argument("--build-description", required=True)
    receipt.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "predict":
            model_path = Path(args.model).resolve()
            model = json.loads(model_path.read_text())
            print(json.dumps({"schema": "numi.science.prediction.v1", "model_sha256": sha256_file(model_path),
                              "prediction": model_prediction(model)}, sort_keys=True, separators=(",", ":"), allow_nan=False))
        elif args.command == "run":
            execute_arm(args)
        elif args.command == "run-native":
            execute_native_scene_arm(args)
        elif args.command == "receipt":
            repository = Path(args.repository).resolve()
            brain_root = Path(args.brain_root).resolve()
            paths = [repository / relative for relative in SOURCE_INPUTS]
            paths.extend((brain_root / "Sources/NumiBrainABI/include/NumiBrainRespiratoryChemoreflexV1.h",
                          brain_root / "Sources/NumiBrainMetal/Shaders/RespiratoryChemoreflexV1.metal"))
            paths.extend(Path(value).resolve() for value in (args.runner, args.matter_metallib,
                           args.respiration_metallib, args.network, args.parameters))
            missing = [str(path) for path in paths if not path.is_file()]
            if missing:
                raise ValueError("build receipt has missing inputs: " + ", ".join(missing))
            receipt_value = {
                "schema": "numi.human-resting.build-receipt.v1",
                "owner_source": get_git_identity(repository),
                "device": "Apple M4 Pro",
                "build_description": args.build_description,
                "files": {str(path): sha256_file(path) for path in paths},
                "source_files": {str(path): sha256_file(path) for path in paths[:len(SOURCE_INPUTS) + 2]},
            }
            receipt_path = Path(args.output).resolve()
            write_json(receipt_path, receipt_value)
            print(receipt_path)
        else:
            prepare(args)
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"resting_intervention_study=failed reason={exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
