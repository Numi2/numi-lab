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

BREATH_LEDGER_COLUMNS = (
    "inspired_volume_accum_ml", "last_inspiration_step", "last_inspiration_time_s",
    "last_inspiration_volume_accum_ml", "last_complete_breath_inspired_ml",
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
        ledger_columns = set(BREATH_LEDGER_COLUMNS).intersection(reader.fieldnames)
        if ledger_columns and ledger_columns != set(BREATH_LEDGER_COLUMNS):
            raise ValueError("native trace contains an incomplete accepted breath ledger")
        columns = TRACE_COLUMNS + (BREATH_LEDGER_COLUMNS if ledger_columns else ())
        if "step" in reader.fieldnames:
            columns += ("step",)
        rows: list[dict[str, float]] = []
        previous = -math.inf
        for line_number, raw in enumerate(reader, start=2):
            row = {column: finite_float(raw[column], f"{column} line {line_number}")
                   for column in columns}
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

    The owner increments `breaths` at a resolved start of inspiration and retains
    its accepted-step event time and inspired volume. Legacy traces fall back to
    interpolation when retained airflow samples resolve every crossing. Neither
    path infers prescribed frequency or uses requested ventilation.
    """
    if rows and all(key in rows[0] for key in BREATH_LEDGER_COLUMNS):
        # The GPU records event roots and cumulative inspiration before the
        # accepted-state gate. A whole positive-flow excursion can occur
        # between two retained display samples; the event ledger preserves
        # its count and volume without inventing a flow crossing from those
        # coarse endpoints. The GPU observer uses explicit flow hysteresis to
        # exclude numerical sign chatter; no breath is removed by this analysis.
        boundaries: list[tuple[float, float, float]] = []
        previous_count = -1.0
        previous_time = previous_volume = 0.0
        previous_sample_time = -math.inf
        for row in rows:
            count, step = row["breaths"], row["last_inspiration_step"]
            time, volume = row["last_inspiration_time_s"], row["last_inspiration_volume_accum_ml"]
            if (count < 0 or count != int(count) or step < 0 or step != int(step) or
                    time < 0 or time > row["time_s"] + 1e-7 or volume < 0 or
                    volume > row["inspired_volume_accum_ml"] + 1e-6 or
                    count < previous_count or (count == 0 and (step != 0 or time != 0 or volume != 0))):
                raise ValueError("invalid accepted breath event ledger")
            if count == previous_count and (time != previous_time or volume != previous_volume):
                raise ValueError("accepted breath event changed without a counter transition")
            if count > previous_count and count > 0:
                if step == 0 or time <= previous_sample_time or volume < previous_volume:
                    raise ValueError("accepted breath event ledger did not advance with its source step")
                if start_s <= time <= end_s:
                    boundaries.append((count, time, volume))
            previous_count, previous_time, previous_volume = count, time, volume
            previous_sample_time = row["time_s"]
        if len(boundaries) < 2:
            return {"available": False, "complete_breath_count": 0,
                    "method": "accepted_gpu_event_ledger",
                    "reason": "fewer than two accepted inspiratory boundaries inside the analysis window"}
        first, last = boundaries[0], boundaries[-1]
        cycles, seconds = int(last[0] - first[0]), last[1] - first[1]
        return {"available": True, "complete_breath_count": cycles,
                "method": "accepted_gpu_event_ledger", "window_s": [first[1], last[1]],
                "events_between_retained_samples": cycles - (len(boundaries) - 1),
                "inspiratory_minute_ventilation_L_min": (last[2] - first[2]) * 60 / (1000 * seconds),
                "respiratory_rate_per_min": 60 * cycles / seconds}
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


def resting_reference_comparison(rows: list[dict[str, float]], start_s: float, end_s: float) -> dict[str, Any]:
    """Descriptive source comparisons, kept separate from numerical admission."""
    samples = window_rows(rows, start_s, end_s)
    elapsed = samples[-1]["time_s"] - samples[0]["time_s"]
    aacn = "https://aacn.s3-us-west-2.amazonaws.com/Courses/ecco/course-resources/resources/common-resources/Normal_Ranges.pdf"
    vital = "https://medlineplus.gov/ency/article/002341.htm"
    cohort = "https://pmc.ncbi.nlm.nih.gov/articles/PMC7253877/"
    comparisons: dict[str, Any] = {}

    def compare(name: str, value: float, bounds: list[float | None], source: str,
                method: str, strict_lower: bool = False) -> None:
        lower, upper = bounds
        inside = ((lower is None or (value > lower if strict_lower else value >= lower)) and
                  (upper is None or value <= upper))
        comparisons[name] = {"measured": value, "reference_bounds": bounds,
                             "lower_bound_inclusive": not strict_lower,
                             "within_reference_bounds": inside, "source": source,
                             "measurement_method": method}

    # These are sampled observations and integrated owner volumes, not
    # independently generated vital-sign curves or a pass/fail simulator gate.
    for name, column, bounds in (("mean_PaCO2_mmhg", "PaCO2_mmhg", [35.0, 45.0]),
                                 ("mean_PaO2_mmhg", "PaO2_mmhg", [80.0, 100.0]),
                                 ("mean_aortic_pressure_mmhg", "aorta_mmhg", [70.0, 105.0]),
                                 ("mean_pulmonary_artery_pressure_mmhg", "pulmonary_artery_mmhg", [15.0, 20.0]),
                                 ("mean_last_complete_LV_stroke_ml", "last_lv_stroke_ml", [50.0, 100.0])):
        compare(name, mean(samples, column), bounds, aacn, "mean of retained accepted samples")
    compare("mean_SaO2_percent", 100 * mean(samples, "SaO2"), [95.0, None], aacn,
            "mean of retained accepted samples", strict_lower=True)
    for name, column in (("aortic_output_L_min", "aortic_ejected_ml"),
                         ("pulmonary_output_L_min", "pulmonary_ejected_ml")):
        value = (samples[-1][column] - samples[0][column]) * 60 / (1000 * elapsed)
        compare(name, value, [4.0, 8.0], aacn, "accepted cumulative forward flow difference / actual sample interval")

    heart_boundaries = [right for left, right in zip(rows, rows[1:])
                        if start_s <= right["time_s"] < end_s and
                        right["complete_filling_ejection_cycles"] > left["complete_filling_ejection_cycles"]]
    if len(heart_boundaries) >= 2:
        first, last = heart_boundaries[0], heart_boundaries[-1]
        rate = 60 * (last["complete_filling_ejection_cycles"] - first["complete_filling_ejection_cycles"]) / (last["time_s"] - first["time_s"])
        compare("complete_heartbeat_rate_per_min", rate, [60.0, 100.0], vital,
                "accepted complete-cycle count between first/last retained cycle transitions; timing limited by retained cadence")
    breaths = complete_breath_metrics(rows, start_s, end_s)
    if breaths["available"]:
        compare("complete_breath_rate_per_min", breaths["respiratory_rate_per_min"], [12.0, 18.0], vital,
                breaths.get("method", "resolved accepted airflow crossings"))
    cohort_values = {
        "tidal_volume_L": {"measured": mean(samples, "tidal_ml") / 1000, "mean": 0.58, "sd": 0.28},
        "minute_ventilation_L_min": {"measured": breaths.get("inspiratory_minute_ventilation_L_min"), "mean": 8.32, "sd": 2.78},
        "breaths_per_min": {"measured": breaths.get("respiratory_rate_per_min"), "mean": 16.15, "sd": 4.72},
    }
    return {"window_s": [start_s, end_s], "accepted_sample_interval_s": [samples[0]["time_s"], samples[-1]["time_s"]],
            "general_adult_resting_reference_comparisons": comparisons,
            "supine_male_cohort_context": {"source": cohort, "source_location": "Table 2, men, supine",
                                          "values": cohort_values,
                                          "scope": "Published cohort mean/SD, not normal bounds or a clinical gate. OEP chest-wall tidal volume is a related measurement, not identical to model airway volume."},
            "supine_pulmonary_pressure_review_context": {
                "source": "https://pubmed.ncbi.nlm.nih.gov/19324955/",
                "doi": "10.1183/09031936.00145608",
                "source_location": "Kovacs et al. 2009, abstract, healthy subjects at rest in the supine position",
                "mean_pulmonary_artery_pressure_mmhg": {
                    "measured": mean(samples, "pulmonary_artery_mmhg"), "mean": 14.0, "sd": 3.3},
                "scope": "Published right-heart-catheterization review mean/SD, not individual normal bounds or a clinical gate. This posture-specific context supplements the unchanged AACN comparison; it does not replace a failed comparison or tune the model."},
            "limitations": "General reference intervals depend on age, altitude and measurement method. Comparisons describe model outputs; they do not establish anatomical validity, numerical conservation, population generalization or clinical validation. Missing complete-cycle rates remain unavailable."}


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
    recovery = window_metrics(rows, expected_seconds - width, expected_seconds)
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
        "resting_reference_comparisons": {
            "pre": resting_reference_comparison(rows, args.start_s - width, args.start_s),
            "dose": resting_reference_comparison(rows, args.end_s - width, args.end_s),
            "recovery": resting_reference_comparison(rows, expected_seconds - width, expected_seconds),
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


def native_scene_runtime_dependency_bindings(invocation: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve build-layout dependencies to the exact invocation-bound files."""
    command = invocation.get("argv")
    bindings = invocation.get("asset_sha256")
    if not isinstance(command, list) or not command or not isinstance(bindings, dict):
        raise ValueError("native invocation lacks argv or asset hash bindings")
    binary = Path(str(command[0]))
    if binary.parent.name != "bin":
        raise ValueError("native invocation does not use the Human owner build layout")
    build = binary.parent.parent
    relatives = ("lib/libmetalrobo.dylib", "shaders/MetalRobo.metallib",
                 "shaders/MetalRoboHyperPolicy.metallib", "shaders/NumiNeuron.metallib",
                 "matter/shaders/HumanRespiration.metallib", "matter/shaders/NumiMatter.metallib",
                 "matter/shaders/NumiMatterPhysicalStateDigest.metallib")
    records = []
    for relative in relatives:
        configured = build / relative
        if not configured.is_file():
            raise ValueError(f"native runtime dependency is missing: {configured}")
        try:
            target = configured.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"native runtime dependency target cannot be resolved: {configured}") from exc
        target_path = str(target)
        components = [build]
        for component in Path(relative).parts:
            components.append(components[-1] / component)
        symlink_components = [str(component) for component in components if component.is_symlink()]
        requires_resolved_binding = bool(symlink_components)
        bound_path = (target_path if requires_resolved_binding or target_path in bindings
                      else str(configured))
        expected = bindings.get(bound_path)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"native runtime dependency target is not hash-bound: {configured} -> {target}")
        actual = sha256_file(target)
        if actual != expected:
            raise ValueError(f"native runtime dependency target hash changed: {configured} -> {target}")
        records.append({"configured_path": str(configured), "resolved_path": target_path,
                        "asset_binding_path": bound_path,
                        "is_symlink": configured.is_symlink(),
                        "symlink_components": symlink_components, "sha256": actual})
    return records


def verify_native_scene_runtime_dependency_resolution(
        invocation: dict[str, Any],
        expected: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    records = native_scene_runtime_dependency_bindings(invocation)
    if expected is not None and records != expected:
        raise ValueError("native runtime dependency symlink target changed since preflight")
    return records


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
    # These dependencies are loaded by the native executable, not named in
    # argv. Match the existing Human launcher's build layout so a manually
    # assembled reference cannot silently omit a mutable runtime library.
    verify_native_scene_runtime_dependency_resolution(invocation)
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
    program = re.search(r"resting_body_source_fingerprint=([0-9]+) coupled_program_fingerprint=([0-9]+)", log)
    if device is None or "eligible dense45 vascular solve" not in runtime or not device[1].startswith("Apple "):
        raise ValueError("native scene did not report the physical Apple GPU/Dense45 owner")
    if not terminal or "physiology_body_clock=matched root_assistance=false" not in summary:
        raise ValueError("native scene did not complete a shared-clock unassisted body trajectory")
    if program is None or min(int(program[1]), int(program[2])) == 0:
        raise ValueError("native scene did not report its coupled body/controller program identity")
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
            "body_source_fingerprint": program[1], "coupled_program_fingerprint": program[2],
            "whole_body_anatomy_qualified": False}


def native_scene_environment(recorded: dict[str, str], ambient: dict[str, str]) -> dict[str, str]:
    """Execute the recorded native settings without inherited solver overrides."""
    env = {key: value for key, value in ambient.items()
           if not key.startswith(("NUMI_", "DYLD_"))}
    env.update(recorded)
    if any(key.startswith("NUMI_HUMAN_STAND_CPU_") and
           (value == "1" or key == "NUMI_HUMAN_STAND_CPU_ACCELERATE_FACTOR")
           for key, value in env.items()):
        raise ValueError("CPU stepping is not admitted for the native scene")
    return env


def native_cycle_coverage(rows: list[dict[str, float]], dt: float) -> dict[str, Any]:
    """Require actual repeated breaths/ejection during 300 s after initialization."""
    initialization_step = round(10.0 / dt)
    start = next((row for row in rows if int(row["step"]) == initialization_step), None)
    if start is None:
        raise ValueError("native trace lacks the accepted initialization boundary")
    duration = rows[-1]["time_s"] - start["time_s"]
    if duration < 300.0 - 1e-5:
        raise ValueError("native trace lacks 300 seconds after initialization")
    breaths = complete_breath_metrics(rows, start["time_s"], rows[-1]["time_s"])
    if not breaths["available"] or breaths["complete_breath_count"] < 2:
        raise ValueError("native trace lacks repeated complete post-initialization breaths")
    after = [start, *[row for row in rows if row["step"] > initialization_step]]
    cycles = after[-1]["complete_filling_ejection_cycles"] - start["complete_filling_ejection_cycles"]
    if cycles < 2 or cycles != int(cycles):
        raise ValueError("native trace lacks repeated complete post-initialization heart cycles")
    for left, right in zip(after, after[1:]):
        change = right["complete_filling_ejection_cycles"] - left["complete_filling_ejection_cycles"]
        if change < 0 or change != int(change):
            raise ValueError("native cardiac cycle counter regressed or became fractional")
        if change and right["last_lv_stroke_ml"] <= 0:
            raise ValueError("native completed cardiac cycle has no positive stroke volume")
        for field in ("aortic_ejected_ml", "pulmonary_ejected_ml"):
            if right[field] < left[field]:
                raise ValueError("native cumulative forward ejection regressed")
    ejection = {field: after[-1][field] - start[field]
                for field in ("aortic_ejected_ml", "pulmonary_ejected_ml")}
    if min(ejection.values()) <= 0:
        raise ValueError("native heart did not eject into both circulations after initialization")
    return {"post_initialization_cycle_coverage": {
        "passed": True, "observed_seconds": duration,
        "initialization_boundary_step": initialization_step,
        "complete_breath_count": breaths["complete_breath_count"],
        "complete_filling_ejection_cycles": int(cycles),
        **ejection,
        "scope": "Numerical duration and actual cycle/flow coverage; not physiological or anatomical validation."}}


def native_body_trace_consistency(trace: Path, steps: int, dt: float) -> dict[str, Any]:
    """Check the body fields on every retained accepted observation."""
    previous_step = 0
    maximum_penetration = 0.0
    with trace.open(newline="") as stream:
        for row in csv.DictReader(stream):
            step = int(row["step"])
            if not 0 < step - previous_step <= 32:
                raise ValueError("native trace skipped an accepted observation segment")
            if abs(finite_float(row["time_s"], "time_s") - step * dt) > max(1e-5, step * dt * 1e-7):
                raise ValueError("body and physiological observation times disagree")
            for key in ("root_assistance_n", "root_assistance_nm"):
                if finite_float(row[key], key) != 0:
                    raise ValueError("native body trace contains root assistance")
            maximum_penetration = max(maximum_penetration, finite_float(row["peak_penetration_m"], "peak_penetration_m"))
            previous_step = step
    if previous_step != steps:
        raise ValueError("native body trace does not reach the declared final accepted step")
    return {"root_assistance_observed": False, "maximum_contact_penetration_m": maximum_penetration}


def native_surface_trace_consistency(trace: Path, steps: int, dt: float, *,
                                        require_whole_mesh: bool = False,
                                        presentation_period_s: float = 0.064,
                                        terminal_accepted_capture: bool = False) -> dict[str, Any]:
    """Validate pre-step displayed states and, when requested, the exact terminal accepted capture.

    The renderer captures step n-1 before step n is evaluated and publishes it
    only after that step is accepted. Never pair this trace with the next
    physiology row: its own chamber/lung target columns are authoritative.
    The optional final step N row is a separate query-only accepted capture.
    These checks cover numerical geometry consistency, not tissue interfaces.
    """
    frame_interval = round(presentation_period_s / dt)
    if frame_interval <= 0 or abs(frame_interval * dt - presentation_period_s) > max(1e-9, presentation_period_s * 1e-7):
        raise ValueError("native surface cadence is not an integer number of native steps")
    expected_steps = [0] + [step for step in range(frame_interval - 1, steps, frame_interval) if step != 0]
    if expected_steps[-1] != steps - 1:
        expected_steps.append(steps - 1)
    if terminal_accepted_capture:
        expected_steps.append(steps)
    minimum_gap, maximum_volume_error, count = math.inf, 0.0, 0
    body_columns = {"body_com_x_m", "body_com_y_m", "body_com_z_m", "represented_body_mass_kg"}
    body_first, body_last, body_mass = None, None, None
    wall_columns = {"ventricular_wall_bound", "ventricular_material_ml", "ventricular_material_target_ml",
                    "ventricular_closure_mm"}
    wall_bound, wall_target = None, None
    wall_error, wall_min_closure, wall_max_closure = 0.0, math.inf, -math.inf
    geometry_mode, maximum_common_residual, mesh_triangles = None, 0.0, None
    common_coordinates = tuple("common_coordinate_" + name for name in
                               ("RA", "RV", "LA", "LV", "RA_material",
                                "ventricular_material", "LA_material"))
    common_columns = set(common_coordinates) | {
        "common_coordinate_solver_status", "common_coordinate_solver_iterations",
        "common_coordinate_domain_box", "common_coordinate_normalized_residual",
        "ventricular_closure_mm_applicable", "ventricular_material_status",
        "functional_geometry_status"}
    mesh_columns = {"mesh_zero_area_triangles", "mesh_nonfinite_area_triangles",
                    "mesh_triangles_checked"}
    with trace.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"step", "time_s", "min_skin_bed_gap_m", "vertices_below_1mm", "nonfinite_skin_vertices",
                    "max_functional_volume_relative_error", "q_ra", "q_rv", "q_la", "q_lv",
                    "diaphragm_swept_ml", "rib_swept_ml", "ra_target_ml", "rv_target_ml",
                    "la_target_ml", "lv_target_ml", "lung_target_ml"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("native surface trace lacks same-frame geometry target columns")
        present_body_columns = body_columns.intersection(reader.fieldnames or [])
        if present_body_columns and present_body_columns != body_columns:
            raise ValueError("native surface trace has an incomplete body mass/COM diagnostic")
        present_wall_columns = wall_columns.intersection(reader.fieldnames or [])
        if present_wall_columns and present_wall_columns != wall_columns:
            raise ValueError("native surface trace has an incomplete ventricular material diagnostic")
        present_mesh_columns = mesh_columns.intersection(reader.fieldnames or [])
        if require_whole_mesh and present_mesh_columns != mesh_columns:
            raise ValueError("native surface trace lacks the required whole-mesh area audit")
        if require_whole_mesh and "functional_geometry_status" not in (reader.fieldnames or []):
            raise ValueError("native surface trace lacks the required functional-geometry audit status")
        if present_mesh_columns and present_mesh_columns != mesh_columns:
            raise ValueError("native surface trace has an incomplete whole-mesh audit")
        wall_status_present = "ventricular_material_status" in (reader.fieldnames or [])
        if wall_status_present and not present_wall_columns:
            raise ValueError("native surface trace material status lacks its ventricular diagnostic")
        for row in reader:
            mode = row.get("geometry_mode", "legacy_ventricular_wall_v2")
            if mode not in ("legacy_ventricular_wall_v2", "common_seven_coordinate_v1"):
                raise ValueError("native surface trace has an unknown cardiac geometry mode")
            if geometry_mode is not None and mode != geometry_mode:
                raise ValueError("native surface trace cardiac geometry mode changes")
            geometry_mode = mode
            common_geometry = mode == "common_seven_coordinate_v1"
            if common_geometry:
                if not (common_columns | wall_columns).issubset(reader.fieldnames or []):
                    raise ValueError("native surface trace lacks common cardiac geometry diagnostics")
                for key in common_coordinates:
                    finite_float(row[key], key)
                if finite_float(row["common_coordinate_solver_status"], "common solver status") != 0:
                    raise ValueError("native surface trace common cardiac solve failed")
                for key in ("common_coordinate_solver_iterations", "common_coordinate_domain_box"):
                    value = finite_float(row[key], key)
                    if value != int(value) or not 0 <= value < 4294967295:
                        raise ValueError("native surface trace has invalid common solve provenance")
                residual = finite_float(row["common_coordinate_normalized_residual"], "common residual")
                # NumiHumanRestingAnatomy bounds this owner solver tolerance at 2e-5.
                if not 0 <= residual <= 2e-5:
                    raise ValueError("native surface trace exceeds the common solver tolerance")
                maximum_common_residual = max(maximum_common_residual, residual)
                if finite_float(row["ventricular_closure_mm_applicable"], "closure applicable") != 0:
                    raise ValueError("common cardiac geometry incorrectly declares legacy closure")
                # The native owner deliberately emits NaN for inapplicable legacy
                # coordinates/closure. This exemption never applies to active state.
                for key in ("q_ra", "q_rv", "q_la", "q_lv", "ventricular_closure_mm"):
                    if not math.isnan(float(row[key])):
                        raise ValueError("common cardiac geometry lacks its legacy NaN sentinel")
            else:
                for key in ("q_ra", "q_rv", "q_la", "q_lv"):
                    finite_float(row[key], key)
                if ("ventricular_closure_mm_applicable" in row and
                        finite_float(row["ventricular_closure_mm_applicable"], "closure applicable") != 1):
                    raise ValueError("legacy cardiac geometry incorrectly disables closure")
            if present_mesh_columns:
                for key in ("mesh_zero_area_triangles", "mesh_nonfinite_area_triangles"):
                    if finite_float(row[key], key) != 0:
                        raise ValueError("native surface trace contains invalid whole-mesh triangles")
                triangles = finite_float(row["mesh_triangles_checked"], "mesh triangles")
                if (triangles != int(triangles) or triangles <= 0 or
                        (mesh_triangles is not None and triangles != mesh_triangles)):
                    raise ValueError("native surface trace whole-mesh audit count changes or is invalid")
                mesh_triangles = int(triangles)
            step = int(row["step"])
            if count >= len(expected_steps) or step != expected_steps[count]:
                raise ValueError("native surface trace skipped or duplicated a displayed accepted step")
            if abs(finite_float(row["time_s"], "surface time_s") - step * dt) > max(1e-5, step * dt * 1e-7):
                raise ValueError("native surface trace clock differs from its displayed accepted step")
            for key in ("vertices_below_1mm", "nonfinite_skin_vertices"):
                if finite_float(row[key], key) != 0:
                    raise ValueError("native surface trace contains invalid skin/bed geometry")
            if "functional_geometry_status" in row and finite_float(row["functional_geometry_status"], "functional_geometry_status") != 0:
                raise ValueError("native surface trace contains invalid functional triangles or volumes")
            gap = finite_float(row["min_skin_bed_gap_m"], "min_skin_bed_gap_m")
            error = finite_float(row["max_functional_volume_relative_error"], "max_functional_volume_relative_error")
            if gap < -.001 or not 0 <= error <= 2e-4:
                raise ValueError("native surface trace exceeds the existing GPU geometry tolerance")
            for key in ("diaphragm_swept_ml", "rib_swept_ml"):
                finite_float(row[key], key)
            for key in ("ra_target_ml", "rv_target_ml", "la_target_ml", "lv_target_ml", "lung_target_ml"):
                if finite_float(row[key], key) <= 0:
                    raise ValueError("native surface trace contains a nonpositive functional volume")
            if present_body_columns:
                center = [finite_float(row[f"body_com_{axis}_m"], f"body_com_{axis}_m") for axis in "xyz"]
                mass = finite_float(row["represented_body_mass_kg"], "represented_body_mass_kg")
                if mass <= 0 or (body_mass is not None and abs(mass - body_mass) > 1e-5):
                    raise ValueError("native surface trace body mass is nonpositive or changes")
                if body_first is None:
                    body_first, body_mass = center, mass
                body_last = center
            if present_wall_columns:
                if wall_status_present and finite_float(row["ventricular_material_status"], "ventricular_material_status") != 0:
                    raise ValueError("native surface trace contains a degenerate ventricular triangle or failed material volume check")
                bound = finite_float(row["ventricular_wall_bound"], "ventricular_wall_bound")
                material = finite_float(row["ventricular_material_ml"], "ventricular_material_ml")
                target = finite_float(row["ventricular_material_target_ml"], "ventricular_material_target_ml")
                closure = None if common_geometry else finite_float(row["ventricular_closure_mm"], "ventricular_closure_mm")
                if bound not in (0, 1) or (wall_bound is not None and bound != wall_bound):
                    raise ValueError("native surface trace ventricular binding changes or is invalid")
                wall_bound = bound
                if common_geometry and bound != 0:
                    raise ValueError("common cardiac geometry incorrectly binds the legacy wall")
                if bound or common_geometry:
                    if target <= 0 or material <= 0 or (wall_target is not None and abs(target - wall_target) > 1e-5):
                        raise ValueError("native surface trace ventricular material target is nonpositive or changes")
                    relative = abs(material - target) / target
                    if relative > 2e-4:
                        raise ValueError("native surface trace ventricular material exceeds the GPU volume tolerance")
                    wall_target, wall_error = target, max(wall_error, relative)
                    if closure is not None:
                        wall_min_closure, wall_max_closure = min(wall_min_closure, closure), max(wall_max_closure, closure)
                elif material != 0 or target != 0 or closure != 0:
                    raise ValueError("native surface trace unbound ventricular wall reports material state")
            minimum_gap = min(minimum_gap, gap)
            maximum_volume_error = max(maximum_volume_error, error)
            count += 1
    if count != len(expected_steps):
        raise ValueError("native surface trace does not reach the final displayed accepted state")
    result = {"displayed_accepted_frames": count, "minimum_full_skin_bed_gap_m": minimum_gap,
            "maximum_rendered_functional_volume_relative_error": maximum_volume_error,
            "displayed_state_lag_steps": 1, "terminal_accepted_capture_included": terminal_accepted_capture,
            "whole_body_interfaces_qualified": False, "geometry_mode": geometry_mode}
    if geometry_mode == "common_seven_coordinate_v1":
        result["common_cardiac_geometry"] = {
            "coordinate_count": 7, "solver_status": 0,
            "maximum_normalized_residual": maximum_common_residual,
            "legacy_coordinates_and_closure_applicable": False}
    if mesh_triangles is not None:
        result["whole_mesh_area_audit"] = {
            "triangles_checked_per_frame": mesh_triangles,
            "zero_area_triangles": 0, "nonfinite_area_triangles": 0,
            "scope": "presented-frame triangle area only; not self-intersection or interface qualification"}
    if body_first is not None:
        result["body_center_of_mass"] = {"first_m": body_first, "last_m": body_last,
            "displacement_m": [last - first for first, last in zip(body_first, body_last)],
            "represented_mass_kg": body_mass,
            "qualification": "diagnostic displacement; stationary rest and drift are not inferred from endpoints"}
    if wall_bound is not None:
        material_bound = bool(wall_bound) or geometry_mode == "common_seven_coordinate_v1"
        result["ventricular_material"] = {"bound": material_bound, "target_ml": wall_target,
            "legacy_wall_bound": bool(wall_bound), "representation": geometry_mode,
            "gpu_degenerate_triangle_check_recorded": wall_status_present,
            "maximum_volume_relative_error": wall_error if material_bound else None,
            "closure_range_mm": [wall_min_closure, wall_max_closure] if wall_bound else None,
            "qualification": "rendered material volume consistency; wall topology and interfaces require separate checks"}
    return result


def native_respiration_trace_consistency(trace: Path, parameters: Path,
                                         windows: dict[str, list[float]]) -> dict[str, Any]:
    """Check observed mechanical identities, without advancing model state."""
    config = json.loads(parameters.read_text())
    keys = ("frc_m3", "diaphragm_area_m2", "rib_effective_area_m2",
            "airway_resistance_pa_s_per_m3", "lung_compliance_m3_per_pa")
    values = {key: finite_float(config.get(key), key) for key in keys}
    if min(values.values()) <= 0:
        raise ValueError("respiratory mechanical parameters must be positive")
    rest = finite_float(config.get("rest_pleural_pressure_pa"), "rest_pleural_pressure_pa")
    columns = ("time_s", "lung_volume_ml", "airflow_ml_s", "alveolar_pa", "pleural_pa",
               "diaphragm_mm", "rib_mm", "diaphragm_excitation", "intercostal_excitation",
               "diaphragm_activation", "intercostal_activation")
    with trace.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(columns).issubset(reader.fieldnames or ()):
            raise ValueError("native trace is missing respiratory mechanical observations")
        rows = [{key: finite_float(row[key], key) for key in columns} for row in reader]
    if len(rows) < 2 or any(b["time_s"] <= a["time_s"] for a, b in zip(rows, rows[1:])):
        raise ValueError("respiratory observations require increasing accepted times")
    maxima = {key: 0.0 for key in ("volume_decomposition_ml", "airway_pressure_pa",
                                  "pleural_compliance_pa")}
    maximum_fraction = 0.0
    for row in rows:
        if any(not 0 <= row[key] <= 1 for key in columns[-4:]):
            raise ValueError("respiratory excitation or activation is outside [0, 1]")
        volume_terms = (row["lung_volume_ml"], -1e6 * values["frc_m3"],
                        -1e3 * values["diaphragm_area_m2"] * row["diaphragm_mm"],
                        -1e3 * values["rib_effective_area_m2"] * row["rib_mm"])
        airway_terms = (row["alveolar_pa"],
                        values["airway_resistance_pa_s_per_m3"] * 1e-6 * row["airflow_ml_s"])
        pleural_terms = (row["pleural_pa"], -rest, -row["alveolar_pa"],
                         (row["lung_volume_ml"] * 1e-6 - values["frc_m3"]) /
                         values["lung_compliance_m3_per_pa"])
        for key, terms in zip(maxima, (volume_terms, airway_terms, pleural_terms)):
            residual = abs(math.fsum(terms))
            # A declared rounding allowance for exported Float32 states and
            # parameters, scaled by the terms in the identity. This is not a
            # physiological reference interval or an integration-error bound.
            allowance = 32 * 2**-23 * max(1.0, *(abs(x) for x in terms))
            if residual > allowance:
                raise ValueError(f"respiratory {key} identity failed at {row['time_s']} s")
            maxima[key] = max(maxima[key], residual)
            maximum_fraction = max(maximum_fraction, residual / allowance)
    summaries = {}
    for name, (start, end) in windows.items():
        samples = window_rows(rows, start, end)
        summaries[name] = {"window_s": [start, end], "accepted_samples": len(samples),
            "means": {key: mean(samples, key) for key in columns[-4:]},
            "ranges": {key: [min(r[key] for r in samples), max(r[key] for r in samples)]
                       for key in ("diaphragm_mm", "rib_mm", "lung_volume_ml", "airflow_ml_s",
                                   "alveolar_pa", "pleural_pa")}}
    return {"respiratory_mechanics": {
        "parameter_sha256": sha256_file(parameters), "trace_sha256": sha256_file(trace),
        "accepted_samples": len(rows), "maximum_absolute_identity_residuals": maxima,
        "rounding_allowance_float32_epsilons": 32,
        "maximum_fraction_of_rounding_allowance": maximum_fraction, "windows": summaries,
        "qualification": "Observed volume, pressure-flow and pressure-compliance identities; not independent proof of causal response, anatomy, or physiological plausibility."}}


def prepared_native_runtime_dependency_resolution(args: argparse.Namespace,
                                                    invocation_path: Path,
                                                    invocation: dict[str, Any]) -> list[dict[str, Any]]:
    identity_path = Path(args.native_build_identity).resolve()
    expected_sha256 = args.native_build_identity_sha256
    if (not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
            or sha256_file(identity_path) != expected_sha256):
        raise ValueError("prepared native build identity hash changed")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if not isinstance(identity, dict) or identity.get("schema") != "numi.human-resting.native-paired-build-identity.v1":
        raise ValueError("prepared native build identity has an unexpected schema")
    native_invocation = identity.get("native_invocation", {})
    if not isinstance(native_invocation, dict):
        raise ValueError("prepared native build identity has no invocation mapping")
    if (native_invocation.get("path") != str(invocation_path)
            or native_invocation.get("sha256") != sha256_file(invocation_path)
            or native_invocation.get("asset_sha256") != invocation.get("asset_sha256")):
        raise ValueError("prepared native build identity does not bind this exact invocation")
    expected_resolution = identity.get("runtime_dependency_resolution")
    if not isinstance(expected_resolution, list) or len(expected_resolution) != 7:
        raise ValueError("prepared native build identity lacks all seven runtime dependency resolutions")
    return verify_native_scene_runtime_dependency_resolution(invocation, expected_resolution)


def native_terminal_accepted_capture_evidence(invocation: dict[str, Any], output: Path,
                                             native: dict[str, Any], log: str,
                                             steps: int, dt: float) -> dict[str, Any]:
    """Verify a declared, exact-N accepted geometry capture from native evidence.

    Older launch templates did not request a terminal geometry frame. They
    retain the pre-terminal surface schedule. A launch that does request N
    must also carry the native terminal query proof and the matching accepted
    geometry receipt; the capture request alone is never completion evidence.
    """
    capture_key = "NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS"
    environment = invocation.get("environment", {})
    if not isinstance(environment, dict):
        raise ValueError("native invocation environment is malformed")
    raw_steps = environment.get(capture_key)
    template = invocation.get("readiness_derived_capture_launch_template")
    if raw_steps is None:
        if isinstance(template, dict) and template.get("capture_environment_key") == capture_key:
            raise ValueError("derived capture template has no matching native capture environment")
        return {"verified": False, "reason": "terminal accepted geometry was not declared"}
    if not isinstance(raw_steps, str) or not raw_steps:
        raise ValueError("native capture environment must be a nonempty step list")
    try:
        text_steps = raw_steps.split(",")
        if any(not re.fullmatch(r"(?:0|[1-9][0-9]*)", value) for value in text_steps):
            raise ValueError
        capture_steps = [int(value) for value in text_steps]
    except ValueError as exc:
        raise ValueError("native capture environment has malformed accepted steps") from exc
    if (capture_steps != sorted(capture_steps) or len(capture_steps) != len(set(capture_steps)) or
            any(step < 0 or step > steps for step in capture_steps)):
        raise ValueError("native capture environment has duplicate, unordered, or out-of-horizon steps")

    if template is not None:
        if not isinstance(template, dict):
            raise ValueError("derived capture launch template is malformed")
        expected_template = {
            "schema": "numi.human.resting.accepted-geometry-launch-template.v1",
            "accepted_horizon_steps": steps,
            "physical_timestep_s": dt,
            "capture_environment_key": capture_key,
            "capture_environment_value": raw_steps,
            "capture_step_ids": capture_steps,
            "terminal_step_id": steps,
        }
        if any(template.get(key) != value for key, value in expected_template.items()):
            raise ValueError("derived capture template does not match the recorded native horizon and step list")
        if steps not in capture_steps:
            raise ValueError("derived capture template must request its exact terminal accepted step")
        nominal_time = finite_float(str(template.get("terminal_nominal_time_s")),
                                    "declared terminal nominal time")
        if abs(nominal_time - steps * dt) > 1e-12:
            raise ValueError("derived capture template has a mismatched nominal terminal time")

    if steps not in capture_steps:
        return {"verified": False, "reason": "terminal accepted geometry was not requested"}
    if native.get("accepted_steps") != steps:
        raise ValueError("terminal accepted geometry was requested but the native run did not accept exact N")

    presentation_prefix = "resting_terminal_presentation="
    presentation_lines = [line for line in log.splitlines() if line.startswith(presentation_prefix)]
    presentation_pattern = re.compile(
        r"resting_terminal_presentation=accepted step=([0-9]+) body_count=([0-9]+) "
        r"respiratory_status=([0-9]+) common_coordinates=accepted_buffer_copied "
        r"physical_steps_advanced=0 controller_steps_advanced=0 "
        r"fk_owner=MetalArticulatedOperator_query_only")
    if len(presentation_lines) != 1:
        raise ValueError("native log must contain exactly one accepted terminal presentation proof")
    presentation = presentation_pattern.fullmatch(presentation_lines[0])
    if (presentation is None or int(presentation[1]) != steps or int(presentation[2]) <= 0 or
            int(presentation[3]) != steps):
        raise ValueError("native log terminal presentation does not prove exact accepted N without advancing")

    identity_prefix = "resting_terminal_capture_identity="
    identity_lines = [line for line in log.splitlines() if line.startswith(identity_prefix)]
    identity_pattern = re.compile(
        r"resting_terminal_capture_identity=accepted_step_([0-9]+) "
        r"q_source=exact_final_accepted_float32 root_source=exact_final_compensated_translation "
        r"fk=MetalArticulatedOperator_pointJacobiansOnly terminal_physical_steps_advanced=0")
    if len(identity_lines) != 1:
        raise ValueError("native log must contain exactly one accepted terminal capture identity")
    terminal_identity = identity_pattern.fullmatch(identity_lines[0])
    if terminal_identity is None or int(terminal_identity[1]) != steps:
        raise ValueError("native log terminal capture identity differs from exact accepted N")

    state_lines = [line for line in log.splitlines() if line.startswith("stand_terminal_state=")]
    if len(state_lines) != 1:
        raise ValueError("native log must contain exactly one terminal accepted body state")
    try:
        terminal_state = json.loads(state_lines[0].split("=", 1)[1])
    except (ValueError, TypeError) as exc:
        raise ValueError("native terminal accepted body state is malformed") from exc
    if (terminal_state.get("step_count") != steps or terminal_state.get("root_assistance") is not False or
            abs(finite_float(str(terminal_state.get("timestep_seconds")), "terminal native timestep") - dt) > 1e-12):
        raise ValueError("native terminal accepted body state does not match exact N and timestep")

    accepted_dir = output / "accepted-geometry"
    pack_path = accepted_dir / f"step-{steps}.mrvpack"
    receipt_path = accepted_dir / f"step-{steps}.receipt.json"
    for path, label in ((pack_path, "terminal accepted geometry pack"),
                         (receipt_path, "terminal accepted geometry receipt")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"native log declares terminal capture but {label} is missing: {path}")
    export_lines = [line for line in log.splitlines() if line.startswith("accepted_geometry_export=")]
    expected_export_prefix = f"accepted_geometry_export={pack_path}"
    matching_exports = [line for line in export_lines
                        if line.split(" ", 1)[0] == expected_export_prefix]
    if len(matching_exports) != 1:
        raise ValueError("native log must bind exactly one export of the exact terminal pack")
    fields = dict(token.split("=", 1) for token in matching_exports[0].split(" ")[1:] if "=" in token)
    if (fields.get("receipt") != str(receipt_path) or
            not re.fullmatch(r"0x[0-9a-fA-F]+", fields.get("accepted_root", "")) or
            fields.get("pack_sha256") != sha256_file(pack_path) or
            fields.get("receipt_sha256") != sha256_file(receipt_path)):
        raise ValueError("native terminal export line does not match its accepted pack and receipt hashes")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    accepted_time = finite_float(str(receipt.get("accepted_time_s")), "terminal receipt accepted time")
    accepted_root_hex = receipt.get("accepted_root_fingerprint_hex")
    if (receipt.get("schema") != "numi.human.accepted-render-geometry.v1" or
            receipt.get("accepted_step") != steps or receipt.get("physical_endpoint") != "accepted" or
            receipt.get("surface_audit_endpoint") != "passed" or
            receipt.get("accepted_pack_path") != str(pack_path) or
            receipt.get("pack_file_sha256") != sha256_file(pack_path) or
            not isinstance(accepted_root_hex, str) or
            fields.get("accepted_root", "").lower() != accepted_root_hex.lower() or
            not isinstance(receipt.get("accepted_root_fingerprint"), int) or
            int(accepted_root_hex, 16) != receipt["accepted_root_fingerprint"] or
            not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("accepted_body_state_sha256", ""))) or
            not re.fullmatch(r"[0-9a-f]{64}", str(receipt.get("accepted_respiration_state_sha256", ""))) or
            abs(accepted_time - native["simulated_s"]) > 1e-8):
        raise ValueError("terminal accepted geometry receipt does not bind the successful native accepted state")
    surface_audit = receipt.get("surface_audit", {})
    if (not isinstance(surface_audit, dict) or surface_audit.get("physical_endpoint") != "accepted" or
            surface_audit.get("surface_audit_endpoint") != "passed" or
            surface_audit.get("mesh_zero_area_triangles") != 0 or
            surface_audit.get("mesh_nonfinite_area_triangles") != 0):
        raise ValueError("terminal accepted geometry receipt lacks its passed surface audit")
    if template is not None:
        declared_accepted_time = finite_float(str(template.get("terminal_accepted_time_s")),
                                              "declared terminal accepted time")
        if abs(declared_accepted_time - accepted_time) > 1e-8:
            raise ValueError("derived capture template terminal time differs from the accepted receipt")
    return {"verified": True, "accepted_step": steps, "accepted_time_s": accepted_time,
            "capture_environment_key": capture_key, "capture_step_ids": capture_steps,
            "pack_sha256": sha256_file(pack_path), "receipt_sha256": sha256_file(receipt_path),
            "accepted_body_state_sha256": receipt["accepted_body_state_sha256"],
            "accepted_respiration_state_sha256": receipt["accepted_respiration_state_sha256"]}


def execute_native_scene_arm(args: argparse.Namespace) -> dict[str, Any]:
    validate_windows(args)
    work, output = Path.cwd().resolve(), Path(args.output).resolve()
    if output.parent != work or output.exists():
        raise ValueError("native scene output must be a new child of the notebook run directory")
    invocation_path = Path(args.invocation).resolve()
    invocation = json.loads(invocation_path.read_text())
    runtime_dependency_resolution = prepared_native_runtime_dependency_resolution(
        args, invocation_path, invocation)
    command = native_scene_command(invocation, output, args)
    bindings = invocation["asset_sha256"]

    def verify_bindings() -> None:
        for path, digest in bindings.items():
            if sha256_file(Path(path)) != digest:
                raise ValueError(f"preregistered native binary/library/asset changed: {path}")
        verify_native_scene_runtime_dependency_resolution(invocation, runtime_dependency_resolution)

    verify_bindings()
    recorded_environment = dict(invocation.get("environment", {}))
    env = native_scene_environment(recorded_environment, dict(os.environ))
    # This is an output, not a frozen input. A replayed preflight receipt must
    # not let an arm overwrite evidence in the preflight or another trial.
    failure_receipt_key = "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"
    if env.get(failure_receipt_key):
        recorded_environment[failure_receipt_key] = str(output / "common-field-failure.json")
        env[failure_receipt_key] = recorded_environment[failure_receipt_key]
    output.mkdir()
    write_json(output / "invocation.json", {**invocation, "argv": command,
               "environment": recorded_environment,
               "runtime_dependency_resolution": runtime_dependency_resolution,
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
    # The vascular world is common to both arms; the delivered intervention is
    # part of the coupled body/controller program identity. Bind both owners.
    if native["world_fingerprint"] != args.world_fingerprint:
        raise ValueError("native scene world differs from this arm's preregistered identity")
    if native["coupled_program_fingerprint"] != args.program_fingerprint:
        raise ValueError("native body/controller program differs from this arm's preregistered identity")
    result = observation(args, output / "resting-coupled.csv", native, log)
    movie, surfaces = output / "native-viewer.mov", output / "resting-surface-audit.csv"
    if not movie.is_file() or movie.stat().st_size == 0 or not surfaces.is_file():
        raise ValueError("native scene did not retain its continuous movie and surface trace")
    terminal_capture = native_terminal_accepted_capture_evidence(
        invocation, output, native, log, args.steps, args.dt)
    result.update(native_body_trace_consistency(output / "resting-coupled.csv", args.steps, args.dt))
    result.update(native_surface_trace_consistency(
        surfaces, args.steps, args.dt, require_whole_mesh=True,
        terminal_accepted_capture=terminal_capture["verified"]))
    result["terminal_accepted_capture_evidence"] = terminal_capture
    if args.steps * args.dt >= 310.0 - 1e-6:
        result.update(native_cycle_coverage(read_trace(output / "resting-coupled.csv"), args.dt))
    parameters = Path(command[command.index("--resting-scene") + 2])
    result.update(native_respiration_trace_consistency(output / "resting-coupled.csv", parameters,
        {name: result[name + "_window_s"] for name in ("pre", "dose", "recovery")}))
    result.update(native_whole_body_executed=True,
                  common_asset_identity=digest_json(bindings),
                  runtime_dependency_resolution=runtime_dependency_resolution,
                  body_source_fingerprint=native["body_source_fingerprint"],
                  coupled_program_fingerprint=native["coupled_program_fingerprint"],
                  recording_sha256=sha256_file(movie), surface_trace_sha256=sha256_file(surfaces),
                  reference_invocation_sha256=sha256_file(invocation_path))
    write_json(output / "intervention-observation.json", result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return result


def native_plan_components(args: argparse.Namespace, invocation: dict[str, Any],
                           source_hashes: dict[str, str], script: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build a v2 plan for the existing whole-body native scene adapter.

    This is plan construction only: it does not register the study or launch
    either arm. The per-arm program fingerprints must come from the native
    owner for the exact frozen invocation and intervention settings.
    """
    validate_windows(args)
    source_revisions = json.loads(Path(args.source_revisions).resolve().read_text(encoding="utf-8"))
    required_repositories = {"numi-lab", "numilab-human", "numi-brain"}
    if not isinstance(source_revisions, dict) or not required_repositories.issubset(source_revisions):
        raise ValueError("source revision inventory must include numi-lab, numilab-human, and numi-brain")
    for repository, revision in source_revisions.items():
        if not isinstance(repository, str) or not isinstance(revision, dict):
            raise ValueError("each source revision entry must be a repository identity object")
        commit, diff = revision.get("revision"), revision.get("diff_sha256")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError(f"{repository} revision must be the exact 40-character Git commit")
        if not isinstance(diff, str) or not re.fullmatch(r"[0-9a-f]{64}", diff):
            raise ValueError(f"{repository} source diff identity must be a SHA-256 digest")
    for name, value in (("world", args.world_fingerprint),
                        ("control program", args.control_program_fingerprint),
                        ("treatment program", args.treatment_program_fingerprint)):
        if not value.isdecimal() or int(value) == 0:
            raise ValueError(f"{name} fingerprint must be a nonzero owner-reported integer")
    if args.control_program_fingerprint == args.treatment_program_fingerprint:
        raise ValueError("control and intervention must have distinct coupled-program fingerprints")
    if not source_hashes or any(not isinstance(path, str) or not Path(path).is_absolute() or
                                not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                                for path, digest in source_hashes.items()):
        raise ValueError("source hash inventory must map absolute source paths to SHA-256 digests")
    for path, expected in source_hashes.items():
        source_path = Path(path)
        if source_path.is_symlink() or not source_path.is_file() or sha256_file(source_path) != expected:
            raise ValueError(f"source hash differs from frozen inventory: {path}")
    bindings = invocation.get("asset_sha256")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("native invocation must bind its consumed binary, libraries, and assets")
    for path, expected in bindings.items():
        if not isinstance(path, str) or not Path(path).is_absolute() or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("native invocation contains an invalid asset binding")
        asset_path = Path(path)
        if asset_path.is_symlink() or not asset_path.is_file() or sha256_file(asset_path) != expected:
            raise ValueError(f"native invocation asset differs from its frozen digest: {path}")
    if not isinstance(invocation.get("argv"), list):
        raise ValueError("native invocation must preserve its exact owner argv")
    if not args.device.startswith("Apple "):
        raise ValueError("native paired plan requires the actual Apple device identity")

    out = Path(args.directory).resolve()
    invocation_path = Path(args.invocation).resolve()
    source_hashes_path = Path(args.source_hashes).resolve()
    source_revisions_path = Path(args.source_revisions).resolve()
    fixture = Path(args.parser_fixture).resolve()
    if not fixture.is_file():
        raise ValueError("a retained accepted native CSV fixture is required for parser calibration")
    runtime_dependency_resolution = verify_native_scene_runtime_dependency_resolution(invocation)
    for arm in ("control", "treatment"):
        preflight_args = argparse.Namespace(steps=args.steps, dt=args.dt, arm=arm,
                                            start_s=args.start_s, end_s=args.end_s, scale=args.scale)
        native_scene_command(invocation, out / "preflight-output", preflight_args)
    unit_basis = {"source_revisions": source_revisions,
                  "source_files": source_hashes,
                  "common_assets": bindings,
                  "runtime_dependency_resolution": runtime_dependency_resolution,
                  "world_fingerprint": args.world_fingerprint,
                  "invocation_sha256": sha256_file(invocation_path),
                  "device": args.device, "steps": args.steps, "dt": args.dt}
    unit_id = "human-resting-reference-v1-" + digest_json(unit_basis)[:20]
    common_asset_identity = digest_json(bindings)

    model = {
        "schema": "numi.human-resting-paco2-model.v1",
        "version": "alveolar-ventilation-envelope-v1",
        "statement": ("For fixed resting CO2 production, the alveolar-ventilation relation predicts a "
                       "nonnegative control-corrected PaCO2 change when effective alveolar ventilation "
                       "under reduced drive is between one-half and all of baseline. From a 40 mmHg "
                       "baseline this gives a broad 0 to +40 mmHg sensitivity envelope; it is not a "
                       "probability interval or a clinical prediction."),
        "parameters": {"baseline_PaCO2_mmhg": 40.0,
                       "minimum_effective_alveolar_ventilation_fraction": 0.5,
                       "maximum_effective_alveolar_ventilation_fraction": 1.0,
                       "assumption": ("Sensitivity only: at fixed CO2 production, effective alveolar "
                                      "ventilation is assumed to remain in [0.5, 1.0] of baseline. "
                                      "The closed-loop native response is measured, not prescribed.")},
        "training_units": [],
        "scope": "One deterministic paired simulation condition; not population inference, physiology calibration, or clinical validation.",
    }
    calibration = known_parser_calibration(fixture)
    calibration["bindings"] = {str(script): sha256_file(script), str(fixture): sha256_file(fixture),
                               str(invocation_path): sha256_file(invocation_path),
                               str(source_hashes_path): sha256_file(source_hashes_path),
                               str(source_revisions_path): sha256_file(source_revisions_path)}
    calibration["evidence"] = {str(fixture): sha256_file(fixture)}

    py = str(Path(sys.executable).resolve())
    common = ["--invocation", str(invocation_path), "--unit-id", unit_id,
              "--world-fingerprint", args.world_fingerprint, "--device", args.device,
              "--steps", str(args.steps), "--dt", repr(args.dt),
              "--start-s", repr(args.start_s), "--end-s", repr(args.end_s),
              "--scale", repr(args.scale), "--window-s", repr(args.window_s)]
    trials = []
    for arm, program_fingerprint, trial_id in (
            ("control", args.control_program_fingerprint, "resting-baseline"),
            ("treatment", args.treatment_program_fingerprint, "resting-drive-half")):
        trials.append({"id": trial_id, "pair": "resting-reference-v1", "arm": arm,
                       "unit": {"unit_id": unit_id},
                       "argv": [py, str(script), "run-native", *common,
                                "--program-fingerprint", program_fingerprint,
                                "--arm", arm, "--output", "{run}/scene"],
                       "env": {}, "timeout_seconds": 86400})

    native_identity_path = out / "native-build-identity.json"
    model_path = out / "model.json"
    calibration_path = out / "calibration.json"
    source_paths = sorted(source_hashes)
    bound_assets = sorted(bindings)
    instrument_artifacts = list(dict.fromkeys([str(script), str(invocation_path),
        str(source_hashes_path), str(source_revisions_path), str(fixture), *source_paths, *bound_assets]))
    # The notebook validates every declared instrument input against this
    # report, including the frozen native binary, libraries and source files.
    calibration["bindings"] = {path: sha256_file(Path(path)) for path in instrument_artifacts}
    identity = {"schema": "numi.human-resting.native-paired-build-identity.v1",
                "source_revisions": source_revisions,
                "source_revisions_file": {"path": str(source_revisions_path),
                                          "sha256": sha256_file(source_revisions_path)},
                "source_file_sha256": source_hashes,
                "device": args.device,
                "native_invocation": {"path": str(invocation_path),
                                      "sha256": sha256_file(invocation_path),
                                      "asset_sha256": bindings},
                "runtime_dependency_resolution": runtime_dependency_resolution,
                "world_fingerprint": args.world_fingerprint,
                "coupled_program_fingerprint_by_arm": {
                    "control": args.control_program_fingerprint,
                    "treatment": args.treatment_program_fingerprint},
                "common_asset_identity": common_asset_identity,
                "unit_id": unit_id,
                "configuration": {"steps": args.steps, "dt_s": args.dt,
                                  "duration_s": args.steps * args.dt,
                                  "drive_intervention_start_s": args.start_s,
                  "drive_intervention_end_s": args.end_s,
                  "drive_intervention_scale": args.scale,
                  "analysis_window_s": args.window_s,
                  "initialization_exclusion_s": 10.0,
                  "observation_after_initialization_s": args.steps * args.dt - 10.0,
                  "solver": "existing native Matter Dense45 circulation and accepted-state body coupling",
                                  "controller": "existing NumiBrain respiratory chemoreflex"},
                "scope": "Frozen native whole-body scene input identity; this is not physiological or clinical qualification."}
    identity_sha256 = hashlib.sha256((json.dumps(identity, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")).hexdigest()
    for trial in trials:
        trial["argv"].extend(("--native-build-identity", str(native_identity_path),
                              "--native-build-identity-sha256", identity_sha256))
    plan = {
        "schema": "numi.science.plan.v2", "purpose": "exploration",
        "question": (f"In the frozen native resting human scene, does reducing delivered respiratory excitation "
                     f"to {args.scale:g} on [{args.start_s:g},{args.end_s:g}) s lower ventilation and raise "
                     "control-corrected PaCO2 during the intervention, while gas measurements return near the "
                     "unchanged control during late recovery?"),
        "hypothesis": model["statement"],
        "owner": "Numi Human native articulated body, Matter circulation, and NumiBrain respiratory control",
        "repository": str(Path(args.repository).resolve()),
        "backend": f"Apple Metal native integrated viewer on {args.device}",
        "evidence_level": "simulation", "model": model, "model_file": str(model_path),
        "predictor": {"argv": [py, str(script), "predict", "--model", str(model_path)],
                      "env": {}, "timeout_seconds": 30},
        "instrument": {"description": (f"Native accepted-state PaCO2 difference-in-differences using {args.window_s:g} s "
                        "pre-dose and dose windows; accepted-flow ventilation, PaO2/SaO2, late recovery, cardiac, "
                        "blood-balance, body-clock, and whole-surface contact diagnostics are retained per arm."),
                       "calibration": str(calibration_path), "artifacts": instrument_artifacts},
        "artifacts": list(dict.fromkeys([*instrument_artifacts, str(native_identity_path),
                                         str(model_path), str(calibration_path)])),
        "design": {"intervention": (f"Existing Brain chemoreflex remains active; only its delivered diaphragm/intercostal "
                                     f"excitation is multiplied by {args.scale:g} on [{args.start_s:g},{args.end_s:g}) s."),
                   "pre_dose_window_s": [args.start_s - args.window_s, args.start_s],
                   "dose_window_s": [args.end_s - args.window_s, args.end_s],
                   "recovery_window_s": [args.steps * args.dt - args.window_s, args.steps * args.dt],
                   "initialization_exclusion_window_s": [0.0, 10.0],
                   "observation_after_initialization_s": args.steps * args.dt - 10.0,
                   "controls": "Matched fresh native process and same frozen invocation/assets/initialization; control drive remains 1.0 throughout.",
                   "experimental_unit": "One deterministic resting initialization and source/configuration identity; paired trajectories are not independent people or independent time samples.",
                   "allocation": "One exploratory pair, control then treatment, no retries/exclusions; all outcomes come from accepted native state.",
                   "unit_paths": {"unit_id": ["unit_id"]}},
        "observable": {"name": (f"Control-corrected change in mean PaCO2: treatment-minus-control of each arm's "
                                 f"dose-window minus pre-dose-window mean ({args.window_s:g} s windows)."),
                        "unit": "mmHg", "path": ["primary_delta_PaCO2_mmhg"]},
        "prediction": model_prediction(model),
        "secondary_predictions": {
            "dose_direction": {"treatment_minus_control_inspiratory_ventilation_L_min": "< 0",
                               "treatment_minus_control_primary_delta_PaCO2_mmhg": ">= 0",
                               "scope": "Mechanistic response-direction checks; not guaranteed by a target curve."},
            "late_recovery_equivalence": {
                "comparison": (f"treatment versus unchanged control, both averaged over the final "
                               f"{args.window_s:g} s"),
                "PaCO2_absolute_difference_max_mmhg": 1.0,
                "inspiratory_minute_ventilation_relative_difference_max_fraction": 0.10,
                "PaO2_absolute_difference_max_mmhg": 5.0,
                "justification": ("Predeclared numerical equivalence margins: 1 mmHg is 2.5% of the model's "
                                  "40 mmHg reference, 10% is a one-tenth relative ventilation tolerance, and "
                                  "5 mmHg is a finite oxygen readout margin. These are study decision limits, "
                                  "not literature-defined clinical cutoffs or guaranteed outcomes."),
                "scope": "Descriptive late recovery comparison; apply to the fixed native trajectories only."}},
        "validity": [{"path": ["schema"], "equals": "numi.human-resting.intervention-observation.v1"},
                     {"path": ["accepted_steps"], "equals": args.steps},
                     {"path": ["nominal_duration_s"], "equals": args.steps * args.dt},
                     {"path": ["duration_valid"], "equals": True},
                     {"path": ["dense45"], "equals": True}, {"path": ["brain_control"], "equals": True},
                     {"path": ["root_assistance_observed"], "equals": False},
                     {"path": ["common_asset_identity"], "equals": common_asset_identity}],
        "paired_equal": [["unit_id"], ["device"], ["world_fingerprint"], ["timestep_s"],
                         ["accepted_steps"], ["dense45"], ["brain_control"],
                         ["common_asset_identity"]],
        "trials": trials,
        "limitations": ("This is one deterministic paired simulation, not a population estimate or clinical validation. "
                        "The primary predictor is the fixed-production alveolar-ventilation relation and a broad "
                        "sensitivity envelope; closed-loop finite-time response may differ. Recovery limits are "
                        "prespecified analysis tolerances, not claimed physiological standards. Body and organ source "
                        "anatomy remain explicitly unqualified wherever the owner receipt says so."),
    }
    return identity, calibration, plan


def prepare_native(args: argparse.Namespace) -> Path:
    """Write a native v2 plan draft from exact frozen owner identities only."""
    invocation_path = Path(args.invocation).resolve()
    source_hashes_path = Path(args.source_hashes).resolve()
    script = Path(__file__).resolve()
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    source_hashes = json.loads(source_hashes_path.read_text(encoding="utf-8"))
    if not isinstance(source_hashes, dict):
        raise ValueError("source hash inventory must be a JSON object")
    out = Path(args.directory).resolve()
    identity, calibration, plan = native_plan_components(args, invocation, source_hashes, script)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "native-build-identity.json", identity)
    write_json(out / "model.json", plan["model"])
    write_json(out / "calibration.json", calibration)
    write_json(out / "plan.json", plan)
    print(out / "plan.json")
    return out / "plan.json"


NATIVE_310S_STEPS = 155_000
NATIVE_310S_DT = 0.002
NATIVE_310S_START_S = 60.0
NATIVE_310S_END_S = 100.0
NATIVE_310S_SCALE = 0.5
NATIVE_310S_WINDOW_S = 30.0

NATIVE_310S_REQUIRED_ENVIRONMENT = {
    "NUMI_HUMAN_SPLIT_STAND": "1",
    "NUMI_HUMAN_EXECUTION_STAGES": "1",
    "NUMI_HUMAN_TRAINING_PROFILE": "1",
    "NUMI_HUMAN_RESTING_TRANSACTION_PROBE": "1",
    "NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT": "1",
    "NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS": "8",
    "NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT": "0",
    "NUMI_HUMAN_STAND_SPARSE_OPERATOR": "1",
    "NUMI_HUMAN_STAND_CACHE_LIMIT_EQUALITY": "1",
    "NUMI_HUMAN_STAND_REDUCED_PROJECTED_RESPONSES": "1",
    "NUMI_HUMAN_STAND_REDUCED_CHOLESKY": "1",
    "NUMI_HUMAN_STAND_REDUCED_BASE_PROJECTION": "1",
    "NUMI_HUMAN_STAND_DEFER_EQUALITY_DATA": "1",
    "NUMI_HUMAN_STAND_DEFER_EQUALITY_DIAGNOSTICS": "0",
    "NUMI_HUMAN_STAND_EQUALITY_DEFERRAL_FAULT": "none",
    "NUMI_HUMAN_STAND_COMPENSATED_BODY_SUM": "1",
    "NUMI_HUMAN_RESTING_ELIMINATE_FIXED_BOUNDS": "1",
    "NUMI_HUMAN_RESTING_CONSOLIDATE_LINEAR_BOUNDS": "1",
    "NUMI_HUMAN_SUPPORT_ANCESTRY_PRUNE": "1",
    "NUMI_HUMAN_PARALLEL_RESPIRATORY_MUSCLES": "1",
    "NUMI_HUMAN_PARALLEL_RESPIRATORY_GAS": "0",
    "NUMI_HUMAN_GAS_TRANSPORT_SUBCYCLING": "1",
    "NUMI_HUMAN_RESPIRATORY_SUBCYCLING": "1",
}

NATIVE_310S_DISABLED_EXPERIMENTS = {
    "NUMI_HUMAN_STAND_ANALYTIC_BODY_SPATIAL_JACOBIANS": "0",
    "NUMI_HUMAN_STAND_CACHE_CONTACT_JACOBIAN": "0",
    "NUMI_HUMAN_STAND_CONTACT_WARMSTART": "0",
    "NUMI_HUMAN_STAND_FIRST_SIMD_CONTACT_SWEEP": "0",
    "NUMI_HUMAN_STAND_FUSE_CANONICAL_BODY_PROBES": "0",
    "NUMI_HUMAN_STAND_HYBRID_EQUALITY_FACTOR_CACHE": "0",
    "NUMI_HUMAN_STAND_ONE_PASS_ORDERED_LIMITS": "0",
    "NUMI_HUMAN_STAND_PGS_VELOCITY_RESIDUAL_TOLERANCE": "0.0",
    "NUMI_HUMAN_STAND_SPECULATIVE_CONTACT_DISTANCE_M": "0.0",
    "NUMI_HUMAN_STAND_ZERO_CONTACT_RESPONSE_FASTPATH": "0",
    "NUMI_HUMAN_STAND_FORCE_PARITY_DIAGNOSTIC": "0",
    "NUMI_HUMAN_STAND_SOURCE_ASSEMBLY_DUMP": "0",
    "NUMI_HUMAN_SKIN_INFLUENCE_TILE32": "0",
}


def validate_native_310s_invocation(invocation: dict[str, Any]) -> None:
    """Require a real owner invocation with the admitted 2 ms GPU configuration."""
    environment = invocation.get("environment")
    argv = invocation.get("argv")
    if not isinstance(environment, dict) or not isinstance(argv, list):
        raise ValueError("310 s preparation requires a Human owner invocation receipt")
    if (invocation.get("qualification") !=
            "native execution receipt; physiological and anatomical acceptance require separate audits" or
            invocation.get("machine") != "arm64" or
            not str(invocation.get("system", "")).startswith(("Darwin", "macOS-"))):
        raise ValueError("310 s preparation requires a completed Apple-silicon Human owner run receipt")
    for key, expected in NATIVE_310S_REQUIRED_ENVIRONMENT.items():
        if environment.get(key) != expected:
            raise ValueError(f"310 s invocation requires {key}={expected}")
    for key, expected in NATIVE_310S_DISABLED_EXPERIMENTS.items():
        if environment.get(key, expected) != expected:
            raise ValueError(f"310 s invocation requires exploratory setting {key}={expected}")
    for key in ("NUMI_HUMAN_STAND_REDUCED_RESPONSE_DIAGNOSTIC_ROOTS",
                "NUMI_HUMAN_STAND_FINISH_COUNTER_ROOTS"):
        if environment.get(key, ""):
            raise ValueError(f"310 s invocation requires diagnostic root setting {key} to be unset or empty")
    if any(key.startswith("NUMI_HUMAN_STAND_CPU_") and
           (value == "1" or key == "NUMI_HUMAN_STAND_CPU_ACCELERATE_FACTOR")
           for key, value in environment.items()):
        raise ValueError("310 s invocation cannot enable a CPU stand solver")
    if "NUMI_HUMAN_RESIDENT_PHYSICS_PILOT" in environment:
        raise ValueError("310 s paired study requires the single-Human scene, not the resident physics pilot")
    if not argv or Path(str(argv[0])).name != "numi-human-native":
        raise ValueError("310 s invocation must use the native Human executable")
    try:
        dt = float(argv[argv.index("--muscle-step-seconds") + 1])
        steps = int(argv[argv.index("--muscle-step-count") + 1])
    except (ValueError, IndexError, TypeError) as exc:
        raise ValueError("310 s invocation is missing its native timestep or step count") from exc
    if dt != NATIVE_310S_DT or steps <= 0:
        raise ValueError("310 s invocation must be a real 2 ms native run")
    if "--resting-movie" not in argv or "--mechanics-only" in argv:
        raise ValueError("310 s invocation must retain the native anatomical viewer movie")
    if "--resting-drive-intervention" in argv:
        raise ValueError("310 s invocation must be an unmodified baseline owner receipt")
    for key in ("NUMI_HUMAN_GPU_TIMING", "NUMI_MATTER_GPU_TIMING",
                "NUMI_HUMAN_SUPPORT_GPU_TIMING", "NUMI_MATTER_GPU_TIMING_DENSE45"):
        if environment.get(key, "0") not in ("0", "false", "False"):
            raise ValueError(f"310 s invocation must disable profiling setting {key}")


def _reference_execution(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} reference is not a regular file: {path}")
    path = path.resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("exit_code") != 0:
        raise ValueError(f"{label} reference did not complete successfully")
    return record



FULL_Q_INTEGRATION_FIELDS = (
    "accepted_step", "time_s", "dt_s", "configuration_count", "velocity_count",
    "q_before_f32_semicolon", "v_before_f32_semicolon",
    "q_preprojection_f32_semicolon", "v_preprojection_f32_semicolon",
    "q_accepted_f32_semicolon", "v_accepted_f32_semicolon",
    "root_before_reference_displacement_correction_xyzw_semicolon",
    "root_after_reference_displacement_correction_xyzw_semicolon",
    "q_before_fingerprint_fnv64", "v_before_fingerprint_fnv64",
    "q_preprojection_fingerprint_fnv64", "v_preprojection_fingerprint_fnv64",
    "q_accepted_fingerprint_fnv64", "v_accepted_fingerprint_fnv64",
    "source_body_linear_momentum_before_x_kg_m_s",
    "source_body_linear_momentum_before_y_kg_m_s",
    "source_body_linear_momentum_before_z_kg_m_s",
    "source_body_linear_momentum_free_same_q_x_kg_m_s",
    "source_body_linear_momentum_free_same_q_y_kg_m_s",
    "source_body_linear_momentum_free_same_q_z_kg_m_s",
    "source_body_linear_momentum_preprojection_velocity_same_q_x_kg_m_s",
    "source_body_linear_momentum_preprojection_velocity_same_q_y_kg_m_s",
    "source_body_linear_momentum_preprojection_velocity_same_q_z_kg_m_s",
    "source_body_linear_momentum_preprojection_qv_x_kg_m_s",
    "source_body_linear_momentum_preprojection_qv_y_kg_m_s",
    "source_body_linear_momentum_preprojection_qv_z_kg_m_s",
    "source_body_linear_momentum_accepted_qv_x_kg_m_s",
    "source_body_linear_momentum_accepted_qv_y_kg_m_s",
    "source_body_linear_momentum_accepted_qv_z_kg_m_s",
)
FULL_Q_VECTOR_FIELDS = (
    "q_before_f32_semicolon", "q_preprojection_f32_semicolon",
    "q_accepted_f32_semicolon",
)
FULL_V_VECTOR_FIELDS = (
    "v_before_f32_semicolon", "v_preprojection_f32_semicolon",
    "v_accepted_f32_semicolon",
)
FULL_ROOT_VECTOR_FIELDS = (
    "root_before_reference_displacement_correction_xyzw_semicolon",
    "root_after_reference_displacement_correction_xyzw_semicolon",
)
FULL_Q_FINGERPRINT_FIELDS = (
    "q_before_fingerprint_fnv64", "v_before_fingerprint_fnv64",
    "q_preprojection_fingerprint_fnv64", "v_preprojection_fingerprint_fnv64",
    "q_accepted_fingerprint_fnv64", "v_accepted_fingerprint_fnv64",
)


def native_full_q_integration_trace_consistency(trace: Path, steps: int, requested_dt: float,
                                                actual_dt: float) -> dict[str, Any]:
    """Validate the exact accepted-q CSV schema, every root, and every numeric payload."""
    if trace.is_symlink() or not trace.is_file():
        raise ValueError("full-q reference is missing its regular accepted q-audit CSV")
    previous_time = None
    q_count = v_count = root_count = None
    count = 0
    momentum_fields = tuple(field for field in FULL_Q_INTEGRATION_FIELDS
                            if field.startswith("source_body_linear_momentum_"))
    with trace.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FULL_Q_INTEGRATION_FIELDS:
            raise ValueError("full-q reference has an unexpected accepted q-audit header")
        for row in reader:
            if None in row or count >= steps:
                raise ValueError("full-q reference has malformed or excess accepted q-audit rows")
            step = count + 1
            try:
                accepted_step = int(row["accepted_step"])
            except (TypeError, ValueError) as exc:
                raise ValueError("full-q reference has a malformed accepted-step field") from exc
            if accepted_step != step:
                raise ValueError("full-q reference does not contain every accepted root in order")
            row_dt = finite_float(row["dt_s"], "accepted q dt_s")
            row_time = finite_float(row["time_s"], "accepted q time_s")
            if abs(row_dt - actual_dt) > 1e-12 or abs(row_dt - requested_dt) > 1e-6:
                raise ValueError("full-q reference accepted timestep differs from its native invocation")
            if abs(row_time - step * row_dt) > 1e-8:
                raise ValueError("full-q reference accepted q clock does not match its root index")
            if previous_time is not None and row_time <= previous_time:
                raise ValueError("full-q reference accepted q clock is not strictly increasing")
            previous_time = row_time
            try:
                row_q_count = int(row["configuration_count"])
                row_v_count = int(row["velocity_count"])
            except (TypeError, ValueError) as exc:
                raise ValueError("full-q reference has invalid q/v dimensions") from exc
            if row_q_count <= 0 or row_v_count <= 0:
                raise ValueError("full-q reference has empty q/v dimensions")
            if q_count is None:
                q_count, v_count = row_q_count, row_v_count
            elif (row_q_count, row_v_count) != (q_count, v_count):
                raise ValueError("full-q reference q/v dimensions change between accepted roots")
            for field in FULL_Q_VECTOR_FIELDS:
                values = row[field].split(";")
                if len(values) != row_q_count:
                    raise ValueError(f"full-q reference {field} length differs from configuration_count")
                for value in values:
                    finite_float(value, field)
            for field in FULL_V_VECTOR_FIELDS:
                values = row[field].split(";")
                if len(values) != row_v_count:
                    raise ValueError(f"full-q reference {field} length differs from velocity_count")
                for value in values:
                    finite_float(value, field)
            for field in FULL_ROOT_VECTOR_FIELDS:
                values = row[field].split(";")
                if not values or any(value == "" for value in values):
                    raise ValueError(f"full-q reference {field} is empty")
                if root_count is None:
                    root_count = len(values)
                elif len(values) != root_count:
                    raise ValueError(f"full-q reference {field} dimension changes")
                for value in values:
                    finite_float(value, field)
            for field in momentum_fields:
                finite_float(row[field], field)
            for field in FULL_Q_FINGERPRINT_FIELDS:
                value = row[field]
                if not value.isdecimal() or int(value) > 0xFFFFFFFFFFFFFFFF:
                    raise ValueError(f"full-q reference {field} is not an unsigned 64-bit fingerprint")
            count += 1
    if count != steps:
        raise ValueError("full-q reference does not retain every accepted q-audit row")
    return {
        "accepted_rows": count, "first_accepted_step": 1,
        "last_accepted_step": count, "configuration_count": q_count,
        "velocity_count": v_count, "root_observer_component_count": root_count,
        "actual_float_dt_s": actual_dt, "schema_fields": len(FULL_Q_INTEGRATION_FIELDS),
        "all_numeric_values_finite": True,
    }




FULL_Q_INDEX_MAP_FIELDS = (
    "record_kind", "local_q_index", "global_q_index", "local_v_index", "global_v_index",
    "joint_index", "joint_name", "dof_name", "local_dof", "q_index_valid",
)


def native_full_q_index_map_consistency(path: Path, q_count: int, v_count: int) -> dict[str, Any]:
    """Validate the owner-emitted component map used to interpret the q/v columns."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("full-q reference is missing its regular q/v index map")
    q_indices, v_indices = set(), set()
    record_count = 0
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FULL_Q_INDEX_MAP_FIELDS:
            raise ValueError("full-q reference has an unexpected q/v index-map header")
        for row in reader:
            if None in row:
                raise ValueError("full-q q/v index map has malformed extra fields")
            record_count += 1
            kind = row["record_kind"]
            try:
                q_valid = int(row["q_index_valid"])
            except (TypeError, ValueError) as exc:
                raise ValueError("full-q q/v index map has an invalid validity bit") from exc
            if q_valid not in (0, 1):
                raise ValueError("full-q q/v index map validity bit is not binary")
            q_value, v_value = row["global_q_index"], row["global_v_index"]
            if kind == "scalar_dof":
                if q_valid != 1 or not q_value.isdecimal() or not v_value.isdecimal():
                    raise ValueError("full-q scalar DOF mapping is incomplete")
                q_index, v_index = int(q_value), int(v_value)
                if q_index in q_indices or v_index in v_indices:
                    raise ValueError("full-q q/v index map duplicates a scalar index")
                q_indices.add(q_index)
                v_indices.add(v_index)
            elif kind == "configuration_without_direct_velocity":
                if q_valid != 0 or not q_value.isdecimal() or v_value:
                    raise ValueError("full-q configuration-only mapping is malformed")
                q_index = int(q_value)
                if q_index in q_indices:
                    raise ValueError("full-q q/v index map duplicates a configuration index")
                q_indices.add(q_index)
            elif kind == "velocity_only":
                if q_valid != 0 or q_value or not v_value.isdecimal():
                    raise ValueError("full-q velocity-only mapping is malformed")
                v_index = int(v_value)
                if v_index in v_indices:
                    raise ValueError("full-q q/v index map duplicates a velocity index")
                v_indices.add(v_index)
            else:
                raise ValueError("full-q q/v index map has an unknown record kind")
    if (not record_count or q_indices != set(range(q_count)) or
            v_indices != set(range(v_count))):
        raise ValueError("full-q q/v index map does not cover its declared component dimensions")
    return {"q_components": len(q_indices), "v_components": len(v_indices),
            "mapping_records": record_count,
            "scope": "Owner-provided component index map for interpreting accepted q/v vectors."}

def _unique_native_arg(argv: list[Any], flag: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"modern full-q invocation requires exactly one {flag} value")
    return str(argv[positions[0] + 1])


def _modern_full_q_reference_manifest(q_path: Path, verification_path: Path, runtime_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Pin a modern native run directly; never manufacture an execution receipt."""
    q_dir = q_path.parent
    invocation_path = q_dir / "invocation.json"
    native_log_path = q_dir / "native.log"
    q_csv = q_dir / "resting-com-q-integration.csv"
    q_index_map = q_dir / "resting-com-q-index-map.csv"
    physiology_csv = q_dir / "resting-coupled.csv"
    surface_csv = q_dir / "resting-surface-audit.csv"
    momentum_csv = q_dir / "resting-com-momentum-diagnostic.csv"
    support_csv = q_dir / "resting-com-support-impulses.csv"
    terminal_pack = q_dir / "accepted-geometry" / "step-10000.mrvpack"
    terminal_receipt = q_dir / "accepted-geometry" / "step-10000.receipt.json"
    files = [
        (q_path, "modern full-q run metadata"),
        (invocation_path, "modern full-q invocation"),
        (native_log_path, "modern full-q native log"),
        (verification_path, "modern full-q terminal-cycle verification"),
        (q_csv, "modern full-q integration trace"),
        (q_index_map, "modern full-q component index map"),
        (physiology_csv, "modern full-q physiology trace"),
        (surface_csv, "modern full-q presented surface trace"),
        (momentum_csv, "modern full-q momentum diagnostics"),
        (support_csv, "modern full-q support impulse diagnostics"),
        (terminal_pack, "modern full-q terminal accepted geometry pack"),
        (terminal_receipt, "modern full-q terminal accepted geometry receipt"),
        (runtime_path, "modern full-q runtime correctness reference"),
    ]
    for path, label in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{label} is not a regular retained artifact: {path}")
    metadata = json.loads(q_path.read_text(encoding="utf-8"))
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if (metadata.get("exit_code") != 0 or metadata.get("source_files_changed_during_run") != [] or
            metadata.get("qualification") !=
            "native execution receipt; physiological and anatomical acceptance require separate audits" or
            metadata.get("machine") != "arm64" or
            not str(metadata.get("system", "")).startswith(("Darwin", "macOS-"))):
        raise ValueError("modern full-q metadata does not attest a completed unchanged Apple-silicon run")
    for key in ("argv", "environment", "asset_sha256"):
        if metadata.get(key) != invocation.get(key):
            raise ValueError(f"modern full-q run metadata {key} differs from its invocation")
    argv = invocation.get("argv")
    environment = invocation.get("environment")
    assets = invocation.get("asset_sha256")
    if not isinstance(argv, list) or not isinstance(environment, dict) or not isinstance(assets, dict):
        raise ValueError("modern full-q invocation has malformed argv/environment/asset bindings")
    if not argv or Path(str(argv[0])).name != "numi-human-native":
        raise ValueError("modern full-q invocation is not from the native Human executable")
    binary_path = Path(str(argv[0]))
    build_root = binary_path.resolve().parent.parent
    build_pins_path = build_root / "evidence" / "build-pins.json"
    source_pins_path = build_root / "evidence" / "source-pins.json"
    for path in (binary_path, build_pins_path, source_pins_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"modern full-q build provenance is not a regular retained file: {path}")
    build_pins = json.loads(build_pins_path.read_text(encoding="utf-8"))
    source_pins = json.loads(source_pins_path.read_text(encoding="utf-8"))
    if not isinstance(build_pins, dict) or not isinstance(source_pins, dict):
        raise ValueError("modern full-q build/source pins must be JSON objects")
    expected_source_pins = {key: value for key, value in build_pins.items() if key != "artifacts"}
    if source_pins != expected_source_pins:
        raise ValueError("modern full-q build and source-file pins disagree")
    source_file_hashes = {
        key: value for key, value in source_pins.items()
        if key not in ("source_revision", "build_script_sha256")
    }
    if (not source_file_hashes or
            any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in source_file_hashes.values()) or
            not re.fullmatch(r"[0-9a-f]{40}", str(source_pins.get("source_revision", ""))) or
            not re.fullmatch(r"[0-9a-f]{64}", str(source_pins.get("build_script_sha256", "")))):
        raise ValueError("modern full-q source pins lack exact source/build-script SHA-256 identities")
    artifact_pins = build_pins.get("artifacts")
    if not isinstance(artifact_pins, dict):
        raise ValueError("modern full-q build pins lack compiled artifact hashes")
    binary_sha = sha256_file(binary_path)
    if artifact_pins.get(str(binary_path.resolve())) != binary_sha:
        raise ValueError("modern full-q executable differs from its retained build pin")
    respiration_metallib = build_root / "matter" / "shaders" / "HumanRespiration.metallib"
    if (respiration_metallib.is_symlink() or not respiration_metallib.is_file() or
            artifact_pins.get(str(respiration_metallib.resolve())) != sha256_file(respiration_metallib)):
        raise ValueError("modern full-q respiration metallib differs from its retained build pin")
    built_library = build_root / "lib" / "libmetalrobo.dylib"
    if (not built_library.is_file() or
            artifact_pins.get(str(built_library)) != sha256_file(built_library)):
        raise ValueError("modern full-q MetalRobo library differs from its retained build pin")
    try:
        requested_dt = float(_unique_native_arg(argv, "--muscle-step-seconds"))
        requested_steps = int(_unique_native_arg(argv, "--muscle-step-count"))
    except (ValueError, TypeError) as exc:
        raise ValueError("modern full-q invocation has invalid timestep or accepted root count") from exc
    if requested_dt != NATIVE_310S_DT or requested_steps != 10000:
        raise ValueError("modern full-q reference must retain 10000 accepted roots at 2 ms")
    if len(argv) <= 4 or Path(str(argv[4])).resolve() != q_dir.resolve():
        raise ValueError("modern full-q invocation output path does not identify its retained run directory")
    for flag in ("--persistent-metal-stand", "--resting-movie", "--resting-release-initialization"):
        if flag not in argv:
            raise ValueError(f"modern full-q invocation is missing {flag}")
    if "--mechanics-only" in argv or "--resting-drive-intervention" in argv:
        raise ValueError("modern full-q reference must be the unmodified coupled control")
    if (environment.get("NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT") != "1" or
            environment.get("NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT") != "1" or
            environment.get("NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS") != "1" or
            environment.get("NUMI_HUMAN_RESTING_TRANSACTION_PROBE") != "1" or
            environment.get("NUMI_HUMAN_RESTING_INSPECTION_TOUR") != "1"):
        raise ValueError("modern full-q invocation lacks accepted-q, transaction, or inspection probe settings")
    loaded = metadata.get("loaded_metal_runtime", {})
    library_path = Path(str(loaded.get("expected_path", "")))
    library_sha = loaded.get("expected_sha256")
    observed_images = loaded.get("observed_images", [])
    if (loaded.get("verified") is not True or not library_sha or
            library_path.is_symlink() or not library_path.is_file() or
            sha256_file(library_path) != library_sha or
            assets.get(str(library_path.resolve())) != library_sha or
            not any(isinstance(image, dict) and image.get("path") == str(library_path)
                    for image in observed_images)):
        raise ValueError("modern full-q metadata lacks a hash-bound loaded Metal runtime")
    runtime_reference = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_hashes = runtime_reference.get("runtime_sha256", {})
    runtime_entry = runtime_hashes.get(str(library_path.resolve())) if isinstance(runtime_hashes, dict) else None
    runtime_record = (runtime_entry.get("sha256") if isinstance(runtime_entry, dict) else None)
    if (runtime_record != library_sha or
            runtime_reference.get("integrated_2ms_check", {}).get("accepted_steps") != 1000):
        raise ValueError("modern full-q loaded library does not match the retained runtime correctness reference")
    native_log = native_log_path.read_text(encoding="utf-8", errors="replace")
    scene = native_scene_summary(native_log)
    if scene["accepted_steps"] != requested_steps or scene["device"] != "Apple M4 Pro":
        raise ValueError("modern full-q native terminal summary differs from its retained 10000-root Apple run")
    q_start = re.search(
        r"^resting_com_q_integration_audit=enabled accepted_only=1 segment_cap_steps=1 "
        r"actual_float_dt_s=([^ ]+) rows_file=resting-com-q-integration\.csv "
        r"q_index_map=resting-com-q-index-map\.csv arrays=before,preprojection,accepted "
        r"root=compensated_reference_displacement_correction observer_only=1$",
        native_log, re.MULTILINE)
    q_end = re.search(
        r"^resting_com_q_integration_rows=(\d+) final_accepted_step=(\d+) actual_float_dt_s=([^ ]+)",
        native_log, re.MULTILINE)
    presentation = re.search(
        r"^resting_terminal_presentation=accepted step=(\d+) body_count=(\d+) respiratory_status=(\d+) "
        r"common_coordinates=accepted_buffer_copied physical_steps_advanced=0 controller_steps_advanced=0 "
        r"fk_owner=MetalArticulatedOperator_query_only$",
        native_log, re.MULTILINE)
    capture = re.search(
        r"^resting_terminal_capture_identity=accepted_step_(\d+) "
        r"q_source=exact_final_accepted_float32 root_source=exact_final_compensated_translation "
        r"fk=MetalArticulatedOperator_pointJacobiansOnly terminal_physical_steps_advanced=0$",
        native_log, re.MULTILINE)
    integrated = re.search(
        r"^resting_integrated_body=completed simulated_s=([^ ]+).*"
        r"physiology_body_clock=matched root_assistance=false(?: |$)",
        native_log, re.MULTILINE)
    throughput = re.search(
        r"^resting_integrated_throughput accepted_start_step=1 accepted_end_step=(\d+) ",
        native_log, re.MULTILINE)
    if not all((q_start, q_end, presentation, capture, integrated, throughput)):
        raise ValueError("modern full-q native log lacks accepted-q, shared-clock, or terminal query proof")
    transaction_probe = re.search(
        r"^resting_integrated_rejection=pass .*"
        r"rejected_after_accepted_predecessor=true .*"
        r"body_q_v_root_myo_unchanged=true .*"
        r"circulation_and_clock_unchanged=true .*"
        r"respiration_brain_history_unchanged=true .*"
        r"retry_matches_uninterrupted_replay=true .*"
        r"all_owners_match_uninterrupted_prefix=true .*"
        r"retry=fresh_context_reseed_and_replay_matches_six_root_baseline "
        r"same_context_retry=unsupported$",
        native_log, re.MULTILINE)
    if transaction_probe is None:
        raise ValueError("modern full-q native log lacks the completed transaction rejection/replay probe")
    actual_dt = finite_float(q_start.group(1), "modern full-q logged dt")
    if (q_end.group(1) != str(requested_steps) or q_end.group(2) != str(requested_steps) or
            abs(finite_float(q_end.group(3), "modern full-q terminal dt") - actual_dt) > 1e-12 or
            presentation.group(1) != str(requested_steps) or int(presentation.group(2)) <= 0 or
            presentation.group(3) != str(requested_steps) or
            capture.group(1) != str(requested_steps) or throughput.group(1) != str(requested_steps)):
        raise ValueError("modern full-q native log does not terminate at the exact accepted root")
    terminal_time = finite_float(integrated.group(1), "modern full-q terminal time")
    q_metrics = native_full_q_integration_trace_consistency(q_csv, requested_steps, requested_dt, actual_dt)
    index_map_metrics = native_full_q_index_map_consistency(
        q_index_map, q_metrics["configuration_count"], q_metrics["velocity_count"])
    if (q_metrics["configuration_count"] != 129 or
            q_metrics["velocity_count"] != 128 or
            q_metrics["root_observer_component_count"] != 12):
        raise ValueError("modern full-q trace dimensions differ from the pinned native scene observer")
    if abs(terminal_time - requested_steps * actual_dt) > 1e-6:
        raise ValueError("modern full-q native clock differs from its requested accepted horizon")
    surface_metrics = native_surface_trace_consistency(
        surface_csv, requested_steps, requested_dt, require_whole_mesh=True,
        terminal_accepted_capture=True)
    if (surface_metrics["displayed_state_lag_steps"] != 1 or
            surface_metrics["terminal_accepted_capture_included"] is not True):
        raise ValueError("modern full-q surface trace lacks pre-step frames or its exact terminal accepted capture")
    metadata_sha = sha256_file(q_path)
    verification_sha = sha256_file(verification_path)
    candidate = verification.get("runs", {}).get(q_dir.name)
    if (verification.get("pass") is not True or
            verification.get("scope") !=
            "Exact accepted-state terminal presentation; not anatomical or physiological qualification" or
            not isinstance(candidate, dict) or candidate.get("metadata_sha256") != metadata_sha or
            candidate.get("argv") != argv):
        raise ValueError("modern full-q verification does not bind the selected successful native run")
    trace_files = {
        "resting-coupled.csv": physiology_csv,
        "resting-com-q-integration.csv": q_csv,
        "resting-com-momentum-diagnostic.csv": momentum_csv,
        "resting-com-support-impulses.csv": support_csv,
    }
    traces = verification.get("traces", {})
    if set(traces) != set(trace_files):
        raise ValueError("modern full-q verification has an incomplete or unexpected trace set")
    trace_metrics = {}
    for name, path in trace_files.items():
        row = traces[name]
        if (not isinstance(row, dict) or row.get("candidate_sha256") != sha256_file(path) or
                row.get("candidate_rows") != row.get("reference_rows") or
                row.get("shared_row_differences") != 0 or row.get("candidate_rows", 0) <= 0):
            raise ValueError(f"modern full-q verification does not match {name}")
        if name in ("resting-coupled.csv", "resting-com-q-integration.csv") and row["candidate_rows"] != requested_steps:
            raise ValueError(f"modern full-q verification has incomplete {name}")
        trace_metrics[name] = {
            "candidate_rows": row["candidate_rows"], "reference_rows": row["reference_rows"],
            "shared_row_differences": row["shared_row_differences"],
            "candidate_sha256": row["candidate_sha256"],
        }
    terminal = verification.get("terminal", {})
    accepted_time = finite_float(str(terminal.get("accepted_time_s")), "verified terminal accepted time")
    timestamp = terminal.get("accepted_timestamp_microseconds")
    if (terminal.get("accepted_step") != requested_steps or
            terminal.get("matches_final_physical_trace_time") is not True or
            terminal.get("no_additional_physical_or_controller_step") is not True or
            abs(accepted_time - terminal_time) > 1e-8 or timestamp != round(accepted_time * 1e6)):
        raise ValueError("modern full-q verification lacks the exact terminal accepted state with no extra step")
    with surface_csv.open("r", encoding="utf-8", newline="") as stream:
        surface_rows = list(csv.DictReader(stream))
    terminal_surface_row = surface_rows[-1] if surface_rows else {}
    if (int(terminal_surface_row.get("step", -1)) != requested_steps or
            abs(finite_float(terminal_surface_row.get("time_s", ""), "terminal surface accepted time") -
                accepted_time) > 1e-8):
        raise ValueError("modern full-q surface trace does not end at the exact accepted terminal state")
    if (terminal.get("accepted_body_state_sha256") is None or
            terminal.get("accepted_respiration_state_sha256") is None or
            terminal.get("pack_file_sha256") != sha256_file(terminal_pack) or
            terminal.get("receipt_sha256") != sha256_file(terminal_receipt)):
        raise ValueError("modern full-q terminal state/geometry hashes do not match retained outputs")
    receipt = json.loads(terminal_receipt.read_text(encoding="utf-8"))
    if (receipt.get("schema") != "numi.human.accepted-render-geometry.v1" or
            receipt.get("accepted_step") != requested_steps or
            receipt.get("physical_endpoint") != "accepted" or
            receipt.get("surface_audit_endpoint") != "passed" or
            receipt.get("pack_file_sha256") != sha256_file(terminal_pack) or
            receipt.get("accepted_body_state_sha256") != terminal.get("accepted_body_state_sha256") or
            receipt.get("accepted_respiration_state_sha256") != terminal.get("accepted_respiration_state_sha256") or
            abs(finite_float(str(receipt.get("accepted_time_s")), "terminal receipt time") - accepted_time) > 1e-8):
        raise ValueError("modern full-q accepted geometry receipt differs from its terminal verification")
    terminal_surface = terminal.get("surface_audit", {})
    if (terminal_surface.get("physical_endpoint") != "accepted" or
            terminal_surface.get("surface_audit_endpoint") != "passed" or
            terminal_surface.get("mesh_zero_area_triangles") != 0 or
            terminal_surface.get("mesh_nonfinite_area_triangles") != 0):
        raise ValueError("modern full-q terminal surface audit did not pass")
    required_files = [
        q_path, invocation_path, native_log_path, q_csv, q_index_map, physiology_csv, surface_csv,
        momentum_csv, support_csv, verification_path, terminal_pack, terminal_receipt, library_path,
        binary_path, build_pins_path, source_pins_path, respiration_metallib,
        built_library, runtime_path,
    ]
    file_pins = {str(path.resolve()): sha256_file(path) for path in required_files}
    manifest = {
        "full_q_2ms_reference": {
            "reference_kind": "modern native run-metadata, invocation, native-log, and terminal-cycle verification",
            "run_metadata_path": str(q_path), "run_metadata_sha256": metadata_sha,
            "invocation_path": str(invocation_path), "invocation_sha256": sha256_file(invocation_path),
            "native_log_path": str(native_log_path), "native_log_sha256": sha256_file(native_log_path),
            "verification_report_path": str(verification_path), "verification_report_sha256": verification_sha,
            "retained_file_sha256": file_pins,
            "native_build": {
                "binary_path": str(binary_path.resolve()), "binary_sha256": binary_sha,
                "build_pins_path": str(build_pins_path.resolve()),
                "build_pins_sha256": sha256_file(build_pins_path),
                "source_pins_path": str(source_pins_path.resolve()),
                "source_pins_sha256": sha256_file(source_pins_path),
                "source_revision": source_pins["source_revision"],
                "build_script_sha256": source_pins["build_script_sha256"],
                "source_file_sha256": source_file_hashes,
                "respiration_metallib_path": str(respiration_metallib.resolve()),
                "respiration_metallib_sha256": sha256_file(respiration_metallib),
                "built_library_path": str(built_library),
                "built_library_sha256": sha256_file(built_library),
            },
            "runtime_correctness_reference": {
                "path": str(runtime_path.resolve()), "sha256": sha256_file(runtime_path),
                "loaded_library_sha256": runtime_record,
                "scope": "Independent retained runtime identity check; does not assert the viewer source or shader identity.",
            },
            "requested_run": {
                "requested_step_count": requested_steps, "requested_dt_s": requested_dt,
                "accepted_step_count": scene["accepted_steps"], "accepted_terminal_time_s": terminal_time,
                "device": scene["device"], "world_fingerprint": scene["world_fingerprint"],
                "body_source_fingerprint": scene["body_source_fingerprint"],
                "coupled_program_fingerprint": scene["coupled_program_fingerprint"],
                "transaction_probe_setting": environment["NUMI_HUMAN_RESTING_TRANSACTION_PROBE"],
                "transaction_probe_log_record": {
                    "rejection_after_accepted_predecessor": True,
                    "all_owner_state_unchanged": True,
                    "fresh_context_replay_matches": True,
                    "source_line": transaction_probe.group(0),
                },
                "accepted_q_audit_setting": environment["NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT"],
                "loaded_metal_runtime": {
                    "path": str(library_path.resolve()), "sha256": library_sha,
                    "observed_images": observed_images, "verified": True,
                },
            },
            "q_integration_audit": {
                **q_metrics, "trace_sha256": sha256_file(q_csv),
                "component_index_map_path": str(q_index_map.resolve()),
                "component_index_map_sha256": sha256_file(q_index_map),
                "component_index_map": index_map_metrics,
                "native_terminal_row_count": int(q_end.group(1)),
            },
            "presented_surface_audit": {
                **surface_metrics, "presentation_period_s": 0.064,
                "scope": "Owner-validated pre-step displayed states and whole-mesh checks; not tissue-interface qualification.",
            },
            "terminal_capture": {
                "accepted_step": requested_steps, "accepted_time_s": accepted_time,
                "accepted_timestamp_microseconds": timestamp, "pack_path": str(terminal_pack),
                "receipt_path": str(terminal_receipt), "no_additional_physical_or_controller_step": True,
                "scope": "Exact terminal accepted state from the GPU viewer owner; not physiology or anatomy qualification.",
            },
            "verified_trace_comparisons": trace_metrics,
            "scope": (
                "Retained 931 run has a full accepted-q audit over 10000 roots at requested 2 ms, "
                "a separately verified query-only terminal capture, and owner-validated 64 ms surface samples. "
                "Its comparison verifies output/runtime identity against the retained 925 reference; it is not a "
                "310 s endurance result or physiological/anatomical qualification."
            ),
        },
    }
    return manifest, [str(path.resolve()) for path in required_files]


def native_310s_reference_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    """Pin the distinct segment-8 and full-q references without equating their scopes."""
    segment_path = Path(args.segment8_reference)
    q_path = Path(args.full_q_reference)
    runtime_path = Path(args.runtime_correctness_reference)
    if any(path.is_symlink() for path in (segment_path, q_path, runtime_path)):
        raise ValueError("audit-schedule references must not be symlinks")
    segment_path, q_path, runtime_path = (path.resolve() for path in (segment_path, q_path, runtime_path))
    segment = _reference_execution(segment_path, "segment-8")
    modern_q = q_path.name == "run-metadata.json"
    q_full = None if modern_q else _reference_execution(q_path, "full-q")
    verification_arg = getattr(args, "full_q_verification", None)
    if modern_q and not verification_arg:
        raise ValueError("modern full-q run-metadata requires --full-q-verification")
    if not modern_q and verification_arg:
        raise ValueError("--full-q-verification is only valid with modern run-metadata")
    if modern_q and Path(verification_arg).is_symlink():
        raise ValueError("modern full-q verification reference must not be a symlink")
    if runtime_path.is_symlink() or not runtime_path.is_file():
        raise ValueError(f"runtime correctness pin is not a regular file: {runtime_path}")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    segment_env = segment.get("environment", {})
    segment_run = segment.get("run_configuration", {})
    if (segment_env.get("NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT") != "1" or
            segment_env.get("NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS") != "8" or
            segment_env.get("NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT") != "0" or
            segment_run.get("accepted_com_audit_segment_steps") != 8 or
            segment_run.get("full_accepted_q_audit") is not False or
            segment_run.get("requested_dt_s") != "0.008" or
            segment_run.get("requested_step_count") != 250 or
            segment_run.get("presentation_period_s") != 0.064):
        raise ValueError("segment-8 reference no longer matches the retained 752 diagnostic schedule")
    if modern_q:
        modern_manifest, modern_artifacts = _modern_full_q_reference_manifest(
            q_path, Path(args.full_q_verification).resolve(), runtime_path)
        segment_dir = segment_path.parent
        segment_surface_metrics = native_surface_trace_consistency(
            segment_dir / "resting-surface-audit.csv", 250, 0.008, require_whole_mesh=True)
        segment_required = [
            segment_path, segment_dir / "resting-coupled.csv",
            segment_dir / "resting-surface-audit.csv",
            segment_dir / "resting-com-support-impulses.csv",
            segment_dir / "resting-com-momentum-diagnostic.csv",
        ]
        for path in segment_required:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"missing regular audit-schedule reference artifact: {path}")
        manifest = {
            "segment8_schedule_reference": {
                "execution_json": str(segment_path),
                "execution_sha256": sha256_file(segment_path),
                "physiology_trace_sha256": sha256_file(segment_dir / "resting-coupled.csv"),
                "surface_trace_sha256": sha256_file(segment_dir / "resting-surface-audit.csv"),
                "aggregate_support_impulses_sha256": sha256_file(segment_dir / "resting-com-support-impulses.csv"),
                "com_momentum_diagnostics_sha256": sha256_file(segment_dir / "resting-com-momentum-diagnostic.csv"),
                "scope": ("Retained 752 run reported physiology, surface, and aggregate contact diagnostics with "
                          "the segment-8 observer schedule for its 2 s, 8 ms condition. It does not qualify 8 ms "
                          "temporal accuracy or the 310 s pair."),
                "displayed_surface_frames": segment_surface_metrics["displayed_accepted_frames"],
                "whole_mesh_triangles_checked_per_frame": segment_surface_metrics["whole_mesh_area_audit"]["triangles_checked_per_frame"],
                "maximum_functional_volume_relative_error": segment_surface_metrics["maximum_rendered_functional_volume_relative_error"],
            },
            **modern_manifest,
        }
        return manifest, list(dict.fromkeys([*(str(path.resolve()) for path in segment_required),
                                             *modern_artifacts]))
    q_env = q_full.get("environment", {})
    q_run = q_full.get("run_configuration", {})
    if (q_env.get("NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT") != "1" or
            q_run.get("full_accepted_q_audit") is not True or
            q_run.get("requested_dt_s") != "0.002" or
            q_run.get("requested_step_count") != 1000 or
            q_run.get("presentation_period_s") != 0.064 or
            q_run.get("transaction_probe") is not True):
        raise ValueError("full-q reference no longer matches the retained 801 2 ms audit")
    q_dir = q_path.parent
    q_csv = q_dir / "resting-com-q-integration.csv"
    q_surface = q_dir / "resting-surface-audit.csv"
    if q_csv.is_symlink() or not q_csv.is_file():
        raise ValueError("full-q reference must retain its accepted 2 ms q-audit CSV")
    with q_csv.open(encoding="utf-8") as stream:
        q_audit_rows = sum(1 for _ in stream) - 1
    if q_audit_rows != 1000:
        raise ValueError("full-q reference must retain all 1000 accepted 2 ms q-audit rows")
    if q_surface.is_symlink() or not q_surface.is_file():
        raise ValueError("full-q reference must retain its 64 ms presentation surface trace")
    with q_surface.open(newline="", encoding="utf-8") as stream:
        surface_rows = list(csv.DictReader(stream))
    surface_steps = [int(row["step"]) for row in surface_rows]
    if (len(surface_steps) != 33 or surface_steps[0] != 0 or surface_steps[-1] != 999 or
            any(b - a > 32 for a, b in zip(surface_steps, surface_steps[1:]))):
        raise ValueError("full-q reference must retain the 64 ms presentation schedule")
    expected_q = {"path": str(q_path), "sha256": sha256_file(q_path)}
    evidence = runtime.get("evidence", {}).get(
        "integrated-final-runtime-2ms-check-801/execution.json", {})
    if (evidence.get("path") != str(q_dir / "execution.json") or
            evidence.get("sha256") != expected_q["sha256"] or
            runtime.get("integrated_2ms_check", {}).get("accepted_steps") != 1000):
        raise ValueError("runtime correctness pin does not match the retained 801 full-q run")
    segment_dir = segment_path.parent
    segment_surface_metrics = native_surface_trace_consistency(
        segment_dir / "resting-surface-audit.csv", 250, 0.008, require_whole_mesh=True)
    q_surface_metrics = native_surface_trace_consistency(
        q_surface, 1000, 0.002, require_whole_mesh=True)
    required_files = [
        segment_path, segment_dir / "resting-coupled.csv",
        segment_dir / "resting-surface-audit.csv",
        segment_dir / "resting-com-support-impulses.csv",
        segment_dir / "resting-com-momentum-diagnostic.csv",
        q_path, q_dir / "resting-coupled.csv", q_csv, q_surface, runtime_path,
    ]
    for path in required_files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing regular audit-schedule reference artifact: {path}")
    manifest = {
        "segment8_schedule_reference": {
            "execution_json": str(segment_path),
            "execution_sha256": sha256_file(segment_path),
            "physiology_trace_sha256": sha256_file(segment_dir / "resting-coupled.csv"),
            "surface_trace_sha256": sha256_file(segment_dir / "resting-surface-audit.csv"),
            "aggregate_support_impulses_sha256": sha256_file(segment_dir / "resting-com-support-impulses.csv"),
            "com_momentum_diagnostics_sha256": sha256_file(segment_dir / "resting-com-momentum-diagnostic.csv"),
            "scope": ("Retained 752 run reported exact physiology/surface outputs and aggregate contact "
                      "extrema/impulse accounting while using the segment-8 observer schedule for its 2 s, "
                      "8 ms condition; its displayed frames retain whole-mesh area and functional-volume checks. "
                      "It does not qualify 8 ms temporal accuracy or the 310 s pair."),
            "displayed_surface_frames": segment_surface_metrics["displayed_accepted_frames"],
            "whole_mesh_triangles_checked_per_frame": segment_surface_metrics["whole_mesh_area_audit"]["triangles_checked_per_frame"],
            "maximum_functional_volume_relative_error": segment_surface_metrics["maximum_rendered_functional_volume_relative_error"]},
        "full_q_2ms_reference": {
            "execution_json": str(q_path), "execution_sha256": sha256_file(q_path),
            "q_integration_trace_sha256": sha256_file(q_csv),
            "physiology_trace_sha256": sha256_file(q_dir / "resting-coupled.csv"),
            "surface_trace_sha256": sha256_file(q_surface),
            "runtime_correctness_pin": str(runtime_path),
            "runtime_correctness_pin_sha256": sha256_file(runtime_path),
            "full_q_audit_rows": q_audit_rows,
            "displayed_surface_frames": len(surface_steps),
            "whole_mesh_triangles_checked_per_frame": q_surface_metrics["whole_mesh_area_audit"]["triangles_checked_per_frame"],
            "maximum_functional_volume_relative_error": q_surface_metrics["maximum_rendered_functional_volume_relative_error"],
            "presentation_period_s": 0.064,
            "scope": ("Retained 801 run has 1000 accepted full-q audit rows at 2 ms, whole-mesh area and "
                      "functional-volume checks on its 33 displayed surface frames, and a 64 ms display cadence; "
                      "it is a bounded 2 s runtime check, not 310 s endurance or physiological qualification.")},
    }
    artifacts = [str(path) for path in required_files]
    return manifest, artifacts


def prepare_native_310s(args: argparse.Namespace) -> Path:
    """Create the fixed 310 s, 2 ms baseline/intervention v2 plan; never launch or register."""
    raw_paths = [Path(args.invocation), Path(args.source_hashes), Path(args.source_revisions)]
    for path in raw_paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"310 s preparation input must be a regular file: {path}")
    invocation_path, source_hashes_path, source_revisions_path = (
        path.resolve() for path in raw_paths)
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    validate_native_310s_invocation(invocation)
    run_metadata_path = invocation_path.with_name("run-metadata.json")
    if run_metadata_path.is_symlink() or not run_metadata_path.is_file():
        raise ValueError("310 s preparation requires the completed Human owner run-metadata receipt")
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    if (run_metadata.get("exit_code") != 0 or
            run_metadata.get("loaded_metal_runtime", {}).get("verified") is not True or
            run_metadata.get("source_files_changed_during_run") != [] or
            run_metadata.get("argv") != invocation.get("argv") or
            run_metadata.get("asset_sha256") != invocation.get("asset_sha256") or
            run_metadata.get("environment") != invocation.get("environment")):
        raise ValueError("completed Human owner run does not verify the selected invocation and loaded runtime")
    native_log_path = invocation_path.with_name("native.log")
    if native_log_path.is_symlink() or not native_log_path.is_file():
        raise ValueError("completed Human owner run is missing its native log")
    native_preflight = native_scene_summary(native_log_path.read_text(encoding="utf-8", errors="replace"))
    try:
        requested_steps = int(invocation["argv"][invocation["argv"].index("--muscle-step-count") + 1])
        requested_dt = float(invocation["argv"][invocation["argv"].index("--muscle-step-seconds") + 1])
    except (ValueError, IndexError, TypeError, KeyError) as exc:
        raise ValueError("Human owner receipt does not declare its actual preflight horizon") from exc
    if (native_preflight["accepted_steps"] != requested_steps or requested_dt != NATIVE_310S_DT or
            abs(native_preflight["simulated_s"] - requested_steps * requested_dt) > 1e-5 or
            native_preflight["world_fingerprint"] != args.world_fingerprint or
            native_preflight["coupled_program_fingerprint"] != args.control_program_fingerprint):
        raise ValueError("completed Human owner preflight differs from the pinned 2 ms control program")
    if not isinstance(json.loads(source_hashes_path.read_text(encoding="utf-8")), dict):
        raise ValueError("source hash inventory must be a JSON object")
    if not source_hashes_path.is_file() or not source_revisions_path.is_file():
        raise ValueError("310 s preparation requires explicit source and revision pins")
    references, reference_artifacts = native_310s_reference_manifest(args)
    args.steps = NATIVE_310S_STEPS
    args.dt = NATIVE_310S_DT
    args.start_s = NATIVE_310S_START_S
    args.end_s = NATIVE_310S_END_S
    args.scale = NATIVE_310S_SCALE
    args.window_s = NATIVE_310S_WINDOW_S
    source_hashes = json.loads(source_hashes_path.read_text(encoding="utf-8"))
    if not isinstance(source_hashes, dict):
        raise ValueError("source hash inventory must be a JSON object")
    output = Path(args.directory).resolve()
    identity, calibration, plan = native_plan_components(
        args, invocation, source_hashes, Path(__file__).resolve())
    modern_full_q = "run_metadata_path" in references["full_q_2ms_reference"]
    if modern_full_q:
        q_reference_scope = (
            "The separately pinned modern 10000-root 2 ms full-q reference is a bounded "
            "runtime integration check, not a long-run q audit.")
        q_limitation = (
            "The long pair disables the full q integration audit; its separately pinned modern reference "
            "retains 10000 roots at 2 ms but is not 310 s endurance or physiological/anatomical qualification.")
    else:
        q_reference_scope = (
            "The full-q audit is separately pinned to the 2 ms 1000-step 801 reference.")
        q_limitation = (
            "The long pair disables the full q integration audit; the separate 801 reference is a "
            "1000-root 2 ms full-q check.")
    audit_schedule = {
        "physical_timestep_s": NATIVE_310S_DT,
        "accepted_native_steps": NATIVE_310S_STEPS,
        "duration_s": NATIVE_310S_STEPS * NATIVE_310S_DT,
        "initialization_exclusion_s": 10.0,
        "post_initialization_observation_s": 300.0,
        "body_physiology_controller_update": "every accepted native root at 2 ms",
        "presentation_period_s": 0.064,
        "accepted_com_momentum_audit": {
            "enabled": True, "segment_steps": 8,
            "scope": "diagnostic aggregation only; no change to physical integration cadence"},
        "accepted_q_integration_audit": {
            "enabled": False,
            "scope": "disabled for the long pair; " + q_reference_scope},
        "presented_surface_geometry_audit": {
            "enabled": True, "cadence_s": 0.064,
            "scope": ("At every scheduled displayed accepted state, retain the complete visible-surface skin/bed "
                      "and finiteness checks, whole-mesh triangle area counts, functional geometry status and "
                      "functional-volume consistency. This samples presentation states; it is not a per-root "
                      "q integration audit or a tissue-interface qualification.")},
        "references": references,
    }
    identity["configuration"]["audit_schedule"] = audit_schedule
    identity["configuration"]["physical_step_contract"] = (
        "The body, circulation, respiration and brain controller continue to execute each accepted 2 ms "
        "native root. COM audit segments and 64 ms display cadence are observer schedules, not timestep coarsening.")
    plan["design"]["native_audit_schedule"] = audit_schedule
    plan["design"]["native_step_contract"] = identity["configuration"]["physical_step_contract"]
    plan.setdefault("validity", []).extend([
        {"path": ["timestep_s"], "equals": NATIVE_310S_DT},
        {"path": ["post_initialization_cycle_coverage", "passed"], "equals": True}])
    plan["artifacts"] = list(dict.fromkeys([*plan["artifacts"], *reference_artifacts]))
    plan["limitations"] += (
        " The 310 s pair is planned at 155000 physical 2 ms roots after a 10 s initialization exclusion; "
        "the 64 ms movie cadence and eight-root COM diagnostic segments do not alter physics/controller timestep. "
        + q_limitation + " "
        "The 752 reference only supports segment-8 observer scheduling at its 8 ms condition. "
        "Neither reference demonstrates 310 s endurance or physiological/anatomical qualification. "
        "The duration/cycle gate requires 300 observed seconds after initialization, repeated complete breaths, "
        "repeated filling/ejection cycles with positive stroke volume, and positive forward ejection into both "
        "circulations. These numerical coverage checks do not establish physiological plausibility or anatomy.")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "native-build-identity.json", identity)
    write_json(output / "model.json", plan["model"])
    write_json(output / "calibration.json", calibration)
    write_json(output / "plan.json", plan)
    print(output / "plan.json")
    return output / "plan.json"


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
                     {"path": ["dense45"], "equals": True}, {"path": ["brain_control"], "equals": True}],
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
    native_run.add_argument("--native-build-identity", required=True,
                            help="registered native-build-identity.json from the prepared paired plan")
    native_run.add_argument("--native-build-identity-sha256", required=True,
                            help="exact prepared identity digest embedded in the registered trial command")
    native_run.add_argument("--unit-id", required=True)
    native_run.add_argument("--world-fingerprint", required=True, help="expected fingerprint for this specific arm")
    native_run.add_argument("--program-fingerprint", required=True,
                            help="expected coupled body/controller program fingerprint, including this arm's intervention")
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
    native_prep = sub.add_parser("prepare-native", help="write a v2 plan draft for the frozen anatomical native scene; never registers or launches")
    native_prep.add_argument("--repository", required=True, help="existing Numi Lab Git owner checkout; evidence output is not a repository")
    native_prep.add_argument("--directory", required=True)
    native_prep.add_argument("--invocation", required=True, help="exact baseline native invocation receipt with asset_sha256 bindings")
    native_prep.add_argument("--source-hashes", required=True, help="JSON map of exact compiled source paths to SHA-256")
    native_prep.add_argument("--source-revisions", required=True,
                             help="JSON map for numi-lab, numilab-human, and numi-brain, each with revision and diff_sha256")
    native_prep.add_argument("--parser-fixture", required=True)
    native_prep.add_argument("--world-fingerprint", required=True, help="actual owner-reported vascular world identity")
    native_prep.add_argument("--control-program-fingerprint", required=True, help="actual owner-reported control program identity")
    native_prep.add_argument("--treatment-program-fingerprint", required=True, help="actual owner-reported intervention program identity")
    native_prep.add_argument("--device", default="Apple M4 Pro")
    native_prep.add_argument("--steps", type=int, default=160000)
    native_prep.add_argument("--dt", type=float, default=0.002)
    native_prep.add_argument("--start-s", type=float, default=60.0)
    native_prep.add_argument("--end-s", type=float, default=100.0)
    native_prep.add_argument("--scale", type=float, default=0.5)
    native_prep.add_argument("--window-s", type=float, default=30.0)
    native_310s = sub.add_parser("prepare-native-310s",
                                 help="prepare fixed 310 s paired study plan at 2 ms; never registers or launches")
    native_310s.add_argument("--repository", required=True)
    native_310s.add_argument("--directory", required=True)
    native_310s.add_argument("--invocation", required=True,
                             help="genuine Human owner invocation.json from the selected 2 ms GPU path")
    native_310s.add_argument("--source-hashes", required=True)
    native_310s.add_argument("--source-revisions", required=True)
    native_310s.add_argument("--parser-fixture", required=True)
    native_310s.add_argument("--world-fingerprint", required=True)
    native_310s.add_argument("--control-program-fingerprint", required=True)
    native_310s.add_argument("--treatment-program-fingerprint", required=True)
    native_310s.add_argument("--device", default="Apple M4 Pro")
    native_310s.add_argument("--segment8-reference", default=
        "/Users/n/numi-human-resting-evidence-20261005/integrated-parallel-contact-batched-752/execution.json")
    native_310s.add_argument("--full-q-reference", default=
        "/Users/n/numi-human-resting-evidence-20261005/integrated-final-runtime-2ms-check-801/execution.json")
    native_310s.add_argument("--full-q-verification", default=None,
                             help="existing terminal-cycle verification JSON for a modern run-metadata full-q reference")
    native_310s.add_argument("--runtime-correctness-reference", default=
        "/Users/n/numi-human-performance-source-014/docs/evidence/human-resting/2026-10-07-native-runtime.json")
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
        elif args.command == "prepare-native":
            prepare_native(args)
        elif args.command == "prepare-native-310s":
            prepare_native_310s(args)
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
