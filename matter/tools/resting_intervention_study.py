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
    build = Path(command[0]).parent.parent
    if Path(command[0]).parent.name != "bin":
        raise ValueError("native invocation does not use the Human owner build layout")
    for relative in ("lib/libmetalrobo.dylib", "shaders/MetalRobo.metallib",
                     "shaders/MetalRoboHyperPolicy.metallib", "shaders/NumiNeuron.metallib",
                     "matter/shaders/HumanRespiration.metallib", "matter/shaders/NumiMatter.metallib",
                     "matter/shaders/NumiMatterPhysicalStateDigest.metallib"):
        if str(build / relative) not in bindings:
            raise ValueError(f"native invocation has an unbound runtime dependency: {build / relative}")
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


def native_surface_trace_consistency(trace: Path, steps: int, dt: float) -> dict[str, Any]:
    """Validate every retained displayed state, whose clock precedes its accepted segment end.

    The renderer captures step n-1 before step n is evaluated and publishes it
    only after that step is accepted. Never pair this trace with the next
    physiology row: its own chamber/lung target columns are authoritative.
    These checks cover numerical geometry consistency, not tissue interfaces.
    """
    expected_steps = [0] + list(range(31, steps, 32))
    if expected_steps[-1] != steps - 1:
        expected_steps.append(steps - 1)
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
            "displayed_state_lag_steps": 1, "whole_body_interfaces_qualified": False,
            "geometry_mode": geometry_mode}
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
    recorded_environment = dict(invocation.get("environment", {}))
    env.update(recorded_environment)
    # This is an output, not a frozen input. A replayed preflight receipt must
    # not let an arm overwrite evidence in the preflight or another trial.
    failure_receipt_key = "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"
    if env.get(failure_receipt_key):
        recorded_environment[failure_receipt_key] = str(output / "common-field-failure.json")
        env[failure_receipt_key] = recorded_environment[failure_receipt_key]
    if any(k.startswith("NUMI_HUMAN_STAND_CPU_") and v == "1" for k, v in env.items()):
        raise ValueError("CPU stepping is not admitted for the native scene")
    output.mkdir()
    write_json(output / "invocation.json", {**invocation, "argv": command,
               "environment": recorded_environment,
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
    result.update(native_body_trace_consistency(output / "resting-coupled.csv", args.steps, args.dt))
    result.update(native_surface_trace_consistency(surfaces, args.steps, args.dt))
    parameters = Path(command[command.index("--resting-scene") + 2])
    result.update(native_respiration_trace_consistency(output / "resting-coupled.csv", parameters,
        {name: result[name + "_window_s"] for name in ("pre", "dose", "recovery")}))
    result.update(native_whole_body_executed=True,
                  common_asset_identity=digest_json(bindings),
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
    for arm in ("control", "treatment"):
        preflight_args = argparse.Namespace(steps=args.steps, dt=args.dt, arm=arm,
                                            start_s=args.start_s, end_s=args.end_s, scale=args.scale)
        native_scene_command(invocation, out / "preflight-output", preflight_args)
    unit_basis = {"source_revisions": source_revisions,
                  "source_files": source_hashes,
                  "common_assets": bindings,
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
