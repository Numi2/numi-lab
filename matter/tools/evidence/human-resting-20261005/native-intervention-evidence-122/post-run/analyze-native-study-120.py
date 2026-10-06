"""Post-run diagnostics for the retained native study; never advances simulation."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("study_owner", args.helper)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    registration_path = args.study / "registration.json"
    plan = json.loads(registration_path.read_text())["payload"]["plan"]
    observations = {}
    report = {
        "qualification": "Post-run numerical and descriptive diagnostics only. The frozen registered study remains the primary intervention analysis. These results do not qualify anatomy, stationary rest, physiology, or clinical use.",
        "analysis_script_sha256": sha(__file__),
        "existing_analysis_owner": {"path": str(args.helper), "sha256": sha(args.helper)},
        "registration_sha256": sha(registration_path),
        "arms": {},
    }
    for arm in ("resting-baseline", "resting-drive-half"):
        root = args.study / "trials" / arm / "output" / "scene"
        observation_path = root / "intervention-observation.json"
        if not observation_path.is_file():
            continue
        observation = json.loads(observation_path.read_text())
        observations[arm] = observation
        invocation = json.loads((root / "invocation.json").read_text())
        command = invocation["argv"]
        parameters = Path(command[command.index("--resting-scene") + 2])
        windows = {name: observation[name + "_window_s"] for name in ("pre", "dose", "recovery")}
        result = owner.native_respiration_trace_consistency(root / "resting-coupled.csv", parameters, windows)
        with (root / "resting-surface-audit.csv").open(newline="") as stream:
            rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
        body_windows = {}
        for name, (start, end) in windows.items():
            samples = [row for row in rows if start <= row["time_s"] <= end]
            assert len(samples) >= 2
            times = [row["time_s"] for row in samples]
            t_mean = math.fsum(times) / len(times)
            t_variance = math.fsum((time - t_mean) ** 2 for time in times)
            assert t_variance > 0
            positions = [[row["body_com_" + axis + "_m"] for row in samples] for axis in "xyz"]
            means = [math.fsum(values) / len(values) for values in positions]
            slopes = [math.fsum((time - t_mean) * (position - mean) for time, position in zip(times, values)) / t_variance
                      for values, mean in zip(positions, means)]
            masses = [row["represented_body_mass_kg"] for row in samples]
            assert all(math.isfinite(value) for values in positions for value in values)
            body_windows[name] = {
                "window_s": [start, end], "samples": len(samples),
                "first_m": [values[0] for values in positions],
                "last_m": [values[-1] for values in positions],
                "range_m": [max(values) - min(values) for values in positions],
                "linear_slope_m_per_s": slopes,
                "linear_slope_norm_m_per_s": math.sqrt(math.fsum(value * value for value in slopes)),
                "represented_mass_range_kg": [min(masses), max(masses)],
            }
        result["center_of_mass_windows"] = body_windows
        result["artifact_sha256"] = {name: sha(root / name) for name in
            ("resting-coupled.csv", "resting-surface-audit.csv", "invocation.json", "intervention-observation.json")}
        report["arms"][arm] = result
    assert report["arms"], "No completed native arm exists"
    if len(report["arms"]) == 2:
        paths = [args.study / "trials" / arm / "output" / "scene" / "resting-coupled.csv" for arm in report["arms"]]
        with paths[0].open(newline="") as a, paths[1].open(newline="") as b:
            rows_a, rows_b = list(csv.DictReader(a)), list(csv.DictReader(b))
        start = windows["pre"][1]
        before_a = [row for row in rows_a if float(row["time_s"]) < start]
        before_b = [row for row in rows_b if float(row["time_s"]) < start]
        report["pre_intervention_exact_trace_comparison"] = {
            "time_exclusive_upper_bound_s": start,
            "control_rows": len(before_a), "treatment_rows": len(before_b),
            "all_exported_fields_equal": before_a == before_b,
            "qualification": "Equality of retained CSV fields before intervention; not whole checkpoint or archive identity.",
        }
        control, treatment = (observations[arm] for arm in ("resting-baseline", "resting-drive-half"))
        differences = {}
        for phase in ("pre", "dose", "recovery"):
            differences[phase] = {key: treatment[key] - control[key] for key in (
                "PaCO2_" + phase + "_mean_mmhg", "PaO2_" + phase + "_mean_mmhg",
                "SaO2_" + phase + "_mean", "inspiratory_minute_ventilation_" + phase + "_L_min")}
        recovery = differences["recovery"]
        margins = plan["secondary_predictions"]["late_recovery_equivalence"]
        ventilation_difference = abs(recovery["inspiratory_minute_ventilation_recovery_L_min"]) / control["inspiratory_minute_ventilation_recovery_L_min"]
        report["paired_diagnostics"] = {
            "treatment_minus_control": differences,
            "dose_ventilation_decreased": differences["dose"]["inspiratory_minute_ventilation_dose_L_min"] < 0,
            "PaCO2_difference_in_differences_mmhg": differences["dose"]["PaCO2_dose_mean_mmhg"] - differences["pre"]["PaCO2_pre_mean_mmhg"],
            "late_recovery_predeclared_margins": margins,
            "late_recovery_checks": {
                "PaCO2_within_margin": abs(recovery["PaCO2_recovery_mean_mmhg"]) <= margins["PaCO2_absolute_difference_max_mmhg"],
                "PaO2_within_margin": abs(recovery["PaO2_recovery_mean_mmhg"]) <= margins["PaO2_absolute_difference_max_mmhg"],
                "ventilation_relative_difference": ventilation_difference,
                "ventilation_within_margin": ventilation_difference <= margins["inspiratory_minute_ventilation_relative_difference_max_fraction"],
            },
            "qualification": "Fixed-window deterministic paired measurements and predeclared descriptive recovery margins; these are not population estimates or clinical cutoffs.",
        }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha(args.output),
                      "completed_arms": list(report["arms"]),
                      "body_windows": {arm: value["center_of_mass_windows"] for arm, value in report["arms"].items()}}, indent=2))


if __name__ == "__main__":
    main()
