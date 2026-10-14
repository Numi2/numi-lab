#!/usr/bin/env python3
"""Run and measure one full 60-step native synthetic wound-traction arm."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


FIELDS = (
    "time_s", "bite_force_n", "mean_gap_m", "minimum_mean_gap_m",
    "minimum_signed_gap_m", "maximum_displacement_m", "minimum_J",
    "maximum_residual", "equilibrium_residual", "volume_residual",
    "pressure_residual", "maximum_volume_residual", "active_tetrahedra",
    "mass_kg", "gpu_ms",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene_fingerprint(material_sha256: str) -> str:
    scene = {
        "length_m": 0.030,
        "width_m": 0.024,
        "thickness_m": 0.0015,
        "incision_length_m": 0.016,
        "initial_gap_m": 0.0006,
        "density_kg_m3": 1050.0,
        "mesh_cells": [24, 8, 2],
        "step_count": 60,
        "timestep_s": 0.001,
        "contact_slop_m": 0.00001,
        "gravity_m_s2": [0.0, 0.0, 0.0],
        "material_sha256": material_sha256,
    }
    payload = json.dumps(scene, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def summary_fields(output: str) -> dict[str, str]:
    lines = [line.strip() for line in output.splitlines()
             if line.strip().startswith("synthetic_skin_wound_diagnostic=")]
    if len(lines) != 1 or not lines[0].startswith("synthetic_skin_wound_diagnostic=ok "):
        raise ValueError("native probe did not emit exactly one successful diagnostic summary")
    return dict(token.split("=", 1) for token in lines[0].split()[1:] if "=" in token)


def load_samples(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not (set(FIELDS) | {"step"}).issubset(reader.fieldnames):
            raise ValueError("native sample CSV is missing required measurement columns")
        return list(reader)


def assess(samples: list[dict[str, str]], force_n: float) -> tuple[list[str], dict[str, float]]:
    errors: list[str] = []
    numbers: list[dict[str, float]] = []
    if len(samples) != 60:
        errors.append(f"sample_count_{len(samples)}_not_60")
    for step, row in enumerate(samples):
        try:
            values = {name: float(row[name]) for name in FIELDS if name != "active_tetrahedra"}
            values["active_tetrahedra"] = float(row["active_tetrahedra"])
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError("nonfinite_measurement")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"step_{step}_invalid_measurement_{exc}")
            continue
        numbers.append(values)
        if int(row["step"]) != step:
            errors.append(f"step_{step}_missing_or_reordered")
        if abs(values["time_s"] - (step + 1) * 0.001) > 1.0e-9:
            errors.append(f"step_{step}_time_mismatch")
        scale = 0.5 * (1.0 - math.cos(math.pi * step / 59.0))
        if abs(values["bite_force_n"] - force_n * scale) > 1.0e-8:
            errors.append(f"step_{step}_force_schedule_mismatch")
        if values["minimum_J"] <= 0.20:
            errors.append(f"step_{step}_determinant_gate")
        if values["minimum_signed_gap_m"] < 0.0:
            errors.append(f"step_{step}_lip_crossing")
        if int(values["active_tetrahedra"]) != 2304:
            errors.append(f"step_{step}_active_tetrahedron_count")
        if values["mass_kg"] <= 0.0 or values["gpu_ms"] < 0.0:
            errors.append(f"step_{step}_invalid_mass_or_time")
        if values["maximum_volume_residual"] > 1.0e-4:
            errors.append(f"step_{step}_native_volume_gate")
    if numbers:
        initial_mass = numbers[0]["mass_kg"]
        if initial_mass <= 0.0 or max(abs(v["mass_kg"] - initial_mass) / initial_mass for v in numbers) > 1.0e-5:
            errors.append("relative_mass_error_gate")
        metrics = {
            "minimum_mean_gap_m": min(v["mean_gap_m"] for v in numbers),
            "final_mean_gap_m": numbers[-1]["mean_gap_m"],
            "minimum_signed_gap_m": min(v["minimum_signed_gap_m"] for v in numbers),
            "maximum_displacement_m": max(v["maximum_displacement_m"] for v in numbers),
            "minimum_J": min(v["minimum_J"] for v in numbers),
            "maximum_residual": max(v["maximum_residual"] for v in numbers),
            "maximum_volume_residual": max(v["maximum_volume_residual"] for v in numbers),
            "maximum_pressure_residual": max(v["pressure_residual"] for v in numbers),
            "mass_relative_error": max(abs(v["mass_kg"] - initial_mass) / initial_mass for v in numbers),
            "total_gpu_ms": sum(v["gpu_ms"] for v in numbers),
            "sample_count": float(len(numbers)),
        }
    else:
        metrics = {"sample_count": 0.0}
    return errors, metrics


def fixture_self_test() -> dict[str, object]:
    samples = []
    for step in range(60):
        force = 0.010 * 0.5 * (1.0 - math.cos(math.pi * step / 59.0))
        samples.append({
            "step": str(step), "time_s": f"{(step + 1) * 0.001:.9f}",
            "bite_force_n": f"{force:.12f}", "mean_gap_m": f"{0.0006 - 0.00002 * step / 59.0:.12f}",
            "minimum_mean_gap_m": f"{0.0006 - 0.00002 * step / 59.0:.12f}",
            "minimum_signed_gap_m": "0.00058", "maximum_displacement_m": "0.00001",
            "minimum_J": "0.999", "maximum_residual": "0.000003",
            "equilibrium_residual": "0.000003", "volume_residual": "0.00002",
            "pressure_residual": "0.000003", "maximum_volume_residual": "0.00002",
            "active_tetrahedra": "2304", "mass_kg": "0.00112", "gpu_ms": "1.0",
        })
    errors, metrics = assess(samples, 0.010)
    if errors or metrics.get("sample_count") != 60.0 or abs(metrics["minimum_mean_gap_m"] - 0.00058) > 1.0e-10:
        raise SystemExit("adapter fixture positive control failed")
    bad = [dict(row) for row in samples]
    bad[17]["maximum_volume_residual"] = "0.00011"
    bad_errors, _ = assess(bad, 0.010)
    if "step_17_native_volume_gate" not in bad_errors:
        raise SystemExit("adapter fixture negative gate failed")
    return {
        "schema": "numi.science.calibration-check.v1",
        "passed": True,
        "checks": ["60-step schedule and metric extraction", "unchanged volume-gate rejection"],
        "synthetic_fixture_only": True,
    }


def emit(record: dict[str, object]) -> None:
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", nargs="?")
    parser.add_argument("probe", nargs="?")
    parser.add_argument("material", nargs="?")
    parser.add_argument("metallib", nargs="?")
    parser.add_argument("force_per_site_n", nargs="?", type=float, choices=(0.0, 0.010))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        emit(fixture_self_test())
        return 0
    if not all((args.run_directory, args.probe, args.material, args.metallib)) or args.force_per_site_n is None:
        parser.error("trial execution requires run_directory, probe, material, metallib and force_per_site_n")

    run_directory = Path(args.run_directory)
    run_directory.mkdir(parents=True, exist_ok=True)
    native_output = run_directory / "native"
    command = [
        str(Path(args.probe)), str(native_output), str(Path(args.material)), str(Path(args.metallib)),
        "--force-per-site-n", "0" if args.force_per_site_n == 0.0 else "0.010",
        "--newton-iterations", "7", "--fgmres-iterations", "20",
    ]
    (run_directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    native_output.mkdir(parents=True, exist_ok=True)
    (run_directory / "native-stdout.log").write_text(result.stdout)
    (run_directory / "native-stderr.log").write_text(result.stderr)
    (run_directory / "native-exit-code.txt").write_text(f"{result.returncode}\n")

    material_sha = sha256(Path(args.material))
    unit = {"scene_fingerprint": scene_fingerprint(material_sha), "deterministic": True}
    record: dict[str, object] = {
        "schema": "numi.science.synthetic-skin-full-horizon-run.v1",
        "valid": False,
        "simulation_only": True,
        "unit": unit,
        "settings": {
            "mesh_cells": [24, 8, 2], "step_count": 60, "timestep_s": 0.001,
            "newton_iterations": 7, "fgmres_max_iterations": 20,
            "contact_slop_m": 0.00001, "initial_wound_gap_m": 0.0006,
            "gravity_m_s2": [0.0, 0.0, 0.0], "force_ramp": "cosine60",
            "force_per_site_n": args.force_per_site_n, "material_sha256": material_sha,
            "solver_policy": "trace-guided-forcing-0.015625-floor-equilibrium-tolerance",
        },
        "native_exit_code": result.returncode,
        "force_ramp": "cosine60",
        "force_per_site_n": args.force_per_site_n,
        "thread_present": False,
        "needle_present": False,
        "robot_execution": False,
        "physical_tissue_calibration": "none",
        "sample_count": 0,
        "all_gates_passed": False,
        "reason": "native_probe_failed_or_output_invalid",
    }
    samples_path = native_output / "traction-samples.csv"
    if result.returncode != 0:
        if samples_path.is_file():
            try:
                record["sample_count"] = sum(1 for _ in samples_path.open(encoding="utf-8")) - 1
            except OSError:
                pass
        record["reason"] = "native_exit_" + str(result.returncode)
        emit(record)
        return 0
    try:
        summary = summary_fields(result.stdout)
        samples = load_samples(samples_path)
        errors, metrics = assess(samples, args.force_per_site_n)
        expected_arm = "zero_force_control" if args.force_per_site_n == 0.0 else "paired_bite_traction"
        expected = {
            "arm": expected_arm, "mesh": "24x8x2", "force_ramp": "cosine60",
            "newton_iterations": "7", "execution_steps": "60", "protocol_steps": "60",
            "physical_calibration": "none", "thread_present": "false", "robot_execution": "false",
        }
        for key, value in expected.items():
            if summary.get(key) != value:
                errors.append("summary_" + key + "_mismatch")
        if float(summary.get("bite_force_per_site_n", "nan")) != args.force_per_site_n:
            errors.append("summary_force_mismatch")
        if float(summary.get("maximum_volume_residual", "inf")) > 1.0e-4:
            errors.append("summary_native_volume_gate")
        if float(summary.get("minimum_J", "-inf")) <= 0.20:
            errors.append("summary_native_determinant_gate")
        if float(summary.get("mass_relative_error", "inf")) > 1.0e-5:
            errors.append("summary_native_mass_gate")
        if errors:
            record["reason"] = ";".join(errors)
            record["sample_count"] = int(metrics.get("sample_count", 0.0))
            record.update({key: value for key, value in metrics.items() if key != "sample_count"})
            emit(record)
            return 0
        record.update(metrics)
        record["valid"] = True
        record["all_gates_passed"] = True
        record["reason"] = "all_60_native_samples_passed_unchanged_gates"
        record["summary"] = summary
    except Exception as exc:  # A parse failure is retained as invalid evidence.
        record["reason"] = "measurement_parse_failure:" + type(exc).__name__ + ":" + str(exc)
    emit(record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
