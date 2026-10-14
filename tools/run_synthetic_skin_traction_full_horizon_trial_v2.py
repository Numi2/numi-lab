#!/usr/bin/env python3
"""Collect the native wound-traction probe's actual CSV and summary schema.

The control-reparse mode is read-only with respect to its source trial. It
copies its raw native artifacts into the new notebook run and recalculates the
same gates, preserving an explicit pointer to the original process receipt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


FIELDS = (
    "time_s", "bite_force_n", "mean_gap_m", "minimum_mean_gap_m",
    "center_gap_m", "minimum_signed_gap_m", "maximum_displacement_m",
    "minimum_J", "maximum_residual", "equilibrium_residual",
    "relative_correction", "volume_residual", "pressure_residual",
    "maximum_volume_residual", "active_tetrahedra", "mass_kg", "gpu_ms",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_fingerprint(material_sha256: str) -> str:
    scene = {
        "length_m": 0.030, "width_m": 0.024, "thickness_m": 0.0015,
        "incision_length_m": 0.016, "initial_gap_m": 0.0006,
        "density_kg_m3": 1050.0, "mesh_cells": [24, 8, 2],
        "step_count": 60, "timestep_s": 0.001, "contact_slop_m": 0.00001,
        "gravity_m_s2": [0.0, 0.0, 0.0], "material_sha256": material_sha256,
    }
    payload = json.dumps(scene, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def parse_summary(output: str) -> dict[str, str]:
    lines = [line.strip() for line in output.splitlines()
             if line.strip().startswith("synthetic_skin_wound_traction=")]
    if len(lines) != 1 or not lines[0].startswith("synthetic_skin_wound_traction=ok "):
        raise ValueError("native probe did not emit exactly one successful traction summary")
    return dict(token.split("=", 1) for token in lines[0].split()[1:] if "=" in token)


def parse_samples(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not ({"step"} | set(FIELDS)).issubset(reader.fieldnames):
            raise ValueError("native traction CSV is missing required measurement columns")
        return list(reader)


def measure(native_output: Path, expected_force_n: float, material_sha256: str) -> dict[str, object]:
    exit_file = native_output.parent / "native-exit-code.txt"
    stdout_file = native_output.parent / "native-stdout.log"
    if not exit_file.is_file() or int(exit_file.read_text().strip()) != 0:
        raise ValueError("retained native process did not exit successfully")
    summary = parse_summary(stdout_file.read_text(encoding="utf-8"))
    samples = parse_samples(native_output / "traction-samples.csv")
    if len(samples) != 60:
        raise ValueError(f"sample count {len(samples)} is not 60")

    numbers: list[dict[str, float]] = []
    for index, row in enumerate(samples):
        values = {name: float(row[name]) for name in FIELDS}
        values["active_tetrahedra"] = int(values["active_tetrahedra"])
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"step {index} contains a non-finite measurement")
        if int(row["step"]) != index or abs(values["time_s"] - (index + 1) * 0.001) > 1.0e-9:
            raise ValueError(f"step {index} is missing, reordered, or has the wrong time")
        scale = 0.5 * (1.0 - math.cos(math.pi * index / 59.0))
        if abs(values["bite_force_n"] - expected_force_n * scale) > 1.0e-8:
            raise ValueError(f"step {index} has the wrong applied-force schedule")
        if values["active_tetrahedra"] != 2304:
            raise ValueError(f"step {index} does not retain all 2,304 tetrahedra")
        if values["minimum_J"] <= 0.20:
            raise ValueError(f"step {index} fails the determinant gate")
        if values["minimum_signed_gap_m"] < 0.0:
            raise ValueError(f"step {index} has crossed wound lips")
        if values["maximum_volume_residual"] > 1.0e-4:
            raise ValueError(f"step {index} fails the unchanged native volume gate")
        if values["mass_kg"] <= 0.0 or values["gpu_ms"] < 0.0:
            raise ValueError(f"step {index} has invalid mass or timing")
        numbers.append(values)

    expected_arm = "zero_force_control" if expected_force_n == 0.0 else "paired_bite_traction"
    expected_summary = {
        "arm": expected_arm,
        "mesh": "24x8x2",
        "top_lip_pairs": "12",
        "force_ramp": "cosine60",
        "newton_iterations": "7",
        "physical_calibration": "none",
        "thread_present": "false",
        "robot_execution": "false",
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise ValueError(f"native summary {key} mismatch")
    if float(summary.get("bite_force_per_site_n", "nan")) != expected_force_n:
        raise ValueError("native summary force mismatch")

    mass_start = numbers[0]["mass_kg"]
    mass_error = abs(numbers[-1]["mass_kg"] - mass_start) / mass_start
    if mass_error > 1.0e-5 or float(summary.get("mass_relative_error", "inf")) > 1.0e-5:
        raise ValueError("native mass-conservation gate failed")
    if expected_force_n == 0.0 and float(summary.get("maximum_mean_gap_drift_m", "inf")) > 1.0e-6:
        raise ValueError("zero-force control drift exceeded 1 um")

    maximum_residual = max(value["maximum_residual"] for value in numbers)
    if abs(maximum_residual - float(summary["maximum_solver_residual"])) > 1.0e-8:
        raise ValueError("CSV and summary solver residual disagree")
    return {
        "schema": "numi.science.synthetic-skin-full-horizon-run.v2",
        "valid": True,
        "all_gates_passed": True,
        "native_exit_code": 0,
        "sample_count": len(numbers),
        "execution_steps": len(numbers),
        "protocol_steps": 60,
        "simulation_only": True,
        "unit": {"scene_fingerprint": scene_fingerprint(material_sha256), "deterministic": True},
        "settings": {
            "mesh_cells": [24, 8, 2], "step_count": 60, "timestep_s": 0.001,
            "newton_iterations": 7, "fgmres_max_iterations": 20,
            "contact_slop_m": 1.0e-5, "initial_wound_gap_m": 0.0006,
            "gravity_m_s2": [0.0, 0.0, 0.0], "force_ramp": "cosine60",
            "force_per_site_n": expected_force_n,
            "material_sha256": material_sha256,
            "solver_policy": "trace-guided-forcing-0.015625-floor-equilibrium-tolerance",
        },
        "force_ramp": "cosine60",
        "force_per_site_n": expected_force_n,
        "thread_present": False,
        "needle_present": False,
        "robot_execution": False,
        "physical_tissue_calibration": "none",
        "minimum_mean_gap_m": min(value["minimum_mean_gap_m"] for value in numbers),
        "final_mean_gap_m": numbers[-1]["mean_gap_m"],
        "minimum_signed_gap_m": min(value["minimum_signed_gap_m"] for value in numbers),
        "maximum_displacement_m": max(value["maximum_displacement_m"] for value in numbers),
        "minimum_J": min(value["minimum_J"] for value in numbers),
        "maximum_residual": maximum_residual,
        "maximum_volume_residual": max(value["maximum_volume_residual"] for value in numbers),
        "secondary_5e-5_volume_target_passed": max(value["maximum_volume_residual"] for value in numbers) <= 5.0e-5,
        "maximum_pressure_residual": max(value["pressure_residual"] for value in numbers),
        "mass_relative_error": max(mass_error, float(summary["mass_relative_error"])),
        "total_gpu_ms": sum(value["gpu_ms"] for value in numbers),
        "summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--reparse-traction-samples", type=Path,
                      help="read-only reparse using a bound retained traction-samples.csv file")
    mode.add_argument("--execute-native", action="store_true")
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--material", type=Path)
    parser.add_argument("--metallib", type=Path)
    parser.add_argument("--force-per-site-n", type=float, choices=(0.0, 0.010))
    args = parser.parse_args()

    run_directory = args.run_directory.resolve()
    run_directory.mkdir(parents=True, exist_ok=True)
    native_output = run_directory / "native"
    source_run: Path | None = None
    if args.reparse_traction_samples is not None:
        if args.material is None or not args.material.is_file():
            parser.error("control reparse requires --material for source-bound unit identity")
        source_native = args.reparse_traction_samples.resolve().parent
        if source_native.name != "native":
            parser.error("control reparse must point to a retained native/traction-samples.csv")
        source_run = source_native.parent
        if args.reparse_traction_samples.name != "traction-samples.csv":
            parser.error("control reparse input must be traction-samples.csv")
        shutil.copytree(source_native, native_output)
        for name in ("native-exit-code.txt", "native-stdout.log", "native-stderr.log"):
            shutil.copy2(source_run / name, run_directory / name)
        original_receipt = source_run.parent / "process.json"
        (run_directory / "reparse-source.json").write_text(json.dumps({
            "method": "read-only reparse of retained native control",
            "source_trial_output": str(source_run),
            "source_process_receipt": str(original_receipt),
            "source_process_receipt_sha256": sha256(original_receipt),
            "source_traction_samples_sha256": sha256(source_native / "traction-samples.csv"),
            "source_lip_geometry_sha256": sha256(source_native / "lip-geometry.csv"),
        }, indent=2, sort_keys=True) + "\n")
        force = 0.0
        method = "retained_native_control_reparsed"
    else:
        if not all((args.probe, args.material, args.metallib, args.force_per_site_n is not None)):
            parser.error("native execution requires --probe, --material, --metallib and --force-per-site-n")
        force = args.force_per_site_n
        command = [str(args.probe.resolve()), str(native_output), str(args.material.resolve()),
                   str(args.metallib.resolve()), "--force-per-site-n", f"{force:.3f}",
                   "--newton-iterations", "7", "--fgmres-iterations", "20"]
        (run_directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        native_output.mkdir(parents=True, exist_ok=True)
        (run_directory / "native-stdout.log").write_text(result.stdout)
        (run_directory / "native-stderr.log").write_text(result.stderr)
        (run_directory / "native-exit-code.txt").write_text(f"{result.returncode}\n")
        method = "fresh_native_execution"

    record: dict[str, object] = {
        "collection_method": method,
        "force_per_site_n": force,
        "reason": "all_60_native_samples_passed_unchanged_gates",
    }
    try:
        record.update(measure(native_output, force, sha256(args.material)))
        if source_run is not None:
            record["reparsed_source_trial_output"] = str(source_run)
            record["reparsed_source_trial_receipt_sha256"] = sha256(source_run.parent / "process.json")
        emit = json.dumps(record, sort_keys=True, separators=(",", ":"))
        (run_directory / "stdout.json").write_text(emit + "\n")
        print(emit)
        return 0
    except Exception as exc:
        record.update({
            "schema": "numi.science.synthetic-skin-full-horizon-run.v2",
            "valid": False,
            "all_gates_passed": False,
            "reason": "measurement_parse_failure:" + type(exc).__name__ + ":" + str(exc),
        })
        emit = json.dumps(record, sort_keys=True, separators=(",", ":"))
        (run_directory / "stdout.json").write_text(emit + "\n")
        print(emit)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
