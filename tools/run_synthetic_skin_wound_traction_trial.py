#!/usr/bin/env python3
"""Run one frozen native trial and emit a calibrated scalar JSON receipt."""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


MESH_CELLS = [24, 8, 2]
STEP_COUNT = 60
TIMESTEP_S = 0.001
CONTACT_SLOP_M = 0.00001
INITIAL_GAP_M = 0.0006
EXPECTED_TETRAHEDRA = 6 * MESH_CELLS[0] * MESH_CELLS[1] * MESH_CELLS[2]
NUMERIC_SAMPLE_FIELDS = (
    "time_s", "bite_force_n", "mean_gap_m", "minimum_mean_gap_m",
    "center_gap_m", "minimum_signed_gap_m", "maximum_displacement_m",
    "minimum_J", "maximum_residual", "mass_kg", "gpu_ms",
)
NUMERIC_LIP_FIELDS = (
    "rest_x_m", "rest_z_m", "lower_x_m", "lower_y_m", "lower_z_m",
    "upper_x_m", "upper_y_m", "upper_z_m",
)


def _scene_definition(material_sha256):
    return {
        "length_m": 0.030,
        "width_m": 0.024,
        "thickness_m": 0.0015,
        "incision_length_m": 0.016,
        "initial_gap_m": INITIAL_GAP_M,
        "density_kg_m3": 1050.0,
        "mesh_cells": MESH_CELLS,
        "step_count": STEP_COUNT,
        "timestep_s": TIMESTEP_S,
        "contact_slop_m": CONTACT_SLOP_M,
        "gravity_m_s2": [0.0, 0.0, 0.0],
        "material_sha256": material_sha256,
    }


def scene_fingerprint(material_sha256):
    encoded = json.dumps(
        _scene_definition(material_sha256),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_rows(path, required_fields):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(required_fields).issubset(reader.fieldnames):
            raise ValueError(f"CSV header is missing fields: {path}")
        return list(reader)


def _finite(row, names):
    values = {}
    for name in names:
        value = float(row[name])
        if not math.isfinite(value):
            raise ValueError(f"nonfinite {name}")
        values[name] = value
    return values


def _key_values(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1 or not lines[0].startswith("synthetic_skin_wound_traction=ok "):
        raise ValueError("native trial did not emit exactly one successful summary")
    result = {}
    for token in lines[0].split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def summarize(samples_path, lips_path, native_stdout, material_path, force_n):
    summary = _key_values(native_stdout)
    expected_arm = "zero_force_control" if force_n == 0.0 else "paired_bite_traction"
    expected_mesh = "x".join(str(value) for value in MESH_CELLS)
    if summary.get("arm") != expected_arm or summary.get("mesh") != expected_mesh:
        raise ValueError("native arm or mesh identity differs from the declared trial")
    if summary.get("force_ramp") != "cosine60":
        raise ValueError("native force ramp differs from the preregistered protocol")
    if int(summary["newton_iterations"]) != 7:
        raise ValueError("native nonlinear solver iteration budget changed")
    if float(summary["bite_force_per_site_n"]) != force_n:
        raise ValueError("native force differs from the declared trial")
    if (summary.get("physical_calibration") != "none" or
            summary.get("thread_present") != "false" or
            summary.get("robot_execution") != "false"):
        raise ValueError("native fidelity boundary differs from the registered model")

    sample_rows = _read_rows(samples_path, ("step", *NUMERIC_SAMPLE_FIELDS,
                                              "active_tetrahedra"))
    if len(sample_rows) != STEP_COUNT:
        raise ValueError(f"expected {STEP_COUNT} complete samples; found {len(sample_rows)}")
    samples = []
    for expected_step, row in enumerate(sample_rows):
        if int(row["step"]) != expected_step:
            raise ValueError("sample steps are missing, duplicated, or reordered")
        values = _finite(row, NUMERIC_SAMPLE_FIELDS)
        values["step"] = expected_step
        values["active_tetrahedra"] = int(row["active_tetrahedra"])
        progress = expected_step / (STEP_COUNT - 1.0)
        scale = 0.5 * (1.0 - math.cos(math.pi * progress))
        if abs(values["bite_force_n"] - force_n * scale) > 1.0e-8:
            raise ValueError("sample force does not match the preregistered ramp")
        if values["minimum_J"] <= 0.20:
            raise ValueError("a sample violated the determinant gate")
        if values["minimum_signed_gap_m"] < 0.0:
            raise ValueError("a sample contains crossed wound lips")
        if values["active_tetrahedra"] != EXPECTED_TETRAHEDRA:
            raise ValueError("a sample changed the active tetrahedron count")
        if values["mass_kg"] <= 0.0 or values["gpu_ms"] < 0.0:
            raise ValueError("a sample has invalid mass or GPU time")
        samples.append(values)

    lip_rows = _read_rows(lips_path, ("step", "pair", *NUMERIC_LIP_FIELDS))
    if not lip_rows:
        raise ValueError("lip-geometry CSV is empty")
    pair_ids = set()
    rows_per_step = [0] * STEP_COUNT
    for row in lip_rows:
        step = int(row["step"])
        if not 0 <= step < STEP_COUNT:
            raise ValueError("lip-geometry step is outside the frozen horizon")
        values = _finite(row, NUMERIC_LIP_FIELDS)
        if values["upper_y_m"] < values["lower_y_m"]:
            raise ValueError("lip-geometry CSV contains crossed lips")
        rows_per_step[step] += 1
        pair_ids.add(int(row["pair"]))
    if len(pair_ids) < 8 or len(pair_ids) != int(summary["top_lip_pairs"]):
        raise ValueError("lip-geometry sample count differs from the native preflight")
    if any(count != len(pair_ids) for count in rows_per_step):
        raise ValueError("lip-geometry samples are incomplete")

    gaps = [sample["mean_gap_m"] for sample in samples]
    initial_gap = gaps[0]
    minimum_gap = min(gaps)
    maximum_drift = max(abs(gap - initial_gap) for gap in gaps)
    peak_reduction = initial_gap - minimum_gap
    mass_error = float(summary["mass_relative_error"])
    if not math.isfinite(mass_error) or mass_error > 1.0e-5:
        raise ValueError("native final mass-conservation gate failed")
    if force_n == 0.0 and maximum_drift > 1.0e-6:
        raise ValueError("zero-force control drift exceeded 1 micrometre")
    material_sha = hashlib.sha256(material_path.read_bytes()).hexdigest()
    return {
        "schema": "numi.science.synthetic-skin-traction-run.v1",
        "valid": True,
        "simulation_only": True,
        "unit": {"scene_fingerprint": scene_fingerprint(material_sha),
                 "deterministic": True},
        "settings": {
            "mesh_cells": MESH_CELLS,
            "step_count": STEP_COUNT,
            "timestep_s": TIMESTEP_S,
            "force_ramp": "cosine60",
            "newton_iterations": 7,
            "contact_slop_m": CONTACT_SLOP_M,
            "initial_wound_gap_m": INITIAL_GAP_M,
            "gravity_m_s2": [0.0, 0.0, 0.0],
            "material_sha256": material_sha,
        },
        "force_per_site_n": force_n,
        "sample_count": len(samples),
        "top_lip_pair_count": len(pair_ids),
        "initial_mean_gap_m": initial_gap,
        "minimum_mean_gap_m": minimum_gap,
        "final_mean_gap_m": gaps[-1],
        "peak_gap_reduction_m": peak_reduction,
        "maximum_control_or_loaded_gap_drift_m": maximum_drift,
        "minimum_signed_gap_m": min(row["minimum_signed_gap_m"] for row in samples),
        "maximum_tissue_displacement_m": max(row["maximum_displacement_m"] for row in samples),
        "minimum_J": min(row["minimum_J"] for row in samples),
        "maximum_solver_residual": max(row["maximum_residual"] for row in samples),
        "active_tetrahedra": EXPECTED_TETRAHEDRA,
        "mass_relative_error": mass_error,
        "total_gpu_ms": float(summary["total_gpu_ms"]),
        "physical_tissue_calibration": "none",
        "thread_present": False,
        "needle_present": False,
        "robot_execution": False,
    }


def self_test():
    with tempfile.TemporaryDirectory(prefix="synthetic-skin-adapter-") as raw:
        root = Path(raw)
        samples_path = root / "samples.csv"
        lips_path = root / "lips.csv"
        fields = ["step", *NUMERIC_SAMPLE_FIELDS, "active_tetrahedra"]
        with samples_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for step in range(STEP_COUNT):
                gap = 0.0006 - (0.00000005 * step)
                progress = step / (STEP_COUNT - 1.0)
                force_scale = 0.5 * (1.0 - math.cos(math.pi * progress))
                writer.writerow({
                    "step": step, "time_s": 0.001 * (step + 1),
                    "bite_force_n": 0.01 * force_scale, "mean_gap_m": gap,
                    "minimum_mean_gap_m": gap, "center_gap_m": gap,
                    "minimum_signed_gap_m": gap, "maximum_displacement_m": 0.0,
                    "minimum_J": 0.9, "maximum_residual": 0.001,
                    "active_tetrahedra": EXPECTED_TETRAHEDRA,
                    "mass_kg": 0.0001, "gpu_ms": 0.5,
                })
        with lips_path.open("w", newline="") as stream:
            fields = ["step", "pair", *NUMERIC_LIP_FIELDS]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for step in range(STEP_COUNT):
                for pair in range(8):
                    x = -0.007 + pair * 0.002
                    writer.writerow({
                        "step": step, "pair": pair, "rest_x_m": x,
                        "rest_z_m": 0.00075, "lower_x_m": x,
                        "lower_y_m": -0.0003, "lower_z_m": 0.00075,
                        "upper_x_m": x, "upper_y_m": 0.0003,
                        "upper_z_m": 0.00075,
                    })
        material = root / "material.nmatter"
        material.write_text("fixture-only-material\n")
        stdout = (
            "synthetic_skin_wound_traction=ok arm=paired_bite_traction "
            "mesh=24x8x2 top_lip_pairs=8 bite_force_per_site_n=0.01 "
            "force_ramp=cosine60 newton_iterations=7 "
            "initial_mean_gap_m=0.0006 minimum_mean_gap_m=0.00059705 "
            "final_mean_gap_m=0.00059705 maximum_mean_gap_drift_m=0.00000295 "
            "minimum_signed_gap_m=0.0006 maximum_tissue_displacement_m=0 "
            "minimum_J=0.9 maximum_solver_residual=0.001 "
            "active_tetrahedra=2304 mass_relative_error=0 "
            "total_gpu_ms=30 physical_calibration=none thread_present=false "
            "robot_execution=false\n"
        )
        report = summarize(samples_path, lips_path, stdout, material, 0.01)
        if (report["sample_count"] != STEP_COUNT or
                report["active_tetrahedra"] != EXPECTED_TETRAHEDRA or
                report["peak_gap_reduction_m"] >= CONTACT_SLOP_M):
            raise ValueError("fixture summary did not preserve the declared sample identity")
        return {"schema": "numi.science.synthetic-skin-traction-adapter-test.v1",
                "passed": True, "checks": ["fixed-horizon-csv", "lip-order", "metric-summary",
                                             "scene-fingerprint", "fidelity-boundary"]}


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        print(json.dumps(self_test(), sort_keys=True, separators=(",", ":")))
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("probe", type=Path)
    parser.add_argument("material", type=Path)
    parser.add_argument("metallib", type=Path)
    parser.add_argument("force_per_site_n", type=float, choices=(0.0, 0.01))
    args = parser.parse_args()
    run_directory = args.run_directory.resolve()
    trajectory = run_directory / "trajectory"
    native_directory = run_directory / "native"
    trajectory.mkdir(parents=True, exist_ok=False)
    native_directory.mkdir(parents=True, exist_ok=False)
    command = [str(args.probe), str(trajectory), str(args.material), str(args.metallib),
               "--force-per-site-n", f"{args.force_per_site_n:.3f}"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    (native_directory / "stdout.log").write_text(result.stdout)
    (native_directory / "stderr.log").write_text(result.stderr)
    if result.returncode != 0:
        print(json.dumps({"schema": "numi.science.synthetic-skin-traction-run.v1",
                          "valid": False, "native_exit_status": result.returncode},
                         sort_keys=True, separators=(",", ":")))
        return result.returncode or 1
    try:
        report = summarize(trajectory / "traction-samples.csv",
                           trajectory / "lip-geometry.csv", result.stdout,
                           args.material, args.force_per_site_n)
    except Exception as error:
        print(json.dumps({"schema": "numi.science.synthetic-skin-traction-run.v1",
                          "valid": False, "adapter_error": str(error)},
                         sort_keys=True, separators=(",", ":")))
        return 1
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
