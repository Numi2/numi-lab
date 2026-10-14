#!/usr/bin/env python3
"""Freeze the v5 treatment study after build and CPU-only instrument checks."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("python", type=Path)
    parser.add_argument("--parent-study", type=Path, required=True)
    parser.add_argument("--model-file", default="SYNTHETIC_SKIN_TRACTION_MODEL_20261004_V5.json")
    args = parser.parse_args()
    root = args.run_root.resolve()
    python = args.python.resolve()
    source = root / "source"
    build = root / "build"
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)

    files = {
        "probe": build / "bin/metalrobo_synthetic_skin_wound_traction_probe",
        "library": build / "lib/libmetalrobo.dylib",
        "metallib": build / "matter/shaders/NumiMatter.metallib",
        "material": source / "matter/materials/synthetic_skin_wound.nmatter",
        "probe_source": source / "apps/synthetic_skin_wound_traction_probe.mm",
        "adapter": source / "tools/run_synthetic_skin_wound_traction_trial.py",
        "control_replay": source / "tools/replay_synthetic_skin_traction_control.py",
        "predictor": source / "tools/predict_synthetic_skin_traction.py",
        "plan_generator": source / "tools/create_synthetic_skin_traction_plan.py",
        "model": source / "numi" / args.model_file,
        "science_notebook": metadata / "science_notebook.py",
        "preflight": metadata / "preflight.log",
        "adapter_test": metadata / "adapter-self-test.json",
    }
    for name, path in files.items():
        if not path.is_file():
            raise SystemExit(f"missing frozen plan input {name}: {path}")

    preflight = files["preflight"].read_text()
    expected_preflight = (
        "synthetic_skin_wound_traction_preflight=ok",
        "source_tetrahedra=2304",
        "expected_tetrahedra=2304",
        "proximal_bite_nodes=",
        "distal_bite_nodes=",
        "force_ramp=cosine60",
        "newton_iterations=7",
        "gpu_dispatch=false",
    )
    if not all(token in preflight for token in expected_preflight):
        raise SystemExit("CPU-only preflight does not match the registered v5 mesh and ramp")
    adapter_test = json.loads(files["adapter_test"].read_text())
    if adapter_test.get("passed") is not True:
        raise SystemExit("adapter fixture calibration failed")

    sys.path.insert(0, str(source / "tools"))
    from run_synthetic_skin_wound_traction_trial import scene_fingerprint

    model_file = files["model"].resolve()
    model = json.loads(model_file.read_text())
    material_sha = sha256(files["material"])
    scene = scene_fingerprint(material_sha)
    unit = {"scene_fingerprint": scene, "deterministic": True}
    parent_study = args.parent_study.resolve()
    parent_control = parent_study / "trials/cosine24-mesh24x8x2-control"
    parent_artifacts = [
        parent_study / "registration.json",
        parent_study / "analysis.json",
        parent_study / "attempts/cosine24-mesh24x8x2-control.json",
        parent_control / "receipt.json",
        parent_control / "output/stdout.json",
        parent_control / "output/native/stdout.log",
        parent_control / "output/native/stderr.log",
        parent_control / "output/trajectory/traction-samples.csv",
        parent_control / "output/trajectory/lip-geometry.csv",
    ]
    if any(not path.is_file() for path in parent_artifacts):
        raise SystemExit("parent study is missing a required sealed control artifact")
    instrument_artifacts = [
        str(files[key].resolve())
        for key in ("adapter", "control_replay", "probe_source", "probe", "library", "metallib", "material")
    ] + [str(path) for path in parent_artifacts]

    receipt = {
        "schema": "numi.synthetic-skin-traction.build-receipt.v1",
        "source_revision": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "compiler": "AppleClang Release on Mac16,11 / Apple M4 Pro",
        "mesh_cells": [24, 8, 2],
        "steps": 60,
        "probe_source_sha256": sha256(files["probe_source"]),
        "probe_sha256": sha256(files["probe"]),
        "library_sha256": sha256(files["library"]),
        "metallib_sha256": sha256(files["metallib"]),
        "material_sha256": material_sha,
    }
    receipt_path = metadata / "build-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    calibration_path = metadata / "calibration.json"
    artifact_paths = [
        files[key].resolve()
        for key in ("adapter", "control_replay", "predictor", "plan_generator", "probe_source", "probe",
                    "library", "metallib", "material", "model", "preflight", "adapter_test")
    ] + [receipt_path.resolve(), calibration_path.resolve(), *parent_artifacts]
    calibration = {
        "schema": "numi.science.calibration.v1",
        "status": "passed",
        "scope": (
            "CPU-only compiled-mesh identity and bite-patch mapping, plus a fixture-tested "
            "CSV-to-JSON adapter for this exact native build. This does not calibrate force, "
            "tissue material, human skin, suture mechanics, or clinical behavior."
        ),
        "checks": [
            {"id": "compiled-authored-node-map", "passed": True},
            {"id": "three-dimensional-mesh-and-bite-patches", "passed": True},
            {"id": "fixed-horizon-csv-and-lip-order-adapter", "passed": True},
            {"id": "runtime-and-material-hash-binding", "passed": True},
        ],
        "bindings": {str(path): sha256(path) for path in map(Path, instrument_artifacts)},
        "evidence": {
            str(files["preflight"].resolve()): sha256(files["preflight"]),
            str(files["adapter_test"].resolve()): sha256(files["adapter_test"]),
            str(receipt_path): sha256(receipt_path),
        },
        "observed_units": [],
    }
    calibration_path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
    artifact_paths.append(calibration_path.resolve())

    plan = {
        "schema": "numi.science.plan.v2",
        "question": (
            "On one deterministic synthetic skin-wound coupon, does the fixed paired "
            "10 mN-per-bite traction lower the minimum mean top-lip gap by at least one "
            "contact slop compared with an otherwise identical zero-force run?"
        ),
        "hypothesis": (
            "The minimum mean top-lip gap in the loaded trajectory will be at least "
            "10 micrometres smaller than in the zero-force control."
        ),
        "purpose": "replication",
        "owner": "Numi Matter synthetic skin-wound traction probe",
        "repository": str(source),
        "backend": "Native Metal FEM on Apple M4 Pro; paired deterministic trajectories",
        "evidence_level": "simulation",
        "model": model,
        "model_file": str(model_file),
        "predictor": {
            "argv": [str(python), str(files["predictor"].resolve()), str(model_file)],
            "env": {},
            "timeout_seconds": 30,
        },
        "instrument": {
            "description": "Native wound-lip FEM trajectory with nonlinear failure diagnostics, full-duration cosine60 treatment ramp, integrity-checked reused deterministic control, fixed-horizon CSV reader, and CPU-only topology preflight",
            "calibration": str(calibration_path),
            "artifacts": instrument_artifacts,
        },
        "artifacts": [str(path) for path in artifact_paths],
        "design": {
            "intervention": "10 mN equal-and-opposite force per bite patch with a full-duration half-cosine ramp over all 60 treatment steps, reaching maximum force at the final sample; the zero-force condition is unchanged by ramp schedule",
            "controls": "Reuse the exact integrity-checked v4 zero-force baseline and its 60 raw samples; run only one new cosine60 loaded treatment. No independent control replicate is claimed. Material, specimen identity, mesh, solver iteration budget, timestep, gravity, and output horizon remain fixed; solver tolerances are unchanged.",
            "experimental_unit": "One authored deterministic synthetic coupon; time samples are repeated measures, not independent units",
            "allocation": "One matched deterministic pair using the prior valid control replay and one new native treatment trajectory",
            "unit_paths": {
                "scene_fingerprint": ["unit", "scene_fingerprint"],
                "deterministic": ["unit", "deterministic"],
            },
        },
        "observable": {
            "name": "minimum_mean_top_lip_gap",
            "unit": "m",
            "path": ["minimum_mean_gap_m"],
        },
        "prediction": {
            "estimand": "paired_difference_mean",
            "minimum": model["predicted_difference_m"]["minimum"],
            "maximum": model["predicted_difference_m"]["maximum"],
        },
        "validity": [
            {"path": ["schema"], "equals": "numi.science.synthetic-skin-traction-run.v1"},
            {"path": ["valid"], "equals": True},
            {"path": ["simulation_only"], "equals": True},
            {"path": ["sample_count"], "equals": 60},
            {"path": ["thread_present"], "equals": False},
            {"path": ["needle_present"], "equals": False},
            {"path": ["robot_execution"], "equals": False},
            {"path": ["physical_tissue_calibration"], "equals": "none"},
        ],
        "paired_equal": [
            ["unit", "scene_fingerprint"],
            ["unit", "deterministic"],
            ["settings", "mesh_cells"],
            ["settings", "step_count"],
            ["settings", "timestep_s"],
            ["settings", "newton_iterations"],
            ["settings", "contact_slop_m"],
            ["settings", "initial_wound_gap_m"],
            ["settings", "gravity_m_s2"],
            ["settings", "material_sha256"],
            ["simulation_only"],
            ["physical_tissue_calibration"],
        ],
        "trials": [],
        "limitations": (
            "One deterministic homogeneous synthetic coupon and one paired condition; no measured skin fit, "
            "puncture, tear, needle, thread, robot, healing, biological replicate, statistical inference, "
            "or clinical qualification. Mesh coarsening is a resolution tradeoff, not a faster or more "
            "accurate solver claim."
        ),
    }
    plan["trials"].append({
        "id": "cosine60-mesh24x8x2-control-replay",
        "pair": "coupon-v3",
        "arm": "control",
        "unit": unit,
        "argv": [str(python), str(files["control_replay"].resolve()), "{run}", str(parent_study / "analysis.json")],
        "env": {},
        "timeout_seconds": 30,
    })
    plan["trials"].append({
        "id": "cosine60-mesh24x8x2-loaded",
        "pair": "coupon-v3",
        "arm": "treatment",
        "unit": unit,
        "argv": [
            str(python), str(files["adapter"].resolve()), "{run}",
            str(files["probe"].resolve()), str(files["material"].resolve()),
            str(files["metallib"].resolve()), "0.010",
        ],
        "env": {},
        "timeout_seconds": 300,
    })
    plan_path = metadata / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "schema": "numi.science.synthetic-skin-traction-plan-receipt.v1",
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "calibration": str(calibration_path),
        "scene_fingerprint": scene,
        "tetrahedra": 2304,
        "steps": 60,
        "gpu_dispatch_during_preregistration": False,
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
