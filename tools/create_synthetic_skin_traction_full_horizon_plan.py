#!/usr/bin/env python3
"""Calibrate the measurement adapter and seal a v2 full-horizon study plan."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


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
    canonical = json.dumps(scene, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: create_synthetic_skin_traction_full_horizon_plan.py RUN_ROOT")
    root = Path(sys.argv[1]).resolve()
    source, build, metadata = root / "source", root / "build", root / "metadata"
    probe = build / "bin/metalrobo_synthetic_skin_wound_traction_probe"
    library = build / "lib/libmetalrobo.dylib"
    metallib = build / "matter/shaders/NumiMatter.metallib"
    material = source / "matter/materials/synthetic_skin_wound.nmatter"
    probe_source = source / "apps/synthetic_skin_wound_traction_probe.mm"
    runtime_source = source / "matter/src/runtime.mm"
    fgmres_source = source / "matter/src/metal/fgmres.metalinc"
    mixed_fem_source = source / "matter/src/metal/mixed_fem.metalinc"
    adapter = metadata / "traction_trial_adapter.py"
    plan_builder = metadata / "create_full_horizon_plan.py"
    plan_doc = metadata / "plan.md"
    notebook = metadata / "science_notebook.py"
    predictor = metadata / "predict_synthetic_skin_traction.py"
    model_path = metadata / "model.json"
    preflight = metadata / "preflight.log"
    library_load = metadata / "library-load.txt"
    self_test_path = metadata / "adapter-self-test.json"
    source_binding = metadata / "source-binding.json"
    source_diff = metadata / "source.diff"
    calibration_path = metadata / "calibration.json"

    files = [probe, library, metallib, material, probe_source, runtime_source,
             fgmres_source, mixed_fem_source, adapter, plan_builder, plan_doc, notebook,
             predictor, model_path, preflight, library_load]
    for path in files:
        if not path.is_file():
            raise SystemExit(f"missing frozen study input: {path}")
    preflight_text = preflight.read_text()
    expected_preflight = (
        "synthetic_skin_wound_traction_preflight=ok", "source_tetrahedra=2304",
        "expected_tetrahedra=2304", "top_lip_pairs=12", "force_ramp=cosine60",
        "newton_iterations=7", "execution_steps=60", "gpu_dispatch=false",
    )
    if not all(item in preflight_text for item in expected_preflight):
        raise SystemExit("CPU-only preflight does not bind the declared 60-step coupon")
    if str(library.resolve()) not in library_load.read_text():
        raise SystemExit("loaded-library evidence does not resolve to this candidate build")

    source_revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    source_status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=normal"], text=True
    )
    diff_bytes = subprocess.check_output(["git", "-C", str(source), "diff", "HEAD", "--binary"])
    source_diff.write_bytes(diff_bytes)
    untracked = subprocess.check_output(
        ["git", "-C", str(source), "ls-files", "--others", "--exclude-standard"], text=True
    ).splitlines()
    untracked_hashes = {
        relative: sha256(source / relative) for relative in untracked
        if (source / relative).is_file()
    }
    source_binding.write_text(json.dumps({
        "schema": "numi.synthetic-skin-source-binding.v1",
        "source_root": str(source.resolve()),
        "base_revision": source_revision,
        "working_tree_status": source_status,
        "tracked_diff_sha256": hashlib.sha256(diff_bytes).hexdigest(),
        "untracked_source_sha256": untracked_hashes,
        "probe_sha256": sha256(probe_source),
        "runtime_sha256": sha256(runtime_source),
        "fgmres_sha256": sha256(fgmres_source),
        "mixed_fem_sha256": sha256(mixed_fem_source),
        "binary_sha256": sha256(probe),
        "library_sha256": sha256(library),
        "metallib_sha256": sha256(metallib),
        "material_sha256": sha256(material),
    }, indent=2, sort_keys=True) + "\n")

    adapter_check = subprocess.run(
        [sys.executable, str(adapter), "--self-test"],
        check=True, capture_output=True, text=True,
    )
    adapter_record = json.loads(adapter_check.stdout)
    if adapter_record.get("passed") is not True or not adapter_record.get("synthetic_fixture_only"):
        raise SystemExit("measurement adapter calibration self-test failed")
    self_test_path.write_text(json.dumps(adapter_record, indent=2, sort_keys=True) + "\n")

    model = json.loads(model_path.read_text())
    unit = {"scene_fingerprint": scene_fingerprint(sha256(material)), "deterministic": True}
    if unit not in model.get("training_units", []):
        raise SystemExit("model training history does not identify this deterministic coupon")

    instrument_files = [adapter, probe_source, runtime_source, fgmres_source,
                        mixed_fem_source, probe, library, metallib, material,
                        source_binding, source_diff, preflight, library_load,
                        self_test_path, plan_doc]
    plan_files = [*instrument_files, notebook, predictor, model_path, plan_builder]
    calibration = {
        "schema": "numi.science.calibration.v1",
        "status": "passed",
        "scope": (
            "CPU-only compiled-mesh identity and bite-map preflight, loaded dylib-path check, "
            "and fixture-tested 60-sample schedule/metric adapter. This does not calibrate "
            "force, material, human skin, suture mechanics, or clinical behavior."
        ),
        "checks": [
            {"id": "compiled-mesh-bite-map-and-60-step-preflight", "passed": True},
            {"id": "candidate-dylib-resolution", "passed": True},
            {"id": "adapter-positive-and-negative-fixture", "passed": True},
        ],
        "bindings": {str(path.resolve()): sha256(path) for path in instrument_files},
        "evidence": {
            str(preflight.resolve()): sha256(preflight),
            str(library_load.resolve()): sha256(library_load),
            str(self_test_path.resolve()): sha256(self_test_path),
        },
        "observed_units": [],
    }
    calibration_path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
    plan_files.append(calibration_path)
    artifacts = sorted({str(path.resolve()) for path in plan_files})
    material_hash = sha256(material)
    shared_settings = [
        ["unit", "scene_fingerprint"], ["unit", "deterministic"],
        ["settings", "mesh_cells"], ["settings", "step_count"],
        ["settings", "timestep_s"], ["settings", "newton_iterations"],
        ["settings", "fgmres_max_iterations"], ["settings", "contact_slop_m"],
        ["settings", "initial_wound_gap_m"], ["settings", "gravity_m_s2"],
        ["settings", "material_sha256"], ["settings", "solver_policy"],
        ["settings", "force_ramp"], ["simulation_only"],
        ["physical_tissue_calibration"],
    ]
    plan = {
        "schema": "numi.science.plan.v2",
        "purpose": "replication",
        "question": (
            "Under the same trace-guided solver candidate, does a complete 60-step paired-bite "
            "traction trajectory preserve every unchanged native gate and reduce the minimum "
            "mean wound-lip gap by at least 10 micrometres relative to a matched zero-force trajectory?"
        ),
        "hypothesis": (
            "The candidate completes all 60 steps in both arms, passes every unchanged native gate, "
            "and the loaded minimum mean top-lip gap is 10 micrometres to 600 micrometres below control."
        ),
        "owner": "Numi Matter synthetic skin-wound traction probe",
        "repository": str(source.resolve()),
        "backend": "Native Apple Metal FEM on Mac16,11 / Apple M4 Pro",
        "evidence_level": "simulation",
        "model": model,
        "model_file": str(model_path.resolve()),
        "predictor": {
            "argv": [sys.executable, str(predictor.resolve()), str(model_path.resolve())],
            "env": {}, "timeout_seconds": 30,
        },
        "instrument": {
            "description": (
                "Source-bound native 60-step cosine60 traction/control trajectories with full raw "
                "samples, stepwise solver and volume residuals, determinant, mass, signed lip gap, "
                "active tetrahedra, and loaded native library identity."
            ),
            "calibration": str(calibration_path.resolve()),
            "artifacts": [str(path.resolve()) for path in instrument_files],
        },
        "artifacts": artifacts,
        "design": {
            "intervention": (
                "10 mN equal-and-opposite traction per bite site on a half-cosine ramp across all "
                "60 samples; the matched control uses zero force under the same binary, solver policy, "
                "mesh, material, initial geometry and horizon."
            ),
            "controls": (
                "A new zero-force trajectory precedes one loaded trajectory, both using the exact same "
                "trace-guided candidate source/build and unchanged native gates. The previous seven-step "
                "prefix is not reused as a checkpoint or a replicate."
            ),
            "experimental_unit": "One deterministic authored synthetic coupon; the 60 time samples are repeated measures, not independent units",
            "allocation": "One matched deterministic pair, zero-force control first and loaded treatment second",
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
            {"path": ["schema"], "equals": "numi.science.synthetic-skin-full-horizon-run.v1"},
            {"path": ["valid"], "equals": True},
            {"path": ["all_gates_passed"], "equals": True},
            {"path": ["native_exit_code"], "equals": 0},
            {"path": ["sample_count"], "equals": 60},
            {"path": ["force_ramp"], "equals": "cosine60"},
            {"path": ["thread_present"], "equals": False},
            {"path": ["needle_present"], "equals": False},
            {"path": ["robot_execution"], "equals": False},
            {"path": ["physical_tissue_calibration"], "equals": "none"},
        ],
        "paired_equal": shared_settings,
        "limitations": (
            "One deterministic uncalibrated homogeneous synthetic coupon and one paired condition. "
            "No measured skin fit, puncture, tear, needle, thread, robot, healing, biological replicate, "
            "population inference, clinical qualification, or 100x whole-task performance claim. "
            "A miss of the secondary 5e-5 residual target is reported without relaxing the native 1e-4 gate."
        ),
        "trials": [],
    }
    for identity, arm, force in (("full60-control", "control", 0.0), ("full60-loaded", "treatment", 0.010)):
        plan["trials"].append({
            "id": identity,
            "pair": "synthetic-coupon-full60",
            "arm": arm,
            "unit": unit,
            "argv": [
                sys.executable, str(adapter.resolve()), "{run}", str(probe.resolve()),
                str(material.resolve()), str(metallib.resolve()), f"{force:.3f}",
            ],
            "env": {},
            "timeout_seconds": 1200,
        })
    plan_path = metadata / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n")
    manifest_path = metadata / "input-SHA256SUMS"
    manifest_path.write_text("".join(f"{sha256(Path(path))}  {path}\n" for path in artifacts))
    print(json.dumps({
        "schema": "numi.science.full-horizon-plan-receipt.v1",
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "calibration": str(calibration_path),
        "calibration_sha256": sha256(calibration_path),
        "scene_fingerprint": unit["scene_fingerprint"],
        "artifact_count": len(artifacts),
        "trial_count": len(plan["trials"]),
        "gpu_dispatch_during_registration": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
