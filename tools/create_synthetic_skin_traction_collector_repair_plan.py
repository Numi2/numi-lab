#!/usr/bin/env python3
"""Seal a corrected-collector plan using the retained control and one fresh load."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: create_synthetic_skin_traction_collector_repair_plan.py RUN_ROOT OLD_STUDY ORIGIN_ROOT")
    root, old_study, origin = (Path(value).resolve() for value in sys.argv[1:])
    metadata = root / "metadata"
    source, build = root / "source", root / "build"
    original_trial = old_study / "trials/full60-control"
    original_output = original_trial / "output"
    base_plan_path = origin / "metadata/plan.json"
    notebook = metadata / "science_notebook.py"
    builder = Path(__file__).resolve()
    collector = metadata / "run_synthetic_skin_traction_full_horizon_trial_v2.py"
    calibration_script = metadata / "calibrate_synthetic_skin_traction_collector_v2.py"
    model_path = metadata / "model.json"
    predictor = metadata / "predict_synthetic_skin_traction.py"
    plan_doc = metadata / "plan.md"
    calibration_path = metadata / "calibration.json"
    calibration_check = metadata / "collector-calibration/calibration-check.json"
    probe = build / "bin/metalrobo_synthetic_skin_wound_traction_probe"
    library = build / "lib/libmetalrobo.dylib"
    metallib = build / "matter/shaders/NumiMatter.metallib"
    material = source / "matter/materials/synthetic_skin_wound.nmatter"

    original_input_files = [
        old_study / "registration.json", old_study / "analysis.json",
        old_study / "stopped.json", old_study / "attempts/full60-control.json",
        original_trial / "started.json", original_trial / "process.json",
        original_output / "command.json", original_output / "stdout.json",
        original_output / "stderr.log", original_output / "native-stdout.log",
        original_output / "native-stderr.log", original_output / "native-exit-code.txt",
        original_output / "native/traction-samples.csv",
        original_output / "native/lip-geometry.csv",
    ]
    instrument_files = [
        probe, library, metallib, material,
        source / "apps/synthetic_skin_wound_traction_probe.mm",
        source / "matter/src/runtime.mm",
        source / "matter/src/metal/fgmres.metalinc",
        source / "matter/src/metal/mixed_fem.metalinc",
        origin / "metadata/source-binding.json", origin / "metadata/source.diff",
        origin / "metadata/preflight-confirmation.log", origin / "metadata/library-load.txt",
        origin / "metadata/traction_trial_adapter.py", origin / "metadata/plan.json",
        origin / "metadata/calibration.json",
        collector, calibration_script, calibration_check, *original_input_files,
    ]
    other_files = [notebook, predictor, model_path, builder, plan_doc]
    for path in instrument_files + other_files:
        if not path.is_file():
            raise SystemExit(f"missing repair-plan artifact: {path}")

    model = json.loads(model_path.read_text())
    plan = json.loads(base_plan_path.read_text())
    plan["model_file"] = str(model_path)
    plan["model"] = model
    plan["predictor"] = {
        "argv": [sys.executable, str(predictor), str(model_path)],
        "env": {}, "timeout_seconds": 30,
    }
    plan["repository"] = str(source)
    plan["instrument"] = {
        "description": (
            "Corrected collector validated against the retained full-horizon native control, then "
            "used for one fresh loaded trajectory. The control is transparently reparsed from its "
            "failed collector attempt; it is not rerun or counted as a replicate. Raw native samples, "
            "per-step residuals, determinant, mass, signed gap, active tetrahedra, binary, library, "
            "and original attempt lineage are all bound."
        ),
        "calibration": str(calibration_path),
        "artifacts": [str(path) for path in instrument_files],
    }
    plan["design"]["controls"] = (
        "The zero-force arm is a read-only reparse of the complete 60-step native control whose raw "
        "data and successful native exit were retained by the stopped first notebook; the previous "
        "collector rejected its summary prefix and that notebook remains inconclusive. The corrected "
        "collector is tested against those exact raw artifacts before this registration. Then run one "
        "fresh 10 mN-per-site loaded trajectory with the same source/build, solver policy, material, "
        "mesh, state, schedule, horizon, and unchanged native gates. This is one matched simulation "
        "comparison on one deterministic synthetic coupon, not two independent replicates."
    )
    plan["design"]["allocation"] = (
        "Retained zero-force native control reparsed first, then one fresh loaded treatment; no "
        "replay checkpoint or new control simulation."
    )
    plan["validity"] = [
        {"path": ["schema"], "equals": "numi.science.synthetic-skin-full-horizon-run.v2"},
        {"path": ["valid"], "equals": True},
        {"path": ["all_gates_passed"], "equals": True},
        {"path": ["native_exit_code"], "equals": 0},
        {"path": ["sample_count"], "equals": 60},
        {"path": ["execution_steps"], "equals": 60},
        {"path": ["protocol_steps"], "equals": 60},
        {"path": ["force_ramp"], "equals": "cosine60"},
        {"path": ["thread_present"], "equals": False},
        {"path": ["needle_present"], "equals": False},
        {"path": ["robot_execution"], "equals": False},
        {"path": ["physical_tissue_calibration"], "equals": "none"},
    ]
    plan["trials"] = [
        {
            "id": "full60-control-reparsed",
            "pair": "synthetic-coupon-full60-repaired-collector",
            "arm": "control",
            "unit": plan["model"]["training_units"][0],
            "argv": [
                sys.executable, str(collector), "{run}", "--reparse-traction-samples",
                str(original_output / "native/traction-samples.csv"),
                "--material", str(material),
            ],
            "env": {}, "timeout_seconds": 1200,
        },
        {
            "id": "full60-loaded-corrected-collector",
            "pair": "synthetic-coupon-full60-repaired-collector",
            "arm": "treatment",
            "unit": plan["model"]["training_units"][0],
            "argv": [
                sys.executable, str(collector), "{run}", "--execute-native",
                "--probe", str(probe), "--material", str(material),
                "--metallib", str(metallib), "--force-per-site-n", "0.010",
            ],
            "env": {}, "timeout_seconds": 1200,
        },
    ]

    base_artifacts = sorted({str(path) for path in instrument_files + other_files})
    plan["artifacts"] = sorted(set(base_artifacts + [str(calibration_path)]))
    hashes = {str(path): sha256(path) for path in instrument_files}
    evidence_paths = [
        origin / "metadata/preflight-confirmation.log",
        origin / "metadata/library-load.txt",
        calibration_check,
        original_trial / "process.json",
        old_study / "registration.json",
        old_study / "analysis.json",
        old_study / "stopped.json",
    ]
    calibration = {
        "schema": "numi.science.calibration.v1",
        "status": "passed",
        "scope": (
            "The CPU-only native preflight, candidate dylib resolution, and corrected collector "
            "were verified against one retained 60-step synthetic zero-force output; a deliberately "
            "corrupted sample above the unchanged 1e-4 volume residual gate was rejected. This is "
            "measurement-pipeline calibration only, not calibration of force, material, or human skin."
        ),
        "checks": [
            {"id": "compiled-mesh-bite-map-and-60-step-preflight", "passed": True},
            {"id": "candidate-dylib-resolution", "passed": True},
            {"id": "actual-retained-control-positive-and-corrupted-volume-negative", "passed": True},
        ],
        "bindings": hashes,
        "evidence": {str(path): sha256(path) for path in evidence_paths},
        "observed_units": [],
    }
    calibration_path.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
    plan["artifacts"] = sorted(set(plan["artifacts"] + [str(calibration_path)]))
    plan_path = metadata / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (metadata / "input-SHA256SUMS").write_text(
        "".join(f"{sha256(Path(path))}  {path}\n" for path in plan["artifacts"])
    )
    print(json.dumps({
        "schema": "numi.science.collector-repair-plan-receipt.v1",
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "calibration_sha256": sha256(calibration_path),
        "artifact_count": len(plan["artifacts"]),
        "trial_count": len(plan["trials"]),
        "fresh_native_trajectories": 1,
        "reused_native_control": True,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
