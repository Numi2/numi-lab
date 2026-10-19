from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time

ROOT = Path("/Users/n/numi-human-separation-20261018/C/continuum-integrated-234")
BUILD = ROOT / "passive-fem-rate-252" / "build"
PLAN_PATH = BUILD / "run-plan.json"
INVOCATION_PATH = Path("/Users/n/numi-human-separation-20261018/C/shared-fem-scene-219/native-fem-cv-009/invocation.json")
WRAPPER = Path("/Users/n/numi-human-final-integration-publish-1397/tools/run_resting_job_exclusive.py")
PYTHON = Path("/Users/n/numi-human-prep-venv-20261005/bin/python3")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def remove_optional_skin_payload(argv):
    result = list(argv)
    positions = [i for i, value in enumerate(result) if value == "--skin-payload"]
    if len(positions) != 1 or positions[0] + 1 >= len(result):
        raise RuntimeError("base invocation must contain exactly one --skin-payload path")
    return result


def main():
    parser = argparse.ArgumentParser(description="Five one-step, 0.25 ms registered wall/skin/prostate FEM owner cases; no visual verification.")
    parser.add_argument("--output", type=Path, help="fresh output root; defaults to the pinned run plan")
    parser.add_argument("--scenario", action="append", required=True, help="select explicit named case(s); required to avoid accidental baseline reruns")
    args = parser.parse_args()
    plan = json.loads(PLAN_PATH.read_text())
    if plan.get("run_script_sha256") and sha256(Path(__file__)) != plan["run_script_sha256"]:
        raise RuntimeError("runner does not match the pinned run-plan script hash")
    respiration_source = Path(plan["respiration_source_path"])
    if sha256(respiration_source) != plan["respiration_source_sha256"]:
        raise RuntimeError("pinned sustained respiration source changed after run-plan freeze")
    invocation = json.loads(INVOCATION_PATH.read_text())
    binary = Path(plan["binary_path"])
    library = Path(plan["runtime_library_path"])
    runtime_source = Path(plan["runtime_source_path"])
    output_root = args.output or Path(plan["output_root"])
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite run output {output_root}")
    if sha256(binary) != plan["binary_sha256"]:
        raise RuntimeError("combined binary does not match the pinned run plan")
    if sha256(library) != plan["runtime_library_sha256"]:
        raise RuntimeError("233 library does not match the pinned run plan")
    if sha256(runtime_source) != plan["runtime_source_sha256"]:
        raise RuntimeError("233 MatterSource does not match the pinned run plan")
    if sha256(INVOCATION_PATH) != plan["source_invocation_sha256"]:
        raise RuntimeError("base native invocation changed after run-plan freeze")
    for path, expected in invocation["asset_sha256"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"base input changed after source invocation freeze: {path}")
    skin_material, adipose_material = [Path(p) for p in plan["tissue_material_paths"]]
    if sha256(skin_material) != plan["tissue_material_sha256"][0] or sha256(adipose_material) != plan["tissue_material_sha256"][1]:
        raise RuntimeError("frozen tissue material asset changed after run-plan freeze")
    prostate_material = Path(plan["prostate_reference_material_path"])
    if sha256(prostate_material) != plan["prostate_reference_material_sha256"]:
        raise RuntimeError("pinned prostate reference material changed after run-plan freeze")
    if not WRAPPER.is_file() or not PYTHON.is_file():
        raise RuntimeError("the pinned exclusive native runner is unavailable")

    output_root.mkdir(parents=True)
    alias = output_root / "lib"
    alias.mkdir()
    (alias / "libmetalrobo.dylib").symlink_to(library.resolve())
    base_argv = remove_optional_skin_payload(invocation["argv"])
    base_argv[0] = str(binary)
    base_argv[base_argv.index("--muscle-step-count") + 1] = str(plan["step_count"])
    base_argv[base_argv.index("--muscle-step-seconds") + 1] = f"{plan['step_seconds']:.8f}"
    base_env = dict(invocation["environment"])
    # Wall/skin/prostate validation keeps the qualified MPM handoff explicitly off.
    base_env["NUMI_HUMAN_BLOOD_MPM_ENABLED"] = "0"
    base_env["DYLD_LIBRARY_PATH"] = str(alias)
    base_env["DYLD_PRINT_LIBRARIES"] = "1"
    for key, path_key, sha_key in (
        ("NUMI_MATTER_SOURCE", "runtime_source_path", "runtime_source_sha256"),
        ("NUMI_MATTER_TEST_SOURCE", "runtime_source_path", "runtime_source_sha256"),
    ):
        base_env[key] = plan[path_key]
    for key in ("NUMI_MATTER_SOURCE_SHA256", "NUMI_MATTER_TEST_SOURCE_SHA256"):
        base_env[key] = plan["runtime_source_sha256"]

    scenarios = plan["scenarios"]
    if args.scenario != ["joint"]:
        raise RuntimeError("budget-16 comparison is pinned to exactly the joint scenario")
    if args.scenario:
        known = {scenario["name"] for scenario in scenarios}
        requested = set(args.scenario)
        unknown = requested - known
        if unknown:
            raise RuntimeError(f"unknown scenario name(s): {', '.join(sorted(unknown))}")
        scenarios = [scenario for scenario in scenarios if scenario["name"] in requested]
    if not scenarios:
        raise RuntimeError("run plan selected no scenarios")
    records = []
    for scenario in scenarios:
        case = output_root / scenario["name"]
        case.mkdir()
        argv = list(base_argv)
        argv[4] = str(case / "output")
        argv[argv.index("--muscle-step-count") + 1] = str(plan["step_count"])
        env = dict(base_env)
        env.pop("NUMI_MATTER_GPU_TIMING_ROOT", None)
        env.pop("NUMI_MATTER_GPU_TIMING_STAGE", None)
        env.pop("NUMI_MATTER_SOLVER_ITERATION_DIAGNOSTICS", None)
        env["NUMI_MATTER_GPU_TIMING"] = "0"
        env["NUMI_HUMAN_RESTING_FGMRES_BUDGET"] = "16"
        env["NUMI_HUMAN_PASSIVE_FEM_MAX_RATE_EXPONENT"] = "0"
        env["NUMI_HUMAN_RESTING_AORTIC_WALL"] = "1" if scenario["aortic_wall"] else "0"
        env["NUMI_HUMAN_RESTING_SKIN_FAT_FEM"] = "1" if scenario["skin_fat_fem"] else "0"
        env["NUMI_HUMAN_RESTING_PROSTATE_FEM"] = "1" if scenario["prostate_fem"] else "0"
        env["NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT"] = str(case / "common-field-failure.json")
        assets = {str(Path(p).resolve()): sha256(p) for p in argv[1:] if Path(p).is_file()}
        for key, value in env.items():
            if key.endswith(("_SOURCE", "_MATERIAL")) and Path(value).is_file():
                assets[str(Path(value).resolve())] = sha256(value)
        assets[str(skin_material.resolve())] = sha256(skin_material)
        assets[str(adipose_material.resolve())] = sha256(adipose_material)
        assets[str(prostate_material.resolve())] = sha256(prostate_material)
        assets[str(runtime_source.resolve())] = sha256(runtime_source)
        assets[str(respiration_source.resolve())] = sha256(respiration_source)
        launcher = [str(PYTHON), str(WRAPPER), "--", *argv]
        record = {
            "schema": "numi.continuum_integrated_vessel_skin_prostate_native_attempt.v1",
            "scenario": scenario,
            "argv": argv,
            "launcher_argv": launcher,
            "environment": env,
            "asset_sha256": assets,
            "binary_sha256": sha256(binary),
            "runtime_library_path": str(library),
            "runtime_library_sha256": sha256(library),
            "runtime_source_path": str(runtime_source),
            "runtime_source_sha256": sha256(runtime_source),
            "requested_step_count": plan["step_count"],
            "requested_step_seconds": plan["step_seconds"],
            "requested_physical_horizon_s": plan["physical_horizon_seconds"],
            "visual_verification": False,
        }
        receipt = case / "invocation.json"
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        process_env = os.environ.copy()
        process_env.update(env)
        started = time.monotonic()
        with (case / "native.log").open("w") as log:
            try:
                proc = subprocess.run(launcher, env=process_env, stdout=log,
                                      stderr=subprocess.STDOUT, timeout=240)
            except subprocess.TimeoutExpired:
                log.write("\nsolver_profile_timeout_seconds=240\n")
                proc = subprocess.CompletedProcess(launcher, 124)
        record.update(process_exit=proc.returncode, wall_seconds=time.monotonic() - started,
                      runtime_library_sha256_after=sha256(library))
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        records.append({"scenario": scenario["name"], "process_exit": proc.returncode,
                        "seconds": record["wall_seconds"], "receipt": str(receipt)})
        if proc.returncode:
            break
    summary = {"schema": "numi.continuum_integrated_vessel_skin_prostate_run_summary.v1",
               "plan_sha256": sha256(PLAN_PATH), "binary_sha256": sha256(binary),
               "runtime_library_sha256": sha256(library), "attempts": records,
               "visual_verification": False}
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if len(records) == len(scenarios) and all(x["process_exit"] == 0 for x in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
