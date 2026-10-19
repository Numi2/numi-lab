from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time

PLAN_DIR = Path(__file__).resolve().parent
PLAN_PATH = PLAN_DIR / "run-plan.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(
        description="Two-step 0.25 ms joint clean-control plus late-reject/same-context retry on pinned rate252 source; no visual verification."
    )
    parser.add_argument("--output", type=Path, help="fresh output root; defaults to the pinned plan")
    parser.add_argument("--validate-only", action="store_true", help="verify all frozen pins without creating output or launching native work")
    args = parser.parse_args()
    plan = json.loads(PLAN_PATH.read_text())
    if sha256(Path(__file__)) != plan["runner_sha256"]:
        raise RuntimeError("runner differs from frozen plan")

    checks = (
        ("base_run_plan_path", "base_run_plan_sha256"),
        ("build_plan_path", "build_plan_sha256"),
        ("source_invocation_path", "source_invocation_sha256"),
        ("binary_path", "binary_sha256"),
        ("runtime_library_path", "runtime_library_sha256"),
        ("runtime_source_path", "runtime_source_sha256"),
        ("respiration_source_path", "respiration_source_sha256"),
        ("source_manifest_path", "source_manifest_sha256"),
    )
    pinned = {}
    for path_key, hash_key in checks:
        path = Path(plan[path_key])
        if sha256(path) != plan[hash_key]:
            raise RuntimeError(f"frozen input changed: {path}")
        pinned[path_key] = path

    for asset_path, expected in plan["extra_pinned_assets"].items():
        if sha256(asset_path) != expected:
            raise RuntimeError(f"extra source asset changed: {asset_path}")
    base_plan = json.loads(pinned["base_run_plan_path"].read_text())
    if base_plan.get("passive_fem_rate_cap") != plan["rate_cap_exponent"]:
        raise RuntimeError("pinned 252 base plan does not declare the requested passive FEM rate cap")
    if base_plan.get("binary_sha256") != plan["binary_sha256"]:
        raise RuntimeError("pinned 252 base plan binary differs from candidate binary")
    if base_plan.get("source_invocation_sha256") != plan["source_invocation_sha256"]:
        raise RuntimeError("pinned 252 base plan invocation differs from candidate invocation")
    invocation = json.loads(pinned["source_invocation_path"].read_text())
    for path, expected in invocation["asset_sha256"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"base native asset changed: {path}")
    skin_args = [i for i, v in enumerate(invocation["argv"]) if v == "--skin-payload"]
    if len(skin_args) != 1 or skin_args[0] + 1 >= len(invocation["argv"]):
        raise RuntimeError("joint scene must retain exactly one skin payload")

    wrapper = Path(plan["wrapper_path"])
    python = Path(plan["python_path"])
    if not wrapper.is_file() or not python.is_file():
        raise RuntimeError("pinned exclusive native runner is unavailable")
    for material_path, expected in zip(plan["skin_material_paths"], plan["skin_material_sha256"]):
        if sha256(material_path) != expected:
            raise RuntimeError(f"pinned skin/adipose material changed: {material_path}")
    if sha256(plan["prostate_reference_material_path"]) != plan["prostate_reference_material_sha256"]:
        raise RuntimeError("pinned prostate proxy material changed")
    if sha256(plan["prostate_registered_payload_path"]) != plan["prostate_registered_payload_sha256"]:
        raise RuntimeError("pinned NHANATOMY source payload changed")
    if args.validate_only:
        print(json.dumps({
            "validated": True,
            "binary_sha256": plan["binary_sha256"],
            "runtime_library_sha256": plan["runtime_library_sha256"],
            "source_invocation_sha256": plan["source_invocation_sha256"],
            "gpu_launched": False
        }, indent=2))
        return 0
    binary = pinned["binary_path"]
    library = pinned["runtime_library_path"]
    runtime_source = pinned["runtime_source_path"]
    respiration_source = pinned["respiration_source_path"]
    output_root = args.output or Path(plan["output_root"])
    if output_root.exists():
        raise RuntimeError(f"refusing to overwrite run output {output_root}")
    output_root.mkdir(parents=True)
    alias = output_root / "lib"
    alias.mkdir()
    (alias / "libmetalrobo.dylib").symlink_to(library.resolve())

    base_argv = list(invocation["argv"])
    base_argv[0] = str(binary)
    for flag, value in (
        ("--muscle-step-count", str(plan["step_count"])),
        ("--muscle-step-seconds", f'{plan["step_seconds"]:.8f}'),
    ):
        if flag not in base_argv:
            raise RuntimeError(f"invocation omits {flag}")
        base_argv[base_argv.index(flag) + 1] = value

    base_env = dict(invocation["environment"])
    base_env.update({
        "DYLD_LIBRARY_PATH": str(alias),
        "DYLD_PRINT_LIBRARIES": "1",
        "NUMI_HUMAN_BLOOD_MPM_ENABLED": "0",
            "NUMI_HUMAN_RESTING_FGMRES_BUDGET": "16",
            "NUMI_HUMAN_PASSIVE_FEM_MAX_RATE_EXPONENT": "0",
            "NUMI_MATTER_GPU_TIMING": "0",
        "NUMI_MATTER_SOURCE": str(runtime_source),
        "NUMI_MATTER_TEST_SOURCE": str(runtime_source),
        "NUMI_MATTER_SOURCE_SHA256": plan["runtime_source_sha256"],
        "NUMI_MATTER_TEST_SOURCE_SHA256": plan["runtime_source_sha256"],
    })

    attempts = []
    for case in plan["cases"]:
        case_dir = output_root / case["name"]
        case_dir.mkdir()
        argv = list(base_argv)
        argv[4] = str(case_dir / "output")
        env = dict(base_env)
        env.update({
            "NUMI_HUMAN_RESTING_AORTIC_WALL": "1",
            "NUMI_HUMAN_RESTING_SKIN_FAT_FEM": "1",
            "NUMI_HUMAN_RESTING_PROSTATE_FEM": "1",
            "NUMI_HUMAN_BLOOD_MPM_ENABLED": "0",
            "NUMI_HUMAN_RESTING_FGMRES_BUDGET": "16",
            "NUMI_HUMAN_PASSIVE_FEM_MAX_RATE_EXPONENT": "0",
            "NUMI_MATTER_GPU_TIMING": "0",
            "NUMI_HUMAN_SHARED_TISSUE_ROLLBACK_TEST": (
                "1" if case["rollback_retry"] else "0"
            ),
            "NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT": str(
                case_dir / "common-field-failure.json"
            ),
        })
        assets = {
            str(Path(p).resolve()): sha256(p)
            for p in argv[1:] if Path(p).is_file()
        }
        for key, value in env.items():
            if key.endswith(("_SOURCE", "_MATERIAL")) and Path(value).is_file():
                assets[str(Path(value).resolve())] = sha256(value)
        assets[str(runtime_source.resolve())] = sha256(runtime_source)
        assets[str(respiration_source.resolve())] = sha256(respiration_source)
        launcher = [str(python), str(wrapper), "--", *argv]
        record = {
            "schema": "numi.continuum_joint_shared_tissue_rollback_retry_attempt_rate252.v1",
            "case": case,
            "argv": argv,
            "launcher_argv": launcher,
            "environment": env,
            "asset_sha256": assets,
            "base_run_plan_sha256": sha256(pinned["base_run_plan_path"]),
            "build_plan_sha256": sha256(pinned["build_plan_path"]),
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
        receipt = case_dir / "invocation.json"
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        process_env = os.environ.copy()
        process_env.update(env)
        started = time.monotonic()
        with (case_dir / "native.log").open("w") as log:
            proc = subprocess.run(
                launcher, env=process_env, stdout=log, stderr=subprocess.STDOUT
            )
        record.update(
            process_exit=proc.returncode,
            wall_seconds=time.monotonic() - started,
            runtime_library_sha256_after=sha256(library),
        )
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        attempts.append({
            "name": case["name"],
            "process_exit": proc.returncode,
            "wall_seconds": record["wall_seconds"],
            "receipt": str(receipt),
        })
        if proc.returncode:
            break

    rollback_checks = {}
    trace_comparison = {"byte_identical": False, "not_run_or_failed": True}
    if len(attempts) == len(plan["cases"]) and all(x["process_exit"] == 0 for x in attempts):
        clean = output_root / plan["cases"][0]["name"]
        retry = output_root / plan["cases"][1]["name"]
        retry_log = (retry / "native.log").read_text(errors="replace")
        rollback_checks = {
            marker: marker in retry_log for marker in plan["rollback_log_markers"]
        }
        if not all(rollback_checks.values()):
            raise RuntimeError("retry log lacks one or more exact accepted-state checks")
        clean_trace = clean / "output/resting-coupled.csv"
        retry_trace = retry / "output/resting-coupled.csv"
        if not clean_trace.is_file() or not retry_trace.is_file():
            raise RuntimeError("both cases must emit resting-coupled.csv")
        clean_bytes = clean_trace.read_bytes()
        retry_bytes = retry_trace.read_bytes()
        trace_comparison = {
            "path": "output/resting-coupled.csv",
            "clean_sha256": hashlib.sha256(clean_bytes).hexdigest(),
            "retry_sha256": hashlib.sha256(retry_bytes).hexdigest(),
            "byte_identical": clean_bytes == retry_bytes,
        }

    summary = {
        "schema": "numi.continuum_joint_shared_tissue_rollback_retry_summary_rate252.v1",
        "plan_sha256": sha256(PLAN_PATH),
        "runner_sha256": sha256(Path(__file__)),
        "binary_sha256": sha256(binary),
        "runtime_library_sha256": sha256(library),
        "attempts": attempts,
        "rollback_exact_state_checks": rollback_checks,
        "clean_vs_retry_trace": trace_comparison,
        "scope": (
            "two 0.25 ms steps on rate252 binary, FEM rate exponent 0, FGMRES budget 16; joint skin/fat, aortic wall, and prostate opt-ins; "
            "late reject at step 1 plus same-context retry; MPM off; no visual verification"
        ),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return int(
        len(attempts) != len(plan["cases"])
        or any(x["process_exit"] != 0 for x in attempts)
        or not all(rollback_checks.values())
        or not trace_comparison.get("byte_identical", False)
    )


if __name__ == "__main__":
    raise SystemExit(main())

