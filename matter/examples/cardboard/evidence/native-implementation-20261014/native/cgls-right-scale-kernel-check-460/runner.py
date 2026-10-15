#!/Users/home/.pyenv/versions/3.10.12/bin/python3
"""One-worker runner for the source-bound CGLS scale kernel checks.

This script creates its output case only when run. It never overwrites an
existing case, and runs the identity and right-scale checkers sequentially.
Use --plan for read-only hash and command validation; it does not create the
case directory or execute Metal code.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/Users/home/cardboard-implementation-20261014")
PACKAGE = BASE / "analysis/native-cgls-right-scale-checker-149-001"
OVERLAY = BASE / "analysis/native-cgls-block-right-preconditioning-overlay-149-002"
CASE = BASE / "native/cgls-right-scale-kernel-check-460"
RUNNER = BASE / "native/run-cgls-right-scale-kernel-check-460.py"
COMMAND_FILE = BASE / "native/cgls-right-scale-kernel-check-460-command.json"
TIMEOUT_SECONDS_EACH = 180
EXPECTED_PACKAGE_MANIFEST_SHA256 = (
    "3619c4e2b5b5208548f36562ffcb1373cdbca4e41971171294b6e1345d67fe0e"
)
EXPECTED_OVERLAY_MANIFEST_SHA256 = (
    "e158080398d47e1974e34ffcd71af5cd7e563d0fd08c4eb4e53482e7f9a94906"
)
EXPECTED_OVERLAY_PATCH_SHA256 = (
    "e23ed54a067f2f2ad822d84172d21df225aa112e28307f0cbe6299a175da8a86"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bound_inputs() -> list[Path]:
    relpaths = [
        "manifest.json",
        "README.md",
        "build-only-receipt.json",
        "source/native-cgls-overlay-149-002.json",
        "source/matter/include/numi/matter/shared.h",
        "source/matter/src/runtime.mm",
        "source/matter/src/metal/fgmres.metalinc",
        "source/matter/src/metal/base_cgls.metalinc",
        "source/matter/src/metal/base_cgls_runtime.metalinc",
        "native/base_cgls_identity_check.mm",
        "native/right_scale_check.mm",
        "native/base_cgls_test_actions.metal",
        "native/compile_only.sh",
        "native/build_only.sh",
        "native/build/BaseCGLSRightScaleCheck.metallib",
        "native/build/base_cgls_identity_check",
        "native/build/base_cgls_right_scale_check",
    ]
    paths = [PACKAGE / rel for rel in relpaths]
    paths += [OVERLAY / "source-overlay.json", OVERLAY / "source-overlay.patch"]
    paths += [RUNNER, COMMAND_FILE]
    return paths


def expected_files_match(paths: list[Path]) -> list[str]:
    problems: list[str] = []
    package_manifest = PACKAGE / "manifest.json"
    overlay_manifest = OVERLAY / "source-overlay.json"
    overlay_patch = OVERLAY / "source-overlay.patch"
    if not package_manifest.is_file():
        problems.append("checker package manifest is missing")
    elif sha256(package_manifest) != EXPECTED_PACKAGE_MANIFEST_SHA256:
        problems.append("checker package manifest hash changed")
    if not overlay_manifest.is_file():
        problems.append("149-002 overlay manifest is missing")
    elif sha256(overlay_manifest) != EXPECTED_OVERLAY_MANIFEST_SHA256:
        problems.append("149-002 overlay manifest hash changed")
    if not overlay_patch.is_file():
        problems.append("149-002 overlay patch is missing")
    elif sha256(overlay_patch) != EXPECTED_OVERLAY_PATCH_SHA256:
        problems.append("149-002 overlay patch hash changed")
    if package_manifest.is_file():
        package = json.loads(package_manifest.read_text())
        for rel, expected in package["artifacts"].items():
            path = PACKAGE / rel
            if not path.is_file() or sha256(path) != expected:
                problems.append(f"checker artifact does not match package manifest: {rel}")
    if overlay_manifest.is_file():
        overlay = json.loads(overlay_manifest.read_text())
        for rel, expected in overlay["overlay_source_hashes"].items():
            path = PACKAGE / "source" / rel
            if not path.is_file() or sha256(path) != expected:
                problems.append(f"copied source does not match 149-002 manifest: {rel}")
    if COMMAND_FILE.is_file():
        command_record = json.loads(COMMAND_FILE.read_text())
        expected_command = [str(Path(sys.executable)), str(RUNNER)]
        if command_record.get("run_command") != expected_command:
            problems.append("command.json run_command does not match the runner invocation")
        if command_record.get("case_path") != str(CASE):
            problems.append("command.json case_path mismatch")
        if RUNNER.is_file() and command_record.get("runner_sha256") != sha256(RUNNER):
            problems.append("command.json runner hash mismatch")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        problems.extend(f"missing bound input: {path}" for path in missing)
    return problems


def command_specs() -> list[dict[str, object]]:
    return [
        {
            "name": "identity-control",
            "argv": [str(PACKAGE / "native/build/base_cgls_identity_check")],
            "timeout_seconds": TIMEOUT_SECONDS_EACH,
            "expected_exit_code": 0,
            "expected_stdout_regex": r"base CGLS direct-kernel checks=678\b",
            "expected_checks": 678,
            "scope": "Existing identity-scale CGLS checker with useRightScale=0 and dummy slot 14.",
        },
        {
            "name": "right-scale",
            "argv": [str(PACKAGE / "native/build/base_cgls_right_scale_check")],
            "timeout_seconds": TIMEOUT_SECONDS_EACH,
            "expected_exit_code": 0,
            "expected_stdout_regex": r"source149 right-scale native checker checks=",
            "required_stdout_fields": [
                "scale_S_abs=",
                "scale_ST_abs=",
                "true_residual_before=",
                "true_residual_after=",
                "physical_transactions=0",
            ],
            "admission_assertions": [
                "inactive unsupported-tag environment and nonfinite block are ignored without errors",
                "selected nonfinite inverse block sets the invalid-right-scale error bit",
                "fixed, quiet, field, and float4 padding rows remain zero",
            ],
            "scope": "S/S-transpose and synthetic right-scaled CGLS recurrence with unscaled residual check.",
        },
    ]


def prediction(paths: list[Path]) -> dict[str, object]:
    return {
        "schema": "numi.matter.native_cgls_right_scale_run_prediction.v1",
        "case_id": CASE.name,
        "execution_order": "identity-control then right-scale, sequentially, one native checker process at a time",
        "serialized_worker_expectation": "one GPU worker; do not start until native459 has completed",
        "timeout_seconds_each": TIMEOUT_SECONDS_EACH,
        "source_overlay": {
            "id": "native-cgls-block-right-preconditioning-149-002",
            "manifest_sha256": EXPECTED_OVERLAY_MANIFEST_SHA256,
            "patch_sha256": EXPECTED_OVERLAY_PATCH_SHA256,
            "runtime_source_sha256": "f859b1ee45390dd844fcbd883356b931ab71d2c36260f53a4b03600b04fb3959",
            "fgmres_source_sha256": "8e0b9da3e28e7997eea9ee9db37e89a0bf3b9753d98967c80221e328e5da2af8",
            "base_cgls_source_sha256": "127d05d6323085ad6abd160ff560b95008981e9421541e9677dddbb3fa253aec",
        },
        "checker_package_manifest_sha256": EXPECTED_PACKAGE_MANIFEST_SHA256,
        "commands": command_specs(),
        "pre_execution_input_hashes": {
            str(path): sha256(path) if path.is_file() else None for path in paths
        },
        "expected_admission_outcomes": {
            "inactive_environment_with_unsupported_tag_and_nonfinite_scale": "ignored; no error",
            "selected_environment_with_nonfinite_inverse_block": "rejected with invalid-right-scale error bit",
            "nonfinite_active row/mask inputs": "must fail checker assertions rather than silently pass",
        },
        "claim_limits": [
            "This is a synthetic kernel/recurrence check, not production FEM preconditioner qualification.",
            "A passing pair of checkers does not establish nonlinear convergence or full-coupon success.",
            "The right-scale checker stdout contains a static build-time marker gpu_checker_not_run_here=1; execution status is determined only by this runner receipt.",
        ],
    }


def check_stdout(spec: dict[str, object], stdout_path: Path) -> list[str]:
    text = stdout_path.read_text(errors="replace")
    problems: list[str] = []
    pattern = spec.get("expected_stdout_regex")
    if pattern and re.search(str(pattern), text) is None:
        problems.append("expected stdout signature missing")
    for token in spec.get("required_stdout_fields", []):
        if token not in text:
            problems.append(f"expected stdout field missing: {token}")
    if spec["name"] == "right-scale":
        before = re.search(r"true_residual_before=([0-9.eE+-]+)", text)
        after = re.search(r"true_residual_after=([0-9.eE+-]+)", text)
        try:
            bvalue = float(before.group(1)) if before is not None else math.nan
            avalue = float(after.group(1)) if after is not None else math.nan
            if not (math.isfinite(bvalue) and math.isfinite(avalue) and avalue < bvalue):
                problems.append("true unscaled residual did not decrease")
        except ValueError:
            problems.append("true residual fields are not parseable finite numbers")
    return problems


def inventory(root: Path) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"):
        result[str(path.relative_to(root))] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    return result


def main() -> int:
    plan_only = "--plan" in sys.argv[1:]
    paths = bound_inputs()
    problems = expected_files_match(paths)
    plan = prediction(paths)
    if plan_only:
        print(json.dumps({"problems": problems, "prediction": plan}, indent=2))
        return 0 if not problems else 2
    # mkdir is the atomic fresh-case reservation. Existing attempts are retained.
    CASE.mkdir(parents=False, exist_ok=False)
    problems = expected_files_match(paths)
    plan = prediction(paths)
    (CASE / "prediction.json").write_text(json.dumps(plan, indent=2) + "\n")
    (CASE / "runner.py").write_bytes(RUNNER.read_bytes())
    (CASE / "command.json").write_text(json.dumps([sys.executable, str(RUNNER)], indent=2) + "\n")
    if problems:
        receipt = {
            "schema": "numi.matter.native_cgls_right_scale_run_receipt.v1",
            "case_id": CASE.name,
            "execution_started": False,
            "native_gpu_execution": False,
            "preflight_problems": problems,
            "input_hashes": {str(path): sha256(path) for path in paths if path.is_file()},
            "result": "rejected_before_execution",
            "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        (CASE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        (CASE / "manifest.json").write_text(json.dumps({"files": inventory(CASE)}, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
        return 2
    before = {str(path): sha256(path) for path in paths}
    if before != plan["pre_execution_input_hashes"]:
        problems.append("one or more bound inputs changed after preflight")
    if problems:
        receipt = {
            "schema": "numi.matter.native_cgls_right_scale_run_receipt.v1",
            "case_id": CASE.name,
            "execution_started": False,
            "native_gpu_execution": False,
            "preflight_problems": problems,
            "input_hashes": before,
            "result": "rejected_before_execution",
            "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        (CASE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        (CASE / "manifest.json").write_text(json.dumps({"files": inventory(CASE)}, indent=2) + "\n")
        print(json.dumps(receipt), flush=True)
        return 2
    run_records: list[dict[str, object]] = []
    overall_problems: list[str] = []
    for spec in command_specs():
        name = str(spec["name"])
        folder = CASE / name
        folder.mkdir()
        stdout_path, stderr_path = folder / "stdout.log", folder / "stderr.log"
        started = time.monotonic()
        started_at_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        timed_out = False
        exit_code: int | None = None
        try:
            with stdout_path.open("x") as out, stderr_path.open("x") as err:
                process = subprocess.run(
                    spec["argv"], stdout=out, stderr=err,
                    timeout=int(spec["timeout_seconds"]), check=False,
                )
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            overall_problems.append(f"{name} timed out")
        except Exception as exc:  # retain a visible launch failure record
            (folder / "launch-error.txt").write_text(repr(exc) + "\n")
            overall_problems.append(f"{name} failed to launch")
        output_problems = check_stdout(spec, stdout_path) if stdout_path.exists() else ["stdout was not produced"]
        if exit_code != spec["expected_exit_code"]:
            output_problems.append(f"exit code {exit_code} != expected {spec['expected_exit_code']}")
        if timed_out:
            output_problems.append("checker timed out")
        overall_problems.extend(f"{name}: {item}" for item in output_problems)
        record = {
            "name": name,
            "argv": spec["argv"],
            "started_at_utc": started_at_utc,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "timeout_seconds": spec["timeout_seconds"],
            "wall_seconds": time.monotonic() - started,
            "expected_exit_code": spec["expected_exit_code"],
            "output_contract_problems": output_problems,
            "admission_assertions": spec.get("admission_assertions", []),
            "assertion_interpretation": "Each listed assertion is embedded in the checker; exit_code=0 means it passed, while a nonzero exit preserves the first reported failure in stderr.",
        }
        (folder / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        run_records.append(record)
    after = {str(path): sha256(path) for path in paths}
    unchanged = before == after
    if not unchanged:
        overall_problems.append("one or more source/binary inputs changed during execution")
    receipt = {
        "schema": "numi.matter.native_cgls_right_scale_run_receipt.v1",
        "case_id": CASE.name,
        "execution_started": True,
        "one_worker_sequential": True,
        "native_gpu_execution": True,
        "physical_transactions": 0,
        "source_inputs_unchanged": unchanged,
        "before_hashes": before,
        "after_hashes": after,
        "runs": run_records,
        "overall_problems": overall_problems,
        "result": "pass" if not overall_problems else "fail_or_inconclusive",
        "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (CASE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (CASE / "manifest.json").write_text(json.dumps({"files": inventory(CASE)}, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
    return 0 if not overall_problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
