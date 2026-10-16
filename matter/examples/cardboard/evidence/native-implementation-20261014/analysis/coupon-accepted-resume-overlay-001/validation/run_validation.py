#!/usr/bin/env python3
"""Re-run the bounded CPU contracts and source157 ObjC++ syntax checks."""
from __future__ import annotations
import hashlib
import json
import pathlib
import shlex
import subprocess
import tempfile

PACKAGE = pathlib.Path(__file__).resolve().parents[1]
BASE = pathlib.Path("/Users/home/cardboard-implementation-20261014/snapshots/source-right-grip-telemetry-157")
BUILD_METADATA = pathlib.Path("/Users/home/cardboard-implementation-20261014/analysis/source157-telemetry-native-build-001/build-metadata/compile_commands.json")
VALIDATION = PACKAGE / "validation"
PROBE = PACKAGE / "candidate/matter/tools/cardboard_probe.mm"
TOOLS_INCLUDE = BASE / "matter/tools"


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_logged(label: str, argv: list[str], cwd: pathlib.Path) -> int:
    completed = subprocess.run(argv, cwd=cwd, text=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, check=False)
    (VALIDATION / f"{label}.log").write_text(
        f"exit_code={completed.returncode}\n" + completed.stdout,
        encoding="utf-8")
    return completed.returncode


def main() -> int:
    VALIDATION.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="coupon-resume-contract-") as temp:
        test_binary = pathlib.Path(temp) / "coupon_resume_contract_test"
        test_source = PACKAGE / "tests/coupon_resume_contract_test.cpp"
        if run_logged("contract-compile", ["/usr/bin/clang++", "-std=c++20",
                "-Wall", "-Wextra", "-Werror", str(test_source), "-o",
                str(test_binary)], BASE) != 0:
            return 1
        if run_logged("contract-run", [str(test_binary)], BASE) != 0:
            return 1

    commands = json.loads(BUILD_METADATA.read_text(encoding="utf-8"))
    variants = [entry for entry in commands
                if entry["file"].endswith("cardboard_probe.mm")]
    if len(variants) != 2:
        raise RuntimeError(f"expected two source-bound probe commands, found {len(variants)}")
    for index, entry in enumerate(variants, 1):
        argv = entry.get("arguments") or shlex.split(entry["command"])
        rewritten: list[str] = []
        skip_output = False
        for arg in argv:
            if skip_output:
                skip_output = False
                continue
            if arg == "-o":
                skip_output = True
                continue
            if arg == "-c":
                continue
            if arg == entry["file"]:
                arg = str(PROBE)
            rewritten.append(arg)
        rewritten.extend([f"-I{TOOLS_INCLUDE}", "-fsyntax-only"])
        label = ("syntax-box-native" if
                 "-DNUMI_CARDBOARD_BOX_NATIVE_DRIVER=1" in rewritten
                 else "syntax-coupon")
        if run_logged(label, rewritten, pathlib.Path(entry["directory"])) != 0:
            return 1

    patch_path = PACKAGE / "overlay.patch"
    patch_check = subprocess.run(
        ["/usr/bin/patch", "--dry-run", "-p1", "-d", str(BASE)],
        input=patch_path.read_text(encoding="utf-8"), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (VALIDATION / "patch-dry-run.log").write_text(
        f"exit_code={patch_check.returncode}\n" + patch_check.stdout,
        encoding="utf-8")
    if patch_check.returncode != 0:
        return 1

    report = {
        "schema": "numi.cardboard.coupon-resume-validation.v1",
        "source157_probe_sha256": digest(BASE / "matter/tools/cardboard_probe.mm"),
        "source157_snapshot_header_sha256": digest(BASE / "matter/tools/cardboard_shipping_snapshot.hpp"),
        "source157_driver_header_sha256": digest(BASE / "matter/tools/cardboard_shipping_driver.hpp"),
        "compile_commands_sha256": digest(BUILD_METADATA),
        "candidate_probe_sha256": digest(PROBE),
        "candidate_resume_header_sha256": digest(PACKAGE / "candidate/matter/tools/cardboard_coupon_resume.hpp"),
        "contract_test_sha256": digest(PACKAGE / "tests/coupon_resume_contract_test.cpp"),
        "overlay_patch_sha256": digest(patch_path),
        "contract_tests": "PASS",
        "ordinary_coupon_probe_syntax": "PASS",
        "box_native_probe_syntax": "PASS",
        "patch_dry_run": "PASS",
        "native_build_or_gpu_run": "NOT_RUN",
    }
    (VALIDATION / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
