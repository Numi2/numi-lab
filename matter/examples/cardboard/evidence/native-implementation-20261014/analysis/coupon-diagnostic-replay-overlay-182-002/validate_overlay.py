#!/usr/bin/env python3
"""Validate the isolated source-181-bound coupon diagnostic replay overlay."""

from __future__ import annotations

import difflib
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import tempfile


PACKAGE = pathlib.Path(__file__).resolve().parent
ROOT = pathlib.Path("/Users/home/cardboard-implementation-20261014")
BASE = ROOT / "snapshots/source-admission-encoder-181"
TOOLS = ("cardboard_coupon_resume.hpp", "cardboard_probe.mm")
TEST = PACKAGE / "tests/coupon_resume_contract_test.cpp"
PATCH = PACKAGE / "overlay.patch"
SYNTAX = PACKAGE / "validation/objcxx-syntax.json"
HOST_BINDINGS = PACKAGE / "validation/host-bindings.json"


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def make_patch() -> str:
    chunks = []
    for name in TOOLS:
        rel = f"matter/tools/{name}"
        before = (BASE / rel).read_text().splitlines(keepends=True)
        after = (PACKAGE / "candidate" / rel).read_text().splitlines(keepends=True)
        chunks.extend(difflib.unified_diff(
            before, after, fromfile=f"a/{rel}", tofile=f"b/{rel}", n=3))
    return "".join(chunks)


def verify_patch_application(expected_patch: str) -> None:
    with tempfile.TemporaryDirectory(prefix="coupon-overlay-182-") as tmp:
        stage = pathlib.Path(tmp)
        for name in TOOLS:
            rel = pathlib.Path("matter/tools") / name
            destination = stage / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(BASE / rel, destination)
        proc = subprocess.run(
            ["patch", "-p1", "--directory", str(stage)],
            input=expected_patch, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False)
        require(proc.returncode == 0, f"patch application failed:\n{proc.stdout}")
        for name in TOOLS:
            rel = pathlib.Path("matter/tools") / name
            require((stage / rel).read_bytes() ==
                    (PACKAGE / "candidate" / rel).read_bytes(),
                    f"patched bytes differ for {rel}")


def main() -> None:
    frozen = json.loads((BASE / "frozen-source.json").read_text())
    for name in TOOLS:
        rel = f"matter/tools/{name}"
        base_file = BASE / rel
        candidate = PACKAGE / "candidate" / rel
        require(base_file.is_file() and not base_file.is_symlink(),
                f"base source is missing or aliased: {rel}")
        require(candidate.is_file() and not candidate.is_symlink(),
                f"candidate source is missing or aliased: {rel}")
        require(frozen["source_sha256"].get(rel) == digest(base_file),
                f"source-181 frozen hash mismatch: {rel}")

    patch_text = make_patch()
    require(PATCH.read_text() == patch_text,
            "overlay.patch differs from the candidate against immutable source 181")
    verify_patch_application(patch_text)

    probe_text = (PACKAGE / "candidate/matter/tools/cardboard_probe.mm").read_text()
    runner_text = (BASE / "matter/tools/cardboard_solver_environment.py").read_text()
    registry_block = probe_text.split(
        "constexpr std::array<std::string_view, 51u> "
        "kSource161DiagnosticEnvironmentKeys{", 1)[1].split("};", 1)[0]
    probe_keys = re.findall(r'"([A-Z0-9_]+)"', registry_block)
    python_block = runner_text.split(
        "NATIVE_ENV_KEYS = tuple(sorted({", 1)[1].split("}))", 1)[0]
    python_keys = sorted(set(re.findall(r'"([A-Z0-9_]+)"', python_block)))
    require(len(probe_keys) == 51 and len(set(probe_keys)) == 51 and
            probe_keys == sorted(probe_keys) and probe_keys == python_keys,
            "probe and source-bound runner 51-key registries differ")
    require("[(NSArray*)keysValue count] == 49u" in probe_text and
            "[(NSArray*)keysValue count] == 50u" in probe_text and
            "[(NSArray*)keysValue count] == 51u" in probe_text and
            "contractHasWarmStartKey = keys.count >= 50u" in probe_text and
            "contractHasIncidenceSortKey = keys.count == 51u" in probe_text,
            "49/50/51 source registry transition predicates changed")
    require(probe_text.count(
                "couponValidatedRunnerDigest(currentRunnerValidated,") == 4 and
            "current_runner_proof_mode" in probe_text and
            "source_runner_proof_mode" in probe_text and
            "couponCurrentRunnerProofMode(currentRunnerValidated" in probe_text,
            "validated runner proof digests/modes are not emitted at the report boundary")
    contract_source = TEST.read_text()
    require("noPathCurrentProof" in contract_source and
            "source_bound_diagnostic_replay_contract" in contract_source and
            "couponValidatedRunnerDigest(false," in contract_source and
            "not-a-sha256" in contract_source,
            "C++ contract test omits validated no-path proof and digest failure cases")
    for count in (49, 50, 51):
        accepted = [key for key in probe_keys
                    if not (key == "NM_FEM_BASE_CGLS_WARM_START_REFINE" and count < 50)
                    and not (key == "NM_CONTACT_NODE_INCIDENCE_SORT" and count != 51)]
        require(len(accepted) == count and accepted == sorted(accepted),
                f"source registry variant {count} is malformed")

    with tempfile.TemporaryDirectory(prefix="coupon-contract-182-") as tmp:
        executable = pathlib.Path(tmp) / "coupon_resume_contract_test"
        command = [
            "clang++", "-std=c++20", "-Wall", "-Wextra", "-Werror",
            "-I", str(PACKAGE / "candidate/matter/tools"), str(TEST),
            "-o", str(executable),
        ]
        compile_result = subprocess.run(command, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        require(compile_result.returncode == 0,
                f"C++ contract test compile failed:\n{compile_result.stdout}")
        run_result = subprocess.run([str(executable)], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        require(run_result.returncode == 0,
                f"C++ contract tests failed:\n{run_result.stdout}")
        require("contract checks passed" in run_result.stdout,
                "C++ contract test did not emit its success marker")
        validation = {
            "schema": "numi.cardboard.coupon-diagnostic-replay-validation.v1",
            "command": command,
            "compile_exit": compile_result.returncode,
            "compile_output": compile_result.stdout,
            "run_exit": run_result.returncode,
            "run_output": run_result.stdout,
        }
        (PACKAGE / "validation/cpp-contract.json").write_text(
            json.dumps(validation, indent=2) + "\n")

    syntax = json.loads(SYNTAX.read_text())
    require(len(syntax) == 2 and
            {entry["target"] for entry in syntax} == {"default", "box-native"} and
            all(entry["exit"] == 0 and "-fsyntax-only" in entry["argv"]
                for entry in syntax),
            "remote ObjC++ syntax evidence is incomplete or failed")
    host = json.loads(HOST_BINDINGS.read_text())
    require(host["source181_probe_sha256"] ==
            digest(BASE / "matter/tools/cardboard_probe.mm"),
            "remote syntax base source differs from local frozen source")
    require(host["frozen_source_sha256"] == digest(BASE / "frozen-source.json"),
            "remote syntax used a different frozen-source manifest")
    host_artifacts = ROOT / "analysis/source181-host-artifacts-001"
    require(host["source_build_json_sha256"] ==
            digest(host_artifacts / "source-build.json"),
            "remote syntax build provenance differs from retained source-181 build")
    for name in TOOLS:
        rel = f"matter/tools/{name}"
        require(host["candidate_sha256"][name] ==
                digest(PACKAGE / "candidate" / rel),
                f"remote syntax candidate differs from local candidate: {name}")

    full = PACKAGE / "snapshots/source181-frozen-source.json"
    require(full.is_file(), "source-181 frozen-source copy is missing")
    copied = json.loads(full.read_text())
    require(copied == frozen, "source-181 frozen-source copy differs from base")

    candidate_hashes = {
        name: digest(PACKAGE / "candidate/matter/tools" / name)
        for name in TOOLS
    }
    (PACKAGE / "validation/candidate-hashes.json").write_text(
        json.dumps(candidate_hashes, indent=2) + "\n")

    validation_result = {
        "schema": "numi.cardboard.coupon-diagnostic-replay-overlay-validation.v1",
        "base_snapshot": str(BASE),
        "base_frozen_source_sha256": digest(BASE / "frozen-source.json"),
        "base_source_build_json_sha256": host["source_build_json_sha256"],
        "base_compile_commands_sha256": host["compile_commands_sha256"],
        "base_file_sha256": {
            name: digest(BASE / "matter/tools" / name) for name in TOOLS},
        "candidate_file_sha256": {
            name: digest(PACKAGE / "candidate/matter/tools" / name)
            for name in TOOLS},
        "patch_sha256": digest(PATCH),
        "contract_test_sha256": digest(TEST),
        "validation_script_sha256": digest(pathlib.Path(__file__)),
        "host_bindings_sha256": digest(HOST_BINDINGS),
        "remote_syntax_runner_sha256": digest(
            PACKAGE / "validation/run_remote_objcxx_syntax.py"),
        "objcxx_syntax_log_sha256": digest(SYNTAX),
        "objcxx_syntax_targets": [entry["target"] for entry in syntax],
        "contract_test": json.loads(
            (PACKAGE / "validation/cpp-contract.json").read_text()),
        "source_registry_count": len(probe_keys),
        "source_registry_consistency": "PASS",
        "patch_application": "PASS",
        "native_resume_replay": "NOT_RUN",
    }
    (PACKAGE / "validation/validation-result.json").write_text(
        json.dumps(validation_result, indent=2) + "\n")

    print("patch application: PASS (2 files)")
    print("C++ contract test: PASS")
    print("ObjC++ syntax: PASS (default, box-native)")
    print("source registry consistency: PASS (49/50/51 contract variants)")


if __name__ == "__main__":
    main()
