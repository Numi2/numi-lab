#!/usr/bin/env python3
"""Compare production output histories from a sealed diagnostic on/off pair.

This tool establishes byte parity for a narrowly admitted pair of native run
receipts. Parity is separate from whether either physical attempt completed or
was qualified. It never changes either run directory.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any


PREDICTION_SCHEMA = "numi.cardboard.native-prediction.v1"
RECEIPT_SCHEMA = "numi.cardboard.native-run-receipt.v2"
RESULT_SCHEMA = "numi.cardboard.probe-result.v1"
REPORT_SCHEMA = "numi.cardboard.diagnostic-history-comparison.v1"

# Only these variables may differ between the diagnostic-enabled and control
# runs. Adding a variable requires an explicit review that it cannot affect the
# physical trajectory. The capture-path variable is separately constrained to
# the corresponding attempt directory below.
DIAGNOSTIC_ENV_KEYS = frozenset({
    "NM_FEM_NEWTON_TRACE_ROOT",
    "NM_FEM_OPERATOR_FD_SELECTOR",
    "NM_FEM_OPERATOR_FD_CAPTURE_PATH",
})


class ComparisonError(ValueError):
    """The pair does not satisfy the diagnostic-history comparison contract."""


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonError(f"cannot read {description}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ComparisonError(f"{description} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_regular_file(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise ComparisonError(f"missing comparison file: {path}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ComparisonError(f"expected a regular non-symlink file: {path}")


def _same_bytes(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            aa = a.read(1024 * 1024)
            bb = b.read(1024 * 1024)
            if aa != bb:
                return False
            if not aa:
                return True


def _load_attempt(case_dir: Path, *, state_fixture: bool = False) -> dict[str, Any]:
    case_dir = case_dir.resolve(strict=True)
    if not case_dir.is_dir():
        raise ComparisonError(f"attempt path is not a directory: {case_dir}")
    prediction_path = case_dir / "prediction.json"
    receipt_path = case_dir / "receipt.json"
    prediction = _read_json(prediction_path, "prediction receipt")
    receipt = _read_json(receipt_path, "run receipt")
    if prediction.get("schema") != PREDICTION_SCHEMA:
        raise ComparisonError(f"unsupported prediction schema in {prediction_path}")
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ComparisonError(f"unsupported run receipt schema in {receipt_path}")
    if prediction.get("executed_build_verified") is not True:
        raise ComparisonError(f"build was not source-bound at launch: {case_dir}")

    # The exact retained file manifest is checked against its archive, rather
    # than merely comparing two self-reported dictionaries.
    bindings = prediction.get("inputs_sha256")
    if not isinstance(bindings, dict) or not bindings:
        raise ComparisonError(f"missing sealed input hashes: {case_dir}")
    for relative, expected in bindings.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ComparisonError(f"malformed input hash entry in {case_dir}")
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ComparisonError(f"unsafe sealed input path {relative!r}")
        path = case_dir / "inputs" / rel
        _safe_regular_file(path)
        if _sha256(path) != expected:
            raise ComparisonError(f"sealed input hash mismatch: {path}")

    source_snapshot = prediction.get("source_snapshot")
    runtime_dir = prediction.get("runtime_dir")
    build_path = prediction.get("build_provenance")
    external_inputs = prediction.get("external_inputs")
    if not all(isinstance(value, str) and value for value in
               (source_snapshot, runtime_dir, build_path)):
        raise ComparisonError(f"incomplete source/build identity: {case_dir}")
    if not isinstance(external_inputs, list) or not external_inputs:
        raise ComparisonError(f"missing external build provenance input: {case_dir}")
    build_file = case_dir / "inputs" / "external" / Path(external_inputs[0]).name
    _safe_regular_file(build_file)
    build = _read_json(build_file, "sealed source-build provenance")
    if (build.get("schema") != "numi.cardboard.source-build.v1" or
            build.get("success") is not True or
            Path(build.get("source_snapshot", "")).resolve() != Path(source_snapshot).resolve() or
            Path(build.get("build_dir", "")).resolve() != Path(runtime_dir).resolve() or
            Path(build_path).resolve() != Path(external_inputs[0]).resolve()):
        raise ComparisonError(f"source-build provenance does not match prediction: {case_dir}")
    external_key = f"external/{build_file.name}"
    if bindings.get(external_key) != _sha256(build_file):
        raise ComparisonError(f"source-build provenance is not bound by input hashes: {case_dir}")
    archived_source = {
        key[len("source/"):]: digest
        for key, digest in bindings.items()
        if key.startswith("source/")
    }
    if archived_source != build.get("source_sha256"):
        raise ComparisonError(f"archived source files do not match the source-build receipt: {case_dir}")

    argv = prediction.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise ComparisonError(f"malformed sealed argv: {case_dir}")
    output_flag = "--diagnostic-state-output" if state_fixture else "--output"
    if argv.count(output_flag) != 1:
        raise ComparisonError(f"expected exactly one {output_flag} argument: {case_dir}")
    output_index = argv.index(output_flag)
    if output_index + 1 >= len(argv):
        raise ComparisonError(f"missing --output value: {case_dir}")
    retained_output = case_dir / ("diagnostic-state" if state_fixture else "output")
    if retained_output.is_symlink():
        raise ComparisonError(f"output directory must not be a symlink: {retained_output}")
    output_dir = retained_output.resolve(strict=True)
    if Path(argv[output_index + 1]).resolve() != output_dir:
        raise ComparisonError(f"argv output directory does not identify retained output: {case_dir}")
    if output_index + 2 != len(argv):
        raise ComparisonError(f"--output must be the final argv pair: {case_dir}")
    artifact_hashes = build.get("artifacts_sha256")
    if not isinstance(artifact_hashes, dict):
        raise ComparisonError(f"source-build receipt lacks artifact hashes: {case_dir}")
    runtime_path = Path(runtime_dir).resolve()
    executed = Path(argv[0]).resolve()
    if executed not in {Path(path).resolve() for path in artifact_hashes}:
        raise ComparisonError(f"executed binary is not listed in source-build artifacts: {case_dir}")
    try:
        executed_key = "runtime/" + executed.relative_to(runtime_path).as_posix()
    except ValueError as error:
        raise ComparisonError(f"executed binary is outside the sealed runtime directory: {case_dir}") from error
    if bindings.get(executed_key) != artifact_hashes[str(executed)]:
        raise ComparisonError(f"executed binary archive does not match source-build artifact hash: {case_dir}")
    if "--metallib" in argv:
        metal_index = argv.index("--metallib")
        if metal_index + 1 >= len(argv):
            raise ComparisonError(f"missing --metallib value: {case_dir}")
        metal_path = Path(argv[metal_index + 1]).resolve()
        try:
            metal_key = "runtime/" + metal_path.relative_to(runtime_path).as_posix()
        except ValueError as error:
            raise ComparisonError(f"Metal library is outside the sealed runtime directory: {case_dir}") from error
        if (str(metal_path) not in artifact_hashes or
                bindings.get(metal_key) != artifact_hashes[str(metal_path)]):
            raise ComparisonError(f"Metal library archive does not match source-build artifact hash: {case_dir}")

    environment = prediction.get("environment")
    if not isinstance(environment, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in environment.items()):
        raise ComparisonError(f"malformed sealed environment: {case_dir}")

    if (receipt.get("execution_started") is not True or
            receipt.get("source_inputs_unchanged") is not True or
            receipt.get("changed_inputs") != [] or
            receipt.get("timed_out") is not False or
            not isinstance(receipt.get("exit_code"), int)):
        raise ComparisonError(f"attempt did not execute cleanly or timed out/changed inputs: {case_dir}")

    result = None
    expected_steps = 0
    if not state_fixture:
        result_path = output_dir / "result.json"
        result = _read_json(result_path, "probe result")
        if result.get("schema") != RESULT_SCHEMA:
            raise ComparisonError(f"unsupported probe result schema: {result_path}")
        manifest = _read_json(output_dir / "manifest.json", "probe manifest")
        solver = manifest.get("solver")
        if not isinstance(solver, dict):
            raise ComparisonError(f"probe manifest lacks solver protocol: {output_dir / 'manifest.json'}")
        phase_keys = ("steps", "hold_steps", "unload_steps", "relax_steps", "release_steps")
        for key in phase_keys:
            value = solver.get(key, 0)
            if not isinstance(value, int) or value < 0:
                raise ComparisonError(f"invalid manifest solver.{key}: {output_dir / 'manifest.json'}")
            expected_steps += value

    _validate_diagnostic_capture(case_dir, environment)
    output_files = _enumerate_output_files(output_dir, require_probe_result=not state_fixture)
    return {
        "path": case_dir,
        "prediction": prediction,
        "receipt": receipt,
        "build": build,
        "argv_without_output": argv[:output_index] + argv[output_index + 2:],
        "environment": environment,
        "result": result,
        "expected_steps": expected_steps,
        "output_files": output_files,
    }


def _validate_diagnostic_capture(case_dir: Path, environment: dict[str, str]) -> None:
    selector = environment.get("NM_FEM_OPERATOR_FD_SELECTOR")
    capture_path = environment.get("NM_FEM_OPERATOR_FD_CAPTURE_PATH")
    if (selector is None) != (capture_path is None):
        raise ComparisonError("FD selector and capture path must be supplied together")
    if selector is not None:
        if not re.fullmatch(r"\d+:\d+", selector):
            raise ComparisonError(f"invalid operator FD selector: {selector!r}")
        path = Path(capture_path)
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(case_dir)
        except (OSError, ValueError) as error:
            raise ComparisonError("diagnostic capture must be retained inside its own attempt directory") from error
        _safe_regular_file(resolved)
    trace = environment.get("NM_FEM_NEWTON_TRACE_ROOT")
    if trace is not None and trace not in {"0", "1"}:
        raise ComparisonError(f"invalid Newton trace selector: {trace!r}")


def _enumerate_output_files(output_dir: Path, *, require_probe_result: bool = True) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_symlink():
            raise ComparisonError(f"output tree contains a symlink: {path}")
        if path.is_dir():
            continue
        _safe_regular_file(path)
        relative = path.relative_to(output_dir).as_posix()
        files[relative] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    if not files:
        raise ComparisonError(f"empty production output tree: {output_dir}")
    if require_probe_result and "result.json" not in files:
        raise ComparisonError(f"production output lacks result.json: {output_dir}")
    return files


def _attempt_verdict(attempt: dict[str, Any]) -> dict[str, Any]:
    receipt = attempt["receipt"]
    result = attempt["result"]
    accepted_steps = result.get("accepted_steps")
    expected_steps = attempt["expected_steps"]
    protocol_complete = accepted_steps == expected_steps
    run_completed = receipt["exit_code"] == 0 and result.get("status") == "completed"
    if receipt["exit_code"] != 0 or result.get("status") == "failed":
        verdict = "failed_attempt"
    elif not run_completed:
        verdict = "incomplete_or_unrecognized_attempt"
    elif not protocol_complete:
        verdict = "completed_process_but_protocol_incomplete"
    elif result.get("physical_validation") is not True:
        verdict = "completed_but_physical_validation_false"
    else:
        verdict = "completed_physical_validation_true"
    return {
        "verdict": verdict,
        "exit_code": receipt["exit_code"],
        "result_status": result.get("status"),
        "accepted_steps": accepted_steps,
        "expected_protocol_steps": expected_steps,
        "protocol_complete": protocol_complete,
        "physical_validation": result.get("physical_validation"),
        "failure": result.get("failure"),
        "wall_seconds": receipt.get("wall_seconds"),
        "timed_out": receipt.get("timed_out"),
    }


def compare_attempts(left_path: str | Path, right_path: str | Path) -> dict[str, Any]:
    left = _load_attempt(Path(left_path))
    right = _load_attempt(Path(right_path))
    lp, rp = left["prediction"], right["prediction"]

    for field in ("source_snapshot", "runtime_dir", "build_provenance", "external_inputs",
                  "inputs_sha256", "executed_build_verified",
                  "source_binding_limitation"):
        if lp.get(field) != rp.get(field):
            raise ComparisonError(f"sealed run identity differs in prediction.{field}")
    if left["build"] != right["build"]:
        raise ComparisonError("source-build provenance content differs")
    if left["argv_without_output"] != right["argv_without_output"]:
        raise ComparisonError("physical argv differs beyond the declared output directory")
    if left["environment"].get("NM_FEM_RESIDUAL_ARMIJO_ROOT") != right["environment"].get(
            "NM_FEM_RESIDUAL_ARMIJO_ROOT"):
        raise ComparisonError("shared solver diagnostic/physics environment differs")
    if (left["environment"].get("NM_FEM_OPERATOR_FD_SELECTOR") is None or
            right["environment"].get("NM_FEM_OPERATOR_FD_SELECTOR") is not None or
            left["environment"].get("NM_FEM_NEWTON_TRACE_ROOT") != "1" or
            right["environment"].get("NM_FEM_NEWTON_TRACE_ROOT") is not None):
        raise ComparisonError("expected a diagnostic-enabled left run and diagnostic-disabled right control")

    changed_env_keys = sorted(key for key in set(left["environment"]) | set(right["environment"])
                              if left["environment"].get(key) != right["environment"].get(key))
    unexpected = sorted(set(changed_env_keys) - DIAGNOSTIC_ENV_KEYS)
    if unexpected:
        raise ComparisonError(f"environment changes are not allowlisted diagnostics: {unexpected}")
    if left["receipt"]["exit_code"] != right["receipt"]["exit_code"]:
        raise ComparisonError("attempt exit codes differ")

    left_files, right_files = left["output_files"], right["output_files"]
    names_left, names_right = set(left_files), set(right_files)
    differing: list[str] = sorted(names_left ^ names_right)
    common = sorted(names_left & names_right)
    for relative in common:
        a = left["path"] / "output" / relative
        b = right["path"] / "output" / relative
        if left_files[relative] != right_files[relative] or not _same_bytes(a, b):
            differing.append(relative)

    identical = not differing
    return {
        "schema": REPORT_SCHEMA,
        "created_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "comparison": "byte_identical" if identical else "different",
        "production_outputs_byte_identical": identical,
        "output_file_count": len(names_left),
        "identical_file_count": len(common) - len(set(differing) & names_left),
        "differing_files": sorted(set(differing)),
        "files": {
            "diagnostic_enabled": left_files,
            "control": right_files,
        },
        "sealed_identity": {
            "source_snapshot": lp["source_snapshot"],
            "runtime_dir": lp["runtime_dir"],
            "build_provenance": lp["build_provenance"],
            "build_provenance_sha256": hashlib.sha256(
                (left["path"] / "inputs" / "external" /
                 Path(lp["external_inputs"][0]).name).read_bytes()).hexdigest(),
            "inputs_sha256_equal": True,
            "argv_equal_except_output_directory": True,
            "changed_environment_keys": changed_env_keys,
            "diagnostic_environment_allowlist": sorted(DIAGNOSTIC_ENV_KEYS),
            "diagnostic_environment_changes_only": True,
            "timeout_seconds": {
                "diagnostic_enabled": lp.get("timeout_seconds"),
                "control": rp.get("timeout_seconds"),
                "matched": lp.get("timeout_seconds") == rp.get("timeout_seconds"),
                "note": "Timeout limits need not match when both receipts completed without timeout.",
            },
        },
        "attempts": {
            "diagnostic_enabled": _attempt_verdict(left),
            "control": _attempt_verdict(right),
        },
        "qualification": {
            "status": "not_qualified",
            "reason": "A byte-identical diagnostic/control pair only establishes output parity. "
                      "Failed attempts remain failed; parity does not establish convergence or physical validity.",
        },
    }


def write_report_exclusive(path: str | Path, report: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    # Exclusive creation prevents evidence from being silently overwritten.
    with target.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", required=True, help="sealed diagnostic-enabled attempt directory")
    parser.add_argument("--control", required=True, help="sealed diagnostic-disabled control directory")
    parser.add_argument("--output", required=True, help="new report JSON path; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        report = compare_attempts(args.diagnostic, args.control)
        write_report_exclusive(args.output, report)
    except (ComparisonError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({
        "comparison": report["comparison"],
        "production_outputs_byte_identical": report["production_outputs_byte_identical"],
        "output_file_count": report["output_file_count"],
        "attempt_verdicts": {
            key: value["verdict"] for key, value in report["attempts"].items()
        },
        "report": str(Path(args.output).resolve()),
    }, sort_keys=True))
    return 0 if report["production_outputs_byte_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
