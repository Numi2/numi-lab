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
    "NM_FEM_REGIONAL_RAP_CAPTURE_ONLY",
    # Changes only the private perturbation pair, never the production update.
    "NM_FEM_OPERATOR_FD_EPSILON_SCALE",
    "NM_FEM_TRUE_RESIDUAL_ROOT",
    # Zero is observational; one replaces the solver residual and is rejected.
    "NM_FEM_TRUE_RESIDUAL_RESTART",
    "NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR",
    "NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT",
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
    regional_only = environment.get("NM_FEM_REGIONAL_RAP_CAPTURE_ONLY")
    if regional_only not in (None, "0", "1"):
        raise ComparisonError("NM_FEM_REGIONAL_RAP_CAPTURE_ONLY must be exactly 0 or 1")
    if regional_only == "1" and (selector is None or not selector or
                                    capture_path is None or not capture_path):
        raise ComparisonError(
            "regional RAP-only capture requires both a nonempty FD selector and capture path")
    if (selector is None) != (capture_path is None):
        raise ComparisonError("FD selector and capture path must be supplied together")
    if selector is not None:
        if not re.fullmatch(r"\d+:\d+", selector):
            raise ComparisonError(f"invalid operator FD selector: {selector!r}")
        if not Path(capture_path).is_absolute():
            raise ComparisonError("diagnostic capture path must be absolute and retained inside its own attempt directory")
        if regional_only == "1":
            _validate_regional_rap_capture(case_dir, selector, capture_path)
        else:
            _retained_regular_file(case_dir, Path(capture_path),
                                   "diagnostic capture must be retained inside its own attempt directory")
    for key in ("NM_FEM_NEWTON_TRACE_ROOT", "NM_FEM_TRUE_RESIDUAL_ROOT",
                "NM_FEM_RESIDUAL_ARMIJO_TRACE_ROOT"):
        trace = environment.get(key)
        if trace is not None:
            if not re.fullmatch(r"\d+", trace) or int(trace) > 0xffffffff:
                raise ComparisonError(f"invalid diagnostic control selector {key}: {trace!r}")
            if selector is not None and int(trace) != int(selector.split(":", 1)[0]):
                raise ComparisonError("diagnostic-enabled traces must select the captured control step")
    replacement = environment.get("NM_FEM_TRUE_RESIDUAL_RESTART")
    if replacement is not None and (replacement != "0" or
            environment.get("NM_FEM_TRUE_RESIDUAL_ROOT") is None):
        raise ComparisonError("numerical algorithm difference: true residual replacement changes the solver; only observational restart=0 is admitted")
    newton = environment.get("NM_FEM_TRUE_RESIDUAL_NEWTON_SELECTOR")
    if newton is not None:
        if (not re.fullmatch(r"\d+", newton) or int(newton) > 256 or
                environment.get("NM_FEM_TRUE_RESIDUAL_ROOT") is None):
            raise ComparisonError("invalid true residual Newton selector")
        if selector is not None and int(newton) != int(selector.split(":", 1)[1]):
            raise ComparisonError("true residual Newton selector must select the captured Newton iteration")


def _retained_regular_file(case_dir: Path, path: Path, error_message: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(case_dir.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise ComparisonError(error_message) from error
    # Inspect the original path before accepting resolve(); otherwise a
    # symlink inside the attempt could masquerade as a retained regular file.
    _safe_regular_file(path)
    return resolved


def _validate_regional_rap_capture(
    case_dir: Path,
    selector: str,
    capture_path: str,
) -> None:
    base = Path(capture_path)
    manifest_path = Path(str(base) + ".regional.json")
    payload_path = Path(str(base) + ".regional.bin")
    _retained_regular_file(
        case_dir, manifest_path,
        "regional RAP manifest is missing or outside its own attempt directory")
    payload_resolved = _retained_regular_file(
        case_dir, payload_path,
        "regional RAP payload is missing or outside its own attempt directory")
    manifest = _read_json(manifest_path, "regional RAP manifest")
    if manifest.get("schema") != "numi.matter.regional_rap_capture.v1":
        raise ComparisonError("unsupported regional RAP capture schema")
    if manifest.get("byte_order") != "little_endian":
        raise ComparisonError("regional RAP payload must declare little_endian byte order")
    if manifest.get("layout") != (
            "C-contiguous; environment-major; each array is tightly concatenated with no padding; "
            "float32x4 elements store four adjacent little-endian lanes"):
        raise ComparisonError("regional RAP payload layout is missing or unsupported")
    try:
        selected_root, selected_iteration = (int(value) for value in selector.split(":"))
    except (TypeError, ValueError) as error:
        raise ComparisonError("invalid regional RAP selector") from error
    for key, expected in (("root", selected_root), ("iteration", selected_iteration)):
        value = manifest.get(key)
        if type(value) is not int or value != expected:
            raise ComparisonError(f"regional RAP manifest {key} does not match the selected iteration")
    if manifest.get("preconditioner_selector") not in (2, 3, 4, 5):
        raise ComparisonError("regional RAP manifest does not identify a regional preconditioner")
    capture_point = manifest.get("capture_point")
    if capture_point not in (
            "after_regional_rap_factor_barrier_before_fgmres_correction",
            "after_regional_rap_and_graph_factor_barrier_before_fgmres_correction"):
        raise ComparisonError("regional RAP capture point is missing or unsupported")

    payload = manifest.get("payload")
    if not isinstance(payload, dict):
        raise ComparisonError("regional RAP manifest lacks payload metadata")
    if payload.get("file") != str(payload_path):
        raise ComparisonError("regional RAP manifest payload path does not match its retained sidecar")
    payload_bytes = payload_resolved.read_bytes()
    payload_hash = hashlib.sha256(payload_bytes).hexdigest()
    if (type(payload.get("byte_count")) is not int or
            payload.get("byte_count") != len(payload_bytes) or
            payload.get("sha256") != payload_hash):
        raise ComparisonError("regional RAP payload size or SHA-256 does not match its manifest")

    arrays = manifest.get("arrays")
    if not isinstance(arrays, list) or not arrays:
        raise ComparisonError("regional RAP manifest has no array table")
    required = {"candidate_nodes", "material_history", "scheduler_exponents",
                "deformation_high", "deformation_low", "P", "effective_P",
                "AP", "pivoted_RAP", "pivots", "factor_valid"}
    seen: set[str] = set()
    offset = 0
    validated: dict[str, tuple[int, int]] = {}
    for array in arrays:
        if not isinstance(array, dict):
            raise ComparisonError("regional RAP array entry must be an object")
        name = array.get("name")
        dtype = array.get("dtype")
        shape = array.get("shape")
        array_offset = array.get("offset")
        byte_count = array.get("byte_count")
        element_bytes = array.get("element_bytes")
        if (not isinstance(name, str) or not name or name in seen or
                not isinstance(dtype, str) or not dtype or
                not isinstance(shape, list) or not shape or
                any(type(extent) is not int or extent <= 0 for extent in shape) or
                type(array_offset) is not int or type(byte_count) is not int or
                type(element_bytes) is not int or element_bytes <= 0):
            raise ComparisonError("regional RAP array metadata is malformed or duplicated")
        element_count = 1
        for extent in shape:
            element_count *= extent
            if element_count > len(payload_bytes) + 1:
                raise ComparisonError("regional RAP array shape exceeds its payload")
        if (array_offset != offset or byte_count != element_count * element_bytes or
                byte_count > len(payload_bytes) - offset):
            raise ComparisonError("regional RAP arrays are not tightly packed within the payload")
        array_bytes = payload_bytes[offset:offset + byte_count]
        if array.get("sha256") != hashlib.sha256(array_bytes).hexdigest():
            raise ComparisonError(f"regional RAP array hash mismatch: {name}")
        seen.add(name)
        validated[name] = (array_offset, byte_count)
        offset += byte_count
    if offset != len(payload_bytes) or not required.issubset(seen):
        raise ComparisonError("regional RAP array table is incomplete or does not cover its payload")

    environment_count = manifest.get("environment_count")
    if type(environment_count) is not int or environment_count <= 0:
        raise ComparisonError("regional RAP environment_count is invalid")
    identities = manifest.get("candidate_linearization_sha256")
    if not isinstance(identities, list) or len(identities) != environment_count:
        raise ComparisonError("regional RAP candidate identity table is incomplete")
    identity_arrays = ["candidate_nodes", "material_history", "scheduler_exponents",
                       "deformation_high", "deformation_low"]
    tangent_cache_used = manifest.get("tangent_cache_used")
    if type(tangent_cache_used) is not bool:
        raise ComparisonError("regional RAP tangent_cache_used must be boolean")
    if tangent_cache_used:
        required_cache = {"algorithmic_tangent_cache", "algorithmic_tangent_cache_valid"}
        if not required_cache.issubset(seen):
            raise ComparisonError("regional RAP tangent-cache identity arrays are missing")
        identity_arrays.extend(sorted(required_cache))
    for environment, entry in enumerate(identities):
        if not isinstance(entry, dict) or entry.get("environment") != environment:
            raise ComparisonError("regional RAP candidate identity order is invalid")
        if entry.get("identity_status") == "missing_arrays":
            raise ComparisonError("regional RAP candidate identity was not captured")
        if not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", ""))):
            raise ComparisonError("regional RAP candidate identity SHA-256 is malformed")
        identity_digest = hashlib.sha256()
        for name in identity_arrays:
            start, size = validated[name]
            if size % environment_count != 0:
                raise ComparisonError(f"regional RAP identity array is not environment-major: {name}")
            per_environment = size // environment_count
            start += environment * per_environment
            identity_digest.update(payload_bytes[start:start + per_environment])
        if identity_digest.hexdigest() != entry["sha256"]:
            raise ComparisonError(f"regional RAP candidate linearization hash mismatch for environment {environment}")
        nodes_start, nodes_size = validated["candidate_nodes"]
        nodes_size //= environment_count
        nodes_start += environment * nodes_size
        if entry.get("candidate_nodes_sha256") != hashlib.sha256(
                payload_bytes[nodes_start:nodes_start + nodes_size]).hexdigest():
            raise ComparisonError(f"regional RAP candidate-node hash mismatch for environment {environment}")


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
            any(right["environment"].get(key) is not None for key in DIAGNOSTIC_ENV_KEYS)):
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
