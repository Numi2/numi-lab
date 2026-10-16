#!/usr/bin/env python3
"""Strict pre-run BCT candidate/condition and executed-manifest binding."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

SCHEMA = "numi.cardboard.source-bound-calibration-execution.v1"
CASE_SCHEMA = "numi.cardboard.bct-native-case-binding.v1"
PLAN_SCHEMA = "numi.cardboard.calibration_ensemble.v1"
PROTOCOL_ID = "shipping_bct_without_offset_first_local_peak_v1"


class CalibrationBindingError(ValueError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def resolve_pointer(root: Any, pointer: str) -> Any:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise CalibrationBindingError(f"invalid JSON pointer: {pointer!r}")
    value = root
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list) and token.isdecimal() and int(token) < len(value):
            value = value[int(token)]
        elif isinstance(value, dict) and token in value:
            value = value[token]
        else:
            raise CalibrationBindingError(f"JSON pointer does not resolve: {pointer}")
    return value


def _json(path: Path, what: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationBindingError(f"cannot parse {what}: {exc}") from exc
    if not isinstance(value, dict):
        raise CalibrationBindingError(f"{what} must be a JSON object")
    return value


def _properties(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise CalibrationBindingError("capacity qualification has a malformed property line")
        key, value = line.split("=", 1)
        if not key or not value or key in result:
            raise CalibrationBindingError("capacity qualification has an empty or duplicate property")
        result[key] = value
    return result


def _single_flag(argv: list[str], flag: str) -> str:
    matches = [index for index, value in enumerate(argv) if value == flag]
    if len(matches) != 1 or matches[0] + 1 >= len(argv):
        raise CalibrationBindingError(f"execution argv must contain exactly one {flag} value")
    value = argv[matches[0] + 1]
    if value.startswith("--"):
        raise CalibrationBindingError(f"execution argv has no value after {flag}")
    return value


def _close_number(actual: Any, expected: float, *, where: str) -> float:
    if isinstance(actual, bool):
        raise CalibrationBindingError(f"{where} is boolean, expected numeric candidate value")
    try:
        parsed = float(actual)
    except (TypeError, ValueError) as exc:
        raise CalibrationBindingError(f"{where} is not numeric") from exc
    if not math.isfinite(parsed) or parsed != expected:
        raise CalibrationBindingError(f"{where}={actual!r} differs from candidate value {expected!r}")
    return parsed


def validate_pre_run(plan_path: Path, binding_path: Path, argv: list[str],
                     expected_output: Path) -> tuple[dict[str, Any], list[Path]]:
    plan_path, binding_path = plan_path.resolve(), binding_path.resolve()
    plan = _json(plan_path, "ensemble plan")
    binding = _json(binding_path, "prepared case binding")
    if plan.get("schema") != PLAN_SCHEMA or binding.get("schema") != CASE_SCHEMA:
        raise CalibrationBindingError("unsupported plan or prepared case-binding schema")
    plan_sha, binding_sha = sha256(plan_path), sha256(binding_path)
    if binding.get("plan_sha256") != plan_sha:
        raise CalibrationBindingError("prepared case binding does not match exact plan bytes")

    candidate = binding.get("candidate")
    condition = binding.get("condition")
    if not isinstance(candidate, dict) or not isinstance(condition, dict):
        raise CalibrationBindingError("prepared case binding lacks candidate or condition")
    candidate_body = {key: value for key, value in candidate.items() if key != "candidate_sha256"}
    if candidate.get("candidate_sha256") != canonical_sha256(candidate_body):
        raise CalibrationBindingError("candidate hash does not match canonical candidate fields")
    if candidate.get("role") != "fit_candidate":
        raise CalibrationBindingError("BCT native execution requires an explicit fit candidate")
    candidates = [row for row in plan.get("candidates", [])
                  if isinstance(row, dict) and row.get("candidate_id") == candidate.get("candidate_id")]
    conditions = [row for row in plan.get("conditions", [])
                  if isinstance(row, dict) and row.get("condition_id") == condition.get("condition_id")]
    if len(candidates) != 1 or candidates[0] != candidate:
        raise CalibrationBindingError("prepared candidate differs from or is absent in the sealed plan")
    if len(conditions) != 1 or conditions[0] != condition:
        raise CalibrationBindingError("prepared condition differs from or is absent in the sealed plan")
    if condition.get("native_protocol_id") != PROTOCOL_ID:
        raise CalibrationBindingError("condition native protocol differs from evaluator protocol")
    if condition.get("dataset_id") != "mrowczynski_2021_box_bct":
        raise CalibrationBindingError("condition is not the supported BCT dataset")
    if not isinstance(condition.get("specimen_id"), str) or not condition["specimen_id"]:
        raise CalibrationBindingError("condition specimen identity is missing")
    if (binding.get("source_data_catalog_sha256") != plan.get("source_data_catalog_sha256") or
            binding.get("target_recipe_id") != plan.get("target_recipe_id") or
            binding.get("target_grade_id") != plan.get("target_grade_id")):
        raise CalibrationBindingError("prepared case recipe, grade, or catalog identity differs from the plan")

    if "--shipping-box" not in argv:
        raise CalibrationBindingError("native argv must enable the shipping-box protocol")
    qualification = Path(_single_flag(argv, "--shipping-capacity-qualification")).resolve()
    if not qualification.is_file():
        raise CalibrationBindingError("capacity qualification file is missing before execution")
    props = _properties(qualification)
    if (props.get("schema") != "numi.cardboard.shipping-capacity-qualification.v2" or
            props.get("status") != "qualified_for_numerical_capacity_only"):
        raise CalibrationBindingError("shipping execution requires a complete v2 numerical-capacity qualification")
    report_paths: list[Path] = []
    report_hashes: dict[str, str] = {}
    for stem in ("mesh_report", "time_report"):
        name, expected = props.get(stem+"_file"), props.get(stem+"_sha256")
        if not name or Path(name).name != name or not expected:
            raise CalibrationBindingError(f"capacity qualification lacks a safe {stem} binding")
        report_path = qualification.parent / name
        if not report_path.is_file() or sha256(report_path) != expected:
            raise CalibrationBindingError(f"capacity qualification {stem} is missing or hash-mismatched")
        report_paths.append(report_path.resolve())
        report_hashes[stem+"_sha256"] = expected
    if Path(_single_flag(argv, "--output")).resolve() != expected_output.resolve():
        raise CalibrationBindingError("native output path must be the sealed case output directory")

    params = candidate.get("parameters")
    mappings = binding.get("parameter_bindings")
    if not isinstance(params, dict) or not params or not isinstance(mappings, dict) or set(params) != set(mappings):
        raise CalibrationBindingError("every fit-candidate parameter needs exactly one execution and manifest binding")
    checked: list[dict[str, Any]] = []
    external_paths = [plan_path, binding_path, qualification, *report_paths]
    seen_flags: set[str] = set()
    seen_pointers: set[str] = set()
    for name, spec in params.items():
        if (not isinstance(spec, dict) or not spec.get("unit") or not spec.get("family") or
                not isinstance(spec.get("fit"), bool)):
            raise CalibrationBindingError(f"candidate parameter {name!r} lacks unit/family/fit metadata")
        value = spec.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise CalibrationBindingError(f"candidate parameter {name!r} is not finite numeric data")
        mapping = mappings[name]
        if (not isinstance(mapping, dict) or mapping.get("kind") != "manifest_json_pointer" or
                not isinstance(mapping.get("pointer"), str) or not mapping["pointer"].startswith("/")):
            raise CalibrationBindingError(f"candidate parameter {name!r} lacks an executed-manifest JSON pointer")
        flag = mapping.get("argv_flag")
        if not isinstance(flag, str) or not flag.startswith("--"):
            raise CalibrationBindingError(f"candidate parameter {name!r} lacks a pre-run argv binding")
        if flag in seen_flags or mapping["pointer"] in seen_pointers:
            raise CalibrationBindingError("candidate parameters must use distinct argv flags and manifest pointers")
        seen_flags.add(flag)
        seen_pointers.add(mapping["pointer"])
        source = _single_flag(argv, flag)
        check: dict[str, Any] = {"name": name, "candidate_value": float(value),
                                 "manifest_pointer": mapping["pointer"], "argv_flag": flag}
        input_pointer = mapping.get("input_json_pointer")
        if input_pointer is None:
            observed = _close_number(source, float(value), where=f"argv {flag}")
            check.update({"kind": "argv_scalar", "argv_value": observed})
        else:
            input_path = Path(source).resolve()
            if not input_path.is_file():
                raise CalibrationBindingError(f"candidate parameter file for {name!r} is missing")
            input_data = _json(input_path, f"parameter input for {name}")
            observed = _close_number(resolve_pointer(input_data, input_pointer), float(value),
                                     where=f"parameter file {input_pointer}")
            check.update({"kind": "argv_json_pointer", "input_file": str(input_path),
                          "input_file_sha256": sha256(input_path),
                          "input_json_pointer": input_pointer, "input_value": observed})
            external_paths.append(input_path)
        checked.append(check)
    identity = {
        "schema": SCHEMA,
        "plan_sha256": plan_sha,
        "case_binding_sha256": binding_sha,
        "candidate_id": candidate["candidate_id"],
        "candidate_sha256": candidate["candidate_sha256"],
        "condition_id": condition["condition_id"],
        "specimen_id": condition["specimen_id"],
        "native_protocol_id": condition["native_protocol_id"],
        "parameter_bindings_sha256": canonical_sha256(mappings),
        "capacity_qualification_sha256": sha256(qualification),
        **report_hashes,
        "pre_run_parameter_checks": checked,
        "external_input_sha256": {str(path.resolve()): sha256(path) for path in external_paths},
    }
    # Preserve stable order while eliminating repeated explicit files.
    return identity, list(dict.fromkeys(external_paths))


def validate_executed_manifest(manifest_path: Path, identity: dict[str, Any],
                               binding_path: Path) -> dict[str, Any]:
    manifest = _json(manifest_path, "executed native manifest")
    binding = _json(binding_path, "prepared case binding")
    candidate = binding["candidate"]
    checks = []
    for name, spec in candidate["parameters"].items():
        pointer = binding["parameter_bindings"][name]["pointer"]
        observed = resolve_pointer(manifest, pointer)
        expected = float(spec["value"])
        parsed = _close_number(observed, expected, where=f"executed manifest {pointer}")
        checks.append({"name": name, "pointer": pointer, "expected": expected, "observed": parsed,
                       "matches": True})
    protocol_field = None
    protocol_status = "pre_run_plan_bound"
    for key in ("native_protocol_id", "shipping_protocol_id"):
        if key in manifest:
            protocol_field = key
            if manifest[key] != identity["native_protocol_id"]:
                raise CalibrationBindingError("executed manifest protocol identity differs from pre-run plan")
            protocol_status = "executed_manifest_match"
            break
    return {"status": "passed", "manifest_sha256": sha256(manifest_path),
            "parameter_checks": checks, "native_protocol_id": identity["native_protocol_id"],
            "protocol_manifest_field": protocol_field, "protocol_validation": protocol_status}
