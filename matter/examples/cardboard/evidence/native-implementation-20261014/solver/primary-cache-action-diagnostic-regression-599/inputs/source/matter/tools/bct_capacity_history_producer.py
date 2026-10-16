#!/usr/bin/env python3
"""Turn an already completed native BCT capacity run into ensemble artifacts.

This wrapper never changes a raw native run. The pre-run case binding and plan
must appear in prediction.json's external-input hash map; parameter values must
map back to concrete executed-manifest fields. It emits a new history,
manifest, receipt, certificate, and plan-record fragment in a fresh directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cardboard_calibration_case_binding import (
    CalibrationBindingError, canonical_sha256 as _binding_canonical_sha256,
    validate_pre_run as _validate_pre_run_binding,
)

SCHEMA = "numi.cardboard.bct-native-case-binding.v1"
BINDING_SCHEMA = "numi.cardboard.native-bct-capacity-binding.v1"
DATASET_ID = "mrowczynski_2021_box_bct"
PROTOCOL_ID = "shipping_bct_without_offset_first_local_peak_v1"


class ProducerError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProducerError(f"cannot read {what} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProducerError(f"{what} must be a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2,
                               ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _resolve_pointer(root: Any, pointer: str) -> Any:
    if pointer == "":
        return root
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ProducerError(f"invalid JSON pointer {pointer!r}")
    value = root
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not token.isdecimal() or int(token) >= len(value):
                raise ProducerError(f"JSON pointer does not resolve: {pointer}")
            value = value[int(token)]
        elif isinstance(value, dict) and token in value:
            value = value[token]
        else:
            raise ProducerError(f"JSON pointer does not resolve: {pointer}")
    return value


def _check_candidate(candidate: dict[str, Any]) -> None:
    body = {key: value for key, value in candidate.items()
            if key != "candidate_sha256"}
    if candidate.get("candidate_sha256") != canonical_sha256(body):
        raise ProducerError("candidate_sha256 does not match the canonical candidate")
    if not isinstance(candidate.get("candidate_id"), str) or not candidate["candidate_id"]:
        raise ProducerError("candidate_id is missing")
    if candidate.get("role") not in {"fit_candidate", "unbonded_control", "tied_control"}:
        raise ProducerError("unsupported candidate role")
    params = candidate.get("parameters", {})
    if not isinstance(params, dict):
        raise ProducerError("candidate parameters must be an object")
    if candidate["role"] == "fit_candidate" and not params:
        raise ProducerError("fit candidate must have explicit parameters")
    for name, spec in params.items():
        if not isinstance(spec, dict):
            raise ProducerError(f"candidate parameter {name!r} must be an object")
        value = spec.get("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ProducerError(f"candidate parameter {name!r} must have a finite numeric value")
        if not spec.get("unit") or not spec.get("family") or not isinstance(spec.get("fit"), bool):
            raise ProducerError(f"candidate parameter {name!r} lacks family/unit/fit metadata")


def prepare_case_binding(plan_path: Path, candidate_id: str, condition_id: str,
                         parameter_bindings: dict[str, Any], output_path: Path) -> dict[str, Any]:
    plan = read_json(plan_path, "ensemble plan")
    if plan.get("schema") != "numi.cardboard.calibration_ensemble.v1":
        raise ProducerError("unsupported calibration ensemble plan schema")
    candidates = [row for row in plan.get("candidates", [])
                  if isinstance(row, dict) and row.get("candidate_id") == candidate_id]
    conditions = [row for row in plan.get("conditions", [])
                  if isinstance(row, dict) and row.get("condition_id") == condition_id]
    if len(candidates) != 1 or len(conditions) != 1:
        raise ProducerError("candidate and condition IDs must each resolve exactly once in the plan")
    candidate, condition = candidates[0], conditions[0]
    _check_candidate(candidate)
    if condition.get("dataset_id") != DATASET_ID:
        raise ProducerError("this producer only supports the published BCT dataset")
    if condition.get("native_protocol_id") != PROTOCOL_ID:
        raise ProducerError("BCT condition does not use the source-bound shipping protocol")
    if not isinstance(condition.get("specimen_id"), str) or not condition["specimen_id"]:
        raise ProducerError("BCT specimen_id is required")
    params = candidate.get("parameters", {})
    if set(parameter_bindings) != set(params):
        raise ProducerError("every candidate parameter needs exactly one executed-manifest JSON pointer")
    seen_pointers: set[str] = set()
    seen_flags: set[str] = set()
    for name, binding in parameter_bindings.items():
        if (not isinstance(binding, dict) or binding.get("kind") != "manifest_json_pointer" or
                not isinstance(binding.get("pointer"), str) or
                not binding["pointer"].startswith("/")):
            raise ProducerError(f"parameter {name!r} needs a manifest_json_pointer binding")
        flag = binding.get("argv_flag")
        if not isinstance(flag, str) or not flag.startswith("--"):
            raise ProducerError(f"parameter {name!r} needs an explicit pre-run argv_flag binding")
        if binding["pointer"] in seen_pointers or flag in seen_flags:
            raise ProducerError("each candidate parameter must have a distinct manifest pointer and argv flag")
        seen_pointers.add(binding["pointer"]); seen_flags.add(flag)
        input_pointer = binding.get("input_json_pointer")
        if input_pointer is not None and (not isinstance(input_pointer, str) or not input_pointer.startswith("/")):
            raise ProducerError(f"parameter {name!r} has an invalid input_json_pointer")
    value = {
        "schema": SCHEMA,
        "plan_sha256": sha256(plan_path),
        "candidate": candidate,
        "condition": condition,
        "source_data_catalog_sha256": plan.get("source_data_catalog_sha256"),
        "target_recipe_id": plan.get("target_recipe_id"),
        "target_grade_id": plan.get("target_grade_id"),
        "parameter_bindings": parameter_bindings,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, value)
    return value


def _verify_published_condition(condition: dict[str, Any], catalog_path: Path,
                                data_dir: Path, expected_catalog_sha: str,
                                target_grade_id: str) -> None:
    tools_dir = catalog_path.resolve().parent.parent.parent.parent / "tools"
    if not tools_dir.is_dir():
        raise ProducerError("cannot locate the checkout's cardboard source-data validators")
    sys.path.insert(0, str(tools_dir))
    try:
        import cardboard_calibration_ensemble as ensemble
        catalog, catalog_sha, datasets = ensemble.load_source_package(catalog_path, data_dir)
    except Exception as exc:
        raise ProducerError(f"published source package failed validation: {exc}") from exc
    if catalog_sha != expected_catalog_sha:
        raise ProducerError("source catalog bytes differ from the pre-run plan binding")
    dataset_id = condition.get("dataset_id")
    dataset = datasets.get(dataset_id)
    if dataset is None or dataset_id != DATASET_ID:
        raise ProducerError("condition dataset is not the validated BCT dataset")
    if condition.get("source_dataset_sha256") != dataset.get("sha256"):
        raise ProducerError("BCT dataset digest differs from the published catalog")
    source_id = condition.get("source_id")
    if source_id not in dataset.get("source_ids", []):
        raise ProducerError("BCT source identity is not declared by the published dataset")
    source = next((row for row in catalog.get("sources", []) if row.get("id") == source_id), None)
    if source is None:
        raise ProducerError("BCT source identity is absent from the catalog")
    source_hash = source.get("external_original_sha256") or source.get("source_text_extract_sha256")
    if not isinstance(source_hash, str) or condition.get("source_asset_sha256") != source_hash:
        raise ProducerError("BCT publisher asset/text hash differs from the source catalog")
    if condition.get("source_catalog_sha256") != catalog_sha:
        raise ProducerError("BCT condition does not bind the validated catalog bytes")
    try:
        selected_rows, _ = ensemble._selected_rows(condition, dataset)
    except Exception as exc:
        raise ProducerError(f"BCT condition selector is not a complete published fixture: {exc}") from exc
    if not selected_rows:
        raise ProducerError("BCT whole-fixture selector matched no source rows")
    transfer = condition.get("transfer")
    if not isinstance(transfer, dict) or transfer.get("kind") != "bounded_analogue":
        raise ProducerError("BCT catalog holdout requires an explicit bounded_analogue transfer")
    if (not target_grade_id or transfer.get("target_grade") != target_grade_id or
            not str(transfer.get("rationale", "")).strip()):
        raise ProducerError("BCT analogue target grade/rationale is missing or differs from the plan")
    selected_grades = {str(row.get("grade", "")).strip() for row in selected_rows}
    if (len(selected_grades) != 1 or "" in selected_grades or
            transfer.get("source_grade") not in selected_grades):
        raise ProducerError("BCT source grade must exactly match the selected published grade")


def _external_hash_key(prediction: dict[str, Any], path: Path) -> str:
    raw_inputs = prediction.get("inputs_sha256")
    external = prediction.get("external_inputs")
    if not isinstance(raw_inputs, dict) or not isinstance(external, list):
        raise ProducerError("prediction lacks external-input hash bindings")
    resolved = path.resolve()
    if not any(Path(str(item)).resolve() == resolved for item in external):
        raise ProducerError(f"pre-run artifact is absent from prediction.external_inputs: {path.name}")
    matches = [key for key, digest in raw_inputs.items()
               if str(key).startswith("external/") and
               Path(str(key)).name == path.name and digest == sha256(path)]
    if len(matches) != 1:
        raise ProducerError(f"prediction does not uniquely hash pre-run artifact {path.name}")
    return matches[0]


def _verify_prediction_inputs(prediction_path: Path, prediction: dict[str, Any],
                              plan_path: Path, binding_path: Path,
                              binding: dict[str, Any]) -> dict[str, Any]:
    _external_hash_key(prediction, plan_path)
    _external_hash_key(prediction, binding_path)
    argv = prediction.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise ProducerError("prediction argv is missing")
    if "--shipping-box" not in argv:
        raise ProducerError("prediction argv does not enable the shipping box protocol")
    if "--shipping-capacity-qualification" not in argv:
        raise ProducerError("prediction argv does not bind a v2 capacity qualification")
    try:
        qualification_path = Path(argv[argv.index("--shipping-capacity-qualification") + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise ProducerError("prediction has a malformed capacity qualification argument") from exc
    _external_hash_key(prediction, qualification_path)
    output_flag = "--output"
    try:
        recorded_output = Path(argv[argv.index(output_flag) + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise ProducerError("prediction argv does not bind the native output directory") from exc
    if recorded_output != (prediction_path.parent / "output").resolve():
        raise ProducerError("prediction output path differs from the supplied native run directory")
    raw_inputs = prediction["inputs_sha256"]
    binary = Path(argv[0]).resolve()
    if not binary.is_file() or not any(
            Path(key).name == binary.name and digest == sha256(binary)
            for key, digest in raw_inputs.items()):
        raise ProducerError("prediction does not hash the executed native binary")
    candidate, condition = binding["candidate"], binding["condition"]
    execution_binding = prediction.get("calibration_case_binding")
    if not isinstance(execution_binding, dict):
        raise ProducerError("source-bound harness did not seal a pre-run calibration identity")
    try:
        expected_identity, required_inputs = _validate_pre_run_binding(
            plan_path, binding_path, argv, prediction_path.parent / "output")
    except CalibrationBindingError as exc:
        raise ProducerError(f"prepared BCT binding failed independent pre-run validation: {exc}") from exc
    if _binding_canonical_sha256(execution_binding) != _binding_canonical_sha256(expected_identity):
        raise ProducerError("sealed pre-run calibration identity differs from the independently recomputed binding")
    if prediction.get("native_protocol_id") != condition["native_protocol_id"]:
        raise ProducerError("prediction protocol identity differs from prepared condition")
    if not isinstance(prediction.get("external_inputs"), list) or not all(
            any(Path(str(item)).resolve() == path.resolve()
                for item in prediction["external_inputs"])
            for path in (plan_path, binding_path)):
        raise ProducerError("plan and prepared binding must be explicit pre-run harness inputs")
    for external_path in required_inputs:
        key = _external_hash_key(prediction, external_path)
        copied = prediction_path.parent / "inputs" / key
        if not copied.is_file() or sha256(copied) != sha256(external_path):
            raise ProducerError(f"source-bound harness did not retain exact external input bytes: {external_path.name}")
    _verify_candidate_argv_bindings(binding, argv, raw_inputs)
    return execution_binding


def _argv_value(argv: list[str], flag: str) -> str:
    matches = [i for i, token in enumerate(argv) if token == flag]
    if len(matches) != 1 or matches[0] + 1 >= len(argv):
        raise ProducerError(f"sealed native argv must contain exactly one value for {flag}")
    value = argv[matches[0] + 1]
    if value.startswith("--"):
        raise ProducerError(f"sealed native argv lacks a value for {flag}")
    return value


def _verify_candidate_argv_bindings(binding: dict[str, Any], argv: list[str],
                                    input_hashes: dict[str, Any]) -> None:
    for name, spec in binding["candidate"].get("parameters", {}).items():
        mapping = binding["parameter_bindings"][name]
        flag = mapping.get("argv_flag")
        if not isinstance(flag, str):
            raise ProducerError(f"candidate parameter {name!r} lacks a sealed argv binding")
        actual = _argv_value(argv, flag)
        expected = float(spec["value"])
        input_pointer = mapping.get("input_json_pointer")
        if input_pointer is None:
            try:
                parsed = float(actual)
            except ValueError as exc:
                raise ProducerError(f"argv value for {name!r} is not numeric") from exc
            if not math.isfinite(parsed) or parsed != expected:
                raise ProducerError(f"argv value for {name!r} differs from prepared candidate")
        else:
            path = Path(actual).resolve()
            if not path.is_file():
                raise ProducerError(f"argv input file for {name!r} is missing")
            matches = [key for key, digest in input_hashes.items()
                       if Path(str(key)).name == path.name and digest == sha256(path)]
            if len(matches) != 1:
                raise ProducerError(f"argv input file for {name!r} is not uniquely source-bound")
            data = read_json(path, f"candidate parameter file {name}")
            parsed = _resolve_pointer(data, input_pointer)
            if isinstance(parsed, bool) or not isinstance(parsed, (int, float)) or float(parsed) != expected:
                raise ProducerError(f"parameter file value for {name!r} differs from prepared candidate")


def _copy_verified(src: Path, destination: Path, expected_sha: str | None = None) -> str:
    if not src.is_file():
        raise ProducerError(f"required native artifact is missing: {src}")
    actual = sha256(src)
    if expected_sha is not None and actual != expected_sha:
        raise ProducerError(f"native artifact digest mismatch: {src.name}")
    shutil.copyfile(src, destination)
    if sha256(destination) != actual:
        raise ProducerError(f"copied artifact digest mismatch: {destination.name}")
    return actual


def _resolve_checkpoint_pair(output_root: Path, matter_sha: str,
                             rigid_sha: str) -> tuple[Path, Path, Path]:
    metadata_path = output_root / "shipping-rigid-accepted-checkpoint.json"
    metadata = read_json(metadata_path, "paired accepted checkpoint metadata")
    if metadata.get("schema") != "numi.cardboard.shipping-rigid-checkpoint.v1":
        raise ProducerError("paired checkpoint metadata has an unsupported schema")
    if metadata.get("fem_checkpoint_sha256") != matter_sha:
        raise ProducerError("paired checkpoint metadata does not bind the qualified Matter checkpoint")
    if metadata.get("paired_scene_sidecar_sha256") != rigid_sha:
        raise ProducerError("paired checkpoint metadata does not bind the qualified rigid sidecar")
    checkpoint_name = metadata.get("fem_checkpoint_file")
    rigid_name = metadata.get("paired_scene_sidecar_file")
    if (not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name or
            not isinstance(rigid_name, str) or Path(rigid_name).name != rigid_name):
        raise ProducerError("paired checkpoint metadata filenames must be safe basenames")
    checkpoint_path = output_root / "checkpoints" / checkpoint_name
    rigid_path = output_root / "checkpoints" / rigid_name
    if not checkpoint_path.is_file() or sha256(checkpoint_path) != matter_sha:
        raise ProducerError("paired Matter checkpoint is missing or its hash differs")
    if not rigid_path.is_file() or sha256(rigid_path) != rigid_sha:
        raise ProducerError("paired rigid sidecar is missing or its hash differs")
    return metadata_path, checkpoint_path, rigid_path


def _verify_parameter_bindings(binding: dict[str, Any], native_manifest: dict[str, Any]) -> None:
    params = binding["candidate"].get("parameters", {})
    mappings = binding.get("parameter_bindings")
    if not isinstance(mappings, dict) or set(mappings) != set(params):
        raise ProducerError("pre-run binding does not cover every candidate parameter")
    for name, spec in params.items():
        pointer = mappings[name].get("pointer")
        observed = _resolve_pointer(native_manifest, pointer)
        expected = float(spec["value"])
        if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(observed):
            raise ProducerError(f"executed manifest parameter {name!r} is not finite numeric data")
        if float(observed) != expected:
            raise ProducerError(f"executed manifest value for parameter {name!r} differs from its pre-run candidate")


def _load_curve(path: Path, expected_count: int, peak_index: int) -> list[dict[str, Any]]:
    required = ["sample_index", "stroke_m", "force_N", "contact_samples"]
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != required:
            raise ProducerError("native nominal curve has an unsupported CSV schema")
        rows = list(reader)
    if len(rows) != expected_count or len(rows) < 3:
        raise ProducerError("native nominal curve count is incomplete")
    samples = []
    strokes: list[float] = []
    forces: list[float] = []
    for index, row in enumerate(rows):
        if row["sample_index"] != str(index):
            raise ProducerError("native nominal curve sample indices are not contiguous")
        stroke, force = float(row["stroke_m"]), float(row["force_N"])
        contacts = int(row["contact_samples"])
        if not (math.isfinite(stroke) and math.isfinite(force) and stroke >= 0 and force >= 0 and contacts >= 0):
            raise ProducerError("native nominal curve contains invalid values")
        if strokes and stroke <= strokes[-1]:
            raise ProducerError("native nominal curve stroke is not strictly increasing")
        strokes.append(stroke); forces.append(force)
        samples.append({"sample_index": index, "phase": "shipping_compression_nominal",
                        "accepted": True, "stroke_m": stroke, "force_N": force,
                        "contact_samples": contacts})
    if forces.index(max(forces)) != peak_index:
        raise ProducerError("native capacity binding peak index differs from the first nominal curve maximum")
    return samples


def _find_prediction_path(prediction: dict[str, Any], flag: str) -> Path:
    argv = prediction["argv"]
    try:
        return Path(argv[argv.index(flag) + 1]).resolve()
    except (ValueError, IndexError) as exc:
        raise ProducerError(f"prediction argv is missing {flag}") from exc


def _verify_case(plan_path: Path, binding_path: Path, binding: dict[str, Any],
                 catalog_path: Path, data_dir: Path) -> None:
    plan = read_json(plan_path, "ensemble plan")
    if binding.get("schema") != SCHEMA or binding.get("plan_sha256") != sha256(plan_path):
        raise ProducerError("case binding does not match the pre-run plan bytes")
    candidate_id = binding.get("candidate", {}).get("candidate_id")
    condition_id = binding.get("condition", {}).get("condition_id")
    candidates = [x for x in plan.get("candidates", []) if x.get("candidate_id") == candidate_id]
    conditions = [x for x in plan.get("conditions", []) if x.get("condition_id") == condition_id]
    if len(candidates) != 1 or len(conditions) != 1 or candidates[0] != binding["candidate"] or conditions[0] != binding["condition"]:
        raise ProducerError("pre-run case binding differs from the plan candidate or condition")
    _check_candidate(binding["candidate"])
    if binding["condition"].get("dataset_id") != DATASET_ID or binding["condition"].get("native_protocol_id") != PROTOCOL_ID:
        raise ProducerError("case binding is not the supported BCT full-capacity condition")
    if binding.get("source_data_catalog_sha256") != plan.get("source_data_catalog_sha256"):
        raise ProducerError("case binding catalog identity differs from the plan")
    if binding.get("target_recipe_id") != plan.get("target_recipe_id"):
        raise ProducerError("case binding target recipe differs from the plan")
    if binding.get("target_grade_id") != plan.get("target_grade_id"):
        raise ProducerError("case binding target grade differs from the plan")
    _verify_published_condition(binding["condition"], catalog_path, data_dir,
                                str(binding.get("source_data_catalog_sha256", "")),
                                str(binding.get("target_grade_id", "")))


def finalize_run(run_dir: Path, plan_path: Path, binding_path: Path,
                 output_dir: Path, catalog_path: Path,
                 data_dir: Path) -> Path:
    run_dir, plan_path, binding_path = run_dir.resolve(), plan_path.resolve(), binding_path.resolve()
    if output_dir.exists():
        raise ProducerError("output directory must not already exist; raw evidence is never overwritten")
    prediction_path = run_dir / "prediction.json"
    raw_receipt_path = run_dir / "receipt.json"
    output_root = run_dir / "output"
    result_path = output_root / "box-native-result.json"
    manifest_path = output_root / "manifest.json"
    checkpoint_metadata_path = output_root / "shipping-rigid-accepted-checkpoint.json"
    if not all(p.is_file() for p in (prediction_path, raw_receipt_path, result_path,
                                     manifest_path, checkpoint_metadata_path)):
        raise ProducerError("native run lacks prediction, receipt, result, manifest, or paired checkpoint metadata")
    binding = read_json(binding_path, "pre-run BCT case binding")
    prediction = read_json(prediction_path, "sealed native prediction")
    execution_binding = _verify_prediction_inputs(
        prediction_path, prediction, plan_path, binding_path, binding)
    _verify_case(plan_path, binding_path, binding, catalog_path, data_dir)
    raw_receipt = read_json(raw_receipt_path, "original native receipt")
    if raw_receipt.get("exit_code") != 0 or raw_receipt.get("timed_out") is not False:
        raise ProducerError("native run did not complete successfully; prefixes and failed receipts cannot produce histories")

    result = read_json(result_path, "native box result")
    native_manifest = read_json(manifest_path, "native input manifest")
    calibration_validation = raw_receipt.get("calibration_parameter_validation")
    if (raw_receipt.get("source_inputs_unchanged") is not True or
            raw_receipt.get("calibration_case_binding_sha256") != sha256(binding_path) or
            raw_receipt.get("calibration_plan_sha256") != sha256(plan_path) or
            raw_receipt.get("calibration_candidate_id") != binding["candidate"]["candidate_id"] or
            raw_receipt.get("calibration_candidate_sha256") != binding["candidate"]["candidate_sha256"] or
            raw_receipt.get("calibration_condition_id") != binding["condition"]["condition_id"] or
            raw_receipt.get("native_protocol_id") != binding["condition"]["native_protocol_id"] or
            not isinstance(calibration_validation, dict) or
            calibration_validation.get("status") != "passed" or
            calibration_validation.get("manifest_sha256") != sha256(manifest_path)):
        raise ProducerError("raw harness receipt does not prove successful post-run parameter/protocol validation")
    if (calibration_validation.get("native_protocol_id") != binding["condition"]["native_protocol_id"] or
            calibration_validation.get("protocol_validation") not in
            {"pre_run_plan_bound", "executed_manifest_match"}):
        raise ProducerError("raw harness post-run protocol identity is inconsistent with the prepared condition")
    checks = calibration_validation.get("parameter_checks")
    expected_names = set(binding["candidate"].get("parameters", {}))
    if (not isinstance(checks, list) or len(checks) != len(expected_names) or
            {row.get("name") for row in checks if isinstance(row, dict)} != expected_names):
        raise ProducerError("raw harness receipt omits or duplicates candidate parameter checks")
    for row in checks:
        name = row["name"]
        spec = binding["candidate"]["parameters"][name]
        pointer = binding["parameter_bindings"][name]["pointer"]
        if (row.get("pointer") != pointer or row.get("matches") is not True or
                isinstance(row.get("observed"), bool) or
                not isinstance(row.get("observed"), (int, float)) or
                float(row["observed"]) != float(spec["value"]) or
                float(row.get("expected", float("nan"))) != float(spec["value"])):
            raise ProducerError(f"raw harness manifest check is inconsistent for candidate parameter {name!r}")
    capacity_binding = result.get("shipping_capacity_binding")
    if not isinstance(capacity_binding, dict) or capacity_binding.get("schema") != BINDING_SCHEMA:
        raise ProducerError("native result does not contain the full source-bound capacity binding")
    if native_manifest.get("preset") != capacity_binding.get("board_recipe_id"):
        raise ProducerError("native preset differs from the capacity binding recipe")
    _verify_parameter_bindings(binding, native_manifest)
    if (result.get("native_slow_compression_completed") is not True or
            result.get("native_half_speed_compression_completed") is not True or
            result.get("native_compression_rate_sensitivity_gate_passed") is not True or
            result.get("native_compression_numerical_capacity_qualification_passed") is not True or
            not math.isclose(float(result.get("compression_stroke_fraction_of_inside_height", float("nan"))), 0.70, rel_tol=0.0, abs_tol=1e-9)):
        raise ProducerError("native result lacks full-stroke, half-speed, or mesh/time numerical-capacity qualification")
    if (result.get("source_physics_fingerprint") != capacity_binding.get("source_physics_fingerprint") or
            result.get("device_program_fingerprint") != capacity_binding.get("device_program_fingerprint") or
            result.get("compiled_world_fingerprint") != capacity_binding.get("compiled_world_fingerprint") or
            native_manifest.get("compiled_world_fingerprint") != capacity_binding.get("compiled_world_fingerprint") or
            result.get("shipping_source_material_binding_sha256") != capacity_binding.get("source_binding_sha256") or
            result.get("assembled_checkpoint_sha256_for_capacity_binding") != capacity_binding.get("assembled_matter_checkpoint_sha256") or
            result.get("assembled_rigid_sidecar_sha256_for_capacity_binding") != capacity_binding.get("assembled_rigid_sidecar_sha256") or
            result.get("nominal_compression_curve_sha256") != capacity_binding.get("nominal_curve_sha256")):
        raise ProducerError("native result and capacity binding fingerprints/artifact identities disagree")

    curve_path = output_root / str(result.get("nominal_compression_curve_file", ""))
    curve_sha = capacity_binding.get("nominal_curve_sha256")
    if not curve_path.is_file() or sha256(curve_path) != curve_sha:
        raise ProducerError("nominal compression curve is missing or not the exact bound artifact")
    samples = _load_curve(curve_path,
                          int(capacity_binding.get("nominal_accepted_sample_count", -1)),
                          int(capacity_binding.get("nominal_peak_sample_index", -1)))
    if not math.isclose(samples[-1]["stroke_m"] / float(capacity_binding["inside_height_m"]), 0.70,
                        rel_tol=0.0, abs_tol=1e-9):
        raise ProducerError("bound nominal curve does not end at the complete 70-percent stroke")

    qualification_path = _find_prediction_path(prediction, "--shipping-capacity-qualification")
    if sha256(qualification_path) != capacity_binding.get("capacity_qualification_sha256"):
        raise ProducerError("capacity qualification input hash differs from the native binding")
    prop_values: dict[str, str] = {}
    for raw in qualification_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ProducerError("capacity qualification contains malformed properties")
        key, value = line.split("=", 1)
        if key in prop_values or not value:
            raise ProducerError("capacity qualification has duplicate or empty properties")
        prop_values[key] = value
    if (prop_values.get("status") != "qualified_for_numerical_capacity_only" or
            prop_values.get("source_binding_sha256") != capacity_binding.get("source_binding_sha256") or
            prop_values.get("assembled_matter_checkpoint_sha256") != capacity_binding.get("assembled_matter_checkpoint_sha256") or
            prop_values.get("assembled_rigid_sidecar_sha256") != capacity_binding.get("assembled_rigid_sidecar_sha256") or
            prop_values.get("nominal_curve_sha256") != curve_sha or
            prop_values.get("mesh_report_sha256") != capacity_binding.get("mesh_report_sha256") or
            prop_values.get("time_report_sha256") != capacity_binding.get("time_report_sha256")):
        raise ProducerError("capacity qualification properties do not match the native binding")
    mesh_report = qualification_path.parent / prop_values["mesh_report_file"]
    time_report = qualification_path.parent / prop_values["time_report_file"]
    branch_path = output_root / "shipping-compression.csv"
    if sha256(mesh_report) != capacity_binding["mesh_report_sha256"] or sha256(time_report) != capacity_binding["time_report_sha256"]:
        raise ProducerError("source-bound mesh/time report hashes do not match the capacity input")
    if not branch_path.is_file() or sha256(branch_path) != capacity_binding.get("compression_branch_history_sha256"):
        raise ProducerError("accepted compression branch ledger is missing or mismatched")

    attempted = result.get("native_steps_attempted")
    accepted = result.get("native_steps_accepted")
    if not isinstance(attempted, int) or not isinstance(accepted, int) or attempted <= 0 or attempted != accepted:
        raise ProducerError("native control-step ledger is incomplete or contains rejected endpoints")

    candidate, condition = binding["candidate"], binding["condition"]
    output_dir = output_dir.resolve()
    if plan_path.parent.resolve() not in output_dir.parents:
        raise ProducerError("new calibration artifacts must be placed inside the ensemble plan directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        # Preserve raw files separately; the raw run remains byte-for-byte untouched.
        raw_copy = temp_dir / "original-native-run"
        raw_copy.mkdir()
        for source in (prediction_path, raw_receipt_path, result_path, manifest_path,
                       checkpoint_metadata_path,
                       output_root / "shipping-compression.csv", curve_path,
                       output_root / "compiled.nmatterpack"):
            if source.is_file():
                shutil.copyfile(source, raw_copy / source.name)
        original_receipt_sha = sha256(raw_receipt_path)
        result_sha = sha256(result_path)

        executable_path = Path(prediction["argv"][0]).resolve()
        if not executable_path.is_file():
            raise ProducerError("executed native binary is unavailable for source-bound packaging")
        exe_sha = sha256(executable_path)
        input_map = prediction.get("inputs_sha256", {})
        if not any(Path(key).name == executable_path.name and digest == exe_sha
                   for key, digest in input_map.items()):
            raise ProducerError("native executable hash is absent from sealed prediction inputs")
        solver_source_manifest = {
            "schema": "numi.cardboard.solver-source-input-hashes.v1",
            "source_snapshot": prediction.get("source_snapshot"),
            "runtime_dir": prediction.get("runtime_dir"),
            "inputs_sha256": input_map,
            "source_binding_limitation": prediction.get("source_binding_limitation"),
        }
        solver_source_path = temp_dir / "solver-source-input-hashes.json"
        write_json(solver_source_path, solver_source_manifest)
        solver_source_sha = sha256(solver_source_path)
        exe_copy = temp_dir / executable_path.name
        _copy_verified(executable_path, exe_copy, exe_sha)
        compiled_src = output_root / "compiled.nmatterpack"
        compiled_copy = temp_dir / "compiled.nmatterpack"
        compiled_sha = _copy_verified(compiled_src, compiled_copy)
        plan_copy = temp_dir / "calibration-ensemble-plan.json"
        plan_sha = _copy_verified(plan_path, plan_copy, sha256(plan_path))
        binding_copy = temp_dir / "prepared-case-binding.json"
        binding_sha = _copy_verified(binding_path, binding_copy, sha256(binding_path))

        def artifact_copy(key: str, source: Path, expected: str | None = None) -> tuple[Path, str]:
            target = temp_dir / source.name
            digest = _copy_verified(source, target, expected)
            return target, digest

        qualification_copy, qualification_sha = artifact_copy(
            "shipping_capacity_qualification", qualification_path,
            capacity_binding["capacity_qualification_sha256"])
        mesh_copy, mesh_sha = artifact_copy(
            "shipping_capacity_mesh_report", mesh_report, capacity_binding["mesh_report_sha256"])
        time_copy, time_sha = artifact_copy(
            "shipping_capacity_time_report", time_report, capacity_binding["time_report_sha256"])
        curve_copy, copied_curve_sha = artifact_copy(
            "shipping_compression_nominal_curve", curve_path, curve_sha)
        branch_copy, branch_sha = artifact_copy(
            "shipping_compression_branch_history", branch_path,
            capacity_binding["compression_branch_history_sha256"])
        checkpoint_sha = capacity_binding["assembled_matter_checkpoint_sha256"]
        rigid_sha = capacity_binding["assembled_rigid_sidecar_sha256"]
        checkpoint_metadata_src, checkpoint_src, rigid_src = _resolve_checkpoint_pair(
            output_root, checkpoint_sha, rigid_sha)
        checkpoint_copy, checkpoint_sha = artifact_copy(
            "shipping_assembled_matter_checkpoint", checkpoint_src, checkpoint_sha)
        rigid_copy, rigid_sha = artifact_copy(
            "shipping_assembled_rigid_sidecar", rigid_src, rigid_sha)
        checkpoint_metadata_copy, checkpoint_metadata_sha = artifact_copy(
            "shipping_rigid_checkpoint_metadata", checkpoint_metadata_src)
        result_copy, copied_result_sha = artifact_copy("native_result", result_path, result_sha)

        input_manifest = {
            "schema": "numi.cardboard.native_input_manifest.v1",
            "candidate_id": candidate["candidate_id"],
            "candidate_sha256": candidate["candidate_sha256"],
            "condition_id": condition["condition_id"],
            "specimen_id": condition["specimen_id"],
            "native_protocol_id": condition["native_protocol_id"],
            "source_data_catalog_sha256": binding["source_data_catalog_sha256"],
            "source_dataset_sha256": condition["source_dataset_sha256"],
            "source_asset_sha256": condition["source_asset_sha256"],
            "target_recipe_id": binding["target_recipe_id"],
            "target_grade_id": binding["target_grade_id"],
            "parameters": candidate.get("parameters", {}),
            "parameter_bindings": binding["parameter_bindings"],
            "preset": native_manifest["preset"],
            "source_physics_fingerprint": capacity_binding["source_physics_fingerprint"],
            "device_program_fingerprint": capacity_binding["device_program_fingerprint"],
            "compiled_world_fingerprint": capacity_binding["compiled_world_fingerprint"],
            "compiled_matter_package_sha256": compiled_sha,
            "native_executable_sha256": exe_sha,
            "solver_source_sha256": solver_source_sha,
            "shipping_source_material_binding_sha256": capacity_binding["source_binding_sha256"],
            "assembled_checkpoint_sha256_for_capacity_binding": checkpoint_sha,
            "assembled_rigid_sidecar_sha256_for_capacity_binding": rigid_sha,
            "shipping_rigid_checkpoint_metadata_sha256": checkpoint_metadata_sha,
            "shipping_capacity_binding": capacity_binding,
            "original_prediction_sha256": sha256(prediction_path),
            "original_native_receipt_sha256": original_receipt_sha,
            "original_native_result_sha256": result_sha,
            "raw_manifest_sha256": sha256(manifest_path),
            "calibration_ensemble_plan_sha256": plan_sha,
            "prepared_case_binding_sha256": binding_sha,
            "pre_run_calibration_execution": execution_binding,
            "calibration_parameter_validation": calibration_validation,
        }
        manifest_path_new = temp_dir / "native_input_manifest.json"
        write_json(manifest_path_new, input_manifest)
        manifest_sha = sha256(manifest_path_new)

        history = {
            "schema": "numi.cardboard.native_forward_history.v1",
            "candidate_id": candidate["candidate_id"],
            "candidate_sha256": candidate["candidate_sha256"],
            "condition_id": condition["condition_id"],
            "specimen_id": condition["specimen_id"],
            "native_protocol_id": condition["native_protocol_id"],
            "source_data_catalog_sha256": binding["source_data_catalog_sha256"],
            "source_dataset_sha256": condition["source_dataset_sha256"],
            "source_asset_sha256": condition["source_asset_sha256"],
            "units": {"stroke_m": "m", "force_N": "N"},
            "samples": samples,
            "outcomes": {"native_capacity_status": "qualified_for_numerical_capacity_only"},
            "shipping_capacity_binding": capacity_binding,
            "calibration_case_binding_sha256": sha256(binding_path),
            "calibration_plan_sha256": sha256(plan_path),
            "calibration_ensemble_plan_sha256": plan_sha,
            "prepared_case_binding_sha256": binding_sha,
            "calibration_parameter_validation": calibration_validation,
            "pre_run_calibration_execution": execution_binding,
        }
        history_path = temp_dir / "history.json"
        write_json(history_path, history)
        history_sha = sha256(history_path)

        receipt = {
            "schema": "numi.cardboard.native_run_receipt.v1",
            "candidate_id": candidate["candidate_id"],
            "candidate_sha256": candidate["candidate_sha256"],
            "condition_id": condition["condition_id"],
            "specimen_id": condition["specimen_id"],
            "native_protocol_id": condition["native_protocol_id"],
            "terminal_status": "completed",
            "accepted_steps": accepted,
            "expected_steps": attempted,
            "physical_fem_steps": accepted,
            "backend": "numi_matter_native_fem",
            "original_receipt_sha256": original_receipt_sha,
            "original_native_result_sha256": result_sha,
            "shipping_capacity_binding": capacity_binding,
            "calibration_case_binding_sha256": sha256(binding_path),
            "calibration_plan_sha256": sha256(plan_path),
            "calibration_ensemble_plan_sha256": plan_sha,
            "prepared_case_binding_sha256": binding_sha,
            "calibration_parameter_validation": calibration_validation,
            "pre_run_calibration_execution": execution_binding,
        }
        receipt_path = temp_dir / "run_receipt.json"
        write_json(receipt_path, receipt)
        receipt_sha = sha256(receipt_path)

        artifacts = {
            "native_input_manifest": {"path": manifest_path_new.name, "sha256": manifest_sha},
            "calibration_ensemble_plan": {"path": plan_copy.name, "sha256": plan_sha},
            "prepared_case_binding": {"path": binding_copy.name, "sha256": binding_sha},
            "native_executable": {"path": exe_copy.name, "sha256": exe_sha},
            "solver_source": {"path": solver_source_path.name, "sha256": solver_source_sha},
            "run_receipt": {"path": receipt_path.name, "sha256": receipt_sha},
            "shipping_capacity_qualification": {"path": qualification_copy.name, "sha256": qualification_sha},
            "shipping_compression_nominal_curve": {"path": curve_copy.name, "sha256": copied_curve_sha},
            "shipping_capacity_mesh_report": {"path": mesh_copy.name, "sha256": mesh_sha},
            "shipping_capacity_time_report": {"path": time_copy.name, "sha256": time_sha},
            "shipping_compression_branch_history": {"path": branch_copy.name, "sha256": branch_sha},
            "shipping_assembled_matter_checkpoint": {"path": checkpoint_copy.name, "sha256": checkpoint_sha},
            "shipping_assembled_rigid_sidecar": {"path": rigid_copy.name, "sha256": rigid_sha},
            "shipping_rigid_checkpoint_metadata": {"path": checkpoint_metadata_copy.name, "sha256": checkpoint_metadata_sha},
            "compiled_matter_package": {"path": compiled_copy.name, "sha256": compiled_sha},
            "native_result": {"path": result_copy.name, "sha256": copied_result_sha},
        }
        certificate = {
            "schema": "numi.cardboard.native_execution_certificate.v1",
            "candidate_id": candidate["candidate_id"],
            "candidate_sha256": candidate["candidate_sha256"],
            "condition_id": condition["condition_id"],
            "specimen_id": condition["specimen_id"],
            "native_protocol_id": condition["native_protocol_id"],
            "source_data_catalog_sha256": binding["source_data_catalog_sha256"],
            "source_dataset_sha256": condition["source_dataset_sha256"],
            "source_asset_sha256": condition["source_asset_sha256"],
            "execution_class": "native_forward_prediction",
            "backend": "numi_matter_native_fem",
            "terminal_status": "completed",
            "protocol_complete": True,
            "prefix_only": False,
            "expected_steps": attempted,
            "accepted_steps": accepted,
            "physical_fem_steps": accepted,
            "sample_count": len(samples),
            "history_sha256": history_sha,
            "artifact_hashes": artifacts,
            "shipping_capacity_binding": capacity_binding,
            "calibration_case_binding_sha256": sha256(binding_path),
            "calibration_plan_sha256": sha256(plan_path),
            "calibration_ensemble_plan_sha256": plan_sha,
            "prepared_case_binding_sha256": binding_sha,
            "calibration_parameter_validation": calibration_validation,
            "pre_run_calibration_execution": execution_binding,
            "original_native_result_sha256": result_sha,
            "original_native_receipt_sha256": original_receipt_sha,
        }
        certificate["certificate_sha256"] = canonical_sha256(certificate)
        certificate_path = temp_dir / "certificate.json"
        write_json(certificate_path, certificate)
        cert_sha = sha256(certificate_path)
        record = {
            "candidate_id": candidate["candidate_id"],
            "candidate_sha256": candidate["candidate_sha256"],
            "condition_id": condition["condition_id"],
            "specimen_id": condition["specimen_id"],
            "native_protocol_id": condition["native_protocol_id"],
            "history_path": str((output_dir / "history.json").relative_to(plan_path.parent)),
            "history_sha256": history_sha,
            "certificate_path": str((output_dir / "certificate.json").relative_to(plan_path.parent)),
            "certificate_file_sha256": cert_sha,
        }
        write_json(temp_dir / "plan-record.json", record)
        os.replace(temp_dir, output_dir)
        return output_dir / "plan-record.json"
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="seal a candidate/condition binding before the native run")
    prepare.add_argument("plan", type=Path)
    prepare.add_argument("candidate_id")
    prepare.add_argument("condition_id")
    prepare.add_argument("--parameter-bindings", type=Path, required=True,
                         help="map each candidate parameter to a raw-manifest pointer and an argv_flag; use input_json_pointer for parameters supplied through a JSON file")
    prepare.add_argument("--output", type=Path, required=True)
    finalize = sub.add_parser("finalize", help="create new ensemble artifacts from an unchanged native run")
    finalize.add_argument("run_dir", type=Path)
    finalize.add_argument("plan", type=Path)
    finalize.add_argument("case_binding", type=Path)
    finalize.add_argument("--source-data-catalog", type=Path, required=True)
    finalize.add_argument("--source-data-dir", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            parameter_bindings = read_json(args.parameter_bindings, "parameter binding map")
            if not isinstance(parameter_bindings, dict):
                raise ProducerError("parameter binding map must be a JSON object")
            value = prepare_case_binding(args.plan, args.candidate_id,
                                        args.condition_id, parameter_bindings,
                                        args.output)
            print(json.dumps({"status": "pre_run_case_binding_written",
                              "schema": value["schema"],
                              "sha256": sha256(args.output),
                              "plan_sha256": value["plan_sha256"]}, sort_keys=True))
        else:
            record = finalize_run(args.run_dir, args.plan, args.case_binding,
                                  args.output, args.source_data_catalog,
                                  args.source_data_dir)
            print(json.dumps({"status": "new_calibration_artifacts_written",
                              "plan_record": str(record),
                              "plan_record_sha256": sha256(record)}, sort_keys=True))
        return 0
    except ProducerError as exc:
        print(f"BCT producer rejected input: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"BCT producer failed closed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
