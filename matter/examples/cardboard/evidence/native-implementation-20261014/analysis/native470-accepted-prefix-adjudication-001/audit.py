#!/usr/bin/env python3
"""Strict read-only adjudicator for the accepted control 0..19 prefix in run 470.

This tool must only be executed after the source-bound runner has written a
terminal receipt. It requires an explicit terminal acknowledgement, rejects a
live native process, and never modifies either run directory. Its result covers
the accepted prefix and checkpoint integrity only; it does not claim resume
parity, a complete trajectory, or physical validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path("/Users/home/cardboard-implementation-20261014")
BASELINE = ROOT / "solver/final-newton-fd-capture-469"
TARGET = ROOT / "solver/accepted-checkpoint-prefix-470"
SOURCE = ROOT / "snapshots/source-diagnostic-and-resume-167-final"
TOOLS = SOURCE / "matter/tools"
SOURCE_HASHES = {
    "matter/tools/cardboard_solver_environment.py":
        "42c0e016b00c765950b8371e543504d2676cc9dd8fc8a559f2c9540b212b586b",
    "matter/tools/cardboard_source_bound_run.py":
        "6ccd93a467860df58d8552b6f14c7d1ea77be2a896c06a77bfd9ad25392ceb9e",
    "matter/tools/cardboard_coupon_resume.hpp":
        "bdb41b93688e924b950aebc68aea19e5cc0988f618430b3e0e38634d37e1d282",
    "matter/tools/cardboard_probe.mm":
        "448f8b4f59a5b287d7cb81e36e6fa077687b407a43e63a45d173393340e16bd1",
}
CHECKPOINT_CONTROL = 19
NEXT_CONTROL = 20
EXPECTED_CAPTURE_KEY = "NM_FEM_REGIONAL_CONSUMER_CAPTURE_PATH"
EXPECTED_SELECTOR_KEY = "NM_FEM_REGIONAL_CONSUMER_CAPTURE_SELECTOR"
EXPECTED_SELECTOR = "20:0:63:1:0:0"
CAPTURE_KEYS = {
    "NM_FEM_OPERATOR_FD_CAPTURE_PATH",
    "NM_FEM_OPERATOR_FD_SELECTOR",
    EXPECTED_CAPTURE_KEY,
    EXPECTED_SELECTOR_KEY,
}
RESULT_IDENTITY_FIELDS = (
    "compiled_world_fingerprint", "nodes", "tetrahedra",
    "material_map_sha256", "authored_region_map_sha256",
    "frame_map_sha256", "damage_overlay_mapping_sha256",
)
COMMON_PREFIX_FILES = (
    "compiled.nmatterpack", "initial.obj", "mesh.json",
)
ENVS = {0: "bent", 1: "held_reference"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class AuditError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    st = path.lstat()
    if path.is_symlink() or not path.is_file() or st.st_nlink != 1:
        raise AuditError(f"not a nonsymlink, single-link regular file: {path}")
    if max_bytes is not None and st.st_size > max_bytes:
        raise AuditError(f"file exceeds declared cap: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_bytes(raw: bytes, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise AuditError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def bad_constant(value: str) -> None:
        raise AuditError(f"{label}: nonstandard JSON constant {value}")

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=bad_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{label}: invalid strict JSON: {error}") from error


def strict_json_file(path: Path, label: str, cap: int = 16 * 1024 * 1024) -> tuple[Any, bytes]:
    st = path.lstat()
    if path.is_symlink() or not path.is_file() or st.st_nlink != 1:
        raise AuditError(f"{label}: expected a nonsymlink, single-link regular file")
    if st.st_size > cap:
        raise AuditError(f"{label}: exceeds {cap} byte cap")
    raw = path.read_bytes()
    if len(raw) > cap:
        raise AuditError(f"{label}: exceeds {cap} byte cap")
    return strict_json_bytes(raw, label), raw


def load_source_environment_module():
    source_path = TOOLS / "cardboard_solver_environment.py"
    actual = sha256_file(source_path)
    expected = SOURCE_HASHES["matter/tools/cardboard_solver_environment.py"]
    if actual != expected:
        raise AuditError(
            f"source-bound validator changed: expected {expected}, got {actual}")
    spec = importlib.util.spec_from_file_location(
        "source167_cardboard_solver_environment", source_path)
    if spec is None or spec.loader is None:
        raise AuditError("cannot load source167 environment validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_source_bindings() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in SOURCE_HASHES.items():
        path = SOURCE / relative
        actual = sha256_file(path)
        if actual != expected:
            raise AuditError(f"source binding mismatch for {relative}")
        observed[relative] = actual
    return observed


def assert_no_live_probe(case: Path, prediction: dict[str, Any]) -> None:
    argv = prediction.get("argv")
    if not isinstance(argv, list) or not argv or not isinstance(argv[0], str):
        raise AuditError("prediction lacks the native executable argv")
    executable = str(Path(argv[0]).resolve())
    try:
        listing = subprocess.run(
            ["/bin/ps", "-axo", "pid=,stat=,command="],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise AuditError(f"could not verify native process is terminal: {error}") from error
    active = []
    for line in listing.stdout.splitlines():
        if executable in line or str(case) in line:
            # Exclude this audit command itself; its command line contains the
            # case path in the Python source/arguments only after invocation.
            if "native470-accepted-prefix-adjudication-001/audit.py" not in line:
                active.append(line.strip())
    if active:
        raise AuditError("native run still appears active; refusing to read its output: " +
                         " | ".join(active[:4]))


def parse_key_values(raw: bytes, label: str) -> dict[str, str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AuditError(f"{label}: invalid UTF-8") from error
    values: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line or "=" not in line:
            raise AuditError(f"{label}:{line_number}: malformed key/value line")
        key, value = line.split("=", 1)
        if not key or key in values or "\x00" in value:
            raise AuditError(f"{label}:{line_number}: empty or duplicate key")
        values[key] = value
    return values


def parse_prefix_manifest(raw: bytes) -> dict[str, Any]:
    values: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if line.startswith("file="):
            pieces = line[5:].split("|")
            if len(pieces) != 4:
                raise AuditError(f"prefix manifest line {line_number}: malformed file record")
            relative, byte_count, digest, observation = pieces
            path = PurePosixPath(relative)
            if (path.is_absolute() or not path.parts or ".." in path.parts or
                    "\\" in relative or "\x00" in relative):
                raise AuditError(f"unsafe prefix path {relative!r}")
            if not byte_count.isdecimal() or observation not in {"0", "1"}:
                raise AuditError(f"invalid prefix entry metadata for {relative}")
            if not HEX64.fullmatch(digest):
                raise AuditError(f"invalid SHA-256 in prefix entry for {relative}")
            entries.append({"path": relative, "bytes": int(byte_count),
                            "sha256": digest, "observations_prefix": observation == "1"})
            continue
        if "=" not in line:
            raise AuditError(f"prefix manifest line {line_number}: malformed header")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise AuditError(f"prefix manifest line {line_number}: duplicate header {key!r}")
        values[key] = value
    expected_headers = {
        "schema", "source_run_directory", "accepted_control_index",
        "observations_prefix_bytes", "observations_prefix_sha256",
    }
    if set(values) != expected_headers:
        raise AuditError("prefix manifest header inventory differs from source167 schema")
    if values["schema"] != "numi.cardboard.accepted-coupon-prefix.v1":
        raise AuditError("unexpected checkpoint prefix schema")
    if not values["accepted_control_index"].isdecimal():
        raise AuditError("invalid accepted control in prefix manifest")
    if not values["observations_prefix_bytes"].isdecimal():
        raise AuditError("invalid observations prefix byte count")
    if not HEX64.fullmatch(values["observations_prefix_sha256"]):
        raise AuditError("invalid observations prefix digest")
    paths = [entry["path"] for entry in entries]
    if len(paths) != len(set(paths)) or paths != sorted(paths):
        raise AuditError("prefix paths are duplicated or not canonically sorted")
    total_bytes = sum(entry["bytes"] for entry in entries)
    if total_bytes > 2 * 1024**3 or not entries or len(entries) > 8192:
        raise AuditError("prefix inventory exceeds source167 count/byte bounds")
    return {
        **values,
        "accepted_control_index": int(values["accepted_control_index"]),
        "observations_prefix_bytes": int(values["observations_prefix_bytes"]),
        "entries": entries,
        "total_bytes": total_bytes,
    }


def expected_accepted_names() -> set[str]:
    # The manifest belongs in the checkpoint inventory but is compared
    # semantically below because its absolute material-source paths differ.
    names = set(COMMON_PREFIX_FILES) | {"manifest.json", "observations.csv"}
    for step in range(1, NEXT_CONTROL + 1):
        for environment in ENVS.values():
            names.add(f"accepted_step_{step:06d}_{environment}.obj")
    return names


def csv_prefix_lines(path: Path) -> tuple[bytes, list[dict[str, str]], list[bytes]]:
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if not lines or not lines[0].endswith(b"\n"):
        raise AuditError(f"{path}: CSV header is missing or not newline terminated")
    try:
        decoded = raw.decode("utf-8")
        records = list(csv.DictReader(decoded.splitlines()))
    except (UnicodeDecodeError, csv.Error) as error:
        raise AuditError(f"{path}: invalid observations CSV: {error}") from error
    selected_records = []
    selected_lines = [lines[0]]
    for record, line in zip(records, lines[1:]):
        try:
            step = int(record["step"])
        except (KeyError, TypeError, ValueError) as error:
            raise AuditError("observations CSV has an invalid step field") from error
        if step <= CHECKPOINT_CONTROL:
            selected_records.append(record)
            selected_lines.append(line)
    return b"".join(selected_lines), selected_records, lines


def verify_csv_accepted_prefix(baseline_output: Path, target_output: Path) -> dict[str, Any]:
    baseline_prefix, base_rows, _ = csv_prefix_lines(baseline_output / "observations.csv")
    target_bytes = (target_output / "observations.csv").read_bytes()
    if target_bytes != baseline_prefix:
        raise AuditError("470 observations.csv is not byte-identical to 469 controls 0..19")
    expected_steps = set(range(NEXT_CONTROL))
    seen: dict[int, set[int]] = {}
    for row in base_rows:
        step = int(row["step"])
        environment = int(row["environment"])
        seen.setdefault(step, set()).add(environment)
        if environment not in ENVS or row["arm"] != ENVS[environment]:
            raise AuditError(f"unexpected environment/arm at accepted prefix step {step}")
        if row["status_code"] != "0" or row["step_accepted"] != "1":
            raise AuditError(f"baseline control {step} contains a nonaccepted row")
    if set(seen) != expected_steps or any(seen[step] != {0, 1} for step in expected_steps):
        raise AuditError("accepted prefix does not have exactly both environments at controls 0..19")
    if len(base_rows) != 2 * NEXT_CONTROL:
        raise AuditError("accepted prefix row count is not exactly 40")
    return {
        "exact_bytes_match": True,
        "controls": [0, CHECKPOINT_CONTROL],
        "accepted_environment_rows": len(base_rows),
        "sha256": sha256_bytes(target_bytes),
        "bytes": len(target_bytes),
    }


def validate_checkpoint(target_output: Path, target_case: Path,
                        target_prediction: dict[str, Any],
                        target_prediction_raw: bytes,
                        target_environment_record: dict[str, Any],
                        target_environment_raw: bytes,
                        target_result: dict[str, Any]) -> dict[str, Any]:
    output_root = target_output.resolve(strict=True)
    if target_output.is_symlink() or not target_output.is_dir():
        raise AuditError("target output root is missing or aliased")
    report_path = target_output / "coupon-checkpoint-prefix.json"
    prefix_path_data, prefix_report_raw = strict_json_file(report_path, "checkpoint prefix report")
    expected_prefix_report_keys = {
        "schema", "status", "accepted_control_index", "next_control_index",
        "checkpoint_file", "checkpoint_sha256", "metadata_file",
        "metadata_sha256", "full_protocol_complete",
        "trajectory_parity_established", "physical_validation",
    }
    if set(prefix_path_data) != expected_prefix_report_keys:
        raise AuditError("checkpoint prefix report keys differ from source167 schema")
    if (prefix_path_data.get("schema") != "numi.cardboard.coupon-checkpoint-prefix.v1"
            or prefix_path_data.get("status") != "accepted_checkpoint_prefix_saved"
            or prefix_path_data.get("accepted_control_index") != CHECKPOINT_CONTROL
            or prefix_path_data.get("next_control_index") != NEXT_CONTROL
            or any(prefix_path_data.get(key) is not False for key in
                   ("full_protocol_complete", "trajectory_parity_established", "physical_validation"))):
        raise AuditError("checkpoint prefix report does not prove the expected incomplete boundary")
    if (prefix_path_data["checkpoint_file"] != "checkpoints/coupon-control-000019.nmcb"
            or prefix_path_data["metadata_file"] != "checkpoints/coupon-control-000019.resume"):
        raise AuditError("checkpoint report names a noncanonical control-19 artifact")

    def output_file(relative: str, cap: int) -> Path:
        rel = PurePosixPath(relative)
        if rel.is_absolute() or not rel.parts or ".." in rel.parts or "\\" in relative:
            raise AuditError(f"unsafe checkpoint reference {relative!r}")
        candidate = target_output.joinpath(*rel.parts)
        if candidate.is_symlink() or not candidate.is_file():
            raise AuditError(f"checkpoint reference is missing or aliased: {relative}")
        if not candidate.resolve(strict=True).is_relative_to(output_root):
            raise AuditError(f"checkpoint reference escaped output root: {relative}")
        if candidate.stat().st_size > cap:
            raise AuditError(f"checkpoint reference exceeds cap: {relative}")
        return candidate

    checkpoint = output_file(prefix_path_data["checkpoint_file"], 2 * 1024**3)
    metadata_file = output_file(prefix_path_data["metadata_file"], 64 * 1024)
    checkpoint_hash = sha256_file(checkpoint, max_bytes=2 * 1024**3)
    metadata_hash = sha256_file(metadata_file, max_bytes=64 * 1024)
    if (checkpoint_hash != prefix_path_data.get("checkpoint_sha256")
            or metadata_hash != prefix_path_data.get("metadata_sha256")):
        raise AuditError("checkpoint/report hash binding mismatch")

    resume_raw = metadata_file.read_bytes()
    resume = parse_key_values(resume_raw, "resume metadata")
    required_resume = {
        "schema", "checkpoint_sha256", "checkpoint_bytes",
        "source_physics_fingerprint", "device_program_fingerprint",
        "protocol_manifest_sha256", "prefix_manifest_sha256",
        "runtime_environment_sha256", "runtime_output_path_map_hex",
        "runtime_output_path_map_sha256", "source_runner_prediction_sha256",
        "source_runner_contract_sha256", "source_runner_case_directory",
        "source_runner_environment_record_sha256",
        "source_runner_non_output_settings_sha256",
        "prefix_manifest_file", "source_run_directory", "source_run_id",
        "environment_count", "node_count", "tetrahedron_count",
        "material_state_stride", "accepted_control_index",
        "next_control_index", "physics_substep", "source_prefix_control_count",
        "timestep_f32_bits", "scheduled_elapsed_seconds_f64_bits",
        "phase", "next_phase", "next_target_angle_f64_bits",
        "both_environments_accepted",
    }
    if set(resume) != required_resume:
        raise AuditError("resume metadata keys differ from the source167 v2 schema")
    if resume["schema"] != "numi.cardboard.accepted-coupon-resume.v2":
        raise AuditError("resume metadata is not source167 v2")
    if resume["both_environments_accepted"] != "true":
        raise AuditError("checkpoint does not certify both environments accepted")
    if (resume["accepted_control_index"] != str(CHECKPOINT_CONTROL)
            or resume["next_control_index"] != str(NEXT_CONTROL)
            or resume["source_prefix_control_count"] != str(NEXT_CONTROL)):
        raise AuditError("resume metadata cursor differs from control-19 boundary")
    if resume["checkpoint_sha256"] != checkpoint_hash:
        raise AuditError("resume metadata checkpoint SHA differs from checkpoint bytes")
    if resume["checkpoint_bytes"] != str(checkpoint.stat().st_size):
        raise AuditError("resume metadata checkpoint byte count differs")
    if resume["protocol_manifest_sha256"] != sha256_file(target_output / "manifest.json"):
        raise AuditError("resume protocol manifest digest does not match output manifest")
    for digest_key in ("runtime_environment_sha256", "runtime_output_path_map_sha256"):
        if not HEX64.fullmatch(resume[digest_key]):
            raise AuditError(f"resume digest is malformed: {digest_key}")
    try:
        output_path_map = bytes.fromhex(resume["runtime_output_path_map_hex"])
    except ValueError as error:
        raise AuditError("resume output-path map hex is malformed") from error
    if sha256_bytes(output_path_map) != resume["runtime_output_path_map_sha256"]:
        raise AuditError("resume output-path map digest does not match its bytes")
    numeric_fields = (
        "source_physics_fingerprint", "device_program_fingerprint",
        "environment_count", "node_count", "tetrahedron_count",
        "material_state_stride", "physics_substep", "timestep_f32_bits",
        "scheduled_elapsed_seconds_f64_bits", "next_target_angle_f64_bits",
    )
    for field in numeric_fields:
        if not resume[field].isdecimal():
            raise AuditError(f"resume numeric field is malformed: {field}")
    if (int(resume["environment_count"]) != 2
            or int(resume["node_count"]) != 882
            or int(resume["tetrahedron_count"]) != 1920):
        raise AuditError("checkpoint dimensions differ from the 469 accepted-prefix fixture")
    if resume["source_runner_case_directory"] != str(target_case.resolve()):
        raise AuditError("resume metadata is bound to a different source-runner case")
    if resume["source_runner_prediction_sha256"] != sha256_bytes(target_prediction_raw):
        raise AuditError("resume metadata prediction binding mismatch")
    if resume["source_runner_environment_record_sha256"] != sha256_bytes(target_environment_raw):
        raise AuditError("resume metadata environment record binding mismatch")
    contract = target_prediction.get("effective_native_environment_contract")
    if not isinstance(contract, dict):
        raise AuditError("prediction lacks its diagnostic environment contract")
    contract_sha = target_prediction.get("effective_native_environment_contract_sha256")
    if resume["source_runner_contract_sha256"] != contract_sha:
        raise AuditError("resume metadata diagnostic contract binding mismatch")
    if resume["prefix_manifest_file"] != Path(prefix_path_data["checkpoint_file"]).with_suffix(".prefix").name:
        raise AuditError("resume metadata names an unexpected prefix-manifest file")
    prefix_file = output_file(str(Path(prefix_path_data["checkpoint_file"]).parent /
                                  resume["prefix_manifest_file"]), 2 * 1024**3)
    prefix_raw = prefix_file.read_bytes()
    prefix_hash = sha256_bytes(prefix_raw)
    if prefix_hash != resume["prefix_manifest_sha256"]:
        raise AuditError("prefix manifest digest differs from resume metadata")

    try:
        prefix = parse_prefix_manifest(prefix_raw)
    except UnicodeDecodeError as error:
        raise AuditError("prefix manifest is not valid UTF-8") from error
    if (prefix["accepted_control_index"] != CHECKPOINT_CONTROL
            or Path(prefix["source_run_directory"]).resolve() != target_output.resolve()):
        raise AuditError("prefix manifest source directory or cursor mismatch")
    expected_names = expected_accepted_names()
    entries = {entry["path"]: entry for entry in prefix["entries"]}
    if set(entries) != expected_names:
        missing = sorted(expected_names - set(entries))
        extra = sorted(set(entries) - expected_names)
        raise AuditError(f"checkpoint inventory mismatch; missing={missing[:6]} extra={extra[:6]}")
    total_bytes = 0
    for relative, entry in entries.items():
        path = output_file(relative, 2 * 1024**3)
        actual_size = path.stat().st_size
        actual_hash = sha256_file(path, max_bytes=2 * 1024**3)
        if actual_size != entry["bytes"] or actual_hash != entry["sha256"]:
            raise AuditError(f"checkpoint prefix entry does not match bytes: {relative}")
        total_bytes += actual_size
    observation = entries["observations.csv"]
    observations_path = target_output / "observations.csv"
    if (not observation["observations_prefix"]
            or observation["bytes"] != prefix["observations_prefix_bytes"]
            or observation["sha256"] != prefix["observations_prefix_sha256"]
            or observations_path.stat().st_size != prefix["observations_prefix_bytes"]):
        raise AuditError("observations prefix binding is inconsistent")
    if prefix_path_data.get("source_physics_fingerprint") is not None:
        raise AuditError("unknown field appeared in source167 checkpoint prefix report")

    protocol_hash = sha256_file(target_output / "manifest.json")
    computed_run_id = sha256_bytes(
        (protocol_hash + checkpoint_hash + prefix["source_run_directory"]).encode("utf-8"))
    if resume["source_run_id"] != computed_run_id:
        raise AuditError("checkpoint source_run_id does not match source167 derivation")

    saved = parse_key_values(
        (target_output / "coupon-checkpoint-saved.txt").read_bytes(),
        "checkpoint saved marker")
    if (saved.get("schema") != "numi.cardboard.accepted-coupon-checkpoint-reference.v1"
            or saved.get("checkpoint") != prefix_path_data["checkpoint_file"]
            or saved.get("control_index") != str(CHECKPOINT_CONTROL)
            or saved.get("next_control_index") != str(NEXT_CONTROL)
            or saved.get("continuation_parity_established") != "false"
            or set(saved) != {"schema", "checkpoint", "control_index",
                              "next_control_index", "continuation_parity_established"}):
        raise AuditError("checkpoint saved marker differs from the prefix")

    return {
        "accepted_control_index": CHECKPOINT_CONTROL,
        "next_control_index": NEXT_CONTROL,
        "checkpoint": {
            "path": str(checkpoint.relative_to(target_output)),
            "bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_hash,
        },
        "resume_metadata": {
            "path": str(metadata_file.relative_to(target_output)),
            "bytes": len(resume_raw),
            "sha256": metadata_hash,
            "source_physics_fingerprint": resume["source_physics_fingerprint"],
            "device_program_fingerprint": resume["device_program_fingerprint"],
            "protocol_manifest_sha256": resume["protocol_manifest_sha256"],
            "runtime_environment_sha256": resume["runtime_environment_sha256"],
            "runtime_output_path_map_sha256": resume["runtime_output_path_map_sha256"],
            "source_run_id": resume["source_run_id"],
            "environment_count": resume["environment_count"],
            "node_count": resume["node_count"],
            "tetrahedron_count": resume["tetrahedron_count"],
            "material_state_stride": resume["material_state_stride"],
            "physics_substep": resume["physics_substep"],
            "phase": resume["phase"],
            "next_phase": resume["next_phase"],
        },
        "prefix_manifest": {
            "path": str(prefix_file.relative_to(target_output)),
            "bytes": len(prefix_raw),
            "sha256": prefix_hash,
            "entry_count": len(entries),
            "total_bytes": total_bytes,
            "observations_prefix_bytes": prefix["observations_prefix_bytes"],
            "observations_prefix_sha256": prefix["observations_prefix_sha256"],
            "all_entries_verified": True,
        },
        "prefix_inventory_verified": True,
    }


def compare_common_outputs(base: Path, target: Path,
                           target_result: dict[str, Any],
                           base_result: dict[str, Any]) -> dict[str, Any]:
    if target_result.get("schema") != "numi.cardboard.probe-result.v1":
        raise AuditError("target result schema mismatch")
    if (target_result.get("status") != "accepted_checkpoint_prefix_saved"
            or target_result.get("accepted_steps") != NEXT_CONTROL
            or target_result.get("physical_validation") is not False):
        raise AuditError("target result does not describe the incomplete accepted prefix")
    if base_result.get("schema") != "numi.cardboard.probe-result.v1":
        raise AuditError("baseline result schema mismatch")
    for key in RESULT_IDENTITY_FIELDS:
        if target_result.get(key) != base_result.get(key):
            raise AuditError(f"result identity field changed: {key}")
    common: dict[str, Any] = {}
    for relative in COMMON_PREFIX_FILES:
        left = base / relative
        right = target / relative
        left_hash = sha256_file(left)
        right_hash = sha256_file(right)
        if left_hash != right_hash:
            raise AuditError(f"common production artifact changed: {relative}")
        common[relative] = {"bytes": left.stat().st_size, "sha256": left_hash,
                            "exact_bytes_match": True}
    left_manifest, _ = strict_json_file(base / "manifest.json", "469 output manifest")
    right_manifest, _ = strict_json_file(target / "manifest.json", "470 output manifest")
    def normalize_manifest(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {k: normalize_manifest(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [normalize_manifest(item, key) for item in value]
        if isinstance(value, str) and key.endswith("_path") and "/matter/materials/" in value:
            return "<source>/matter/materials/" + Path(value).name
        return value
    if normalize_manifest(left_manifest) != normalize_manifest(right_manifest):
        raise AuditError("production output manifest differs beyond source-root material paths")
    common["manifest.json"] = {
        "bytes_469": (base / "manifest.json").stat().st_size,
        "bytes_470": (target / "manifest.json").stat().st_size,
        "sha256_469": sha256_file(base / "manifest.json"),
        "sha256_470": sha256_file(target / "manifest.json"),
        "semantic_match_after_material_source_path_normalization": True,
    }
    for key in ("liner_sha256", "medium_sha256", "glue_sha256"):
        if left_manifest["materials"].get(key) != right_manifest["materials"].get(key):
            raise AuditError(f"production material digest changed: {key}")
    return {
        "result_identity_fields_exact": list(RESULT_IDENTITY_FIELDS),
        "common_artifacts": common,
        "production_manifest_equal": True,
        "target_result_status": target_result["status"],
        "target_accepted_steps": target_result["accepted_steps"],
        "physical_validation": False,
    }


def compare_predictions(base_case: Path, target_case: Path,
                        base_prediction: dict[str, Any],
                        target_prediction: dict[str, Any]) -> dict[str, Any]:
    if target_prediction.get("run_mode") != "diagnostic":
        raise AuditError("470 is not explicitly diagnostic-only")
    if target_prediction.get("schema") != "numi.cardboard.native-diagnostic-prediction.v1":
        raise AuditError("470 prediction schema is not diagnostic")
    if base_prediction.get("schema") != "numi.cardboard.native-prediction.v1":
        raise AuditError("469 baseline prediction schema mismatch")
    if target_prediction.get("source_snapshot") != str(SOURCE):
        raise AuditError("470 is not bound to source167 snapshot")
    if target_prediction.get("executed_build_verified") is not True:
        raise AuditError("470 prediction does not bind a successful source build receipt")
    if not target_prediction.get("build_provenance"):
        raise AuditError("470 prediction does not name build provenance")
    if not isinstance(target_prediction.get("inputs_sha256"), dict):
        raise AuditError("470 prediction lacks retained input hashes")

    def normalize_argv(prediction: dict[str, Any], *, target: bool) -> list[str]:
        argv = prediction.get("argv")
        if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
            raise AuditError("prediction argv is malformed")
        result = []
        i = 0
        checkpoint_seen = False
        while i < len(argv):
            value = argv[i]
            if value in {"--output", "--metallib", "--liner-material",
                         "--medium-material"}:
                if i + 1 >= len(argv):
                    raise AuditError(f"argv value missing after {value}")
                replacement = {
                    "--output": "<OUTPUT>",
                    "--metallib": "<METALLIB>",
                    "--liner-material": "<LINER_MATERIAL>",
                    "--medium-material": "<MEDIUM_MATERIAL>",
                }[value]
                result.extend([value, replacement])
                i += 2
                continue
            if value == "--coupon-checkpoint-after-control":
                if not target or checkpoint_seen or i + 1 >= len(argv) or argv[i + 1] != "19":
                    raise AuditError("unexpected checkpoint CLI argument")
                checkpoint_seen = True
                i += 2
                continue
            if i == 0:
                result.append("<EXECUTABLE>")
            else:
                result.append(value)
            i += 1
        if target and not checkpoint_seen:
            raise AuditError("470 argv lacks the explicit control-19 checkpoint request")
        if not target and checkpoint_seen:
            raise AuditError("469 baseline unexpectedly requested a checkpoint")
        return result

    normalized_base = normalize_argv(base_prediction, target=False)
    normalized_target = normalize_argv(target_prediction, target=True)
    if normalized_base != normalized_target:
        raise AuditError("native physical argv differs beyond executable/output/metallib/checkpoint")

    base_env = base_prediction.get("environment")
    target_env = target_prediction.get("environment")
    if not isinstance(base_env, dict) or not isinstance(target_env, dict):
        raise AuditError("prediction environment is malformed")
    if {k: v for k, v in base_env.items() if k not in CAPTURE_KEYS} != \
            {k: v for k, v in target_env.items() if k not in CAPTURE_KEYS}:
        raise AuditError("non-capture native environment overrides differ")
    expected_base_captures = {
        "NM_FEM_OPERATOR_FD_SELECTOR": "20:63",
    }
    if any(base_env.get(k) != v for k, v in expected_base_captures.items()):
        raise AuditError("469 baseline capture controls differ from its sealed baseline")
    baseline_capture_path = base_env.get("NM_FEM_OPERATOR_FD_CAPTURE_PATH")
    if (not isinstance(baseline_capture_path, str)
            or not baseline_capture_path.startswith(str(base_case / "diagnostics/"))):
        raise AuditError("469 baseline FD path differs from its sealed case")
    if EXPECTED_CAPTURE_KEY not in target_env or EXPECTED_SELECTOR_KEY not in target_env:
        raise AuditError("470 lacks the selected regional-consumer diagnostic")
    if target_env[EXPECTED_SELECTOR_KEY] != EXPECTED_SELECTOR:
        raise AuditError("470 regional capture selector differs from control 20")
    if ("NM_FEM_OPERATOR_FD_SELECTOR" in target_env
            or "NM_FEM_OPERATOR_FD_CAPTURE_PATH" in target_env):
        raise AuditError("470 unexpectedly retains the baseline operator-FD selector")
    for material in ("hajali2009_liner_hill_bilinear_fit.nmatter",
                     "hajali2009_medium_hill_bilinear_fit.nmatter"):
        key = f"source/matter/materials/{material}"
        base_hash = base_prediction.get("inputs_sha256", {}).get(key)
        target_hash = target_prediction.get("inputs_sha256", {}).get(key)
        if not isinstance(base_hash, str) or base_hash != target_hash:
            raise AuditError(f"source-bound material input changed: {material}")
    return {
        "physical_argv_equal_after_declared_path_and_checkpoint_normalization": True,
        "non_capture_environment_equal": True,
        "diagnostic_only_difference": {
            "469": {
                "selector": base_env["NM_FEM_OPERATOR_FD_SELECTOR"],
                "path": base_env["NM_FEM_OPERATOR_FD_CAPTURE_PATH"],
            },
            "470": {
                "selector": target_env[EXPECTED_SELECTOR_KEY],
                "path": target_env[EXPECTED_CAPTURE_KEY],
            },
        },
        "source_snapshots_differ": base_prediction.get("source_snapshot") !=
                                   target_prediction.get("source_snapshot"),
        "full_physics_trajectory_claim": False,
    }


def verify_capture_pending(target_case: Path, target_prediction: dict[str, Any],
                           target_prediction_raw: bytes,
                           target_receipt: dict[str, Any],
                           target_environment_record: dict[str, Any]) -> dict[str, Any]:
    module = load_source_environment_module()
    try:
        module.validate_run_record(target_environment_record)
        module.validate_prediction_binding(target_environment_record, target_prediction_raw)
        contract = module.validate_diagnostic_contract(
            target_prediction.get("effective_native_environment_contract"))
    except (ValueError, TypeError) as error:
        raise AuditError(f"source167 environment contract validation failed: {error}") from error
    if target_prediction.get("effective_native_environment_contract_sha256") != contract["sha256"]:
        raise AuditError("prediction environment contract digest mismatch")
    if target_environment_record.get("prediction_sha256") != sha256_bytes(target_prediction_raw):
        raise AuditError("native environment record prediction hash mismatch")
    if target_receipt.get("diagnostic_capture_validation") is None:
        raise AuditError("terminal runner receipt omits diagnostic validation result")
    prefixes = contract.get("capture_prefixes")
    if not isinstance(prefixes, dict) or set(prefixes) != {EXPECTED_CAPTURE_KEY}:
        raise AuditError("470 capture contract does not contain only the regional consumer path")
    if contract["effective"][EXPECTED_SELECTOR_KEY] != {
            "set": True, "value": EXPECTED_SELECTOR}:
        raise AuditError("470 effective regional selector is not the expected future control")
    try:
        recomputed = module.validate_diagnostic_outputs(
            target_case, prefixes, native_argv=target_prediction["argv"],
            native_exit_code=target_receipt["exit_code"],
            native_timed_out=target_receipt["timed_out"],
            effective_contract=contract)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise AuditError(f"source167 pending-capture validation failed: {error}") from error
    if recomputed != target_receipt.get("diagnostic_capture_validation"):
        raise AuditError("receipt diagnostic validation differs from source167 recomputation")
    capture_record, _ = strict_json_file(
        target_case / "diagnostic-captures.json", "diagnostic captures inventory", 1 << 20)
    expected_record = {
        "schema": "numi.cardboard.diagnostic-capture-files.v1",
        "environment_contract_sha256": contract["sha256"],
        **recomputed,
    }
    if capture_record != expected_record:
        raise AuditError("diagnostic-captures.json differs from source167 recomputation")
    pending = recomputed.get("pending_captures")
    if (recomputed.get("status") != "validated_with_pending_checkpoint_targets"
            or recomputed.get("files") != [] or recomputed.get("total_bytes") != 0
            or not isinstance(pending, list) or len(pending) != 1):
        raise AuditError("regional capture was not correctly recorded as pending")
    item = pending[0]
    if (item.get("key") != EXPECTED_CAPTURE_KEY
            or item.get("selector_key") != EXPECTED_SELECTOR_KEY
            or item.get("selector") != EXPECTED_SELECTOR
            or item.get("target_control") != NEXT_CONTROL
            or item.get("accepted_control") != CHECKPOINT_CONTROL
            or item.get("status") != "not_yet_visited_at_accepted_checkpoint"
            or item.get("positive_qualification_allowed") is not False):
        raise AuditError("pending capture record has incorrect boundary or scope")
    return {
        "source167_validator_recomputed": True,
        "status": recomputed["status"],
        "pending_capture": item,
        "capture_files": 0,
        "capture_payload_bytes": 0,
        "positive_qualification_allowed": False,
    }


def audit(case469: Path, case470: Path) -> dict[str, Any]:
    if case469.is_symlink() or case470.is_symlink():
        raise AuditError("case directory must not be a symlink")
    case469 = case469.resolve(strict=True)
    case470 = case470.resolve(strict=True)
    if case469 == case470:
        raise AuditError("baseline and target cases must be distinct")
    for name in ("receipt.json", "prediction.json"):
        if not (case470 / name).is_file():
            raise AuditError(f"terminal target file is missing: {name}")

    base_prediction, base_prediction_raw = strict_json_file(
        case469 / "prediction.json", "469 prediction", 32 * 1024 * 1024)
    target_prediction, target_prediction_raw = strict_json_file(
        case470 / "prediction.json", "470 prediction", 32 * 1024 * 1024)
    assert_no_live_probe(case470, target_prediction)
    input_bindings = target_prediction.get("inputs_sha256")
    if not isinstance(input_bindings, dict) or not input_bindings:
        raise AuditError("470 prediction lacks retained build/source input hashes")
    input_copy_root = case470 / "inputs"
    input_inventory_digest = hashlib.sha256()
    input_total_bytes = 0
    for relative, expected_hash in sorted(input_bindings.items()):
        rel = PurePosixPath(relative)
        if (not isinstance(relative, str) or rel.is_absolute() or not rel.parts
                or ".." in rel.parts or "\\" in relative):
            raise AuditError(f"unsafe retained input binding: {relative!r}")
        if not isinstance(expected_hash, str) or not HEX64.fullmatch(expected_hash):
            raise AuditError(f"malformed retained input SHA-256: {relative}")
        bound_file = input_copy_root.joinpath(*rel.parts)
        actual_hash = sha256_file(bound_file)
        if actual_hash != expected_hash:
            raise AuditError(f"retained source/build input changed: {relative}")
        size = bound_file.stat().st_size
        input_total_bytes += size
        input_inventory_digest.update(relative.encode("utf-8") + b"\0")
        input_inventory_digest.update(bytes.fromhex(actual_hash))
    base_receipt, _ = strict_json_file(case469 / "receipt.json", "469 receipt", 1 << 20)
    target_receipt, _ = strict_json_file(case470 / "receipt.json", "470 receipt", 1 << 20)
    if (target_receipt.get("schema") != "numi.cardboard.native-run-receipt.v2"
            or target_receipt.get("execution_started") is not True
            or target_receipt.get("source_inputs_unchanged") is not True
            or target_receipt.get("changed_inputs") != []
            or target_receipt.get("timed_out") is not False
            or target_receipt.get("exit_code") != 0
            or target_receipt.get("run_mode") != "diagnostic"
            or target_receipt.get("termination_errors") != []):
        raise AuditError("470 receipt is not a clean terminal diagnostic prefix run")
    if (base_receipt.get("schema") != "numi.cardboard.native-run-receipt.v2"
            or base_receipt.get("exit_code") != 2
            or base_receipt.get("timed_out") is not False
            or base_receipt.get("source_inputs_unchanged") is not True):
        raise AuditError("469 baseline receipt differs from the retained failed baseline")

    source_bindings = verify_source_bindings()
    prediction_comparison = compare_predictions(case469, case470,
                                                base_prediction, target_prediction)
    base_output = case469 / "output"
    target_output = case470 / "output"
    base_result, _ = strict_json_file(base_output / "result.json", "469 result")
    target_result, _ = strict_json_file(target_output / "result.json", "470 result")
    production = compare_common_outputs(base_output, target_output,
                                        target_result, base_result)
    csv_result = verify_csv_accepted_prefix(base_output, target_output)
    base_frames: dict[str, str] = {}
    target_frames: dict[str, str] = {}
    for step in range(1, NEXT_CONTROL + 1):
        for arm in ENVS.values():
            relative = f"accepted_step_{step:06d}_{arm}.obj"
            base_path = base_output / relative
            target_path = target_output / relative
            base_hash = sha256_file(base_path)
            target_hash = sha256_file(target_path)
            if base_hash != target_hash:
                raise AuditError(f"accepted OBJ prefix mismatch: {relative}")
            base_frames[relative] = base_hash
            target_frames[relative] = target_hash
    environment_record, environment_raw = strict_json_file(
        case470 / "native-environment.json", "470 native environment", 2 * 1024 * 1024)
    checkpoint = validate_checkpoint(
        target_output, case470, target_prediction, target_prediction_raw,
        environment_record, environment_raw, target_result)
    capture = verify_capture_pending(
        case470, target_prediction, target_prediction_raw,
        target_receipt, environment_record)

    # Retain hashes for the inputs that define the comparison and complete
    # source-side contract; no 470 files are read until this function runs.
    inputs = {
        "469_prediction_sha256": sha256_bytes(base_prediction_raw),
        "470_prediction_sha256": sha256_bytes(target_prediction_raw),
        "469_manifest_sha256": sha256_file(base_output / "manifest.json"),
        "470_manifest_sha256": sha256_file(target_output / "manifest.json"),
        "469_observations_sha256": sha256_file(base_output / "observations.csv"),
        "470_observations_sha256": sha256_file(target_output / "observations.csv"),
        "470_retained_input_count": len(input_bindings),
        "470_retained_input_bytes": input_total_bytes,
        "470_retained_input_inventory_sha256": input_inventory_digest.hexdigest(),
        "source_files": source_bindings,
    }
    return {
        "schema": "numi.cardboard.native470-accepted-prefix-adjudication.v1",
        "verdict": "PASS_ACCEPTED_PREFIX_ONLY",
        "cases": {"baseline": str(case469), "target": str(case470)},
        "comparison": {
            "prefix_controls": [0, CHECKPOINT_CONTROL],
            "accepted_control_count": NEXT_CONTROL,
            "environment_rows_per_control": 2,
            "csv": csv_result,
            "accepted_obj_frame_count": len(target_frames),
            "accepted_obj_frames_exact": True,
            "accepted_obj_sha256": target_frames,
            "common_production": production,
            "argv_and_environment": prediction_comparison,
            "checkpoint": checkpoint,
            "future_capture": capture,
        },
        "input_sha256": inputs,
        "limits": {
            "full_protocol_complete": False,
            "full_trajectory_claim": False,
            "checkpoint_restore_parity_established": False,
            "physical_validation": False,
            "post_control_19_behavior_compared": False,
            "future_regional_capture_payload_present": False,
        },
    }


def self_test() -> None:
    sample = (
        "schema=numi.cardboard.accepted-coupon-prefix.v1\n"
        "source_run_directory=/tmp/run/output\n"
        "accepted_control_index=19\n"
        "observations_prefix_bytes=5\n"
        "observations_prefix_sha256=" + "a" * 64 + "\n"
        "file=observations.csv|5|" + "a" * 64 + "|1\n"
    ).encode()
    parsed = parse_prefix_manifest(sample)
    assert parsed["accepted_control_index"] == 19
    assert parsed["entries"][0]["observations_prefix"] is True
    assert len(expected_accepted_names()) == 45
    for bad in (
        sample + b"accepted_control_index=19\n",
        sample.replace(b"observations.csv", b"../observations.csv"),
        sample.replace(b"schema=numi.cardboard.accepted-coupon-prefix.v1",
                       b"schema=numi.cardboard.wrong.v1"),
    ):
        try:
            parse_prefix_manifest(bad)
        except (AuditError, UnicodeDecodeError):
            pass
        else:
            raise AssertionError("invalid synthetic prefix was admitted")
    for bad_json in (b'{"x":1,"x":2}', b'{"x":NaN}'):
        try:
            strict_json_bytes(bad_json, "synthetic strict JSON")
        except AuditError:
            pass
        else:
            raise AssertionError("invalid strict JSON was admitted")
    print("self-test: 6 parser/schema cases passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-469", type=Path, default=BASELINE)
    parser.add_argument("--case-470", type=Path, default=TARGET)
    parser.add_argument("--output", type=Path,
                        help="write the completed report to this path")
    parser.add_argument("--terminal-ack",
                        help="must equal '470 terminal confirmed by operator'")
    parser.add_argument("--self-test", action="store_true",
                        help="run synthetic parser tests without reading either case")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.terminal_ack != "470 terminal confirmed by operator":
        parser.error("--terminal-ack must explicitly confirm that 470 is terminal")
    try:
        report = audit(args.case_469, args.case_470)
    except (AuditError, OSError, ValueError, TypeError, KeyError) as error:
        report = {
            "schema": "numi.cardboard.native470-accepted-prefix-adjudication.v1",
            "verdict": "FAIL_CLOSED",
            "error": str(error),
            "cases": {"baseline": str(args.case_469), "target": str(args.case_470)},
            "limits": {
                "full_protocol_complete": False,
                "full_trajectory_claim": False,
                "checkpoint_restore_parity_established": False,
                "physical_validation": False,
            },
        }
    rendered = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output:
        output = args.output
        if output.exists():
            raise AuditError(f"refusing to overwrite report: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["verdict"] == "PASS_ACCEPTED_PREFIX_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
