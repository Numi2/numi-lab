#!/usr/bin/env python3
"""Strict reader for numi.matter.base_cgls_state_capture.v1 sidecars.

The payload is an exact byte snapshot. This reader validates structure and
hashes only; record: arrays remain opaque and this tool makes no physics claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

SCHEMA = "numi.matter.base_cgls_state_capture.v1"
CAPTURE_POINT = "after_newton_correction_before_cgls_direction_construction_or_installation"
MAX_PAYLOAD_BYTES = 96 * 1024 * 1024
EXPECTED_ARRAYS = {
    "accepted_nodes",
    "candidate_nodes",
    "accepted_tetrahedra",
    "candidate_tetrahedra",
    "accepted_material_history",
    "candidate_material_state_buffer",
    "objects",
    "materials",
    "scalar_programs",
    "expression_instructions",
    "environment_parameters",
    "schedulers",
    "adaptive_state",
    "original_newton_correction",
    "base_residual_rhs",
    "algorithmic_tangent_cache",
    "algorithmic_tangent_cache_valid",
    "fgmres_state",
    "mixed_solver",
    "armijo_metrics",
    "armijo_statuses",
    "production_statuses",
    "fem_line_search",
    "environment_line_search",
}
RECORD_WIDTHS = {
    "record:NMFEMNodeStateGPU": 80,
    "record:NMTetrahedronGPU": 112,
    "float32": 4,
    "float32x4": 16,
    "uint32": 4,
}
HEX16 = re.compile(r"^[0-9a-f]{16}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CaptureError(ValueError):
    pass


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CaptureError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise CaptureError(f"nonstandard JSON numeric token: {token}")


def _read_json_strict(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CaptureError(f"cannot read strict JSON manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise CaptureError("manifest root must be an object")
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(obj: dict[str, Any], key: str, minimum: int = 0) -> int:
    value = obj.get(key)
    if not _is_int(value) or value < minimum:
        raise CaptureError(f"{key} must be an integer >= {minimum}")
    return value


def _require_float(obj: dict[str, Any], key: str) -> float:
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaptureError(f"{key} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise CaptureError(f"{key} must be finite")
    return result


def _sha(path_or_bytes: Path | bytes) -> str:
    digest = hashlib.sha256()
    if isinstance(path_or_bytes, bytes):
        digest.update(path_or_bytes)
    else:
        with path_or_bytes.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_metadata(doc: dict[str, Any]) -> tuple[int, int, int, int]:
    if doc.get("schema") != SCHEMA:
        raise CaptureError("unsupported state-capture schema")
    if doc.get("complete") is not True:
        raise CaptureError("manifest does not carry complete=true")
    if doc.get("byte_order") != "little_endian":
        raise CaptureError("payload byte order must be little_endian")
    if doc.get("capture_point") != CAPTURE_POINT:
        raise CaptureError("unexpected capture point")
    if doc.get("physical_state_changed") is not False:
        raise CaptureError("capture must explicitly state physical_state_changed=false")
    root = _require_int(doc, "root")
    if root != _require_int(doc, "control_step"):
        raise CaptureError("root and control_step disagree")
    _require_int(doc, "physics_substep")
    _require_int(doc, "microtick")
    _require_int(doc, "microtick_count", 1)
    _require_int(doc, "newton_iteration")
    e = _require_int(doc, "environment_count", 1)
    objects = _require_int(doc, "object_count")
    materials = _require_int(doc, "material_count")
    parameters = _require_int(doc, "parameter_count")
    n = _require_int(doc, "node_count", 1)
    t = _require_int(doc, "tetrahedron_count", 1)
    stride = _require_int(doc, "material_state_stride", 1)
    unknowns = _require_int(doc, "unknown_count", 1)
    cgls_iterations = _require_int(doc, "cgls_iterations", 1)
    if e > 2 or n > 4096 or t > 32768 or stride > 32:
        raise CaptureError("capture extent exceeds the admitted diagnostic scope")
    if unknowns != 2 * e * n:
        raise CaptureError("unknown_count must equal 2*environment_count*node_count")
    if cgls_iterations not in (32, 64):
        raise CaptureError("cgls_iterations must be 32 or 64")
    if _require_int(doc, "array_count", 1) != len(EXPECTED_ARRAYS):
        raise CaptureError("array_count must equal the complete v1 array table")
    if doc.get("layout", {}).get("unknown_count") != unknowns:
        raise CaptureError("layout unknown_count disagrees with top-level count")
    if doc.get("layout", {}).get("mechanical_rows") != "first E*N float4 entries":
        raise CaptureError("unexpected mechanical row layout")
    if doc.get("layout", {}).get("field_backing_rows") != (
        "second E*N float4 entries; CGLS requires exact pure-FEM layout and preserves captured bits"
    ):
        raise CaptureError("unexpected field backing layout")
    _require_float(doc, "dt_seconds")
    _require_float(doc, "inverse_dt_seconds")
    source_fp = doc.get("source_physics_fingerprint_u64_hex")
    world_fp = doc.get("world_fingerprint_u64_hex")
    program_fp = doc.get("device_program_fingerprint_u64_hex")
    if not all(isinstance(v, str) and HEX16.fullmatch(v)
               for v in (source_fp, world_fp, program_fp)):
        raise CaptureError("source, world, and device fingerprints must be 16 lowercase hex digits")
    device = doc.get("device")
    if not isinstance(device, dict) or not isinstance(device.get("name"), str):
        raise CaptureError("device identity missing")
    _require_int(device, "registry_id")
    tangent = doc.get("tangent_cache")
    if not isinstance(tangent, dict) or not isinstance(tangent.get("enabled_by_solver_policy"), bool):
        raise CaptureError("tangent-cache metadata missing")
    if not isinstance(tangent.get("force_rebuild_requested"), bool):
        raise CaptureError("tangent-cache rebuild metadata missing")
    _require_int(tangent, "valid_entry_count")
    if not isinstance(tangent.get("generation_claim"), str):
        raise CaptureError("tangent-cache generation claim missing")
    projection = doc.get("material_history_projection")
    if not isinstance(projection, dict):
        raise CaptureError("material-history projection status missing")
    if projection.get("current_newton_status") != "unavailable_not_persisted":
        raise CaptureError("projected material history must remain explicitly unavailable")
    if projection.get("candidate_buffer_name") != "candidate_material_state_buffer" or \
       projection.get("candidate_buffer_is_current_projection") is not False:
        raise CaptureError("raw candidate material state must not be misidentified as projected history")
    if not isinstance(projection.get("reason"), str) or not projection["reason"]:
        raise CaptureError("projection unavailability reason missing")
    cgls_name = doc.get("cgls_capture_file")
    if not isinstance(cgls_name, str) or not cgls_name or Path(cgls_name).name != cgls_name:
        raise CaptureError("cgls_capture_file must be a basename")
    _ = (objects, materials, parameters)
    return e, n, t, stride


def validate_capture(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    doc = _read_json_strict(manifest_path)
    e, n, t, stride = _validate_metadata(doc)
    payload = doc.get("payload")
    if not isinstance(payload, dict):
        raise CaptureError("payload descriptor missing")
    payload_name = payload.get("file")
    if not isinstance(payload_name, str) or not payload_name or Path(payload_name).name != payload_name:
        raise CaptureError("payload filename must be an adjacent basename")
    payload_path = manifest_path.parent / payload_name
    if not payload_path.is_file():
        raise CaptureError("manifest-adjacent payload is missing; no path fallback is allowed")
    declared_bytes = _require_int(payload, "byte_count", 1)
    if declared_bytes > MAX_PAYLOAD_BYTES:
        raise CaptureError("payload exceeds the 96 MiB capture bound")
    if payload_path.stat().st_size != declared_bytes:
        raise CaptureError("payload byte count mismatch")
    declared_sha = payload.get("sha256")
    if not isinstance(declared_sha, str) or not HEX64.fullmatch(declared_sha):
        raise CaptureError("payload SHA-256 malformed")
    actual_sha = _sha(payload_path)
    if actual_sha != declared_sha:
        raise CaptureError("payload SHA-256 mismatch")

    arrays = doc.get("arrays")
    if not isinstance(arrays, list) or len(arrays) != len(EXPECTED_ARRAYS):
        raise CaptureError("array table must contain exactly the v1 required arrays")
    names: set[str] = set()
    spans: list[tuple[int, int, str]] = []
    with payload_path.open("rb") as stream:
        for item in arrays:
            if not isinstance(item, dict):
                raise CaptureError("array descriptor must be an object")
            name = item.get("name")
            if not isinstance(name, str) or name not in EXPECTED_ARRAYS or name in names:
                raise CaptureError(f"unknown or duplicate array name: {name!r}")
            names.add(name)
            dtype = item.get("dtype")
            if not isinstance(dtype, str) or not dtype:
                raise CaptureError(f"array {name}: dtype missing")
            shape = item.get("shape")
            if not isinstance(shape, list) or any(not _is_int(x) or x < 0 for x in shape):
                raise CaptureError(f"array {name}: invalid shape")
            elem = _require_int(item, "element_bytes", 1)
            count = _require_int(item, "byte_count")
            offset = _require_int(item, "offset")
            expected = math.prod(shape) * elem
            if expected != count:
                raise CaptureError(f"array {name}: shape/element extent mismatch")
            if dtype in RECORD_WIDTHS and elem != RECORD_WIDTHS[dtype]:
                raise CaptureError(f"array {name}: unexpected element width for {dtype}")
            if offset % 16 != 0:
                raise CaptureError(f"array {name}: offset is not 16-byte aligned")
            if offset + count > declared_bytes:
                raise CaptureError(f"array {name}: extent exceeds payload")
            span = (offset, offset + count, name)
            for other_start, other_end, other_name in spans:
                if max(offset, other_start) < min(offset + count, other_end):
                    raise CaptureError(f"arrays {other_name} and {name} overlap")
            spans.append(span)
            digest = item.get("sha256")
            if not isinstance(digest, str) or not HEX64.fullmatch(digest):
                raise CaptureError(f"array {name}: SHA-256 malformed")
            stream.seek(offset)
            array_bytes = stream.read(count)
            if len(array_bytes) != count or _sha(array_bytes) != digest:
                raise CaptureError(f"array {name}: byte count or SHA-256 mismatch")
    if names != EXPECTED_ARRAYS:
        raise CaptureError("array table is incomplete")

    expected_shapes = {
        "accepted_nodes": [e, n],
        "candidate_nodes": [e, n],
        "accepted_tetrahedra": [e, t],
        "candidate_tetrahedra": [e, t],
        "accepted_material_history": [e, t, stride],
        "candidate_material_state_buffer": [e, t, stride],
        "original_newton_correction": [2 * e * n],
        "base_residual_rhs": [2 * e * n],
        "algorithmic_tangent_cache": [e, t, 9, 9],
        "algorithmic_tangent_cache_valid": [e, t],
        "fgmres_state": [e],
        "mixed_solver": [1],
        "armijo_metrics": [e],
        "armijo_statuses": [e],
        "production_statuses": [e],
        "environment_parameters": [e, doc["parameter_count"]],
        "schedulers": [e, doc["object_count"]],
        "adaptive_state": [e, doc["object_count"]],
        "fem_line_search": [e, doc["object_count"]],
        "environment_line_search": [e],
    }
    expected_dtypes = {
        "accepted_nodes": "record:NMFEMNodeStateGPU",
        "candidate_nodes": "record:NMFEMNodeStateGPU",
        "accepted_tetrahedra": "record:NMTetrahedronGPU",
        "candidate_tetrahedra": "record:NMTetrahedronGPU",
        "accepted_material_history": "float32",
        "candidate_material_state_buffer": "float32",
        "environment_parameters": "float32",
        "original_newton_correction": "float32x4",
        "base_residual_rhs": "float32x4",
        "algorithmic_tangent_cache": "float32",
        "algorithmic_tangent_cache_valid": "uint32",
        "armijo_metrics": "float32x4",
        "fem_line_search": "float32x4",
        "environment_line_search": "float32",
    }
    for item in arrays:
        if item["name"] in expected_shapes and item["shape"] != expected_shapes[item["name"]]:
            raise CaptureError(f"array {item['name']}: shape disagrees with capture counts")
        expected_dtype = expected_dtypes.get(item["name"])
        if expected_dtype is not None and item["dtype"] != expected_dtype:
            raise CaptureError(f"array {item['name']}: expected dtype {expected_dtype}")
        if not isinstance(item.get("semantic"), str) or not item["semantic"]:
            raise CaptureError(f"array {item['name']}: semantic label missing")
    cursor = 0
    with payload_path.open("rb") as stream:
        for start, end, name in sorted(spans):
            if start > cursor:
                stream.seek(cursor)
                padding = stream.read(start - cursor)
                if any(padding):
                    raise CaptureError(f"nonzero padding before array {name}")
            cursor = max(cursor, end)
        if cursor < declared_bytes:
            stream.seek(cursor)
            if any(stream.read(declared_bytes - cursor)):
                raise CaptureError("nonzero trailing payload bytes")
    return {
        "manifest": str(manifest_path),
        "payload": str(payload_path),
        "schema": SCHEMA,
        "root": doc["root"],
        "newton_iteration": doc["newton_iteration"],
        "environment_count": e,
        "node_count": n,
        "tetrahedron_count": t,
        "array_count": len(arrays),
        "payload_bytes": declared_bytes,
        "payload_sha256": actual_sha,
        "projected_history_status": "unavailable_not_persisted",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        result = validate_capture(args.manifest)
    except (CaptureError, OSError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"valid": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
