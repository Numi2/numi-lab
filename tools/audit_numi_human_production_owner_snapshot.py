#!/usr/bin/env python3
"""Audit one persistent Numi Human production-owner snapshot evidence file.

The evidence envelope hashes the literal JSON bytes occupied by ``payload``.
This tool verifies that byte range without reserializing it, validates the v1
shape and coverage contract, reconstructs every word as little-endian bytes,
and independently reproduces the recorder's FP64 dynamics derivation before
requiring bit-exact FP32 witnesses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any, Callable


EVIDENCE_SCHEMA = "persistent-production-owner-snapshot-evidence.v1"
PAYLOAD_SCHEMA = "persistent-production-owner-snapshot.v1"
AUDIT_SCHEMA = "numi.human.production-owner-snapshot-audit.v1"
MAX_INPUT_BYTES = 1 << 30
UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1
FLOAT32_MAX = struct.unpack("<f", bytes.fromhex("ffff7f7f"))[0]
FLOAT32_MIN_NORMAL = struct.unpack("<f", bytes.fromhex("00008000"))[0]
FLOAT32_EPSILON = 2.0 ** -23
PUBLISHED_REQUIRED_COVERAGE_MASK = 0x7BFFFF
REJECTED_REQUIRED_COVERAGE_MASK = 0x5DFFFF
INVALID_INDEX = UINT32_MAX
HEX32 = re.compile(r"[0-9a-f]{8}\Z")
HEX64 = re.compile(r"[0-9a-f]{16}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
JSON_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


class AuditError(RuntimeError):
    """The artifact violates the production-owner evidence contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _strict_json(text: str, context: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            require(key not in result, f"{context} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise AuditError(f"{context} has non-finite JSON constant {value}")

    try:
        value = json.loads(
            text, object_pairs_hook=object_pairs, parse_constant=invalid_constant
        )
    except (json.JSONDecodeError, TypeError, ValueError, RecursionError) as error:
        raise AuditError(f"{context} is not strict JSON") from error
    require(isinstance(value, dict), f"{context} is not a JSON object")
    return value


def _skip_space(text: str, offset: int) -> int:
    while offset < len(text) and text[offset] in " \t\r\n":
        offset += 1
    return offset


def _scan_string(text: str, offset: int) -> int:
    require(offset < len(text) and text[offset] == '"', "invalid JSON string")
    offset += 1
    while offset < len(text):
        character = text[offset]
        if character == '"':
            return offset + 1
        if character == "\\":
            offset += 2
        else:
            offset += 1
    raise AuditError("unterminated JSON string")


def _scan_value(text: str, offset: int) -> int:
    offset = _skip_space(text, offset)
    require(offset < len(text), "missing JSON value")
    character = text[offset]
    if character == '"':
        return _scan_string(text, offset)
    if character == "{":
        offset = _skip_space(text, offset + 1)
        if offset < len(text) and text[offset] == "}":
            return offset + 1
        while True:
            key_end = _scan_string(text, offset)
            offset = _skip_space(text, key_end)
            require(offset < len(text) and text[offset] == ":", "missing JSON colon")
            offset = _scan_value(text, offset + 1)
            offset = _skip_space(text, offset)
            require(offset < len(text), "unterminated JSON object")
            if text[offset] == "}":
                return offset + 1
            require(text[offset] == ",", "invalid JSON object separator")
            offset = _skip_space(text, offset + 1)
    if character == "[":
        offset = _skip_space(text, offset + 1)
        if offset < len(text) and text[offset] == "]":
            return offset + 1
        while True:
            offset = _scan_value(text, offset)
            offset = _skip_space(text, offset)
            require(offset < len(text), "unterminated JSON array")
            if text[offset] == "]":
                return offset + 1
            require(text[offset] == ",", "invalid JSON array separator")
            offset = _skip_space(text, offset + 1)
    for literal in ("true", "false", "null"):
        if text.startswith(literal, offset):
            return offset + len(literal)
    match = JSON_NUMBER.match(text, offset)
    require(match is not None, "invalid JSON value")
    return match.end()


def _top_level_spans(text: str) -> dict[str, tuple[int, int]]:
    """Return literal value spans for members of the top-level JSON object."""
    offset = _skip_space(text, 0)
    require(offset < len(text) and text[offset] == "{", "evidence is not an object")
    offset = _skip_space(text, offset + 1)
    spans: dict[str, tuple[int, int]] = {}
    while offset < len(text) and text[offset] != "}":
        key_end = _scan_string(text, offset)
        try:
            key = json.loads(text[offset:key_end])
        except json.JSONDecodeError as error:
            raise AuditError("invalid top-level JSON key") from error
        require(isinstance(key, str), "top-level JSON key is not a string")
        require(key not in spans, f"evidence has duplicate JSON key {key!r}")
        offset = _skip_space(text, key_end)
        require(offset < len(text) and text[offset] == ":", "missing top-level colon")
        start = _skip_space(text, offset + 1)
        end = _scan_value(text, start)
        spans[key] = (start, end)
        offset = _skip_space(text, end)
        require(offset < len(text), "unterminated evidence object")
        if text[offset] == "}":
            break
        require(text[offset] == ",", "invalid top-level object separator")
        offset = _skip_space(text, offset + 1)
    require(offset < len(text) and text[offset] == "}", "unterminated evidence object")
    require(_skip_space(text, offset + 1) == len(text), "trailing data after evidence")
    return spans


def _exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    missing = sorted(expected - value.keys())
    extra = sorted(value.keys() - expected)
    require(not missing, f"{context} is missing fields: {', '.join(missing)}")
    require(not extra, f"{context} has unexpected fields: {', '.join(extra)}")


IDENTITY_FIELDS = {
    "base_state_fingerprint",
    "treatment_history_fingerprint",
    "human_source_fingerprint",
    "matter_source_physics_fingerprint",
    "matter_device_program_fingerprint",
    "owner_program_fingerprint",
    "transaction_fingerprint",
    "previous_transaction_fingerprint",
    "linearization_epoch",
    "slot_generation",
    "physics_generation",
    "previous_physics_generation",
    "brain_generation",
    "sensor_generation",
    "human_io_program_fingerprint",
    "sensor_fingerprint",
    "transaction_instance_fingerprint",
    "joint_fence_fingerprint",
    "equality_program_fingerprint",
    "limit_program_fingerprint",
    "coverage_mask",
    "required_coverage_mask",
    "truncation_mask",
}

COUNT_FIELDS = {
    "q_coordinate_count",
    "dof_count",
    "muscle_count",
    "muscle_site_count",
    "muscle_wrap_count",
    "muscle_route_node_count",
    "support_row_count",
    "equality_row_count",
    "limit_row_count",
    "tendon_row_count",
    "contact_sample_count",
    "terminal_accepted_matter_rigid_generalized_state_count",
    "terminal_accepted_matter_rigid_reaction_count",
}

PAYLOAD_FIELDS = {
    "schema",
    "format_version",
    "comparison_identity_algorithm",
    "comparison_identity_is_cryptographic_proof",
    "rhs_bias_origin",
    "constraint_witness_origin",
    "inertial_operator_capture",
    "effective_tangent_factor_storage_layout",
    "rollback_identity_scope",
    "terminal_matter_state_role",
    "work_energy_scope",
    "treatment",
    "disposition",
    *IDENTITY_FIELDS,
    "control_step",
    "candidate_timestamp_nanoseconds",
    "publication_epoch",
    "timestep_nanoseconds",
    "support_payload_byte_count",
    "support_payload_abi",
    "support_payload_sha256",
    "matter_control_step",
    *COUNT_FIELDS,
    "publication_identity_available",
    "rollback_identity_available",
    "arrays",
}

# name -> (logical element count, bytes per element).  Count callbacks mirror
# serializeNumiHumanProductionOwnerSnapshotV1 exactly.
ARRAY_SPECS: dict[str, tuple[Callable[[dict[str, Any]], int], int]] = {
    "initial_q": (lambda p: p["q_coordinate_count"], 4),
    "initial_v": (lambda p: p["dof_count"], 4),
    "initial_root": (lambda p: 1, 48),
    "initial_muscles": (lambda p: p["muscle_count"], 16),
    "muscle_records": (lambda p: p["muscle_count"], 224),
    "muscle_sites": (lambda p: p["muscle_site_count"], 32),
    "muscle_wraps": (lambda p: p["muscle_wrap_count"], 96),
    "muscle_route_nodes": (lambda p: p["muscle_route_node_count"], 16),
    "checkpoint_q": (lambda p: p["q_coordinate_count"], 4),
    "checkpoint_v": (lambda p: p["dof_count"], 4),
    "checkpoint_root": (lambda p: 1, 48),
    "checkpoint_muscles": (lambda p: p["muscle_count"], 16),
    "effective_tangent_factor_storage": (
        lambda p: p["dof_count"] * p["dof_count"],
        4,
    ),
    "source_generalized_force": (lambda p: p["dof_count"], 4),
    "source_predicted_velocity": (lambda p: p["dof_count"], 4),
    "candidate_q": (lambda p: p["q_coordinate_count"], 4),
    "candidate_v": (lambda p: p["dof_count"], 4),
    "candidate_root": (lambda p: 1, 48),
    "candidate_muscles": (lambda p: p["muscle_count"], 16),
    "muscle_results": (lambda p: p["muscle_count"], 96),
    "muscle_generalized_forces": (
        lambda p: p["muscle_count"] * p["dof_count"],
        4,
    ),
    "reduced_muscle_generalized_force": (lambda p: p["dof_count"], 4),
    "tendon_transfers": (lambda p: p["tendon_row_count"], 112),
    "tendon_generalized_corrections": (
        lambda p: p["tendon_row_count"] * p["dof_count"],
        4,
    ),
    "matter_generalized_reaction": (lambda p: p["dof_count"], 4),
    "support_rows": (lambda p: p["support_row_count"], 80),
    "support_plane": (lambda p: 2, 16),
    "initial_support_histories": (lambda p: p["support_row_count"], 16),
    "candidate_support_histories": (lambda p: p["support_row_count"], 16),
    "candidate_support_consequences": (lambda p: p["support_row_count"], 64),
    "terminal_accepted_support_histories": (
        lambda p: p["support_row_count"],
        16,
    ),
    "terminal_accepted_support_consequences": (
        lambda p: p["support_row_count"],
        64,
    ),
    "equality_rows": (lambda p: p["equality_row_count"], 112),
    "equality_reconstructed_linearization_impulses": (
        lambda p: p["equality_row_count"],
        32,
    ),
    "limit_rows": (lambda p: p["limit_row_count"], 80),
    "limit_reconstructed_linearization_impulses": (
        lambda p: 2 * p["limit_row_count"],
        32,
    ),
    "contact_samples": (lambda p: p["contact_sample_count"], 160),
    "terminal_accepted_matter_rigid_generalized_state": (
        lambda p: p["terminal_accepted_matter_rigid_generalized_state_count"],
        4,
    ),
    "terminal_accepted_matter_rigid_reactions": (
        lambda p: p["terminal_accepted_matter_rigid_reaction_count"],
        32,
    ),
    "stand_status": (lambda p: 1, 272),
    "human_matter_owner_status": (lambda p: 1, 96),
    "acceleration": (lambda p: p["dof_count"], 4),
    "source_rhs": (lambda p: p["dof_count"], 4),
    "source_bias": (lambda p: p["dof_count"], 4),
    "work_energy_components": (lambda p: 3, 4),
}

COVERAGE_NAMES = (
    "initial_state",
    "checkpoint_state",
    "effective_tangent_factor",
    "source_generalized_force",
    "free_velocity",
    "candidate_state",
    "muscle_state",
    "muscle_results",
    "muscle_generalized_forces",
    "tendon",
    "matter_reaction",
    "support_rows",
    "support_history",
    "equality_rows",
    "limit_rows",
    "rhs_bias_acceleration",
    "work_energy",
    "publication_identity",
    "rollback_identity",
    "contact_samples",
    "muscle_program",
    "matter_integration_update",
    "status_records",
)


def _uint(value: Any, maximum: int, context: str, *, nonzero: bool = False) -> int:
    require(type(value) is int, f"{context} is not an integer")
    require(0 <= value <= maximum, f"{context} is outside its unsigned range")
    require(not nonzero or value != 0, f"{context} must be nonzero")
    return value


def _identity(payload: dict[str, Any], name: str) -> int:
    value = payload[name]
    require(isinstance(value, str) and HEX64.fullmatch(value) is not None,
            f"payload {name} is not 16-character lowercase hex")
    return int(value, 16)


def _validate_payload_header(payload: dict[str, Any]) -> dict[str, int]:
    _exact_keys(payload, PAYLOAD_FIELDS, "payload")
    require(payload["schema"] == PAYLOAD_SCHEMA, "payload schema is not v1")
    require(payload["format_version"] == 1 and type(payload["format_version"]) is int,
            "payload format_version is not 1")
    fixed = {
        "comparison_identity_algorithm": "fnv1a64-domain-v1",
        "rhs_bias_origin": "host-reconstructed-from-captured-A0-v0-vfree-tau",
        "constraint_witness_origin": "host-reconstructed-from-captured-production-inputs",
        "inertial_operator_capture": "source-effective-tangent-factor-storage",
        "effective_tangent_factor_storage_layout":
            "row-major-lower-cholesky-upper-source-A0",
        "rollback_identity_scope":
            "lifecycle-terminal-disposition-only-not-byte-restoration-proof",
        "work_energy_scope": "solver-components-not-physical-energy-closure",
    }
    for field, expected in fixed.items():
        require(payload[field] == expected, f"payload {field} changed")
    require(payload["comparison_identity_is_cryptographic_proof"] is False,
            "payload promotes comparison identity to cryptographic proof")
    require(payload["treatment"] in ("cold", "seeded"), "payload treatment invalid")
    disposition = payload["disposition"]
    require(disposition in ("published", "rejected"), "payload disposition invalid")
    expected_role = (
        "published-attempt-accepted-state" if disposition == "published"
        else "prior-accepted-state-context-after-rejection"
    )
    require(payload["terminal_matter_state_role"] == expected_role,
            "payload terminal Matter role contradicts disposition")

    identities = {name: _identity(payload, name) for name in IDENTITY_FIELDS}
    for name in (
        "base_state_fingerprint", "treatment_history_fingerprint",
        "human_source_fingerprint", "matter_source_physics_fingerprint",
        "matter_device_program_fingerprint", "owner_program_fingerprint",
        "transaction_fingerprint", "slot_generation", "physics_generation",
        "brain_generation", "sensor_generation", "human_io_program_fingerprint",
        "sensor_fingerprint", "transaction_instance_fingerprint",
    ):
        require(identities[name] != 0, f"payload {name} must be nonzero")

    for name in (
        "control_step", "candidate_timestamp_nanoseconds", "publication_epoch",
        "timestep_nanoseconds", "support_payload_byte_count",
    ):
        _uint(payload[name], UINT64_MAX, f"payload {name}")
    for name in ("support_payload_abi", "matter_control_step"):
        _uint(payload[name], UINT32_MAX, f"payload {name}")
    require(payload["matter_control_step"] == payload["control_step"],
            "payload matter_control_step does not equal control_step")
    for name in (
        "q_coordinate_count", "dof_count", "muscle_count",
        "support_row_count", "equality_row_count", "limit_row_count",
        "tendon_row_count",
    ):
        _uint(payload[name], UINT32_MAX, f"payload {name}")
    for name in (
        "muscle_site_count", "muscle_wrap_count", "muscle_route_node_count",
        "contact_sample_count",
        "terminal_accepted_matter_rigid_generalized_state_count",
        "terminal_accepted_matter_rigid_reaction_count",
    ):
        _uint(payload[name], UINT64_MAX, f"payload {name}")
    for name in ("q_coordinate_count", "dof_count", "muscle_count"):
        require(payload[name] != 0, f"payload {name} must be nonzero")
    require(payload["timestep_nanoseconds"] != 0,
            "payload timestep_nanoseconds must be nonzero")
    require(payload["support_payload_byte_count"] != 0,
            "payload support_payload_byte_count must be nonzero")
    require(payload["support_payload_abi"] != 0,
            "payload support_payload_abi must be nonzero")
    digest = payload["support_payload_sha256"]
    require(isinstance(digest, str) and SHA256.fullmatch(digest) is not None,
            "payload support_payload_sha256 is not lowercase SHA-256")
    require(int(digest, 16) != 0, "payload support_payload_sha256 is zero")
    require((payload["equality_row_count"] == 0) ==
            (identities["equality_program_fingerprint"] == 0),
            "payload equality row/program identity is inconsistent")
    require((payload["limit_row_count"] == 0) ==
            (identities["limit_program_fingerprint"] == 0),
            "payload limit row/program identity is inconsistent")
    require(type(payload["publication_identity_available"]) is bool and
            type(payload["rollback_identity_available"]) is bool,
            "payload terminal identity availability is not Boolean")
    if disposition == "published":
        require(payload["publication_identity_available"] is True and
                payload["rollback_identity_available"] is False and
                payload["publication_epoch"] != 0 and
                identities["joint_fence_fingerprint"] != 0,
                "published payload lacks exclusive publication identity")
    else:
        require(payload["rollback_identity_available"] is True and
                payload["publication_identity_available"] is False and
                payload["publication_epoch"] == 0 and
                identities["joint_fence_fingerprint"] == 0,
                "rejected payload lacks exclusive rollback identity")
    expected_required = (
        PUBLISHED_REQUIRED_COVERAGE_MASK
        if disposition == "published"
        else REJECTED_REQUIRED_COVERAGE_MASK
    )
    require(identities["required_coverage_mask"] == expected_required,
            "payload required_coverage_mask is not the exact disposition contract")
    return identities


def _validate_arrays(payload: dict[str, Any]) -> tuple[dict[str, bytes], dict[str, bool]]:
    arrays = payload["arrays"]
    require(isinstance(arrays, dict), "payload arrays is not an object")
    _exact_keys(arrays, set(ARRAY_SPECS), "payload arrays")
    reconstructed: dict[str, bytes] = {}
    available: dict[str, bool] = {}
    for name, (count_for, element_bytes) in ARRAY_SPECS.items():
        value = arrays[name]
        require(isinstance(value, dict), f"array {name} is not an object")
        _exact_keys(
            value,
            {"available", "expected_elements", "captured_elements", "element_bytes", "words"},
            f"array {name}",
        )
        require(type(value["available"]) is bool,
                f"array {name} availability is not Boolean")
        expected_elements = _uint(
            value["expected_elements"], UINT64_MAX, f"array {name} expected_elements"
        )
        captured_elements = _uint(
            value["captured_elements"], UINT64_MAX, f"array {name} captured_elements"
        )
        width = _uint(value["element_bytes"], UINT32_MAX,
                      f"array {name} element_bytes", nonzero=True)
        require(width == element_bytes, f"array {name} element width changed")
        logical_count = count_for(payload)
        require(logical_count <= UINT64_MAX, f"array {name} logical shape overflows u64")
        if value["available"]:
            require(expected_elements == logical_count,
                    f"array {name} available shape does not match payload counts")
            require(captured_elements == expected_elements,
                    f"array {name} is partially captured")
        else:
            require(captured_elements == 0,
                    f"array {name} is unavailable but claims captured elements")
        words = value["words"]
        require(isinstance(words, list), f"array {name} words is not an array")
        expected_word_count = captured_elements * width // 4
        require(width % 4 == 0 and len(words) == expected_word_count,
                f"array {name} word count does not match its shape")
        output = bytearray()
        for index, word in enumerate(words):
            require(isinstance(word, str) and HEX32.fullmatch(word) is not None,
                    f"array {name} word {index} is not 8-character lowercase hex")
            output.extend(struct.pack("<I", int(word, 16)))
        require(len(output) == captured_elements * width,
                f"array {name} reconstructed byte count changed")
        reconstructed[name] = bytes(output)
        available[name] = value["available"]
    return reconstructed, available


def _coverage_mask(payload: dict[str, Any], available: dict[str, bool]) -> int:
    result = 0

    def cover(bit: int, *names: str) -> None:
        nonlocal result
        if all(available[name] for name in names):
            result |= 1 << bit

    cover(0, "initial_q", "initial_v", "initial_root", "initial_muscles")
    cover(1, "checkpoint_q", "checkpoint_v", "checkpoint_root", "checkpoint_muscles")
    cover(2, "effective_tangent_factor_storage")
    cover(3, "source_generalized_force")
    cover(4, "source_predicted_velocity")
    cover(5, "candidate_q", "candidate_v", "candidate_root")
    cover(6, "candidate_muscles")
    cover(7, "muscle_results")
    cover(8, "muscle_generalized_forces", "reduced_muscle_generalized_force")
    cover(9, "tendon_transfers", "tendon_generalized_corrections")
    cover(10, "matter_generalized_reaction")
    cover(11, "support_rows", "support_plane")
    cover(12, "initial_support_histories", "candidate_support_histories",
          "candidate_support_consequences", "terminal_accepted_support_histories",
          "terminal_accepted_support_consequences")
    cover(13, "equality_rows", "equality_reconstructed_linearization_impulses")
    cover(14, "limit_rows", "limit_reconstructed_linearization_impulses")
    cover(15, "acceleration", "source_rhs", "source_bias")
    cover(16, "work_energy_components")
    if payload["publication_identity_available"]:
        result |= 1 << 17
    if payload["rollback_identity_available"]:
        result |= 1 << 18
    cover(19, "contact_samples")
    cover(20, "muscle_records", "muscle_sites", "muscle_wraps", "muscle_route_nodes")
    if payload["disposition"] == "published" and all(
        available[name] for name in (
            "terminal_accepted_matter_rigid_generalized_state",
            "terminal_accepted_matter_rigid_reactions",
        )
    ):
        result |= 1 << 21
    cover(22, "stand_status", "human_matter_owner_status")
    return result


def _floats(data: bytes, count: int, context: str) -> list[float]:
    require(len(data) == count * 4, f"{context} has the wrong byte count")
    values = list(struct.unpack(f"<{count}f", data)) if count else []
    for index, value in enumerate(values):
        require(math.isfinite(value), f"{context}[{index}] is nonfinite")
    return values


def _pack_f32(value: float, context: str) -> bytes:
    require(math.isfinite(value) and abs(value) <= FLOAT32_MAX,
            f"{context} is not FP32 representable")
    try:
        return struct.pack("<f", value)
    except (OverflowError, struct.error) as error:
        raise AuditError(f"{context} is not FP32 representable") from error


def _to_f32(value: float, context: str) -> float:
    return struct.unpack("<f", _pack_f32(value, context))[0]


def _compare_f32(name: str, expected: list[float], actual: bytes) -> list[float]:
    stored = _floats(actual, len(expected), f"stored {name}")
    for index, value in enumerate(expected):
        packed = _pack_f32(value, f"recomputed {name}[{index}]")
        begin = 4 * index
        require(actual[begin:begin + 4] == packed,
                f"stored {name}[{index}] does not exactly match FP64 recomputation")
    return stored


def _dominant(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    index = max(range(len(values)), key=lambda item: (abs(values[item]), -item))
    return {"dof": index, "value": values[index]}


def _f32_ulp(value: float) -> float:
    """Return the spacing to the next larger finite FP32 magnitude."""
    magnitude = abs(value)
    bits = struct.unpack("<I", struct.pack("<f", magnitude))[0]
    if bits == 0:
        return struct.unpack("<f", struct.pack("<I", 1))[0]
    if bits >= 0x7F7FFFFF:
        current = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]
        previous = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFE))[0]
        return current - previous
    current = struct.unpack("<f", struct.pack("<I", bits))[0]
    following = struct.unpack("<f", struct.pack("<I", bits + 1))[0]
    return following - current


def _require_close(actual: float, expected: float, context: str,
                   *, operations: int = 32) -> None:
    scale = max(abs(actual), abs(expected), 1.0)
    tolerance = operations * FLOAT32_EPSILON * scale
    require(abs(actual - expected) <= tolerance,
            f"{context} is inconsistent with captured row/state inputs")


def _require_positive_zero_word(data: bytes, word: int, context: str) -> None:
    begin = word * 4
    require(data[begin:begin + 4] == b"\x00\x00\x00\x00",
            f"{context} is not raw FP32 +0")


def _audit_effective_tangent(factor: list[float], count: int) -> dict[str, Any]:
    """Check positive L pivots and the source A0 retained above the diagonal.

    v1 stores L in the lower triangle (including the diagonal) and retains the
    original FP32 A0 only in the strict upper triangle.  The diagonal of A0 is
    therefore reconstructible but is not independently retained.
    """
    minimum_pivot = math.inf
    for row in range(count):
        pivot = factor[row * count + row]
        require(pivot > 0.0,
                f"effective tangent Cholesky diagonal {row} is not positive")
        minimum_pivot = min(minimum_pivot, pivot)

    checked = 0
    maximum_error = 0.0
    maximum_bound_ratio = 0.0
    # A lane-zero FP32 Cholesky has an O(n*u) backward error.  Use a
    # deliberately conservative 128*n*u envelope, scaled by |L||L^T|, plus a
    # handful of source-word ulps.  This remains scale-aware (no unit floor).
    # The constant is local to this reconstruction gate; exact dynamics/work
    # witnesses below retain bit-exact FP32 comparison.
    gamma = 128.0 * max(count, 1) * FLOAT32_EPSILON
    require(gamma < 0.25, "effective tangent dimension exceeds FP32 audit bound")
    for row in range(count):
        for column in range(row + 1, count):
            reconstructed = 0.0
            product_scale = 0.0
            for inner in range(row + 1):
                product = (
                    float(factor[row * count + inner]) *
                    float(factor[column * count + inner])
                )
                reconstructed += product
                product_scale += abs(product)
            source_a0 = factor[row * count + column]
            scale = max(product_scale, abs(source_a0), FLOAT32_MIN_NORMAL)
            bound = max(gamma * scale, 8.0 * _f32_ulp(source_a0))
            error = abs(reconstructed - source_a0)
            require(error <= bound,
                    "effective tangent strict-upper source A0 is inconsistent "
                    f"with L*L^T at ({row},{column})")
            checked += 1
            maximum_error = max(maximum_error, error)
            maximum_bound_ratio = max(maximum_bound_ratio, error / bound)
    return {
        "positive_cholesky_diagonal": True,
        "minimum_cholesky_diagonal": None if count == 0 else minimum_pivot,
        "strict_upper_source_a0_entries_checked": checked,
        "maximum_absolute_reconstruction_error": maximum_error,
        "maximum_error_to_bound_ratio": maximum_bound_ratio,
        "diagonal_source_a0_independently_retained": False,
    }


def _witness_summary(data: bytes, count: int, *, limits: bool) -> dict[str, Any]:
    values = _floats(data, count * 8, "constraint witnesses")
    records = [values[index * 8:(index + 1) * 8] for index in range(count)]
    active: list[tuple[int, list[float]]] = []
    for index, record in enumerate(records):
        record_bytes = data[index * 32:(index + 1) * 32]
        active_bits = struct.unpack("<I", record_bytes[24:28])[0]
        require(active_bits in (0, 0x3F800000),
                f"constraint witness {index} active flag is not raw +0/+1")
        _require_positive_zero_word(
            record_bytes, 7, f"constraint witness {index} reserved word"
        )
        if limits and active_bits == 0:
            require(record_bytes == bytes(32),
                    f"inactive limit witness {index} is not raw all +0")
        if not limits:
            require(active_bits == 0x3F800000,
                    f"equality witness {index} is not active")
        if active_bits == 0x3F800000:
            active.append((index, record))

    def largest(component: int) -> dict[str, Any] | None:
        if not active:
            return None
        index, record = max(active, key=lambda pair: (abs(pair[1][component]), -pair[0]))
        result: dict[str, Any] = {"value": record[component]}
        if limits:
            result.update({"row": index // 2, "side": "lower" if index % 2 == 0 else "upper"})
        else:
            result["row"] = index
        return result

    def largest_equality_solve_violation() -> dict[str, Any] | None:
        if not active:
            return None
        index, record = max(
            active, key=lambda pair: (abs(pair[1][5] - pair[1][1]), -pair[0])
        )
        return {"row": index, "value": record[5] - record[1]}

    result = {
        "record_count": count,
        "active_count": len(active),
        "largest_impulse": largest(4),
        "largest_position_violation": largest(3),
    }
    if limits:
        if not active:
            result["largest_violation"] = None
        else:
            # A limit violation is adverse in the negative direction.  An
            # absolute-magnitude maximum can incorrectly select a large
            # positive (satisfied) side of the same two-sided row.
            index, record = min(active, key=lambda pair: (pair[1][5], pair[0]))
            result["largest_violation"] = {
                "row": index // 2,
                "side": "lower" if index % 2 == 0 else "upper",
                "value": record[5],
            }
    else:
        # Equality word 5 is J0*(v_candidate-v_free), not a violation.
        # Subtract the stored bDelta (word 1) to report the solve violation.
        result["largest_row_delta_velocity"] = largest(5)
        result["largest_violation"] = largest_equality_solve_violation()
    return result


def _f32(value: float, context: str) -> float:
    return _to_f32(value, context)


def _within_f32_bound(actual: float, expected: float,
                      *, operations: int = 128) -> bool:
    scale = max(abs(actual), abs(expected), 1.0)
    return abs(actual - expected) <= operations * FLOAT32_EPSILON * scale


def _scalar_impedance(solimp0: list[float], solimp1: list[float],
                      phi: float) -> float:
    d0 = min(max(solimp0[0], 0.0001), 0.9999)
    dw = min(max(solimp0[1], 0.0001), 0.9999)
    width = max(solimp0[2], 0.0)
    midpoint = min(max(solimp0[3], 0.0001), 0.9999)
    power = max(solimp1[0], 1.0)
    if d0 == dw or width <= 1.0e-15:
        return _f32(0.5 * (d0 + dw), "constraint impedance")
    x = min(max(abs(phi) / width, 0.0), 1.0)
    try:
        if power == 1.0:
            y = x
        elif x <= midpoint:
            y = x ** power / midpoint ** (power - 1.0)
        else:
            y = 1.0 - (1.0 - x) ** power / (1.0 - midpoint) ** (power - 1.0)
    except (ArithmeticError, ValueError) as error:
        raise AuditError("constraint impedance cannot be reconstructed") from error
    return _f32(d0 + y * (dw - d0), "constraint impedance")


def _bdelta_for_flag(
    solref: list[float], solimp0: list[float], solimp1: list[float],
    phi: float, source_velocity: float, free_increment: float,
    timestep: float, reference_safe: bool,
) -> float:
    dw = min(max(solimp0[1], 0.0001), 0.9999)
    impedance = _scalar_impedance(solimp0, solimp1, phi)
    positive = solref[0] > 0.0
    time_constant = max(solref[0], 2.0 * timestep) if reference_safe else solref[0]
    if positive:
        denominator = max(
            1.0e-15,
            dw * dw * time_constant * time_constant * solref[1] * solref[1],
        )
        stiffness = _f32(1.0 / denominator, "constraint stiffness")
        damping = _f32(
            2.0 / max(1.0e-15, dw * time_constant),
            "constraint damping",
        )
    else:
        stiffness = _f32(-solref[0] / max(1.0e-15, dw * dw),
                         "constraint stiffness")
        damping = _f32(-solref[1] / max(1.0e-15, dw),
                       "constraint damping")
    reference_acceleration = _f32(
        -damping * source_velocity - stiffness * impedance * phi,
        "constraint reference acceleration",
    )
    return _f32(
        timestep * reference_acceleration - free_increment,
        "constraint bDelta",
    )


def _matching_reference_safe_flags(
    stored_bdelta: float, solref: list[float], solimp0: list[float],
    solimp1: list[float], phi: float, source_velocity: float,
    free_increment: float, timestep: float,
) -> set[int]:
    matches: set[int] = set()
    for flag in (0, 1):
        expected = _bdelta_for_flag(
            solref, solimp0, solimp1, phi, source_velocity,
            free_increment, timestep, bool(flag),
        )
        if _within_f32_bound(stored_bdelta, expected, operations=256):
            matches.add(flag)
    return matches


def _audit_constraint_witnesses(
    payload: dict[str, Any], arrays: dict[str, bytes]
) -> dict[str, Any]:
    nq = payload["q_coordinate_count"]
    nv = payload["dof_count"]
    require(nv >= 6 and nq == nv + 1,
            "captured scalar joint program dimensions are not floating-base nq=nv+1")
    q0 = _floats(arrays["checkpoint_q"], nq, "checkpoint_q")
    v0 = _floats(arrays["checkpoint_v"], nv, "checkpoint_v")
    free = _floats(arrays["source_predicted_velocity"], nv,
                   "source_predicted_velocity")
    candidate = _floats(arrays["candidate_v"], nv, "candidate_v")
    delta = [
        _f32(candidate[index] - free[index], f"candidate delta velocity[{index}]")
        for index in range(nv)
    ]
    timestep = _f32(
        float(payload["timestep_nanoseconds"]) * 1.0e-9,
        "constraint timestep",
    )

    equality_rows = arrays["equality_rows"]
    equality_witnesses = arrays["equality_reconstructed_linearization_impulses"]
    dependent: set[int] = set()
    equality_records: list[tuple[tuple[int, int, int, int], list[float]]] = []
    possible_flags = {0, 1}
    for index in range(payload["equality_row_count"]):
        row = equality_rows[index * 112:(index + 1) * 112]
        indices = struct.unpack("<4I", row[:16])
        scalars = _floats(row[16:], 24, f"equality row {index}")
        fixed = indices[2] == INVALID_INDEX
        require(indices[0] < nq and indices[1] >= 6 and indices[1] < nv and
                indices[0] == indices[1] + 1,
                f"equality row {index} dependent coordinate identity is invalid")
        require((fixed and indices[3] == INVALID_INDEX) or
                (not fixed and indices[2] < nq and indices[3] >= 6 and
                 indices[3] < nv and indices[2] == indices[3] + 1 and
                 indices[1] != indices[3]),
                f"equality row {index} master coordinate identity is invalid")
        require(indices[1] not in dependent,
                f"equality row {index} duplicates a dependent dof")
        dependent.add(indices[1])
        for word in (11, 14, 15, 21, 22, 23, 26, 27):
            _require_positive_zero_word(row, word, f"equality row {index} reserved word {word}")
        if fixed:
            _require_positive_zero_word(
                row, 25, f"fixed equality row {index} master inverse weight"
            )
        require(scalars[20] > 0.0 and
                ((fixed and scalars[21] == 0.0) or
                 (not fixed and scalars[21] > 0.0)),
                f"equality row {index} inverse weights are invalid")
        require((scalars[8] > 0.0 and scalars[9] > 0.0) or
                (scalars[8] <= 0.0 and scalars[9] <= 0.0),
                f"equality row {index} solref sign pairing is invalid")
        equality_records.append((indices, scalars))
    for index, (indices, _) in enumerate(equality_records):
        require(indices[3] == INVALID_INDEX or indices[3] not in dependent,
                f"equality row {index} forms an unqualified dependent chain")

    for index, (indices, scalars) in enumerate(equality_records):
        witness_bytes = equality_witnesses[index * 32:(index + 1) * 32]
        witness = _floats(witness_bytes, 8, f"equality witness {index}")
        require(struct.unpack("<I", witness_bytes[24:28])[0] == 0x3F800000,
                f"equality witness {index} active flag is not raw +1")
        _require_positive_zero_word(
            witness_bytes, 7, f"equality witness {index} reserved word"
        )
        fixed = indices[2] == INVALID_INDEX
        references0 = scalars[0:4]
        coefficients1 = scalars[4:8]
        solref = scalars[8:12]
        solimp0 = scalars[12:16]
        solimp1 = scalars[16:20]
        inverse_weights = scalars[20:24]
        x = 0.0 if fixed else _f32(
            q0[indices[2]] - references0[1], f"equality row {index} x"
        )
        polynomial = _f32(
            (((coefficients1[2] * x + coefficients1[1]) * x + coefficients1[0])
             * x + references0[3]) * x + references0[2],
            f"equality row {index} polynomial",
        )
        derivative = 0.0 if fixed else _f32(
            ((4.0 * coefficients1[2] * x + 3.0 * coefficients1[1]) * x +
             2.0 * coefficients1[0]) * x + references0[3],
            f"equality row {index} derivative",
        )
        phi = _f32(
            q0[indices[0]] - references0[0] - polynomial,
            f"equality row {index} phi",
        )
        row_delta = _f32(
            delta[indices[1]] -
            (0.0 if fixed else derivative * delta[indices[3]]),
            f"equality row {index} row delta velocity",
        )
        source_velocity = _f32(
            v0[indices[1]] -
            (0.0 if fixed else derivative * v0[indices[3]]),
            f"equality row {index} source velocity",
        )
        free_increment = _f32(
            free[indices[1]] - v0[indices[1]] -
            (0.0 if fixed else derivative * (free[indices[3]] - v0[indices[3]])),
            f"equality row {index} free increment",
        )
        impedance = _scalar_impedance(solimp0, solimp1, phi)
        regularizer = max(
            1.0e-15,
            (1.0 - impedance) / impedance *
            (inverse_weights[0] + inverse_weights[1]),
        )
        inverse_regularizer = _f32(1.0 / regularizer,
                                   f"equality row {index} inverse regularizer")
        _require_close(witness[0], derivative, f"equality witness {index} derivative")
        _require_close(witness[2], inverse_regularizer,
                       f"equality witness {index} inverse regularizer", operations=256)
        _require_close(witness[3], phi, f"equality witness {index} phi")
        _require_close(witness[5], row_delta,
                       f"equality witness {index} row delta velocity")
        _require_close(
            witness[4], _f32((witness[5] - witness[1]) * witness[2],
                             f"equality witness {index} impulse identity"),
            f"equality witness {index} impulse",
        )
        row_flags = _matching_reference_safe_flags(
            witness[1], solref, solimp0, solimp1, phi,
            source_velocity, free_increment, timestep,
        )
        require(row_flags, f"equality witness {index} bDelta matches no v1 policy flag")
        possible_flags &= row_flags

    limit_rows = arrays["limit_rows"]
    limit_witnesses = arrays["limit_reconstructed_linearization_impulses"]
    limited: set[int] = set()
    source_ids: set[int] = set()
    for row_index in range(payload["limit_row_count"]):
        row = limit_rows[row_index * 80:(row_index + 1) * 80]
        indices = struct.unpack("<4I", row[:16])
        scalars = _floats(row[16:], 16, f"limit row {row_index}")
        require(all(value == 0.0 or abs(value) >= FLOAT32_MIN_NORMAL
                    for value in scalars),
                f"limit row {row_index} contains a subnormal scalar")
        require(indices[0] < nq and indices[1] >= 6 and indices[1] < nv and
                indices[0] == indices[1] + 1 and
                indices[2] != INVALID_INDEX and indices[3] == 0,
                f"limit row {row_index} source coordinate identity is invalid")
        require(indices[1] not in limited and indices[2] not in source_ids,
                f"limit row {row_index} duplicates a dof or source identity")
        limited.add(indices[1])
        source_ids.add(indices[2])
        for word in (10, 11, 17, 18, 19):
            _require_positive_zero_word(row, word, f"limit row {row_index} reserved word {word}")
        bounds = scalars[0:4]
        solref = scalars[4:8]
        solimp0 = scalars[8:12]
        solimp1 = scalars[12:16]
        require(bounds[0] <= bounds[1] and bounds[3] >= FLOAT32_MIN_NORMAL,
                f"limit row {row_index} range/inverse weight is invalid")
        require((solref[0] > 0.0 and solref[1] > 0.0) or
                (solref[0] <= 0.0 and solref[1] <= 0.0),
                f"limit row {row_index} solref sign pairing is invalid")
        position = q0[indices[0]]
        for side in range(2):
            witness_index = 2 * row_index + side
            witness_bytes = limit_witnesses[
                witness_index * 32:(witness_index + 1) * 32
            ]
            witness = _floats(witness_bytes, 8, f"limit witness {witness_index}")
            lower = side == 0
            direction = 1.0 if lower else -1.0
            distance = _f32(
                position - bounds[0] if lower else bounds[1] - position,
                f"limit row {row_index} distance",
            )
            active = distance < bounds[2]
            if not active:
                require(witness_bytes == bytes(32),
                        f"inactive limit witness {witness_index} is not raw all +0")
                continue
            expected_direction_bits = 0x3F800000 if lower else 0xBF800000
            require(struct.unpack("<I", witness_bytes[:4])[0] == expected_direction_bits,
                    f"limit witness {witness_index} direction is invalid")
            require(struct.unpack("<I", witness_bytes[24:28])[0] == 0x3F800000,
                    f"limit witness {witness_index} active flag is not raw +1")
            _require_positive_zero_word(
                witness_bytes, 7, f"limit witness {witness_index} reserved word"
            )
            phi = _f32(distance - bounds[2], f"limit witness {witness_index} phi")
            source_velocity = _f32(
                direction * v0[indices[1]],
                f"limit witness {witness_index} source velocity",
            )
            free_increment = _f32(
                direction * (free[indices[1]] - v0[indices[1]]),
                f"limit witness {witness_index} free increment",
            )
            impedance = _scalar_impedance(solimp0, solimp1, phi)
            regularizer = max(
                1.0e-15,
                (1.0 - impedance) / impedance * bounds[3],
            )
            inverse_regularizer = _f32(
                1.0 / regularizer,
                f"limit witness {witness_index} inverse regularizer",
            )
            violation = _f32(
                direction * delta[indices[1]] - witness[1],
                f"limit witness {witness_index} violation identity",
            )
            impulse = _f32(
                min(0.0, violation) * witness[2],
                f"limit witness {witness_index} impulse identity",
            )
            _require_close(witness[2], inverse_regularizer,
                           f"limit witness {witness_index} inverse regularizer",
                           operations=256)
            _require_close(witness[3], phi, f"limit witness {witness_index} phi")
            _require_close(witness[5], violation,
                           f"limit witness {witness_index} violation")
            _require_close(witness[4], impulse,
                           f"limit witness {witness_index} impulse")
            row_flags = _matching_reference_safe_flags(
                witness[1], solref, solimp0, solimp1, phi,
                source_velocity, free_increment, timestep,
            )
            require(row_flags,
                    f"limit witness {witness_index} bDelta matches no v1 policy flag")
            possible_flags &= row_flags

    require(possible_flags,
            "constraint witnesses do not share one omitted v1 reference-safe flag")
    return {
        "equality": _witness_summary(
            equality_witnesses, payload["equality_row_count"], limits=False
        ),
        "limit": _witness_summary(
            limit_witnesses, 2 * payload["limit_row_count"], limits=True
        ),
        "reference_safe_flag_candidates": sorted(possible_flags),
        "independent_bdelta_policy_identity_available": False,
    }


def _audit_dynamics(payload: dict[str, Any], arrays: dict[str, bytes],
                    available: dict[str, bool]) -> dict[str, Any]:
    required = (
        "checkpoint_q", "checkpoint_v", "source_predicted_velocity",
        "source_generalized_force",
        "matter_generalized_reaction", "candidate_v",
        "effective_tangent_factor_storage", "acceleration", "source_rhs",
        "source_bias", "work_energy_components",
        "equality_rows", "limit_rows",
        "equality_reconstructed_linearization_impulses",
        "limit_reconstructed_linearization_impulses",
    )
    for name in required:
        require(available[name], f"required dynamics array {name} is unavailable")
    nv = payload["dof_count"]
    v0 = _floats(arrays["checkpoint_v"], nv, "checkpoint_v")
    free = _floats(arrays["source_predicted_velocity"], nv,
                   "source_predicted_velocity")
    source = _floats(arrays["source_generalized_force"], nv,
                     "source_generalized_force")
    reaction = _floats(arrays["matter_generalized_reaction"], nv,
                       "matter_generalized_reaction")
    candidate = _floats(arrays["candidate_v"], nv, "candidate_v")
    factor = _floats(arrays["effective_tangent_factor_storage"], nv * nv,
                     "effective_tangent_factor_storage")
    tangent_audit = _audit_effective_tangent(factor, nv)
    timestep = float(payload["timestep_nanoseconds"]) * 1.0e-9
    require(math.isfinite(timestep) and timestep > 0.0, "timestep is not finite positive")

    acceleration = [
        _to_f32((free[index] - v0[index]) / timestep,
                f"acceleration[{index}]")
        for index in range(nv)
    ]
    transpose_action = [0.0] * nv
    for column in range(nv):
        value = 0.0
        for row in range(column, nv):
            value += float(factor[row * nv + column]) * acceleration[row]
        transpose_action[column] = value
    rhs64 = [0.0] * nv
    for row in range(nv):
        value = 0.0
        for column in range(row + 1):
            value += float(factor[row * nv + column]) * transpose_action[column]
        rhs64[row] = value
    bias64 = [float(source[row]) - rhs64[row] for row in range(nv)]

    source_work = 0.0
    reaction_work = 0.0
    for index in range(nv):
        source_work += (
            0.5 * timestep * (float(v0[index]) + free[index]) * source[index]
        )
        reaction_work += (
            0.5 * timestep * (float(free[index]) + candidate[index])
            * reaction[index]
        )
    candidate_delta = [float(candidate[index]) - v0[index] for index in range(nv)]
    delta_transpose = [0.0] * nv
    for column in range(nv):
        for row in range(column, nv):
            delta_transpose[column] += (
                float(factor[row * nv + column]) * candidate_delta[row]
            )
    effective_energy = 0.0
    for value in delta_transpose:
        effective_energy += 0.5 * value * value

    stored_acceleration = _compare_f32(
        "acceleration", acceleration, arrays["acceleration"]
    )
    stored_rhs = _compare_f32("source_rhs", rhs64, arrays["source_rhs"])
    _compare_f32("source_bias", bias64, arrays["source_bias"])
    work = _compare_f32(
        "work_energy_components",
        [source_work, reaction_work, effective_energy],
        arrays["work_energy_components"],
    )
    constraints = _audit_constraint_witnesses(payload, arrays)
    return {
        "dominant_dofs": {
            "acceleration": _dominant(stored_acceleration),
            "source_rhs": _dominant(stored_rhs),
            "matter_reaction": _dominant(reaction),
        },
        "effective_tangent": tangent_audit,
        "constraint_witnesses": constraints,
        "work_energy_components": {
            "source_force_midpoint_work": work[0],
            "matter_reaction_midpoint_work": work[1],
            "effective_tangent_delta_v_energy": work[2],
        },
    }


def audit_text(text: str, source: str = "<memory>",
               expected_payload_sha256: str | None = None) -> dict[str, Any]:
    require(isinstance(text, str), "evidence input is not text")
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise AuditError("evidence text is not valid UTF-8") from error
    require(len(encoded) <= MAX_INPUT_BYTES,
            "evidence input exceeds the 1 GiB audit bound")
    envelope = _strict_json(text, "evidence envelope")
    spans = _top_level_spans(text)
    _exact_keys(envelope, {"evidence_schema", "payload_sha256", "payload"},
                "evidence envelope")
    require(envelope["evidence_schema"] == EVIDENCE_SCHEMA,
            "evidence_schema is not persistent production-owner v1")
    digest = envelope["payload_sha256"]
    require(isinstance(digest, str) and SHA256.fullmatch(digest) is not None,
            "payload_sha256 is not lowercase SHA-256")
    start, end = spans["payload"]
    payload_text = text[start:end]
    actual_digest = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
    require(actual_digest == digest,
            "payload_sha256 does not match the exact embedded payload bytes")
    if expected_payload_sha256 is not None:
        require(isinstance(expected_payload_sha256, str) and
                SHA256.fullmatch(expected_payload_sha256) is not None,
                "detached expected payload SHA-256 is not 64-character lowercase hex")
        require(actual_digest == expected_payload_sha256,
                "exact embedded payload does not match detached expected SHA-256")
    payload = _strict_json(payload_text, "embedded payload")
    require(payload == envelope["payload"], "embedded payload parse is inconsistent")
    identities = _validate_payload_header(payload)
    arrays, available = _validate_arrays(payload)
    actual_coverage = _coverage_mask(payload, available)
    coverage = identities["coverage_mask"]
    required = identities["required_coverage_mask"]
    require(identities["truncation_mask"] == 0, "payload truncation_mask is nonzero")
    require(coverage == actual_coverage,
            "payload coverage_mask does not exactly match array/identity availability")
    require((coverage & required) == required,
            "payload coverage_mask does not contain required_coverage_mask")
    dynamics = _audit_dynamics(payload, arrays, available)
    return {
        "audit_schema": AUDIT_SCHEMA,
        "artifact": source,
        "payload_digest": {
            "embedded_self_sha256": digest,
            "detached_expected_sha256": expected_payload_sha256,
            "detached_expected_match": expected_payload_sha256 is not None,
            "authenticated_provenance_claimed": False,
        },
        "payload_schema": payload["schema"],
        "treatment": payload["treatment"],
        "disposition": payload["disposition"],
        "dimensions": {
            "q_coordinates": payload["q_coordinate_count"],
            "dofs": payload["dof_count"],
            "muscles": payload["muscle_count"],
            "equality_rows": payload["equality_row_count"],
            "limit_rows": payload["limit_row_count"],
        },
        "coverage": {
            "mask": f"{coverage:016x}",
            "required_mask": f"{required:016x}",
            "qualified": [
                name for bit, name in enumerate(COVERAGE_NAMES)
                if coverage & (1 << bit)
            ],
        },
        "verified": {
            "exact_embedded_payload_self_digest": True,
            "detached_expected_payload_digest":
                expected_payload_sha256 is not None,
            "schemas": True,
            "zero_truncation": True,
            "coverage_and_shapes": True,
            "little_endian_words": True,
            "positive_cholesky_and_strict_upper_source_a0": True,
            "fp64_dynamics_to_exact_fp32": True,
            "constraint_witness_internal_identities": True,
        },
        "limitations": [
            "The embedded payload digest is a self-integrity check, not authenticated provenance.",
            "A detached digest match proves identity only relative to the separately trusted digest source.",
            "Snapshot v1 does not retain the equality/limit dispatch reference-safe flag; the audit checks the common set of policy values consistent with all witnesses but cannot identify that flag independently.",
            "Snapshot v1 retains source A0 only above the diagonal; its diagonal is reconstructed from L and is not independently cross-checked.",
            "Constraint witnesses are host reconstructions, not device readbacks, and work terms are not physical whole-body energy closure.",
        ],
        **dynamics,
    }


def audit_path(path: Path, expected_payload_sha256: str | None = None) -> dict[str, Any]:
    try:
        # A single bounded read avoids the stat/read race and cannot allocate
        # beyond one byte past the declared input ceiling.
        with path.open("rb") as stream:
            raw = stream.read(MAX_INPUT_BYTES + 1)
    except (OSError, MemoryError) as error:
        raise AuditError(f"cannot read evidence file {path}: {error}") from error
    require(len(raw) <= MAX_INPUT_BYTES,
            "evidence file exceeds the 1 GiB audit bound")
    require(raw, "evidence file is empty")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AuditError("evidence file is not UTF-8") from error
    return audit_text(text, str(path), expected_payload_sha256)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-closed audit of one Numi Human production-owner snapshot"
    )
    parser.add_argument("evidence", type=Path, help="snapshot evidence JSON file")
    parser.add_argument(
        "--expected-payload-sha256",
        help="detached trusted SHA-256 of the literal embedded payload bytes",
    )
    parser.add_argument("--pretty", action="store_true", help="pretty-print audit JSON")
    arguments = parser.parse_args(argv)
    try:
        report = audit_path(
            arguments.evidence, arguments.expected_payload_sha256
        )
        print(json.dumps(
            report,
            indent=2 if arguments.pretty else None,
            sort_keys=True,
            separators=None if arguments.pretty else (",", ":"),
            allow_nan=False,
        ))
        return 0
    except AuditError as error:
        print(f"audit_numi_human_production_owner_snapshot: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
