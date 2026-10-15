#!/usr/bin/env python3
"""Fail-closed validator for full-capacity deformable contact captures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any

CAP_BYTES = 32 * 1024 * 1024
CONTACT_RECORD_BYTES = 432
CANDIDATE_RECORD_BYTES = 16
HISTORY_RECORD_BYTES = 48
INCIDENCE_RANGE_BYTES = 16
VALID_FLAG = 1


class CaptureError(ValueError):
    pass


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise CaptureError(message)


def _sha(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_capture(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as exc:
        raise CaptureError(f"invalid manifest JSON: {exc}") from exc
    _require(manifest.get("schema") == "numi.matter.coupled_newton_capture.v1",
             "unsupported capture schema")
    _require(manifest.get("byte_order") == "little_endian",
             "capture byte order is not explicit little endian")
    payload_desc = manifest.get("payload")
    _require(isinstance(payload_desc, dict), "payload descriptor missing")
    byte_count = payload_desc.get("byte_count")
    _require(isinstance(byte_count, int) and 0 <= byte_count <= CAP_BYTES,
             "payload is outside the fixed 32 MiB bound")
    payload_path = manifest_path.parent / Path(str(payload_desc.get("file", ""))).name
    _require(payload_path.is_file(), "payload file is missing")
    payload = payload_path.read_bytes()
    _require(len(payload) == byte_count, "payload extent differs from manifest")
    _require(_sha(payload) == payload_desc.get("sha256"), "payload hash mismatch")

    arrays_raw = manifest.get("arrays")
    _require(isinstance(arrays_raw, list), "array table missing")
    arrays: dict[str, dict[str, Any]] = {}
    spans: list[tuple[int, int, str]] = []
    for row in arrays_raw:
        _require(isinstance(row, dict), "array descriptor is malformed")
        name = row.get("name")
        _require(isinstance(name, str) and name and name not in arrays,
                 "array name is missing or duplicated")
        count, elem, offset, size = (row.get(k) for k in
                                     ("count", "element_bytes", "offset", "byte_count"))
        _require(all(isinstance(v, int) and v >= 0
                     for v in (count, elem, offset, size)) and elem > 0,
                 f"invalid extent for {name}")
        _require(count * elem == size, f"count/size mismatch for {name}")
        end = offset + size
        _require(end <= len(payload), f"{name} exceeds payload")
        view = memoryview(payload)[offset:end]
        _require(_sha(view) == row.get("sha256"), f"{name} hash mismatch")
        arrays[name] = row
        spans.append((offset, end, name))
    cursor = 0
    for start, end, name in sorted(spans):
        _require(start == cursor, f"payload gap or overlap at {name}")
        cursor = end
    _require(cursor == len(payload), "unclaimed payload bytes")

    coverage = manifest.get("deformable_contact_capture")
    _require(isinstance(coverage, dict), "contact coverage proof missing")
    capacity = manifest.get("deformable_contact_capacity")
    environments = manifest.get("environment_count")
    _require(isinstance(capacity, int) and capacity >= 0,
             "invalid deformable capacity")
    _require(isinstance(environments, int) and environments == 1,
             "capture checker supports only the authored one-environment scope")
    if capacity == 0:
        _require(coverage.get("mode") == "no_deformable_contact_arena" and
                 coverage.get("complete") is True and
                 coverage.get("truncated") is False,
                 "empty-arena proof is malformed")
        return {"complete": True, "capacity": 0, "active_counts": []}

    _require(coverage.get("mode") == "full_capacity_arenas",
             "partial or unknown contact arena mode")
    _require(coverage.get("complete") is True and
             coverage.get("truncated") is False,
             "capture is marked incomplete or truncated")
    _require(coverage.get("active_list_verified_against_valid_flags") is True,
             "native active-prefix verification is absent")
    captured_records = coverage.get("captured_contact_records_per_environment")
    _require(captured_records == [capacity],
             "full contact record arena was not captured")
    for field, expected in (
        ("captured_candidate_records_per_environment", [capacity]),
        ("captured_accepted_history_records_per_environment", [capacity]),
        ("captured_candidate_history_records_per_environment", [capacity]),
        ("captured_incidence_entries_per_environment", [6 * capacity]),
    ):
        _require(coverage.get(field) == expected,
                 f"full arena coverage metadata mismatch: {field}")
    contacts = arrays.get("deformable_contacts")
    candidates = arrays.get("deformable_contact_candidates")
    accepted = arrays.get("deformable_contact_histories_accepted")
    candidate_history = arrays.get("deformable_contact_histories_candidate")
    indices = arrays.get("deformable_contact_active_indices")
    counts = arrays.get("deformable_contact_active_counts")
    candidate_counts = arrays.get("deformable_contact_candidate_counts")
    incidence = arrays.get("deformable_contact_node_incidence")
    ranges = arrays.get("deformable_contact_node_ranges")
    offsets = arrays.get("deformable_contact_active_offsets")
    global_indices = arrays.get("deformable_contact_global_active_indices")
    dispatch = arrays.get("deformable_contact_active_dispatch")
    required = {
        "deformable_contacts": (contacts, CONTACT_RECORD_BYTES, capacity),
        "deformable_contact_candidates": (candidates, CANDIDATE_RECORD_BYTES, capacity),
        "deformable_contact_histories_accepted": (accepted, HISTORY_RECORD_BYTES, capacity),
        "deformable_contact_histories_candidate": (candidate_history, HISTORY_RECORD_BYTES, capacity),
        "deformable_contact_active_indices": (indices, 4, capacity),
        "deformable_contact_node_incidence": (incidence, 4, 6 * capacity),
        "deformable_contact_node_ranges": (
            ranges, INCIDENCE_RANGE_BYTES,
            manifest.get("grid_node_count", 0) + manifest.get("node_count", 0)),
        "deformable_contact_global_active_indices": (global_indices, 4, capacity),
    }
    for name, (row, width, minimum_count) in required.items():
        _require(row is not None, f"required arena {name} missing")
        _require(row.get("element_bytes") == width and
                 row.get("count", -1) >= minimum_count,
                 f"required arena {name} is truncated or has wrong record width")
    for name, row in (("deformable_contact_active_counts", counts),
                      ("deformable_contact_candidate_counts", candidate_counts)):
        _require(row is not None and row.get("element_bytes") == 16 and
                 row.get("byte_count", 0) >= 4,
                 f"required scalar-count arena {name} missing or short")
    _require(offsets is not None and offsets.get("byte_count", 0) >= 4,
             "active offsets are missing or short")
    _require(dispatch is not None and dispatch.get("byte_count", 0) >= 16,
             "active dispatch record is missing or short")

    active_list_counts = coverage.get("active_index_valid_prefix_count_per_environment")
    _require(isinstance(active_list_counts, list) and len(active_list_counts) == environments,
             "active-index prefix counts missing")
    active_counts = coverage.get("active_count_per_environment")
    _require(active_counts == active_list_counts,
             "active count and captured prefix length disagree")

    def raw(name: str) -> memoryview:
        row = arrays[name]
        return memoryview(payload)[row["offset"]:row["offset"] + row["byte_count"]]

    active_count = active_counts[0]
    _require(isinstance(active_count, int) and 0 <= active_count <= capacity,
             "active count is outside arena capacity")
    active_count_word = struct.unpack_from(
        "<I", memoryview(payload)[counts["offset"]:counts["offset"] + 4], 0
    )[0]
    _require(active_count_word == active_count,
             "captured active-count buffer disagrees with manifest")
    raw_indices = raw("deformable_contact_active_indices")
    raw_contacts = raw("deformable_contacts")
    seen: set[int] = set()
    for i in range(active_count):
        slot = struct.unpack_from("<I", raw_indices, 4 * i)[0]
        _require(slot < capacity and slot not in seen,
                 "active-index prefix is out of range or duplicated")
        seen.add(slot)
        flags = struct.unpack_from("<I", raw_contacts,
                                   slot * CONTACT_RECORD_BYTES + 28)[0]
        _require(flags & VALID_FLAG, "active index points to an invalid contact record")
    valid_slots = {
        slot for slot in range(capacity)
        if struct.unpack_from("<I", raw_contacts,
                              slot * CONTACT_RECORD_BYTES + 28)[0] & VALID_FLAG
    }
    _require(seen.issubset(valid_slots),
             "active prefix references a record without the valid bit")
    valid_counts = coverage.get("valid_record_count_per_environment")
    unused_counts = coverage.get("unused_valid_record_count_per_environment")
    equal_counts = coverage.get("valid_records_equal_active_count_per_environment")
    _require(valid_counts == [len(valid_slots)],
             "full-arena valid-record count does not match captured records")
    _require(unused_counts == [len(valid_slots) - active_count],
             "unused valid-record count is inconsistent")
    _require(equal_counts == [len(valid_slots) == active_count],
             "active/full valid-record comparison metadata is inconsistent")
    return {
        "complete": True,
        "capacity": capacity,
        "active_counts": active_counts,
        "valid_slots": len(valid_slots),
        "unused_valid_slots": len(valid_slots) - active_count,
        "valid_records_equal_active_count": len(valid_slots) == active_count,
        "payload_bytes": len(payload),
        "payload_sha256": _sha(payload),
    }
