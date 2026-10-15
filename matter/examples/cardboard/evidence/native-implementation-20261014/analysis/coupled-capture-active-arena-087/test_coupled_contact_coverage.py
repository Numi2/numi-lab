#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from coupled_contact_coverage import CaptureError, validate_capture

CAP = 32 * 1024 * 1024
C = 4
E = 1


def make_capture(root: Path, *, active=(1, 3), truncate=False,
                 truncate_indices=False, duplicate=False,
                 wrong_prefix_count=False, stale_valid=False,
                 wrong_buffer_count=False) -> Path:
    arrays = []
    payload = bytearray()

    def add(name, dtype, elem, raw):
        offset = len(payload)
        payload.extend(raw)
        arrays.append({
            "name": name, "dtype": dtype, "element_bytes": elem,
            "count": len(raw) // elem, "offset": offset,
            "byte_count": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        })

    contacts = bytearray(432 * C)
    for slot in active:
        struct.pack_into("<I", contacts, slot * 432 + 28, 1)
    if stale_valid:
        struct.pack_into("<I", contacts, 28, 1)
    add("deformable_contacts", "record:NMDeformableContactGPU", 432,
        contacts[:432 * (C - 1)] if truncate else contacts)
    add("deformable_contact_candidates", "record:NMDeformableContactCandidateGPU",
        16, bytes(16 * C))
    add("deformable_contact_candidate_counts", "uint32x4", 16,
        bytes(16))
    add("deformable_contact_histories_accepted",
        "record:NMDeformableContactHistoryGPU", 48, bytes(48 * C))
    add("deformable_contact_histories_candidate",
        "record:NMDeformableContactHistoryGPU", 48, bytes(48 * C))
    indices = list(active)
    if duplicate:
        indices = [indices[0], indices[0]]
    if truncate_indices and indices:
        indices[0] = 0xFFFFFFFF
    indices = indices + [0xFFFFFFFF] * (C - len(indices))
    add("deformable_contact_active_indices", "uint32", 4,
        struct.pack("<" + "I" * C, *indices))
    count = len(active)
    add("deformable_contact_active_counts", "uint32x4", 16,
        struct.pack("<4I", count + (1 if wrong_buffer_count else 0), 0, 0, 0))
    add("deformable_contact_node_incidence", "uint32", 4,
        bytes(24 * C))
    add("deformable_contact_node_ranges", "record:NMIncidenceRangeGPU", 16,
        bytes(16))
    add("deformable_contact_active_offsets", "uint32", 4,
        struct.pack("<I", 0))
    add("deformable_contact_global_active_indices", "uint32", 4,
        bytes(4 * C))
    add("deformable_contact_active_dispatch", "record:NMIndirectDispatchGPU", 16,
        bytes(16))

    binary = root / "capture.coupled.bin"
    binary.write_bytes(payload)
    manifest = {
        "schema": "numi.matter.coupled_newton_capture.v1",
        "byte_order": "little_endian",
        "payload": {"file": binary.name, "byte_count": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest()},
        "arrays": arrays,
        "environment_count": E,
        "node_count": 1,
        "grid_node_count": 0,
        "deformable_contact_capacity": C,
        "deformable_contact_capture": {
            "mode": "full_capacity_arenas",
            "capacity_per_environment": C,
            "captured_contact_records_per_environment": [C - 1 if truncate else C],
            "captured_candidate_records_per_environment": [C],
            "captured_accepted_history_records_per_environment": [C],
            "captured_candidate_history_records_per_environment": [C],
            "captured_incidence_entries_per_environment": [6 * C],
            "active_count_per_environment": [count],
            "active_index_valid_prefix_count_per_environment": [
                count + (1 if wrong_prefix_count else 0)],
            "valid_record_count_per_environment": [
                len(set(active)) + (1 if stale_valid and 0 not in active else 0)],
            "unused_valid_record_count_per_environment": [
                len(set(active)) + (1 if stale_valid and 0 not in active else 0) - count],
            "valid_records_equal_active_count_per_environment": [
                len(set(active)) + (1 if stale_valid and 0 not in active else 0) == count],
            "active_list_verified_against_valid_flags": True,
            "truncated": truncate,
            "complete": not truncate,
        },
    }
    path = root / "capture.coupled.json"
    path.write_text(json.dumps(manifest))
    return path


class ContactCoverageTests(unittest.TestCase):
    def test_full_arena_and_active_prefix_pass(self):
        with tempfile.TemporaryDirectory() as td:
            path = make_capture(Path(td))
            result = validate_capture(path)
            self.assertTrue(result["complete"])
            self.assertEqual(result["active_counts"], [2])
            self.assertEqual(result["valid_slots"], 2)

    def test_truncated_record_arena_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(CaptureError, "incomplete or truncated"):
                validate_capture(make_capture(Path(td), truncate=True))

    def test_truncated_active_prefix_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(CaptureError, "out of range or duplicated"):
                validate_capture(make_capture(Path(td), truncate_indices=True))

    def test_duplicate_active_index_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(CaptureError, "duplicated"):
                validate_capture(make_capture(Path(td), duplicate=True))

    def test_active_count_and_prefix_metadata_disagreement_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(CaptureError, "active count and captured prefix"):
                validate_capture(make_capture(Path(td), wrong_prefix_count=True))

    def test_native_active_count_buffer_must_match_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(CaptureError, "active-count buffer disagrees"):
                validate_capture(make_capture(Path(td), wrong_buffer_count=True))

    def test_unused_valid_full_arena_record_is_reported_not_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            result = validate_capture(make_capture(Path(td), stale_valid=True))
            self.assertTrue(result["complete"])
            self.assertEqual(result["active_counts"], [2])
            self.assertEqual(result["valid_slots"], 3)
            self.assertEqual(result["unused_valid_slots"], 1)
            self.assertFalse(result["valid_records_equal_active_count"])

    def test_expected_fixture_full_arena_fits_without_truncation(self):
        # sizeof-derived records: contact 432, candidate 16, two histories
        # 2*48, active+global indices 2*4, six-node incidence 6*4.
        capacity = 18_336
        full_contact_bytes = capacity * (432 + 16 + 2 * 48 + 2 * 4 + 6 * 4)
        small_env_records = 16 + 16 + 16 + 36 * 16 + 16
        self.assertEqual(full_contact_bytes, 10_561_536)
        self.assertEqual(full_contact_bytes + small_env_records, 10_562_176)
        self.assertLess(full_contact_bytes + small_env_records, CAP)
        self.assertGreater(65_536 * 576, CAP)


if __name__ == "__main__":
    unittest.main()
