#!/usr/bin/env python3
"""Verify the source binding and bounded full-arena capture contract for 087."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import unittest

HERE = Path(__file__).resolve().parent
BASE = Path(
    "/Users/home/cardboard-implementation-20261014/snapshots/"
    "source-captured-point-and-contact-scope-086/matter/src/runtime.mm"
)
FIXTURE = Path(
    "/Users/home/cardboard-implementation-20261014/snapshots/"
    "source-captured-point-and-contact-scope-086/matter/tools/"
    "cardboard_full_contact_armijo_check.mm"
)
MANIFEST = HERE / "overlay.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OverlayTests(unittest.TestCase):
    def test_frozen_source_bindings(self):
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual(sha(BASE), manifest["base_runtime_sha256"])
        self.assertEqual(sha(FIXTURE), manifest["fixture_sha256"])
        self.assertEqual(sha(HERE / "runtime.mm"), manifest["overlay_runtime_sha256"])

    def test_runtime_keeps_fixed_payload_bound_and_checks_used_prefix(self):
        source = (HERE / "runtime.mm").read_text()
        self.assertIn("32u * 1024u * 1024u", source)
        self.assertIn("captureArrayCap = 192u", source)
        self.assertNotIn("deformableContactCapacity > 16384u", source)
        self.assertIn("minimum_full_contact_arena_bytes", source)
        self.assertIn("requested_present_payload_bytes", source)
        self.assertIn("active_prefix_count_range_uniqueness_and_valid_bit_scan", source)
        self.assertIn("valid_record_count_per_environment", source)
        self.assertIn("unused_valid_record_count_per_environment", source)
        self.assertNotIn("validRecordCount != activeCount", source)

    def test_authored_fixture_capacity_and_contact_bytes(self):
        text = FIXTURE.read_text()
        self.assertIn("pad.tetrahedra.size() == 48u", text)
        self.assertIn("authoredFaceSlots =", text)
        self.assertIn("authoredFaceSlots * (authoredFaceSlots - 1u) / 2u", text)
        capacity = (4 * 48) * ((4 * 48) - 1) // 2
        contact_slot_bytes = 432 + 16 + 2 * 48 + 2 * 4 + 6 * 4
        env_overhead = 3 * 16 + 36 * 16 + 16
        self.assertEqual(capacity, 18_336)
        self.assertEqual(capacity * contact_slot_bytes, 10_561_536)
        self.assertEqual(capacity * contact_slot_bytes + env_overhead, 10_562_176)
        self.assertLess(10_562_176, 32 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
