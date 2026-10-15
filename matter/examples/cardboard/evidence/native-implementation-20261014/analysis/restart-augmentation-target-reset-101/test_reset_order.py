#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path("/Users/home/cardboard-implementation-20261014")
PACKAGE = ROOT / "analysis/restart-augmentation-target-reset-101"
BASE = ROOT / "snapshots/source-exact-tiling-inactive-fields-100/matter/src/runtime.mm"
PATCHED = PACKAGE / "runtime.mm"

spec = importlib.util.spec_from_file_location("check_reset_order", PACKAGE / "check_reset_order.py")
assert spec is not None and spec.loader is not None
check_reset_order = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_reset_order)


class ResetOrderTests(unittest.TestCase):
    def test_patched_source_resets_before_rap_and_scale_capture(self):
        report = check_reset_order.check(PATCHED, check_reset_order.PATCHED_SHA256)
        self.assertTrue(report["reset_precedes_target_capture"])
        self.assertEqual(report["dispatch_order"], [
            "reset_metadata", "project_basis", "build_rap", "capture_restart_scale"])
        self.assertEqual(len(report["reset_buffers"]), 7)

    def test_frozen_base_retains_reset_after_scale_capture(self):
        with self.assertRaisesRegex(ValueError, "unsafe dispatch order"):
            check_reset_order.check(BASE, check_reset_order.BASE_SHA256)


if __name__ == "__main__":
    unittest.main()
