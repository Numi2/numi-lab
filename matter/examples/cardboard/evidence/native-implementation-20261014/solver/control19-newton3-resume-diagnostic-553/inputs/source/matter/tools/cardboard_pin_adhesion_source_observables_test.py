#!/usr/bin/env python3
"""Contract checks for the published pin-adhesion source observable extractor."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


TOOL_PATH = Path(__file__).with_name("cardboard_pin_adhesion_source_observables.py")
SPEC = importlib.util.spec_from_file_location("cardboard_pin_adhesion_source_observables", TOOL_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class PinSourceObservableTests(unittest.TestCase):
    def test_source_observables_are_bounded_and_not_calibration(self) -> None:
        report = module.analyze()
        low = report["presentation_curve_observables"]["low_adhesion_white_glue_lines"]
        high = report["presentation_curve_observables"]["applicator_roll_too_fast"]

        self.assertEqual(report["status"], "source_observables_extracted_not_calibration")
        self.assertEqual(low["visible_sampled_peak"]["sampled_centerline_estimate"], 78.0)
        self.assertEqual(low["visible_sampled_peak"]["digitization_interval"], [68.0, 88.0])
        secant = low["first_to_peak_visible_secant"]
        self.assertAlmostEqual(secant["estimate"], 32.0)
        self.assertAlmostEqual(secant["interval"][0], 22.89156626506)
        self.assertAlmostEqual(secant["interval"][1], 43.28358208955)
        area = low["visible_window_trapezoid_area"]
        self.assertAlmostEqual(area["estimate"], 76.625)
        self.assertLessEqual(area["interval"][0], area["estimate"])
        self.assertGreaterEqual(area["interval"][1], area["estimate"])
        self.assertEqual(area["unit"], "N*mm")
        self.assertEqual(high["global_peak"]["status"], "unbounded_from_visible_trace; later branch is obscured")
        self.assertAlmostEqual(high["first_to_last_visible_branch_secant"]["estimate"], 185.5555555556)
        self.assertLess(high["first_to_last_visible_branch_secant"]["interval"][0],
                        high["first_to_last_visible_branch_secant"]["estimate"])
        self.assertGreater(high["first_to_last_visible_branch_secant"]["interval"][1],
                           high["first_to_last_visible_branch_secant"]["estimate"])
        self.assertEqual(report["presentation_curve_observables"]["failure_locus"]["status"], "unavailable_in_source")
        self.assertEqual(report["article_narrative_summary"]["reported_pin_strength_range_N_per_mm"], [0.17, 0.8])
        self.assertEqual(report["slide16_summary_markers"]["strength_center_range_N_per_mm"], [0.13, 0.82])
        self.assertFalse(report["engineering_profile"]["source_equivalent"])
        self.assertFalse(report["engineering_profile"]["published_nuisance_bounds_available"])
        self.assertEqual(report["identifiability"]["calibration_decision"],
                         "blocked; retain user-supplied feasible parameter sets and report sensitivity only")
        self.assertNotIn("feasible_parameter_sets", report)

    def test_catalog_hash_mismatch_is_rejected(self) -> None:
        catalog = json.loads(module.DEFAULT_CATALOG.read_text(encoding="utf-8"))
        record = next(row for row in catalog["datasets"] if row["id"] == module.FORCE_DATASET)
        record["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.json"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            with self.assertRaises(module.PinObservableError):
                module.analyze(path, module.DEFAULT_DATA)

    def test_article_cannot_be_promoted_to_direct_slide_source(self) -> None:
        catalog = json.loads(module.DEFAULT_CATALOG.read_text(encoding="utf-8"))
        record = next(row for row in catalog["datasets"] if row["id"] == module.FORCE_DATASET)
        record["source_ids"] = [module.PLOT_SOURCE_ID, module.PROTOCOL_SOURCE_ID]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.json"
            path.write_text(json.dumps(catalog), encoding="utf-8")
            with self.assertRaises(module.PinObservableError):
                module.analyze(path, module.DEFAULT_DATA)


if __name__ == "__main__":
    unittest.main()
