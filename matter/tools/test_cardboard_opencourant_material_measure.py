#!/usr/bin/env python3
"""Focused checks for the source-fit material coupon collector."""
from __future__ import annotations

import math
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cardboard_opencourant_material_measure import (  # noqa: E402
    main,
    _prescribed_displacement,
    _validate_history,
    nominal_axial_stress,
    source_fit_at_engineering_strain,
)


class MaterialMeasureTests(unittest.TestCase):
    def test_cli_selects_run_prefix_and_case_suite_without_changing_default_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "measurement-07.json"
            report = {"schema": "test", "coupons": []}
            with patch("cardboard_opencourant_material_measure.collect_study", return_value=report) as collect:
                result = main([
                    "--study-dir", str(root), "--output", str(output),
                    "--run-prefix", "paper-predictor-07-endpoint-window-law112-",
                    "--case-suite-dir", "cases-material-transfer-07-endpoint-window",
                ])
            self.assertEqual(result, 0)
            collect.assert_called_once_with(
                root, run_prefix="paper-predictor-07-endpoint-window-law112-",
                case_suite_dir="cases-material-transfer-07-endpoint-window",
            )
            self.assertEqual(json.loads(output.read_text()), report)

    def test_nominal_axial_traction_uses_deformed_area_and_shear(self) -> None:
        # Sigma_xx=100, sigma_xy=10, and a slightly tilted current area vector
        # produce a 98.2 MPa reference traction after division by A0=100.
        measured = nominal_axial_stress(
            (100.0, 50.0, 20.0, 10.0, 0.0, 5.0),
            (98.0, 2.0, 0.0),
            reference_area=100.0,
            axis=0,
        )
        self.assertAlmostEqual(measured, 98.2, places=12)

    def test_fit_line_and_pixel_envelope_use_uniaxial_H_conversion(self) -> None:
        curve = {
            "elastic_modulus_MPa": 1000.0,
            "yield_anchor": {"stress_MPa": 10.0, "strain_mm_per_mm": 0.01},
            "plastic_hardening_modulus_MPa": 100.0,
            "plastic_hardening_modulus_pixel_bound_MPa": [80.0, 120.0],
        }
        result = source_fit_at_engineering_strain(curve, 0.02)
        self.assertAlmostEqual(result["central_MPa"], 10.0 + (1000.0 * 100.0 / 1100.0) * 0.01)
        self.assertLess(result["pixel_bound_low_MPa"], result["central_MPa"])
        self.assertGreater(result["pixel_bound_high_MPa"], result["central_MPa"])
        at_yield = source_fit_at_engineering_strain(curve, 0.01)
        self.assertEqual(at_yield["central_MPa"], 10.0)

    def test_history_requires_t0_monotonicity_and_requested_endpoint(self) -> None:
        report = _validate_history([0.0, 0.1, 0.5], 0.5)
        self.assertTrue(report["reaches_requested_horizon_within_one_sample_interval"])
        self.assertTrue(report["exact_requested_endpoint_frame"])
        self.assertEqual(report["sample_count"], 3)
        near_but_not_endpoint = _validate_history([0.0, 0.2, 0.4], 0.5)
        self.assertTrue(near_but_not_endpoint["reaches_requested_horizon_within_one_sample_interval"])
        self.assertFalse(near_but_not_endpoint["exact_requested_endpoint_frame"])
        for invalid in ([0.1, 0.3, 0.5], [0.0, 0.2, 0.2, 0.5], [0.0, 0.1, 0.2]):
            with self.subTest(times=invalid), self.assertRaises(ValueError):
                _validate_history(list(invalid), 0.5)

    def test_nonfinite_source_curve_is_rejected(self) -> None:
        curve = {
            "elastic_modulus_MPa": math.inf,
            "yield_anchor": {"stress_MPa": 10.0, "strain_mm_per_mm": 0.01},
            "plastic_hardening_modulus_MPa": 100.0,
            "plastic_hardening_modulus_pixel_bound_MPa": [80.0, 120.0],
        }
        with self.assertRaises(ValueError):
            source_fit_at_engineering_strain(curve, 0.02)

    def test_deck_motion_integrates_velocity_at_actual_history_time(self) -> None:
        drive = {
            "active_start_s": 0.0,
            "active_stop_s": 2.0,
            "abscissa_scale": 1.0,
            "velocity_scale_mm_s": 0.1,
            "function_knots": [(0.0, 1.0), (2.0, 1.0)],
        }
        self.assertAlmostEqual(_prescribed_displacement(drive, 1.25), 0.125)
        self.assertAlmostEqual(_prescribed_displacement(drive, 3.0), 0.2)


if __name__ == "__main__":
    unittest.main()
