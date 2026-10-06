"""Synthetic schema/regression tests for the crease evidence analyzer.

These fixtures exercise analysis logic only. They are not Matter runs or
cardboard measurements.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import cardboard_tooling_analysis as analysis


class CardboardToolingAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp.name) / "synthetic-tooling-fixture"
        self.run_dir.mkdir()
        digest = "a" * 64
        self.manifest = {
            "schema": analysis.MANIFEST_SCHEMA,
            "compiled_world_fingerprint": 42,
            "solver": {
                "dt_s": 2.5e-5,
                "contact_slop_m": 1.91e-6,
                "steps": 2,
                "hold_steps": 0,
                "unload_steps": 0,
                "relax_steps": 0,
                "release_steps": 0,
                "relative_residual_tolerance": 1.0e-4,
                "volume_tolerance": 1.0e-4,
                "pressure_tolerance": 1.0e-4,
            },
            "geometry": {
                "length_m": 0.02,
                "width_m": 0.002,
                "total_height_m": 0.005,
            },
            "tooling": {
                "enabled": True,
                "timing": "start-of-step body pose plus consistent velocity; commanded end pose for independent gap audit",
                "indentation_m": 2.0e-5,
                "punch_radius_m": 7.5e-4,
                "initial_nose_clearance_m": 1.0e-5,
                "initial_anvil_clearance_m": 1.0e-5,
            },
            "materials": {
                "liner_path": "missing-liner.nmatter",
                "liner_sha256": digest,
                "medium_path": "missing-medium.nmatter",
                "medium_sha256": digest,
                "glue_path": "",
                "glue_sha256": "",
                "regional_map_sha256": digest,
                "frame_map_sha256": digest,
            },
        }
        self.result = {
            "schema": analysis.RESULT_SCHEMA,
            "status": "completed",
            "accepted_steps": 2,
            "compiled_world_fingerprint": 42,
        }
        self._write_json("manifest.json", self.manifest)
        self._write_json("result.json", self.result)
        # Coordinates bound the expected FP32 allowance to about 19 nm.
        self._write_json("mesh.json", {
            "schema": "numi.cardboard.authored-mesh.v1",
            "units": "metres",
            "nodes_m": [[0.0, 0.0, 0.0], [0.02, 0.002, 0.005]],
        })
        self.observations = []
        self.tool_rows = []
        for step, travel in enumerate((1.0e-5, 2.0e-5)):
            for environment in (0, 1):
                self.observations.append({
                    "step": step,
                    "time_s": (step + 1) * 2.5e-5,
                    "environment": environment,
                    "arm": "indented" if environment == 0 else "stationary_tool_reference",
                    "target_angle_deg": 0,
                    "status_code": 0,
                    "step_accepted": 1,
                    "certificate_raw_accepted_flag": 1,
                    "certificate_residual": 1.0e-5,
                    "certificate_correction": 1.0e-5,
                    "certificate_volume": 1.0e-5,
                    "certificate_pressure": 1.0e-5,
                })
                self.tool_rows.append({
                    "step": step,
                    "environment": environment,
                    "commanded_end_travel_m": travel if environment == 0 else 0.0,
                    "commanded_speed_m_s": -0.4 if environment == 0 else 0.0,
                    "punch_force_z_N": 0.5 if environment == 0 and step == 1 else 0.0,
                    "punch_contact_count": 1 if environment == 0 and step == 1 else 0,
                    "min_end_punch_node_gap_m": 2.0e-6,
                    "min_anvil_node_gap_m": 2.0e-6,
                    "step_accepted": 1,
                })
        self._write_csv("observations.csv", self.observations)
        self._write_csv("tool-observations.csv", self.tool_rows)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_json(self, name: str, value: dict) -> None:
        (self.run_dir / name).write_text(json.dumps(value), encoding="utf-8")

    def _write_csv(self, name: str, rows: list[dict]) -> None:
        path = self.run_dir / name
        with path.open("w", encoding="utf-8", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _rewrite_tools(self) -> None:
        self._write_csv("tool-observations.csv", self.tool_rows)

    def test_complete_instrument_fixture_passes_without_qualification(self) -> None:
        report = analysis.analyze_run(self.run_dir)
        self.assertEqual(report["verdict"], "instrument_checks_passed")
        self.assertEqual(report["exit_code"], 0)
        self.assertFalse(report["fold_qualification"])
        self.assertFalse(report["physical_validation"])
        self.assertEqual(report["summary"]["verified_accepted_step_node_gaps"]["negative_gap_count"], 0)
        self.assertEqual(report["summary"]["verified_accepted_indented_punch_force_receipt"]["punch_contact_count_total"], 1)

    def test_no_moving_arm_native_contact_fails(self) -> None:
        for row in self.tool_rows:
            if row["environment"] == 0:
                row["punch_contact_count"] = 0
                row["punch_force_z_N"] = 0.0
        self._rewrite_tools()
        report = analysis.analyze_run(self.run_dir)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["checks"]["moving_arm_native_punch_contact"], "fail")

    def test_stationary_control_contact_fails(self) -> None:
        self.tool_rows[1]["punch_contact_count"] = 1
        self.tool_rows[1]["punch_force_z_N"] = 0.25
        self._rewrite_tools()
        report = analysis.analyze_run(self.run_dir)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["checks"]["stationary_control_punch_contact"], "fail")

    def test_nonfinite_measurement_fails_without_becoming_missing_row(self) -> None:
        self.tool_rows[0]["punch_force_z_N"] = "nan"
        self._rewrite_tools()
        report = analysis.analyze_run(self.run_dir)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["coverage"]["tool_observations_rows"], 4)
        self.assertEqual(report["checks"]["finite_measurements"], "fail")

    def test_commanded_end_penetration_fails_beyond_fp32_roundoff(self) -> None:
        self.tool_rows[2]["min_end_punch_node_gap_m"] = -2.0e-6
        self._rewrite_tools()
        report = analysis.analyze_run(self.run_dir)
        self.assertEqual(report["verdict"], "failed")
        summary = report["summary"]["verified_accepted_step_node_gaps"]
        self.assertEqual(summary["negative_gap_count"], 1)
        self.assertEqual(summary["first_fp32_roundoff_violation"]["step"], 1)
        self.assertEqual(summary["first_fp32_roundoff_violation"]["environment"], 0)
        self.assertLess(summary["minimum_node_gap_m"], -report["summary"]["minimum_gap_roundoff_tolerance_m"])
        self.assertEqual(report["checks"]["commanded_end_pose_node_nonpenetration"], "fail")

    def test_small_negative_gap_within_roundoff_allowance_is_reported(self) -> None:
        self.tool_rows[0]["min_end_punch_node_gap_m"] = -1.0e-9
        self._rewrite_tools()
        report = analysis.analyze_run(self.run_dir)
        self.assertEqual(report["verdict"], "instrument_checks_passed")
        gaps = report["summary"]["verified_accepted_step_node_gaps"]
        self.assertEqual(gaps["negative_gap_count"], 1)
        self.assertIsNone(gaps["first_fp32_roundoff_violation"])

    def test_travel_curve_discontinuity_fails(self) -> None:
        self.tool_rows[2]["commanded_end_travel_m"] = 4.0e-5
        self._rewrite_tools()
        report = analysis.analyze_run(self.run_dir)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["checks"]["command_travel_matches_manifest_and_velocity"], "fail")

    def test_rejected_or_incomplete_run_is_inconclusive(self) -> None:
        self.result["status"] = "failed"
        self.result["accepted_steps"] = 1
        self._write_json("result.json", self.result)
        rejected_obs = next(row for row in self.observations
                            if row["step"] == 1 and row["environment"] == 0)
        rejected_obs["status_code"] = 6
        rejected_obs["step_accepted"] = 0
        self._write_csv("observations.csv", self.observations)
        rejected_tool = next(row for row in self.tool_rows
                             if row["step"] == 1 and row["environment"] == 0)
        rejected_tool["step_accepted"] = 0
        rejected_tool["min_end_punch_node_gap_m"] = -2.0e-6
        self._rewrite_tools()
        report = analysis.analyze_run(self.run_dir)
        self.assertEqual(report["verdict"], "inconclusive")
        self.assertEqual(report["exit_code"], 2)
        self.assertTrue(report["coverage"]["incomplete_reasons"])
        self.assertEqual(
            report["summary"]["verified_accepted_step_node_gaps"]["negative_gap_count"], 0)
        self.assertEqual(
            report["summary"]["rejected_or_unverified_step_node_gap_diagnostics"]["negative_gap_count"], 1)

    def test_false_accepted_solver_certificate_fails_independent_audit(self) -> None:
        row = next(row for row in self.observations
                   if row["step"] == 0 and row["environment"] == 0)
        row["certificate_residual"] = 2.0e-4
        self._write_csv("observations.csv", self.observations)
        report = analysis.analyze_run(self.run_dir)
        self.assertEqual(report["verdict"], "failed")
        self.assertEqual(report["checks"]["accepted_solver_certificate_within_manifest_tolerance"], "fail")
        self.assertEqual(report["summary"]["verified_accepted_step_node_gaps"]["samples"], 6)
        self.assertEqual(len(report["coverage"]["certificate_acceptance_errors"]), 1)


if __name__ == "__main__":
    unittest.main()
