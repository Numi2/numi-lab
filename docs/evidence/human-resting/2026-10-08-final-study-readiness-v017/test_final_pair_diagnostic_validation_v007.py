#!/usr/bin/env python3
"""Fail-closed diagnostics regressions using retained completed study 867."""
import importlib.util
import json
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("analyze_final_pair_v007", ROOT / "analyze_final_pair.py")
analyzer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyzer)

EVIDENCE = Path("/Users/n/numi-human-resting-evidence-20261005")
STUDY = EVIDENCE / "native-coupled-endurance-study-867"
TRACE_DIR = STUDY / "trials"
PRIMARY = STUDY / "analysis.json"
TRACE_PATHS = {
    "control": TRACE_DIR / "resting-baseline/output/scene/resting-coupled.csv",
    "treatment": TRACE_DIR / "resting-drive-half/output/scene/resting-coupled.csv",
}
EXPECTED_DT = 0.0020000000949949026
EXPECTED_STEPS = 155000


def load_arm(arm):
    header, rows = analyzer.audit.load_csv(str(TRACE_PATHS[arm]))
    horizon = analyzer.audit.check_horizon(rows, EXPECTED_STEPS, EXPECTED_DT)
    return header, rows, horizon


class FinalPairDiagnosticValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.traces = {arm: load_arm(arm) for arm in ("control", "treatment")}

    def test_retained_867_primary_trace_calculation_is_unchanged(self):
        arm_changes = {}
        for arm, (header, rows, horizon) in self.traces.items():
            analyzer.validate_sampled_terminal(rows, horizon, EXPECTED_STEPS)
            analyzer.validate_trace_fields(header, rows, horizon["duration"])
            pre, _ = analyzer.window_mean(rows, "PaCO2_mmhg", 30.0, 60.0)
            dose, _ = analyzer.window_mean(rows, "PaCO2_mmhg", 70.0, 100.0)
            arm_changes[arm] = dose - pre
        raw_did = arm_changes["treatment"] - arm_changes["control"]
        primary = json.loads(PRIMARY.read_text(encoding="utf-8"))["payload"]["mean_difference"]
        self.assertTrue(math.isclose(raw_did, primary, rel_tol=0.0, abs_tol=1e-9))
        self.assertTrue(math.isclose(raw_did, 1.835244624837813, rel_tol=0.0, abs_tol=1e-9))

    def test_exact_nine_numerical_fields_are_finite_and_summarized(self):
        expected = tuple(analyzer.NUMERICAL_DIAGNOSTIC_FIELDS)
        self.assertEqual(len(expected), 9)
        for header, rows, horizon in self.traces.values():
            analyzer.validate_trace_fields(header, rows, horizon["duration"])
            summary = analyzer.numerical_diagnostic_highwater(rows, horizon["duration"], EXPECTED_STEPS)
            self.assertEqual(tuple(summary), expected)
            self.assertTrue(all(math.isfinite(value) and value >= 0.0 for value in summary.values()))


    def test_terminal_only_budget_peak_is_included_despite_float32_time(self):
        _, rows, _ = self.traces["control"]
        changed = list(rows)
        changed[-1] = dict(rows[-1])
        self.assertEqual(analyzer.audit.iv(changed[-1], "step"), EXPECTED_STEPS)
        self.assertGreater(float(changed[-1]["time_s"]), 310.0)
        changed[-1]["blood_error_ml"] = "123"
        summary = analyzer.numerical_diagnostic_highwater(changed, 310.0, EXPECTED_STEPS)
        self.assertEqual(summary["blood_error_ml"], 123.0)

    def test_nonfinite_terminal_budget_value_cannot_be_dropped(self):
        _, rows, _ = self.traces["control"]
        changed = list(rows)
        changed[-1] = dict(rows[-1])
        changed[-1]["oxygen_balance_error_stpd_ml"] = "nan"
        with self.assertRaisesRegex(ValueError, "missing/nonfinite trace field"):
            analyzer.numerical_diagnostic_highwater(changed, 310.0, EXPECTED_STEPS)

    def test_missing_required_diagnostic_header_fails(self):
        header, rows, horizon = self.traces["control"]
        incomplete = list(header)
        incomplete.remove("blood_physical_delta_accum_ml")
        with self.assertRaisesRegex(ValueError, "missing or duplicates"):
            analyzer.validate_trace_fields(incomplete, rows, horizon["duration"])

    def test_duplicate_required_diagnostic_header_fails(self):
        header, rows, horizon = self.traces["control"]
        duplicated = list(header) + ["oxygen_balance_error_stpd_ml"]
        with self.assertRaisesRegex(ValueError, "missing or duplicates"):
            analyzer.validate_trace_fields(duplicated, rows, horizon["duration"])

    def test_nonfinite_required_diagnostic_fails_closed(self):
        header, rows, horizon = self.traces["control"]
        broken = list(rows)
        index = next(i for i, row in enumerate(rows) if float(row["time_s"]) >= 10.0)
        broken[index] = dict(rows[index])
        broken[index]["blood_error_ml"] = "nan"
        with self.assertRaisesRegex(ValueError, "missing/nonfinite trace field blood_error_ml"):
            analyzer.validate_trace_fields(header, broken, horizon["duration"])

    def test_empty_selected_window_fails(self):
        _, rows, _ = self.traces["control"]
        with self.assertRaisesRegex(ValueError, "no PaCO2_mmhg samples"):
            analyzer.window_mean(rows, "PaCO2_mmhg", 311.0, 312.0)

    def test_missing_exact_terminal_sample_fails(self):
        _, rows, horizon = self.traces["control"]
        truncated = rows[:-1]
        with self.assertRaisesRegex(ValueError, "exact accepted terminal step"):
            analyzer.validate_sampled_terminal(truncated, horizon, EXPECTED_STEPS)

    def test_medlineplus_band_is_added_to_retained_per_arm_report(self):
        header, rows, horizon = self.traces["control"]
        base = (EVIDENCE / "native-paired-physiology-audit-872/completed-pair-867-verified/control-physiology.md").read_text(encoding="utf-8")
        report = analyzer.annotate_per_arm_report(base, rows, horizon["duration"])
        self.assertIn("| PaO2 (MedlinePlus ABG) |", report)
        self.assertIn("| PaO2 |", report)
        self.assertIn("75-100 mmHg; below 75:", report)
        self.assertIn("diagnostic summary reports maximum absolute sampled values", report)

    def test_medlineplus_band_and_field_semantics_are_explicit(self):
        sample = [
            {"time_s": "10", "PaO2_mmhg": "75"},
            {"time_s": "11", "PaO2_mmhg": "79"},
            {"time_s": "12", "PaO2_mmhg": "101"},
        ]
        report = analyzer.annotate_per_arm_report("\n| SaO2 |\n## References\n", sample, 20.0)
        self.assertIn("75-100 mmHg; below 75: 0, 75-<80: 2, above 100: 1 samples", report)
        self.assertIn("blood_error_ml are owner-maintained running maxima", report)
        self.assertIn("respiratory_net_volume_ml and respiratory_volume_balance_ml describe one physical step", report)


if __name__ == "__main__":
    unittest.main()
