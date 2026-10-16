import hashlib
import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run-006'
REPORT_PATH = RUN / 'report.json'
ARCHIVE_PATH = RUN / 'mixed_reference_arrays.npz'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


class ExplicitMPFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(REPORT_PATH.read_text())
        cls.arrays = np.load(ARCHIVE_PATH)

    def test_output_hash_and_all_five_mixed_state_tensors_are_finite(self):
        self.assertEqual(self.report['mixed_reference_arrays_sha256'], sha(ARCHIVE_PATH))
        for state in ('center', 's1', 's2', 's3', 's4'):
            self.assertTrue(np.isfinite(self.arrays[f'{state}_tangent']).all())
            self.assertTrue(np.isfinite(self.arrays[f'{state}_stress']).all())
            self.assertEqual(self.arrays[f'{state}_tangent'].shape, (1920, 9, 9))
            self.assertEqual(self.arrays[f'{state}_stress'].shape, (1920, 3, 3))

    def test_only_original_fp64_failed_rows_are_given_mp_fallbacks(self):
        original = self.report['original_fp64_failures_preserved']
        output = self.report['mp_results']
        self.assertEqual(len(original), 17)
        self.assertEqual(len(output), 17)
        self.assertEqual([(r['state'], r['tet'], r['reason']) for r in original],
                         [(r['state'], r['tet'], r['original_fp64_failure']) for r in output])
        self.assertTrue(all(r['fallback_status'] == 'mp_gates_pass' for r in output))

    def test_strict_residual_correction_and_precision_stability_gates_are_unchanged(self):
        self.assertEqual(self.report['strict_gates']['scaled_residual_max'], '2e-12')
        self.assertEqual(self.report['strict_gates']['dynamic_scaled_correction_max'], '2e-11')
        self.assertEqual(self.report['strict_gates']['mp_precision_root_spread_max'], '1e-12')
        for row in self.report['mp_results']:
            self.assertTrue(row['mp_root_stability_gate_1e-12'])
            self.assertLessEqual(float(row['mp110_final_scaled_residual']), 2e-12)
            self.assertLessEqual(float(row['mp110_final_dynamic_correction']), 2e-11)

    def test_mp_ift_and_fp64_ift_match_at_same_rounded_mp_root(self):
        self.assertLessEqual(self.report['summary_metrics']['max_fp64_ift_vs_mp_ift_relative_l2_at_same_rounded_mp_root'], 1e-6)
        self.assertLessEqual(self.report['summary_metrics']['max_mp_ift_90_vs_110_relative_l2'], 1e-12)
        self.assertLessEqual(self.report['summary_metrics']['max_center_native_tangent_vs_mp_ift_relative_l2'], 1e-3)

    def test_native_side_stress_discrepancy_at_captured_state_is_not_hidden(self):
        # The s1/tet1317 captured stress disagrees with the independent source166
        # stress evaluated at that exact captured state. Keep this unresolved
        # outlier visible instead of treating MP roots as a native stress pass.
        rows = [r for r in self.report['mp_results'] if r['state'] == 's1' and r['tet'] == 1317]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertIn('projection_diagnostics_raw', row['native_projection_metadata_raw'])
        self.assertGreater(row['native_captured_stress_vs_fp64_stress_at_native_state']['relative_l2'], 0.06)
        self.assertLessEqual(self.report['summary_metrics']['max_native_captured_stress_vs_fp64_stress_at_native_state_relative_l2_by_state']['center'], 1e-6)
        self.assertFalse(self.report['qualification_axes']['complete_native_tangent'])
        self.assertFalse(self.report['qualification_axes']['full_assembled_action'])

    def test_branch_mismatches_and_unqualified_axes_remain_visible(self):
        mismatches = self.report['summary_metrics']['native_vs_mp_branch_predicate_mismatches']
        self.assertEqual(len(mismatches), self.report['summary_metrics']['native_vs_mp_branch_predicate_mismatch_count'])
        self.assertGreater(len(mismatches), 0)
        axes = self.report['qualification_axes']
        self.assertFalse(axes['native_projection'])
        self.assertFalse(axes['complete_native_tangent'])
        self.assertFalse(axes['full_assembled_action'])
        self.assertFalse(axes['physical_or_calibration_qualification'])


if __name__ == '__main__':
    unittest.main()
