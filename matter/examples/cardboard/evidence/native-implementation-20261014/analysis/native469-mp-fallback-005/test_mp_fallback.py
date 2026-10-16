import hashlib
import json
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'run-002'
REPORT_PATH = RUN / 'report.json'
ARCHIVE_PATH = RUN / 'mixed_reference_arrays.npz'


class ExplicitMPFallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(REPORT_PATH.read_text())
        cls.arrays = np.load(ARCHIVE_PATH)

    def test_all_five_mixed_state_tensors_are_finite_and_shaped(self):
        self.assertEqual(self.report['mixed_reference_arrays_sha256'], hashlib.sha256(ARCHIVE_PATH.read_bytes()).hexdigest())
        for state in ('center', 's1', 's2', 's3', 's4'):
            self.assertTrue(np.isfinite(self.arrays[f'{state}_tangent']).all())
            self.assertTrue(np.isfinite(self.arrays[f'{state}_stress']).all())
            self.assertEqual(self.arrays[f'{state}_tangent'].shape, (1920, 9, 9))
            self.assertEqual(self.arrays[f'{state}_stress'].shape, (1920, 3, 3))

    def test_parameter_words_follow_each_captured_tet_material_offset(self):
        materials = self.report['paper_material_parameter_bindings']
        indexes = self.arrays['paper_material_index']
        parameters = self.arrays['paper_parameters_f32']
        self.assertEqual(indexes.shape, (1728,))
        self.assertEqual(parameters.shape, (1728, 28))
        for material_index in (0, 1):
            expected = np.asarray(materials[str(material_index)]['parameter_words_f32'], dtype=np.float32)
            self.assertEqual(len(expected), 28)
            self.assertEqual(materials[str(material_index)]['parameter_offset'], 28 * material_index)
            self.assertTrue(np.array_equal(parameters[indexes == material_index],
                                           np.broadcast_to(expected, (int(np.sum(indexes == material_index)), 28))))
        self.assertFalse(np.array_equal(np.asarray(materials['0']['parameter_words_f32']),
                                        np.asarray(materials['1']['parameter_words_f32'])))

    def test_all_original_fp64_failures_are_preserved_and_material_bound(self):
        original = self.report['original_fp64_failures_preserved']
        output = self.report['mp_results']
        self.assertEqual(len(original), 17)
        self.assertEqual(len(output), 17)
        self.assertEqual([(r['state'], r['tet'], r['reason']) for r in original],
                         [(r['state'], r['tet'], r['original_fp64_failure']) for r in output])
        self.assertTrue(all(r['fallback_status'] == 'mp_gates_pass' for r in output))
        for row in output:
            self.assertIn(row['material_index'], (0, 1))
            self.assertEqual(row['material_parameter_offset'], 28 * row['material_index'])
            self.assertEqual(row['material_name'], self.report['paper_material_parameter_bindings'][str(row['material_index'])]['name'])

    def test_unchanged_mp_residual_correction_and_precision_gates(self):
        gates = self.report['strict_gates']
        self.assertEqual(gates['scaled_residual_max'], '2e-12')
        self.assertEqual(gates['dynamic_scaled_correction_max'], '2e-11')
        self.assertEqual(gates['mp_precision_root_spread_max'], '1e-12')
        for row in self.report['mp_results']:
            self.assertTrue(row['mp_root_stability_gate_1e-12'])
            self.assertLessEqual(float(row['mp110_final_scaled_residual']), 2e-12)
            self.assertLessEqual(float(row['mp110_final_dynamic_correction']), 2e-11)

    def test_mp_ift_fp64_ift_and_native_center_tangent_comparisons(self):
        metrics = self.report['summary_metrics']
        self.assertLessEqual(metrics['max_fp64_ift_vs_mp_ift_relative_l2_at_same_rounded_mp_root'], 1e-6)
        self.assertLessEqual(metrics['max_mp_ift_90_vs_110_relative_l2'], 1e-12)
        self.assertLessEqual(metrics['max_center_native_tangent_vs_mp_ift_relative_l2'], 1e-3)

    def test_successful_rows_retain_correct_source166_material_tangents(self):
        checks = self.report['successful_row_material_binding_checks']
        self.assertEqual(len(checks), 10)
        for state in ('center', 's1', 's2', 's3', 's4'):
            selected = [x for x in checks if x['state'] == state]
            self.assertEqual({x['material_index'] for x in selected}, {0, 1})
        for row in checks:
            self.assertLessEqual(row['mixed_output_vs_same_source166_row']['relative_l2'], 1e-12)
            # This fixture deliberately selects a state where the other authored
            # grade produces a materially different tangent, catching a shared-
            # parameter-vector regression rather than merely checking labels.
            self.assertGreater(row['opposite_material_tangent_vs_correct']['relative_l2'], 0.06)

    def test_all_center_source166_fp64_success_tangents_are_bitwise_preserved(self):
        check = self.report['center_source166_reference_C_preservation']
        self.assertEqual(check['valid_source166_paper_row_count'], 1724)
        self.assertEqual(check['compared_float64_entries'], 1724 * 9 * 9)
        self.assertTrue(check['tangent_values_bitwise_preserved'])

    def test_s1_tet1317_old_medium_parameter_error_is_retracted(self):
        result = self.report['material_binding_regression']
        self.assertEqual((result['state'], result['tet'], result['native_material_index']), ('s1', 1317, 0))
        self.assertEqual(result['native_material_name'], 'hajali2009_liner_hill_bilinear_fit')
        self.assertEqual(result['native_parameter_offset'], 0)
        self.assertLessEqual(result['correct_liner_parameter_stress_error']['relative_l2'], 1e-6)
        self.assertGreater(result['wrong_medium_parameter_stress_error']['relative_l2'], 0.06)
        self.assertEqual(self.report['supersedes_fallback004']['status'],
                         '004 material binding was incorrect for all liner tets: it hardcoded the medium parameter range [28:56] for every row.')

    def test_unqualified_native_branch_and_assembly_axes_remain_visible(self):
        mismatches = self.report['summary_metrics']['native_vs_mp_branch_predicate_mismatches']
        self.assertEqual(len(mismatches), self.report['summary_metrics']['native_vs_mp_branch_predicate_mismatch_count'])
        axes = self.report['qualification_axes']
        self.assertFalse(axes['native_projection'])
        self.assertFalse(axes['complete_native_tangent'])
        self.assertFalse(axes['full_assembled_action'])
        self.assertFalse(axes['physical_or_calibration_qualification'])


if __name__ == '__main__':
    unittest.main()
