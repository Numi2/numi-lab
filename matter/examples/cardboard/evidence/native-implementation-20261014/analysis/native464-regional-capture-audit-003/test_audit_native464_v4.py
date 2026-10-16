#!/usr/bin/env python3
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
REPORT = HERE / 'native464-audit-v4.json'

class Native464FrobeniusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = json.loads(REPORT.read_text())
        cls.patch = cls.doc['operator_comparison']['patch_matrices']

    def test_global_frobenius_metrics_are_consistent(self):
        p = self.patch
        self.assertAlmostEqual(p['all_entry_frobenius_difference_l2'] /
                               p['all_entry_frobenius_reference_l2'],
                               p['all_entry_frobenius_relative_l2'], places=14)
        self.assertAlmostEqual(p['free_free_frobenius_difference_l2'] /
                               p['free_free_frobenius_reference_l2'],
                               p['free_free_frobenius_relative_l2'], places=14)
        self.assertEqual(len(p['per_patch']), 27)
        self.assertTrue(all('free_free_relative_l2' in row for row in p['per_patch']))

    def test_failed_entry_gate_is_retained(self):
        p = self.patch
        self.assertEqual(p['atol'], 1e-9)
        self.assertFalse(p['within_atol'])
        self.assertGreater(p['free_free_max_abs'], p['atol'])
        w = p['worst_free_free_entry']
        self.assertEqual(w['patch_index'], 0)
        self.assertEqual(w['global_row_dof'], 1029)
        self.assertEqual(w['global_column_dof'], 1029)
        self.assertAlmostEqual(w['abs_error'], p['free_free_max_abs'], places=16)
        self.assertAlmostEqual(w['reference_magnitude'], abs(w['reference_value']), places=15)

    def test_frobenius_result_does_not_promote_entry_gate(self):
        p = self.patch
        self.assertLess(p['free_free_frobenius_relative_l2'], 1e-7)
        self.assertGreater(p['free_free_max_abs'], 1e-9)
        self.assertFalse(p['within_atol'])
        self.assertFalse(self.doc.get('native_or_physical_qualification', False))

if __name__ == '__main__':
    unittest.main(verbosity=2)
