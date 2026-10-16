#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
CASE = Path('/Users/home/cardboard-implementation-20261014/solver/cgls-regional-state-capture-464')
REPORT = HERE / 'native464-audit-v3.json'
SPEC = importlib.util.spec_from_file_location('audit_native464_v3', HERE/'audit_native464_v3.py')
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)

class Native464AuditTests(unittest.TestCase):
    def test_final_capture_selection_segment_is_well_formed(self):
        doc = json.loads((CASE/'base-cgls.bin.state.json').read_text())
        result = MOD.parse_cgls_capture_selection(CASE/'base-cgls.bin', 2, 882, 3528)
        self.assertEqual(result['selected_environment_words'], [1, 0])
        self.assertTrue(result['header_and_all_segment_extents_valid'])
        self.assertEqual(result['iteration_cap'], 64)
        self.assertEqual(result['segment_count'], 12)

    def test_bad_header_rejected(self):
        raw = bytearray((CASE/'base-cgls.bin').read_bytes())
        raw[0] ^= 0xff
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'bad.bin'; path.write_bytes(raw)
            with self.assertRaises(MOD.AuditError):
                MOD.parse_cgls_capture_selection(path, 2, 882, 3528)

    def test_selected_environment_cross_check_and_sidecar_erratum(self):
        d = json.loads(REPORT.read_text())
        selection = d['authoritative_environment_selection']
        self.assertTrue(selection['derived_and_final_selection_agree'])
        self.assertEqual(selection['marker_and_healthy_environments'], [0])
        self.assertEqual(selection['final_cgls_selected_environments'], [0])
        self.assertFalse(selection['regional_primary_policy_selected_environment_words_are_authoritative'])
        self.assertEqual(d['operator_comparison']['view_admission']['primary_policy_selection_words_pre_layout_stale'], [0, 0])

    def test_numeric_gates_and_refusal_are_preserved(self):
        d = json.loads(REPORT.read_text())
        self.assertTrue(d['state_to_419_exact_input_join']['exact_operator_join_admissible'])
        self.assertFalse(d['operator_comparison']['patch_matrices']['within_atol'])
        self.assertGreater(d['operator_comparison']['patch_matrices']['free_free_max_abs'], 1e-9)
        self.assertTrue(d['operator_comparison']['base_RAP']['base_factor_valid'])
        self.assertTrue(d['operator_comparison']['view_admission']['actual_view_request_refused']['refused'])
        self.assertEqual(d['run_receipts']['464']['exit_code'], 2)
        self.assertTrue(d['run_receipts']['464']['source_inputs_unchanged'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
