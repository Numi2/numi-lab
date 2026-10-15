#!/usr/bin/env python3
"""Keep the known unexpected RHS metadata delta visible in the sealed audit."""
import json
from pathlib import Path
import unittest

REPORT=Path(__file__).with_name('report.json')

class SealedParityReportContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads(REPORT.read_text())

    def test_overall_verdict_remains_fail_closed(self):
        self.assertEqual(self.report['overall_verdict'],'FAIL_UNEXPECTED_METADATA')

    def test_rhs_flag_is_never_normalized(self):
        capture=self.report['center_capture']
        self.assertEqual(capture['rhs_preserved_before_fgmres'],
                         {'362':True,'369':False,'equal':False})
        self.assertEqual([x['field'] for x in capture['unexpected_manifest_differences']],
                         ['rhs_preserved_before_fgmres'])

    def test_physical_payload_parity_is_separate_and_scoped(self):
        self.assertEqual(self.report['classification']['physical_payload_parity'],
                         'PASS_WITH_DEVICE_PROGRAM_FINGERPRINT_MASK')
        self.assertEqual(self.report['center_capture']['array_count'],145)
        self.assertTrue(self.report['center_capture']['all_array_descriptors_and_bytes_equal'])
        self.assertEqual(self.report['native_logs_and_attempt_outcome']['armijo_records'],32)
        self.assertEqual(len(self.report['diagnostic_state']['state_binary']),6)
        self.assertTrue(all(x['all_bytes_outside_device_program_fingerprint_equal']
                            for x in self.report['diagnostic_state']['state_binary']))

if __name__=='__main__':
    unittest.main(verbosity=2)
