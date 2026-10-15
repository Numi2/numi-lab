import json
import unittest
from pathlib import Path

REPORT=Path(__file__).with_name('report.json')

class NativeLifetimeParityReportContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads(REPORT.read_text())

    def test_rhs_callback_metadata_observation_is_not_normalized(self):
        self.assertEqual(self.report['overall_verdict'],'FAIL_UNEXPECTED_STATUS')
        rows=self.report['capture_comparison']['runs']
        self.assertEqual({k:v['rhs_preserved_before_fgmres'] for k,v in rows.items()},
                         {'362':True,'369':False,'371':True})
        self.assertEqual({v['payload_sha256'] for v in rows.values()},
                         {'965d38af46af2bbc311cb1534193b8892581cb4d4548936dfc8d7ba9db0cc2e4'})
        self.assertTrue(all(v['array_count']==145 for v in rows.values()))

    def test_only_the_rejected_newton7_trial_status_differs(self):
        differences=self.report['diagnostic_states_logs_and_armijo']['armijo_status_differences']
        self.assertEqual(len(differences),4) # stderr and mirrored fixture log for two pairings
        for item in differences:
            self.assertEqual(item['newton_iteration'],7)
            fields={d['field'] for d in item['differences']}
            self.assertEqual(fields,{
                'environments[0].trial_status.failing_index',
                'environments[0].trial_status.diagnostics[0]',
                'environments[0].trial_status.diagnostics[2]'})
        self.assertFalse(any('rhs_preserved_before_fgmres' in str(d) for d in differences))

    def test_transactions_and_payloads_remain_byte_bound(self):
        pairs=self.report['diagnostic_states_logs_and_armijo']['pairwise']
        for pair in pairs.values():
            self.assertEqual(len(pair['diagnostic_state']['transaction_json']),3)
            self.assertTrue(all(x['exact_equal'] for x in pair['diagnostic_state']['transaction_json']))
            self.assertEqual(len(pair['diagnostic_state']['transaction_state_binary']),6)
            self.assertTrue(all(x['bytes_equal_outside_device_fingerprint']
                                for x in pair['diagnostic_state']['transaction_state_binary']))
        self.assertEqual(self.report['runs']['362']['exit_code'],1)
        self.assertEqual(self.report['runs']['369']['exit_code'],1)
        self.assertEqual(self.report['runs']['371']['exit_code'],1)

if __name__=='__main__':
    unittest.main(verbosity=2)
