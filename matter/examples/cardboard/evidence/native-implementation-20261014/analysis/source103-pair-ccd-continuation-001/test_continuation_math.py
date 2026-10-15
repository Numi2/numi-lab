#!/usr/bin/env python3
"""CPU-only recurrence checks; these do not execute or certify Metal CCD."""
import importlib.util
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('audit_native367',HERE/'audit_native367.py')
AUDIT=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(AUDIT)

class ContinuationRecurrenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result=AUDIT.strict_json(AUDIT.NATIVE/'native-result.json')
        cls.case=next(c for c in result['cases'] if c['name']=='trial358_ee_edge0_edge0')

    def test_captured_pair_exhausts_original_bound_but_reaches_band_in_extended_cpu_recurrence(self):
        short=AUDIT.fp64_distance_trace(self.case,17)
        extended=AUDIT.fp64_distance_trace(self.case,33)
        self.assertTrue(short['iteration_exhausted'])
        self.assertIsNone(short['hit_candidate_index'])
        self.assertEqual(extended['hit_candidate_index'],27)
        self.assertLessEqual(extended['trace'][-1]['distance_m'],
                             self.case['trace_clone_result']['effective_thickness_m'])

    def test_exact_start_contact_is_seen_before_any_advance(self):
        case={**self.case}
        points=case['points_start_m']
        # Move both endpoints of the second segment to the first segment's
        # start point. This is a deliberately degenerate CPU boundary fixture.
        point=points[:3]
        case['points_start_m']=points[:6]+point+point
        case['points_finish_m']=case['points_start_m']
        case['points_start_f32_bits']=[f'{int(x):08x}' for x in
            __import__('numpy').asarray(case['points_start_m'],dtype='<f4').view('<u4')]
        case['points_finish_f32_bits']=list(case['points_start_f32_bits'])
        result=AUDIT.fp64_distance_trace(case,33)
        self.assertEqual(result['hit_candidate_index'],0)
        self.assertEqual(result['evaluations'],1)

    def test_a_short_budget_remains_exhaustion_not_a_miss(self):
        result=AUDIT.fp64_distance_trace(self.case,17)
        self.assertTrue(result['iteration_exhausted'])
        self.assertNotIn('certified_miss',result)

if __name__=='__main__':
    unittest.main(verbosity=2)
