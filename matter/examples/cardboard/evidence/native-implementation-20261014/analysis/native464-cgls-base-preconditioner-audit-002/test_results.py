#!/usr/bin/env python3
import json, math, unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
R=json.loads((HERE/'results.json').read_text())
class Captured464CGLSTests(unittest.TestCase):
    def test_all_cases_run_full_bounded_cap_with_finite_outputs(self):
        self.assertEqual(R['cgls_cap'],64)
        for precision,strategies in R['precision_methods'].items():
            self.assertEqual(set(strategies),{'identity','block3','base_graph'})
            for name,case in strategies.items():
                self.assertEqual(case['status'],'iteration_cap',(precision,name))
                self.assertEqual(case['iterations'],64,(precision,name))
                def finite(v):
                    if isinstance(v,float): return math.isfinite(v)
                    if isinstance(v,dict): return all(finite(x) for x in v.values())
                    if isinstance(v,list): return all(finite(x) for x in v)
                    return True
                self.assertTrue(finite(case),(precision,name))
                self.assertEqual([x['iteration'] for x in case['checkpoints']],[8,16,32,64])
    def test_measured_truncated_residual_ordering_is_data_bound(self):
        f64=R['precision_methods']['fp64']; f32=R['precision_methods']['fp32_cpu_arithmetic']
        self.assertLess(f64['block3']['final_relative_true_residual'],f64['identity']['final_relative_true_residual'])
        self.assertGreater(f64['base_graph']['final_relative_true_residual'],f64['identity']['final_relative_true_residual'])
        self.assertLess(f32['block3']['final_relative_true_residual'],f32['identity']['final_relative_true_residual'])
        self.assertGreater(f32['base_graph']['final_relative_true_residual'],f32['identity']['final_relative_true_residual'])
    def test_source_map_actions_and_transpose_are_checked(self):
        self.assertTrue(all(x['S_rel_l2']<2e-13 and x['ST_rel_l2']<2e-13 for x in R['fast_map_vs_source_reference']))
        self.assertLess(R['source_map_transpose_dot_identity']['relative_gap'],2e-13)
    def test_actual_consumed_view_not_claimed(self):
        self.assertFalse(R['lineage']['actual_selected_view_admissible'])
        self.assertTrue(R['lineage']['completed_tail_latch_captured'])
        self.assertIn('reference clone only',R['lineage']['benchmark_map_latch_override'])
        self.assertIn('that selector-5 augmented map was consumed',R['not_claimed'])
    def test_entrywise_gate_failure_is_retained(self):
        p=R['patch_matrix_gate_context']
        self.assertEqual(p['per_entry_atol'],1e-9)
        self.assertFalse(p['within_atol'])
        self.assertGreater(p['max_entry_error'],p['per_entry_atol'])
    def test_warning_control_keeps_finite_result(self):
        d=json.loads((HERE/'warning-control.json').read_text())
        self.assertTrue(d['input_values_finite']);self.assertTrue(d['output_finite'])
        self.assertEqual(len(d['warnings']),3)
if __name__=='__main__':unittest.main(verbosity=2)
