from pathlib import Path
import importlib.util
import unittest
import numpy as np

p=Path(__file__).with_name("graded_cut_step_447.py")
spec=importlib.util.spec_from_file_location("cuts447_test",p)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class FixedGuardScaleCutTests(unittest.TestCase):
    def test_fixed_guard_rows_join_seed_lp_and_preserve_anchor_certificate(self):
        d=np.array([[0.,0,0],[1.,0,0],[0.,0,0]])
        packed={"indptr":np.array([0,1]),"nodes":np.array([2]),
                "coefficients":np.array([[1.,0,0]]),"lower":np.array([-.5])}
        fixed=[
            {"nodes":np.array([1]),"coefficients":np.array([[-1.,0,0]]),
             "lower":-.25,"label":"fixed-test"},
            {"nodes":np.array([0]),"coefficients":np.array([[1.,0,0]]),
             "lower":-1.0,"label":"anchor-test"},
        ]
        alpha,proof=m.solve_with_fixed_guard_rows(
            m.cuts431,np.array([[0,1,2]]),np.ones(3),np.array([0]),
            np.empty((0,2),dtype=np.int64),np.empty(0),d,[],packed,
            "447-synthetic-fixed-rows",fixed,{"enabled":True,"typed_pair_count":1})
        self.assertEqual(alpha[0],0.0)
        self.assertLessEqual(alpha[1],.25+1e-8)
        self.assertEqual(proof["seed_rows_checked"],1)
        cert=proof["fixed_guard_certificate"]
        self.assertTrue(cert["passed"])
        self.assertEqual(cert["row_count"],2)
        self.assertEqual(cert["anchor_rows_checked"],[0])
        self.assertEqual(cert["input_receipt"]["typed_pair_count"],1)
        bad=m.fixed_guard_certificate(fixed,d,np.array([0.,.5,1.]),[0])
        self.assertFalse(bad["passed"])
        self.assertEqual(bad["violated_rows"],1)

    def test_tiny_baseline_roundoff_is_clamped_only_for_lp_and_certified_original(self):
        d=np.array([[1.,0,0],[0.,0,0]])
        guard=[{"nodes":np.array([0]),"coefficients":np.array([[1.,0,0]]),
                "lower":1e-8}]
        alpha,proof=m.solve_with_fixed_guard_rows(
            m.cuts431,np.array([[0,1,0]]),np.ones(2),np.array([1]),
            np.empty((0,2),dtype=np.int64),np.empty(0),d,[],
            {"indptr":np.array([0]),"nodes":np.empty(0,dtype=np.int64),
             "coefficients":np.empty((0,3)),"lower":np.empty(0)},
            "447-synthetic-roundoff",guard)
        cert=proof["fixed_guard_certificate"]
        self.assertTrue(cert["passed"])
        self.assertEqual(cert["lp_positive_lower_clamped_rows"],1)
        self.assertAlmostEqual(cert["lp_max_positive_lower_clamp_mm"],1e-8)

    def test_row_validation_rejects_out_of_domain_global_nodes(self):
        row={"nodes":np.array([3]),"coefficients":np.ones((1,3)),"lower":0.0}
        with self.assertRaisesRegex(ValueError,"global node IDs"):
            m.validate_rows([row],3)

if __name__=="__main__": unittest.main()
