import importlib.util
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse

SOURCE=Path(__file__).with_name("basis_feasibility_398.py")
spec=importlib.util.spec_from_file_location("basis_feasibility_398_test",SOURCE)
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class BasisFeasibilityTests(unittest.TestCase):
    def test_identity_basis_feasible_certificate(self):
        A=sparse.eye(2,format="csr")
        report,arrays=mod.compare_feasibility(A,np.array([1.,-1.]),np.array([-2.,-2.]),np.array([2.,2.]),sparse.eye(2,format="csr"),time_limit_s=5)
        self.assertTrue(report["reduced_basis"]["success"])
        self.assertTrue(report["full_output_coordinates"]["success"])
        self.assertTrue(report["reduced_basis"]["output_primal_certificate"]["passed"])
        self.assertTrue(report["full_output_coordinates"]["primal_certificate"]["passed"])
        self.assertIn("full_output_primal_d",arrays)

    def test_full_output_feasible_but_reduced_basis_infeasible(self):
        # d1>=1 is feasible in R2, but the one-column basis fixes d1=0.
        A=sparse.csr_matrix([[0.,1.]])
        report,arrays=mod.compare_feasibility(A,np.array([1.]),np.array([-2.,-2.]),np.array([2.,2.]),sparse.csr_matrix([[1.],[0.]]),time_limit_s=5)
        self.assertFalse(report["reduced_basis"]["success"])
        self.assertTrue(report["full_output_coordinates"]["success"])
        self.assertTrue(report["full_output_coordinates"]["primal_certificate"]["passed"])
        self.assertIn("full_output_primal_d",arrays)

    def test_infeasible_same_rows_remain_infeasible(self):
        A=sparse.csr_matrix([[1.],[-1.]])
        report,arrays=mod.compare_feasibility(A,np.array([1.,1.]),np.array([-2.]),np.array([2.]),sparse.eye(1,format="csr"),time_limit_s=5)
        self.assertFalse(report["reduced_basis"]["success"])
        self.assertFalse(report["full_output_coordinates"]["success"])
        self.assertEqual(arrays,{})

    def test_bad_dimensions_fail_closed(self):
        with self.assertRaises(ValueError):
            mod.compare_feasibility(sparse.eye(2),np.ones(2),np.zeros(2),np.ones(2),sparse.eye(3),time_limit_s=5)


if __name__=="__main__":
    unittest.main()
