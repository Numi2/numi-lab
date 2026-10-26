import importlib.util
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linprog

SOURCE=Path(__file__).with_name("elastic_precision_phase_one_398.py")
spec=importlib.util.spec_from_file_location("elastic_precision_phase_one_398_test",SOURCE)
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


class ElasticPhaseOneTests(unittest.TestCase):
    def test_only_precision_duplicate_is_relaxed_original_remains_hard(self):
        # Row 0 is the original hard row d>=1; row 1 duplicates it with a +1 margin.
        A=sparse.csr_matrix([[1.],[1.]])
        p=mod.build_elastic_problem(A,np.array([1.,2.]),np.array([-3.]),np.array([3.]),
                                    sparse.eye(1,format="csr"),1,2,np.array([0]),[("disjoint",(4,9))])
        res=linprog(p["objective"],A_ub=p["A_ub"],b_ub=p["b_ub"],bounds=p["bounds"],method="highs")
        self.assertTrue(res.success)
        d=float(res.x[0]); slack=float(res.x[1])
        self.assertGreaterEqual(d-1.,-1e-8) # original row has no slack column
        self.assertAlmostEqual(d+slack,2.,places=7)
        self.assertAlmostEqual(slack,1.,places=7)

    def test_original_infeasibility_cannot_be_hidden_by_phase_slack(self):
        # Original d>=2 conflicts with hard upper bound d<=1; precision d>=3 is soft.
        A=sparse.csr_matrix([[1.],[1.]])
        with self.assertRaises(ValueError):
            mod.build_elastic_problem(A,np.array([2.,3.]),np.array([-1.]),np.array([1.]),
                                      sparse.eye(1,format="csr"),1,2,np.array([0]),[("wedge",(1,8))])

    def test_pair_balanced_normalized_weights(self):
        A=sparse.eye(3,format="csr")
        b=np.array([-1.,-1.,-1.,0.2,0.4]) if False else np.array([-1.,-1.,-1.])
        # Two appended rows strengthen their own source rows by different amounts.
        A=sparse.vstack((A,A[:2]),format="csr")
        b=np.array([-1.,-1.,-1.,0.2,0.4])
        p=mod.build_elastic_problem(A,b,np.full(3,-2.),np.full(3,2.),sparse.eye(3,format="csr"),
                                    3,5,np.array([0,1]),[("wedge",(2,4)),("wedge",(2,4))])
        self.assertAlmostEqual(float(np.sum(p["weights"]*p["extra_rhs"])),1.0)
        self.assertEqual(p["hard_mask"].tolist(),[True,True,True,False,False])

    def test_bad_ranges_nonpositive_strengthening_or_row_mismatch_fail(self):
        A=sparse.eye(2,format="csr"); M=sparse.eye(2,format="csr")
        args=(A,np.array([-1.,-1.]),np.full(2,-2.),np.full(2,2.),M)
        with self.assertRaises(ValueError):
            mod.build_elastic_problem(*args,1,3,np.array([0,0]),[("x",(1,2)),("x",(1,3))])
        with self.assertRaises(ValueError):
            mod.build_elastic_problem(A,np.array([-1.,-0.5]),np.full(2,-2.),np.full(2,2.),M,1,2,np.array([0]),[("x",(1,2))])


if __name__=="__main__": unittest.main()
