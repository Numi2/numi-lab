import importlib.util
from pathlib import Path
import unittest
import numpy as np

p=Path(__file__).with_name("graded_cut_step_431.py")
spec=importlib.util.spec_from_file_location("cuts431_test",p)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class SeedScaleCutTests(unittest.TestCase):
    def test_nonuniform_scales_cannot_break_cached_seed_row(self):
        d=np.array([[1.,0,0],[0.8,0,0],[0,0,0]])
        # Row: alpha[0] - .8*alpha[1] >= -.5. The raw full delta
        # satisfies it (.2), while alpha=(.2,1,0) violates it (-.6).
        packed={"indptr":np.array([0,2]),"nodes":np.array([0,1]),
                "coefficients":np.array([[1.,0,0],[-1.,0,0]]),"lower":np.array([-.5])}
        cap=dict(nodes=np.array([0]),coefficients=np.array([[-1.,0,0]]),lower=-.2)
        raw,lower,valid,A,b=m._project_seed_rows(packed,d,3,"synthetic-cut-test")
        self.assertGreater(float((lower-raw@np.array([.2,1.,0.]))[0]),0.09)
        x,proof=m._solve_with_seed_cuts(np.array([[0,1,2]],dtype=np.int64),np.ones(3),np.array([2]),
            np.empty((0,2),dtype=np.int64),np.empty(0),d,[cap],packed,"synthetic-cut-test")
        self.assertLessEqual(float((lower-raw@x)[0]),m.ROW_TOL_MM)
        self.assertEqual(x[2],0.0)
        self.assertEqual(proof["seed_rows_active"],1)
        self.assertEqual(proof["seed_violated_rows"],0)
        self.assertLessEqual(proof["normalized_max_row_violation"],m.LP_TOL)

    def test_all_rows_checked_even_when_only_violations_become_cuts(self):
        d=np.array([[1.,0,0],[0.8,0,0]])
        packed={"indptr":np.array([0,2,4]),"nodes":np.array([0,1,0,1]),
                "coefficients":np.array([[1.,0,0],[-1.,0,0],[0.,1,0],[0.,-1.,0]]),
                "lower":np.array([-.5,-1.0])}
        cap=dict(nodes=np.array([0]),coefficients=np.array([[-1.,0,0]]),lower=-.2)
        x,proof=m._solve_with_seed_cuts(np.array([[0,1,2]],dtype=np.int64),np.ones(2),np.array([],dtype=np.int64),
            np.empty((0,2),dtype=np.int64),np.empty(0),d,[cap],packed,"synthetic-cut-test-2")
        self.assertEqual(proof["seed_rows_checked"],2)
        self.assertEqual(proof["seed_violated_rows"],0)
        self.assertLessEqual(float(np.maximum(packed["lower"]-m._project_seed_rows(packed,d,2,"synthetic-cut-test-2")[0]@x,0).max()),m.ROW_TOL_MM)

if __name__=="__main__": unittest.main()
